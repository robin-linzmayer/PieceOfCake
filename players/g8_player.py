
import logging
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
from scipy.optimize import linear_sum_assignment
import miniball
from tqdm import tqdm
import constants
from piece_of_cake_state import PieceOfCakeState
from multiprocessing import cpu_count
import time
from concurrent.futures import ProcessPoolExecutor, wait


def fits_on_plate(cake_piece, radius=12.5):
    """
    Check if the cake can fit inside a plate of radius 12.5.
    """
    if cake_piece.area < 0.25: # Quentin's optimization 
        return True
    cake_points = np.array(
        list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
    )
    res = miniball.miniball(cake_points)
    return res["radius"] <= radius

def calculate_penalties(pieces, requests, tolerance):
    """Calculate penalties exactly as the game does"""
    n_pieces = len(pieces)
    n_requests = len(requests)
    max_size = max(n_pieces, n_requests)

    # Create cost matrix
    cost_matrix = np.full((max_size, max_size), 100.0)

    # Fill in penalties exactly as game calculates them
    for i, piece in enumerate(pieces):
        if not fits_on_plate(piece):
            continue

        area = piece.area
        for j, request in enumerate(requests):
            percentage_diff = 100 * abs(area - request) / request
            if percentage_diff <= tolerance:
                cost_matrix[i, j] = 0  # No penalty within tolerance
            else:
                cost_matrix[i, j] = percentage_diff

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Convert to assignment list format that game uses
    assignments = [-1] * n_requests
    for piece_idx, request_idx in zip(row_ind, col_ind):
        if piece_idx < n_pieces and request_idx < n_requests:
            assignments[request_idx] = piece_idx

    # Calculate penalty exactly as game does
    total_penalty = 0
    for request_idx, piece_idx in enumerate(assignments):
        if piece_idx == -1 or not fits_on_plate(pieces[piece_idx]):
            total_penalty += 100
        else:
            percentage_diff = (
                100
                * abs(pieces[piece_idx].area - requests[request_idx])
                / requests[request_idx]
            )
            if percentage_diff > tolerance:
                total_penalty += percentage_diff

    return total_penalty

def calculate_cut_length(points: list[tuple[float, float]]) -> float:
    """Calculate total length of cut sequence"""
    total_length = 0

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        total_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return total_length

def evaluate_cut_sequence(args): # Note args is a tuple of (new_points, pieces, cake, requests, tolerance)
    """
    Evaluate a sequence of cut endpoints, returning (score, new_points, new_beam_uuid, new_pieces, penalty)
    """
    # Unpack arguments
    new_points, pieces, cake, requests, tolerance = args

    # Convert points to cuts
    cuts = []
    for i in range(len(new_points) - 1):
        cuts.append(LineString([new_points[i], new_points[i + 1]]))

    cut = cuts[0] if len(cuts) == 1 else cuts[-1]

    new_pieces = []
    for piece in pieces:
        try:
            if cut.intersects(piece):
                # print("2")
                split_result = split(piece, cut)
                # print("3")

                new_pieces.extend(list(split_result.geoms))
            else:
                new_pieces.append(piece)
        except Exception as e:
            print(f"Error: {e}")
            print("3")
            new_pieces.append(piece)

    # Calculate penalties
    penalties = calculate_penalties(new_pieces, requests, tolerance)
    cut_length = calculate_cut_length(new_points)
    score = penalties + cut_length * 1e-6

    return (score, new_points, new_pieces, penalties)

# The G8_Player class
class G8_Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        tolerance: int,
    ) -> None:
        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        self.requests = None
        self.cut_path = []
        self.assignment = []
        self.inital_state = None
        self.edges = None
        self.cake = None
        self.solution = None

    def move(self, current_percept: PieceOfCakeState) -> tuple[int, list[int]]:
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        # On turn 1 we need to decide where to start.
        if turn_number == 1:
            # initialize these variables
            self.assignment = [-1 for _ in range(len(requests))]
            self.requests = requests
            self.cake = polygons[0]
            self.cake_len = cake_len
            self.cake_width = cake_width

            self.edges = [
                ((0, 0), (cake_width, 0)),                  # top
                ((cake_width, 0), (cake_width, cake_len)),  # right
                ((cake_width, cake_len), (0, cake_len)),    # bottom
                ((0, cake_len), (0, 0)),                    # left
            ]
            max_workers = max(cpu_count(), 1)
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                self.solution = self.solve(executor=executor)
            x, y = self.solution.pop(0)

            return constants.INIT, [round(x, 2), round(y, 2)]

        if len(self.solution) == 0:
            return self.assign_polygons(current_percept.polygons)

        x, y = self.solution.pop(0)

        return constants.CUT, [round(x, 2), round(y, 2)]

    def solve(self, executor) -> list[tuple[float, float]]:
        """Find optimal cutting sequence using beam search"""
        beam_width = min(50, len(self.requests) * 5)
        max_depth = len(self.requests) + 10
        counter = 0

        # Initialize beam with possible first points
        initial_points = self.generate_initial_points()
        beam = [(0, [point], [self.cake]) for point in initial_points]

        best_solution = None
        best_score = float("inf")
        best_beam_score = float("inf")
        best_score_depth = 0

        # Keep track of runtime
        start_time = time.time()
        time_limit = 3600 - 180 # 57 mins
        
        with tqdm(total=max_depth) as pbar:
            while counter < max_depth:
                new_beam = []

                # Prepare the args to be passed to the evaluate_cut_sequence function
                args_list = []
                for _, current_points, pieces in beam:
                    next_points_list = self.generate_next_points(current_points[-1])

                    for next_point in next_points_list:
                        if len(current_points) > 1:
                            if not self.is_valid_cut(
                                current_points[-1], next_point, current_points
                            ):
                                continue

                        new_points = current_points + [next_point]
                        args = (new_points, pieces, self.cake, self.requests, self.tolerance)
                        args_list.append(args)
                
                futures = [executor.submit(evaluate_cut_sequence, args) for args in args_list]
                print(f"Number of tasks submitted: {len(futures)}")

                timeout = 90
                # Wait max 90 seconds for tasks to complete
                done, not_done = wait(futures, timeout=timeout)
                # Process the completed futures
                for i, future in enumerate(done):
                    try:
                        result = future.result()
                        score, new_points, new_pieces, penalty = result

                        if penalty > best_score + 250:
                            continue

                        if penalty <= best_score:
                            best_score = penalty

                        new_beam.append((score, new_points, new_pieces))

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                # Sometime things hang so we need to cancel the not done tasks
                if not_done:
                    print(f"{len(not_done)} tasks did not complete and will be skipped.")
                    for future in not_done:
                        time.sleep(1)
                        future.cancel()


                # Keep best beam_width solutions
                beam = sorted(new_beam, key=lambda x: x[0])[:beam_width]

                # Early return if score is not improving
                if beam and beam[0][0] < best_beam_score:
                    best_score_depth = counter
                    best_beam_score = beam[0][0]
                    best_solution = beam[0][1]
                    print(f"Best score depth: {best_score_depth}")
                    print(f"Best score : {beam[0][0]}")

                if best_score_depth + 5 == counter:
                    break

                if best_beam_score < 0.1:
                    break

                counter += 1
                pbar.update(1)

                # Exit if we have reached the time limit
                elapsed_time = time.time() - start_time
                if elapsed_time >= time_limit:
                    print("Time limit reached after iteration. Returning the best solution found so far.")
                    break 

            if best_solution is None:
                raise ValueError("No valid solution found")

        return best_solution

    def generate_initial_points(self):
        """Generate possible starting points along the perimeter"""
        points = []
        samples = 5

        for (x1, y1), (x2, y2) in self.edges:
            for t in np.linspace(0, 1, samples):
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                points.append((x, y))

        return points

    def generate_next_points(self, current_point: tuple[float, float]):
        """Generate possible next points on valid edges"""
        points = []
        samples = 12

        current_edge = self.get_edge(current_point)

        for i, ((x1, y1), (x2, y2)) in enumerate(self.edges):
            if i != current_edge:  # Can't cut to same edge
                for t in np.linspace(0, 1, samples):
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    points.append((x, y))

        return points
    
    def get_edge(self, point: tuple[int, int], tolerance: float = 1e-10):
        """Get what edge a point lies on"""
        x, y = point

        if abs(y - 0) <= tolerance:
            return 0  # top

        if abs(x - self.cake_width) <= tolerance:
            return 1  # right

        if abs(y - self.cake_len) <= tolerance:
            return 2  # bottom

        if abs(x - 0) <= tolerance:
            return 3  # left

        raise ValueError(f"Point {point} not on any edge")

    def assign_polygons(self, pieces: list[Polygon]) -> tuple[str, list[int]]:
        """
        Get optimal mapping of requests to pieces using exact game penalty calculation.
        """
        n_pieces = len(pieces)
        n_requests = len(self.requests)
        max_size = max(n_pieces, n_requests)

        # Create cost matrix
        cost_matrix = np.full(
            (max_size, max_size), 100.0
        )  # Default to 100 for invalid assignments

        # Fill in costs exactly as game calculates them
        for i, piece in enumerate(pieces):
            if not fits_on_plate(piece):
                continue

            area = piece.area
            for j, request in enumerate(self.requests):
                # Calculate percentage deviation exactly as game does
                penalty_percentage = 100 * abs(area - request) / request

                # Only add penalty if it exceeds tolerance
                if penalty_percentage > self.tolerance:
                    cost_matrix[i, j] = penalty_percentage
                else:
                    cost_matrix[i, j] = 0  # No penalty if within tolerance

        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Create assignments list
        assignments = [-1] * n_requests
        for piece_idx, request_idx in zip(row_ind, col_ind):
            if piece_idx < n_pieces and request_idx < n_requests:
                assignments[request_idx] = int(piece_idx)

        return constants.ASSIGN, assignments

    def is_valid_cut(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        existing_points: list[tuple[float, float]],
    ) -> bool:
        """Check if a cut is valid, allowing sequential cuts but preventing close parallel cuts"""
        new_line = LineString([start, end])

        for point in existing_points[:-1]:
            if point not in (start, end):
                if Point(point).distance(new_line) < 0.001:
                    return False

        return True