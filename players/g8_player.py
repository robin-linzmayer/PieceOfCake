import os
import pickle
from typing import List

import logging

import numpy as np

from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split

from scipy.optimize import linear_sum_assignment

import miniball

import constants
from piece_of_cake_state import PieceOfCakeState


class G8_Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        tolerance: int,
    ) -> None:
        """Initialise the player with the basic information

        Args:
            rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
            logger (logging.Logger): logger use this like logger.info("message")
            precomp_dir (str): Directory path to store/load pre-computation
            tolerance (int): tolerance for the cake distribution
            cake_len (int): Length of the smaller side of the cake
        """

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

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

    def move(self, current_percept: PieceOfCakeState) -> tuple[int, List[int]]:
        """Function that retrieves the current state of the cake map and returns an cake movement

        Args:
            current_percept(TimingMazeState): contains current state information
        Returns:
            int: This function returns the next move of the user:
                WAIT = -1
                LEFT = 0
                UP = 1
                RIGHT = 2
                DOWN = 3
        """
        print(current_percept)
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        # On turn 1 we need to decide where to start.
        if turn_number == 1:
            # initialize these variables
            self.assignment = [-1 for _ in range(len(requests))]
            self.requests = sorted(requests)
            self.cake = polygons
            self.cake_len = cake_len
            self.cake_width = cake_width

            self.edges = [
                ((0, 0), (cake_width, 0)),  # top
                ((cake_width, 0), (cake_width, cake_len)),  # right
                ((cake_width, cake_len), (0, cake_len)),  # bottom
                ((0, cake_len), (0, 0)),  # left
            ]

            start_position = self.decide_where_to_start(current_percept)
            self.cut_path.append(start_position)
            return constants.INIT, start_position

        if len(self.requests) == 0:
            return self.assign_polygons(polygons, requests)

        # Cut the max request remaining
        max_request = self.requests.pop()

        # We could also assign here early instead of doing it later
        print("cutting for request: ", max_request)
        request_index = requests.index(max_request)
        self.assignment[request_index] = len(polygons)

        if len(self.cut_path) == 1:
            base_left = 0
        else:
            base_left = self.cut_path[-2][0]  # Since we go up and down

        end_x = (2 * max_request / cake_len) + base_left
        end_x = round(end_x, 2)
        end_x = min(end_x, cake_width)

        end_y = cake_len if cur_pos[1] == 0 else 0
        self.cut_path.append([end_x, end_y])
        return constants.CUT, [end_x, end_y]

    def decide_where_to_start(self, current_percept):
        return [0, 0]

    def evaluate_cut_sequence(
        self, points: list[tuple[float, float]]
    ) -> tuple[float, float]:
        """Evaluate a sequence of cut endpoints, returning (penalty, total_cut_length)"""
        # Convert points to cuts
        cuts = []
        for i in range(len(points) - 1):
            cuts.append(LineString([points[i], points[i + 1]]))

        # Split cake into pieces
        pieces = [self.cake]
        for cut in cuts:
            new_pieces = []
            for piece in pieces:
                if cut.intersects(piece):
                    split_result = split(piece, cut)
                    new_pieces.extend(list(split_result.geoms))
                else:
                    new_pieces.append(piece)
            pieces = new_pieces

        # Calculate penalties (handles both area mismatches and plate fitting)
        penalties = self.calculate_penalties(pieces)
        return penalties, self.calculate_cut_length(points)

    def generate_initial_points(self):
        """Generate possible starting points along the perimeter"""
        points = []
        samples = 10

        for (x1, y1), (x2, y2) in self.edges:
            for t in np.linspace(0, 1, samples):
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                points.append((x, y))

        return points

    def generate_next_points(self, current_point: tuple[int, int]):
        """Generate possible next points on valid edges"""
        points = []
        samples = 10

        current_edge = self.get_edge(current_point)

        for i, (x1, y1), (x2, y2) in enumerate(self.edges):
            if i != current_edge:  # Can't cut to same edge
                for t in np.linspace(0, 1, samples):
                    x = x1 + t * (x2 - x1)
                    y = y1 + t * (y2 - y1)
                    points.append((x, y))

        return points

    def calculate_penalties(self, pieces: list[Polygon]):
        """Calculate penalties using Hungarian algorithm"""
        n_pieces = len(pieces)
        n_requests = len(self.requests)

        # Make the cost matrix rectangular instead of square
        cost_matrix = np.zeros((max(n_pieces, n_requests), max(n_pieces, n_requests)))

        # Fill with high penalties for the empty/excess slots
        cost_matrix.fill(100)

        # Fill in the actual penalties for existing pieces and requests
        for i, piece in enumerate(pieces):
            if not self.fits_on_plate(piece):
                # If piece doesn't fit, set penalty to 100 for all possible assignments
                cost_matrix[i, :n_requests] = 100
            else:
                area = piece.area
                for j, request in enumerate(self.requests):
                    deviation = abs(area - request) / request * 100
                    penalty = max(0, deviation - self.tolerance)
                    cost_matrix[i, j] = penalty

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return cost_matrix[row_ind, col_ind].sum()

    def calculate_cut_length(self, points: list[tuple[float, float]]) -> float:
        """Calculate total length of cut sequence"""
        total_length = 0

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            total_length += np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        return total_length

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

    # def cut_psuedo_cake(self):
    #     """Works to find the optimal cuts givin the initial state"""
    #     new_pieces = []
    #     for polygon in self.inital_state.polygons:
    #         line_points = LineString([tuple(self.cur_pos), tuple(action[1])])
    #         slices = self.divide_polygon(polygon, line_points)
    #         for cut in slices:
    #             new_pieces.append(cut)
    #
    def divide_polygon(self, polygon, line):
        """
        Divide a convex polygon by a line segment into two polygons.

        Parameters:
        - polygon: A convex polygon (as a Shapely Polygon object)
        - line_points: A list containing two points that represent the line segment

        Returns:
        - Two polygons (as shapely Polygon objects) that result from dividing the original polygon
        """
        # Create the convex polygon and the line segment using Shapely
        # polygon = Polygon(polygon_points)
        # line = LineString(line_points)

        # Check if the line intersects with the polygon
        if not line.intersects(polygon):
            return [polygon]
        # Split the polygon into two pieces
        result = split(polygon, line)

        # Convert the result into individual polygons

        polygons = []
        for p in result.geoms:
            polygons.append(p)

        return polygons

    def fits_on_plate(self, cake_piece, radius=12.5):
        """
        Check if the cake can fit inside a plate of radius 12.5.

        Parameters:
        - cake_pieces: Cake pieces (as shapely Polygon object)
        - radius: The radius of the circle

        Returns:
        - True if the cake can fit inside the plate, False otherwise
        """
        # Step 1: Get the points on the cake piece and store as numpy array

        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )

        # Step 2: Find the minimum bounding circle of the cake piece
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def assign_polygons(self, polygons, requests: List[float]):
        assignments = [-1] * len(requests)
        request_indices = sorted(
            range(len(requests)), key=lambda i: requests[i], reverse=True
        )  # Sort requests by size (largest to smallest)

        # Keep track of available polygons
        available_polygons = list(range(len(polygons)))

        for req_idx in request_indices:
            request_size = requests[req_idx]
            best_polygon = None
            min_diff = float("inf")  # Set a very high initial value for the difference

            # Find the polygon with the closest area to the current request
            for poly_idx in available_polygons:
                polygon_area = polygons[poly_idx].area
                diff = abs(polygon_area - request_size)

                if diff < min_diff:
                    min_diff = diff
                    best_polygon = poly_idx

            # Assign the best polygon and remove it from the available list
            assignments[req_idx] = best_polygon
            available_polygons.remove(best_polygon)

            print(
                f"Assigning polygon {best_polygon} to request {req_idx}, area: {polygons[best_polygon].area:.2f}, request: {request_size:.2f}"
            )

        return constants.ASSIGN, assignments
