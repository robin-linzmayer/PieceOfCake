from typing import List, Tuple, Callable
from sklearn.mixture import GaussianMixture
import numpy as np
import math

import logging
import constants
from shapely.geometry import Polygon

import miniball

def get_polygon_areas(polygons):
    """Calculates and returns sorted polygon indices based on their area."""
    polygons_with_area = [(i, polygon.area) for i, polygon in enumerate(polygons)]
    sorted_polygons = sorted(polygons_with_area, key=lambda x: x[1])
    return sorted_polygons  # Each tuple is (index, area)

class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, tolerance: int) -> None:
        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        self.cut_coords = None
        self.splits = 0
        self.best_ratio = [float('inf'), -1]

    def can_cake_fit_in_plate(self, cake_piece, radius=12.5):
        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def _calculate_penalty(
        self, assign_func: Callable[[List[Polygon], List[float]], List[int]]
    ) -> float:
        penalty = 0
        assignments: List[int] = assign_func(self.polygons, self.requests)

        for request_index, assignment in enumerate(assignments):
            # Check if the cake piece fits on a plate of diameter 25
            if assignment == -1 or not self.can_cake_fit_in_plate(self.polygons[assignment]):
                penalty += 100
            else:
                # Calculate penalty based on the difference in area
                penalty_percentage = (
                    100
                    * abs(self.polygons[assignment].area - self.requests[request_index])
                    / self.requests[request_index]
                )
                # Add penalty if it exceeds tolerance
                if penalty_percentage > self.tolerance:
                    penalty += penalty_percentage
        return penalty

    def move(self, current_percept) -> Tuple[int, List[int]]:
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width

        self.polygons = polygons
        self.requests = requests
                
        # Compute assignment and penalty for this configuration
        # penalty = self._calculate_penalty(lambda polys, reqs: self.assign_pieces(reqs, polys))
        # print("penalty",num_hor_cuts, penalty)
        # # Update best configuration if this one has a lower penalty
        # if penalty < best_penalty:
        #     best_penalty = penalty
        #    best_cut_config = (num_hor_cuts, cut_coords)
        
        # Use the best configuration’s cuts
        # best_num_hor_cuts, best_cut_coords = best_cut_config

        if turn_number == 1:
                    # num_hor_cuts = 2
            self.splits = calc_num_splits(requests, self.cake_len)

            if self.splits >= 1:
                # Generate vertical cuts based on requests and grouping ratio
                vertical_cuts = self.generate_vertical_cuts(requests, num_cuts=self.splits) # Preserving vertical grouping ratios
            else:
                vertical_cuts = self.generate_vertical_cuts(requests, num_cuts=0, final=True) # Preserving vertical grouping ratios
            
            # Inject the specified number of horizontal cuts
            self.cut_coords = inject_crumb_coords(vertical_cuts, self.cake_len, self.cake_width)
            self.cut_coords = inject_horizontal_cuts(self.cut_coords, self.cake_len, self.cake_width, self.splits)

            return constants.INIT, self.cut_coords[0]
        
        if turn_number < len(self.cut_coords) + 1:
            return constants.CUT, self.cut_coords[turn_number - 1]
        
        # areas =[i.area for i in polygons]
        # assignment = sorted(range(len(areas)), key=lambda x: areas[x], reverse=True)
        # print(assignment[:len(requests)])
        # return constants.ASSIGN, assignment[:len(requests)][::-1]

        assignment = self.assign_pieces(requests, polygons)
        return constants.ASSIGN, assignment
    
    def assign_pieces(self, requests: List[float], polygons: List[float]) -> List[int]:
        """Assign each request to the polygon with the closest area, minimizing the sum of differences."""
        
        assignment = [-1] * len(requests)
        
        requests_sorted = sorted(enumerate(requests), key=lambda x: x[1])
        polygons_sorted = get_polygon_areas(polygons)
        
        j = 0 
        for req_idx, req_size in requests_sorted:
            best_poly_idx = -1
            best_diff = float('inf')
            
            for k in range(j, len(polygons_sorted)):
                poly_idx, poly_size = polygons_sorted[k]
                diff = abs(poly_size - req_size)
                
                if diff < best_diff:
                    best_diff = diff
                    best_poly_idx = k
                
                if poly_size > req_size:
                    break

            if best_poly_idx != -1:
                poly_idx, poly_size = polygons_sorted.pop(best_poly_idx)
                assignment[req_idx] = poly_idx
                j = max(0, j - 1)

        self.polygons = polygons

        return assignment

    def generate_vertical_cuts(self, requests, num_cuts, final=False):
        """Generates vertical cuts with consistent ratio grouping for requests."""
        
        # Copy requests to avoid modifying the original list
        requests_copy = requests[:]
        group_size = num_cuts + 1
        #group_size = num_cuts + 1 if isinstance(num_cuts, int) else len(requests_copy) // 3

        # Ensure the number of requests is a multiple of the group size
        if len(requests_copy) % group_size != 0:
            required_requests = group_size - (len(requests_copy) % group_size)
            
            # Add "fake" requests based on the median to keep distribution consistent
            gmm = GaussianMixture(n_components=min(2, len(requests_copy)), random_state=0)
            gmm.fit(np.array(requests_copy).reshape(-1, 1))
            smallest_cluster_mean = gmm.means_.flatten()[np.argmin(np.unique(gmm.predict(np.array(requests_copy).reshape(-1, 1)), return_counts=True)[1])]
            requests_copy.extend([smallest_cluster_mean] * required_requests)

        # Sort requests and divide into groups based on the specified group size
        sorted_requests = sorted(requests_copy)
        groups = [sorted_requests[i:i + group_size] for i in range(0, len(sorted_requests), group_size) if len(sorted_requests[i:i + group_size]) == group_size]
  
        # Calculate average ratios for consistent group scaling
        avg_ratios = [sum(group[j + 1] / group[j] for group in groups if group[j] != 0) / len(groups) for j in range(group_size - 1)]

        # bail on this split if it sucks
        if not final:
            ratio = max(avg_ratios) ** self.splits
            if ratio > 1 + self.tolerance * 0.01:
                if ratio < self.best_ratio[0]:
                    self.best_ratio = [ratio, self.splits]

                if self.splits < len(requests):
                    self.splits += 1
                    return self.generate_vertical_cuts(requests, self.splits)
                else:
                    self.splits = self.best_ratio[1]
                    return self.generate_vertical_cuts(requests, self.splits, final=True)
        
        # Adjust groups to align with average ratios for consistency
        for group in groups:
            for j in range(group_size - 1):
                group[j + 1] = group[j] * avg_ratios[j]

        # Generate x-widths for vertical cuts based on adjusted groups
        x_widths = [sum(group) / self.cake_len for group in groups]
        x_widths.append(self.cake_width - sum(x_widths))
        x_coords = [x_widths[0]]
        for i in range(1, len(x_widths)):
            x_coords.append(x_coords[i - 1] + x_widths[i])
        # x_coords = [round(sum(x_widths[:i+1]), 2) for i in range(len(x_widths))]
        x_coords = [round(x, 2) for x in x_coords for _ in range(2)]
        y_coords = [0, self.cake_len, self.cake_len, 0]
        # Format x and y coordinates for vertical cuts
        # x_coords = [x for coord in x_coords for x in [coord, coord]]
        # y_coords = [0, self.cake_len, self.cake_len, 0]
        vertical_cuts = [[x, y] for x, y in zip(x_coords, y_coords * (len(x_coords) // 4))]

        return vertical_cuts

def inject_crumb_coords(vertical_cuts, cake_len, cake_width):
    final_cuts = []

    for i, cut in enumerate(vertical_cuts):
        final_cuts.append(cut)

        if (i + 1) % 2 == 0:
            final_cuts.append(get_crumb_coord(cut, cake_len, cake_width))

    return final_cuts

def get_crumb_coord(cut, cake_len, cake_width):
    x = cake_width if cut[0] > (cake_width / 2) else 0
    knife_error = 0.01
    y = round(cake_len - knife_error, 2) if cut[1] == cake_len else knife_error
    return [x, y]

def inject_horizontal_cuts(vertical_cuts, cake_len, cake_width, num_hor_cuts):
    """Injects a specified number of horizontal cuts into the cake."""

    horizontal_cuts = []
    cake_width = round(cake_width, 2)
    total_y = 0
    for i in range(num_hor_cuts):
        total_y = round((cake_len / (num_hor_cuts + 1)) * (i + 1), 2)
        horizontal_cuts.extend([ [0, total_y], [cake_width, total_y], [0, round(total_y+0.01, 2)], [0.01*(i+1), 0]])

    horizontal_cuts.append([0, 0.01])

    return horizontal_cuts + vertical_cuts

def calc_num_splits(requests, cake_len):
    """Calculates the number of horizontal cuts based on the number of requests and the number of vertical cuts."""
    largest_req = max(requests)

    for i in range(50):
        length = cake_len / (i + 1)
        width = largest_req / length

        diagonal = (length ** 2 + width ** 2) ** 0.5

        if diagonal <= 25:
            return i
        
    return 51
