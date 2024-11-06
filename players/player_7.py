import os
import pickle
from typing import List, Tuple
import numpy as np
import logging
import constants
from shapely.geometry import Polygon

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

    def move(self, current_percept) -> Tuple[int, List[int]]:
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width

        print('Turn', turn_number, '\n')
        
        vertical_cuts = get_vertical_cuts(requests, self.cake_len, self.cake_width)
        cut_coords = inject_crumb_coords(vertical_cuts, self.cake_len, self.cake_width)
        cut_coords = inject_horizontal_cut(cut_coords, self.cake_len, self.cake_width)

        if turn_number == 1:
            return constants.INIT, cut_coords[0]        

        # print(len(cut_coords), cut_coords)

        if turn_number < len(cut_coords) + 1:
            return constants.CUT, cut_coords[turn_number - 1]
        
        assignment = self.assign_pieces(requests, polygons)
        return constants.ASSIGN, assignment
    
    def assign_pieces(self, requests: List[float], polygons: List[float]) -> List[int]:
        """Assigns pieces to requests based on area similarity within tolerance."""
        assignment = [-1] * len(requests)
        requests_sorted = sorted(enumerate(requests), key=lambda x: x[1])
        polygons_sorted = get_polygon_areas(polygons)

        for req_idx, req_size in requests_sorted:
            for i, (poly_idx, poly_size) in enumerate(polygons_sorted):
                if abs(poly_size - req_size) <= (req_size * self.tolerance / 100):
                    assignment[req_idx] = poly_idx
                    polygons_sorted.pop(i)
                    break
        return assignment

def get_vertical_cuts(requests, cake_len, cake_width):
    sorted_requests = sorted(requests, reverse=True)

    # TODO handle odd number of requests!
    pairs = [sorted_requests[i] + sorted_requests[i + 1] for i in range(0, len(sorted_requests), 2)]

    x_widths = [pair / cake_len for pair in pairs]
    x_widths.append(cake_width - sum(x_widths))

    x_coords = [x_widths[0]]
    for i in range(1, len(x_widths)):
        x_coords.append(x_coords[i - 1] + x_widths[i])

    x_coords = [round(x, 2) for x in x_coords for _ in range(2)]
    y_coords = [0, cake_len, cake_len, 0]
        
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

def inject_horizontal_cut(vertical_cuts, cake_len, cake_width):
    # define start as the x coordinate of the first vertical cut
    return [[cake_width, cake_len/2], [0, cake_len/2], [0.01, 0], [0, 0.01]] + vertical_cuts