import os
import pickle
from typing import List

import numpy as np
import logging

import constants


class Player:
    def __init__(self, rng: np.random.Generator, logger: logging.Logger,
                 precomp_dir: str, tolerance: int) -> None:
        """Initialise the player with the basic information

            Args:
                rng (np.random.Generator): numpy random number generator, use this for same player behavior across run
                logger (logging.Logger): logger use this like logger.info("message")
                precomp_dir (str): Directory path to store/load pre-computation
                tolerance (int): tolerance for the cake distribution
                cake_len (int): Length of the smaller side of the cake
        """
        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.knife_pos = [0, 0]  # Persistent knife position for continuity
        self.num_requests_cut = 0

    def move(self, current_percept) -> (int, List[int]):
        """Perform valid cuts based on area-based strategy and assign pieces."""
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        requests = sorted(current_percept.requests)  # Sorted in ascending order
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        # First turn initializes knife position at the top-left corner
        if turn_number == 1:
            self.knife_pos = [0, 0]
            return constants.INIT, [0, 0]

        # Cutting Strategy
        if len(polygons) < len(requests):
            # Start by dividing the cake in the shorter dimension (width or length)
            if turn_number == 2:
                if cake_len < cake_width:
                    cut_position = cake_len / 2
                    self.knife_pos = [0, cut_position]
                    return constants.CUT, [cake_width, cut_position]  # Horizontal cut
                else:
                    cut_position = cake_width / 2
                    self.knife_pos = [cut_position, 0]
                    return constants.CUT, [cut_position, cake_len]  # Vertical cut

            # For subsequent cuts, alternate between width and length divisions
            next_request_area = requests[len(polygons)]
            if self.knife_pos[1] == cake_len:  # Current position is at top or bottom
                cut_x = round(self.knife_pos[0] + next_request_area / cake_len, 2)
                self.knife_pos = [cut_x, 0] if self.knife_pos[1] == cake_len else [cut_x, cake_len]
                return constants.CUT, [cut_x, self.knife_pos[1]]
            else:  # Current position is at left or right side
                cut_y = round(self.knife_pos[1] + next_request_area / cake_width, 2)
                self.knife_pos = [0, cut_y] if self.knife_pos[0] == cake_width else [cake_width, cut_y]
                return constants.CUT, [self.knife_pos[0], cut_y]

        # Assignment Strategy
        assignment = self._assign_requests(polygons, requests)
        return constants.ASSIGN, assignment

    def _assign_requests(self, polygons, requests) -> List[int]:
        """Assign each polygon piece to the closest-sized request to minimize penalty."""
        assignments = []
        polygon_areas = [p.area for p in polygons]
        
        # Sort polygons and requests by area in ascending order
        polygon_indices = list(np.argsort(polygon_areas))
        request_indices = list(np.argsort(np.argsort(requests)))

        # Assign each request to a polygon based on area matching
        for request_idx in request_indices:
            assignments.append(polygon_indices[request_idx])

        return assignments