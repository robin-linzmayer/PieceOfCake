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
        self.cuts = []  # Planned cuts
        self.assignments = []
        self.excess_cut_active = False  # Flag for managing excess cuts

    def plan_cuts(self, requests: List[float]) -> List[Tuple[int, float]]:
        """Plan initial cuts based on the customer requests, dividing the cake accordingly."""
        total_area = self.cake_len * self.cake_width
        sorted_requests = sorted(requests, reverse=True)
        cut_positions = []
        current_y = 0
        
        for request_size in sorted_requests:
            if current_y >= self.cake_len:
                break
            requested_length = request_size / self.cake_width
            end_y = min(current_y + requested_length, self.cake_len)
            cut_positions.append((end_y, request_size))
            current_y = end_y
        
        self.logger.info(f"Planned cuts: {cut_positions}")
        return cut_positions

    def move(self, current_percept) -> (int, List[int]):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width

        if turn_number == 1:
            self.cuts = self.plan_cuts(requests)
            self.assignments = self.cuts
            return constants.INIT, [0, 0]

        # Check if all requests have been fulfilled
        if len(polygons) >= len(requests):
            assignment = self.assign_pieces(requests, polygons)
            
            # End condition if all requests have matching pieces
            if all(a != -1 for a in assignment):
                if not self.excess_cut_active:
                    self.excess_cut_active = True
                    self.cuts = self.plan_excess_cuts(polygons)
                    return self.make_excess_cut(cur_pos)
                return constants.ASSIGN, assignment

        # Continue performing planned cuts if there are any left
        if self.cuts and not self.excess_cut_active:
            end_y, _ = self.cuts.pop(0)
            return constants.CUT, [self.cake_width if cur_pos[0] == 0 else 0, round(end_y, 2)]

        # Make excess cuts if needed
        if self.excess_cut_active and self.cuts:
            return self.make_excess_cut(cur_pos)

        # Fallback if no cuts left
        self.logger.warning("Fallback: no cuts left but game didn't end.")
        return constants.ASSIGN, self.assign_pieces(requests, polygons)

    def make_excess_cut(self, cur_pos):
        """Handles additional cuts to remove any remaining excess."""
        end_y, _ = self.cuts.pop(0)
        if cur_pos[0] == 0:
            return constants.CUT, [self.cake_width, round(end_y, 2)]
        else:
            return constants.CUT, [0, round(end_y, 2)]

    def plan_excess_cuts(self, polygons) -> List[Tuple[int, float]]:
        """Plans additional cuts to remove excess unassigned portions of the cake."""
        excess_cuts = []
        polygons_sorted = get_polygon_areas(polygons)
        
        for i, (index, area) in enumerate(polygons_sorted):
            if area > self.cake_width * 10:  # Arbitrary threshold to identify larger excess pieces
                excess_cut_pos = (i + 1) * (self.cake_len / len(polygons_sorted))  # Evenly space cuts
                excess_cuts.append((excess_cut_pos, area))
        
        self.logger.info(f"Planned excess cuts: {excess_cuts}")
        return excess_cuts

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
