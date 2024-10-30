import os
import pickle
from typing import List, Tuple
from collections import deque

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
        self.cake_len = None
        
        self.num_splits = 1
        self.horizontal_split_gap = 24.6
        self.triangle_viable = 23.507
        self.vertical_split_gap = 4
        self.preplanned_moves = deque()
        self.request_served = 0

    def move(self, current_percept) -> Tuple[int, List[int]]:
        self.init_cuts(current_percept=current_percept)  # TODO: just for debug and demonstration, move elsewhere later
        if current_percept.cake_len <= self.triangle_viable:
            return self.triangle(current_percept)
        return self.quadrangle(current_percept)

    def init_cuts(self, current_percept) -> List[List[List[int]]]:
        """
        Given current_percept, returns the list of signficant (non-crumb) cuts to be made in the cake in format:
        [
            [[0, y1], [width, y1]],  # horizontal cut
            [[0, y2], [width, y2]],  # horizontal cut
            ...
            [[x3, 0], [x3, length]],  # vertical cut
            [[x4, 0], [x4, length]],  # vertical cut
        ]
        """
        cake_len, cake_width = current_percept.cake_len, current_percept.cake_width
        row_count, col_count = 2, 4
        row_height, col_width = cake_len / row_count, cake_width / col_count
        cuts = []

        for i in range(1, row_count):
            cuts.append([[0, i * row_height], [cake_width, i * row_height]])
        for i in range(1, col_count):
            cuts.append([[i * col_width, 0], [i * col_width, cake_len]])

        print(f"{cake_len=}, {cake_width=}")
        print(f"init_{cuts=}")
        return cuts

    def quadrangle(self, current_percept):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        if turn_number == 1:
            return constants.INIT, [0, 0.01]
        
        if turn_number == 2:
            cur_x, cur_y = cur_pos[0], cur_pos[1]
            
            while cur_y + self.horizontal_split_gap < cake_len:
                self.shift_along((cur_x, cur_y), (cur_x, cur_y + self.horizontal_split_gap), cake_len, cake_width)
                cur_x = cake_width if cur_x == 0 else 0
                cur_y = cur_y + self.horizontal_split_gap
                self.preplanned_moves.append([cur_x, cur_y])
                self.num_splits += 1
                
            if cur_x == cake_width:
                self.preplanned_moves.append([cake_width - 0.01, cake_len])
                self.preplanned_moves.append([0, cake_len - 0.01])
            self.preplanned_moves.append([0.01, cake_len])
        
        if self.preplanned_moves:
            dest_x, dest_y = self.preplanned_moves.popleft()
            return constants.CUT, [round(dest_x, 2), round(dest_y, 2)]
        
        valid_polygons = [poly for poly in polygons if poly.area > 0.5]
        if len(valid_polygons) < len(requests):
            if cur_pos[1] == 0:
                self.shift_along(cur_pos, [cur_pos[0] + self.vertical_split_gap, 0], cake_len, cake_width)
                self.preplanned_moves.append([cur_pos[0] + self.vertical_split_gap, cake_len])
            else:
                self.shift_along(cur_pos, [cur_pos[0] + self.vertical_split_gap, cake_len], cake_len, cake_width)
                self.preplanned_moves.append([cur_pos[0] + self.vertical_split_gap, 0])
                
            dest_x, dest_y = self.preplanned_moves.popleft()
            return constants.CUT, [round(dest_x, 2), round(dest_y, 2)]
        
        assignment = []
        for i in range(len(requests)):
            assignment.append(i)

        return constants.ASSIGN, assignment

    def shift_along(self, cur_pos, target_pos, cake_len, cake_width):
        if cur_pos[0] == 0:
            if cur_pos[1] < cake_len / 2:
                self.preplanned_moves.append([0.01, 0])
            else:
                self.preplanned_moves.append([0.01, cake_len])
        elif cur_pos[0] == cake_width:
            if cur_pos[1] < cake_len / 2:
                self.preplanned_moves.append([cake_width - 0.01, 0])
            else:
                self.preplanned_moves.append([cake_width - 0.01, cake_len])
        elif cur_pos[1] == 0:
            if cur_pos[0] < cake_width / 2:
                self.preplanned_moves.append([0, 0.01])
            else:
                self.preplanned_moves.append([cake_width, 0.01])
        else:
            if cur_pos[0] < cake_width / 2:
                self.preplanned_moves.append([0, cake_len - 0.01])
            else:
                self.preplanned_moves.append([cake_width, cake_len - 0.01])
        self.preplanned_moves.append([target_pos[0], target_pos[1]])

    def triangle(self, current_percept):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = sorted(current_percept.requests)
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width

        if turn_number == 1:
            self.preplanned_moves.append([0,0])
            return constants.INIT, [0,0]
        
        if self.request_served < len(requests):
            area = requests[self.request_served]
            base = round(2 * area / cake_len, 2)

            cur_x, cur_y = cur_pos[0], cur_pos[1]

            if turn_number == 2:
                if cake_len ** 2 + base ** 2 > 25 ** 2:
                    raise Exception("First cut doesn't fit on plate.")
                dest_x, dest_y = base, cake_len
            else:
                dest_x = round(self.preplanned_moves[-2][0] + base, 2)
                dest_y = cake_len if cur_y == 0 else 0

                if dest_x > cake_width:
                    l1 = dest_x - cake_width
                    l2 = cake_width - self.preplanned_moves[-1][0]
                    h1 = (cake_len * (l1)) / l2
                    h2 = cake_len - h1
                    h3 = (h1 * (l1)) / l2
                    new_y = round(h2 - h3, 2)
                    dest_y = new_y if cur_y == 0 else round(cake_len - new_y, 2)
                    dest_x = cake_width

            self.preplanned_moves.append([dest_x, dest_y])
            self.request_served += 1
            return constants.CUT, [dest_x, dest_y]
        
        assignment = []
        polygon_idx = list(np.argsort([polygon.area for polygon in polygons]))

        if len(requests) == 1:
            polygon_idx.remove(0)
        elif len(polygon_idx) > 1:
            polygon_idx.remove(len(polygon_idx) // 2)
        
        req_idx = list(np.argsort(np.argsort(current_percept.requests)))
        
        for idx in req_idx:
            assignment.append(int(polygon_idx[idx]))
        return constants.ASSIGN, assignment 


