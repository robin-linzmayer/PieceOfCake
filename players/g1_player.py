import os
import pickle
from typing import List
from scipy.optimize import linear_sum_assignment


import numpy as np
import logging

import constants

def sorted_assignment(R, V):
    assignment = []  
    # list of indices of polygons sorted by area
    polygon_indices = list(np.argsort(V))

    # remove the last piece from the list of polygons
    if len(R) == 1:
        polygon_indices.remove(0)
    elif len(V) > 1:
        last_piece_idx = len(V) // 2
        polygon_indices.remove(last_piece_idx)

    # list of indices of requests sorted by area in ascending order
    request_indices = list(np.argsort(np.argsort(R)))

    # Assign polygons to requests by area in ascending order
    for request_idx in request_indices:
        assignment.append(polygon_indices[request_idx])
    return assignment
    
def optimal_assignment(R, V):
    V.remove(V[len(V) // 2])
    num_requests = len(R)
    num_values = len(V)
    
    cost_matrix = np.zeros((num_requests, num_values))
    
    # Fill the cost matrix with relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    # Solving the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Assignment array where assignment[i] is the index of V matched to R[i]
    assignment = [col_indices[i] for i in range(num_requests)]
    assignment = [i+1 if (i >= len(assignment) // 2) else i for i in assignment]
    
    return assignment

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
        self.EASY_LEN_BOUND = 23.507
        self.num_requests_cut = 0
        self.knife_pos = []

    def get_starting_pos(self, requests):
        area = sum(requests) * 1.05
        h = np.sqrt(area / 1.6)
        w = h * 1.6
        return [0, round(h * (1/11), 2)]
        # TODO: Actual logic

    


    def move(self, current_percept) -> (int, List[int]):
        """Function which returns an action.

            Args:
                current_percept(PieceOfCakeState): contains current cake state information
            Returns:
                a tuple of format ([action=1,2,3], [list])
                if action = 1: list is [0,0]
                if action = 2: list is [x, y] (coordinates for knife position after cut)
                if action = 3: list is [p1, p2, ..., pl] (the order of polygon assignment)

        """
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width


        ####################
        # CUTTING STRATEGY #
        ####################

        # sort requests by area in ascending order
        requests = sorted(requests)
        num_requests = len(requests)
        cake_area = cake_len * cake_width

        # slice off 5% extra if only one request
        if num_requests == 1:
            if turn_number == 1:
                return constants.INIT, [0,0]
            elif self.num_requests_cut < num_requests:
                extra_area = 0.05 * requests[0]
                extra_base = round(2 * extra_area / cake_len, 2)
                self.num_requests_cut += 1
                return constants.CUT, [extra_base, cake_len]

        # all other non-edge cases
        elif num_requests > 1:
            if cake_len <= self.EASY_LEN_BOUND:
                # initialize starting knife position
                if turn_number == 1:
                    self.knife_pos.append([0,0])
                    return constants.INIT, [0,0]
                
                if self.num_requests_cut < num_requests:
                    # compute size of base from current polygon area
                    curr_polygon_area = requests[self.num_requests_cut]
                    curr_polygon_base = round(2 * curr_polygon_area / cake_len, 2)

                    if cur_pos[1] == 0:
                        # knife is currently on the top cake edge
                        if turn_number == 2:
                            next_knife_pos = [curr_polygon_base, cake_len]
                        else:
                            next_x = round(self.knife_pos[-2][0] + curr_polygon_base, 2)
                            next_y = cake_len
                            # when knife goes over the cake width
                            if next_x > cake_width:
                                next_x = cake_width
                                next_y = round(2 * cake_area * 0.05 / (cake_width - self.knife_pos[-2][0]), 2)
                            next_knife_pos = [next_x, next_y]

                        self.knife_pos.append(next_knife_pos)
                        self.num_requests_cut += 1
                        return constants.CUT, next_knife_pos
                    else:
                        # knife is currently on the bottom cake edge
                        next_x = round(self.knife_pos[-2][0] + curr_polygon_base, 2)
                        next_y = 0
                        # when knife goes over the cake width
                        if next_x > cake_width:
                            next_x = cake_width
                            next_y = cake_len - round(2 * cake_area * 0.05 / (cake_width - self.knife_pos[-2][0]), 2)
                        next_knife_pos = [next_x, next_y]
                        self.knife_pos.append(next_knife_pos)
                        self.num_requests_cut += 1
                        return constants.CUT, next_knife_pos

            # case where cake is larger than EASY_LEN_BOUND
            else:
                if turn_number == 1:
                    self.angle = np.radians(17)
                    # self.angle = 10
                    starting_pos = self.get_starting_pos(requests)
                    self.knife_pos.append(starting_pos)
                    return constants.INIT, starting_pos
                    

                if self.num_requests_cut < num_requests:
                    # Left side
                    if cur_pos[0] == 0:
                        # Right/Top -> Left -> Down
                        if current_percept.turn_number > 2 and (
                            self.knife_pos[-2][1] == 0 or (self.knife_pos[-2][0] == cake_width and self.knife_pos[-3][1] == 0)):
                            l = (cake_len - cur_pos[1]) / np.tan(self.angle)
                            if l > cake_width:
                                x = cake_width
                                y = cake_len - ((l - cake_width) * np.tan(self.angle))
                            else:
                                x = l
                                y = cake_len

                        # Bottom -> Left -> Up
                        else:
                            l = cur_pos[1] / np.tan(self.angle)
                            if l > cake_width:
                                x = cake_width
                                y = (l - cake_width) * np.tan(self.angle)
                            else:
                                x = l
                                y = 0

                    # Right side
                    elif cur_pos[0] == cake_width:
                        # Left/Down -> Right -> Up
                        if self.knife_pos[-2][0] == 0 or self.knife_pos[-2][1] == cake_len:
                            l = cur_pos[1] / np.tan(self.angle)
                            if l > cake_width:
                                x = 0
                                y = np.tan(self.angle) * (l - cake_width)
                            else:
                                x = cake_width - l
                                y = 0
                        # Top -> Right -> Down
                        else:
                            l = (cake_len - cur_pos[1]) / np.tan(self.angle)
                            if l > cake_width:
                                x = 0
                                y = cake_len - (np.tan(self.angle) * (l - cake_width))
                            else:
                                x = cake_width - l
                                y = cake_len
                        # self.knife_pos[-2][1] / cake_width - cur_pos[0]       


                    # Top
                    elif cur_pos[1] == 0:
                        # This only works now if starting from left/right
                        l = (cake_width - cur_pos[1]) * np.tan(self.angle)
                        if l > cake_len:
                            x = cake_width - ((l - cake_len) * np.tan(self.angle))
                            y = cake_len
                        else:
                            # Left -> Right
                            if self.knife_pos[-2][0] == 0 or (self.knife_pos[-2][1] == cake_len and self.knife_pos[-3][0] == cake_len):
                                x = cake_width
                                y = np.tan(self.angle) * (cake_width - cur_pos[0])
                            # Right -> Left
                            else:
                                x = 0
                                y = np.tan(self.angle) * cur_pos[0]
                    
                    # Bottom
                    else:
                        # This only works now if starting from left/right
                        l = cur_pos[1] * np.tan(self.angle)
                        if l > cake_len:
                            x = (l - cake_len) * np.tan(self.angle)
                            y = 0
                        else:
                            # Left -> Right
                            if self.knife_pos[-2][0] == 0 or (self.knife_pos[-2][1] == 0 and self.knife_pos[-3][0] == 0):
                                x = cake_width
                                y = cake_len - (np.tan(self.angle) * (cake_width - cur_pos[0]))
                            # Right -> Left
                            else:
                                x = 0
                                y = cake_len - (np.tan(self.angle) * cur_pos[0])
                    
                    next_knife_pos = [round(x, 2), round(y, 2)]
                    self.knife_pos.append(next_knife_pos)
                    self.num_requests_cut += 1
                    return constants.CUT, next_knife_pos


        #######################
        # ASSIGNMENT STRATEGY #
        #######################
        V = [p.area for p in polygons]
        if cake_len <= self.EASY_LEN_BOUND:
            assignment = sorted_assignment(current_percept.requests, V)
        else:
            assignment = optimal_assignment(current_percept.requests, V)

        return constants.ASSIGN, assignment
    
    
    
