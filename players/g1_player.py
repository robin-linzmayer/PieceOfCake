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

        num_requests = len(requests)

        if cake_len <= self.EASY_LEN_BOUND:
            # initialize starting knife position
            if turn_number == 1:
                self.knife_pos.append([0,0])
                return constants.INIT, [0,0]
            
            if self.num_requests_cut != num_requests:
                curr_polygon_area = requests[self.num_requests_cut]
                curr_polygon_base = round(2 * curr_polygon_area / cake_len, 2)    # size of base needed for current polygon area

                if self.num_requests_cut == num_requests - 2:
                        # making the final cut, choose the opposite corner of cake
                        self.num_requests_cut += 2
                        return constants.CUT, [cake_width, 0] if num_requests % 2 == 1 else [cake_width, cake_len]

                if cur_pos[1] == 0:
                    # knife is currently on the top cake edge
                    if turn_number == 2:
                        next_knife_pos = [curr_polygon_base, cake_len]
                    else:
                        next_knife_pos = [round(self.knife_pos[-2][0] + curr_polygon_base, 2), cake_len]

                    self.knife_pos.append(next_knife_pos)
                    self.num_requests_cut += 1
                    return constants.CUT, next_knife_pos
                else:
                    # knife is currently on the bottom cake edge
                    next_knife_pos = [round(self.knife_pos[-2][0] + curr_polygon_base, 2), 0]
                    self.knife_pos.append(next_knife_pos)
                    self.num_requests_cut += 1
                    return constants.CUT, next_knife_pos

        # EDIT BELOW TO PROPERLY ASSIGN POLYGONS
        # Hacky assignment for now; only works for EASY_LEN_BOUND
        assignment = []
        indices = list(range(len(requests)))
        for i in range(len(requests)):
            idx = indices.pop(0) if i % 2 == 0 else indices.pop(-1)
            assignment.append(idx)

        return constants.ASSIGN, assignment