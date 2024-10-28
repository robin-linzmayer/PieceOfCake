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
        self.cake_width = None
        self.cuts = set()


    # Move the knife to the side in the clockwise direction
    def move_knife_clockwise(self, cur_pos):
        if cur_pos[0] == 0 and cur_pos[1] != 0:
            val = 0.01
            while [cur_pos[0], cur_pos[1], val, 0]  in self.cuts or [val, 0, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [val, 0]
        elif cur_pos[1] == 0 and cur_pos[0] != self.cake_width:
            val = 0.01
            while [cur_pos[0], cur_pos[1], self.cake_width, val] in self.cuts or [self.cake_width, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [self.cake_width, val]
        elif cur_pos[0] == self.cake_width and cur_pos[1] != self.cake_len:
            val = self.cake_width - 0.01
            while [cur_pos[0], cur_pos[1], val, self.cake_len] in self.cuts or [val, self.cake_len, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [val, self.cake_len]
        elif cur_pos[1] == self.cake_len and cur_pos[0] != 0:
            val = self.cake_len - 0.01
            while [cur_pos[0], cur_pos[1], 0, val] in self.cuts or [0, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [0, val]

    # Move the knife to the side in the anticlockwise direction
    def move_knife_anticlockwise(self, cur_pos):
        if cur_pos[0] == 0 and cur_pos[1] != self.cake_len:
            val = 0.01
            while [cur_pos[0], cur_pos[1], val, self.cake_len] in self.cuts or [val, self.cake_len, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [val, self.cake_len]
        elif cur_pos[1] == 0 and cur_pos[0] != 0:
            val = 0.01
            while [cur_pos[0], cur_pos[1], 0, val] in self.cuts or [0, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [0, val]
        elif cur_pos[0] == self.cake_width and cur_pos[1] != 0:
            val = self.cake_width - 0.01
            while [cur_pos[0], cur_pos[1], val, 0] in self.cuts or [val, 0, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [val, 0]
        elif cur_pos[1] == self.cake_len and cur_pos[0] != self.cake_width:
            val = self.cake_len - 0.01
            while [cur_pos[0], cur_pos[1], self.cake_width, val] in self.cuts or [self.cake_width, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [self.cake_width, val]

    def move(self, current_percept) -> (int, List[int]):
        """Function which retrieves the current state of the cake

            Args:
                current_percept(PieceOfCakeState): contains current state information
            Returns:
                (int, List[int]): This function returns the next move of the user:
                The integer return value should be one of the following:
                    constants.INIT - If wants to initialize the knife position
                    constants.CUT - If wants to cut the cake
                    constants.ASSIGN - If wants to assign the pieces
        """
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width

        if turn_number == 1:
            return constants.INIT, [0,0]

        if len(polygons) < len(requests):
            if cur_pos[0] == 0:
                return constants.CUT, [self.cake_width, round((cur_pos[1] + 5)%self.cake_len, 2)]
            else:
                return constants.CUT, [0, round((cur_pos[1] + 5)%self.cake_len, 2)]

        # Assign the pieces
        assignment = []
        for i in range(len(requests)):
            assignment.append(i)


        return constants.ASSIGN, assignment
