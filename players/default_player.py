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
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        time_remaining = current_percept.time_remaining

        if turn_number == 1:
            print()
            return constants.INIT, [0,0]

        if len(polygons) < len(requests):
            if cur_pos[0] == 0:
                return constants.CUT, [cake_width, round((cur_pos[1] + 5)%cake_len, 2)]
            else:
                return constants.CUT, [0, round((cur_pos[1] + 5)%cake_len, 2)]

        assignment = []
        for i in range(len(requests)):
            assignment.append(i)

        return constants.ASSIGN, assignment
