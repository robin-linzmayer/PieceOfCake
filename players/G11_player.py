import os
import pickle
from typing import List

import numpy as np
import logging

import constants
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import copy


class Player:
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

        if turn_number == 1:
            print()
            return constants.INIT, [0, 0]

        new_percept = copy.deepcopy(current_percept)
        new_polygons = copy.deepcopy(polygons)

        for i in range(50):
            if new_percept.cur_pos[0] == 0:
                try:
                    new_polygons, new_percept = self.check_and_apply_action(
                        [
                            constants.CUT,
                            [
                                new_percept.cake_width,
                                round(
                                    (new_percept.cur_pos[1] + 5) % new_percept.cake_len,
                                    2,
                                ),
                            ],
                        ],
                        new_polygons,
                        new_percept,
                    )

                    print(len(new_polygons))
                    print(new_percept.cur_pos)
                except ValueError as e:
                    print(f"Invalid cut 1 {e}")
            else:
                try:
                    new_polygons, new_percept = self.check_and_apply_action(
                        [
                            constants.CUT,
                            [
                                0,
                                round(
                                    (new_percept.cur_pos[1] + 5) % new_percept.cake_len,
                                    2,
                                ),
                            ],
                        ],
                        new_polygons,
                        new_percept,
                    )
                    print(len(new_polygons))
                    print(new_percept.cur_pos)

                except ValueError as e:
                    print(f"Invalid cut 2 {e}")

        assignment = []
        for i in range(len(requests)):
            assignment.append(i)
        return constants.ASSIGN, []

    def check_and_apply_action(self, action, polygons, current_percept):
        if not action[0] == constants.CUT:
            raise ValueError("Invalid action")

        cur_x, cur_y = action[1]

        # Check if the next position is on the boundary of the cake
        if invalid_knife_position(action[1], current_percept):
            raise ValueError("Invalid knife position")

        # Check if the cut is horizontal across the cake boundary
        if cur_x == 0 or cur_x == current_percept.cake_width:
            if current_percept.cur_pos[0] == cur_x:
                raise ValueError("Invalid cut")

        # Check if the cut is vertical across the cake boundary
        if cur_y == 0 or cur_y == current_percept.cake_len:
            if current_percept.cur_pos[1] == cur_y:
                raise ValueError("Invalid cut")

        # Cut the cake piece
        newPieces = []
        for polygon in polygons:
            line_points = LineString([tuple(current_percept.cur_pos), tuple(action[1])])
            slices = divide_polygon(polygon, line_points)
            for slice in slices:
                newPieces.append(slice)

        current_percept.cur_pos = action[1]
        return newPieces, current_percept


def invalid_knife_position(pos, current_percept):
    cur_x, cur_y = pos
    if (cur_x != 0 and cur_x != current_percept.cake_width) and (
        cur_y != 0 and cur_y != current_percept.cake_len
    ):
        return True

    if cur_x == 0 or cur_x == current_percept.cake_width:
        if cur_y < 0 or cur_y > current_percept.cake_len:
            return True

    if cur_y == 0 or cur_y == current_percept.cake_len:
        if cur_x < 0 or cur_x > current_percept.cake_width:
            return True
    return False


def divide_polygon(polygon, line):
    if not line.intersects(polygon):
        return [polygon]
    result = split(polygon, line)

    polygons = []
    for i in range(len(result.geoms)):
        polygons.append(result.geoms[i])

    return polygons
