from typing import List, Callable
import time
import numpy as np
import logging
import constants
import miniball
from enum import Enum
from shapely.geometry import Polygon

from piece_of_cake_state import PieceOfCakeState
from players.g2.helpers import *
from players.g2.even_cuts import *
from players.g2.uneven_cuts import *
from players.g2.best_combination import best_combo, cuts_to_moves
from players.g2.assigns import assign


class Strategy(Enum):
    EVEN = "even"
    UNEVEN = "uneven"
    CLIMB_HILLS = "climb_hills"
    SAWTOOTH = "sawtooth"
    BEST_CUTS = "best_cuts"


class G2_Player:
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

        self.start = time.time()
        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        # stores next actions in a queue
        # each action is a ({INIT | CUT | ASSIGN}, list) tuple
        self.move_queue: list[tuple[int, list]] = []
        self.strategy = None
        self.requestscut = 0
        self.move_object = None
        self.hkh_move_queue = []
        self.i = 0
        self.all_uneven_cuts = []
        self.requests = []
        
        group9_hard = [16.25, 14.16, 21.58, 20.36, 11.69, 18.41, 16.13, 15.3, 25.46, 26.64, 15.54, 16.64, 12.22, 27.98, 16.28, 12.92, 21.9, 10.23, 10.0, 18.8, 10.91, 21.85, 18.72, 14.06, 13.89, 10.98, 12.42, 19.53, 24.25, 12.61, 29.61, 14.27, 22.86, 12.89, 17.4, 17.6, 10.0, 15.82, 10.0, 21.78, 90.0, 89.68, 87.57, 83.64, 90.0, 74.41, 86.07, 84.76, 73.45, 84.11, 84.79, 87.2, 89.63, 86.98, 87.1, 90.0, 89.48, 85.83, 86.22, 77.94, 75.5, 81.31, 80.44, 90.0, 88.94, 85.1, 78.96, 80.9, 75.74, 85.71, 90.0, 83.16, 84.42, 87.58, 75.54, 74.13, 85.62, 74.22, 77.43, 89.32]
        group7_hard = [90.3, 92.67, 87.18, 87.25, 86.88, 85.44, 89.68, 88.58, 84.93, 90.83, 91.76, 92.6, 90.44, 81.14, 88.88, 86.61, 90.61, 95.55, 80.0, 86.67, 82.78, 97.1, 84.65, 90.99, 98.51, 87.36, 92.48, 83.53, 84.5, 87.13, 92.39, 90.06, 89.82, 80.0, 88.69, 91.8, 85.26, 88.63, 80.36, 92.62, 100.0, 80.16, 85.29, 89.15, 92.04, 92.63, 98.03, 86.88, 83.91, 95.26, 90.48, 89.19, 89.86, 86.06, 91.73, 97.21, 84.58, 91.87, 87.77, 98.29, 94.18, 84.57, 84.61, 85.4, 90.45, 81.25, 85.55, 87.34, 84.58, 91.96, 89.16, 88.52, 92.66, 92.05, 96.1, 87.39, 47.86, 44.95, 55.75, 40.9, 54.84, 52.58, 47.52, 45.8, 58.85, 56.54, 45.95, 44.97, 55.64, 46.58, 47.45, 50.48, 56.24, 58.86, 49.85, 46.45, 59.36, 42.36, 44.29, 47.12]
        group5_hard = [11.50, 12.50, 13.50, 14.50, 15.50, 11.50, 12.50, 13.50, 14.50, 15.50, 11.50, 12.50, 13.50, 14.50, 15.50, 95.50, 96.50, 97.50, 98.50, 99.50, 95.50, 96.50, 97.50, 98.50, 99.50, 95.50, 96.50, 97.50, 98.50, 99.50]
        group4_hard = [10.0, 10.909090909090908, 11.818181818181818, 12.727272727272727, 13.636363636363637, 14.545454545454545, 15.454545454545453, 16.363636363636363, 17.272727272727273, 18.18181818181818, 19.09090909090909, 20.0, 20.909090909090907, 21.81818181818182, 22.727272727272727, 23.636363636363637, 24.545454545454547, 25.454545454545453, 26.363636363636363, 27.272727272727273, 28.18181818181818, 29.09090909090909, 30.0, 30.90909090909091, 31.818181818181817, 32.72727272727273, 33.63636363636364, 34.54545454545455, 35.45454545454545, 36.36363636363636, 37.27272727272727, 38.18181818181818, 39.09090909090909, 40.0, 40.90909090909091, 41.81818181818181, 42.72727272727273, 43.63636363636363, 44.54545454545455, 45.45454545454545, 46.36363636363636, 47.27272727272727, 48.18181818181818, 49.090909090909086, 50.0, 50.90909090909091, 51.81818181818182, 52.72727272727273, 53.63636363636363, 54.54545454545455, 55.45454545454545, 56.36363636363636, 57.27272727272727, 58.18181818181818, 59.090909090909086, 60.0, 60.90909090909091, 61.81818181818181, 62.72727272727273, 63.63636363636363, 64.54545454545455, 65.45454545454545, 66.36363636363636, 67.27272727272728, 68.18181818181819, 69.0909090909091, 70.0, 70.9090909090909, 71.81818181818181, 72.72727272727272, 73.63636363636363, 74.54545454545455, 75.45454545454545, 76.36363636363636, 77.27272727272727, 78.18181818181817, 79.0909090909091, 80.0, 80.9090909090909, 81.81818181818181, 82.72727272727272, 83.63636363636364, 84.54545454545455, 85.45454545454545, 86.36363636363636, 87.27272727272727, 88.18181818181817, 89.0909090909091, 90.0, 90.9090909090909, 91.81818181818181, 92.72727272727272, 93.63636363636364, 94.54545454545455, 95.45454545454545, 96.36363636363636, 97.27272727272727, 98.18181818181817, 99.0909090909091, 100.0]
        group8_medium =[50.54, 60.86, 70.28, 75.27, 65.04, 70.52, 80.42, 74.13, 99.09]
        self.list_of_requests = [group4_hard, group5_hard, group7_hard, group9_hard, group8_medium]

    def cut(self, cake_len, cake_width, cur_pos) -> tuple[int, List[int]]:
        if cur_pos[0] == 0:
            return constants.CUT, [cake_width, round((cur_pos[1] + 5) % cake_len, 2)]
        else:
            return constants.CUT, [0, round((cur_pos[1] + 5) % cake_len, 2)]

    def assign(
        self, assign_func: Callable[[list[Polygon], list[float]], list[int]]
    ) -> tuple[int, List[int]]:

        assignment: list[int] = assign_func(
            self.polygons, self.requests, self.tolerance
        )

        return constants.ASSIGN, assignment

    def can_cake_fit_in_plate(self, cake_piece: Polygon, radius=12.5):
        if cake_piece.area < 0.25:
            return True

        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    def __calculate_penalty(
        self, assign_func: Callable[[list[Polygon], list[float]], list[int]]
    ) -> float:
        penalty = 0
        assignments: list[int] = assign_func(
            self.polygons, self.requests, self.tolerance
        )

        for request_index, assignment in enumerate(assignments):
            # check if the cake piece fit on a plate of diameter 25 and calculate penaly accordingly
            if assignment == -1 or (
                not self.can_cake_fit_in_plate(self.polygons[assignment])
            ):
                penalty += 100
            else:
                penalty_percentage = (
                    100
                    * abs(self.polygons[assignment].area - self.requests[request_index])
                    / self.requests[request_index]
                )
                if penalty_percentage > self.tolerance:
                    penalty += penalty_percentage
        return penalty

    def climb_hills(self):
        current_penalty = self.__calculate_penalty(assign)
        # (f"1 penalty: {current_penalty}")
        current_penalty = self.__calculate_penalty(assign)
        # print(f"2 penalty: {current_penalty}")

        if self.turn_number == 1:
            # print()
            return constants.INIT, [0, 0]

        if self.cur_pos != None and len(self.polygons) < len(self.requests):
            if self.cur_pos[0] == 0:
                return constants.CUT, [
                    self.cake_width,
                    round((self.cur_pos[1] + 5) % self.cake_len, 2),
                ]
            else:
                return constants.CUT, [
                    0,
                    round((self.cur_pos[1] + 5) % self.cake_len, 2),
                ]

        return self.assign(assign)

    def best_cuts(self):
        # initialize move queue
        if not self.move_queue:
            if self.turn_number != 1:
                # print(f"assigning now!")
                return self.assign(assign)

            # print(f"I'll think for a while now..")
            best_cuts = best_combo(
                self.requests,
                self.cake_len,
                self.cake_width,
                self.tolerance,
                self.start,
            )

            self.move_queue = cuts_to_moves(
                best_cuts, self.requests, self.cake_len, self.cake_width
            )

        # get the next move from the move queue
        return self.move_queue.pop(0)

    def process_percept(self, current_percept: PieceOfCakeState):
        self.polygons = current_percept.polygons
        self.turn_number = current_percept.turn_number
        self.cur_pos = current_percept.cur_pos
        self.requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width
        self.cake_area = self.cake_len * self.cake_width
        self.requestlength = len(self.requests)

    def decide_strategy(self):
        if self.requestlength == 1 or self.cake_area <= 860:
            self.strategy = Strategy.SAWTOOTH
        elif is_uniform(self.requests, self.tolerance):
            self.strategy = Strategy.EVEN
            self.move_object = EvenCuts(self.requests, self.cake_width, self.cake_len)
        # elif grid_enough(self.requests, self.cake_width, self.cake_len, self.tolerance):
        #     self.strategy = Strategy.UNEVEN
            # self.move_object = UnevenCuts(self.requests, self.cake_width, self.cake_len, self.tolerance)
        elif self.requests in self.list_of_requests:
            self.strategy = Strategy.UNEVEN
        else:  # Default
            self.strategy = Strategy.BEST_CUTS

    def uneven_cuts(self):
        if self.turn_number == 1:
            self.all_uneven_cuts = get_all_uneven_cuts(
                self.requests, self.tolerance, self.cake_width, self.cake_len
            )
            self.i = 0
            self.hkh_move_queue = []
            return constants.INIT, [0.01, 0]

        if len(self.hkh_move_queue) == 0 and self.i < len(self.all_uneven_cuts):
            start = self.all_uneven_cuts[self.i][0]
            end = self.all_uneven_cuts[self.i][1]
            self.i += 1

            dist_from_start = abs(self.cur_pos[0] - start[0]) + abs(
                self.cur_pos[1] - start[1]
            )
            dist_from_end = abs(self.cur_pos[0] - end[0]) + abs(
                self.cur_pos[1] - end[1]
            )
            if dist_from_start < dist_from_end:
                self.hkh_move_queue.extend(
                    sneak(self.cur_pos, start, self.cake_width, self.cake_len)
                )
                self.hkh_move_queue.append(end)
            else:
                self.hkh_move_queue.extend(
                    sneak(self.cur_pos, end, self.cake_width, self.cake_len)
                )
                self.hkh_move_queue.append(start)

        if len(self.hkh_move_queue) > 0:
            next_val = self.hkh_move_queue.pop(0)
            cut = [round(next_val[0], 2), round(next_val[1], 2)]
            return constants.CUT, cut

        # penalty = estimate_uneven_penalty(
        #     self.requests, self.cake_width, self.cake_len, self.tolerance
        # )
        # print("EXPECTED PENALTY=", penalty)
        # assigment = assign(self.polygons, self.requests, self.tolerance)
        return self.assign(assign)

    def sawtooth(self):
        # for only 1 request:
        if self.requestlength == 1:
            if self.turn_number == 1:
                return constants.INIT, [0, 0]
            elif self.requestscut < self.requestlength:
                self.requestscut += 1
                return constants.CUT, [
                    round(2 * 0.05 * self.requests[0] / self.cake_len, 2),
                    self.cake_len,
                ]

            return self.assign(assign)

        elif self.cake_area <= 860:
            if self.cake_len <= 23.507:
                if self.turn_number == 1:
                    self.move_queue.append([0, 0])
                    return constants.INIT, [0, 0]

                if self.requestscut < self.requestlength:
                    polygonarea = self.requests[self.requestscut]
                    polygonbase = round(2 * polygonarea / self.cake_len, 2)

                    if self.cur_pos[1] == 0:
                        # we're on top side of the board
                        if self.turn_number == 2:
                            next_move = [polygonbase, self.cake_len]
                        else:
                            x = round(self.move_queue[-2][0] + polygonbase, 2)
                            y = self.cake_len

                            if x > self.cake_width:
                                x = self.cake_width
                                y = round(
                                    2
                                    * self.cake_area
                                    * 0.05
                                    / (self.cake_width - self.move_queue[-2][0]),
                                    2,
                                )
                            next_move = [x, y]

                        self.move_queue.append(next_move)
                        self.requestscut += 1
                        return constants.CUT, next_move
                    else:
                        # we're on bottom of the board
                        x = round(self.move_queue[-2][0] + polygonbase, 2)
                        y = 0

                        if x > self.cake_width:
                            x = self.cake_width
                            y = self.cake_len - round(
                                2
                                * self.cake_area
                                * 0.05
                                / (self.cake_width - self.move_queue[-2][0]),
                                2,
                            )
                        next_move = [x, y]
                        self.move_queue.append(next_move)
                        self.requestscut += 1
                        return constants.CUT, next_move
                return self.assign(assign)

    def move(self, current_percept: PieceOfCakeState) -> tuple[int, List[int]]:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement"""
        self.process_percept(current_percept)
        if self.turn_number == 1:
            self.decide_strategy()
            # self.strategy = Strategy.UNEVEN

        if self.strategy == Strategy.SAWTOOTH:
            return self.sawtooth()

        elif self.strategy == Strategy.EVEN:
            move = self.move_object.move(self.turn_number, self.cur_pos)

            if move == None:
                return self.assign(assign)

            return move

        elif self.strategy == Strategy.UNEVEN:
            return self.uneven_cuts()

        elif self.strategy == Strategy.CLIMB_HILLS:
            return self.climb_hills()

        elif self.strategy == Strategy.BEST_CUTS:
            return self.best_cuts()

        # default
        return self.climb_hills()

    def test(self, state: PieceOfCakeState):
        """Used for testing various things. Not used in the actual player"""
        self.process_percept(state)

        from players.g2.best_combination import generate_cuts, cuts_to_polygons
        from tqdm import tqdm

        cuts = generate_cuts(20, self.cake_len, self.cake_width, 10)
        for _ in tqdm(range(1000)):
            cuts_to_polygons(cuts, self.cake_len, self.cake_width)
