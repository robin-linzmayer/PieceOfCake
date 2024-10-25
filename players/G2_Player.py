from typing import List
import numpy as np
import logging
import constants
import miniball


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

        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.move_queue = []

    def can_cake_fit_in_plate(self, cake_piece, radius=12.5):
        cake_points = np.array(
            list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double
        )
        res = miniball.miniball(cake_points)

        return res["radius"] <= radius

    # use just any assignment for now,
    # ideally, we want to find the assignment with the smallest penalty
    # instead of this random one
    def __get_assignments(self) -> float:
        # TODO: Find a way to match polygons with requests
        # with a low penalty

        # sorted_requests = sorted(
        #     [(i, req) for i, req in enumerate(self.requests)], key=lambda x: x[1]
        # )

        if len(self.requests) > len(self.polygons):
            # specify amount of -1 padding needed
            padding = len(self.requests) - len(self.polygons)
            return padding * [-1] + list(range(len(self.polygons)))

        # return an amount of polygon indexes
        # without exceeding the amount of requests
        return list(range(len(self.polygons)))[: len(self.requests)]

    def __calculate_penalty(self) -> float:
        penalty = 0
        assignments = self.__get_assignments()

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

    def move(self, current_percept) -> tuple[int, List[int]]:
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement"""
        self.polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        self.requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        current_penalty = self.__calculate_penalty()
        print(f"current penalty: {current_penalty}")

        if turn_number == 1:
            return constants.INIT, [0, 0]

        if len(self.polygons) < len(self.requests):
            if cur_pos[0] == 0:
                return constants.CUT, [
                    cake_width,
                    round((cur_pos[1] + 5) % cake_len, 2),
                ]
            else:
                return constants.CUT, [0, round((cur_pos[1] + 5) % cake_len, 2)]

        assignment = []
        for i in range(len(self.requests)):
            assignment.append(i)

        print(assignment)

        return constants.ASSIGN, assignment

    def sneak(start_pos, goal_pos):
        return
