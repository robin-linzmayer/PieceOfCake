import os
import pickle
from typing import List

import numpy as np
import logging

import constants
from piece_of_cake_state import PieceOfCakeState


class G8_Player:
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
        self.remaining_requests = None
        self.cut_path = []
        self.assignment = []

    def move(self, current_percept: PieceOfCakeState) -> (int, List[int]):
        """Function which retrieves the current state of the amoeba map and returns an amoeba movement

        Args:
            current_percept(TimingMazeState): contains current state information
        Returns:
            int: This function returns the next move of the user:
                WAIT = -1
                LEFT = 0
                UP = 1
                RIGHT = 2
                DOWN = 3
        """
        print(current_percept)
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        
        # On turn 1 we need to decide where to start.
        if turn_number == 1:
            # initialize these variables
            self.assignment = [-1 for _ in range(len(requests))]
            self.remaining_requests = sorted(requests)
            
            start_position = self.decide_where_to_start(current_percept)
            self.cut_path.append(start_position)
            return constants.INIT, start_position

        if len(self.remaining_requests) == 0:
            return self.assign_polygons(polygons, requests)
            # assignment = self.assignment
            # return constants.ASSIGN, self.assignment
    
                
        # Cut the max request remaining
        max_request = self.remaining_requests.pop()
        
        # We could also assign here early instead of doing it later
        print("cutting for request: ", max_request)
        request_index = requests.index(max_request)
        self.assignment[request_index] = len(polygons)
        
        # request_index, assignment in enumerate(action[1]):
        # the index of assignment corresponds to index of request in requests
        # Therefore if we want to say Polygon 1 is assigned to request 7 we would have assignment[7] = 1
        
        #If we start at 0,0 we will want to cut down and to the right.
        # Area of triangle = 1/2 * base * height
        # base = (end_x - base_left), height = cake_len
        # request = 1/2 * (end_x - base_left) * cake_len 
        # end_x = (2 * request / cake_len) + base_left
        
        if len(self.cut_path) == 1:
            base_left = 0
        else: 
            base_left = self.cut_path[-2][0] # Since we go up and down
        
        end_x = (2 * max_request / cake_len) + base_left
        end_x = round(end_x, 2)
        if end_x > cake_width:
            end_x = cake_width
        end_y = cake_len if cur_pos[1] == 0 else 0
        self.cut_path.append([end_x, end_y])
        return constants.CUT, [end_x, end_y]
    
    
        if len(polygons) != len(requests):
            if cur_pos[0] == 0:
                return constants.CUT, [
                    cake_width,
                    round((cur_pos[1] + 5) % cake_len, 2),
                ]
            else:
                return constants.CUT, [0, round((cur_pos[1] + 5) % cake_len, 2)]

        assignment = []
        for i in range(len(requests)):
            assignment.append(i)

        return constants.ASSIGN, assignment

    def decide_where_to_start(self, current_percept):
        return [0, 0]
    
    def assign_polygons(self, polygons, requests: List[float]):
        assignments = [-1] * len(requests)
        request_indices = sorted(range(len(requests)), key=lambda i: requests[i], reverse=True)  # Largest request first
        polygon_indices = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)  # Largest polygon first

        for _, req in enumerate(request_indices):
            best_polygon = polygon_indices.pop(0)  # Assign largest available polygon to the current largest request
            assignments[req] = best_polygon
            self.logger.info(f"Assigning polygon {best_polygon} to request {req}, area: {polygons[best_polygon].area:.2f}, request: {requests[req]:.2f}")
        
        return constants.ASSIGN, assignments

