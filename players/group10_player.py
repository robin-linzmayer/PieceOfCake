from collections import defaultdict
import os
import pickle
from typing import List

import numpy as np
import math
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
        self.cake_width = None
        self.cake_diagonal = None
        self.requests = None
        self.cuts = []
        self.base_case_switch = False
        self.working_height = None

    def move(self, current_percept) -> tuple[int, List[int]]:
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
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cut_number = current_percept.turn_number - 1
        cur_pos = current_percept.cur_pos
    
        if turn_number == 1:
            # initialize instance variables, sorted requests
            self.requests = current_percept.requests
            self.cake_len = current_percept.cake_len
            self.cake_width = current_percept.cake_width
            self.cake_diagonal = self.calcDiagonal()
            print ("cake_len:", self.cake_len)
            print ("cake_width:", self.cake_width)
            print ("cake_diagonal:", self.cake_diagonal)

            self.cuts.append((0, 0))
            return constants.INIT, [0,0]

        # TODO: adjust for the case when the base the triangle needs surpasses the cake width we have (first occurence, switch to the length, after this switch, make sure we continue working on that edge). CURRENT PROGRAM UNABLE TO MOVE IF WE SWITCH SIDES FROM THE BOTTOM EDGE
        # case if the diagonal of the total cake is <= 25
        if self.cake_diagonal <= 25:
            # assign pieces
            if (cut_number > len(self.requests)):
                # print("I AM TRYING TO ASSIGN PIECES")
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            
            current_area = self.requests[cut_number - 1]
            
            x = cur_pos[0]
            y = cur_pos[1]

            if self.base_case_switch:
                # TODO: this currently retraces the past cut and creates triangles from the same poin
                x = self.cuts[cut_number - 2][0]
                y = self.cuts[cut_number - 2][1]
                if y != 0:
                    y = self.cuts[cut_number - 2][1] - round(2 * current_area / self.working_height, 2)
                
                self.cuts.append((x, y))
                return constants.CUT, [x, y]

            if (cut_number == 1):
                x = round(2 * current_area / self.cake_len, 2)
            else:
                x = round(self.cuts[cut_number - 2][0] + (2 * current_area / self.cake_len), 2)

            y = (0, self.cake_len) [cut_number % 2]

            if x > self.cake_width:
                area_left = (self.cake_len * self.cake_width) / 1.05 * .05 # finding the extra cake portion
                self.working_height = self.cake_width - cur_pos[0]
                if (cut_number < len(self.requests)): # not on our last request
                    area_left += sum(self.requests[cut_number:])
                x = self.cake_width
                y = round(2 * area_left / self.working_height, 2)
                self.base_case_switch = True

            self.cuts.append((x, y))
            return constants.CUT, [x, y]
            
        else:
            if len(polygons) != len(self.requests):
                if cur_pos[0] == 0:
                    return constants.CUT, [self.cake_width, round((cur_pos[1] + 5)%self.cake_len, 2)]
                else:
                    return constants.CUT, [0, round((cur_pos[1] + 5)%self.cake_len, 2)]

            assignment = []
            for i in range(len(self.requests)):
                assignment.append(i)

            return constants.ASSIGN, assignment
    
    def calcDiagonal(self):
        return (math.sqrt((self.cake_len * self.cake_len) + (self.cake_width * self.cake_width)))
    
    def assignPolygons(self, polygons) -> list[int]:
        # parse polygons to polygon_areas: dict(rank: (area, i))
        # print(polygons)
        polygon_areas = []
        requests_items = []
        for i in range(len(polygons)):
            polygon_areas.append((polygons[i].area, i)) 
        for i in range(len(self.requests)):
            requests_items.append((self.requests[i], i))
        # print("AREAS: ", polygon_areas)
        # print("REQUESTS: ", requests_items)

        matches = {} # request : polygon

        for i in range(len(requests_items)):
            # print(f"I AM ON {i} ITERATION")
            min_diff = math.inf
            polygon = -1
            polygon_index = -1
            # print("INIT COMPLETE")
            for j in range(len(polygon_areas)):
                # print("area ", polygon_areas[j][0])
                # print("request ", requests_items[i][0])
                temp_diff = abs(float(polygon_areas[j][0]) - float(requests_items[i][0]))
                # print("TEMP DIFF SET")
                if temp_diff < min_diff:
                    min_diff = temp_diff
                    polygon = polygon_areas[j][1]
                    polygon_index = j
            # print ("polygon:", polygon)
            matches[i] = polygon
            polygon_areas.pop(polygon_index)
        
        assignment = []
        for i in range(len(requests_items)):
            if i in matches:
                assignment.append(matches[i])
            else:
                assignment.append(-1)
        print("THIS IS MY ASSIGNMENT", assignment)
        return assignment
                




        
        
