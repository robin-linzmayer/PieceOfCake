from collections import defaultdict
import os
import pickle
from typing import List
import pprint

import numpy as np
import math
import logging

import miniball
from shapely import points, centroid
from shapely.geometry import Polygon, LineString, Point

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
        self.original_requests = None
        self.cuts = []
        self.base_case_switch = False, None
        self.x_y_switch = True # True: y working height, False: x working height
        self.working_height = None
        self.turn_number = None
        self.cut_number = None
        self.cur_pos = None
        self.uniform_mode = False
        self.uniform_cuts = []
        self.used_crumb = [[0.00, 0.00], [0.00, 0.00], [0.00, 0.00], [0.00, 0.00]] #LEFT-TOP, RIGHT-TOP, LEFT-BOTTOM, RIGHT-BOTTOM. The values inside is [x, y]
        ################# Tom (11/6):
        self.acceptable_range = []
        ############################
        self.angle_cuts = []
        self.margin = 0.01
        self.zigzag_cuts = []
        self.zigzag_found = False
        self.set_assignments = {}

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
        extra_tol = 0 #Change if we suck
        polygons = current_percept.polygons
        self.turn_number = current_percept.turn_number
        self.cut_number = current_percept.turn_number - 1
        self.cur_pos = current_percept.cur_pos
    
        if self.turn_number == 1:
            # initialize instance variables, sorted requests
            self.original_requests = current_percept.requests
            self.requests = current_percept.requests
            self.cake_len = current_percept.cake_len
            self.cake_width = current_percept.cake_width
            self.requests.sort(reverse=False)
            if not self.uniform_mode:
                is_uniform, grid_area = self.if_uniform(current_percept.requests, extra_tol)
                if is_uniform:
                    self.grid_cut(current_percept, grid_area)
                    self.uniform_mode = True

        if self.zigzag_found:
            if (self.cut_number > len(self.zigzag_cuts) - 1):
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            return constants.CUT, list(self.zigzag_cuts[self.cut_number])

        # TODO: adjust for the case when the base the triangle needs surpasses the cake width we have (first occurence, switch to the length, after this switch, make sure we continue working on that edge). CURRENT PROGRAM UNABLE TO MOVE IF WE SWITCH SIDES FROM THE BOTTOM EDGE
        # case if the diagonal of the total cake is <= 25
        if self.cake_len <= 24.67:
            if self.turn_number == 1:
                self.requests.sort(reverse=True)
                self.cuts.append((0, 0))
                return constants.INIT, [0,0]

            # assign pieces
            if (self.cut_number > len(self.requests)):
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            
            return self.smallCake()
    
        elif self.uniform_mode:
            if (self.cut_number > len(self.uniform_cuts) - 1):
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            if self.turn_number == 1:
                return constants.INIT, self.uniform_cuts[self.cut_number]

            return constants.CUT, self.uniform_cuts[self.cut_number]
        else:
            # Optimal ZigZag Strategy
            list_of_factors = self.find_factors(len(self.requests), False) #From closest to furthest. Element is list.

            if not list_of_factors:
                #IS PRIME
                list_of_factors = self.find_factors(len(self.requests) - 1, False)

            list_of_factors.pop(-1) #Remove last element, since it is always (1, something)

            factor_penalty = dict()
            factor_cuts = dict()
            factor_assignments = dict()
            
            for factor in list_of_factors[::-1]:
                penalty, cuts, all_assignments = self.simulate_cuts(factor)
                factor_penalty[tuple(factor)] = penalty
                factor_cuts[tuple(factor)] = cuts
                factor_assignments[tuple(factor)] = all_assignments


            best_factor = min(factor_penalty, key=factor_penalty.get)
            best_cut = factor_cuts[best_factor]
            self.set_assignments = factor_assignments[best_factor]
            zigzag = []

            working_length = self.cake_len - (2 * self.margin)
            row_height = round(working_length/best_factor[0], 2)

            # Horizontal Cuts
            for i in range(best_factor[0], 0, -1):
                # If even, go from left to right
                if i % 2 == 0:
                    if (i != best_factor[0]):
                        temp = (0, round(0.01 + (i * row_height), 2))
                        zigzag.append(tuple(self.add_crumbs(temp, True)))
                    zigzag.append((0, round(0.01 + (i * row_height), 2))) #Left-Coord
                    zigzag.append((self.cake_width, round(0.01 + (i * row_height), 2))) #Right-Coord
                # If odd, right to left
                else:
                    if (i != best_factor[0]):
                        temp = (self.cake_width, round(0.01 + (i * row_height), 2))
                        zigzag.append(tuple(self.add_crumbs(temp, True)))
                    zigzag.append((self.cake_width, round(0.01 + (i * row_height), 2))) #Right-Coord
                    zigzag.append((0, round(0.01 + (i * row_height), 2))) #Left-Coord
                    
            # If small factor is even, then we know we end on the right. Hence, we will perform a breadcrumb in the right corner
            if best_factor[0] % 2 == 0:
                zigzag.append((0.01, 0))
            else:
                zigzag.append((round(self.cake_width - 0.01, 2), 0))
            
            # Vertical Cuts
            for i in range(len(best_cut)):
                # If even, cut from top to bot
                if i % 2 == 0:
                    temp = (float(best_cut[i][0][0])), float(best_cut[i][0][1])
                    zigzag.append(tuple(self.add_crumbs(temp, False)))
                    zigzag.append((round(float(best_cut[i][0][0]), 2), round(float(best_cut[i][0][1]), 2))) #Top-Coord
                    zigzag.append((round(float(best_cut[i][1][0]), 2), round(float(best_cut[i][1][1]), 2))) #Bot-Coord

                # If odd, bot to top.
                else:
                    temp = (float(best_cut[i][1][0])), float(best_cut[i][1][1])
                    zigzag.append(tuple(self.add_crumbs(temp, False)))
                    zigzag.append((round(float(best_cut[i][1][0]), 2), round(float(best_cut[i][1][1]), 2))) #Bot-Coord
                    zigzag.append((round(float(best_cut[i][0][0]), 2), round(float(best_cut[i][0][1]), 2))) #Top-Coord

            self.zigzag_cuts = zigzag

            self.zigzag_found = True
            return constants.INIT, list(self.zigzag_cuts[0])
    
    # Add crumbs to existing cuts
    def add_crumbs(self, target, is_horizontal) -> tuple[float, float]:
        crumb_coord = [-1, -1]
        if not is_horizontal:
            if target[0] <= self.cake_width/2:
                crumb_coord[0] = 0
            else:
                crumb_coord[0] = self.cake_width

            if target[1] == 0: #We are at Top
                crumb_coord[1] = 0.01
            else:
                crumb_coord[1] = round(self.cake_len - 0.01, 2)
        else:
            if target[1] <= self.cake_len/2:
                crumb_coord[1] = 0
            else:
                crumb_coord[1] = self.cake_len
            if target[0] == 0: #We are at left
                crumb_coord[0] = 0.01
            else:
                crumb_coord[0] = round(self.cake_width - 0.01, 2)
        return crumb_coord
                

    # Returns penalty given 
    def simulate_cuts(self, factor) -> tuple[float, list]:
        total_penalty = 0
        self.working_height = round(float((self.cake_len - (2 * self.margin))/factor[0]), 2)
        horizontals = []
        all_assignments = {}
        for j in range (factor[0] + 1):
            horizontals.append(round(self.margin + (self.working_height * j), 2))
        for i in range(factor[1]):
            tolerances = self.find_acceptable_range(factor[0], i) # Returns tolerances of the smaller factors
            penalty, assignments = self.angle_sweep(tolerances, factor[0], i, horizontals)
            total_penalty += penalty

        all_assignments.update(assignments)
        cut_arr = self.angle_cuts.copy()
        self.angle_cuts = []

        return total_penalty, cut_arr, all_assignments
    
    def calcDiagonal(self) -> float:
        return (math.sqrt((self.cake_len * self.cake_len) + (self.cake_width * self.cake_width)))
    
    def smallCake(self) -> tuple[int, list[float, float]]:
        current_area = self.requests[self.cut_number - 1]
        
        x = self.cur_pos[0]
        y = self.cur_pos[1]

        
        if self.base_case_switch[0]:
            x = self.cuts[self.cut_number - 2][0]
            y = self.cuts[self.cut_number - 2][1] # on the first occurrence, this will be 0 if we went from bottom to right and cake_len if we went from top to right
            if self.x_y_switch and self.base_case_switch[1]:
                self.working_height = self.cuts[self.cut_number - 1][1]
                self.x_y_switch = not self.x_y_switch
                x = round(x + 2 * current_area / self.working_height, 2)
                y = 0
            elif self.x_y_switch and not self.base_case_switch[1]:
                self.working_height = self.cake_width - self.cuts[self.cut_number - 1][0]
                self.x_y_switch = not self.x_y_switch
                x = round(x + 2 * current_area / self.working_height, 2)
                y = self.cake_len
            elif not self.x_y_switch and self.base_case_switch[1]:
                self.working_height = self.cake_width - self.cuts[self.cut_number - 1][0]
                self.x_y_switch = not self.x_y_switch
                x = self.cake_width
                y = round(y - 2 * current_area / self.working_height, 2)
            elif not self.x_y_switch and not self.base_case_switch[1]:
                self.working_height = self.cake_width - self.cuts[self.cut_number - 1][0]
                self.x_y_switch = not self.x_y_switch
                x = self.cake_width
                y = round(y + 2 * current_area / self.working_height, 2)
            self.cuts.append((x, y))
            return constants.CUT, [x, y]

        if (self.cut_number == 1):
            x = round(2 * current_area / self.cake_len, 2)
        else:
            x = round(self.cuts[self.cut_number - 2][0] + (2 * current_area / self.cake_len), 2)

        y = (0, self.cake_len) [self.cut_number % 2]

        if x > self.cake_width:
            area_left = (self.cake_len * self.cake_width) / 1.05 * .05 # finding the extra cake portion
            self.working_height = self.cake_width - self.cur_pos[0]
            if (self.cut_number < len(self.requests)): # not on our last request
                area_left += sum(self.requests[self.cut_number:])
            x = self.cake_width
            if (y == 0): 
                y = round(self.cake_len - 2 * area_left / self.working_height, 2)
                self.base_case_switch = True, False
            elif (y == self.cake_len):
                y = round(2 * area_left / self.working_height, 2)
                self.base_case_switch = True, True

        self.cuts.append((x, y))
        return constants.CUT, [x, y]
    
    def assignPolygons(self, polygons) -> list[int]:
        # parse polygons to polygon_areas: dict(rank: (area, i))
        polygon_areas = []
        requests_items = []
        for i in range(len(polygons)):
            polygon_areas.append((polygons[i].area, i)) 
        for i in range(len(self.requests)):
            requests_items.append((self.original_requests[i], i))
        requests_items.sort()
        polygon_areas.sort()

        matches = {} # request : polygon
        for i in range(len(requests_items)):
            if requests_items[i][0] in self.set_assignments:
                matches[i] = self.set_assignments[requests_items[i][0]]
                continue
            min_penalty= math.inf
            polygon = -1
            polygon_index = -1
            for j in range(len(polygon_areas)):
                temp_penalty = 100 * (abs(float(polygon_areas[j][0]) - float(requests_items[i][0]))/float(requests_items[i][0]))
                if temp_penalty <= self.tolerance: 
                    temp_penalty = 0
                if temp_penalty < min_penalty:
                    min_penalty = temp_penalty
                    polygon = polygon_areas[j][1]
                    polygon_index = j
            matches[i] = polygon
            polygon_areas.pop(polygon_index)
        
        assignment = []
        for i in range(len(requests_items)):
            if i in matches:
                assignment.append(matches[i])
            else:
                assignment.append(-1)
        return assignment
                
    #Returns whether if it is uniform, and the area of the uniform. If not, area is -1
    def if_uniform(self, requests, extra_tol=0.0) -> tuple[bool, float]:

        # If prime return false and use zig zag.
        if self.find_factors(len(requests), True) == False:
            return (False, -1)

        tolerance = (self.tolerance + extra_tol) / 100
        skewed_average = (1 - tolerance**2)*(sum(requests) / len(requests)) if tolerance < 1 else (tolerance**2)**(sum(requests) / len(requests))
        for req in requests:
            if (abs(req-skewed_average)/req) >= tolerance:
                return (False, -1)
            
        return (True, skewed_average)
    
    #Cuts into grid using crumbs method, returns the coordinate to cut to.
    def grid_cut(self, current_percept, grid_area) -> list[float, float]:
        # length_shave = 0.01266028 * self.cake_len
        
        # width_shave = round(1.6 * length_shave, 2)
        # length_shave = round(length_shave, 2)

        width_shave = 0.01
        length_shave = 0.01

        s_factor, l_factor = self.find_factors(closest=True)

        x_cuts = [width_shave] # first cut
        y_cuts = [length_shave] # first cut
        x_increment = round((self.cake_width - 2*width_shave)/l_factor, 2)
        y_increment = round((self.cake_len - 2*length_shave)/s_factor, 2)
        # x_increment * y_increment should be equal to grid_area;
        # but do we need to check it? They might be some lil decimal points off tho;
        # but does it matter?
        for ind in range(s_factor):
            y_cuts.append(round(y_cuts[ind] + y_increment, 2))
        for ind in range(l_factor):
            x_cuts.append(round(x_cuts[ind] + x_increment, 2))



        for i in range(len(y_cuts)):
            if i == 0:
                self.uniform_cuts.append([0, y_cuts[i]])
                self.uniform_cuts.append([self.cake_width, y_cuts[i]])
            elif i % 2 == 1:
                self.uniform_cuts.append([round(self.cake_width - (0.01 + self.used_crumb[1][0]), 2), 0])
                # self.used_crumb[1][0] += 0.01
                self.uniform_cuts.append([self.cake_width, y_cuts[i]])
                self.uniform_cuts.append([0, y_cuts[i]])
            elif i % 2 == 0:
                self.uniform_cuts.append([round(0.01 + self.used_crumb[0][0], 2), 0])
                # self.used_crumb[0][0] += 0.01
                self.uniform_cuts.append([0, y_cuts[i]])
                self.uniform_cuts.append([self.cake_width, y_cuts[i]])

        for i in range(len(x_cuts)):
            if l_factor % 2 == 1:
                if i == 0:
                    self.uniform_cuts.append([x_cuts[i], self.cake_len])
                    self.uniform_cuts.append([x_cuts[i], 0])
                elif i % 2 == 1:
                    self.uniform_cuts.append([0, round((0.01 + self.used_crumb[0][1]), 2)])
                    # self.used_crumb[0][1] += 0.01
                    self.uniform_cuts.append([x_cuts[i], 0])
                    self.uniform_cuts.append([x_cuts[i], self.cake_len])
                elif i % 2 == 0:
                    self.uniform_cuts.append([0, round(self.cake_len - (0.01 + self.used_crumb[2][1]), 2)])
                    # self.used_crumb[2][1] += 0.01
                    self.uniform_cuts.append([x_cuts[i], self.cake_len])
                    self.uniform_cuts.append([x_cuts[i], 0])
            elif l_factor % 2 == 0:
                if i == 0:
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), self.cake_len])
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), 0])
                elif i % 2 == 1:
                    self.uniform_cuts.append([self.cake_width, round((0.01 + self.used_crumb[1][1]), 2)])
                    # self.used_crumb[1][1] += 0.01
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), 0])
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), self.cake_len])
                elif i % 2 == 0:
                    self.uniform_cuts.append([self.cake_width, round(self.cake_len - (0.01 + self.used_crumb[3][1]), 2)])
                    # self.used_crumb[3][1] += 0.01
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), self.cake_len])
                    self.uniform_cuts.append([round(self.cake_width - x_cuts[i], 2), 0])
                
           
    #Finds the closest factor with the number of requests
    def find_factors(self, num_requests, closest: bool) -> list[int, int]: # [smaller_closest_factor, bigger_closest_factor]
        num_of_requests = num_requests
        factors = []
        for num in range(int(num_of_requests**0.5),0,-1):
            if num_of_requests % num == 0:
                factors.append([num, num_of_requests//num])
            if factors and closest:
                return factors[0]
        if len(factors) == 1:
            return False
        return factors
        # TODO: Here we can add a check to see if num_of_requests is prime, by checking if factors[-1][0] == 1

    def find_acceptable_range(self, factor, iteration): # find_acceptable_range(factor[0], i, target_requests)
        # For the three smallest requests, find the upper bound and lower bound of the acceptable area based on the tolerance;
        # divide the upper and lower bounds by the height of that row (fixed for all rows) respectively;
        # get the lower and upper bounds of the widths, name it acceptable_range
        temp = []
        if iteration == 0:
            for req in range(0,factor):
                lower_area = self.requests[req + iteration*factor] * (1-(self.tolerance/100))
                upper_area = self.requests[req + iteration*factor] * (1+(self.tolerance/100))
                lower_wid = lower_area/self.working_height
                upper_wid = upper_area/self.working_height
                temp.append([lower_wid,upper_wid])
        else:
            for req in range(0,factor):
                lower_area = self.requests[req + iteration*factor] * (1-(self.tolerance/100))
                upper_area = self.requests[req + iteration*factor] * (1+(self.tolerance/100))
                
                # Because the diff between the longer and shorter edges is fixed and certain
                diff_a_b = abs(self.angle_cuts[iteration - 1][0][0]-self.angle_cuts[iteration - 1][1][0])/factor

                # [(x + x + diff_a_b)/2] * h = area
                lower_wid = ((lower_area/self.working_height)*2 - diff_a_b) / 2
                upper_wid = ((upper_area/self.working_height)*2 - diff_a_b) / 2

                # Add previous x values
                if self.angle_cuts[iteration - 1][0][0]-self.angle_cuts[iteration - 1][1][0] < 0: 
                    lower_wid = lower_wid + diff_a_b*(req+1) + self.angle_cuts[iteration - 1][0][0]
                    upper_wid = upper_wid + diff_a_b*(req+1) + self.angle_cuts[iteration - 1][0][0]
                else:
                    lower_wid = lower_wid - diff_a_b*(req) + self.angle_cuts[iteration - 1][0][0]
                    upper_wid = upper_wid - diff_a_b*(req) + self.angle_cuts[iteration - 1][0][0]
                temp.append([lower_wid,upper_wid])
        return temp

    # Does a search of all possible lines and adds the one with least penalty to the self.angle_cuts variable, returns the penalty
    """
    self.angle_sweep(tolerances: list[list[lower_tolerance, upper_tolerance]], target_requests: list[requests])
    - using knowledge of the previous cut (from self.angle_cuts), find the next cut that minimizes the penalty for the list of target_requests. 
    - chooses the line with the least penalty, and if there are several, choose the one with the highest slope
    - adds the coordinates from the top and bottom of cake to self.angle_cuts and returns the penalty
    """
    def angle_sweep(self, tolerances, factor, iteration, horizontals) -> float:
        x_range = (
            (tolerances[0][0], tolerances[-1][0]), 
            (tolerances[0][1], tolerances[-1][1])
        )
        target_reqs = self.requests[iteration * factor : (iteration + 1) * factor]
        min_penalty = (math.inf, ((0, 0), (0, 0)), 0, {}) # penalty, cuts, slope, set_assignments
        for coord1x in np.arange(x_range[0][0], x_range[1][0], 0.01):
            if coord1x <= 0.01:
                continue
            for coord2x in np.arange(x_range[0][1], x_range[1][1], 0.01):
                if coord2x <= 0.01:
                    continue
                set_assignments = {}
                cuts = None
                penalty = 0
                coord1 = (coord1x, self.margin + (self.working_height / 2))
                coord2 = (coord2x, self.cake_len - (self.margin + (self.working_height / 2)))
                slope, intercept = self.equation_of_line(coord1, coord2)
                if not slope:
                    cuts = self.rounded((
                        (coord1[0], 0), (coord1[0], self.cake_len)
                    ))
                else:
                    cuts = self.rounded((
                        self.point_on_line(slope, intercept, 0, False), # coordinate for top of cake
                        self.point_on_line(slope, intercept, -1 * self.cake_len, False) # coordinate for bottom of cake
                    ))
                
                # CHECK IF THIS LINE RUNS INTO THE LAST ONE
                if self.angle_cuts:
                    last_cuts = self.angle_cuts[-1] # tuple[tuple, tuple]
                    # checking top coordinates
                    if (last_cuts[0][0] > cuts[0][0]):
                        continue
                    elif (last_cuts[1][0] > cuts[1][0]):
                        continue

                # check penalty for all other requests 
                for i in range(factor):
                    req = target_reqs[i]
                    if (coord1x == coord2x):
                        if coord1x < tolerances[i][0]:
                            penalty += 100 * ((tolerances[i][0] - coord1x) * self.working_height) / req
                        elif coord1x > tolerances[i][1]:
                            penalty += 100 * ((coord1x - tolerances[i][1]) * self.working_height) / req
                    else:
                        curr_penalty = self.calculate_penalty(req, slope, intercept, (horizontals[i], horizontals[i + 1]))
                        if curr_penalty > 100:
                            set_assignments[req] = -1
                            penalty += 100
                        else:
                            penalty += curr_penalty
                if penalty == min_penalty[0] and not slope and cuts[0][0] + cuts[1][0] <= min_penalty[1][0][0] + min_penalty[1][1][0]:
                    min_penalty = (penalty, cuts, math.inf, set_assignments)
                elif penalty < min_penalty[0] or (penalty == min_penalty[0] and abs(slope) > min_penalty[2]):
                    min_penalty = (penalty, cuts, slope, set_assignments)
        self.angle_cuts.append(min_penalty[1])
        return min_penalty[0], min_penalty[3]
        
    def equation_of_line(self, coord1: tuple, coord2: tuple) -> tuple[bool|float, float]:
        # putting cake coordinates on a cartesian plane, all y coordinates are negated
        if (coord2[0] - coord1[0] == 0):
            return (False, 0.0)
        slope = (coord1[1] - coord2[1])/(coord2[0] - coord1[0])
        intercept = slope * (-1 * coord1[0]) - coord1[1]
        return (slope, intercept)
        
    def point_on_line(self, slope: float, intercept: float, given_coord: float, x: bool) -> tuple[float, float]:
        if x:
            return (given_coord, round((slope * given_coord) + intercept), 2)
        elif not x:
            return (round((given_coord - intercept) / slope, 2), -1 * given_coord)
        
    def calculate_penalty(self, req: float, slope: float, intercept: float, y_bounds: tuple[float, float]) -> float:
        corner1 = self.point_on_line(slope, intercept, -1 * y_bounds[0], False)
        corner2 = self.point_on_line(slope, intercept, -1 * y_bounds[1], False)
        corner3 = (0, y_bounds[0])
        corner4 = (0, y_bounds[1])

        if self.angle_cuts:
            cut = self.angle_cuts[-1]
            slope, intercept = self.equation_of_line(cut[0], cut[1])
            if not slope:
                corner3 = (cut[0][0], y_bounds[0])
                corner4 = (cut[0][0], y_bounds[1])
            else:
                corner3 = self.point_on_line(slope, intercept, -1 * y_bounds[0], False)
                corner4 = self.point_on_line(slope, intercept, -1 * y_bounds[1], False)
        x = Polygon([corner3, corner1, corner2, corner4, corner3])
        diff = 100 * abs((req - x.area) / req)
        if diff < self.tolerance:
            return 0
        else:
            return diff - self.tolerance

        # from ../piece_of_cake_game.py
    
    def fits_on_plate(poly: Polygon):
        if poly.area < 0.25:
            return True

        # Step 1: Get the points on the cake piece and store as numpy array
        cake_points = np.array(list(zip(*poly.exterior.coords.xy)), dtype=np.double)

        # Step 2: Find the minimum bounding circle of the cake piece
        res = miniball.miniball(cake_points)

        return res["radius"] <= 12.5
    
    def rounded(self, cuts: tuple[tuple, tuple]) -> tuple[tuple, tuple]:
        return (
            (round(cuts[0][0], 2), round(cuts[0][1], 2)),
            (round(cuts[1][0], 2), round(cuts[1][1], 2))
                )
    
    def major_outliers(self):
        q1 = np.percentile(self.requests, 25)
        q3 = np.percentile(self.requests, 75)
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        lower = q1 - 1.5 * iqr
        new_requests = []
        outlier_requests = []
        for req in self.requests:
            if req < lower or req > upper: 
                outlier_requests.append(req)
            else:
                new_requests.append(req)
        return new_requests, outlier_requests
        
    
    def find_outliers(self, factor: tuple[int, int]):
        outlier_count = 0
        new_skew = 0
        i = 0
        
        # each iteration (larger factor)
        while i * factor[1] + new_skew < factor[0] * factor[1]:
            # each request in each iteration (smaller factor)
            outlier_lists = {} # count: outliers
            least_outliers = None # outlier_count w the least interference
            while outlier_count == 0 or (len(outlier_req) > 0.5 * factor[0] and outlier_count % factor[0] != 0):
                outlier = False
                outlier_req = []
                outlier_count += 1
                target_reqs = self.requests[i * factor[0] + new_skew + outlier_count : (i + 1) * factor[0] + new_skew + outlier_count]
                first = target_reqs[outlier_count]
                for j in range(outlier_count + 1, factor[0]):
                    if target_reqs[j] / first > (j + 1) ** 2 or target_reqs[j] / first < (j + 1) ** -2:
                        outlier = True
                        outlier_req.append(target_reqs[j])
                outlier_lists[outlier_count] = outlier_req
                least_outliers = min(outlier_count, key=len(outlier_lists.get))
            new_skew = least_outliers
        return 0

            
            

            