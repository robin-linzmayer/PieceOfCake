from collections import defaultdict
import os
import pickle
from typing import List

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
            self.requests = current_percept.requests
            self.cake_len = current_percept.cake_len
            self.cake_width = current_percept.cake_width
            self.requests.sort(reverse=False)
            if not self.uniform_mode:
                # print("BEFORE")
                is_uniform, grid_area = self.if_uniform(current_percept.requests, extra_tol)
                # print("AFTER")
                print(is_uniform, grid_area)
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
                # print("I AM TRYING TO ASSIGN PIECES")
                assignment = self.assignPolygons(polygons=polygons)
                return constants.ASSIGN, assignment
            # print("I AM IN UNIFORM MODE")
            # print("Turn Number: ", turn_number)
            # print("Cut Number: ", cut_number)
            if self.turn_number == 1:
                # print(self.uniform_cuts)
                # print(self.uniform_cuts[cut_number])
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
            
            for factor in list_of_factors[::-1]:
                penalty, cuts = self.simulate_cuts(factor)
                factor_penalty[tuple(factor)] = penalty
                factor_cuts[tuple(factor)] = cuts

            best_factor = min(factor_penalty, key=factor_penalty.get)
            best_cut = factor_cuts[best_factor]
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
            # print("self.angle_cuts is: ", self.angle_cuts)
            # print("best cuts is: ", best_cut)
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
            print("CHECK IF TARGET [0] = 0,", target[0], type(target[0]))
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
        for j in range (factor[0] + 1):
            horizontals.append(round(self.margin + (self.working_height * j), 2))
        print ("working_height:", self.working_height)
        print ("horizontals:", horizontals)
        for i in range(factor[1]):
            tolerances = self.find_acceptable_range(factor[0], i) # Returns tolerances of the smaller factors
            penalty = self.angle_sweep(tolerances, factor[0], i, horizontals)
            total_penalty += penalty

        cut_arr = self.angle_cuts.copy()
        self.angle_cuts = []

        return total_penalty, cut_arr
    
    def calcDiagonal(self) -> float:
        return (math.sqrt((self.cake_len * self.cake_len) + (self.cake_width * self.cake_width)))
    
    def smallCake(self) -> tuple[int, list[float, float]]:
        current_area = self.requests[self.cut_number - 1]
        
        x = self.cur_pos[0]
        y = self.cur_pos[1]

        
        if self.base_case_switch[0]:
            # TODO: this currently retraces the past cut and creates triangles from the same point
            x = self.cuts[self.cut_number - 2][0]
            y = self.cuts[self.cut_number - 2][1] # on the first occurrence, this will be 0 if we went from bottom to right and cake_len if we went from top to right
            print ("x, y: ", x, y)
            # TODO: make this align with everything else
            print ("y from two turns ago:", self.cuts[self.cut_number - 2][1])
            print ("from one turn ago:", self.cuts[self.cut_number - 1][0], self.cuts[self.cut_number - 1][1])
            print ("adjustment for next area:", 2 * current_area / self.working_height)
            print ("x_y_switch:", self.x_y_switch)
            print ("base_case_switch:", self.base_case_switch)

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
            print ("area left:", area_left)
            print ("working height:", self.working_height)
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
                
    #Returns whether if it is uniform, and the area of the uniform. If not, area is -1
    def if_uniform(self, requests, extra_tol=0.0) -> tuple[bool, float]:

        # If prime return false and use zig zag.
        if self.find_factors(len(requests), True) == False:
            return (False, -1)

        tolerance = (self.tolerance + extra_tol) / 100
        skewed_average = (1 - tolerance**2)*(sum(requests) / len(requests)) if tolerance < 1 else (tolerance**2)**(sum(requests) / len(requests))
        print("THIS IS OUR TOLERANCE: ", tolerance)
        print("THIS IS OUR TRUE AVG: ", sum(requests)/len(requests))
        print("THIS IS OUR SKEWED AVG: ", skewed_average)
        for req in requests:
            if (abs(req-skewed_average)/req) >= tolerance:
                return (False, -1)
            
        return (True, skewed_average)
    
    #Cuts into grid using crumbs method, returns the coordinate to cut to.
    def grid_cut(self, current_percept, grid_area) -> list[float, float]:
        # print("IN GRID CUT")
        print(self.cake_len)
        # length_shave = 0.01266028 * self.cake_len
        
        # width_shave = round(1.6 * length_shave, 2)
        # length_shave = round(length_shave, 2)

        width_shave = 0.01
        length_shave = 0.01

        # print("BEFORE FACTORS")
        s_factor, l_factor = self.find_factors(closest=True)

        print("MY FACTORS: ", s_factor, l_factor)
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

        print("X CUTS: ", x_cuts)
        print("Y CUTS: ", y_cuts)

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
                
           
        print("PATH TO VICTORY: ", self.uniform_cuts)

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
                #print(self.angle_cuts)
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
        #print ("tolerances:", temp)
        return temp

    # angle sweep does a search of all possible lines and adds the one with least penalty to the self.angle_cuts variable, returns the penalty
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
        min_penalty = (math.inf, ((0, 0), (0, 0)), 0) # penalty, cuts, slope 
        for coord1x in np.arange(x_range[0][0], x_range[1][0], 0.01):
            for coord2x in np.arange(x_range[0][1], x_range[1][1], 0.01):
                cuts = None
                penalty = 0
                coord1 = (coord1x, self.margin + (self.working_height / 2))
                coord2 = (coord2x, self.cake_len - (self.margin + (self.working_height / 2)))
                slope, intercept = self.equation_of_line(coord1, coord2)
                if not slope:
                    cuts = (
                        (coord1[0], 0), (coord1[0], self.cake_len)
                    )
                else:
                    cuts = (
                        self.point_on_line(slope, intercept, 0, False), # coordinate for top of cake
                        self.point_on_line(slope, intercept, -1 * self.cake_len, False) # coordinate for bottom of cake
                    )
                # check penalty for all other requests 
                for i in range(factor):
                    req = target_reqs[i]
                    if (coord1x == coord2x):
                        if coord1x < tolerances[i][0]:
                            penalty += 100 * ((tolerances[i][0] - coord1x) * self.working_height) / req
                        elif coord1x > tolerances[i][1]:
                            penalty += 100 * ((coord1x - tolerances[i][1]) * self.working_height) / req
                    else:
                        penalty += self.calculate_penalty(req, slope, intercept, (horizontals[i], horizontals[i + 1]))
                if penalty == min_penalty[0] and not slope:
                    min_penalty = (penalty, cuts, math.inf)
                elif penalty < min_penalty[0] or (penalty == min_penalty[0] and abs(slope) > min_penalty[2]):
                    min_penalty = (penalty, cuts, slope)
        self.angle_cuts.append(min_penalty[1])
        return min_penalty[0]
        
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
        #print ("corners:", corner3, corner1, corner2, corner4)
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
    
