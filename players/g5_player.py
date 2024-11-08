import os
import pickle
from typing import List
from scipy.optimize import linear_sum_assignment

import numpy as np
import math
import logging
import miniball
from shapely.geometry import Polygon

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
        self.zigzag_complete = False  # Tracks if the initial zigzag is done
        self.zigzag_positions = []    # Stores calculated zigzag points
        self.current_zigzag_index = 0
        self.requests_list = []
        #self.polygon_list = [
            #Polygon([(0, 0), (0, self.cake_len), (self.cake_width, self.cake_len), (self.cake_width, 0)])]
       

    def generate_zigzag_positions(self, cake_len, cake_width, segments):
        """Generates zigzag positions from the middle of the left side to the right side."""
        # Start at the middle of the left side
       
        positions = [(0, cake_len)]

        # Move horizontally in increments of `cake_len / segments`
        horizontal_step = 2 * cake_width / segments
        counter = 1
        for i in range(1, segments):
            # Calculate the x-coordinate for this segment
            if i % 2 == 1: 
                next_x = counter * horizontal_step
                counter +=1
            
            # Alternate between top (y=0) and bottom (y=cake_width) positions
            if i % 2 == 1:  # Odd index, move to the bottom
                next_y = 0
            else:  # Even index, move to the top
                next_y = cake_len
            
            positions.append((next_x, next_y))

        # Log the generated positions for clarity
        print(f"Generated zigzag positions across the cake dimensions " f"({cake_len} x {cake_width}) with {segments} segments: {positions}")

        self.zigzag_positions = positions


    def generate_corner_cuts(self, requests):
        """Greedy algorithm that cuts small pieces out of corners"""
        # Sort the list from least to greatest
        self.requests_list = sorted(requests)
        boundary_points = []

        # Add the first isosceles piece
        positions = [(round(np.sqrt(2*self.requests_list[0]), 2), 0), (0, round(np.sqrt(2*self.requests_list[0]), 2))]

        x = 1
        corner = "NW"
        area = self.requests_list[0]

        # Greedy triangle algorithm
        while x < len(self.requests_list):
            points = []

            area += self.requests_list[x]
            base = round(np.sqrt(2*area), 2)

            if corner == "NW":
                horizontal = (base, 0)
                vertical = (0, base)
            elif corner == "SE":
                vertical = (self.cake_width, round(self.cake_len - base, 2))
                horizontal = (round(self.cake_width - base, 2), self.cake_len)
        
            points = np.array([list(vertical), list(horizontal), list(positions[-1]), list(positions[-2])])
            res = miniball.miniball(points)

            # Check to see if future parallelograms will fit on plate
            if corner == "NW":
                para_point = (self.cake_width, round(self.cake_len - max(vertical[1], horizontal[1]), 2))
                para_point2 = (round(self.cake_width - max(vertical[0], horizontal[0]), 2), self.cake_len)
                para_base1 = math.dist(vertical, horizontal)
                para_base2 = min(math.dist(para_point, vertical), math.dist(para_point, horizontal))
                pgram = [vertical, horizontal, para_point, para_point2]
                a = Polygon(pgram).area
                if a < 1:
                    pgram = [pgram[0], pgram[1], pgram[3], pgram[2]]
                    a = Polygon(pgram).area
                height = a / para_base2
                short_base = self.requests_list[-1]/height
                angle = np.pi - np.pi/4 - math.acos((self.cake_len - horizontal[0]) /
                                                        min(math.dist(para_point, horizontal), 
                                                            math.dist(para_point2, horizontal)))
                diagonal = np.sqrt(para_base1**2 + short_base**2 - 2*short_base*para_base1 * np.cos(angle))
                other = np.sqrt(para_base1**2 + short_base**2 + 2*short_base*para_base1 * np.cos(angle))
                if (diagonal > 25 or other > 25) and short_base < height:
                    res["radius"] = max(diagonal, other)

            # Align SE corner with NW corner
            if corner == "SE" and base > max(boundary_points[0][1], boundary_points[1][1]):
                #if res["radius"] < 12.5:
                    if positions[-1][0] == self.cake_width:
                        positions.append((round(self.cake_width-0.01, 2), self.cake_len))
                        positions.append((self.cake_width, 
                                          round(self.cake_len - max(boundary_points[0][1], boundary_points[1][1]) ,2)))
                        positions.append((round(self.cake_width - max(boundary_points[0][1], boundary_points[1][1]) ,2),
                                          self.cake_len))
                        
                    elif positions[-1][1] == self.cake_len:
                        print("hi")
                        positions.append((self.cake_width, round(self.cake_len-0.01, 2)))
                        positions.append((round(self.cake_width - max(boundary_points[0][1], boundary_points[1][1]) ,2),
                                          self.cake_len))
                        positions.append((self.cake_width, 
                                          round(self.cake_len - max(boundary_points[0][1], boundary_points[1][1]) ,2)))
                    res["radius"] = 30


            if res["radius"] <= 12.5:
                # Starting on left edge
                if positions[-1][0] == 0:
                    positions.append((0.01, 0))
                    positions.append(vertical)
                    positions.append(horizontal)
                    
                # Starting on top edge
                elif positions[-1][1] == 0:
                    positions.append((0, 0.01))
                    positions.append(horizontal)
                    positions.append(vertical)

                # Starting on right edge
                elif positions[-1][0] == self.cake_width:
                    positions.append((round(self.cake_width-0.01, 2), self.cake_len))
                    positions.append(vertical)
                    positions.append(horizontal)

                # Starting on bottom edge
                elif positions[-1][1] == self.cake_len:
                    positions.append((self.cake_width, round(self.cake_len-0.01, 2)))
                    positions.append(horizontal)
                    positions.append(vertical)


            else:   # Jump to new corner
                boundary_points.append(positions[-1])
                boundary_points.append(positions[-2])
                if corner == "SE":
                    break

                potential_point_1 = [self.cake_width, self.cake_len]
                potential_point_2 = [round(self.cake_width - np.sqrt(2*self.requests_list[x]), 2), self.cake_len]
                potential_point_3 = [self.cake_width, round(self.cake_len - np.sqrt(2*self.requests_list[x]), 2)]
                points = np.array([potential_point_1, potential_point_2, potential_point_3])
                res = miniball.miniball(points)

                # If next piece too big for southeast corner, stop algorithm
                if res["radius"] > 12.5:
                    break

                # Go to southeast corner from top edge                           
                if positions[-1][1] == 0:
                    positions.append((self.cake_width, 0.01))
                    positions.append((round(self.cake_width-0.01, 2), self.cake_len))
                    if np.sqrt(2*self.requests_list[x]) > max(boundary_points[0][1], boundary_points[1][1]):
                        if positions[-1][0] == self.cake_width:
                            positions.append((round(self.cake_width-0.01, 2), self.cake_len))
                            positions.append((self.cake_width, 
                                            round(self.cake_len - max(boundary_points[0][1], boundary_points[1][1]) ,2)))
                            positions.append((round(self.cake_width - max(boundary_points[0][1], boundary_points[1][1]) ,2),
                                            self.cake_len))
                            
                        elif positions[-1][1] == self.cake_len:
                            positions.append((self.cake_width, round(self.cake_len-0.01, 2)))
                            positions.append((round(self.cake_width - max(boundary_points[0][1], boundary_points[1][1]) ,2),
                                            self.cake_len))
                            positions.append((self.cake_width, 
                                            round(self.cake_len - max(boundary_points[0][1], boundary_points[1][1]) ,2)))
                        break
                    positions.append(tuple(potential_point_3))
                    positions.append(tuple(potential_point_2))
                    corner = "SE"
                    area = self.requests_list[x]

                # Go to southeast corner from left edge
                elif positions[-1][0] == 0:
                    positions.append((0.01, self.cake_len))
                    positions.append((self.cake_width, round(self.cake_len-0.01, 2)))
                    if np.sqrt(2*self.requests_list[x]) > max(boundary_points[0][1], boundary_points[1][1]):
                        if positions[-1][0] == self.cake_width:
                            positions.append((round(self.cake_width-0.01, 2), self.cake_len))
                            positions.append((self.cake_width, 
                                                round(self.cake_len - max(boundary_points[0][1], boundary_points[1][1]) ,2)))
                            positions.append((round(self.cake_width - max(boundary_points[0][1], boundary_points[1][1]) ,2),
                                                self.cake_len))
                            
                        elif positions[-1][1] == self.cake_len:
                            positions.append((self.cake_width, round(self.cake_len-0.01, 2)))
                            positions.append((round(self.cake_width - max(boundary_points[0][1], boundary_points[1][1]) ,2),
                                                self.cake_len))
                            positions.append((self.cake_width, 
                                                round(self.cake_len - max(boundary_points[0][1], boundary_points[1][1]) ,2)))
                            
                        boundary_points.append(positions[-1])
                        boundary_points.append(positions[-2])
                        break
                    positions.append(tuple(potential_point_2))
                    positions.append(tuple(potential_point_3))
                    corner = "SE"
                    area = self.requests_list[x]

            x += 1
            

        # Make parallelogram with last point on bottom edge
        if positions[-1][1] == self.cake_len:
            for b in boundary_points:
                if b[0] == 0:
                    positions.append(b)
                    for bb in boundary_points:
                        if bb[1] == 0:
                            positions.append((round(bb[0]+0.01, 2), bb[1]))
                    positions.append(positions[-4])

        # Make parallelogram
        elif positions[-1][0] == self.cake_width:
                for b in boundary_points:
                    if b[1] == 0:
                        positions.append(b)
                        for bb in boundary_points:
                            if bb[0] == 0:
                                positions.append((bb[0], round(bb[1]+0.01, 2)))
                        positions.append(positions[-4])
        print(positions)
        parallelogram = [positions[-1], positions[-2], positions[-3], positions[-4]]
        area = Polygon(parallelogram).area
        height = round(area/(np.sqrt((positions[-1][1]-positions[-2][1])**2 + (positions[-1][0]-positions[-2][0])**2)), 2)

        angle_1 = math.atan(min(boundary_points[-1][1], boundary_points[-2][1])/min(boundary_points[-1][0], boundary_points[-2][0]))
        angle_2 = np.pi/2 - angle_1
        print(angle_1, angle_2)
        # Angles of side triangles are 28.27 and 61.73 deg

        # Make first parallelogram
        base = self.requests_list[x]/height
        
        # Start from bottom edge
        if positions[-1][1] == self.cake_len:
            horizontal = math.cos(angle_1)*base
            vertical = math.sin(angle_1)*base
            distance = horizontal + vertical/math.tan(np.pi/4)

            temp = self.cake_width - positions[-1][0]

            positions.append((self.cake_width, round(self.cake_len - 0.01, 2)))    
            positions.append((round(positions[-2][0] - distance, 2), self.cake_len))
            positions.append((self.cake_width, round(self.cake_len - temp - distance, 2)))
        
        # Start from right edge
        elif positions[-1][0] == self.cake_width:
            vertical = math.cos(angle_2) * base
            horizontal = math.sin(angle_2) * base
            distance = vertical + horizontal/math.tan(np.pi/4)

            temp = self.cake_len - positions[-1][1]
            positions.append((round(self.cake_width - 0.01, 2), self.cake_len))
            positions.append((self.cake_width, round(positions[-2][1] - distance, 2)))
            
            positions.append((round(self.cake_width - temp - distance, 2), self.cake_len))
        x += 1

        # Make long parallelograms
        while x < len(self.requests_list):
            base = self.requests_list[x]/height

            # Last point on bottom edge
            if positions[-1][1] == self.cake_len:
                horizontal = math.cos(angle_1)*base
                vertical = math.sin(angle_1)*base
                distance = horizontal + vertical/math.tan(np.pi/4)

                temp = self.cake_width - positions[-1][0]

                if positions[-1][0] < distance:
                    leftover = distance - positions[-1][0]
                    positions.append((0, round(self.cake_len - 0.01, 2)))
                    positions.append((0.01, self.cake_len))
                    positions.append((0, round(self.cake_len - leftover, 2)))

                else:
                    positions.append((self.cake_width, round(self.cake_len - 0.01, 2)))    
                    positions.append((round(positions[-2][0] - distance, 2), self.cake_len))

                # If we hit the corner
                if self.cake_len - temp - distance < 0:
                    leftover = distance - (self.cake_len - temp)
                    positions.append((round(self.cake_width - leftover, 2), 0))
                else:
                    positions.append((self.cake_width, round(self.cake_len - temp - distance, 2)))


            # Last point on right edge
            elif positions[-1][0] == self.cake_width:
                vertical = math.cos(angle_2) * base
                horizontal = math.sin(angle_2) * base
                distance = vertical + horizontal/math.tan(np.pi/4)

                temp = self.cake_len - positions[-1][1]

                if positions[-1][1] < distance:
                    leftover = distance - positions[-1][1]
                    positions.append((round(self.cake_width - 0.01, 2), 0))
                    positions.append((self.cake_width, 0.01))
                    positions.append((round(self.cake_width - leftover, 2), 0))

                else:
                    positions.append((round(self.cake_width - 0.01, 2), self.cake_len))
                    positions.append((self.cake_width, round(positions[-2][1] - distance, 2)))
                
                positions.append((round(self.cake_width - temp - distance, 2), self.cake_len))

            # Last point on top edge
            elif positions[-1][1] == 0:
                vertical = math.cos(angle_2) * base
                horizontal = math.sin(angle_2) * base
                distance = vertical + horizontal/math.tan(np.pi/4)

                if positions[-1][0] - distance < max(boundary_points[0][0], boundary_points[1][0]):
                    break

                temp = self.cake_width - positions[-1][0] + self.cake_len

                positions.append((self.cake_width, 0.01))
                positions.append((round(positions[-2][0] - distance, 2), 0))

                if self.cake_width - temp - distance < 0:
                    leftover = distance + (temp - self.cake_width)
                    positions.append((0, round(self.cake_len-leftover, 2)))

                else:
                    positions.append((round(self.cake_width - temp - distance, 2), self.cake_len))

            # Last point on left edge
            elif positions[-1][0] == 0:
                horizontal = math.cos(angle_1)*base
                vertical = math.sin(angle_1)*base
                distance = horizontal + vertical/math.tan(np.pi/4)

                if positions[-1][1] - distance < max(boundary_points[0][1], boundary_points[1][1]):
                    break

                temp = self.cake_width - positions[-1][0]

                positions.append((0.01, self.cake_len))
                positions.append((0, round(positions[-2][1] - distance, 2)))
                positions.append((round(positions[-4][0] - distance, 2), 0))
            
            x += 1


        
        print(positions)
        print(boundary_points)
        self.zigzag_positions = positions


    
    def move(self, current_percept) -> (int, List[int]):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        self.cake_len = cake_len
        self.cake_width = cake_width

        # Generate zigzag positions if it's the first move
        if turn_number == 1:
            if all(x==requests[0] for x in requests):
                self.generate_zigzag_positions(cake_len, cake_width, len(requests))
                next_pos = self.zigzag_positions[self.current_zigzag_index]
                return constants.INIT, [round(next_pos[0], 2), round(next_pos[1], 2)]
            else:
                self.generate_corner_cuts(requests)
                next_pos = self.zigzag_positions[self.current_zigzag_index]
                print("=============")
                return constants.INIT, [round(next_pos[0], 2), round(next_pos[1], 2)]


        # Continue zigzag cutting
        if turn_number > 1 and not self.zigzag_complete:
            self.current_zigzag_index += 1

         
            if self.current_zigzag_index < len(self.zigzag_positions):
                next_pos = self.zigzag_positions[self.current_zigzag_index]
                
                return constants.CUT, [round(next_pos[0], 2), round(next_pos[1], 2)]
            else:
                self.zigzag_complete = True  # Zigzag is done

        
            
        # # After zigzag, make perpendicular cuts
        # if self.zigzag_complete and len(polygons) != len(requests):
        #     # Cut vertically to further divide each segment
        #     x_cut = round(cur_pos[0] + (cake_width / 2), 2) % cake_width
        #     return constants.CUT, [x_cut, cur_pos[1]]

        # Assign pieces to requests after all cuts are done

        print("Matching Polygons to Requests")

        # Assign the pieces
        areas = [i.area for i in polygons]
        #print("areas" + str(areas))
        match = 0
        assignment = []
        final = []
        for s in requests:
            final.append(-1)


        req = sorted(requests)

        for r in req:
            diff = 100
            for a in areas:
                #print(a)
                if abs(r-a) < diff:
                    if areas.index(a) not in assignment:
                        match = a
                        diff = abs(r-a)
            assignment.append(areas.index(match))

        for r in range(len(req)):
            for s in range(len(requests)):
                if req[r] == requests[s] and final[s] == -1 and assignment[r] not in final:
                    final[s] = assignment[r]


        '''assignment = sorted(range(len(areas)), key=lambda x: areas[x], reverse=True)
        print(assignment)
        print(assignment[:len(requests)])
        return constants.ASSIGN, assignment[:len(requests)][::-1]'''

        return constants.ASSIGN, final

        '''assignment = self.return_matches(polygons, requests)
        return constants.ASSIGN, assignment
        

        # assignment = list(range(len(requests)))
        # return constants.ASSIGN, assignment

    def round_position(self, position: List[float]) -> List[float]:
        return [round(position[0], 2), round(position[1], 2)]
    

    def validate_position(self, position: List[float], cake_len: int, cake_width: int) -> List[float]:
        x, y = position
        x = max(0, min(x, cake_width))  # Bound x within the width
        y = max(0, min(y, cake_len))    # Bound y within the length
        return self.round_position([x, y])
    

    def return_matches(self, polygons, requests):
        matches = self.hungarian_algorithm(polygons, requests)
        print(matches)

        # return the indices of the polygons in order of the requests
        assignment = [match[1] for match in matches]

        return assignment
    
    def hungarian_algorithm(self, polygons, requests):
        """
            Function to implement the Hungarian algorithm for optimal assignment
        """
        cost_matrix = self.create_cost_matrix(polygons, requests)

        # Use the Hungarian method to find the optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i][j] < np.inf:
                penalty = cost_matrix[i][j]

                # THIS RETURNS AN INDEX!!!! MAY NEED TO BE ADJUSTED
                matches.append((j, i, penalty))


        return matches
    
    def create_cost_matrix(self, polygons, requests):
            n = len(polygons)
            m = len(requests)

            cost_matrix = np.full((n, m), np.inf)  # Initialize with infinity

            for i in range(n):
                for j in range(m):

                    difference = abs(polygons[i].area - requests[j])

                    if difference <= self.tolerance:
                        cost_matrix[i][j] = 0		# No penalty
                    else:
                        cost_matrix[i][j] = (difference / requests[j]) * 100 # Penalty



            return cost_matrix'''