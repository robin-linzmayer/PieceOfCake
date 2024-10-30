import os
import pickle
from typing import List

import numpy as np
import math
import logging
import miniball

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
        print(self.requests_list)
        boundary_points = []

        # Add the first isosceles piece
        positions = [(round(np.sqrt(2*self.requests_list[0]), 2), 0), (0, round(np.sqrt(2*self.requests_list[0]), 2))]

        x = 1
        corner = "NW"
        area = self.requests_list[0]
        while x < len(self.requests_list):

            '''# In northwest corner, go from left edge to top edge
            if positions[-1][0] == 0:
                base = round(positions[-2][0] + (2*self.requests_list[x]/positions[-1][1]), 2)
                potential_point = [base, 0]

            # In northwest corner, go from top edge to left edge
            elif positions[-1][1] == 0:
                base = round(positions[-2][1] + (2*self.requests_list[x]/positions[-1][0]), 2)
                potential_point = [0, base]

            # In southeast corner, go from right edge to bottom edge
            elif positions[-1][0] == self.cake_width:
                base = round(positions[-2][0] - (2*self.requests_list[x]/(self.cake_len-positions[-1][1])), 2)
                potential_point = [base, self.cake_len]

            # In southeast corner, go from bottom edge to right edge
            elif positions[-1][1] == self.cake_len:
                base = round(positions[-2][1] - (2*self.requests_list[x]/(self.cake_width-positions[-1][0])), 2)
                potential_point = [self.cake_width, base]

            # Check if piece fits on plate
            points = np.array([list(positions[-1]), list(positions[-2]), potential_point])
            res = miniball.miniball(points)'''

            points = []

            area += self.requests_list[x]
            base = round(np.sqrt(2*area), 2)

            if corner == "NW":
                horizontal = (base, 0)
                vertical = (0, base)
            elif corner == "SE":
                vertical = (self.cake_width, self.cake_len - base)
                horizontal = (self.cake_width - base, self.cake_len)
        
            points = np.array([list(vertical), list(horizontal), list(positions[-1]), list(positions[-2])])
            res = miniball.miniball(points)

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
                    positions.append((self.cake_width-0.01, self.cake_len))
                    positions.append(vertical)
                    positions.append(horizontal)

                # Starting on bottom edge
                elif positions[-1][1] == self.cake_len:
                    positions.append((self.cake_width, self.cake_len-0.01))
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
                    positions.append((self.cake_width-0.01, self.cake_len))
                    positions.append(tuple(potential_point_3))
                    positions.append(tuple(potential_point_2))
                    corner = "SE"
                    area = self.requests_list[x]

                # Go to southeast corner from left edge
                elif positions[-1][0] == 0:
                    positions.append((0.01, self.cake_len))
                    positions.append((self.cake_width, self.cake_len-0.01))
                    positions.append(tuple(potential_point_2))
                    positions.append(tuple(potential_point_3))
                    corner = "SE"
                    area = self.requests_list[x]

            x += 1

        if positions[-1][1] == self.cake_len:
            for b in boundary_points:
                if b[0] == 0:
                    positions.append(b)
                    for bb in boundary_points:
                        if bb[1] == 0:
                            positions.append((bb[0]+0.01, bb[1]))
                    positions.append(positions[-4])
        elif positions[-1][0] == self.cake_width:
                for b in boundary_points:
                    if b[1] == 0:
                        positions.append(b)
                        for bb in boundary_points:
                            if bb[0] == 0:
                                positions.append((bb[0], bb[1]+0.01))
                        positions.append(positions[-4])

        
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
                print(next_pos)
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

        # After zigzag, make perpendicular cuts
        if self.zigzag_complete and len(polygons) != len(requests):
            # Cut vertically to further divide each segment
            x_cut = round(cur_pos[0] + (cake_width / 2), 2) % cake_width
            return constants.CUT, [x_cut, cur_pos[1]]

        # Assign pieces to requests after all cuts are done
        assignment = list(range(len(requests)))
        return constants.ASSIGN, assignment

    def round_position(self, position: List[float]) -> List[float]:
        return [round(position[0], 2), round(position[1], 2)]

    def validate_position(self, position: List[float], cake_len: int, cake_width: int) -> List[float]:
        x, y = position
        x = max(0, min(x, cake_width))  # Bound x within the width
        y = max(0, min(y, cake_len))    # Bound y within the length
        return self.round_position([x, y])