import os
import pickle
from typing import List

import numpy as np
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
        # Sort the list from least to greatest
        self.requests_list = sorted(requests)
        corners = [1,2,3,4]
        corner = 1

        positions = [(0, round(np.sqrt(2*self.requests_list[0]), 2)), (round(np.sqrt(2*self.requests_list[0]), 2), 0)]

        x = 1
        while x < len(self.requests_list):
            if corner == 1:
                if positions[-1][0] == 0:
                    base = round(positions[-2][0] + (2*self.requests_list[x]/positions[-1][1]), 2)
                    potential_point = [base, 0]
                elif positions[-1][1] == 0:
                    base = round(positions[-2][1] + (2*self.requests_list[x]/positions[-1][0]), 2)
                    potential_point = [0, base]

            elif corner == 2:
                if positions[-1][1] == 0:
                    base = round(positions[-2][1] + (2*self.requests_list[x]/(self.cake_width-positions[-1][0])), 2)
                    potential_point = [self.cake_width, base]
                elif positions[-1][0] == self.cake_width:
                    base = round(positions[-2][0] - (2*self.requests_list[x]/positions[-1][1]), 2)
                    potential_point = [base, 0]


            points = np.array([list(positions[-1]), list(positions[-2]), potential_point])
            res = miniball.miniball(points)

            if res["radius"] <= 12.5:
                positions.append(tuple(potential_point))

            else:   # Jump to new corner
                if positions[-1][0] == 0 and positions[-2][1] == 0: #TODO
                    positions.append((0.1, self.cake_len))
                    corners.remove(corner)
                    corner = 4

                elif positions[-1][1] == 0 and positions[-2][0] == 0:
                    positions.append((self.cake_width, 0.1))
                    positions.append((round(self.cake_width - np.sqrt(2*self.requests_list[x+1]), 2), 0))
                    positions.append((self.cake_width, round(np.sqrt(2*self.requests_list[x+1]), 2)))
                    corners.remove(corner)
                    corner = 2
                    x += 1

                elif positions[-1][1] == 0 and positions[-2][0] == self.cake_width: #TODO
                    positions.append((self.cake_len, self.cake_width-1))
                    corner = 3

                elif positions[-1][0] == self.cake_width and positions[-2][1] == 0:
                    positions.append((self.cake_width-0.1, self.cake_len))
                    positions.append((self.cake_width, round(self.cake_len - np.sqrt(2*self.requests_list[x+1]), 2)))
                    positions.append((round(self.cake_width - np.sqrt(2*self.requests_list[x+1]), 2), self.cake_len))
                    corners.remove(corner)
                    corner = 3
                    x += 1

            x += 1

        print(positions)
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