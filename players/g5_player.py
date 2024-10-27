import os
import pickle
from typing import List

import numpy as np
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
        self.zigzag_complete = False  # Tracks if the initial zigzag is done
        self.zigzag_positions = []    # Stores calculated zigzag points
        self.current_zigzag_index = 0
       

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
    
    def move(self, current_percept) -> (int, List[int]):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        # Generate zigzag positions if it's the first move
        if turn_number == 1:
            self.generate_zigzag_positions(cake_len, cake_width, len(requests))
            next_pos = self.zigzag_positions[self.current_zigzag_index]
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