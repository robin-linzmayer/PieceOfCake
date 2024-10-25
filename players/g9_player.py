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

    def move(self, current_percept) -> (int, List[int]):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width
        noise = 0.04

        print(" ")
        print(f"----------------------------------- Turn {turn_number} -----------------------------------")
        vertical_cut_coords = get_vertical_cuts(requests, cake_len, cake_width, cake_area, noise)
        print(f"Vertical Cuts: {len(vertical_cut_coords)}, {len(vertical_cut_coords)/2}, {vertical_cut_coords}")

        if turn_number == 1:
            # First turn initialize knife to start at first vertical cut.
            print(f"RETURNS ROUND {turn_number} ---> {constants.INIT, vertical_cut_coords[turn_number-1]}")
            return constants.INIT, vertical_cut_coords[turn_number-1]

        if len(polygons) < len(requests)+(round(len(requests)/3, 0)):

            # Every 3rd turn is crumbs
            if turn_number % 3 == 0:
                crumb_coord = get_crumb_coord(cur_pos, cake_len, cake_width)
                return constants.CUT, crumb_coord

            print(f"RETURNS ROUND {turn_number} ---> {vertical_cut_coords[turn_number - 1]}")
            return constants.CUT, vertical_cut_coords[turn_number - 1]

        assignment = []
        for i in range(len(requests)):
            assignment.append(i)

        print(f"RETURNS ROUND {turn_number} ---> {constants.ASSIGN, assignment}")
        return constants.ASSIGN, assignment


def get_vertical_cuts(requests, cake_len, cake_width, cake_area, noise):

    # Sort requests in descending orders (largest to smallest)
    requests.sort(reverse=True)

    # Calculate the percentage of the area of the cake that each request needs
    # Note: The cake is always 5% larger than the total requests so an exact calculation will only account for 95% of the cake leaving 5% on the end. This current setup adds a noise factor to make all pieces evenly larger by % noise. This can be optimized later for pieces on the end to be larger (to account for crumbs etc if that is an issue).
    perc_area = [(r + (noise * r)) / cake_area for r in requests]

    # Get the x coordinate needed for each piece
    x_width = []
    prev_x = 0
    for frac in perc_area:
        next_x = round(cake_width * frac, 2)
        x_width.append(round(prev_x + next_x, 2))
        prev_x = prev_x + next_x

    # Translate the calculated width into coordinates where a vertical line is cut
    coord_width = [[x_coord, 0] for x_coord in x_width]
    coord_length = [[x_coord, cake_len] for x_coord in x_width]

    # List of coordinates [cut1_top, cut1_bottom, ..., cutN_top, cutNbottom]
    vertical_cuts = [val for pair in zip(coord_width, coord_length) for val in pair]

    return vertical_cuts

def get_crumb_coord(cur_pos, cake_len, cake_width):

    # Set x based on which half of the cake we are in
    crumb_x = cake_width if cur_pos[0] > (cake_width / 2) else 0

    # Set y based on if we are currently at the top or bottom of the cake
    knife_error = 0.02
    crumb_y = cake_len-knife_error if cur_pos[1] == cake_len else knife_error

    return [crumb_x, crumb_y]