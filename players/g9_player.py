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

    def move(self, current_percept) -> tuple[int, List[int]]:
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width
        noise = 0

        print(" ")
        print(f"----------------------------------- Turn {turn_number} -----------------------------------")
        vertical_cut_coords = get_vertical_cuts(requests, cake_len, cake_width, cake_area, noise)
        cut_coords = inject_crumb_coords(vertical_cut_coords, cake_len, cake_width)

        # First turn initialize knife to start at first vertical cut.
        if turn_number == 1:
            print(f"Cut coordinates {len(cut_coords)}: {cut_coords}")
            return constants.INIT, cut_coords[turn_number - 1]

        # Cut the cake
        if turn_number < len(cut_coords) + 1:
            return constants.CUT, cut_coords[turn_number - 1]

        # Assign polygons to each request
        # Find the extra 5% piece and remove it from the list of polygons
        closest_index = min(range(len(polygons)), key=lambda i: abs(polygons[i].area - 0.05 * cake_area))
        polygons.pop(closest_index)
        polygons_sorted_indices = sorted(range(len(polygons)), key=lambda i: polygons[i].area, reverse=True)
        # polygons_sorted = [polygons[i].area for i in polygons_sorted_indices]
        # print("Polygons_sorted:", polygons_sorted)
        # requests_sorted = sorted(requests, reverse=True)
        # print("Requested_sorted:", requests_sorted)
        # Todo - the issue is this list is tuples. The next step is to get the index related to each request and polygon from the original lists.
        assignment = polygons_sorted_indices[:len(requests)]
        return constants.ASSIGN, assignment


def get_vertical_cuts(requests, cake_len, cake_width, cake_area, noise):
    # Sort requests in descending orders (largest to smallest)
    requests.sort(reverse=True)

    # Calculate the percentage of the area of the cake that each request needs
    # Note: The cake is always 5% larger than the total requests so an exact calculation will only account for 95% of the cake leaving 5% on the end. This current setup adds a noise factor to make all pieces evenly larger by % noise. This can be optimized later for pieces on the end to be larger (to account for crumbs etc if that is an issue).

    perc_area = [(r + noise*r) / cake_area for r in requests] + [0.05]
                                                     
    # Get the x coordinate needed for each piece
    x_coords = []
    prev_x = 0
    for frac in perc_area:
        prev_x = round(prev_x + cake_width * frac, 2)
        x_coords.append(prev_x)

    # Duplicate the x coordinates because a single vertical cut will require two x coordinates.
    x_coords = [x for x in x_coords for _ in range(2)]

    # Establish the cutting pattern of down then up (over 4 coordinates) which will repeat.
    y_coords = [0, cake_len, cake_len, 0]

    # Combine x and y coordinates to create tuples. two sets of consecutive coordinates represents a single vertical cut. These are already ordered.
    vertical_cuts = [[x, y] for x, y in zip(x_coords, y_coords * (len(x_coords) // 4))]

    return vertical_cuts


def inject_crumb_coords(vertical_cuts, cake_len, cake_width):
    complete_cut_coords = []
    for i, vert_coord in enumerate(vertical_cuts):
        complete_cut_coords.append(vert_coord)
        # Add an extra element after every second string
        if (i + 1) % 2 == 0:
            complete_cut_coords.append(get_crumb_coord(vert_coord, cake_len, cake_width))
    return complete_cut_coords


def get_crumb_coord(xy_coord, cake_len, cake_width):
    # Set x based on which half of the cake we are in
    crumb_x = cake_width if xy_coord[0] > (cake_width / 2) else 0

    # Set y based on if we are currently at the top or bottom of the cake
    knife_error = 0.01
    crumb_y = round(cake_len - knife_error, 2) if xy_coord[1] == cake_len else knife_error

    return [crumb_x, crumb_y]
