from typing import List
from itertools import combinations
from scipy.optimize import linear_sum_assignment

import math
import numpy as np
import logging
import constants

"""
Constants
"""
MIN_CUT_INCREMENT = 0.01
EASY_LEN_BOUND = 23.507

"""
GLOBAL FUNCTIONS
"""
def optimal_assignment(R, V):
    """
    Provide optimal assignment using the Hungarian algorithm.
    """
    num_requests = len(R)
    num_values = len(V)
    
    cost_matrix = np.zeros((num_requests, num_values))
    
    # Fill the cost matrix with relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    # Solving the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # Assignment array where assignment[i] is the index of V matched to R[i]
    assignment = [int(col_indices[i]) for i in range(num_requests)]
    
    return assignment


def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def find_ratio_groupings(requests, m, tolerance):
    """
    Find all valid groupings of requests that meet the required ratio within tolerance.
    Return groupings that group as many requests as possible, possibly leaving some ungrouped.
    Each valid group includes the lower bound of the overlapping range.
    """
    # Define target ratios for each group size
    if m == 2:
        target_ratios = [1, 3]
    elif m == 3:
        target_ratios = [1, 3, 5]
    elif m == 4:
        target_ratios = [1, 3, 5, 7]
    else:
        print("Invalid group size.")
        return []

    valid_groupings = []
    min_remaining_requests = len(requests)
    
    def is_valid_group(group):
        """
        Check if a group meets the required ratio within tolerance and return the lower bound.
        """
        normalized_group = [group[i] / target_ratios[i] for i in range(len(group))]
        # Find overlapping range
        lower_bounds = [nr - (tolerance / 100) * nr for nr in normalized_group]
        upper_bounds = [nr + (tolerance / 100) * nr for nr in normalized_group]
        lower_bound = max(lower_bounds)
        upper_bound = min(upper_bounds)
        if lower_bound <= upper_bound:
            return True, lower_bound
        return False, None
    
    def backtrack(remaining_requests, current_grouping):
        """Backtracking function to find the valid groupings."""
        nonlocal min_remaining_requests
        nonlocal valid_groupings

        # If all requests are grouped, update the results
        if not remaining_requests:
            if min_remaining_requests > 0:
                min_remaining_requests = 0
                valid_groupings.clear()
                valid_groupings.append({'grouping': current_grouping, 'ungrouped': []})
            elif min_remaining_requests == 0:
                valid_groupings.append({'grouping': current_grouping, 'ungrouped': []})
            return
        
        # If current ungrouped requests exceed min_remaining_requests, prune this path
        if len(remaining_requests) > min_remaining_requests:
            return
        
        # Try forming groups from remaining requests
        for group in combinations(remaining_requests, m):
            is_valid, lower_bound = is_valid_group(group)
            if is_valid:
                new_remaining = remaining_requests[:]
                for num in group:
                    new_remaining.remove(num)
                # Recur with the new list and updated grouping, including the lower bound
                backtrack(new_remaining, current_grouping + [(group, lower_bound)])

        # Consider the possibility of leaving some requests ungrouped
        if len(current_grouping) > 0:
            if len(remaining_requests) < min_remaining_requests:
                min_remaining_requests = len(remaining_requests)
                valid_groupings.clear()
                valid_groupings.append({'grouping': current_grouping, 'ungrouped': remaining_requests})
            elif len(remaining_requests) == min_remaining_requests:
                valid_groupings.append({'grouping': current_grouping, 'ungrouped': remaining_requests})

    # Start backtracking
    backtrack(requests, [])
    
    return valid_groupings

"""
Player Class
"""
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
        self.num_requests_cut = 0
        self.knife_pos = []
        self.num_horizontal = None
        self.cuts_created = False
        self.pending_cuts = []

    
    def add_available_cut(self, origin, dest, coord, increment):
        """
        Add cut to the list of pending cuts. If the cut is in the list of old cuts,
        increment the appropriate coordinate by increment.
        """
        new_dest = [round(dest[0], 2), round(dest[1], 2)]
        cut = (origin[0], origin[1], new_dest[0], new_dest[1])
        sym_cut = (cut[2], cut[3], cut[0], cut[1])
        while cut in self.pending_cuts or sym_cut in self.pending_cuts:
            new_dest[coord] += increment
            cut = (origin[0], origin[1], new_dest[0], new_dest[1])
            sym_cut = (cut[2], cut[3], cut[0], cut[1])
        self.pending_cuts.append(cut)
        self.knife_pos.append(new_dest)
        return new_dest


    def traverse_borders(self, from_pos, to_pos):
        """
        Helper function that moves the knife from current position to next position by
        traversing the borders.
        """
        # Move to same border (should require 2 cuts)
        if from_pos[0] == 0 and to_pos[0] == 0:
            # Traverse along the left border
            if from_pos[1] + to_pos[1] < self.cake_len:
                # Move to top border, which is closer
                interim_pos = self.add_available_cut(from_pos, [MIN_CUT_INCREMENT, 0], 0, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 1, -MIN_CUT_INCREMENT)
            else:
                # Move to bottom border, which is closer
                interim_pos = self.add_available_cut(from_pos, [MIN_CUT_INCREMENT, self.cake_len], 0, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 1, MIN_CUT_INCREMENT)
        elif from_pos[0] == self.cake_width and to_pos[0] == self.cake_width:
            # Traverse along the right border
            if from_pos[1] + to_pos[1] < self.cake_len:
                # Move to top border, which is closer
                interim_pos = self.add_available_cut(from_pos, [self.cake_width-MIN_CUT_INCREMENT, 0], 0, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 1, -MIN_CUT_INCREMENT)
            else:
                # Move to bottom border, which is closer
                interim_pos = self.add_available_cut(from_pos, [self.cake_width-MIN_CUT_INCREMENT, self.cake_len], 0, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 1, MIN_CUT_INCREMENT)
        elif from_pos[1] == 0 and to_pos[1] == 0:
            # Traverse along the top border
            if from_pos[0] + to_pos[0] < self.cake_width:
                # Move to left border, which is closer
                interim_pos = self.add_available_cut(from_pos, [0, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 0, -MIN_CUT_INCREMENT)
            else:
                # Move to right border, which is closer
                interim_pos = self.add_available_cut(from_pos, [self.cake_width, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 0, MIN_CUT_INCREMENT)
        elif from_pos[1] == self.cake_len and to_pos[1] == self.cake_len:
            # Traverse along the bottom border
            if from_pos[0] + to_pos[0] < self.cake_width:
                # Move to left border, which is closer
                interim_pos = self.add_available_cut(from_pos, [0, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 0, -MIN_CUT_INCREMENT)
            else:
                # Move to right border, which is closer
                interim_pos = self.add_available_cut(from_pos, [self.cake_width, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos, to_pos, 0, MIN_CUT_INCREMENT)

        # Move to adjacent border (should require 3 cuts)
        elif from_pos[0] == 0 and to_pos[1] == 0:
            # Traverse from left to top border
            interim_pos_1 = self.add_available_cut(from_pos, [MIN_CUT_INCREMENT, 0], 0, MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [0, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 0, -MIN_CUT_INCREMENT)
        elif from_pos[0] == 0 and to_pos[1] == self.cake_len:
            # Traverse from left to bottom border
            interim_pos_1 = self.add_available_cut(from_pos, [MIN_CUT_INCREMENT, self.cake_len], 0, MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [0, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 0, -MIN_CUT_INCREMENT)
        elif from_pos[0] == self.cake_width and to_pos[1] == 0:
            # Traverse from right to top border
            interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width-MIN_CUT_INCREMENT, 0], 0, -MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 0, MIN_CUT_INCREMENT)
        elif from_pos[0] == self.cake_width and to_pos[1] == self.cake_len:
            # Traverse from right to bottom border
            interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width-MIN_CUT_INCREMENT, self.cake_len], 0, -MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 0, MIN_CUT_INCREMENT)
        elif from_pos[1] == 0 and to_pos[0] == 0:
            # Traverse from top to left border
            interim_pos_1 = self.add_available_cut(from_pos, [0, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [MIN_CUT_INCREMENT, 0], 0, MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 1, -MIN_CUT_INCREMENT)
        elif from_pos[1] == 0 and to_pos[0] == self.cake_width:
            # Traverse from top to right border
            interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width-MIN_CUT_INCREMENT, 0], 0, -MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 1, -MIN_CUT_INCREMENT)
        elif from_pos[1] == self.cake_len and to_pos[0] == 0:
            # Traverse from bottom to left border
            interim_pos_1 = self.add_available_cut(from_pos, [0, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [MIN_CUT_INCREMENT, self.cake_len], 0, MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 1, MIN_CUT_INCREMENT)
        elif from_pos[1] == self.cake_len and to_pos[0] == self.cake_width:
            # Traverse from bottom to right border
            interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
            interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width-MIN_CUT_INCREMENT, self.cake_len], 0, -MIN_CUT_INCREMENT)
            return self.add_available_cut(interim_pos_2, to_pos, 1, MIN_CUT_INCREMENT)

        # Move to opposite border (should require 4 cuts)
        elif from_pos[0] == 0 and to_pos[0] == self.cake_width:
            # Traverse from left to right border
            if from_pos[1] + to_pos[1] < self.cake_len:
                # Move to top border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [MIN_CUT_INCREMENT, 0], 0, MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [self.cake_width-MIN_CUT_INCREMENT, 0], 0, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 1, -MIN_CUT_INCREMENT)
            else:
                # Move to bottom border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [MIN_CUT_INCREMENT, self.cake_len], 0, MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [self.cake_width-MIN_CUT_INCREMENT, self.cake_len], 0, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 1, MIN_CUT_INCREMENT)
        elif from_pos[0] == self.cake_width and to_pos[0] == 0:
            # Traverse from right to left border
            if from_pos[1] + to_pos[1] < self.cake_len:
                # Move to top border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width-MIN_CUT_INCREMENT, 0], 0, -MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [0, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [MIN_CUT_INCREMENT, 0], 0, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 1, -MIN_CUT_INCREMENT)
            else:
                # Move to bottom border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width-MIN_CUT_INCREMENT, self.cake_len], 0, -MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [0, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [MIN_CUT_INCREMENT, self.cake_len], 0, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 1, MIN_CUT_INCREMENT)
        elif from_pos[1] == 0 and to_pos[1] == self.cake_len:
            # Traverse from top to bottom border
            if from_pos[0] + to_pos[0] < self.cake_width:
                # Move to left border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [0, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [MIN_CUT_INCREMENT, self.cake_len], 0, MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [0, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 0, -MIN_CUT_INCREMENT)
            else:
                # Move to right border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width-MIN_CUT_INCREMENT, self.cake_len], 0, -MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [self.cake_width, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 0, MIN_CUT_INCREMENT)
        elif from_pos[1] == self.cake_len and to_pos[1] == 0:
            # Traverse from bottom to top border
            if from_pos[0] + to_pos[0] < self.cake_width:
                # Move to left border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [0, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [MIN_CUT_INCREMENT, 0], 0, MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [0, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 0, -MIN_CUT_INCREMENT)
            else:
                # Move to right border, which is closer
                interim_pos_1 = self.add_available_cut(from_pos, [self.cake_width, self.cake_len-MIN_CUT_INCREMENT], 1, -MIN_CUT_INCREMENT)
                interim_pos_2 = self.add_available_cut(interim_pos_1, [self.cake_width-MIN_CUT_INCREMENT, 0], 0, -MIN_CUT_INCREMENT)
                interim_pos_3 = self.add_available_cut(interim_pos_2, [self.cake_width, MIN_CUT_INCREMENT], 1, MIN_CUT_INCREMENT)
                return self.add_available_cut(interim_pos_3, to_pos, 0, MIN_CUT_INCREMENT)


    def set_starting_pos(self):
        """
        Set starting knife position based on num_horizontal.
        """
        if self.num_horizontal == 2:
            # start from the right
            y_pos = round(self.cake_len / 2, 2)
            self.knife_pos.append([self.cake_width, y_pos])
            return constants.INIT, [self.cake_width, y_pos]
        elif self.num_horizontal == 3:
            # start from the left
            y_pos = round(self.cake_len / 3, 2)
            self.knife_pos.append([0, y_pos])
            return constants.INIT, [0, y_pos]
        elif self.num_horizontal == 4:
            # start from the right
            y_pos = round(self.cake_len / 4, 2)
            self.knife_pos.append([self.cake_width, y_pos])
            return constants.INIT, [self.cake_width, y_pos]

    
    def divide_horizontally(self):
        """
        Divide the cake into horizontal slices of thickness < EASY_LEN_BOUND.
        """
        cur_pos = self.knife_pos[-1]
        if self.num_horizontal == 2:
            # make one horizontal cut right to left
            cut = (cur_pos[0], cur_pos[1], 0, cur_pos[1])
            self.pending_cuts.append(cut)
            self.knife_pos.append([cut[2], cut[3]])

        elif self.num_horizontal == 3:
            # make two horizontal cuts left to right, then right to left
            cut_1 = (cur_pos[0], cur_pos[1], self.cake_width, cur_pos[1])
            self.pending_cuts.append(cut_1)
            self.knife_pos.append([cut_1[2], cut_1[3]])

            interim_y = cut_1[3] + round(self.cake_len / 3, 2)
            interim_pos = self.traverse_borders([cut_1[2], cut_1[3]], [cut_1[2], interim_y])
            
            cut_2 = (interim_pos[0], interim_pos[1], 0, interim_pos[1])
            self.pending_cuts.append(cut_2)
            self.knife_pos.append([cut_2[2], cut_2[3]])
                                  
        elif self.num_horizontal == 4:
            # make three horizontal cuts right to left, then left to right, then right to left
            cut_1 = (cur_pos[0], cur_pos[1], 0, cur_pos[1])
            self.pending_cuts.append(cut_1)
            self.knife_pos.append([cut_1[2], cut_1[3]])

            interim1_y = cut_1[3] + round(self.cake_len / 4, 2)
            interim1_pos = self.traverse_borders([cut_1[2], cut_1[3]], [cut_1[2], interim1_y])

            cut_2 = (interim1_pos[0], interim1_pos[1], self.cake_width, interim1_pos[1])
            self.pending_cuts.append(cut_2)
            self.knife_pos.append([cut_2[2], cut_2[3]])

            interim2_y = cut_2[3] + round(self.cake_len / 4, 2)
            interim2_pos = self.traverse_borders([cut_2[2], cut_2[3]], [cut_2[2], interim2_y])

            cut_3 = (interim2_pos[0], interim2_pos[1], 0, interim2_pos[1])
            self.pending_cuts.append(cut_3)
            self.knife_pos.append([cut_3[2], cut_3[3]])


    def make_rectangles(self, unassigned_requests):
        """
        Find groups of m requests (where m is the number of horizontal slices) of the
        same size within the tolerance. Make verticle cuts to serve rectangular pieces
        of same size for each group.
        """
        m = self.num_horizontal
        i = 0

        while len(unassigned_requests) >= m and i <= len(unassigned_requests) - m:
            # find m requests within tolerance from their mean
            group = [unassigned_requests[i]]
            group_mean = round(sum(unassigned_requests[i:i+m]) / m, 2)
            for j in range(i+1, i+m):
                if abs(unassigned_requests[j] - group_mean) / unassigned_requests[j] * 100 <= self.tolerance:
                    group.append(unassigned_requests[j])

            # make verticle cut to serve rectangular pieces of similar size
            if len(group) == m:
                cur_pos = self.knife_pos[-1]
                width = round(m * group_mean / self.cake_len, 2)
                y_dest = 0 if cur_pos[1] == 0 else self.cake_len
                interim_pos = self.traverse_borders(cur_pos, [cur_pos[0]+width, y_dest])
                vert_cut = (interim_pos[0], interim_pos[1], interim_pos[0], self.cake_len-y_dest)
                self.pending_cuts.append(vert_cut)
                self.knife_pos.append([vert_cut[2], vert_cut[3]])
                # remove assigned requests
                for req in group:
                    unassigned_requests.remove(req)
            else:
                i += 1


    def make_triangles(self, unassigned_requests):
        """
        Optimally allocate remaining pieces by making diagonal cuts and serving
        triangular pieces.
        """
        triangle_groups = find_ratio_groupings(unassigned_requests, self.num_horizontal, self.tolerance)
        # for group in triangle_groups:

            


    def move(self, current_percept) -> (int, List[int]):
        """Function which returns an action.

            Args:
                current_percept(PieceOfCakeState): contains current cake state information
            Returns:
                a tuple of format ([action=1,2,3], [list])
                if action = 1: list is [0,0]
                if action = 2: list is [x, y] (coordinates for knife position after cut)
                if action = 3: list is [p1, p2, ..., pl] (the order of polygon assignment)

        """
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests

        # store cake dimensions in class variables
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width


        ####################
        # CUTTING STRATEGY #
        ####################

        # sort requests by area in ascending order
        requests = sorted(requests)
        num_requests = len(requests)

        # initialize unassigned requests for large cake algorithm
        unassigned_requests = requests.copy()
        cake_area = self.cake_len * self.cake_width

        # slice off 5% extra if only one request
        if num_requests == 1:
            if turn_number == 1:
                return constants.INIT, [0,0]
            elif self.num_requests_cut < num_requests:
                extra_area = 0.05 * requests[0]
                extra_base = round(2 * extra_area / self.cake_len, 2)
                self.num_requests_cut += 1
                return constants.CUT, [extra_base, self.cake_len]

        # all other non-edge cases
        elif num_requests > 1:
            # case where cake is smaller than EASY_LEN_BOUND
            if self.cake_len <= EASY_LEN_BOUND:
                # initialize starting knife position
                if turn_number == 1:
                    self.knife_pos.append([0,0])
                    return constants.INIT, [0,0]
                
                if self.num_requests_cut < num_requests:
                    # compute size of base from current polygon area
                    curr_polygon_area = requests[self.num_requests_cut]
                    curr_polygon_base = round(2 * curr_polygon_area / self.cake_len, 2)

                    if cur_pos[1] == 0:
                        # knife is currently on the top cake edge
                        if turn_number == 2:
                            next_knife_pos = [curr_polygon_base, self.cake_len]
                        else:
                            next_x = round(self.knife_pos[-2][0] + curr_polygon_base, 2)
                            next_y = self.cake_len
                            # when knife goes over the cake width
                            if next_x > self.cake_width:
                                next_x = self.cake_width
                                next_y = round(2 * cake_area * 0.05 / (self.cake_width - self.knife_pos[-2][0]), 2)
                            next_knife_pos = [next_x, next_y]

                        self.knife_pos.append(next_knife_pos)
                        self.num_requests_cut += 1
                        return constants.CUT, next_knife_pos
                    else:
                        # knife is currently on the bottom cake edge
                        next_x = round(self.knife_pos[-2][0] + curr_polygon_base, 2)
                        next_y = 0
                        # when knife goes over the cake width
                        if next_x > self.cake_width:
                            next_x = self.cake_width
                            next_y = self.cake_len - round(2 * cake_area * 0.05 / (self.cake_width - self.knife_pos[-2][0]), 2)
                        next_knife_pos = [next_x, next_y]
                        self.knife_pos.append(next_knife_pos)
                        self.num_requests_cut += 1
                        return constants.CUT, next_knife_pos

            # case where cake is larger than EASY_LEN_BOUND
            else:
                if not self.cuts_created:
                    # set number of horizontal slices
                    # [TODO -- ADD ONE MORE LAYER TO ADDRESS MINIBALL ISSUE]
                    if self.cake_len <= 2 * EASY_LEN_BOUND:
                        self.num_horizontal = 2
                    elif self.cake_len <= 3 * EASY_LEN_BOUND:
                        self.num_horizontal = 3
                    else:
                        self.num_horizontal = 4

                    # initialize starting knife position
                    if self.knife_pos == []:
                        return self.set_starting_pos()

                    # main cutting algorithm
                    self.divide_horizontally()
                    self.make_rectangles(unassigned_requests)
                    self.make_triangles(unassigned_requests)
                    self.cuts_created = True

                if len(self.pending_cuts) > 0:
                    # return pending cuts
                    next_cut = self.pending_cuts.pop(0)
                    return constants.CUT, [next_cut[2], next_cut[3]]


        #######################
        # ASSIGNMENT STRATEGY #
        #######################
        V = [p.area for p in polygons]
        assignment = optimal_assignment(current_percept.requests, V)

        return constants.ASSIGN, assignment
    
    
    
