import os
import pickle
from typing import List

import numpy as np
import logging
import traceback

from numpy.distutils.system_info import wx_info

import constants
import math
import itertools
from scipy.optimize import linear_sum_assignment

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

        # precomp_path = os.path.join(precomp_dir, "{}.pkl".format(map_path))

        # # precompute check
        # if os.path.isfile(precomp_path):
        #     # Getting back the objects:
        #     with open(precomp_path, "rb") as f:
        #         self.obj0, self.obj1, self.obj2 = pickle.load(f)
        # else:
        #     # Compute objects to store
        #     self.obj0, self.obj1, self.obj2 = _

        #     # Dump the objects
        #     with open(precomp_path, 'wb') as f:
        #         pickle.dump([self.obj0, self.obj1, self.obj2], f)

        self.rng = rng
        self.logger = logger
        self.tolerance = tolerance
        self.cake_len = None
        self.cake_width = None
        self.cuts = set()
        self.cutList =[]
        self.requests = None

    # Move the knife to the side in the clockwise direction
    def move_knife_clockwise(self, cur_pos):
        if cur_pos[0] == 0 and cur_pos[1] != 0:
            val = 0.01
            while [cur_pos[0], cur_pos[1], val, 0]  in self.cuts or [val, 0, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [val, 0]
        elif cur_pos[1] == 0 and cur_pos[0] != self.cake_width:
            val = 0.01
            while [cur_pos[0], cur_pos[1], self.cake_width, val] in self.cuts or [self.cake_width, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [self.cake_width, val]
        elif cur_pos[0] == self.cake_width and cur_pos[1] != self.cake_len:
            val = self.cake_width - 0.01
            while [cur_pos[0], cur_pos[1], val, self.cake_len] in self.cuts or [val, self.cake_len, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [val, self.cake_len]
        elif cur_pos[1] == self.cake_len and cur_pos[0] != 0:
            val = self.cake_len - 0.01
            while [cur_pos[0], cur_pos[1], 0, val] in self.cuts or [0, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [0, val]

    # Move the knife to the side in the anticlockwise direction
    def move_knife_anticlockwise(self, cur_pos):
        if cur_pos[0] == 0 and cur_pos[1] != self.cake_len:
            val = 0.01
            while [cur_pos[0], cur_pos[1], val, self.cake_len] in self.cuts or [val, self.cake_len, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [val, self.cake_len]
        elif cur_pos[1] == 0 and cur_pos[0] != 0:
            val = 0.01
            while [cur_pos[0], cur_pos[1], 0, val] in self.cuts or [0, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val += 0.01
            return [0, val]
        elif cur_pos[0] == self.cake_width and cur_pos[1] != 0:
            val = self.cake_width - 0.01
            while [cur_pos[0], cur_pos[1], val, 0] in self.cuts or [val, 0, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [val, 0]
        elif cur_pos[1] == self.cake_len and cur_pos[0] != self.cake_width:
            val = self.cake_len - 0.01
            while [cur_pos[0], cur_pos[1], self.cake_width, val] in self.cuts or [self.cake_width, val, cur_pos[0], cur_pos[1]] in self.cuts:
                val -= 0.01
            return [self.cake_width, val]
    
    def move_straight(self,cur_pos, dir='R'):
        switch ={
            'L': (0,round(cur_pos[1], 2)),
            'R': (round(self.cake_width, 2), round(cur_pos[1], 2)),
            'U': (round(cur_pos[0],2), 0),
            'D': (round(cur_pos[0], 2), round(self.cake_len), 2)
        }
        return switch.get(dir)
    
    def move_angle(self, cur_pos, deg):
        deg = deg % 360
        theta_radians = np.radians(deg)

        dx = np.cos(theta_radians)
        dy = np.sin(theta_radians)

        intersection = None
        intersection = self.check_intersections(dx, dy, cur_pos)

        if intersection is None:
            # Take 180-angle i.e. line other side
            dx = np.cos(theta_radians + np.pi)
            dy = np.sin(theta_radians + np.pi)
            intersection = self.check_intersections(dx, dy, cur_pos)

        return intersection

    def check_intersections(self, dx, dy, cur_pos):
        L, W = self.cake_width, self.cake_len
        x0, y0 = cur_pos
        intersection = None

        # left side (x = 0)
        if dx != 0:
            tl = -x0 / dx
            if tl >= 0:
                y = round(y0 + tl * dy,2)
                if 0 <= y <= W and (0, y) != cur_pos:
                    intersection = (0, y)

        #  right side (x = L)
        if dx != 0:
            tr = (L - x0) / dx
            if tr >= 0:
                y = round(y0 + tr * dy,2)
                if 0 <= y <= W and (L, y) != cur_pos:
                    if intersection is None or tr < tl:
                        intersection = (L, y)

        # down side (y = 0)
        if dy != 0:
            td = -y0 / dy
            if td >= 0:
                x = round(x0 + td * dx,2)
                if 0 <= x <= L and (x, 0) != cur_pos:
                    if intersection is None or td < min(tl, tr):
                        intersection = (x, 0)

        #  up side (y = W)
        if dy != 0:
            tu = (W - y0) / dy
            if tu >= 0:
                x = round(x0 + tu * dx,2)
                if 0 <= x <= L and (x, W) != cur_pos:
                    if intersection is None or tu < min(tl, tr):
                        intersection = (x, W)

        return intersection

    def get_max_diff_average(self, areas, excess = 100):
        # Find the average of the two pieces with the maximum difference
        max_diff = 0
        max_diff_index = 1
        for i in range(1, len(areas)):
            diff = areas[i] - areas[i - 1]
            if (areas[i-1] + areas[i])/2 > excess:
                return (areas[max_diff_index] + areas[max_diff_index - 1]) / 2
            if diff > max_diff:
                max_diff = diff
                max_diff_index = i

        return (areas[max_diff_index] + areas[max_diff_index - 1]) / 2

    def get_greater_than(self, areas, val):
        # Find the average of the two pieces with the maximum difference
        for i in range(1, len(areas)):
            diff = areas[i] - areas[i - 1]
            if diff >= val:
                return areas[i-1] + val/2

        return -1

    def make_cuts(self):
        # print(self.cake_len, self.cake_width)
        area = self.cake_len * self.cake_width
        areas = sorted(self.requests)
        excess = 4.76*area/100

        if area < 945:
            vertical_stack = 1
            areas += [4*area/100]
            areas = sorted(areas)
        # print(f"Area: {area}")
        elif area < 4000:
            vertical_stack = 2
        elif area < 6500:
            vertical_stack = 3
        else:
            vertical_stack = 4

        if len(areas)%vertical_stack != 0:
            required = vertical_stack - len(areas)%vertical_stack
            while required > 0:
                new_slice = min(self.get_max_diff_average(areas, excess), excess)
                # print("added slice", new_slice)
                areas += [new_slice]
                areas = sorted(areas)
                excess -= new_slice
                required -= 1

        # print("vertical_stack", vertical_stack)
        # print("excess", excess)
        if vertical_stack > 1:
            # Add slices to reduce gap between consecutive slices by finding max diff average
            while excess > 0:
                new_areas = areas.copy()
                sum_new_slices = 0
                for i in range(vertical_stack):
                    new_slice = self.get_max_diff_average(new_areas, excess)
                    new_areas += [new_slice]
                    new_areas = sorted(new_areas)
                    sum_new_slices += new_slice
                if sum_new_slices < excess:
                    # print("Old areas", areas)
                    # print("New areas", new_areas)
                    areas = new_areas
                    excess -= sum_new_slices
                else:
                    break

            # Add slices to reduce gap between consecutive slices by finding smallest gap of over 10
            diff = 60
            while excess > 0 :
                new_areas = areas.copy()
                sum_new_slices = 0
                for i in range(vertical_stack):
                    new_slice = self.get_greater_than(new_areas, diff)
                    if new_slice == -1:
                        sum_new_slices = excess
                        break
                    new_areas += [new_slice]
                    new_areas = sorted(new_areas)
                    sum_new_slices += new_slice
                    if excess < 0:
                        break
                if sum_new_slices < excess:
                    # print("Old areas", areas)
                    # print("New areas", new_areas)
                    areas = new_areas
                    excess -= sum_new_slices
                elif diff > 4:
                    diff -= 4
                else:
                    break

        groups = {i: [] for i in range(vertical_stack)}
        
        i=0
        k=0

        while i<len(areas):
            groups[k].append(areas[i])
            i+=1
            if k+1 == vertical_stack:
                k=0
            else:
                k+=1

        # print("Groups", groups)

        # Increase the size of last piece of each stack to accommodate for crumbs
        if vertical_stack > 1:
            for i in range(vertical_stack):
                groups[i][-1] = 1.01*groups[i][-1]

        lengths = [round(sum(groups[group]) / self.cake_width, 2) for group in groups]
        cum_lengths = [round(s, 2) for s in itertools.accumulate(lengths)]
        if vertical_stack > 1:
            for i,l in enumerate(cum_lengths):
                # Place knife
                if i==0:
                    self.cutList.append([0, l])
                    continue
                cur = self.cutList[-1]
                breadcrumb_goto =  round(0 if l < self.cake_len//2 else self.cake_len, 2)

                #Right to left
                if i%2 == 0:
                    self.cutList.append(self.move_straight(cur, 'L'))
                    self.cutList.append([0.02, breadcrumb_goto])
                    self.cutList.append([0, l])
                #Left to right
                else:
                    self.cutList.append(self.move_straight(cur, 'R'))
                    self.cutList.append([round(self.cake_width-0.02, 2), breadcrumb_goto])
                    self.cutList.append([self.cake_width, l])

            cur = self.cutList[-1]
            if vertical_stack % 2 == 0:
                self.cutList.append(self.move_straight(cur, 'L'))
            else:
                self.cutList.append(self.move_straight(cur, 'R'))
        else:
            self.cutList.append([0, round(self.cake_len - 0.01, 2)])
        # For vertical cuts
        l1 = [round(small / lengths[0], 2) for i, small in enumerate(groups[0])]
        l2 = [round(big / lengths[vertical_stack - 1], 2) for i, big in enumerate(groups[vertical_stack - 1])]
        cum_l1 = [round(s, 2) for s in itertools.accumulate(l1)]
        cum_l2 = [round(s, 2) for s in itertools.accumulate(l2)]
        # print(cum_l1, cum_l2, l1, l2)

        if self.cutList[-1][0] == self.cake_width:
            self.cutList.append([round(self.cake_width - 0.03, 2), self.cake_len])
            self.cutList.append([self.cake_width, round(self.cake_len - 0.03, 2)])
        else:
            self.cutList.append([0.03, self.cake_len])
            self.cutList.append([0, round(self.cake_len - 0.03, 2)])

        # print("Vertical Cuts")
        # print(cum_l1, cum_l2)

        for i,(l1,l2) in enumerate(zip(cum_l1,cum_l2)):
            # Down to up from l2 to l1
            if i%2 == 0:
                self.cutList.append([l2, self.cake_len])
                self.cutList.append([l1, 0])
            # Up to down from l1 to l2
            else:
                self.cutList.append([l1, 0])
                self.cutList.append([l2, self.cake_len])

            if i < len(cum_l1)-1:
                cur = self.cutList[-1]
                breadcrumb_goto =  0 if cur[0]<self.cake_width//2 else self.cake_width
                if cur[1] == 0:
                    self.cutList.append([breadcrumb_goto, 0.03])
                else:
                    self.cutList.append([breadcrumb_goto, round(self.cake_len-0.03,2)])
        

    def move(self, current_percept) -> (int, List[int]):
        """Function which retrieves the current state of the cake

            Args:
                current_percept(PieceOfCakeState): contains current state information
            Returns:
                (int, List[int]): This function returns the next move of the user:
                The integer return value should be one of the following:
                    constants.INIT - If wants to initialize the knife position
                    constants.CUT - If wants to cut the cake
                    constants.ASSIGN - If wants to assign the pieces
        """
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        self.cake_len = current_percept.cake_len
        self.cake_width = current_percept.cake_width
        self.requests = requests

        try:
            if turn_number == 1:
                self.make_cuts()
                init = self.cutList.pop(0)
                return constants.INIT, [init[0],init[1]]

            while self.cutList:
                next_move = self.cutList.pop(0)
                return constants.CUT, [next_move[0], next_move[1]]

            if len(polygons) < len(requests):
                if cur_pos[0] == 0:
                    return constants.CUT, [self.cake_width, round((cur_pos[1] + 5)%self.cake_len, 2)]
                else:
                    return constants.CUT, [0, round((cur_pos[1] + 5)%self.cake_len, 2)]

            # Assign the pieces
            areas =[round(i.area,2) for i in polygons]            
            c = np.array([
                [abs(request - area) / request * 100 if abs(request - area) / request * 100 > self.tolerance else 0
                for area in areas] for request in self.requests
            ])
                    
            _, assignment = linear_sum_assignment(c)
            # print(assignment.tolist()[:len(requests)])
            # for i in range(len(requests)):
            #     print(f"Request: {requests[i]} Assigned: {areas[assignment[i]]} Percent Error: {abs(requests[i] - areas[assignment[i]]) / requests[i] * 100 if abs(requests[i] - areas[assignment[i]]) / requests[i] * 100 > self.tolerance else 0}")

            # print("Unassigned")
            # for i in range(len(polygons)):
            #     if i not in assignment and areas[i] > 8:
            #         print(areas[i])
            return constants.ASSIGN, assignment.tolist()[:len(requests)]
        
        except Exception as e:
            print(e)
            traceback.print_exc()

