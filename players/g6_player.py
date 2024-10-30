import os
import pickle
from typing import List

import numpy as np
import logging
import traceback
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
            'L': (0,cur_pos[1]),
            'R': (self.cake_width, cur_pos[1]),
            'U': (cur_pos[0], 0),
            'D': (cur_pos[0], self.cake_len)
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

    # def make_cuts(self):
    #     print(self.cake_len, self.cake_width)
    #     n = len(self.requests)
    #     angles = [i * (360 / n) for i in range(n)]
    #     start = (round(self.cake_width/2,2),0)
    #     print(angles)
    #     self.cutList.append(start)
    #     for i in range(n):
    #         start = self.move_angle(start, angles[i])
    #         self.cutList.append(start)
    #         print(start)

    def make_cuts(self):
        self.requests = sorted(self.requests)
        print(self.cake_len, self.cake_width)

        w= 4
        l=[i/w for i in self.requests]
        self.cutList.append([w,0])
        for i in range(len(self.requests)//2 +1):
            v= (i+2)*w
            cur = self.cutList[-1]
            goto = 0 if v<self.cake_width//2 else self.cake_width
            if v-w> self.cake_width:
                break
            if i%2 == 0:
                self.cutList.append(self.move_straight(cur, 'D'))
                self.cutList.append([goto,self.cake_len-0.02])
                self.cutList.append([v,self.cake_len])
            else:
                self.cutList.append(self.move_straight(cur, 'U'))
                self.cutList.append([goto,0.02])
                self.cutList.append([v,0])
        self.cutList.append([self.cake_width-0.02, 0])
        self.cutList.append([self.cake_width, l[0]])
        self.cutList.append([0, l[-1]])
        #self.cutList.append([self.cake_width, l[1]])
        
        

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
            areas =[i.area for i in polygons]
            assignment = sorted(range(len(areas)), key=lambda x: areas[x], reverse=True)
            print(assignment[:len(requests)])
            return constants.ASSIGN, assignment[:len(requests)][::-1]
        
        except Exception as e:
            print(e)
            traceback.print_exc()

