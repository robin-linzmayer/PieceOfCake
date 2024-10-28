from typing import List

import numpy as np
import logging

import constants


class G2_Player:
    def __init__(
        self,
        rng: np.random.Generator,
        logger: logging.Logger,
        precomp_dir: str,
        tolerance: int,
    ) -> None:
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
        self.move_queue = []
        self.move_queue = []

        self.phase = 'HORIZONTAL'
        self.direction = ''

    def cut(self, cake_len, cake_width, cur_pos) -> (int, List[int]):
        if cur_pos[0] == 0:
            return constants.CUT, [cake_width, round((cur_pos[1] + 5)%cake_len, 2)]
        else:
            return constants.CUT, [0, round((cur_pos[1] + 5)%cake_len, 2)]
        
    def assign(self, polygons, requests) -> (int, List[int]):
        print("Polygons: ", polygons)
        print("Requests: ", requests)
        assignment = []
        for i in range(len(requests)):
            assignment.append(i)

        return constants.ASSIGN, assignment
    
    def sneak(self, start_pos, end_pos, cake_width, cake_len):
        '''
        Given a start position & goal position, uses the 1-pixel shaving technique
        to append the necessary steps to the move_queue
        '''
        nearest_x, x_dist = nearest_edge_x(start_pos, cake_width)
        nearest_y, y_dist = nearest_edge_y(start_pos, cake_len)

        end_x, end_x_dist = nearest_edge_x(end_pos, cake_width)
        end_y, end_y_dist = nearest_edge_y(end_pos, cake_len)

        # if we are on the top or bottom or the board and require more than 1 move
        if y_dist == 0 and (x_dist > 0.1 or nearest_x != end_x):
            bounce_y = bounce(nearest_y)
            self.move_queue.append([end_x, bounce_y])

            # if the end is not on the same line so we must ricochet off of corner
            if end_y_dist > 0 or nearest_y != end_y:
                bounce_x = bounce(end_x)
                self.move_queue.append([bounce_x, nearest_y])

                # if the end position is on the opposite side
                if end_y_dist == 0:
                    bounce_y = bounce(end_y)
                    self.move_queue.append([bounce_x, end_y])
                    self.move_queue.append([end_x, bounce_y])

        # if we are on the left or right side of the board and require more than 1 move
        elif x_dist == 0 and (y_dist > 0.1 or nearest_y != end_y):
            bounce_x = bounce(nearest_x)
            self.move_queue.append([bounce_x, end_y])

            # if the end is not on the same line so we must ricochet off of corner
            if end_x_dist > 0 or nearest_x != end_x:
                bounce_y = bounce(end_y)
                self.move_queue.append([nearest_x, bounce_y])

                # if the end position is on the opposite side
                if end_x_dist == 0:
                    bounce_x = bounce(end_x)
                    self.move_queue.append([end_x, bounce_y])
                    self.move_queue.append([bounce_x, end_y])
            
        self.move_queue.append(end_pos)
        return
    
    def even_cuts(self, current_percept):
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        n = len(requests)
        s = round(np.sqrt(cake_len * cake_width / n), 2)
        pos = cur_pos

        if turn_number == 2:
            self.move_queue.append([0,s])
            self.move_queue.append([cake_width,s])
            return
        
        if self.phase == 'HORIZONTAL' and pos[1]+s >= cake_len:
            self.phase = 'VERTICAL'
            if pos[0] == 0: new_x = s
            else:
                new_x = cake_width-s
                self.direction = 'RIGHT'
            self.sneak(pos, [new_x, cake_len], cake_width, cake_len)
            self.move_queue.append([new_x, 0])

            return

        if self.phase == 'HORIZONTAL':
            self.sneak(pos, [pos[0], pos[1]+s], cake_width, cake_len)
            if pos[0]==0: opposite = cake_width
            else: opposite = 0
            self.move_queue.append([opposite, round(pos[1]+s, 2)])

        else:
            if self.direction == 'RIGHT': new_x = pos[0]-s
            else: new_x = pos[0]+s

            if new_x <= 0 or new_x >= cake_width:
                self.phase = 'DONE'
                return

            self.sneak(pos, [new_x, pos[1]], cake_width, cake_len)
            if pos[1]==0: opposite = cake_len
            else: opposite = 0
            self.move_queue.append([new_x, opposite])

        return

    def move(self, current_percept) -> (int, List[int]):
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
        polygons = current_percept.polygons
        turn_number = current_percept.turn_number
        cur_pos = current_percept.cur_pos
        requests = current_percept.requests
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        
        if turn_number == 1:
            return constants.INIT, [0.01, 0]
        
        if len(self.move_queue) == 0 and self.phase != 'DONE':
            self.even_cuts(current_percept)
        
        if len(self.move_queue) > 0:
            next_val = self.move_queue.pop(0)
            cut = [round(next_val[0], 2), round(next_val[1], 2)]
            return constants.CUT, cut

        if len(polygons) < len(requests):
            return self.cut(cake_len, cake_width, cur_pos)
        else:
            return self.assign(polygons, requests)
        
def nearest_edge_x(pos, cake_width):
    '''
    Returns the nearest X-edge and the distance to said edge
    X-edge is 0 if the position is closer to the left side, or cake_width
        if it is closer to the right side.
    '''
    min_x = pos[0]
    x_edge = 0
    if cake_width - pos[0] < min_x:
        min_x = cake_width - pos[0]
        x_edge = cake_width
    return x_edge, min_x
    
def nearest_edge_y(pos, cake_len):
    '''
    Returns the nearest Y-edge and the distance to said edge
    Y-edge is 0 if the position is closer to the top, or cake_len
        if it is closer to the bottom.
    '''
    min_y = pos[1]
    y_edge = 0
    if cake_len - pos[1] < min_y:
        min_y = cake_len - pos[1]
        y_edge = cake_len
    return y_edge, min_y

def bounce(margin):
    '''
    Returns a value 0.01 away from the provided margin
    '''
    if margin == 0:
        return 0.01
    return round(margin-0.01, 2)
