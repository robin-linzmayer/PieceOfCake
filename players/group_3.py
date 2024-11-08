import bisect
import logging
import math
import os
import pickle
from collections import deque
from enum import Enum
from itertools import accumulate
from typing import List, Tuple

import numpy as np
import pulp
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Polygon
from shapely.ops import split
from sympy import divisors
import miniball
import matplotlib.pyplot as plt
import optuna

import constants

optuna.logging.set_verbosity(optuna.logging.ERROR)

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
        
        self.triangle_viable = 23.507
        self.preplanned_moves = deque()
        self.request_served = 0
        
        self.current_percept = None

    def move(self, current_percept) -> Tuple[int, List[int]]:
        self.current_percept = current_percept
        if current_percept.cake_len <= self.triangle_viable:
            return self.triangle()
        return self.quadrangle()

    def quadrangle(self):
        polygons = self.current_percept.polygons
        turn_number = self.current_percept.turn_number
        cur_pos = self.current_percept.cur_pos
        requests = self.current_percept.requests
        cake_len = self.current_percept.cake_len
        cake_width = self.current_percept.cake_width

        if turn_number == 1:
            self.grid_optimizer = GridOptimizer(
                cake_width=cake_width,
                cake_len=cake_len,
                requests=requests,
                tolerance=self.tolerance,
            )
            self.grid_optimizer.run_optimization(max_evals=2000)
            
            return constants.INIT, [0, 0.01]
        
        if turn_number == 2:
            cur_x, cur_y = cur_pos[0], cur_pos[1]
            horizontal_cuts = self.grid_optimizer.horizontal_cuts
            vertical_cuts = self.grid_optimizer.vertical_cuts
            
            for cut in horizontal_cuts:
                if cur_x == 0:
                    self.shift_along((cur_x, cur_y), cut[0])
                    cur_x, cur_y = cut[1][0], cut[1][1]
                    self.preplanned_moves.append([cur_x, cur_y])
                else:
                    self.shift_along((cur_x, cur_y), cut[1])
                    cur_x, cur_y = cut[0][0], cut[0][1]
                    self.preplanned_moves.append([cur_x, cur_y])
            
            if cur_x == cake_width:
                self.preplanned_moves.append([cake_width - 0.01, cake_len])
                self.preplanned_moves.append([0, cake_len - 0.01])
            self.preplanned_moves.append([0.01, cake_len])
            cur_x, cur_y = 0.01, cake_len
            
            for cut in vertical_cuts:
                if cur_y == cake_len:
                    self.shift_along((cur_x, cur_y), cut[1])
                    cur_x, cur_y = cut[0][0], cut[0][1]
                    self.preplanned_moves.append([cur_x, cur_y])
                else:
                    self.shift_along((cur_x, cur_y), cut[0])
                    cur_x, cur_y = cut[1][0], cut[1][1]
                    self.preplanned_moves.append([cur_x, cur_y])

        if self.preplanned_moves:
            dest_x, dest_y = self.preplanned_moves.popleft()
            return constants.CUT, [round(dest_x, 2), round(dest_y, 2)]
        
        _, assignment = assign_polygons_to_requests(polygons, requests, self.tolerance) 
        return constants.ASSIGN, assignment

    def shift_along(self, cur_pos, target_pos):
        cake_len = self.current_percept.cake_len
        cake_width = self.current_percept.cake_width
        
        if cur_pos[0] == 0:
            if cur_pos[1] < cake_len / 2:
                self.preplanned_moves.append([0.01, 0])
            else:
                self.preplanned_moves.append([0.01, cake_len])
        elif cur_pos[0] == cake_width:
            if cur_pos[1] < cake_len / 2:
                self.preplanned_moves.append([cake_width - 0.01, 0])
            else:
                self.preplanned_moves.append([cake_width - 0.01, cake_len])
        elif cur_pos[1] == 0:
            if cur_pos[0] < cake_width / 2:
                self.preplanned_moves.append([0, 0.01])
            else:
                self.preplanned_moves.append([cake_width, 0.01])
        else:
            if cur_pos[0] < cake_width / 2:
                self.preplanned_moves.append([0, cake_len - 0.01])
            else:
                self.preplanned_moves.append([cake_width, cake_len - 0.01])
        self.preplanned_moves.append([target_pos[0], target_pos[1]])

    def triangle(self):
        turn_number = self.current_percept.turn_number
        cur_pos = self.current_percept.cur_pos
        requests = sorted(self.current_percept.requests)
        cake_len = self.current_percept.cake_len
        cake_width = self.current_percept.cake_width

        if turn_number == 1:
            self.preplanned_moves.append([0,0])
            return constants.INIT, [0,0]
        
        if self.request_served < len(requests):
            area = requests[self.request_served]
            base = round(2 * area / cake_len, 2)

            cur_x, cur_y = cur_pos[0], cur_pos[1]

            if turn_number == 2:
                if cake_len ** 2 + base ** 2 > 25 ** 2:
                    raise Exception("First cut doesn't fit on plate.")
                dest_x, dest_y = base, cake_len
            else:
                dest_x = round(self.preplanned_moves[-2][0] + base, 2)
                dest_y = cake_len if cur_y == 0 else 0

                if dest_x > cake_width:
                    l1 = dest_x - cake_width
                    l2 = cake_width - self.preplanned_moves[-1][0]
                    h1 = (cake_len * (l1)) / l2
                    h2 = cake_len - h1
                    h3 = (h1 * (l1)) / l2
                    new_y = round(h2 - h3, 2)
                    dest_y = new_y if cur_y == 0 else round(cake_len - new_y, 2)
                    dest_x = cake_width

            self.preplanned_moves.append([dest_x, dest_y])
            self.request_served += 1
            return constants.CUT, [dest_x, dest_y]
        
        _, assignment = assign_polygons_to_requests(self.current_percept.polygons, self.current_percept.requests, self.tolerance) 
        return constants.ASSIGN, assignment

class GridOptimizer:
    def __init__(self, cake_width, cake_len, requests, tolerance=5):
        self.cake_width = cake_width
        self.cake_len = cake_len
        self.requests = requests
        self.tolerance = tolerance

        self.horizontal_cuts = []
        self.vertical_cuts = []

        self.polygons = []

    def generate_polygons(self):
        cake_polygon = Polygon([
            (0, 0),
            (self.cake_width, 0),
            (self.cake_width, self.cake_len),
            (0, self.cake_len)
        ])

        polygons = [cake_polygon]

        for cut in self.vertical_cuts:
            line = LineString(cut)
            new_polygons = []
            for poly in polygons:
                new_pieces = self.divide_polygon(poly, line)
                new_polygons.extend(new_pieces)
            polygons = new_polygons

        for cut in self.horizontal_cuts:
            line = LineString(cut)
            new_polygons = []
            for poly in polygons:
                new_pieces = self.divide_polygon(poly, line)
                new_polygons.extend(new_pieces)
            polygons = new_polygons

        self.polygons = polygons
        return polygons

    def divide_polygon(self, polygon, line):
        if not line.intersects(polygon):
            return [polygon]

        result = split(polygon, line)
        return list(result.geoms)

    def cost_function(self):
        polygons = self.generate_polygons()
        total_penalty, _ = assign_polygons_to_requests(
            polygons=polygons,
            requests=self.requests,
            tolerance=self.tolerance
        )
        if len(polygons) > self.num_col_divisions * self.num_row_divisions:
            total_penalty += 100000
        return total_penalty
    
    def get_best_grid_cuts(self):
        num_requests = len(self.requests)
        min_penalty = float("inf")
        best_horizontal_cuts = []
        best_vertical_cuts = []
        
        for rows in range(1, 11):
            cols = (num_requests + rows - 1) // rows
            
            self.num_row_divisions = rows
            self.num_col_divisions = cols

            row_height = self.cake_len / rows
            col_width = self.cake_width / cols

            horizontal_cuts = [row_height * i for i in range(1, rows)]
            vertical_cuts = [col_width * i for i in range(1, cols)]

            self.horizontal_cuts = [
                [(0, y), (self.cake_width, y)] for y in horizontal_cuts
            ]
            self.vertical_cuts = [
                [(x, 0), (x, self.cake_len)] for x in vertical_cuts
            ]

            penalty = self.cost_function()

            if penalty < min_penalty:
                min_penalty = penalty
                best_horizontal_cuts = horizontal_cuts
                best_vertical_cuts = vertical_cuts
                best_rows = rows
                best_cols = cols

        self.num_row_divisions = best_rows
        self.num_horizontal_cuts = best_rows - 1
        self.num_col_divisions = best_cols
        self.num_vertical_cuts = best_cols - 1
        
        self.horizontal_cuts = [
            [(0, y), (self.cake_width, y)] for y in best_horizontal_cuts
        ]
        self.vertical_cuts = [
            [(x, 0), (x, self.cake_len)] for x in best_vertical_cuts
        ]
        
        return best_vertical_cuts, best_horizontal_cuts, min_penalty

    def objective(self, params):
        for key in params:
            if 'h_y' in key:
                params[key] = max(0, min(self.cake_len, params[key]))
            elif 'v_x' in key:
                params[key] = max(0, min(self.cake_width, params[key]))

        self.horizontal_cuts = [
            [(0, params[f'h_y1_{i}']), (self.cake_width, params[f'h_y2_{i}'])]
            for i in range(self.num_horizontal_cuts)
        ]

        self.vertical_cuts = [
            [(params[f'v_x1_{i}'], 0), (params[f'v_x2_{i}'], self.cake_len)]
            for i in range(self.num_vertical_cuts)
        ]

        return self.cost_function()
    
    def run_optimization(self, max_evals=1000):
        """
        Run the optimization using Optuna with an initial trial based on precomputed cuts.

        Updates the cut positions to minimize the total penalty.
        """
        v_x_positions, h_y_positions, _ = self.get_best_grid_cuts()

        temp_x = [0] + v_x_positions.copy() + [self.cake_width]
        temp_y = [0] + h_y_positions.copy() + [self.cake_len]
        
        x_ranges = []
        y_ranges = []
        
        for i in range(1, len(temp_x) - 1):
            x_ranges.append([(temp_x[i] - temp_x[i - 1]) / 2, (temp_x[i + 1] - temp_x[i]) / 2])

        for i in range(1, len(temp_y) - 1):
            y_ranges.append([(temp_y[i] - temp_y[i - 1]) / 2, (temp_y[i + 1] - temp_y[i]) / 2])

        def objective(trial):
            params = {
                **{f'h_y1_{i}': trial.suggest_float(
                    f'h_y1_{i}', 
                    max(0, y - y_ranges[i][0]), 
                    min(self.cake_len, y + y_ranges[i][1])
                ) for i, y in enumerate(h_y_positions)},
                
                **{f'h_y2_{i}': trial.suggest_float(
                    f'h_y2_{i}', 
                    max(0, y - y_ranges[i][0]), 
                    min(self.cake_len, y + y_ranges[i][1])
                ) for i, y in enumerate(h_y_positions)},
                
                **{f'v_x1_{i}': trial.suggest_float(
                    f'v_x1_{i}', 
                    max(0, x - x_ranges[i][0]), 
                    min(self.cake_width, x + x_ranges[i][1])
                ) for i, x in enumerate(v_x_positions)},
                
                **{f'v_x2_{i}': trial.suggest_float(
                    f'v_x2_{i}', 
                    max(0, x - x_ranges[i][0]), 
                    min(self.cake_width, x + x_ranges[i][1])
                ) for i, x in enumerate(v_x_positions)}
            }

            return self.objective(params)

        study = optuna.create_study(direction="minimize")
        
        initial_trial = {
            **{f'h_y1_{i}': y for i, y in enumerate(h_y_positions)},
            **{f'h_y2_{i}': y for i, y in enumerate(h_y_positions)},
            **{f'v_x1_{i}': x for i, x in enumerate(v_x_positions)},
            **{f'v_x2_{i}': x for i, x in enumerate(v_x_positions)}
        }
        
        study.enqueue_trial(initial_trial)
        study.optimize(objective, n_trials=max_evals, timeout=1800)
        best_params = study.best_params

        self.horizontal_cuts = [
            [(0, best_params[f'h_y1_{i}']), (self.cake_width, best_params[f'h_y2_{i}'])]
            for i in range(self.num_horizontal_cuts)
        ]

        self.vertical_cuts = [
            [(best_params[f'v_x1_{i}'], 0), (best_params[f'v_x2_{i}'], self.cake_len)]
            for i in range(self.num_vertical_cuts)
        ]
            
    def display_polygons(self, polygons):
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        
        for poly in polygons:
            if isinstance(poly, Polygon):
                x, y = poly.exterior.xy
                ax.fill(x, y, alpha=0.5, edgecolor='black')
                
                area = poly.area
                
                centroid = poly.centroid
                ax.text(centroid.x, centroid.y, f"{area:.2f}", ha='center', va='center', fontsize=8, color='blue')
        
        ax.set_title("Polygon Grid Visualization")
        plt.xlabel("Width")
        plt.ylabel("Length")
        plt.show()

def assign_polygons_to_requests(polygons, requests, tolerance):
    num_requests = len(requests)
    num_polygons = len(polygons)

    cost_matrix = np.full((num_requests, num_polygons), fill_value=1e6)

    for i, request_area in enumerate(requests):
        for j, polygon in enumerate(polygons):
            piece_area = polygon.area

            size_difference = abs(piece_area - request_area)
            percentage_difference = (size_difference / request_area) * 100
            
            if percentage_difference <= tolerance:
                area_penalty = 0
            else:
                area_penalty = percentage_difference
                
            fits_on_plate = can_cake_fit_in_plate(polygon)
            plate_penalty = 0 if fits_on_plate else 1000

            total_penalty = area_penalty + plate_penalty

            cost_matrix[i, j] = total_penalty

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_penalty = cost_matrix[row_ind, col_ind].sum()

    assignment = [-1] * num_requests
    assigned_polygons = set()

    for req_idx, poly_idx in zip(row_ind, col_ind):
        penalty = cost_matrix[req_idx, poly_idx]
        if penalty < 1e5:
            assignment[req_idx] = int(poly_idx)
            assigned_polygons.add(poly_idx)
        else:
            pass

    return total_penalty, assignment
    
def can_cake_fit_in_plate(cake_piece, radius=12.5):
    cake_points = np.array(list(zip(*cake_piece.exterior.coords.xy)), dtype=np.double)
    res = miniball.miniball(cake_points)
    return res["radius"] <= radius
