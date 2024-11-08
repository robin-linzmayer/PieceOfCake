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
            self.grid_optimizer.run_optimization(max_evals=1500)
            
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
        v_x_positions, h_y_positions = compute_cuts(
            self.requests, self.cake_len, self.cake_width, self.tolerance
        )
        
        self.num_col_divisions = len(v_x_positions) + 1
        self.num_vertical_cuts = len(v_x_positions)
        self.num_row_divisions = len(h_y_positions) + 1
        self.num_horizontal_cuts = len(h_y_positions)

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

def get_best_grid_cuts(
        requests, cake_len, cake_width, min_horz_cuts, max_horz_cuts, tolerance, altered_piece
):
    requests_sorted = sorted(requests, reverse=True)

    tolerance = max(1, tolerance - 1)
    t = tolerance / 100

    requests_factors = divisors(len(requests))
    l_i = bisect.bisect_left(requests_factors, min_horz_cuts)
    r_i = bisect.bisect_right(requests_factors, max_horz_cuts)
    possible_horz_cuts = requests_factors[l_i:r_i]

    min_total_penalty = float("inf")
    best_num_horz_cuts = None
    best_vert_cut_diffs = None

    for num_horz_cuts in possible_horz_cuts:
        num_vert_cuts = len(requests) // num_horz_cuts
        horz_cut_diff = cake_len / num_horz_cuts

        h = horz_cut_diff

        prob = pulp.LpProblem("Minimize_Penalty", pulp.LpMinimize)

        x_int = pulp.LpVariable.dicts(
            "x_int", (i for i in range(num_vert_cuts)), lowBound=1, cat="Integer"
        )
        x = {i: 0.01 * x_int[i] for i in range(num_vert_cuts)}

        s_j = pulp.LpVariable.dicts(
            "s_j", (j for j in range(len(requests))), lowBound=0, cat="Continuous"
        )

        y_j = pulp.LpVariable.dicts(
            "y_j", (j for j in range(len(requests))), cat="Binary"
        )

        delta_j = pulp.LpVariable.dicts(
            "delta_j", (j for j in range(len(requests))), lowBound=0, cat="Continuous"
        )

        alt_index = requests.index(altered_piece) if altered_piece is not None else -1
        prob += pulp.lpSum([s_j[j] for j in range(len(requests)) if j != alt_index])

        prob += pulp.lpSum([x[i] for i in range(num_vert_cuts)]) <= cake_width

        M = 1e6

        request_idx = 0
        for i in range(num_vert_cuts):
            curr_requests = requests_sorted[i * num_horz_cuts: (i + 1) * num_horz_cuts]
            A_i = h * x[i]

            for r in curr_requests:
                r_value = r

                prob += A_i >= (1 - t) * r_value - M * y_j[request_idx]
                prob += A_i <= (1 + t) * r_value + M * y_j[request_idx]

                prob += delta_j[request_idx] >= A_i - r_value - M * (
                        1 - y_j[request_idx]
                )
                prob += delta_j[request_idx] >= r_value - A_i - M * (
                        1 - y_j[request_idx]
                )
                prob += delta_j[request_idx] >= 0

                beta_j = 100 / r_value
                prob += s_j[request_idx] >= delta_j[request_idx] * beta_j
                prob += s_j[request_idx] <= M * y_j[request_idx]

                request_idx += 1

        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if prob.status != pulp.LpStatusOptimal:
            continue

        vert_cut_diffs = [pulp.value(x[i]) for i in range(num_vert_cuts)]
        total_penalty = sum(pulp.value(s_j[j]) for j in range(len(requests)))

        if total_penalty <= min_total_penalty:
            min_total_penalty = total_penalty
            best_num_horz_cuts = num_horz_cuts
            best_vert_cut_diffs = vert_cut_diffs.copy()

    if best_num_horz_cuts is None:
        return None, None, None

    accum_vert_cuts = list(accumulate(best_vert_cut_diffs))
    accum_vert_cuts = [round(diff, 2) for diff in accum_vert_cuts]
    return best_num_horz_cuts, accum_vert_cuts, min_total_penalty

def test_best_grid_cuts(requests, cake_len, cake_width, tolerance):
    min_horz_cuts = math.ceil(cake_len / 24.6)
    max_horz_cuts = math.floor(math.sqrt(len(requests)))

    altered_piece = None
    best_num_horz_cuts, best_x_coords, min_total_penalty = get_best_grid_cuts(
        requests, cake_len, cake_width, min_horz_cuts, max_horz_cuts, tolerance, altered_piece
    )

    for r in set(requests):
        altered_piece = r
        altered_requests = requests + [altered_piece]
        num_horz_cuts, x_coords, curr_penalty = get_best_grid_cuts(
            altered_requests, cake_len, cake_width, min_horz_cuts, max_horz_cuts, tolerance, altered_piece
        )

        if curr_penalty is not None and (min_total_penalty is None or curr_penalty <= min_total_penalty):
            min_total_penalty = curr_penalty
            best_x_coords = x_coords
            best_num_horz_cuts = num_horz_cuts

    return best_num_horz_cuts, best_x_coords, min_total_penalty

def compute_cuts(requests, cake_len, cake_width, tolerance):
    num_horz_cuts, x_coords, total_penalty = test_best_grid_cuts(
        requests,
        cake_len,
        cake_width,
        tolerance,
    )
    
    y_coords = []
    delta = cake_len / num_horz_cuts
    for i in range(1, num_horz_cuts):
        y_coords.append(delta * i)
        
    return x_coords, y_coords