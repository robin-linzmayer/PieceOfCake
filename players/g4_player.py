import os
import pickle
from typing import List

import numpy as np
import logging
from scipy.optimize import linear_sum_assignment

import constants
from shapely.geometry import Polygon, LineString
from shapely.ops import split
import copy
from tqdm import tqdm


class grid_cut_strategy:
    def __init__(self, width, height, requests):
        self.width = width
        self.height = height
        self.requests = requests
        self.factors = self.factor_pairs(len(requests))

    def factor_pairs(x):
        min_pairs = 5

        def get_factor_pairs(n):
            pairs = []
            limit = int(abs(n) ** 0.5) + 1
            for i in range(1, limit):
                if n % i == 0 and i != 1:
                    pairs.append((i, n // i))
            return pairs

        pairs = get_factor_pairs(x)

        offset = 1
        while len(pairs) < min_pairs:
            higher_pairs = get_factor_pairs(x + offset)
            for pair in higher_pairs:
                if (1 not in pair) and (pair not in pairs):
                    pairs.append(pair)
            offset += 1

        return pairs

    def calculate_piece_areas(self, x_cuts, y_cuts):
        x_coords = np.sort(np.concatenate(([0], x_cuts, [self.width])))
        y_coords = np.sort(np.concatenate(([0], y_cuts, [self.height])))

        piece_widths = np.diff(x_coords)
        piece_heights = np.diff(y_coords)

        areas = np.concatenate(np.outer(piece_widths, piece_heights))

        return areas

    def loss_function(self, areas, requests):
        R = requests
        V = areas

        num_requests = len(R)
        num_values = len(V)

        cost_matrix = np.zeros((num_requests, num_values))

        for i, r in enumerate(R):
            for j, v in enumerate(V):
                cost_matrix[i][j] = abs(r - v) / r

        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        total_cost = sum(
            cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
        )

        return total_cost

    def calculate_gradient(self, x_cuts, y_cuts, curr_loss, epsilon=1e-3):
        grad_x_cuts = np.zeros_like(x_cuts, dtype=float)
        grad_y_cuts = np.zeros_like(y_cuts, dtype=float)

        for i in range(len(x_cuts)):
            x_cuts_eps = x_cuts.copy()
            x_cuts_eps[i] += epsilon
            areas_eps = self.calculate_piece_areas(x_cuts_eps, y_cuts)
            loss_eps = self.loss_function(areas_eps, self.requests)
            grad_x_cuts[i] = (loss_eps - curr_loss) / epsilon

        for i in range(len(y_cuts)):
            y_cuts_eps = y_cuts.copy()
            y_cuts_eps[i] += epsilon
            areas_eps = self.calculate_piece_areas(x_cuts, y_cuts_eps)
            loss_eps = self.loss_function(areas_eps, self.requests)
            grad_y_cuts[i] = (loss_eps - curr_loss) / epsilon

        return grad_x_cuts, grad_y_cuts

    def gradient_descent(
        self,
        learning_rate=1,
        num_iterations=500,
        epsilon=1e-3,
        learning_rate_decay=0.99,
    ):
        best_loss = float("inf")
        best_x_cuts = None
        best_y_cuts = None
        all_losses = []

        for factor in self.factors:
            # print(f"Factor pair: {factor}")
            num_horizontal, num_vertical = factor

            x_cuts = np.array(
                np.random.randint(1, self.width, num_vertical), dtype=float
            )
            y_cuts = np.array(
                np.random.randint(1, self.height, num_horizontal), dtype=float
            )

            best_x_cuts = x_cuts.copy()
            best_y_cuts = y_cuts.copy()

            losses = []
            lr = learning_rate
            for i in range(num_iterations):
                lr = max(lr * learning_rate_decay, 1e-2)

                areas = self.calculate_piece_areas(x_cuts, y_cuts)
                loss = self.loss_function(areas, self.requests)
                losses.append(loss)

                if loss < best_loss:
                    best_loss = loss
                    best_x_cuts = x_cuts.copy()
                    best_y_cuts = y_cuts.copy()

                grad_x_cuts, grad_y_cuts = self.calculate_gradient(
                    x_cuts, y_cuts, loss, epsilon
                )

                x_cuts -= lr * grad_x_cuts
                y_cuts -= lr * grad_y_cuts
                # print(f'Iteration {i + 1}: Loss = {loss}, Best loss = {best_loss}')
            all_losses.append(losses)
        all_losses = np.array(all_losses)

        return best_x_cuts, best_y_cuts, all_losses


EASY_LENGTH = 23.5


class Player:
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
        self.cuts = None
        self.knife_pos = []
        self.cut_count = 0

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
        turn_number = current_percept.turn_number
        requests = current_percept.requests
        polygons = current_percept.polygons
        cur_pos = current_percept.cur_pos
        num_requests = len(requests)
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width

        if cake_len <= EASY_LENGTH:
            if turn_number == 1:
                self.knife_pos.append([0, 0])
                return constants.INIT, [0, 0]
            
            if self.cut_count < num_requests:
                # Calculate the base length needed for the current polygon area
                base_length = round(2 * requests[self.cut_count] / cake_len, 2)
                knife_x = round(self.knife_pos[-2][0] + base_length, 2) if turn_number > 2 else base_length
                knife_y = cake_len if cur_pos[1] == 0 else 0
                
                # Adjust if the knife position goes beyond the cake width
                if knife_x > cake_width:
                    adjustment = round(2 * cake_area * 0.05 / (cake_width - self.knife_pos[-2][0]), 2)
                    knife_x = cake_width
                    knife_y = cake_len - adjustment if cur_pos[1] != 0 else adjustment

                next_knife_pos = [knife_x, knife_y]
                self.knife_pos.append(next_knife_pos)
                self.cut_count += 1
                return constants.CUT, next_knife_pos

            return constants.ASSIGN, optimal_assignment(
                requests, [polygon.area for polygon in polygons]
            )

        else:
            if turn_number == 1:

                grid_cut = grid_cut_strategy(cake_width, cake_len, requests)
                best_x_cuts, best_y_cuts, grid_cut_losses = grid_cut.gradient_descent()
                print(f"Best loss: {grid_cut_losses.min()}")

                cake_len = current_percept.cake_len
                cake_width = current_percept.cake_width

                num_cuts = len(requests)
                num_restarts = 30
                stagnant_limit = 20
                min_loss = float("inf")
                best_cuts = None
                # num_steps = 100

                for restart in range(num_restarts):
                    cuts = generate_random_cuts(num_cuts, (cake_width, cake_len))
                    loss = self.get_loss_from_cuts(cuts, current_percept)
                    print(f"Restart {restart} Loss: {loss}")

                    stagnant_steps = 0
                    prev_loss = loss

                    if loss < min_loss:
                        best_cuts = copy.deepcopy(cuts)
                        min_loss = loss

                    # Gradient descent
                    learning_rate = 0.1

                    step = 0
                    # while step < num_steps:
                    while loss > 0.01 and stagnant_steps < stagnant_limit:
                        gradients = self.get_gradient(loss, cuts, current_percept)

                        cur_x, cur_y = cuts[0]
                        for j in range(len(cuts)):
                            cuts[j] = get_shifted_cut(
                                cuts[j],
                                -learning_rate * gradients[j],
                                (cake_width, cake_len),
                                (cur_x, cur_y),
                            )
                            cur_x, cur_y = cuts[j]
                        loss = self.get_loss_from_cuts(cuts, current_percept)
                        if loss < min_loss:
                            best_cuts = copy.deepcopy(cuts)
                            min_loss = loss

                        # Check for stagnation
                        if prev_loss - loss < 0.01:
                            stagnant_steps += 1
                        else:
                            stagnant_steps = 0
                        prev_loss = loss

                        print(f"Step: {step}, Loss: {loss}")
                        step += 1

                print(f"Best penalty: {min_loss * 100}")

                self.cuts = [[round(cut[0], 2), round(cut[1], 2)] for cut in best_cuts]
                return constants.INIT, self.cuts[0]
            elif turn_number <= len(self.cuts):
                return constants.CUT, self.cuts[turn_number - 1]

            return constants.ASSIGN, optimal_assignment(
                requests, [polygon.area for polygon in polygons]
            )

    def get_loss_from_cuts(self, cuts, current_percept):
        new_percept = copy.deepcopy(current_percept)
        new_polygons = new_percept.polygons

        new_percept.cur_pos = cuts[0]

        for cut in cuts[1:]:
            new_polygons, new_percept = self.cut_cake(
                cut,
                new_polygons,
                new_percept,
            )
        loss = cost_function(new_polygons, current_percept.requests)

        return loss

    def get_gradient(self, loss, cuts, current_percept):
        dw = current_percept.cake_width / 100
        gradients = np.zeros(len(cuts))

        cur_x, cur_y = cuts[0]
        for i in range(len(cuts)):
            # for i in tqdm(range(len(cuts))):
            new_cuts = copy.deepcopy(cuts)
            new_cuts[i] = get_shifted_cut(
                cuts[i],
                dw,
                (current_percept.cake_width, current_percept.cake_len),
                (cur_x, cur_y),
            )
            cur_x, cur_y = new_cuts[i]

            new_loss = self.get_loss_from_cuts(new_cuts, current_percept)
            gradients[i] = (new_loss - loss) / dw
        return gradients

    def cut_cake(self, cut, polygons, current_percept):
        # Check if the next position is on the boundary of the cake
        if invalid_knife_position(cut, current_percept):
            raise ValueError("Invalid knife position")

        # Cut the cake piece
        newPieces = []
        for polygon in polygons:
            line_points = LineString([tuple(current_percept.cur_pos), tuple(cut)])
            slices = divide_polygon(polygon, line_points)
            for slice in slices:
                newPieces.append(slice)

        current_percept.cur_pos = cut
        return newPieces, current_percept


def invalid_knife_position(pos, current_percept):
    cur_x, cur_y = pos
    if (cur_x != 0 and cur_x != current_percept.cake_width) and (
        cur_y != 0 and cur_y != current_percept.cake_len
    ):
        return True

    if cur_x == 0 or cur_x == current_percept.cake_width:
        if cur_y < 0 or cur_y > current_percept.cake_len:
            return True

    if cur_y == 0 or cur_y == current_percept.cake_len:
        if cur_x < 0 or cur_x > current_percept.cake_width:
            return True
    return False


def divide_polygon(polygon, line):
    if not line.intersects(polygon):
        return [polygon]
    result = split(polygon, line)

    polygons = []
    for i in range(len(result.geoms)):
        polygons.append(result.geoms[i])

    return polygons


def generate_random_cuts(num_cuts, cake_dims):
    cake_width, cake_len = cake_dims
    corner_gap = 1e-3
    cur_x, cur_y = [
        [0, np.random.uniform(corner_gap, cake_len - corner_gap)],
        [cake_width, np.random.uniform(corner_gap, cake_len - corner_gap)],
        [np.random.uniform(corner_gap, cake_width - corner_gap), 0],
        [np.random.uniform(corner_gap, cake_width - corner_gap), cake_len],
    ][np.random.choice(4)]

    cuts = [[cur_x, cur_y]]
    for i in range(num_cuts):
        # Generate random cuts
        top = [
            np.random.uniform(corner_gap, cake_width - corner_gap),
            0,
        ]
        bottom = [
            np.random.uniform(corner_gap, cake_width - corner_gap),
            cake_len,
        ]
        right = [
            cake_width,
            np.random.uniform(corner_gap, cake_len - corner_gap),
        ]
        left = [
            0,
            np.random.uniform(corner_gap, cake_len - corner_gap),
        ]

        if cur_x == 0:  # Start from left
            cuts.append([top, bottom, right][np.random.choice(3)])
        elif cur_x == cake_width:  # Start from right
            cuts.append([top, bottom, left][np.random.choice(3)])
        elif cur_y == 0:  # Start from top
            cuts.append([bottom, left, right][np.random.choice(3)])
        elif cur_y == cake_len:  # Start from bottom
            cuts.append([top, left, right][np.random.choice(3)])

        cur_x, cur_y = cuts[-1]

    return cuts


def cost_function(polygons, requests):
    R = requests
    V = [polygon.area for polygon in polygons]

    num_requests = len(R)
    num_values = len(V)

    cost_matrix = np.zeros((num_requests, num_values))

    # Fill the cost matrix with relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            cost_matrix[i][j] = abs(r - v) / r

    # Solving the assignment problem
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Calculate the total cost by summing the optimal assignment costs
    total_cost = sum(
        cost_matrix[row_indices[i], col_indices[i]] for i in range(len(row_indices))
    )

    return total_cost


def get_shifted_cut(cut, shift, cake_dims, pos):
    cake_width, cake_len = cake_dims
    cur_x, cur_y = pos

    shifted_cut = copy.deepcopy(cut)
    if cut[0] == 0:  # Left
        if cut[1] + shift > cake_len:
            if cur_y != cake_len:
                remainder = (
                    cut[1] + shift - cake_len
                    if cut[1] + shift - cake_len < cake_width
                    else cake_width
                )
                shifted_cut = [remainder, cake_len]
        elif cut[1] + shift < 0:
            if cur_y != 0:
                remainder = (
                    -(cut[1] + shift) if -(cut[1] + shift) < cake_width else cake_width
                )
                shifted_cut = [remainder, 0]
        else:
            shifted_cut = [0, cut[1] + shift]
    elif cut[0] == cake_width:  # Right
        if cut[1] + shift > cake_len:
            if cur_y != cake_len:
                remainder = cut[1] - shift if cake_width - (cut[1] - shift) > 0 else 0
                shifted_cut = [cake_width - remainder, cake_len]
        elif cut[1] + shift < 0:
            if cur_y != 0:
                remainder = (
                    -(cut[1] + shift) if cake_width - (cut[1] + shift) > 0 else 0
                )
                shifted_cut = [cake_width - remainder, 0]
        else:
            shifted_cut = [cake_width, cut[1] + shift]
    elif cut[1] == 0:  # Top
        if cut[0] + shift > cake_width:
            if cur_x != cake_width:
                remainder = (
                    cut[0] + shift - cake_width
                    if cut[0] + shift - cake_width < cake_len
                    else cake_len
                )
                shifted_cut = [cake_width, remainder]
        elif cut[0] + shift < 0:
            if cur_x != 0:
                remainder = (
                    -(cut[0] + shift) if -(cut[0] + shift) < cake_len else cake_len
                )
                shifted_cut = [0, remainder]
        else:
            shifted_cut = [cut[0] + shift, 0]
    elif cut[1] == cake_len:  # Bottom
        if cut[0] + shift > cake_width:
            if cur_x != cake_width:
                remainder = cut[0] - shift if cake_len - (cut[0] - shift) > 0 else 0
                shifted_cut = [
                    cake_width,
                    cake_len - remainder,
                ]
        elif cut[0] + shift < 0:
            if cur_x != 0:
                remainder = -(cut[0] + shift) if cake_len - (cut[0] + shift) > 0 else 0
                shifted_cut = [0, cake_len - remainder]
        else:
            shifted_cut = [cut[0] + shift, cake_len]
    return shifted_cut


def optimal_assignment(R, V):
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
