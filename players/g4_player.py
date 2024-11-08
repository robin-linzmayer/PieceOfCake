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
import time
import miniball
import math


SAVE_DATA = False


class grid_cut_strategy:
    def __init__(self, width, height, requests):
        self.width = width
        self.height = height
        self.requests = requests
        self.factors = self.factor_pairs(len(requests))

    def factor_pairs(self, x):
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
        start_time = time.time()

        turn_number = current_percept.turn_number
        requests = current_percept.requests
        polygons = current_percept.polygons
        cur_pos = current_percept.cur_pos
        num_requests = len(requests)
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width

        if turn_number == 1:

            strategies = []
            zig_zag_loss = float("inf")
            grid_loss = float("inf")
            gd_loss = float("inf")

            try:
                if cake_len < 24:
                    zig_zag_cuts = self.zig_zag(current_percept, requests)
                    zig_zag_loss = self.get_loss_from_cuts(
                        zig_zag_cuts,
                        current_percept,
                        plate=True,
                        tolerance=self.tolerance,
                    )
                    strategies.append((zig_zag_cuts, zig_zag_loss))
                    print(f"Zig zag loss: {zig_zag_loss}")
            except Exception as e:
                print(e)

            if zig_zag_loss > 0:
                try:
                    grid_cut_strat = grid_cut_strategy(cake_width, cake_len, requests)

                    if num_requests < 50:
                        best_x_cuts, best_y_cuts, grid_cut_losses = (
                            grid_cut_strat.gradient_descent(num_iterations=1000)
                        )
                    else:
                        best_x_cuts, best_y_cuts, grid_cut_losses = (
                            grid_cut_strat.gradient_descent()
                        )

                    print(grid_cut_losses.min())

                    grid_cuts = []
                    grid_cuts.extend(
                        self.vertical_cut(
                            list(best_x_cuts),
                            cake_len,
                            cake_width,
                        )
                    )
                    grid_cuts.extend(
                        self.horizontal_cut(
                            list(best_y_cuts),
                            cake_len,
                            cake_width,
                            grid_cuts[-1],
                        )
                    )

                    # Uncomment below after cuts are generated
                    grid_loss = self.get_loss_from_cuts(
                        grid_cuts,
                        current_percept,
                        plate=True,
                        tolerance=self.tolerance,
                    )
                    strategies.append((grid_cuts, grid_loss))
                    print(f"Grid cut loss: {grid_loss}")
                except Exception as e:
                    print(e)

                try:
                    gd_cuts = self.gradient_descent(
                        requests, start_time, current_percept
                    )
                    gd_loss = self.get_loss_from_cuts(
                        gd_cuts,
                        current_percept,
                        plate=True,
                        tolerance=self.tolerance,
                    )
                    strategies.append((gd_cuts, gd_loss))
                    print(f"Gradient descent loss: {gd_loss}")
                except Exception as e:
                    print(e)

            if grid_loss == gd_loss and grid_loss != float("inf"):
                self.cuts = [[round(cut[0], 2), round(cut[1], 2)] for cut in gd_cuts]
            else:
                best_loss = float("inf")
                best_cuts = []
                for cuts, loss in strategies:
                    if loss < best_loss and len(cuts) > 0:
                        best_loss = loss
                        best_cuts = cuts
                self.cuts = [[round(cut[0], 2), round(cut[1], 2)] for cut in best_cuts]
                return constants.INIT, self.cuts[0]

        elif turn_number <= len(self.cuts):
            return constants.CUT, self.cuts[turn_number - 1]

        return constants.ASSIGN, optimal_assignment(
            requests, [polygon.area for polygon in polygons]
        )

    def zig_zag(self, current_percept, requests):
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width
        cake_area = cake_len * cake_width

        num_restarts = 30

        min_loss = float("inf")
        best_cuts = []

        for restart in range(num_restarts):
            try:
                cuts = [[0, 0]]
                scrambled_requests = np.random.permutation(requests)
                for request in scrambled_requests:
                    # Calculate the base length needed for the current polygon area
                    base_length = round(2 * request / cake_len, 2)
                    knife_x = (
                        round(cuts[-2][0] + base_length, 2)
                        if len(cuts) > 2
                        else base_length
                    )
                    knife_y = cake_len if cuts[-1][1] == 0 else 0

                    # Adjust if the knife position goes beyond the cake width
                    if knife_x > cake_width:
                        adjustment = round(
                            2 * cake_area * 0.05 / (cake_width - cuts[-2][0]), 2
                        )
                        knife_x = cake_width
                        knife_y = (
                            cake_len - adjustment if cuts[-1][1] != 0 else adjustment
                        )
                    cuts.append([knife_x, knife_y])
                loss = self.get_loss_from_cuts(cuts, current_percept, plate=True)

                if loss < min_loss:
                    min_loss = loss
                    best_cuts = copy.deepcopy(cuts)
            except:
                pass

        return best_cuts

    def gradient_descent(self, requests, start_time, current_percept):
        cake_len = current_percept.cake_len
        cake_width = current_percept.cake_width

        num_restarts = 30
        stagnant_limit = 20
        min_loss = float("inf")
        best_cuts = None

        all_losses = []

        while True:
            if len(requests) > 50:
                num_cuts = math.floor(np.abs(np.random.normal(len(requests) // 3, 5)))
            elif len(requests) > 20:
                num_cuts = math.floor(np.abs(np.random.normal(len(requests) // 2, 5)))
            else:
                num_cuts = math.floor(np.abs(np.random.normal(len(requests), 2)))

            # Time check
            if current_percept.time_remaining - time.time() + start_time < 60:
                break

            cuts = generate_random_cuts(num_cuts, (cake_width, cake_len))
            loss = self.get_loss_from_cuts(cuts, current_percept)
            losses = [loss]

            stagnant_steps = 0
            prev_loss = loss

            if loss < min_loss:
                best_cuts = copy.deepcopy(cuts)
                min_loss = loss

            # Gradient descent
            learning_rate = 1

            step = 0
            # while step < num_steps:
            while loss > 0.01 and stagnant_steps < stagnant_limit:
                learning_rate = max(0.1, learning_rate * 0.995)
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
                losses.append(loss)
                if loss < min_loss:
                    best_cuts = copy.deepcopy(cuts)
                    min_loss = loss

                # Check for stagnation
                if prev_loss - loss < 0.01:
                    stagnant_steps += 1
                else:
                    stagnant_steps = 0
                prev_loss = loss

                # Time check
                if current_percept.time_remaining - time.time() + start_time < 60:
                    break

                step += 1
            all_losses.append(losses)

        if SAVE_DATA:
            try:
                np.save(f"loss_{num_cuts}_{len(requests)}.npy", all_losses)
            except:
                pass

        return best_cuts

    def get_loss_from_cuts(self, cuts, current_percept, plate=True, tolerance=0):
        new_percept = copy.deepcopy(current_percept)
        new_polygons = new_percept.polygons

        new_percept.cur_pos = cuts[0]

        for cut in cuts[1:]:
            new_polygons, new_percept = self.cut_cake(
                cut,
                new_polygons,
                new_percept,
            )
        loss = cost_function(new_polygons, current_percept.requests, plate, tolerance)
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

    def is_on_top_half(self, cur_y):
        return cur_y == 0

    def is_on_bottom_half(self, cur_y, cake_len):
        return cur_y == cake_len

    def is_on_left_half(self, cur_x):
        return cur_x == 0

    def is_on_right_half(self, cur_x, cake_width):
        return cur_x == cake_width

    def go_to_top_right_corner(self, cake_width, default=True):
        if default:
            return [cake_width, 0.01]
        return [cake_width - 0.01, 0]

    def go_to_top_left_corner(self, default=True):
        if default:
            return [0, 0.01]
        return [0.01, 0]

    def go_to_bottom_right_corner(self, cake_width, cake_len, default=True):
        if default:
            return [cake_width, round(cake_len - 0.01, 2)]
        return [round(cake_width - 0.01, 2), cake_len]

    def go_to_bottom_left_corner(self, cake_len, default=True):
        if default:
            return [0, round(cake_len - 0.01, 2)]
        return [0.01, cake_len]

    def vertical_cut(self, x_cuts_indices, cake_len, cake_width):
        cuts = []

        # add cut to (index, 0)
        cuts.append([x_cuts_indices[0], 0])
        # print("Init at to (", x_cuts_indices[0], ", 0)")

        for cut in x_cuts_indices:

            # if location is on bottom edge, add cut to the top edge
            if self.is_on_bottom_half(cuts[-1][1], cake_len):
                # print("Location is on bottom edge")
                # print("Appending cut to traverse to top edge")
                cuts.append([cut, 0])

            # if location is on top edge, add cut to the bottom edge
            elif self.is_on_top_half(cuts[-1][1]):
                # print("Location is on top edge")
                # print("Appending cut to traverse to bottom edge")
                cuts.append([cut, cake_len])

            # if there is a next cut after current cut
            if cut != x_cuts_indices[-1]:
                next_cut = x_cuts_indices[x_cuts_indices.index(cut) + 1]
                # print("There is a next cut after current cut")
                # if next cut is on left half of the cake
                if next_cut < cake_width / 2 and self.is_on_bottom_half(
                    cuts[-1][1], cake_len
                ):
                    # print("Next cut is on bottom left half of the cake")
                    # print("Appending cut to bottom left corner")
                    cuts.append(self.go_to_bottom_left_corner(cake_len))
                    # Go to next index
                    # print(
                    #     "Going to next index, appending cut to (",
                    #     x_cuts_indices[x_cuts_indices.index(cut) + 1],
                    #     ",",
                    #     cake_len,
                    #     ")",
                    # )
                    cuts.append(
                        [x_cuts_indices[x_cuts_indices.index(cut) + 1], cake_len]
                    )
                # if next cut is on right half of the cake
                elif next_cut >= cake_width / 2 and self.is_on_bottom_half(
                    cuts[-1][1], cake_len
                ):
                    # print("Next cut is on bottom right half of the cake")
                    cuts.append(self.go_to_bottom_right_corner(cake_width, cake_len))
                    # Go to next index
                    # print(
                    #     "Going to next index, appending cut to (",
                    #     x_cuts_indices[x_cuts_indices.index(cut) + 1],
                    #     ",",
                    #     cake_len,
                    #     ")",
                    # )
                    cuts.append(
                        [x_cuts_indices[x_cuts_indices.index(cut) + 1], cake_len]
                    )
                # if next cut is on left half of the cake
                elif next_cut < cake_width / 2 and self.is_on_top_half(cuts[-1][1]):
                    # print("Next cut is to the top left half of the cake")
                    cuts.append(self.go_to_top_left_corner())
                    # Go to next index
                    # print(
                    #     "Going to next index, appending cut to (",
                    #     x_cuts_indices[x_cuts_indices.index(cut) + 1],
                    #     ", 0)",
                    # )
                    cuts.append([x_cuts_indices[x_cuts_indices.index(cut) + 1], 0])
                    # if next cut is on right half of the cake
                elif next_cut >= cake_width / 2 and self.is_on_top_half(cuts[-1][1]):
                    # print("Next cut is to the top right half of the cake")
                    cuts.append(self.go_to_top_right_corner(cake_width))
                    # Go to next index
                    # print(
                    #     "Going to next index, appending cut to (",
                    #     x_cuts_indices[x_cuts_indices.index(cut) + 1],
                    #     ", 0)",
                    # )
                    cuts.append([x_cuts_indices[x_cuts_indices.index(cut) + 1], 0])

        return cuts

    def horizontal_cut(self, y_cuts_indices, cake_len, cake_width, curr_pos):
        cuts = []

        if self.is_on_top_half(curr_pos[1]):
            cuts.append(self.go_to_top_right_corner(cake_width))
            cuts.append(self.go_to_top_right_corner(cake_width, False))
            cuts.append([cake_width, y_cuts_indices[0]])

        # if on bottom half of cake, go to bottom right corner
        if self.is_on_bottom_half(curr_pos[1], cake_len):
            cuts.append(self.go_to_bottom_right_corner(cake_width))
            cuts.append(self.go_to_bottom_right_corner(cake_width, False))
            cuts.append([cake_width, y_cuts_indices[0]])

        # Start the first cut at (0, y_cuts_indices[0])
        # cuts.append([0, y_cuts_indices[0]])
        # print(f"Init at (0, {y_cuts_indices[0]})")

        for cut in y_cuts_indices:

            # if the current location is on the left half of the cake, add cut to the right edge
            if self.is_on_left_half(cuts[-1][0]):
                # print("Location is on the left edge")
                # print("Appending cut to traverse to right edge")
                cuts.append([cake_width, cut])

            # if the location is on the right half of the cake, add cut to the left edge
            elif self.is_on_right_half(cuts[-1][0], cake_width):
                # print("Location is on the right edge")
                # print("Appending cut to traverse to left edge")
                cuts.append([0, cut])

            # if there is a next cut after the current cut
            if cut != y_cuts_indices[-1]:
                next_cut = y_cuts_indices[y_cuts_indices.index(cut) + 1]
                # print("There is a next cut after the current cut")

                # if the next cut is in the top half of the cake
                if next_cut < cake_len / 2 and self.is_on_left_half(cuts[-1][0]):
                    # print("Next cut is in the top left half of the cake")
                    cuts.append(self.go_to_top_left_corner(False))
                    # move to next index and append cut to (cake_width, next_cut)
                    # print(f"Going to next index, appending cut to ({0}, {next_cut})")
                    cuts.append([0, next_cut])

                # if the next cut is in the bottom half of the cake
                elif next_cut >= cake_len / 2 and self.is_on_left_half(cuts[-1][0]):
                    # print("Next cut is in the bottom left half of the cake")
                    cuts.append(self.go_to_bottom_left_corner(cake_len, False))
                    # move to next index and append cut to (cake_width, next_cut)
                    # print(f"Going to next index, appending cut to ({0}, {next_cut})")
                    cuts.append([0, next_cut])

                # if the next cut is in the top half of the cake
                elif next_cut < cake_len / 2 and self.is_on_right_half(
                    cuts[-1][0], cake_width
                ):
                    # print("Next cut is in the top right half of the cake")
                    cuts.append(self.go_to_top_right_corner(cake_width, False))
                    # move to next index and append cut to (0, next_cut)
                    # print(f"Going to next index, appending cut to (0, {next_cut})")
                    cuts.append([cake_width, next_cut])

                # if the next cut is in the bottom half of the cake
                elif next_cut >= cake_len / 2 and self.is_on_right_half(
                    cuts[-1][0], cake_width
                ):
                    # print("Next cut is in the bottom right half of the cake")
                    cuts.append(
                        self.go_to_bottom_right_corner(cake_width, cake_len, False)
                    )
                    # move to next index and append cut to (0, next_cut)
                    # print(f"Going to next index, appending cut to (0, {next_cut})")
                    cuts.append([cake_width, next_cut])

        return cuts


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


def cost_function(polygons, requests, plate, tolerance):
    if plate:
        V = []
        for polygon in polygons:
            cake_points = np.array(
                list(zip(*polygon.exterior.coords.xy)), dtype=np.double
            )
            res = miniball.miniball(cake_points)
            if res["radius"] <= 12.5:
                V.append(polygon.area)
    else:
        V = [polygon.area for polygon in polygons]

    R = requests

    num_requests = len(R)
    num_values = len(V)

    cost_matrix = np.zeros((num_requests, num_values))

    # Fill the cost matrix with relative differences
    for i, r in enumerate(R):
        for j, v in enumerate(V):
            penalty = abs(r - v) / r * 100
            if penalty > tolerance:
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
