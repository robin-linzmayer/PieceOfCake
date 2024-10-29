import os
import pickle
from typing import List
from scipy.optimize import linear_sum_assignment

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
		
	def which_wall_am_i_on(self, cur_pos, cake_width, cake_len):
		"""
			Function to determine which wall the amoeba is currently on
		"""
		if cur_pos[0] == 0:
			return "left"
		elif cur_pos[0] == cake_width:
			return "right"
		elif cur_pos[1] == 0:
			return "top"
		elif cur_pos[1] == cake_len:
			return "bottom"
		else:
			return "middle"
	

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
			return constants.INIT, [0,0]

		if len(polygons) < len(requests):
			if cur_pos[0] == 0:
				return constants.CUT, [cake_width, round((cur_pos[1] + 5)%cake_len, 2)]
			else:
				return constants.CUT, [0, round((cur_pos[1] + 5)%cake_len, 2)]

		assignment = []
		for i in range(len(requests)):
			assignment.append(i)

		return constants.ASSIGN, assignment

	def match_pieces(self, current_percept):
		"""
			Function to match the pieces of cake to the order based on the polygons formed
		"""
		# Implement your logic here
		pass

	def create_cost_matrix(self, polygons, requests):
		"""
			Function to create a cost matrix based on the areas of polygons and requests
		"""
		# Implement your logic here
		n = len(polygons)
		m = len(requests)

		cost_matrix = np.full((n, m), np.inf)  # Initialize with infinity

		for i in range(n):
			for j in range(m):

				difference = abs(polygons[i].area - requests[j])

				if difference <= self.tolerance:
					cost_matrix[i][j] = 0		# No penalty
				else:
					cost_matrix[i][j] = (difference / requests[j]) * 100 # Penalty



		return cost_matrix

	def hungarian_algorithm(self, polygons, requests):
		"""
			Function to implement the Hungarian algorithm for optimal assignment
		"""

		cost_matrix = self.create_cost_matrix(polygons, requests)

		# Use the Hungarian method to find the optimal assignment
		row_ind, col_ind = linear_sum_assignment(cost_matrix)

		matches = []

		for i, j in zip(row_ind, col_ind):
			if cost_matrix[i][j] < np.inf:
				penalty = cost_matrix[i][j]

				# THIS RETURNS AN INDEX!!!! MAY NEED TO BE ADJUSTED
				matches.append((i, j, penalty))


		return matches

		


