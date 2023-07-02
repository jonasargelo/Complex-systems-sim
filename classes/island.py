"""
Complex System Simulation Group Project
"""


from classes.algorithm import Voter

import numpy as np
import random

class Island:
    """
    Creates one island which simulates an algorithm.
    Parameters:
    - grid_size (int): size of the grid
    - coordinates (numpy array): x and y coordinates within the large grid
    - algorithm (object): instance of an algorithm to be used for individual
    island dynamics.
    """
    def __init__(self, x_coordinate, y_coordinate, grid_size, n_iters, algorithm='Voter') -> None:
        self.grid_size = grid_size
        self.coordinates = np.array([x_coordinate, y_coordinate])
        self.xs = np.random.choice(np.arange(0, self.grid_size), size=n_iters)
        self.ys = np.random.choice(np.arange(0, self.grid_size), size=n_iters)
        self.species = list()
        self.width = grid_size
        self.height = grid_size

        # Check whether the chosen algorithm is Voter
        if algorithm == 'Voter':
            self.algorithm = Voter(grid_size)
        # self.num_species = len(np.unique(self.algorithm.current_grid))

    def get_distance(self, coordinates_other) -> float:
        """
        Returns distance between two island objects.
        """
        return np.linalg.norm(self.coordinates - coordinates_other)

    def get_coordinates(self) -> list:
        """
        Returns coordinates of an island.
        """
        return self.coordinates

    def step(self, alpha, i) -> None:
        """
        Performs a single step of the voter algorithm
        """

        if random.random() < alpha:
            self.algorithm.current_grid[self.xs[i], self.ys[i]] = random.random()

        # Set species in cell to that in one of its 4 neighbors with equal probability
        else:
            neighbors = self.algorithm.get_4_neighbors(self.xs[i], self.ys[i], self.grid_size)
            neighbor_idx = np.random.choice(len(neighbors))
            new_type = self.algorithm.current_grid[neighbors[neighbor_idx]]
            self.algorithm.current_grid[self.xs[i], self.ys[i]] = new_type

        # Save the number of species every thousand steps
        if i % 1000 == 0:
            self.species.append(len(np.unique(self.algorithm.current_grid)))
