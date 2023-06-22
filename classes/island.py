from classes.algorithm import Voter

import numpy as np
import random

class Island:
    """
    Creates one island which simulates an algorithm.
    """
    def __init__(self, x_coordinate, y_coordinate, algorithm='Voter', grid_size=6) -> None:
        self.grid_size = grid_size
        self.coordinates = np.array([x_coordinate, y_coordinate])
        self.num_species = list()
        self.width = grid_size
        self.height = grid_size
        # self.x = x_coordinate
        # self.y = y_coordinate
        if algorithm == 'Voter':
            self.algorithm = Voter(grid_size)


    def get_distance(self, coordinates_other) -> float:
        return np.linalg.norm(self.coordinates - coordinates_other)

    def get_coordinates(self) -> list:
        return self.coordinates

    def get_boundaries(self):
        if self.grid_size % 2 == 0:
            return
        else:
            return

    def step(self, alpha):
        #self.algorithm.update_grid()
        x = np.random.choice(np.arange(0, self.width))
        y = np.random.choice(np.arange(0, self.height))

        if random.random() < alpha:
            self.algorithm.current_grid[x, y] = random.random()
        # Set species in cell to that in one of its 4 neighbors with equal probability
        else:
            neighbors = self.algorithm.get_4_neighbors(x, y, self.grid_size)
            neighbor_idx = np.random.choice(len(neighbors))
            new_type = self.algorithm.current_grid[neighbors[neighbor_idx]]
            self.algorithm.current_grid[x, y] = new_type

        return self.algorithm.current_grid, self.num_species

