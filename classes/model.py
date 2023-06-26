"""
Complex System Simulation Group Project
"""


from classes.island import Island

import math
import numpy as np

class Model:
    """
    Creates a model with multiple islands.
    """

    def __init__(self, island_grid_size, total_grid, iterations, alpha, no_islands=4) -> None:
        self.no_islands = no_islands
        self.grid_size = island_grid_size
        self.total_grid_size = total_grid
        self.islands = self.create_islands()
        self.iterations = iterations
        self.alpha = alpha
        self.counter = 0

    def create_islands(self) -> list:
        new_islands = list()
        half_size = int(self.grid_size / 2) + 1

        for i in range(self.no_islands):
            x_coordinate = np.random.randint(half_size, self.total_grid_size - half_size)
            y_coordinate = np.random.randint(half_size, self.total_grid_size - half_size)
            if self.check_location(x_coordinate, y_coordinate, half_size, new_islands) and i > 0:
                continue
            new_islands.append(Island(x_coordinate, y_coordinate))
        return new_islands

    def check_location(self, x, y, size, islands) -> bool:
        for island in islands:
            if island.get_distance([x, y]) < size:
                return True
        return False

    def simple_prob(self, distance) -> float:
        return 1 / (2 * distance)

    def interaction_prob(self, distance) -> float:
        pass

    def interaction(self, i) -> None:
        unique_values, counts = np.unique(self.islands[i].algorithm.current_grid.ravel(), return_counts=True)
        max_count = np.argmax(counts)
        most = unique_values[max_count]

        x = np.random.choice(np.arange(0, self.islands[i-1].width))
        y = np.random.choice(np.arange(0, self.islands[i-1].height))
        self.islands[i-1].algorithm.current_grid[x, y] = most

    def step(self) -> None:

        # For every island perform the single island dynamics
        for island in self.islands:
            island.step(self.alpha)

        # Loop through every island to see if migration takes place
        length_islands = len(self.islands)
        for i in range(length_islands):

            # Get random index from other island
            indices = [j for j in range(length_islands) if j != i]
            random_index = np.random.choice(indices)

            # Calculate distance towards other island and calculate probability to migrate
            distance = self.islands[i].get_distance([self.islands[random_index].get_coordinates()])
            prob = self.simple_prob(distance)

            # Check if migrations takes place and perform the interaction
            if prob > np.random.uniform():
                self.counter += 1
                self.interaction(i)
