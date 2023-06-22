from classes.island import Island

import numpy as np

class Model:
    """
    Creates a model with multiple islands.
    """
    def __init__(self, grid_size, iterations, alpha, no_islands=4) -> None:
        self.no_islands = no_islands
        self.grid_size = grid_size
        self.islands = self.create_islands()
        self.iterations = iterations
        self.alpha = alpha

    def create_islands(self) -> list:
        new_islands = list()

        for i in range(self.no_islands):
            x_coordinate = np.random.randint(self.grid_size)
            y_coordinate = np.random.randint(self.grid_size)
            new_islands.append(Island(x_coordinate, y_coordinate))
        return new_islands

    def interaction_formula(self) -> float:
        pass

    def interaction(self):
        pass

    def step(self) -> None:
        for island in self.islands:
            island.step(self.alpha)

