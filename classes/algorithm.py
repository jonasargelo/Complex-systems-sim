"""
Complex System Simulation Group Project
"""


import numpy as np

class Voter:
    """
    Creates an istance of the voter forward algorithm.
    Parameters:
    - K (int): size of the grid
    """

    def __init__(self, K):
        self.starting_grid = np.random.rand(K, K)
        self.current_grid = self.starting_grid.copy()

    def get_8_neighbors(self, i, j, K):
        '''Finds all 8 neighbors of a site in a K by K grid'''
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                # Skip the current site
                if di == 0 and dj == 0:
                    continue
                # Apply periodic boundary conditions
                neighbor_i = int((i + di) % K)
                neighbor_j = int((j + dj) % K)
                neighbors.append((neighbor_i, neighbor_j))
        return neighbors

    def get_4_neighbors(self, i, j, K):
        '''Finds upper, lower, left and right neighbor of a site in a K by K grid'''
        neighbors = []
        # Up, down, right, left
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for di, dj in directions:
            # Apply periodic boundary conditions
            neighbor_i = int((i + di) % K)
            neighbor_j = int((j + dj) % K)
            neighbors.append((neighbor_i, neighbor_j))
        return neighbors

    def get_starting_grid(self):
        """
        Return the starting grid.
        """
        return self.starting_grid

    def get_current_grid(self):
        """
        Return the current grid
        """
        return self.current_grid
