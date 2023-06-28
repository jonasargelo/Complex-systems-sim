"""
Complex System Simulation Group Project
"""


from classes.algorithm import Voter

import itertools
import numpy as np
import random

class Island:
    """
    Creates one island which simulates an algorithm.
    """
    def __init__(self, x_coordinate, y_coordinate, grid_size, n_iters, algorithm='Voter') -> None:
        self.grid_size = grid_size
        self.coordinates = np.array([x_coordinate, y_coordinate])
        self.num_species = list()
        self.rand_walkers = list(itertools.product(range(self.grid_size), range(self.grid_size)))
        self.rand_walker_pos = list(self.rand_walkers.copy())
        self.lineages = self.initialize_lineages()
        self.xs = np.random.choice(np.arange(0, self.grid_size), size=n_iters)
        self.ys = np.random.choice(np.arange(0, self.grid_size), size=n_iters)
        self.species = list()
        self.width = grid_size
        self.height = grid_size
        self.x = x_coordinate
        self.y = y_coordinate
        if algorithm == 'Voter':
            self.algorithm = Voter(grid_size)

    def initialize_lineages(self):
        lin = [set() for _ in range(len(self.rand_walkers))]
        for i, element in enumerate(self.rand_walkers):
            lin[i].add(element)
        return lin

    def get_distance(self, coordinates_other) -> float:
        return np.linalg.norm(self.coordinates - coordinates_other)

    def get_coordinates(self) -> list:
        return self.coordinates

    def get_boundaries(self):
        if self.grid_size % 2 == 0:
            return
        else:
            return

    def step(self, alpha, i):
        #self.algorithm.update_grid()
        # x = np.random.choice(np.arange(0, self.width))
        # y = np.random.choice(np.arange(0, self.height))

        if random.random() < alpha:
            self.algorithm.current_grid[self.xs[i], self.ys[i]] = random.random()
        # Set species in cell to that in one of its 4 neighbors with equal probability
        else:
            neighbors = self.algorithm.get_4_neighbors(self.xs[i], self.ys[i], self.grid_size)
            neighbor_idx = np.random.choice(len(neighbors))
            new_type = self.algorithm.current_grid[neighbors[neighbor_idx]]
            self.algorithm.current_grid[self.xs[i], self.ys[i]] = new_type

        return self.algorithm.current_grid, self.num_species

    def new_step(self, alpha):
        # Draw full list of random numbers to save time
        rand_nums = np.random.uniform(size=self.width**2)

        for j in range(self.width**2):
            if len(self.rand_walker_pos) == 0:
                return self.lineages, self.species

            # Select active walker
            cur_walker_loc = np.random.choice(len(self.rand_walker_pos))
            cur_walker = self.rand_walker_pos[cur_walker_loc]
            # Find current walker's place in lineages
            cur_walker_idx = self.width*cur_walker[0] + cur_walker[1]

            # Select parent
            potential_parents = self.algorithm.get_4_neighbors(cur_walker[0], cur_walker[1], self.width)
            parent_walker_loc = np.random.choice(len(potential_parents))
            parent_walker = potential_parents[parent_walker_loc]

            # Find parent's place in lineages
            parent_walker_idx = self.width*parent_walker[0] + parent_walker[1]

            # Speciate with probability alpha
            speciate = False
            if rand_nums[j] < alpha:
                speciate = True
                self.species.append(self.lineages[cur_walker_idx])
                # Insert pointer to correct species where the walker is in
                for site in self.lineages[cur_walker_idx]:
                    site_idx = self.width*site[0] + site[1]
                    self.lineages[site_idx] = len(self.species) - 1
            else:
                # Check if parent hasn't speciated yet
                if type(self.lineages[parent_walker_idx]) == set:
                    # Merge lineages if walkers are from other walks
                    if parent_walker not in self.lineages[cur_walker_idx]:
                        # Unify lineages of parent and child
                        self.lineages[cur_walker_idx] = self.lineages[cur_walker_idx].union(self.lineages[parent_walker_idx])
                        for site in self.lineages[cur_walker_idx]:
                            site_idx = self.width*site[0] + site[1]
                            self.lineages[site_idx] = self.lineages[cur_walker_idx]

                        for site in self.lineages[parent_walker_idx]:
                            site_idx = self.width*site[0] + site[1]
                            if type(self.lineages[site_idx]) == set:
                                self.lineages[site_idx] = self.lineages[cur_walker_idx]

                    # Handle case when walker has moved back into its own lineage
                    else:
                        # Change position of active walker
                        if parent_walker not in self.rand_walker_pos:
                            self.rand_walker_pos.append(parent_walker)
                # Let speciation occur
                elif type(self.lineages[parent_walker_idx]) == int:
                    self.species[self.lineages[parent_walker_idx]] = self.species[self.lineages[parent_walker_idx]].union(self.lineages[cur_walker_idx])
                    for site in self.lineages[cur_walker_idx]:
                        site_idx = self.width*site[0] + site[1]
                        self.lineages[site_idx] = self.lineages[parent_walker_idx]

            # Remove current walker from list of active walkers
            self.rand_walker_pos.remove(cur_walker)
