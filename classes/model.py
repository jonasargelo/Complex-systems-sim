"""
Complex System Simulation Group Project
"""


from classes.island import Island

import itertools
import matplotlib.pyplot as plt
import numpy as np

class Model:
    """
    Creates a model with multiple islands.
    """

    def __init__(self, island_grid_size, total_grid, iterations, distance=-1, no_islands=2, alpha=3e-4) -> None:
        self.no_islands = no_islands
        self.grid_size = island_grid_size
        self.total_grid_size = total_grid
        self.iterations = iterations
        self.islands = self.create_islands()
        self.alpha = alpha
        self.counter = 0
        self.fixed_distance = distance

    def create_islands(self) -> list:
        """
        Create the number of islands desired.
        """
        new_islands = list()
        half_size = int(self.grid_size / 2) + 1

        for i in range(self.no_islands):
            x_coordinate = np.random.randint(half_size, self.total_grid_size - half_size)
            y_coordinate = np.random.randint(half_size, self.total_grid_size - half_size)
            if self.check_location(x_coordinate, y_coordinate, half_size, new_islands) and i > 0:
                continue
            new_islands.append(Island(x_coordinate, y_coordinate, self.grid_size, self.iterations))
        #print(len(new_islands))
        return new_islands

    def check_location(self, x, y, size, islands) -> bool:
        for island in islands:
            if island.get_distance([x, y]) < size:
                return True
        return False

    def simple_prob(self, distance) -> float:
        """
        Simple probability for testing.
        """
        return 0.5 / (distance)

    def interaction_prob(self, distance) -> float:
        pass

    def interaction(self, i, random_index) -> None:
        """
        Performs an interaction (migration) between two islands. It replaces the species
        living on a random square on the other island.
        """
        unique_values, counts = np.unique(self.islands[i].algorithm.current_grid.ravel(), return_counts=True)
        max_count = np.argmax(counts)
        most = unique_values[max_count]

        # Choose random coordinate to migrate to
        x = np.random.choice(np.arange(0, self.islands[random_index].width))
        y = np.random.choice(np.arange(0, self.islands[random_index].height))
        self.islands[random_index].algorithm.current_grid[x, y] = most

    def run(self) -> None:

        for i in range(self.iterations):
            # For every island perform the single island dynamics
            for island in self.islands:
                island.step(self.alpha, i)

            # # Loop through every island to see if migration takes place
            length_islands = len(self.islands)
            for i in range(length_islands):

                # Get random index from other island
                indices = [j for j in range(length_islands) if j != i]
                random_index = np.random.choice(indices)

                # Calculate distance towards other island and calculate probability to migrate
                if self.fixed_distance > 0:
                    prob = self.simple_prob(self.fixed_distance)
                else:
                    distance = self.islands[i].get_distance([self.islands[random_index].get_coordinates()])
                    prob = self.simple_prob(distance)

                # Check if migrations takes place and perform the interaction
                if prob > np.random.uniform():
                    self.counter += 1
                    self.interaction(i, random_index)


    def new_step(self):
        '''Run experiment with the voter model
        Inputs:
        grid_0 (numpy array): Initial grid
        alpha (float): Value of alpha parameter
        MC_steps (int): number of monte carlo steps

        Returns:
        cur_grid (numpy array): Grid after n_iter iterations
        num_species (list): Contains amount of different species at each tenth iteration
        '''

        for i in range(self.iterations):
            for island in self.islands:
                island.new_step(self.alpha)

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
                    self.interaction(i, random_index)

    def visualise(self, plot_island) -> None:
        # fig, (ax1, ax2) = plt.subplots(2, 4, figsize=(7, 7))

        # ax1.imshow(list(self.islands[0].algorithm.starting_grid, cmap='hot'))
        # ax1.set_title('Initial grid')

        # ax2.imshow(list(self.islands[0].algorithm.current_grid))
        # #ax2.set_title(f'Grid after {n_iters} iterations')

        # fig.show()
        plt.imshow(plot_island.algorithm.starting_grid)
        plt.show()
        plt.imshow(plot_island.algorithm.current_grid)
        plt.show()

    def sa_curve(self, grid) -> None:
        height, width = grid.shape

        n_centers = 10
        centers_x = np.random.choice(np.arange(0, width), n_centers) + width
        centers_y = np.random.choice(np.arange(0, height), n_centers) + height

        areas = []
        species = []

        torus_grid = np.vstack((grid, grid, grid))
        torus_grid = np.hstack((torus_grid, torus_grid, torus_grid))

        torus_grid.shape

        for i, (x, y) in enumerate(zip(centers_x, centers_y)):
            cur_species = []
            for j in range(width//2):
                cur_species.append(len(np.unique(torus_grid[x-j:x+j+1, y-j:y+j+1])))
                if i == 0:
                    areas.append((j+1)**2)
            species.append(cur_species)

        spec_std_dev = np.std(species, axis=0)
        spec_mean = np.mean(species, axis=0)

        poly_coeffs = np.polyfit(np.log(areas), np.log(spec_mean), 1)
        print(poly_coeffs)

        plt.loglog(areas, spec_mean, label='Mean of 10 centers')
        plt.loglog([areas[0], areas[-1]],
                np.exp(poly_coeffs[1]) * np.array([areas[0], areas[-1]])**poly_coeffs[0],
                color='grey',
                linestyle='dashed',
                label=f'Lin. regress, power={round(poly_coeffs[0], 2)}')
        plt.fill_between(areas, spec_mean-spec_std_dev, spec_mean+spec_std_dev, alpha=0.2, label='Std. dev')
        plt.ylabel('Log Number of Species')
        plt.xlabel('Log Area')
        plt.title(f'Species area curve for L={self.grid_size}')
        plt.legend()
        plt.show()
