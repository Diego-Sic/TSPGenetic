import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
from graph import Graph

def make_random_gaussian_graph(num_nodes=100):
    '''Function borrowed from Prof Bailey'''
    rng = np.random.default_rng(seed=42)
    cities = rng.normal(0, 1, (num_nodes, 2))
    dists = np.full(shape=(num_nodes, num_nodes), fill_value=0.0)

    for i in range(len(cities)-1):
        for j in range(i+1, len(cities)):
            dists[i, j] = np.linalg.norm(cities[i]-cities[j])

    dists += np.transpose(dists)
    return Graph([f"v{i}" for i in range(len(cities))], dists, city_locations=cities)

class TravelingSalesperson:
    def __init__(self, num_cities, num_iterations, mutation_rate):
        self.num_cities = num_cities
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.graph = make_random_gaussian_graph(num_cities)
        self.population_size = 100
        self.best_route = None
        self.best_fitness = None
        self.best_fitness_history = []

    def fitness(self, route):
        '''We calculate how good something based on the inverse of the total weight
        as the number is bigger we have a better candidates'''
        return 1 / self.graph.get_tour_weight(route)

    def generate_population(self):
        '''Create a random selection of routes'''
        population = []
        for i in range(self.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(route)
        return population

    def crossover(self, parent1, parent2):
        """        
        We selecting a random segment from the first parent
        and filling the rest of the route with non-duplicated cities from the second parent
        """
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start + 1, len(parent1))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child]
        index = 0
        for i in range(len(child)):
            if child[i] is None:
                child[i] = remaining[index]
                index += 1
        return child

    def mutate(self, route):
        """
        This mutation allows the genetic algorithm to explore a wider variety of routes and
        helps to avoid local minima by introducing randomness.
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    def run_genetic_algorithm(self):
        population = self.generate_population()

        for _ in range(self.num_iterations):
            fitness_values = [self.fitness(route) for route in population]
            best_route_index = fitness_values.index(max(fitness_values))
            best_route = population[best_route_index]
            best_fitness = fitness_values[best_route_index]
            self.best_fitness_history.append(best_fitness)

            new_population = [best_route]
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(population, 2)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population

        best_fitness_values = [self.fitness(route) for route in population]
        best_route_index = best_fitness_values.index(max(best_fitness_values))
        self.best_route = population[best_route_index]
        self.best_fitness = best_fitness_values[best_route_index]

    def plot_fitness_history(self):
        generations = range(len(self.best_fitness_history))
        distances = [1 / fitness for fitness in self.best_fitness_history]
        plt.figure(figsize=(10, 5))
        plt.plot(generations, distances, marker='o', linestyle='-', color='b')
        plt.title('Solution Quality Over Time')
        plt.xlabel('Generation')
        plt.ylabel('Total Distance')
        plt.grid(True)
        plt.show()

    def plot_cities_and_route(self):
        cities = self.graph.get_node_locations()
        x_coords = cities[:, 0]
        y_coords = cities[:, 1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_coords, y_coords, color='b', label='Cities')
        x_route = [x_coords[idx] for idx in self.best_route] + [x_coords[self.best_route[0]]]
        y_route = [y_coords[idx] for idx in self.best_route] + [y_coords[self.best_route[0]]]
        ax.plot(x_route, y_route, color='r', label='Best Route')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Cities and Best Route')
        ax.legend()
        plt.show()

    def plot_initial_graph(self):
        cities = self.graph.get_node_locations()
        x_coords = cities[:, 0]
        y_coords = cities[:, 1]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x_coords, y_coords, color='blue', label='Cities')


        max_dist = np.max(self.graph.weights)

        # Draw edges between each pair of nodes
        for i in range(self.num_cities):
            for j in range(i + 1, self.num_cities):
                distance = self.graph.weight_from_i_to_j(i, j)
                alpha = 1 - (distance / max_dist)
                ax.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], 'gray')

        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_title('Initial Graph of Cities')
        ax.legend()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Traveling Salesperson Problem")
    parser.add_argument("num_cities", type=int, help="Number of cities")
    parser.add_argument("num_iterations", type=int, help="Number of iterations")
    parser.add_argument("mutation_rate", type=float, help="Probability of random mutations")
    args = parser.parse_args()

    tsp = TravelingSalesperson(args.num_cities, args.num_iterations, args.mutation_rate)
    tsp.plot_initial_graph()
    tsp.run_genetic_algorithm()
    print(f"Best route: {tsp.best_route}")
    print(f"Best fitness (reciprocal of total distance): {tsp.best_fitness}")
    tsp.plot_fitness_history()
    tsp.plot_cities_and_route()

if __name__ == "__main__":
    main()
