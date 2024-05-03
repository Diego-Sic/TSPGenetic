import argparse
import math
import random
import matplotlib.pyplot as plt
import numpy as np

class TravelingSalesperson:
    def __init__(self, num_cities, num_iterations, mutation_rate, city_bounds=(-10, 10)):
        self.num_cities = num_cities
        self.num_iterations = num_iterations
        self.mutation_rate = mutation_rate
        self.city_bounds = city_bounds
        self.cities = self.create_cities()
        self.population_size = 100
        self.best_route = None
        self.best_fitness = None
        self.best_fitness_history = []

    def create_cities(self):
        min_x, max_x = self.city_bounds
        min_y, max_y = self.city_bounds
        cities = []

        # Create the first city randomly
        x = random.uniform(min_x, max_x)
        y = random.uniform(min_y, max_y)
        cities.append((x, y))

        # Create the remaining cities by connecting them to the previous city
        for i in range(1, self.num_cities):
            distance = random.uniform(1, 5)  # Adjust the distance range as needed
            angle = random.uniform(0, 2 * math.pi)
            x = cities[-1][0] + distance * math.cos(angle)
            y = cities[-1][1] + distance * math.sin(angle)
            cities.append((x, y))

        return np.array(cities)

    def distance(self, city1, city2):
        return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

    def fitness(self, route):
        total_distance = 0
        for i in range(len(route)):
            total_distance += self.distance(self.cities[route[i]], self.cities[route[(i + 1) % len(route)]])
        return 1 / total_distance

    def generate_population(self):
        population = []
        for i in range(self.population_size):
            route = list(range(self.num_cities))
            random.shuffle(route)
            population.append(route)
        return population

    def crossover(self, parent1, parent2):
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
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(route)), 2)
            route[idx1], route[idx2] = route[idx2], route[idx1]
        return route

    def run_genetic_algorithm(self):
        population = self.generate_population()

        for generation in range(self.num_iterations):
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
        x_coords = self.cities[:, 0]
        y_coords = self.cities[:, 1]
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

    def plot_initial_cities(self):
        x_coords = self.cities[:, 0]
        y_coords = self.cities[:, 1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(x_coords, y_coords, color='b', label='Cities')
        ax.plot(x_coords, y_coords, color='r', label='Initial Path')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Initial Connected Cities')
        ax.legend()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Traveling Salesperson Problem")
    parser.add_argument("num_cities", type=int, help="Number of cities")
    parser.add_argument("num_iterations", type=int, help="Number of iterations")
    parser.add_argument("mutation_rate", type=float, help="Probability of random mutations")
    args = parser.parse_args()

    tsp = TravelingSalesperson(args.num_cities, args.num_iterations, args.mutation_rate, city_bounds=(-20, 20))
    tsp.plot_initial_cities()
    tsp.run_genetic_algorithm()

    print(f"Best route: {tsp.best_route}")
    print(f"Best fitness (reciprocal of total distance): {tsp.best_fitness}")

    tsp.plot_fitness_history()
    tsp.plot_cities_and_route()