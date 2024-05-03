import argparse
import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Define the distance function
def distance(city1, city2):
    # calculate the distance between two points
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

# Define the fitness function
def fitness(route, cities):
    total_distance = 0
    for i in range(len(route)):
        total_distance += distance(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return 1 / total_distance

# Generate an initial population
def generate_population(population_size, cities):
    population = []
    for i in range(population_size):
        route = list(range(len(cities)))
        random.shuffle(route)
        population.append(route)
    return population

# Perform crossover
def crossover(parent1, parent2):
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

# Perform mutation
def mutate(route, mutation_rate):
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route

# Genetic algorithm
def genetic_algorithm(num_cities, num_iterations, mutation_rate):
    # Generate random cities
    cities = np.random.normal(0, 1, (num_cities, 2))

    population_size = 100
    population = generate_population(population_size, cities)
    best_fitness_history = []

    for generation in range(num_iterations):
        fitness_values = [fitness(route, cities) for route in population]
        best_route_index = fitness_values.index(max(fitness_values))
        best_route = population[best_route_index]
        best_fitness = fitness_values[best_route_index]
        best_fitness_history.append(best_fitness)

        new_population = [best_route]
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(population, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        population = new_population

    best_fitness_values = [fitness(route, cities) for route in population]
    best_route_index = best_fitness_values.index(max(best_fitness_values))
    best_route = population[best_route_index]
    best_fitness = best_fitness_values[best_route_index]

    return best_route, best_fitness, best_fitness_history, cities

def plot_fitness_history(best_fitness_history):
    """
    Plot the fitness history of the genetic algorithm, showing improvement over generations.
    
    Args:
    best_fitness_history: List of fitness values (reciprocal of total distance) for the best route in each generation.
    """
    # Calculate the number of generations
    generations = range(len(best_fitness_history))
    
    # Convert fitness values to distances for a more intuitive plot (higher is better)
    distances = [1 / fitness for fitness in best_fitness_history]
    
    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(generations, distances, marker='o', linestyle='-', color='b')
    plt.title('Solution Quality Over Time')
    plt.xlabel('Generation')
    plt.ylabel('Total Distance')
    plt.grid(True)
    plt.show()

def plot_cities_and_route(cities, best_route):
    """
    Plot the cities and the best route found by the genetic algorithm.
    
    Args:
    cities: NumPy array of city coordinates.
    best_route: List representing the best route as indices of cities.
    """
    # Extract the coordinates of the cities
    x_coords = cities[:, 0]
    y_coords = cities[:, 1]
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the cities
    ax.scatter(x_coords, y_coords, color='b', label='Cities')
    
    # Plot the best route
    x_route = [x_coords[idx] for idx in best_route]
    y_route = [y_coords[idx] for idx in best_route]
    x_route.append(x_route[0])  # Connect the route back to the start
    y_route.append(y_route[0])
    ax.plot(x_route, y_route, color='r', label='Best Route')
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Cities and Best Route')
    ax.legend()
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Traveling Salesperson Problem")
    parser.add_argument("num_cities", type=int, help="Number of cities")
    parser.add_argument("num_iterations", type=int, help="Number of iterations")
    parser.add_argument("mutation_rate", type=float, help="Probability of random mutations")
    args = parser.parse_args()

    best_route, best_fitness, best_fitness_history, cities = genetic_algorithm(args.num_cities, args.num_iterations, args.mutation_rate)

    print(f"Best route: {best_route}")
    print(f"Best fitness (reciprocal of total distance): {best_fitness}")

    plot_fitness_history(best_fitness_history)
    plot_cities_and_route(cities, best_route)