import numpy as np
class Graph:
    def __init__(self, node_names: list, weights: np.ndarray, city_locations=None):
        assert len(weights) == len(weights[0]) == len(node_names), 'Size mismatch!'
        self.node_names = node_names
        self.nodes = list(range(len(node_names)))
        self.weights = weights
        self.city_locations = city_locations

    def weight_from_i_to_j(self, i: int, j: int) -> int:
        return self.weights[i][j]

    def get_tour_weight(self, tour):
        return sum(self.weights[tour[i-1]][tour[i]] for i in range(1, len(tour))) + self.weights[tour[-1]][tour[0]]

    def get_node_locations(self) -> np.ndarray:
        return self.city_locations

    def get_city_name_for_index(self, i):
        return self.node_names[i]

    def generate_nna_tour(self, start_index=0) -> list:
        N = len(self.nodes)
        unvisited = set(self.nodes)
        tour = [start_index]
        unvisited.remove(start_index)
        current_index = start_index

        while unvisited:
            next_index = min(unvisited, key=lambda x: self.weights[current_index][x])
            tour.append(next_index)
            unvisited.remove(next_index)
            current_index = next_index

        tour.append(start_index)  # To complete the loop
        return tour
