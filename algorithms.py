"""
TSP Algorithm implementations
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import KMeans
from utils import SpatialIndex, euclidean_distance_matrix, tour_length
import random

class TSPAlgorithm:
    """Base class for TSP algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        """Solve TSP and return ordered tour"""
        raise NotImplementedError
        
    def get_info(self) -> str:
        """Get algorithm information"""
        return f"Algorithm: {self.name}"

class NearestNeighborTSP(TSPAlgorithm):
    """Greedy nearest neighbor TSP solver"""
    
    def __init__(self):
        super().__init__("Nearest Neighbor")
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 1:
            return points
            
        n = len(points)
        unvisited = set(range(1, n))
        tour = [0]
        current = 0
        
        while unvisited:
            distances = euclidean_distance_matrix(points[current:current+1], points)[0]
            distances[list(set(range(n)) - unvisited)] = np.inf
            
            next_city = np.argmin(distances)
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            
        return points[tour]

class TwoOptTSP(TSPAlgorithm):
    """2-opt local search TSP solver"""
    
    def __init__(self, max_iterations: int = 1000):
        super().__init__("2-Opt")
        self.max_iterations = max_iterations
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 3:
            return points
            
        # Start with nearest neighbor solution
        nn_solver = NearestNeighborTSP()
        tour = nn_solver.solve(points)
        
        # Get indices of the tour
        tour_indices = []
        for point in tour:
            idx = np.where(np.all(points == point, axis=1))[0][0]
            tour_indices.append(idx)
            
        best_tour = tour_indices[:]
        best_distance = self._tour_distance(points, best_tour)
        
        improved = True
        iteration = 0
        
        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1
            
            for i in range(len(best_tour)):
                for j in range(i + 2, len(best_tour)):
                    if j - i == len(best_tour) - 1:  # Skip if adjacent in circular tour
                        continue
                        
                    new_tour = self._two_opt_swap(best_tour, i, j)
                    new_distance = self._tour_distance(points, new_tour)
                    
                    if new_distance < best_distance:
                        best_tour = new_tour
                        best_distance = new_distance
                        improved = True
                        
        return points[best_tour]
    
    def _two_opt_swap(self, tour: List[int], i: int, j: int) -> List[int]:
        """Perform 2-opt swap"""
        new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
        return new_tour
    
    def _tour_distance(self, points: np.ndarray, tour_indices: List[int]) -> float:
        """Calculate tour distance"""
        tour_points = points[tour_indices]
        return tour_length(tour_points)

class GeneticTSP(TSPAlgorithm):
    """Genetic algorithm TSP solver"""
    
    def __init__(self, population_size: int = 100, generations: int = 50,
                 mutation_rate: float = 0.02, elite_size: int = 20):
        super().__init__("Genetic Algorithm")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 3:
            return points
            
        n = len(points)
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = list(range(n))
            random.shuffle(individual)
            population.append(individual)
            
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                distance = self._tour_distance(points, individual)
                fitness = 1 / (1 + distance)  # Higher fitness for shorter tours
                fitness_scores.append(fitness)
                
            # Selection and reproduction
            new_population = []
            
            # Elite selection
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx][:])
                
            # Generate offspring
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)
                
            population = new_population
            
        # Return best solution
        fitness_scores = [1 / (1 + self._tour_distance(points, ind)) for ind in population]
        best_idx = np.argmax(fitness_scores)
        best_tour = population[best_idx]
        
        return points[best_tour]
    
    def _tournament_selection(self, population: List[List[int]], 
                            fitness_scores: List[float], tournament_size: int = 3) -> List[int]:
        """Tournament selection"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx][:]
    
    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX)"""
        n = len(parent1)
        start, end = sorted(random.sample(range(n), 2))
        
        child = [-1] * n
        child[start:end] = parent1[start:end]
        
        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in child:
                child[pointer % n] = city
                pointer += 1
                
        return child
    
    def _mutate(self, individual: List[int]) -> List[int]:
        """Swap mutation"""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual
    
    def _tour_distance(self, points: np.ndarray, tour_indices: List[int]) -> float:
        """Calculate tour distance"""
        tour_points = points[tour_indices]
        return tour_length(tour_points)

class SimulatedAnnealingTSP(TSPAlgorithm):
    """Simulated annealing TSP solver"""
    
    def __init__(self, initial_temp: float = 1000, cooling_rate: float = 0.995,
                 min_temp: float = 1e-8):
        super().__init__("Simulated Annealing")
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        if len(points) <= 3:
            return points
            
        n = len(points)
        
        # Start with random tour
        current_tour = list(range(n))
        random.shuffle(current_tour)
        current_distance = self._tour_distance(points, current_tour)
        
        best_tour = current_tour[:]
        best_distance = current_distance
        
        temperature = self.initial_temp
        
        while temperature > self.min_temp:
            # Generate neighbor by swapping two random cities
            new_tour = current_tour[:]
            i, j = random.sample(range(n), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            
            new_distance = self._tour_distance(points, new_tour)
            delta = new_distance - current_distance
            
            # Accept or reject the new solution
            if delta < 0 or random.random() < np.exp(-delta / temperature):
                current_tour = new_tour
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_tour = current_tour[:]
                    best_distance = current_distance
                    
            temperature *= self.cooling_rate
            
        return points[best_tour]
    
    def _tour_distance(self, points: np.ndarray, tour_indices: List[int]) -> float:
        """Calculate tour distance"""
        tour_points = points[tour_indices]
        return tour_length(tour_points)

class AssociationTSP(TSPAlgorithm):
    """Association-based iterative TSP solver with adaptive vertex management"""
    
    def __init__(self, n_vertices: int = 120, max_iterations: int = 100, 
                 adaptive_vertices: bool = True, subdivision_threshold: float = 0.05):
        super().__init__("Association")
        self.n_vertices = n_vertices
        self.max_iterations = max_iterations
        self.adaptive_vertices = adaptive_vertices
        self.subdivision_threshold = subdivision_threshold
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        from utils import (init_circular_loop, adaptive_smooth_loop, smooth_loop,
                          calculate_adaptive_vertex_count, subdivide_vertices,
                          remove_redundant_vertices, optimize_vertex_distribution)
        from config import Config
        
        if len(points) <= 3:
            return points
        
        # Calculate adaptive vertex count if enabled
        if self.adaptive_vertices:
            self.n_vertices = calculate_adaptive_vertex_count(
                points, 
                min_vertices=getattr(Config, 'MIN_VERTICES', 60),
                max_vertices=getattr(Config, 'MAX_VERTICES', 300),
                density_factor=getattr(Config, 'VERTEX_DENSITY_FACTOR', 0.4)
            )
            print(f"Using adaptive vertex count: {self.n_vertices}")
            
        # Initialize loop
        vertices = init_circular_loop(self.n_vertices)
        
        # Iterative refinement
        for iteration in range(self.max_iterations):
            # Assign points to nearest vertices
            if len(points) > 0:
                try:
                    spatial_index = SpatialIndex(vertices)
                    _, nearest_indices = spatial_index.query_nearest(points)
                    
                    # Update vertices based on assigned points
                    new_vertices = vertices.copy()
                    for v in range(len(vertices)):
                        assigned_mask = (nearest_indices == v)
                        if np.any(assigned_mask):
                            assigned_points = points[assigned_mask]
                            centroid = np.mean(assigned_points, axis=0)
                            
                            # Adaptive move rate that decreases over time
                            move_rate = 0.3 * (1 - iteration / self.max_iterations)
                            move_rate = max(move_rate, 0.01)  # Minimum move rate
                            
                            new_vertices[v] = vertices[v] * (1 - move_rate) + centroid * move_rate
                    
                    vertices = new_vertices
                    
                    # Apply enhanced smoothing with multiple iterations
                    smoothing_iterations = getattr(Config, 'SMOOTHING_ITERATIONS', 2)
                    smooth_rate = 0.4 * (1 - iteration / self.max_iterations)
                    smooth_rate = max(smooth_rate, 0.1)  # Minimum smooth rate
                    
                    # Use adaptive smoothing for better curve handling
                    for _ in range(smoothing_iterations):
                        vertices = adaptive_smooth_loop(vertices, smooth_rate * 0.5, curvature_weight=0.3)
                    
                    # Periodically optimize vertex distribution
                    if iteration % 20 == 0 and iteration > 0:
                        # Subdivide long edges for finer resolution
                        vertices = subdivide_vertices(vertices, self.subdivision_threshold)
                        
                        # Remove redundant vertices that are too close
                        vertices = remove_redundant_vertices(vertices, min_distance=0.008)
                        
                        # Optimize vertex distribution for uniform spacing
                        vertices = optimize_vertex_distribution(vertices)
                        
                        print(f"Iteration {iteration}: Optimized to {len(vertices)} vertices")
                    
                    # Ensure vertices stay within bounds
                    vertices = np.clip(vertices, 0.02, 0.98)
                    
                except Exception as e:
                    print(f"Error in association iteration {iteration}: {e}")
                    break
                
        return vertices

class ClusteringTSP(TSPAlgorithm):
    """K-means clustering based TSP solver with interpolated fine vertices"""
    
    def __init__(self, n_clusters: int = 40, n_interpolated_vertices: int = 120):
        super().__init__("K-means Clustering")
        self.n_clusters = n_clusters
        self.n_interpolated_vertices = n_interpolated_vertices
        
    def solve(self, points: np.ndarray) -> np.ndarray:
        from utils import resample_curve
        
        if len(points) <= 3:
            return points
            
        # Perform k-means clustering
        n_clusters = min(self.n_clusters, len(points))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_
        
        # Order centers using nearest neighbor
        nn_solver = NearestNeighborTSP()
        ordered_centers = nn_solver.solve(centers)
        
        # Interpolate more vertices for finer resolution
        if len(ordered_centers) >= 3:
            # Resample the curve to have more vertices
            fine_vertices = resample_curve(ordered_centers, self.n_interpolated_vertices)
            return fine_vertices
        
        return ordered_centers

def get_algorithm(algorithm_name: str, **kwargs) -> TSPAlgorithm:
    """Factory function to get TSP algorithm by name"""
    
    algorithms = {
        'nearest_neighbor': NearestNeighborTSP,
        'two_opt': TwoOptTSP,
        'genetic': GeneticTSP,
        'simulated_annealing': SimulatedAnnealingTSP,
        'association': AssociationTSP,
        'clustering': ClusteringTSP
    }
    
    if algorithm_name not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
    return algorithms[algorithm_name](**kwargs)
