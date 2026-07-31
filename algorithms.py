"""
TSP Algorithm implementations

Every solver exposes the same three entry points:

``solve_tour(points)``
    The actual answer to the TSP: an ordering of the *input point indices*.
``solve_loop(points)``
    The geometry to draw. For the permutation solvers this is just the tour
    laid out in order; for the association and clustering solvers it is the
    continuous loop they fit through the data.
``evaluate(points)``
    Both of the above plus the tour length and wall-clock runtime, bundled in
    a :class:`TSPSolution`.

Keeping ``solve_tour`` separate from ``solve_loop`` is what makes the solvers
comparable. A smoothed loop of 120 vertices threaded through 40 data points is
shorter than any tour of those points, so measuring the loop instead of the
tour it induces would flatter the loop-based solvers in every comparison.
"""

import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from sklearn.cluster import KMeans

from utils import (SpatialIndex, adaptive_smooth_loop, calculate_adaptive_vertex_count,
                   init_circular_loop, optimize_vertex_distribution,
                   order_points_along_loop, remove_redundant_vertices, resample_curve,
                   subdivide_vertices, tour_length, tour_length_of_indices, two_opt_refine)

logger = logging.getLogger(__name__)


@dataclass
class TSPSolution:
    """The result of running a solver on a set of points"""

    algorithm: str
    points: np.ndarray
    tour: np.ndarray
    length: float
    runtime: float
    loop: Optional[np.ndarray] = None
    loop_length: Optional[float] = None
    metadata: dict = field(default_factory=dict)

    @property
    def tour_points(self) -> np.ndarray:
        """The data points in visiting order"""
        return self.points[self.tour]


def nearest_neighbor_tour(points: np.ndarray, start: int = 0) -> np.ndarray:
    """Greedy nearest-neighbour tour construction.

    Returns an ordering of point indices, so duplicate coordinates stay
    distinguishable (matching points by value would collapse them).
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n <= 2:
        return np.arange(n)
    if not 0 <= start < n:
        raise ValueError(f"start index {start} is out of range for {n} points")

    visited = np.zeros(n, dtype=bool)
    tour = np.empty(n, dtype=int)

    current = int(start)
    visited[current] = True
    tour[0] = current

    for step in range(1, n):
        distances = np.linalg.norm(points - points[current], axis=1)
        distances[visited] = np.inf
        current = int(np.argmin(distances))
        visited[current] = True
        tour[step] = current

    return tour


class TSPAlgorithm:
    """Base class for TSP algorithms"""

    #: True when the solver fits a continuous loop rather than permuting points.
    produces_loop = False

    def __init__(self, name: str, seed: Optional[int] = None):
        self.name = name
        self.seed = seed
        self._random = random.Random(seed)
        self._rng = np.random.default_rng(seed)

    def reseed(self, seed: Optional[int]) -> None:
        """Reset the solver's random streams"""
        self.seed = seed
        self._random = random.Random(seed)
        self._rng = np.random.default_rng(seed)

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        """Return an ordering of point indices visiting every point once.

        The base implementation is a greedy nearest-neighbour construction;
        subclasses override it with their own strategy.
        """
        return nearest_neighbor_tour(np.asarray(points, dtype=float))

    def solve_loop(self, points: np.ndarray) -> np.ndarray:
        """Return the geometry to draw, as an ordered ``(k, 2)`` array"""
        points = np.asarray(points, dtype=float)
        return points[self.solve_tour(points)]

    def solve(self, points: np.ndarray) -> np.ndarray:
        """Solve TSP and return the ordered array of 2-D coordinates.

        Args:
            points: Array of 2D points

        Returns:
            Ordered array of points representing the tour (or, for loop-based
            solvers, the fitted loop).
        """
        return self.solve_loop(points)

    def evaluate(self, points: np.ndarray, refine: bool = False) -> TSPSolution:
        """Solve and report the tour, the drawable loop and the timings.

        Args:
            points: ``(n, 2)`` array of data points.
            refine: polish the resulting tour with 2-opt before reporting.
        """
        points = np.asarray(points, dtype=float)

        start = time.perf_counter()
        loop = self.solve_loop(points) if self.produces_loop else None
        tour = (order_points_along_loop(points, loop)
                if loop is not None and len(points) > 2
                else self.solve_tour(points))
        if refine and len(points) > 3:
            tour = two_opt_refine(points, tour)
        runtime = time.perf_counter() - start

        if loop is None:
            loop = points[tour] if len(points) else points

        return TSPSolution(
            algorithm=self.name,
            points=points,
            tour=np.asarray(tour, dtype=int),
            length=tour_length_of_indices(points, tour),
            runtime=runtime,
            loop=loop,
            loop_length=tour_length(loop),
            metadata={'refined': bool(refine), 'seed': self.seed},
        )

    def get_info(self) -> str:
        """Get algorithm information"""
        return f"Algorithm: {self.name}"


class NearestNeighborTSP(TSPAlgorithm):
    """Greedy nearest neighbour TSP solver"""

    def __init__(self, start: int = 0, seed: Optional[int] = None):
        super().__init__("Nearest Neighbor", seed=seed)
        self.start = start

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) <= 2:
            return np.arange(len(points))
        return nearest_neighbor_tour(points, start=min(self.start, len(points) - 1))


class TwoOptTSP(TSPAlgorithm):
    """2-opt local search TSP solver seeded with a nearest-neighbour tour"""

    def __init__(self, max_iterations: int = 1000, seed: Optional[int] = None):
        super().__init__("2-Opt", seed=seed)
        self.max_iterations = max_iterations

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) <= 3:
            return np.arange(len(points))

        tour = nearest_neighbor_tour(points)
        return two_opt_refine(points, tour, max_passes=self.max_iterations)


class GeneticTSP(TSPAlgorithm):
    """Genetic algorithm TSP solver using order crossover and swap mutation"""

    def __init__(self, population_size: int = 100, generations: int = 50,
                 mutation_rate: float = 0.02, elite_size: int = 20,
                 tournament_size: int = 3, seed: Optional[int] = None):
        super().__init__("Genetic Algorithm", seed=seed)
        if population_size < 2:
            raise ValueError("population_size must be at least 2")
        self.population_size = population_size
        self.generations = max(0, generations)
        self.mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))
        # An elite larger than the population would leave no room for
        # offspring and stall the search.
        self.elite_size = int(np.clip(elite_size, 1, population_size - 1))
        self.tournament_size = int(np.clip(tournament_size, 1, population_size))

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        n = len(points)
        if n <= 3:
            return np.arange(n)

        # Seeding one greedy tour gives the search a sane starting point
        # instead of asking it to find structure in pure noise.
        population = [list(nearest_neighbor_tour(points))]
        for _ in range(self.population_size - 1):
            individual = list(range(n))
            self._random.shuffle(individual)
            population.append(individual)

        best_tour = min(population, key=lambda ind: self._tour_distance(points, ind))
        best_distance = self._tour_distance(points, best_tour)

        for _ in range(self.generations):
            distances = [self._tour_distance(points, ind) for ind in population]
            fitness_scores = [1.0 / (1.0 + d) for d in distances]

            ranked = np.argsort(distances)
            if distances[ranked[0]] < best_distance:
                best_distance = distances[ranked[0]]
                best_tour = population[ranked[0]][:]

            # Elite selection keeps the best tours intact.
            new_population = [population[idx][:] for idx in ranked[:self.elite_size]]

            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child))

            population = new_population

        for individual in population:
            distance = self._tour_distance(points, individual)
            if distance < best_distance:
                best_distance = distance
                best_tour = individual[:]

        return np.asarray(best_tour, dtype=int)

    def _tournament_selection(self, population: List[List[int]],
                              fitness_scores: List[float]) -> List[int]:
        """Tournament selection"""
        size = min(self.tournament_size, len(population))
        contenders = self._random.sample(range(len(population)), size)
        winner = max(contenders, key=lambda idx: fitness_scores[idx])
        return population[winner][:]

    def _crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover (OX)"""
        n = len(parent1)
        start, end = sorted(self._random.sample(range(n), 2))

        child = [-1] * n
        child[start:end] = parent1[start:end]
        # A set membership test keeps the fill loop linear; scanning the
        # partially built child for every city made crossover quadratic.
        taken = set(parent1[start:end])

        pointer = end
        for city in parent2[end:] + parent2[:end]:
            if city not in taken:
                child[pointer % n] = city
                taken.add(city)
                pointer += 1

        return child

    def _mutate(self, individual: List[int]) -> List[int]:
        """Swap mutation applied per position.

        Rolling once per individual meant a 2% rate produced a swap in only
        one child in fifty, far too little churn to escape a local optimum.
        Rolling per position makes the rate mean what the docs say it does.
        """
        n = len(individual)
        if n < 2:
            return individual

        for i in range(n):
            if self._random.random() < self.mutation_rate:
                j = self._random.randrange(n)
                individual[i], individual[j] = individual[j], individual[i]
        return individual

    def _tour_distance(self, points: np.ndarray, tour_indices: Sequence[int]) -> float:
        """Calculate tour distance"""
        return tour_length_of_indices(points, tour_indices)


class SimulatedAnnealingTSP(TSPAlgorithm):
    """Simulated annealing TSP solver using 2-opt segment reversals"""

    def __init__(self, initial_temp: Optional[float] = None, cooling_rate: float = 0.995,
                 min_temp: Optional[float] = None,
                 iterations_per_temp: Optional[int] = None,
                 seed: Optional[int] = None):
        super().__init__("Simulated Annealing", seed=seed)
        if not 0.0 < cooling_rate < 1.0:
            raise ValueError("cooling_rate must be in (0, 1)")
        if initial_temp is not None and initial_temp <= 0:
            raise ValueError("initial_temp must be positive")
        if min_temp is not None and min_temp <= 0:
            raise ValueError("min_temp must be positive")

        #: ``None`` calibrates the schedule to the data (see :meth:`_schedule`).
        self.initial_temp = None if initial_temp is None else float(initial_temp)
        self.cooling_rate = float(cooling_rate)
        self.min_temp = None if min_temp is None else float(min_temp)
        self.iterations_per_temp = (None if iterations_per_temp is None
                                    else max(1, int(iterations_per_temp)))

    def _schedule(self, points: np.ndarray, tour: Sequence[int]):
        """Resolve the cooling schedule, calibrating to the data if needed.

        A fixed temperature is meaningless without knowing the scale of the
        coordinates: on the unit square a tour edge is ~0.05 long, so a
        temperature of 1000 accepts every proposal and degenerates into a
        random walk. Anchoring the start temperature to the mean edge length
        makes the schedule behave the same on any coordinate scale.
        """
        n = len(tour)
        mean_edge = tour_length_of_indices(points, tour) / max(n, 1)

        initial = self.initial_temp
        if initial is None:
            initial = max(mean_edge, 1e-9)

        minimum = self.min_temp
        if minimum is None:
            minimum = initial * 1e-4

        per_temp = self.iterations_per_temp
        if per_temp is None:
            per_temp = max(10, n)

        return initial, min(minimum, initial * 0.999), per_temp

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        n = len(points)
        if n <= 3:
            return np.arange(n)

        # A greedy start beats a random permutation and lets the cooling
        # schedule spend its budget on refinement rather than untangling.
        current = list(nearest_neighbor_tour(points))
        current_distance = tour_length_of_indices(points, current)

        best_tour = current[:]
        best_distance = current_distance

        temperature, min_temp, iterations_per_temp = self._schedule(points, current)

        while temperature > min_temp:
            for _ in range(iterations_per_temp):
                i, j = sorted(self._random.sample(range(n), 2))
                # Reversing the entire tour leaves its length unchanged.
                if i == 0 and j == n - 1:
                    continue

                delta = self._reversal_delta(points, current, i, j)

                if delta < 0 or self._random.random() < self._acceptance(delta, temperature):
                    current[i:j + 1] = current[i:j + 1][::-1]
                    current_distance += delta

                    if current_distance < best_distance:
                        best_distance = current_distance
                        best_tour = current[:]

            temperature *= self.cooling_rate

        return np.asarray(best_tour, dtype=int)

    @staticmethod
    def _acceptance(delta: float, temperature: float) -> float:
        """Metropolis acceptance probability, guarded against overflow"""
        exponent = -delta / temperature
        if exponent < -700:  # np.exp underflows to 0 well before this
            return 0.0
        return float(np.exp(exponent))

    @staticmethod
    def _reversal_delta(points: np.ndarray, tour: Sequence[int], i: int, j: int) -> float:
        """Length change from reversing ``tour[i:j+1]`` in a closed tour"""
        n = len(tour)
        a = points[tour[i - 1]]
        b = points[tour[i]]
        c = points[tour[j]]
        d = points[tour[(j + 1) % n]]
        removed = np.linalg.norm(a - b) + np.linalg.norm(c - d)
        added = np.linalg.norm(a - c) + np.linalg.norm(b - d)
        return float(added - removed)


class AssociationTSP(TSPAlgorithm):
    """Association-based iterative solver: an elastic loop fitted to the data.

    A closed loop of vertices is pulled toward the centroids of the points that
    select it as their nearest vertex, then relaxed by curvature-aware Laplacian
    smoothing. Repeating that contracts the loop onto the data. The tour is read
    off by ordering the points along the fitted loop.
    """

    produces_loop = True

    def __init__(self, n_vertices: Optional[int] = None, max_iterations: int = 100,
                 adaptive_vertices: Optional[bool] = None,
                 subdivision_threshold: float = 0.05,
                 min_move_rate: float = 0.05, min_smooth_rate: float = 0.0,
                 initial_move_rate: float = 0.3, initial_smooth_rate: float = 0.4,
                 smoothing_iterations: int = 2, reoptimize_every: int = 20,
                 min_vertices: int = 60, max_vertices: int = 300,
                 vertex_density_factor: float = 0.4,
                 seed: Optional[int] = None):
        super().__init__("Association", seed=seed)
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")

        self.n_vertices = n_vertices
        self.max_iterations = max_iterations
        # An explicit vertex count is honoured unless the caller explicitly
        # asks for the adaptive count as well.
        self.adaptive_vertices = (adaptive_vertices if adaptive_vertices is not None
                                  else n_vertices is None)
        self.subdivision_threshold = subdivision_threshold
        self.min_move_rate = min_move_rate
        self.min_smooth_rate = min_smooth_rate
        self.initial_move_rate = initial_move_rate
        self.initial_smooth_rate = initial_smooth_rate
        self.smoothing_iterations = max(1, smoothing_iterations)
        self.reoptimize_every = reoptimize_every
        self.min_vertices = min_vertices
        self.max_vertices = max_vertices
        self.vertex_density_factor = vertex_density_factor

        #: Vertex count of the most recent solve, after adaptive resizing.
        self.last_vertex_count: Optional[int] = None

    def _resolve_vertex_count(self, points: np.ndarray) -> int:
        """Pick the loop resolution for this data set"""
        if self.adaptive_vertices:
            count = calculate_adaptive_vertex_count(
                points,
                min_vertices=self.min_vertices,
                max_vertices=self.max_vertices,
                density_factor=self.vertex_density_factor,
            )
            logger.debug("Using adaptive vertex count: %d", count)
            return count

        if self.n_vertices is None:
            return max(3, min(self.max_vertices, len(points)))
        return max(3, int(self.n_vertices))

    def solve_loop(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)

        if len(points) <= 3:
            return points

        n_vertices = self._resolve_vertex_count(points)
        vertices = init_circular_loop(n_vertices, seed=self._rng)

        for iteration in range(self.max_iterations):
            progress = iteration / self.max_iterations
            move_rate = max(self.initial_move_rate * (1 - progress), self.min_move_rate)
            smooth_rate = max(self.initial_smooth_rate * (1 - progress), self.min_smooth_rate)

            vertices = self._attract_step(points, vertices, move_rate)

            for _ in range(self.smoothing_iterations):
                vertices = adaptive_smooth_loop(vertices, smooth_rate * 0.5,
                                                curvature_weight=0.3)

            if self.reoptimize_every > 0 and iteration > 0 and iteration % self.reoptimize_every == 0:
                vertices = self._reoptimize(vertices)
                logger.debug("Iteration %d: loop resized to %d vertices",
                             iteration, len(vertices))

            vertices = np.clip(vertices, 0.02, 0.98)

        self.last_vertex_count = len(vertices)
        return vertices

    @staticmethod
    def _attract_step(points: np.ndarray, vertices: np.ndarray,
                      move_rate: float) -> np.ndarray:
        """Pull each vertex toward the centroid of the points assigned to it"""
        _, nearest = SpatialIndex(vertices).query_nearest(points)

        # Accumulate the centroid of every vertex's catchment in one pass
        # instead of scanning the assignment array once per vertex.
        sums = np.zeros_like(vertices)
        counts = np.bincount(nearest, minlength=len(vertices)).astype(float)
        np.add.at(sums, nearest, points)

        assigned = counts > 0
        centroids = np.zeros_like(vertices)
        centroids[assigned] = sums[assigned] / counts[assigned, None]

        moved = vertices.copy()
        moved[assigned] = (vertices[assigned] * (1 - move_rate)
                           + centroids[assigned] * move_rate)
        return moved

    def _reoptimize(self, vertices: np.ndarray) -> np.ndarray:
        """Subdivide long edges, drop crowded vertices, respace the loop"""
        vertices = subdivide_vertices(vertices, self.subdivision_threshold)
        vertices = remove_redundant_vertices(vertices, min_distance=0.008)
        if len(vertices) > self.max_vertices:
            vertices = resample_curve(vertices, self.max_vertices)
        return optimize_vertex_distribution(vertices)

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) <= 3:
            return np.arange(len(points))
        return order_points_along_loop(points, self.solve_loop(points))


class ClusteringTSP(TSPAlgorithm):
    """K-means clustering based solver with interpolated fine vertices.

    Cluster centres are ordered greedily to form a coarse route, then resampled
    into a finer loop; the tour is the data points ordered along that loop.
    """

    produces_loop = True

    def __init__(self, n_clusters: int = 40, n_interpolated_vertices: int = 120,
                 n_init: int = 10, seed: Optional[int] = None):
        super().__init__("K-means Clustering", seed=seed)
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = n_clusters
        self.n_interpolated_vertices = max(3, n_interpolated_vertices)
        self.n_init = n_init

    def solve_loop(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)

        if len(points) <= 3:
            return points

        # k-means cannot ask for more clusters than it has samples.
        n_clusters = min(self.n_clusters, len(points))
        kmeans = KMeans(n_clusters=n_clusters,
                        random_state=self.seed if self.seed is not None else 42,
                        n_init=self.n_init)
        kmeans.fit(points)
        centers = kmeans.cluster_centers_

        ordered_centers = centers[nearest_neighbor_tour(centers)]

        if len(ordered_centers) >= 3:
            return resample_curve(ordered_centers, self.n_interpolated_vertices)

        return ordered_centers

    def solve_tour(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if len(points) <= 3:
            return np.arange(len(points))
        return order_points_along_loop(points, self.solve_loop(points))


ALGORITHMS = {
    'nearest_neighbor': NearestNeighborTSP,
    'two_opt': TwoOptTSP,
    'genetic': GeneticTSP,
    'simulated_annealing': SimulatedAnnealingTSP,
    'association': AssociationTSP,
    'clustering': ClusteringTSP,
}


def get_algorithm(algorithm_name: str, **kwargs) -> TSPAlgorithm:
    """Factory function to get a TSP algorithm by name.

    Unknown keyword arguments are dropped rather than raising, so a caller can
    pass a single parameter bundle across several solvers.
    """
    if algorithm_name not in ALGORITHMS:
        known = ', '.join(sorted(ALGORITHMS))
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Choose one of: {known}")

    algorithm_class = ALGORITHMS[algorithm_name]

    accepted = inspect.signature(algorithm_class.__init__).parameters
    supported = {k: v for k, v in kwargs.items() if k in accepted}
    for unsupported in sorted(set(kwargs) - set(supported)):
        logger.debug("%s does not accept '%s'; ignoring it",
                     algorithm_name, unsupported)

    return algorithm_class(**supported)
