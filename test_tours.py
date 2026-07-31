"""
Tests for tour construction: projecting loops onto tours, 2-opt refinement and
the guarantee that every solver returns a valid permutation of the input points.

Everything here is deterministic and runs offline; point sets are either
hand-written or generated from an explicit seed.
"""

import unittest

import numpy as np

from algorithms import (ALGORITHMS, AssociationTSP, ClusteringTSP, GeneticTSP,
                        NearestNeighborTSP, SimulatedAnnealingTSP, TwoOptTSP,
                        get_algorithm, nearest_neighbor_tour)
from utils import (generate_clustered_points, is_valid_tour,
                   order_points_along_loop, project_points_onto_loop,
                   tour_length, tour_length_of_indices, two_opt_refine)


def unit_square(n_per_side=5):
    """Points on the perimeter of the unit square, in scrambled order"""
    ts = np.linspace(0, 1, n_per_side, endpoint=False)
    bottom = np.column_stack([ts, np.zeros_like(ts)])
    right = np.column_stack([np.ones_like(ts), ts])
    top = np.column_stack([1 - ts, np.ones_like(ts)])
    left = np.column_stack([np.zeros_like(ts), 1 - ts])
    ring = np.vstack([bottom, right, top, left])
    rng = np.random.default_rng(0)
    return ring[rng.permutation(len(ring))]


# Small, fast parameter sets so the whole suite stays quick.
FAST_KWARGS = {
    'nearest_neighbor': {},
    'two_opt': {'max_iterations': 20},
    'genetic': {'population_size': 20, 'generations': 10},
    'simulated_annealing': {'cooling_rate': 0.9, 'iterations_per_temp': 5},
    'association': {'n_vertices': 30, 'max_iterations': 20},
    'clustering': {'n_clusters': 8, 'n_interpolated_vertices': 30},
}


class TestProjection(unittest.TestCase):
    """Projecting points onto a closed loop"""

    def test_point_on_loop_projects_to_zero_distance(self):
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        points = np.array([[0.5, 0.0], [1.0, 0.25]])

        arc, distance = project_points_onto_loop(points, loop)

        np.testing.assert_allclose(distance, [0.0, 0.0], atol=1e-12)
        # 0.5 along the first edge, then 1.0 + 0.25 along the second.
        np.testing.assert_allclose(arc, [0.5, 1.25], atol=1e-12)

    def test_distance_to_offset_point(self):
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        points = np.array([[0.5, 0.3]])

        _, distance = project_points_onto_loop(points, loop)

        self.assertAlmostEqual(float(distance[0]), 0.3, places=12)

    def test_projection_matches_across_chunk_sizes(self):
        """Chunking is an internal memory optimisation, not a behaviour change"""
        points = generate_clustered_points(60, 4, 0.05, 0.6, seed=11)
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        whole = project_points_onto_loop(points, loop, chunk_size=10_000)
        chunked = project_points_onto_loop(points, loop, chunk_size=7)

        np.testing.assert_allclose(whole[0], chunked[0])
        np.testing.assert_allclose(whole[1], chunked[1])

    def test_rejects_degenerate_loop(self):
        with self.assertRaises(ValueError):
            project_points_onto_loop(np.zeros((3, 2)), np.zeros((1, 2)))


class TestOrderAlongLoop(unittest.TestCase):
    """Turning a fitted loop into a tour over the data points"""

    def test_square_ring_recovers_the_ring_order(self):
        """Points sitting on a loop come back in the loop's own order"""
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        points = unit_square(5)

        order = order_points_along_loop(points, loop)

        self.assertTrue(is_valid_tour(order, len(points)))
        # The perimeter is the optimal tour of points lying on it.
        self.assertAlmostEqual(tour_length_of_indices(points, order), 4.0, places=9)

    def test_returns_a_permutation(self):
        points = generate_clustered_points(80, 5, 0.05, 0.6, seed=3)
        loop = np.array([[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])

        order = order_points_along_loop(points, loop)

        self.assertTrue(is_valid_tour(order, len(points)))

    def test_handles_duplicate_points(self):
        """Duplicated coordinates stay distinct entries in the tour"""
        points = np.array([[0.2, 0.2], [0.2, 0.2], [0.8, 0.2], [0.8, 0.8]])
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

        order = order_points_along_loop(points, loop)

        self.assertTrue(is_valid_tour(order, 4))

    def test_tiny_inputs_are_returned_unchanged(self):
        loop = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        for n in (0, 1, 2):
            with self.subTest(n=n):
                points = np.zeros((n, 2))
                np.testing.assert_array_equal(
                    order_points_along_loop(points, loop), np.arange(n))


class TestTwoOptRefine(unittest.TestCase):
    """2-opt polishing"""

    def test_untangles_a_crossed_square(self):
        """The classic crossed tour is repaired into the perimeter"""
        points = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        crossed = [0, 1, 2, 3]
        self.assertGreater(tour_length_of_indices(points, crossed), 4.0)

        refined = two_opt_refine(points, crossed)

        self.assertTrue(is_valid_tour(refined, 4))
        self.assertAlmostEqual(tour_length_of_indices(points, refined), 4.0, places=9)

    def test_never_lengthens_a_tour(self):
        points = generate_clustered_points(40, 4, 0.05, 0.6, seed=17)
        start = np.arange(len(points))
        before = tour_length_of_indices(points, start)

        refined = two_opt_refine(points, start)

        self.assertTrue(is_valid_tour(refined, len(points)))
        self.assertLessEqual(tour_length_of_indices(points, refined), before + 1e-9)

    def test_optimal_tour_is_left_alone(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        refined = two_opt_refine(points, [0, 1, 2, 3])
        self.assertAlmostEqual(tour_length_of_indices(points, refined), 4.0, places=9)

    def test_short_tours_pass_through(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(two_opt_refine(points, [0, 1, 2]), [0, 1, 2])


class TestNearestNeighborTour(unittest.TestCase):
    """Greedy construction"""

    def test_returns_permutation(self):
        points = generate_clustered_points(30, 3, 0.05, 0.6, seed=2)
        tour = nearest_neighbor_tour(points)
        self.assertTrue(is_valid_tour(tour, len(points)))

    def test_handles_duplicate_coordinates(self):
        """Duplicates used to collapse when tours were matched by value"""
        points = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        tour = nearest_neighbor_tour(points)
        self.assertTrue(is_valid_tour(tour, 4))

    def test_respects_start_index(self):
        points = generate_clustered_points(20, 2, 0.05, 0.6, seed=4)
        self.assertEqual(int(nearest_neighbor_tour(points, start=5)[0]), 5)

    def test_rejects_out_of_range_start(self):
        with self.assertRaises(ValueError):
            nearest_neighbor_tour(np.zeros((4, 2)), start=9)


class TestSolverContract(unittest.TestCase):
    """Every solver must honour the same interface"""

    def setUp(self):
        self.points = generate_clustered_points(40, 4, 0.05, 0.6, seed=21)

    def test_every_solver_returns_a_valid_tour(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                algo = get_algorithm(name, seed=0, **FAST_KWARGS[name])
                tour = algo.solve_tour(self.points)
                self.assertTrue(is_valid_tour(tour, len(self.points)),
                                f"{name} did not return a permutation of the points")

    def test_evaluate_reports_tour_length_not_loop_length(self):
        """A fitted loop is shorter than the tour it induces, and both are reported"""
        algo = AssociationTSP(n_vertices=40, max_iterations=30, seed=0)
        solution = algo.evaluate(self.points)

        self.assertTrue(is_valid_tour(solution.tour, len(self.points)))
        self.assertAlmostEqual(
            solution.length, tour_length_of_indices(self.points, solution.tour), places=9)
        self.assertAlmostEqual(solution.loop_length, tour_length(solution.loop), places=9)
        # The whole point of separating the two: the loop hugs the data and is
        # shorter than any tour that has to visit every point.
        self.assertLess(solution.loop_length, solution.length)

    def test_permutation_solvers_report_equal_loop_and_tour_length(self):
        algo = NearestNeighborTSP(seed=0)
        solution = algo.evaluate(self.points)
        self.assertAlmostEqual(solution.length, solution.loop_length, places=9)

    def test_refine_never_worsens_the_result(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                plain = get_algorithm(name, seed=0, **FAST_KWARGS[name])
                refined = get_algorithm(name, seed=0, **FAST_KWARGS[name])
                self.assertLessEqual(
                    refined.evaluate(self.points, refine=True).length,
                    plain.evaluate(self.points).length + 1e-9)

    def test_solutions_are_reproducible_for_a_fixed_seed(self):
        for name in ALGORITHMS:
            with self.subTest(algorithm=name):
                first = get_algorithm(name, seed=99, **FAST_KWARGS[name])
                second = get_algorithm(name, seed=99, **FAST_KWARGS[name])
                np.testing.assert_array_equal(
                    first.solve_tour(self.points), second.solve_tour(self.points))

    def test_solvers_handle_tiny_inputs(self):
        for name in ALGORITHMS:
            for n in (0, 1, 2, 3):
                with self.subTest(algorithm=name, n=n):
                    points = np.linspace(0.1, 0.9, max(n * 2, 2)).reshape(-1, 2)[:n]
                    algo = get_algorithm(name, seed=0, **FAST_KWARGS[name])
                    self.assertTrue(is_valid_tour(algo.solve_tour(points), n))


class TestSolverQuality(unittest.TestCase):
    """Sanity checks on solution quality"""

    def setUp(self):
        self.points = generate_clustered_points(50, 4, 0.05, 0.6, seed=31)
        self.baseline = tour_length_of_indices(
            self.points, NearestNeighborTSP().solve_tour(self.points))

    def test_two_opt_beats_nearest_neighbor(self):
        length = tour_length_of_indices(
            self.points, TwoOptTSP().solve_tour(self.points))
        self.assertLess(length, self.baseline)

    def test_annealing_improves_on_its_greedy_start(self):
        """A miscalibrated temperature degenerates into a random walk"""
        algo = SimulatedAnnealingTSP(seed=1, cooling_rate=0.95)
        length = tour_length_of_indices(self.points, algo.solve_tour(self.points))
        self.assertLess(length, self.baseline)

    def test_genetic_improves_on_its_greedy_seed(self):
        algo = GeneticTSP(population_size=40, generations=60, seed=1)
        length = tour_length_of_indices(self.points, algo.solve_tour(self.points))
        self.assertLessEqual(length, self.baseline)

    def test_loop_solvers_produce_usable_tours(self):
        """The induced tours should be in the same league as the greedy baseline"""
        for algo in (AssociationTSP(n_vertices=60, max_iterations=80, seed=1),
                     ClusteringTSP(n_clusters=25, seed=1)):
            with self.subTest(algorithm=algo.name):
                length = tour_length_of_indices(
                    self.points, algo.solve_tour(self.points))
                self.assertLess(length, self.baseline * 1.5)


class TestAlgorithmFactory(unittest.TestCase):
    """Factory behaviour"""

    def test_unknown_algorithm_lists_the_valid_names(self):
        with self.assertRaises(ValueError) as ctx:
            get_algorithm('does_not_exist')
        self.assertIn('nearest_neighbor', str(ctx.exception))

    def test_unsupported_keywords_are_ignored(self):
        """One parameter bundle can be shared across solvers"""
        algo = get_algorithm('nearest_neighbor', population_size=10, seed=3)
        self.assertIsInstance(algo, NearestNeighborTSP)
        self.assertEqual(algo.seed, 3)

    def test_elite_size_is_clamped_below_population(self):
        """An elite as large as the population would leave no room for offspring"""
        algo = GeneticTSP(population_size=10, elite_size=50)
        self.assertLess(algo.elite_size, algo.population_size)

    def test_invalid_cooling_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            SimulatedAnnealingTSP(cooling_rate=1.5)


if __name__ == '__main__':
    unittest.main()
