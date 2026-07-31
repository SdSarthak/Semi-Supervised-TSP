"""
Tests for failure modes and numerical correctness.

Everything here is deterministic and offline: point sets are hand-written or
generated from an explicit seed, and no test needs a display or a download.

The three properties under test are the ones the solvers previously got wrong:

* the elastic-loop solver must behave identically at any coordinate scale,
  because its geometry constants used to be hard-coded for the unit square;
* a loop that carries no ordering information (one vertex, or all vertices
  collapsed together) must not crash or silently produce an arbitrary tour;
* NaN and infinite coordinates must be rejected rather than propagated into a
  NaN tour length that still looks like an answer.
"""

import json
import os
import tempfile
import unittest

import numpy as np

from algorithms import (AssociationTSP, ClusteringTSP, NearestNeighborTSP,
                        nearest_neighbor_tour)
from utils import (SpatialIndex, attract_vertices, calculate_adaptive_vertex_count,
                   data_bounds, generate_clustered_points, init_circular_loop,
                   is_valid_tour, load_points, smooth_loop, tour_length,
                   tour_length_of_indices, validate_points)


def affine(points, scale, offset):
    """Uniformly rescale and translate a point set"""
    return np.asarray(points, dtype=float) * scale + offset


class TestScaleInvariance(unittest.TestCase):
    """The loop solver must not assume unit-square coordinates.

    The fit is built entirely from distances, so scaling and translating the
    input has to scale and translate the answer and change nothing else. It
    used to clip every vertex into ``[0.02, 0.98]``, which on real coordinates
    pinned the whole loop into one corner and produced a meaningless tour.
    """

    def setUp(self):
        self.points = generate_clustered_points(60, 4, 0.05, 0.6, seed=5)
        self.cases = {
            'scaled up': affine(self.points, 1000.0, 5000.0),
            'scaled down': affine(self.points, 1e-3, 0.0),
            'translated negative': affine(self.points, 20.0, -10.0),
        }

    def solver(self):
        return AssociationTSP(n_vertices=60, max_iterations=60,
                              adaptive_vertices=False, seed=1)

    def test_association_tour_is_unchanged_by_rescaling(self):
        reference = self.solver().solve_tour(self.points)
        self.assertTrue(is_valid_tour(reference, len(self.points)))

        for label, transformed in self.cases.items():
            with self.subTest(case=label):
                np.testing.assert_array_equal(
                    self.solver().solve_tour(transformed), reference)

    def test_association_quality_is_unchanged_by_rescaling(self):
        """Relative to greedy, the answer must be the same on any scale"""
        def ratio(pts):
            greedy = tour_length_of_indices(pts, nearest_neighbor_tour(pts))
            fitted = tour_length_of_indices(pts, self.solver().solve_tour(pts))
            return fitted / greedy

        reference = ratio(self.points)
        for label, transformed in self.cases.items():
            with self.subTest(case=label):
                self.assertAlmostEqual(ratio(transformed), reference, places=6)

    def test_fitted_loop_tracks_the_data_not_the_unit_square(self):
        """On large coordinates the loop used to collapse into [0.02, 0.98]"""
        points = self.cases['scaled up']
        loop = self.solver().solve_loop(points)

        lo, hi = points.min(axis=0), points.max(axis=0)
        span = float(np.max(hi - lo))
        self.assertTrue(np.all(loop.min(axis=0) > lo - span))
        self.assertTrue(np.all(loop.max(axis=0) < hi + span))
        # A loop threaded through the data spans a good fraction of its box.
        self.assertGreater(float(np.max(loop.max(axis=0) - loop.min(axis=0))),
                           0.5 * span)

    def test_adaptive_vertex_count_is_scale_free(self):
        baseline = calculate_adaptive_vertex_count(self.points)
        for label, transformed in self.cases.items():
            with self.subTest(case=label):
                self.assertEqual(calculate_adaptive_vertex_count(transformed),
                                 baseline)


class TestDataBounds(unittest.TestCase):
    """Bounding box helper"""

    def test_scale_and_padding(self):
        points = np.array([[0.0, 0.0], [4.0, 2.0]])
        lo, hi, scale = data_bounds(points, margin=0.25)
        self.assertEqual(scale, 4.0)
        np.testing.assert_allclose(lo, [-1.0, -1.0])
        np.testing.assert_allclose(hi, [5.0, 3.0])

    def test_degenerate_input_never_yields_a_zero_scale(self):
        """Every caller divides or multiplies by the scale"""
        for points in (np.empty((0, 2)), np.tile([0.3, 0.7], (5, 1))):
            with self.subTest(n=len(points)):
                _, _, scale = data_bounds(points)
                self.assertGreater(scale, 0.0)


class TestDegenerateLoops(unittest.TestCase):
    """A loop with no ordering information must not crash or lie"""

    def setUp(self):
        self.points = generate_clustered_points(20, 3, 0.05, 0.6, seed=9)

    def test_single_cluster_produces_a_valid_tour(self):
        """One centre is a one-vertex loop; projection used to raise"""
        tour = ClusteringTSP(n_clusters=1, seed=1).solve_tour(self.points)
        self.assertTrue(is_valid_tour(tour, len(self.points)))

    def test_single_cluster_evaluates(self):
        solution = ClusteringTSP(n_clusters=1, seed=1).evaluate(self.points)
        self.assertTrue(is_valid_tour(solution.tour, len(self.points)))
        self.assertAlmostEqual(
            solution.length,
            tour_length_of_indices(self.points, solution.tour), places=12)

    def test_collapsed_loop_falls_back_to_greedy(self):
        """A zero-length loop projects everything to arc position zero"""
        algo = ClusteringTSP(seed=1)
        collapsed = np.tile([0.5, 0.5], (10, 1))

        tour = algo._tour_from_loop(self.points, collapsed)

        self.assertTrue(is_valid_tour(tour, len(self.points)))
        np.testing.assert_array_equal(tour, nearest_neighbor_tour(self.points))

    def test_two_cluster_loop_still_orders_points(self):
        tour = ClusteringTSP(n_clusters=2, seed=1).solve_tour(self.points)
        self.assertTrue(is_valid_tour(tour, len(self.points)))


class TestPointValidation(unittest.TestCase):
    """Non-finite input must be rejected, not propagated"""

    def test_accepts_well_formed_points(self):
        points = [[0.0, 1.0], [2.0, 3.0]]
        np.testing.assert_allclose(validate_points(points), points)

    def test_accepts_an_empty_set(self):
        self.assertEqual(validate_points(np.empty((0, 2))).shape, (0, 2))

    def test_rejects_nan_and_infinity(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(value=bad):
                points = np.array([[0.0, 0.0], [bad, 0.5], [1.0, 1.0]])
                with self.assertRaises(ValueError) as ctx:
                    validate_points(points)
                self.assertIn('1', str(ctx.exception))

    def test_rejects_wrong_shape(self):
        for bad in (np.zeros((3, 3)), np.zeros(4), np.zeros((2, 2, 2))):
            with self.subTest(shape=bad.shape):
                with self.assertRaises(ValueError):
                    validate_points(bad)

    def test_rejects_ragged_input(self):
        with self.assertRaises(ValueError):
            validate_points([[0.0, 1.0], [2.0]])

    def test_nan_input_produces_a_nan_length_without_validation(self):
        """Why this matters: the raw pipeline reports NaN as if it were a score"""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [np.nan, 0.5],
                           [1.0, 1.0], [0.0, 1.0]])
        self.assertTrue(np.isnan(
            tour_length_of_indices(points, nearest_neighbor_tour(points))))

        with self.assertRaises(ValueError):
            NearestNeighborTSP().evaluate(points)

    def test_every_solver_rejects_nan_from_evaluate(self):
        points = generate_clustered_points(12, 2, 0.05, 0.6, seed=2)
        points[3, 1] = np.nan
        for algo in (NearestNeighborTSP(),
                     AssociationTSP(n_vertices=20, max_iterations=5, seed=0),
                     ClusteringTSP(n_clusters=4, seed=0)):
            with self.subTest(algorithm=algo.name):
                with self.assertRaises(ValueError):
                    algo.evaluate(points)


class TestLoadPointsFailures(unittest.TestCase):
    """File parsing boundaries"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def write(self, name, text):
        target = self.path(name)
        with open(target, 'w', encoding='utf-8') as f:
            f.write(text)
        return target

    def test_single_row_csv_loads_as_one_point(self):
        target = self.write('one.csv', '0.25,0.75\n')
        np.testing.assert_allclose(load_points(target), [[0.25, 0.75]])

    def test_empty_file_reports_that_it_is_empty(self):
        target = self.write('empty.csv', '')
        with self.assertRaises(ValueError) as ctx:
            load_points(target)
        self.assertIn('no points', str(ctx.exception))

    def test_empty_json_point_list_is_rejected(self):
        target = self.write('empty.json', json.dumps({'points': []}))
        with self.assertRaises(ValueError):
            load_points(target)

    def test_malformed_json_names_the_file(self):
        target = self.write('broken.json', '{not json')
        with self.assertRaises(ValueError) as ctx:
            load_points(target)
        self.assertIn('broken.json', str(ctx.exception))

    def test_nan_coordinates_are_rejected(self):
        target = self.write('nan.csv', '0.1,0.2\nnan,0.4\n')
        with self.assertRaises(ValueError) as ctx:
            load_points(target)
        self.assertIn('NaN', str(ctx.exception))

    def test_missing_file_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            load_points(self.path('absent.csv'))

    def test_unicode_path_round_trips(self):
        target = self.write('pöints-数据.csv', '0.1,0.2\n0.3,0.4\n')
        np.testing.assert_allclose(load_points(target), [[0.1, 0.2], [0.3, 0.4]])


class TestAttractVertices(unittest.TestCase):
    """The shared association step, previously copied into four modules"""

    @staticmethod
    def reference(points, vertices, move_rate):
        """The original per-vertex implementation, kept as the oracle"""
        _, nearest = SpatialIndex(vertices).query_nearest(points)
        moved = vertices.copy()
        for v in range(len(vertices)):
            assigned = points[nearest == v]
            if len(assigned):
                moved[v] = (vertices[v] * (1 - move_rate)
                            + np.mean(assigned, axis=0) * move_rate)
        return moved

    def setUp(self):
        self.points = generate_clustered_points(120, 5, 0.05, 0.6, seed=13)
        self.vertices = init_circular_loop(40, seed=13)

    def test_matches_the_per_vertex_reference(self):
        np.testing.assert_allclose(
            attract_vertices(self.points, self.vertices, 0.25),
            self.reference(self.points, self.vertices, 0.25))

    def test_does_not_mutate_its_input(self):
        before = self.vertices.copy()
        attract_vertices(self.points, self.vertices, 0.5)
        np.testing.assert_array_equal(self.vertices, before)

    def test_unassigned_vertices_stay_put(self):
        """A vertex with an empty catchment has no centroid to move toward"""
        points = np.array([[0.0, 0.0]])
        vertices = np.array([[0.0, 0.5], [1.0, 0.5], [0.5, 1.0]])

        moved = attract_vertices(points, vertices, 1.0)

        np.testing.assert_allclose(moved[0], [0.0, 0.0])
        np.testing.assert_array_equal(moved[1:], vertices[1:])

    def test_empty_point_set_is_a_no_op(self):
        moved = attract_vertices(np.empty((0, 2)), self.vertices, 0.3)
        np.testing.assert_array_equal(moved, self.vertices)

    def test_full_move_rate_lands_on_the_centroid(self):
        vertices = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        points = np.array([[1.0, 1.0], [1.0, -1.0],     # catchment of vertex 0
                           [11.0, 2.0], [11.0, -2.0]])  # catchment of vertex 1

        moved = attract_vertices(points, vertices, 1.0)

        np.testing.assert_allclose(moved[0], [1.0, 0.0])
        np.testing.assert_allclose(moved[1], [11.0, 0.0])
        np.testing.assert_allclose(moved[2], [0.0, 10.0])

    def test_move_rate_interpolates_linearly(self):
        vertices = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
        points = np.array([[4.0, 0.0]])

        moved = attract_vertices(points, vertices, 0.25)

        np.testing.assert_allclose(moved[0], [1.0, 0.0])

    def test_smoothing_leaves_the_centroid_alone(self):
        """Laplacian smoothing must not translate the loop"""
        smoothed = smooth_loop(self.vertices, 0.5, iterations=5)
        np.testing.assert_allclose(smoothed.mean(axis=0),
                                   self.vertices.mean(axis=0), atol=1e-12)


class TestLoopInitialisation(unittest.TestCase):
    """init_circular_loop guards"""

    def test_bounds_are_respected(self):
        lo, hi = np.array([10.0, 20.0]), np.array([12.0, 22.0])
        loop = init_circular_loop(50, center=(11.0, 21.0), radius=5.0,
                                  noise_std=1.0, seed=3, bounds=(lo, hi))
        self.assertTrue(np.all(loop >= lo))
        self.assertTrue(np.all(loop <= hi))

    def test_default_bounds_are_the_unit_square(self):
        loop = init_circular_loop(30, seed=3)
        self.assertTrue(np.all(loop >= 0.02))
        self.assertTrue(np.all(loop <= 0.98))

    def test_rejects_a_non_positive_radius(self):
        for radius in (0.0, -1.0, np.nan):
            with self.subTest(radius=radius):
                with self.assertRaises(ValueError):
                    init_circular_loop(10, radius=radius)

    def test_rejects_too_few_vertices(self):
        with self.assertRaises(ValueError):
            init_circular_loop(2)

    def test_seed_fixes_the_loop(self):
        np.testing.assert_array_equal(init_circular_loop(20, seed=7),
                                      init_circular_loop(20, seed=7))
        self.assertFalse(np.array_equal(init_circular_loop(20, seed=7),
                                        init_circular_loop(20, seed=8)))


class TestDuplicateAndCollinearPoints(unittest.TestCase):
    """Degenerate geometry that a real data set can easily contain"""

    def test_all_points_identical(self):
        points = np.tile([0.4, 0.6], (12, 1))
        for algo in (NearestNeighborTSP(),
                     AssociationTSP(n_vertices=10, max_iterations=5,
                                    adaptive_vertices=False, seed=0),
                     ClusteringTSP(n_clusters=3, seed=0)):
            with self.subTest(algorithm=algo.name):
                solution = algo.evaluate(points)
                self.assertTrue(is_valid_tour(solution.tour, len(points)))
                self.assertAlmostEqual(solution.length, 0.0, places=12)

    def test_collinear_points(self):
        points = np.column_stack([np.linspace(0.1, 0.9, 15),
                                  np.full(15, 0.5)])
        algo = AssociationTSP(n_vertices=20, max_iterations=20,
                              adaptive_vertices=False, seed=0)
        solution = algo.evaluate(points)

        self.assertTrue(is_valid_tour(solution.tour, len(points)))
        # Out and back along the line is the optimal closed tour.
        self.assertGreaterEqual(solution.length, 1.6 - 1e-9)

    def test_tour_length_of_a_single_point_is_zero(self):
        self.assertEqual(tour_length(np.array([[0.5, 0.5]])), 0.0)
        self.assertEqual(tour_length(np.empty((0, 2))), 0.0)


if __name__ == '__main__':
    unittest.main()
