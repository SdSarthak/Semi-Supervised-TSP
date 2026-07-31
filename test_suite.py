"""
Comprehensive test suite for Semi-Supervised TSP Visualizer
"""

import importlib
import os
import tempfile
import unittest

import numpy as np

from config import Config
from utils import (euclidean_distance_matrix, tour_length, smooth_loop,
                   generate_clustered_points, init_circular_loop,
                   compute_convergence_metric, adaptive_parameters,
                   SpatialIndex, Timer)
from algorithms import (get_algorithm, NearestNeighborTSP, TwoOptTSP,
                       GeneticTSP, SimulatedAnnealingTSP, AssociationTSP,
                       ClusteringTSP)

# Every test here is deterministic and needs no downloaded data: point sets
# are either hand-written or generated from an explicit seed.

class TestConfig(unittest.TestCase):
    """Test configuration validation"""
    
    def test_config_validation(self):
        """Test config validation works"""
        Config.validate()  # Should not raise
        
    def test_config_to_dict(self):
        """Test config conversion to dictionary"""
        config_dict = Config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn('N_POINTS', config_dict)

class TestUtils(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.points1 = np.array([[0, 0], [1, 1], [2, 2]])
        self.points2 = np.array([[0.5, 0.5], [1.5, 1.5]])
        
    def test_euclidean_distance_matrix(self):
        """Test distance matrix calculation"""
        dist_matrix = euclidean_distance_matrix(self.points1, self.points2)
        self.assertEqual(dist_matrix.shape, (3, 2))
        
        # Test known distance
        expected_dist = np.sqrt((0-0.5)**2 + (0-0.5)**2)
        self.assertAlmostEqual(dist_matrix[0, 0], expected_dist, places=6)
        
    def test_tour_length(self):
        """Test tour length calculation"""
        # Simple square tour
        tour = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        expected_length = 4.0  # Perimeter of unit square
        self.assertAlmostEqual(tour_length(tour), expected_length, places=6)
        
    def test_tour_length_empty(self):
        """Test tour length with empty/small tours"""
        self.assertEqual(tour_length(np.array([])), 0.0)
        self.assertEqual(tour_length(np.array([[0, 0]])), 0.0)
        
    def test_smooth_loop(self):
        """Test loop smoothing"""
        # Create a noisy square
        vertices = np.array([[0, 0], [1, 0.1], [1, 1], [0, 0.9]])
        smoothed = smooth_loop(vertices, 0.5)
        
        self.assertEqual(smoothed.shape, vertices.shape)
        # Smoothing should change the vertices (test that it runs without error)
        self.assertFalse(np.array_equal(smoothed, vertices))
        
    def test_generate_clustered_points(self):
        """Test point generation"""
        points = generate_clustered_points(100, 3, 0.05, 0.6, seed=0)

        self.assertEqual(len(points), 100)
        self.assertEqual(points.shape[1], 2)
        self.assertTrue(np.all(points >= 0.02))
        self.assertTrue(np.all(points <= 0.98))

    def test_generate_clustered_points_is_seed_reproducible(self):
        """The same seed gives the same points; a different seed does not"""
        first = generate_clustered_points(50, 3, 0.05, 0.6, seed=123)
        again = generate_clustered_points(50, 3, 0.05, 0.6, seed=123)
        other = generate_clustered_points(50, 3, 0.05, 0.6, seed=124)

        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other))

    def test_generate_clustered_points_leaves_global_rng_alone(self):
        """Generating data must not reseed numpy's global RNG for everyone else"""
        np.random.seed(7)
        expected = np.random.rand(3)

        np.random.seed(7)
        generate_clustered_points(50, 3, 0.05, 0.6, seed=99)
        actual = np.random.rand(3)

        np.testing.assert_array_equal(expected, actual)


    def test_init_circular_loop(self):
        """Test circular loop initialization"""
        vertices = init_circular_loop(8)
        
        self.assertEqual(len(vertices), 8)
        self.assertEqual(vertices.shape[1], 2)
        
        # Should be roughly circular
        center = np.mean(vertices, axis=0)
        distances = np.sqrt(np.sum((vertices - center)**2, axis=1))
        # All points should be roughly same distance from center
        self.assertLess(np.std(distances), 0.2)
        
    def test_compute_convergence_metric(self):
        """Test convergence metric calculation"""
        vertices1 = np.array([[0, 0], [1, 1]])
        vertices2 = np.array([[0.1, 0.1], [1.1, 1.1]])
        
        convergence = compute_convergence_metric(vertices2, vertices1)
        expected = np.sqrt(0.02)  # sqrt((0.1^2 + 0.1^2 + 0.1^2 + 0.1^2)/2)
        self.assertAlmostEqual(convergence, expected, places=6)
        
    def test_convergence_metric_none(self):
        """Test convergence metric with None previous"""
        vertices = np.array([[0, 0], [1, 1]])
        convergence = compute_convergence_metric(vertices, None)
        self.assertEqual(convergence, float('inf'))
        
    def test_adaptive_parameters(self):
        """Test adaptive parameter calculation"""
        # At start
        move, smooth = adaptive_parameters(0, 100, 0.5, 0.4)
        self.assertAlmostEqual(move, 0.5, places=6)
        self.assertAlmostEqual(smooth, 0.4, places=6)
        
        # At end
        move, smooth = adaptive_parameters(100, 100, 0.5, 0.4, 0.01, 0.1)
        self.assertGreaterEqual(move, 0.01)
        self.assertGreaterEqual(smooth, 0.1)
        self.assertLess(move, 0.5)
        self.assertLess(smooth, 0.4)

class TestSpatialIndex(unittest.TestCase):
    """Test spatial indexing"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        self.points = np.random.rand(10, 2)
        self.index = SpatialIndex(self.points)
        
    def test_query_nearest(self):
        """Test nearest neighbor query"""
        query_points = np.array([[0.5, 0.5]])
        distances, indices = self.index.query_nearest(query_points)
        
        self.assertEqual(len(distances), 1)
        self.assertEqual(len(indices), 1)
        self.assertGreaterEqual(indices[0], 0)
        self.assertLess(indices[0], len(self.points))
        
    def test_query_k_nearest(self):
        """Test k-nearest neighbor query"""
        query_points = np.array([[0.5, 0.5]])
        distances, indices = self.index.query_k_nearest(query_points, k=3)
        
        self.assertEqual(distances.shape, (1, 3))
        self.assertEqual(indices.shape, (1, 3))
        
        # Distances should be sorted
        self.assertLessEqual(distances[0, 0], distances[0, 1])
        self.assertLessEqual(distances[0, 1], distances[0, 2])

class TestAlgorithms(unittest.TestCase):
    """Test TSP algorithms"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create a simple 4-point square
        self.simple_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.random_points = np.random.rand(10, 2)
        
    def test_get_algorithm(self):
        """Test algorithm factory function"""
        algo = get_algorithm('nearest_neighbor')
        self.assertIsInstance(algo, NearestNeighborTSP)
        
        algo = get_algorithm('two_opt')
        self.assertIsInstance(algo, TwoOptTSP)
        
        with self.assertRaises(ValueError):
            get_algorithm('invalid_algorithm')
            
    def test_nearest_neighbor_tsp(self):
        """Test nearest neighbor algorithm"""
        algo = NearestNeighborTSP()
        tour = algo.solve(self.simple_points)
        
        self.assertEqual(len(tour), len(self.simple_points))
        self.assertEqual(tour.shape[1], 2)
        
        # Should return a valid tour
        tour_len = tour_length(tour)
        self.assertGreater(tour_len, 0)
        
    def test_nearest_neighbor_empty(self):
        """Test nearest neighbor with edge cases"""
        algo = NearestNeighborTSP()
        
        # Empty points
        tour = algo.solve(np.array([]))
        self.assertEqual(len(tour), 0)
        
        # Single point
        single_point = np.array([[0.5, 0.5]])
        tour = algo.solve(single_point)
        np.testing.assert_array_equal(tour, single_point)
        
    def test_two_opt_tsp(self):
        """Test 2-opt algorithm"""
        algo = TwoOptTSP(max_iterations=10)  # Small number for testing
        tour = algo.solve(self.simple_points)
        
        self.assertEqual(len(tour), len(self.simple_points))
        
        # Should improve upon nearest neighbor
        nn_algo = NearestNeighborTSP()
        nn_tour = nn_algo.solve(self.simple_points)
        
        two_opt_length = tour_length(tour)
        nn_length = tour_length(nn_tour)
        
        # 2-opt should be at least as good as nearest neighbor
        self.assertLessEqual(two_opt_length, nn_length + 1e-6)
        
    def test_genetic_tsp(self):
        """Test genetic algorithm"""
        algo = GeneticTSP(population_size=10, generations=5)  # Small for testing
        tour = algo.solve(self.simple_points)
        
        self.assertEqual(len(tour), len(self.simple_points))
        self.assertGreater(tour_length(tour), 0)
        
    def test_simulated_annealing_tsp(self):
        """Test simulated annealing"""
        algo = SimulatedAnnealingTSP(initial_temp=100, cooling_rate=0.9)
        tour = algo.solve(self.simple_points)
        
        self.assertEqual(len(tour), len(self.simple_points))
        self.assertGreater(tour_length(tour), 0)
        
    def test_association_tsp(self):
        """An explicit vertex count is honoured and shapes the fitted loop"""
        algo = AssociationTSP(n_vertices=8, max_iterations=5)
        loop = algo.solve(self.random_points)

        self.assertEqual(len(loop), 8)  # Explicit n_vertices wins over adaptive sizing
        self.assertEqual(loop.shape[1], 2)

    def test_association_adaptive_vertex_count(self):
        """Asking for adaptive sizing overrides the explicit vertex count"""
        algo = AssociationTSP(n_vertices=8, max_iterations=5,
                              adaptive_vertices=True, min_vertices=40)
        loop = algo.solve(self.random_points)

        self.assertGreaterEqual(len(loop), 40)

    def test_clustering_tsp(self):
        """Clustering interpolates the ordered centres into a finer loop"""
        algo = ClusteringTSP(n_clusters=5, n_interpolated_vertices=40)
        loop = algo.solve(self.random_points)

        self.assertEqual(len(loop), 40)
        self.assertEqual(loop.shape[1], 2)

class TestExportImport(unittest.TestCase):
    """Test data export and import functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.points = np.array([[0.5, 0.5], [0.2, 0.8]])
        self.tour_length_val = 4.0
        
    def test_export_import_data(self):
        """Test data export and import"""
        from utils import export_data, load_data
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_filename = f.name
            
        try:
            # Export data
            export_data(self.vertices, self.points, temp_filename, self.tour_length_val)
            
            # Check file exists
            self.assertTrue(os.path.exists(temp_filename))
            
            # Import data
            loaded_vertices, loaded_points, loaded_length = load_data(temp_filename)
            
            # Check data matches
            np.testing.assert_array_almost_equal(loaded_vertices, self.vertices)
            np.testing.assert_array_almost_equal(loaded_points, self.points)
            self.assertAlmostEqual(loaded_length, self.tour_length_val)
            
        finally:
            # Clean up
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)

class TestTimer(unittest.TestCase):
    """Test timer utility"""
    
    def test_timer_context_manager(self):
        """Test timer as context manager"""
        import time
        
        with Timer("Test operation") as timer:
            time.sleep(0.01)  # Sleep for 10ms
            
        # Timer should have measured some time
        self.assertGreater(timer.start_time, 0)

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test complete pipeline from data generation to solution"""
        # Generate data
        points = generate_clustered_points(20, 3, 0.05, 0.6, seed=5)

        # Test multiple algorithms with appropriate parameters
        algorithms = [
            ('nearest_neighbor', {}),
            ('association', {'n_vertices': 10, 'max_iterations': 20}),
            ('clustering', {'n_clusters': 5})
        ]

        for algo_name, kwargs in algorithms:
            with self.subTest(algorithm=algo_name):
                algo = get_algorithm(algo_name, **kwargs)
                tour = algo.solve(points)

                self.assertGreater(len(tour), 0)
                self.assertEqual(tour.shape[1], 2)

                # Tour length should be positive
                length = tour_length(tour)
                self.assertGreater(length, 0)

# Performance benchmarks (not run by default)
class TestPerformance(unittest.TestCase):
    """Performance tests"""
    
    @unittest.skip("Performance test - run manually")
    def test_algorithm_performance(self):
        """Test algorithm performance with larger datasets"""
        sizes = [50, 100, 200]
        algorithms = ['nearest_neighbor', 'two_opt', 'association']
        
        for size in sizes:
            points = generate_clustered_points(size)
            
            for algo_name in algorithms:
                with self.subTest(size=size, algorithm=algo_name):
                    with Timer(f"{algo_name} with {size} points"):
                        algo = get_algorithm(algo_name)
                        tour = algo.solve(points)
                        length = tour_length(tour)
                        
                    print(f"{algo_name} ({size} points): {length:.6f}")

def run_tests():
    """Run all tests"""
    test_classes = [
        TestConfig,
        TestUtils,
        TestSpatialIndex,
        TestAlgorithms,
        TestExportImport,
        TestTimer,
        TestIntegration,
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Pull in the sibling test modules too, so `python test_suite.py` runs
    # everything rather than silently covering a subset.
    for module_name in ('test_tours', 'test_interfaces'):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        suite.addTests(loader.loadTestsFromModule(module))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    raise SystemExit(0 if success else 1)
