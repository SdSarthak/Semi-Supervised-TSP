"""
Tests for the command-line entry points, config handling and file IO.

These run headless: the matplotlib backend is pinned to Agg before anything
imports pyplot, and no test opens a window or needs downloaded data.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np

import cli
import main as main_module
from config import Config
from utils import export_data, load_data, load_points, tour_length


class TestConfigValidation(unittest.TestCase):
    """Config guards"""

    def test_valid_config_passes(self):
        self.assertTrue(Config.validate())

    def test_out_of_range_value_raises_valueerror(self):
        """Explicit raises, not asserts, so `python -O` cannot skip validation"""
        original = Config.N_POINTS
        try:
            Config.N_POINTS = 0
            with self.assertRaises(ValueError):
                Config.validate()
        finally:
            Config.N_POINTS = original

    def test_to_dict_excludes_callables(self):
        config = Config.to_dict()
        self.assertIn('N_POINTS', config)
        self.assertNotIn('validate', config)
        self.assertNotIn('to_dict', config)


class TestBackendSelection(unittest.TestCase):
    """Backend choice must never explode on a headless machine"""

    def test_non_interactive_selection_returns_a_backend(self):
        from backend import select_backend
        self.assertTrue(select_backend(interactive=False))

    def test_gui_module_imports_without_a_display(self):
        import gui
        self.assertTrue(hasattr(gui, 'TSPVisualizer'))


class TestFileIO(unittest.TestCase):
    """Export and import round trips"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_export_to_bare_filename(self):
        """A filename with no directory used to crash os.makedirs("")"""
        cwd = os.getcwd()
        os.chdir(self.tmpdir.name)
        try:
            vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
            export_data(vertices, vertices, 'bare.json', tour_length(vertices))
            self.assertTrue(os.path.exists('bare.json'))
        finally:
            os.chdir(cwd)

    def test_export_creates_missing_directories(self):
        target = self.path(os.path.join('nested', 'deeper', 'out.json'))
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

        export_data(vertices, vertices, target, tour_length(vertices))

        self.assertTrue(os.path.exists(target))

    def test_round_trip_preserves_tour(self):
        target = self.path('solution.json')
        vertices = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        tour = [2, 0, 3, 1]

        export_data(vertices, vertices, target, 4.0, tour=tour, algorithm='2-Opt')
        loaded_vertices, loaded_points, length = load_data(target)

        np.testing.assert_allclose(loaded_vertices, vertices)
        np.testing.assert_allclose(loaded_points, vertices)
        self.assertAlmostEqual(length, 4.0)

        with open(target, encoding='utf-8') as f:
            payload = json.load(f)
        self.assertEqual(payload['tour'], tour)
        self.assertEqual(payload['algorithm'], '2-Opt')

    def test_load_data_reports_missing_fields(self):
        target = self.path('broken.json')
        with open(target, 'w', encoding='utf-8') as f:
            json.dump({'points': [[0, 0]]}, f)

        with self.assertRaises(ValueError) as ctx:
            load_data(target)
        self.assertIn('vertices', str(ctx.exception))

    def test_load_points_accepts_json_and_csv(self):
        points = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        json_path = self.path('points.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'points': points.tolist()}, f)
        np.testing.assert_allclose(load_points(json_path), points)

        csv_path = self.path('points.csv')
        np.savetxt(csv_path, points, delimiter=',')
        np.testing.assert_allclose(load_points(csv_path), points)

    def test_load_points_rejects_wrong_shape(self):
        csv_path = self.path('wide.csv')
        np.savetxt(csv_path, np.zeros((3, 4)), delimiter=',')
        with self.assertRaises(ValueError):
            load_points(csv_path)


class TestCliDispatch(unittest.TestCase):
    """End-to-end runs of the CLI subcommands"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_no_command_prints_help(self):
        self.assertEqual(cli.main([]), 0)

    def test_generate_writes_points(self):
        target = self.path('pts.json')
        self.assertEqual(cli.main(['generate', '--points', '20', '--output', target]), 0)

        with open(target, encoding='utf-8') as f:
            self.assertEqual(len(json.load(f)['points']), 20)

    def test_solve_every_algorithm(self):
        """`solve clustering --vertices N` used to raise TypeError"""
        for algorithm in Config.AVAILABLE_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                target = self.path(f'{algorithm}.json')
                code = cli.main(['solve', algorithm, '--points', '25',
                                 '--vertices', '10', '--output', target])
                self.assertEqual(code, 0)
                self.assertTrue(os.path.exists(target))

    def test_solve_reads_generated_points(self):
        points_path = self.path('roundtrip.json')
        cli.main(['generate', '--points', '15', '--output', points_path])
        self.assertEqual(
            cli.main(['solve', 'nearest_neighbor', '--input', points_path]), 0)

    def test_compare_runs(self):
        self.assertEqual(
            cli.main(['compare', 'nearest_neighbor', 'two_opt',
                      '--points', '25', '--runs', '2']), 0)

    def test_benchmark_writes_results(self):
        target = self.path('bench.json')
        code = cli.main(['benchmark', 'nearest_neighbor',
                         '--sizes', '15', '25', '--output', target])
        self.assertEqual(code, 0)

        with open(target, encoding='utf-8') as f:
            results = json.load(f)
        self.assertEqual(sorted(results), ['15', '25'])
        self.assertIn('distance', results['15']['nearest_neighbor'])

    def test_seed_makes_runs_reproducible(self):
        first, second = self.path('a.json'), self.path('b.json')
        for target in (first, second):
            cli.main(['--seed', '5', 'solve', 'two_opt',
                      '--points', '20', '--output', target])

        with open(first, encoding='utf-8') as f:
            a = json.load(f)
        with open(second, encoding='utf-8') as f:
            b = json.load(f)
        self.assertEqual(a['tour'], b['tour'])
        self.assertAlmostEqual(a['tour_length'], b['tour_length'])

    def test_missing_input_file_returns_error_code(self):
        self.assertEqual(
            cli.main(['solve', 'two_opt', '--input', self.path('nope.json')]), 1)


class TestMainDispatch(unittest.TestCase):
    """The simple-mode entry point"""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    def path(self, name):
        return os.path.join(self.tmpdir.name, name)

    def test_simple_solve_and_export(self):
        target = self.path('result.json')
        code = main_module.main(['--points', '30', '--algorithm', 'two_opt',
                                 '--no-plot', '--export', target])
        self.assertEqual(code, 0)
        self.assertTrue(os.path.exists(target))

    def test_save_plot_writes_an_image(self):
        target = self.path('plot.png')
        code = main_module.main(['--points', '25', '--algorithm', 'association',
                                 '--no-plot', '--save-plot', target])
        self.assertEqual(code, 0)
        self.assertGreater(os.path.getsize(target), 0)

    def test_compare_mode(self):
        self.assertEqual(
            main_module.main(['--compare', 'nearest_neighbor', 'two_opt',
                              '--points', '25', '--no-plot']), 0)

    def test_cli_delegation_keeps_shared_flags(self):
        """--points must reach the CLI parser, not be eaten by this one"""
        self.assertEqual(
            main_module.split_cli_delegation(
                ['--mode', 'cli', 'solve', 'two_opt', '--points', '40']),
            ['solve', 'two_opt', '--points', '40'])

        self.assertEqual(
            main_module.split_cli_delegation(
                ['--mode=cli', 'compare', 'two_opt']),
            ['compare', 'two_opt'])

        self.assertIsNone(
            main_module.split_cli_delegation(['--mode', 'simple', '--points', '40']))

    def test_delegated_run_uses_the_forwarded_point_count(self):
        target = self.path('delegated.json')
        code = main_module.main(['--mode', 'cli', 'solve', 'nearest_neighbor',
                                 '--points', '17', '--output', target])
        self.assertEqual(code, 0)

        with open(target, encoding='utf-8') as f:
            self.assertEqual(json.load(f)['n_points'], 17)

    def test_algorithm_kwargs_are_accepted_by_every_solver(self):
        from algorithms import get_algorithm
        for algorithm in Config.AVAILABLE_ALGORITHMS:
            with self.subTest(algorithm=algorithm):
                kwargs = main_module.algorithm_kwargs(Config, algorithm, seed=1)
                self.assertIsNotNone(get_algorithm(algorithm, **kwargs))


if __name__ == '__main__':
    unittest.main()
