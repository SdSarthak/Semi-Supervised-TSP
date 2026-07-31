"""
Command-line interface for Semi-Supervised TSP Visualizer
"""

import argparse
import json
import os
import sys

import numpy as np

from backend import has_display, select_backend
from config import Config
from main import algorithm_kwargs
from utils import (generate_clustered_points, tour_length, export_data,
                   load_points)
from algorithms import get_algorithm


class TSPCommandLine:
    """Command-line interface for TSP visualization"""

    def __init__(self, config: Config, seed: int = None, refine: bool = False):
        self.config = config
        self.seed = config.SEED if seed is None else seed
        self.refine = refine
        self.algorithm = None
        self.points = None
        self.vertices = None
        self.solution = None
        self.tour_history = []
        self.metrics = {'distances': [], 'times': []}

    def generate_data(self, n_points: int = None):
        """Generate random test data"""
        n_points = n_points or self.config.N_POINTS

        print(f"Generating {n_points} random points...")
        self.points = generate_clustered_points(
            n_points,
            self.config.N_CLUSTERS_DATA,
            self.config.CLUSTER_STD,
            self.config.UNIFORM_RATIO,
            seed=self.seed,
        )
        print(f"Generated {len(self.points)} points")
        return self.points

    def load_points_from_file(self, filename: str):
        """Load points from a JSON, CSV or whitespace-delimited file"""
        print(f"Loading points from {filename}...")
        self.points = load_points(filename)
        print(f"Loaded {len(self.points)} points")
        return self.points

    def solve_tsp(self, algorithm_name: str, seed: int = None, **overrides):
        """Solve TSP using the specified algorithm.

        Returns:
            Length of the tour through every input point.
        """
        if self.points is None:
            raise ValueError("No points loaded. Generate or load data first.")

        print(f"Solving TSP with {algorithm_name} algorithm...")

        self.algorithm = get_algorithm(
            algorithm_name,
            **algorithm_kwargs(self.config, algorithm_name,
                               seed=self.seed if seed is None else seed,
                               **overrides)
        )
        self.solution = self.algorithm.evaluate(self.points, refine=self.refine)
        self.vertices = self.solution.loop

        print(f"{algorithm_name} TSP took {self.solution.runtime:.3f} seconds")
        print(f"Tour length: {self.solution.length:.6f}")
        if self.algorithm.produces_loop:
            print(f"Fitted loop length: {self.solution.loop_length:.6f} "
                  f"({len(self.solution.loop)} vertices)")

        return self.solution.length

    def compare_algorithms(self, algorithms: list, runs: int = 1):
        """Compare multiple algorithms on the same points"""
        if self.points is None:
            raise ValueError("No points loaded. Generate or load data first.")

        results = {}

        print(f"\nComparing {len(algorithms)} algorithms with {runs} run(s) each...")
        print("-" * 80)

        for algo_name in algorithms:
            print(f"\nTesting {algo_name}...")
            distances = []
            times = []

            for run in range(runs):
                try:
                    # A fresh stream per run, otherwise repeat runs of a
                    # stochastic solver all report the same number.
                    distance = self.solve_tsp(algo_name, seed=self.seed + run)
                    distances.append(distance)
                    times.append(self.solution.runtime)

                    if runs > 1:
                        print(f"  Run {run+1}: {distance:.6f} ({self.solution.runtime:.3f}s)")

                except Exception as e:
                    print(f"  Run {run+1}: FAILED - {e}")
                    continue

            if distances:
                results[algo_name] = {
                    'avg_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'best_distance': float(np.min(distances)),
                    'avg_time': float(np.mean(times)),
                    'distances': [float(d) for d in distances],
                    'times': [float(t) for t in times],
                }

                print(f"  Average: {results[algo_name]['avg_distance']:.6f} "
                      f"+/- {results[algo_name]['std_distance']:.6f}")
                print(f"  Time: {results[algo_name]['avg_time']:.3f}s")

        print("\n" + "=" * 80)
        print("COMPARISON SUMMARY (tour length over all input points)")
        print("=" * 80)

        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_distance'])

        for i, (algo_name, result) in enumerate(sorted_results):
            print(f"{i+1:2d}. {algo_name:20s} | "
                  f"Distance: {result['avg_distance']:8.6f} "
                  f"+/- {result['std_distance']:8.6f} | "
                  f"Best: {result['best_distance']:8.6f} | "
                  f"Time: {result['avg_time']:6.3f}s")

        if not results:
            print("No algorithm completed successfully.")

        return results


    def animate_solution(self, algorithm_name: str, save_video: str = None, **overrides):
        """Create animated visualization of algorithm"""
        if self.points is None:
            raise ValueError("No points loaded. Generate or load data first.")

        print(f"Creating animation for {algorithm_name}...")

        # Writing a file never needs a window; a live animation does.
        select_backend(interactive=save_video is None)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        self.algorithm = get_algorithm(
            algorithm_name,
            **algorithm_kwargs(self.config, algorithm_name, seed=self.seed, **overrides)
        )

        # Initialize
        if algorithm_name == 'association':
            from utils import init_circular_loop
            self.vertices = init_circular_loop(
                overrides.get('n_vertices', self.config.N_VERTICES), seed=self.seed)
        else:
            self.vertices = self.algorithm.solve(self.points)

        # Setup plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Main plot
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_title(f'TSP - {algorithm_name}')
        
        # Metrics plot
        ax2.set_title('Tour Length Over Time')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Tour Length')
        
        # Plot elements
        ax1.scatter(self.points[:, 0], self.points[:, 1],
                    c='red', s=20, alpha=0.7, label='Points')
        tour_line, = ax1.plot([], [], 'b-', linewidth=2, label='Tour')
        vertices_scatter = ax1.scatter([], [], c='blue', s=10, alpha=0.8, label='Vertices')
        
        metrics_line, = ax2.plot([], [], 'g-', linewidth=2)
        
        ax1.legend()
        
        # Animation data
        self.tour_history = []
        distance_history = []
        
        def update_frame(frame):
            # Update algorithm (for iterative ones)
            if algorithm_name == 'association':
                self._update_association_step(frame)
                
            # Record data
            self.tour_history.append(self.vertices.copy())
            distance = tour_length(self.vertices)
            distance_history.append(distance)
            
            # Update tour plot
            tour_x = np.append(self.vertices[:, 0], self.vertices[0, 0])
            tour_y = np.append(self.vertices[:, 1], self.vertices[0, 1])
            tour_line.set_data(tour_x, tour_y)
            
            vertices_scatter.set_offsets(self.vertices)
            
            # Update metrics plot
            metrics_line.set_data(range(len(distance_history)), distance_history)
            ax2.relim()
            ax2.autoscale_view()
            
            # Update title
            ax1.set_title(f'TSP - {algorithm_name} | Iteration: {frame} | Length: {distance:.6f}')
            
            return tour_line, vertices_scatter, metrics_line
            
        # Create animation
        anim = FuncAnimation(fig, update_frame, frames=self.config.STEPS,
                           interval=self.config.INTERVAL_MS, blit=False)
        
        if save_video:
            print(f"Saving animation to {save_video}...")
            os.makedirs(os.path.dirname(os.path.abspath(save_video)), exist_ok=True)
            try:
                anim.save(save_video, fps=self.config.VIDEO_FPS,
                          dpi=self.config.VIDEO_DPI)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not write {save_video}. Saving MP4 needs ffmpeg on PATH; "
                    f"use a .gif extension to fall back to the bundled Pillow writer. "
                    f"Original error: {exc}"
                ) from exc
            print(f"Animation saved to {save_video}")
        elif has_display():
            plt.show()
        else:
            print("No interactive display available; "
                  "pass --output to write the animation to a file instead.")

        return anim

    def _update_association_step(self, iteration):
        """Update association algorithm step"""
        from utils import attract_vertices, smooth_loop, adaptive_parameters

        # Get adaptive parameters
        move_rate, smooth_rate = adaptive_parameters(
            iteration, self.config.STEPS,
            self.config.INITIAL_MOVE_RATE,
            self.config.INITIAL_SMOOTH_RATE,
            self.config.MIN_MOVE_RATE,
            self.config.MIN_SMOOTH_RATE,
        )

        if len(self.points) > 0:
            new_vertices = attract_vertices(self.points, self.vertices, move_rate)
            self.vertices = smooth_loop(new_vertices, smooth_rate)

    def export_solution(self, filename: str):
        """Export current solution"""
        if self.solution is None:
            raise ValueError("No solution to export. Solve first.")

        export_data(self.solution.loop, self.points, filename,
                    self.solution.length, tour=self.solution.tour,
                    algorithm=self.solution.algorithm)
        return filename

    def benchmark_performance(self, n_points_list: list, algorithms: list,
                              output: str = None):
        """Benchmark algorithms across a range of problem sizes"""
        results = {}

        print(f"\nBenchmarking {len(algorithms)} algorithms with problem sizes: {n_points_list}")
        print("=" * 100)

        for n_points in n_points_list:
            print(f"\nProblem size: {n_points} points")
            print("-" * 50)

            # Generate data for this size
            self.generate_data(n_points)

            size_results = {}

            for algo_name in algorithms:
                try:
                    distance = self.solve_tsp(algo_name)
                    size_results[algo_name] = {
                        'distance': float(distance),
                        'time': float(self.solution.runtime),
                    }
                    print(f"  {algo_name:20s}: {distance:8.6f} "
                          f"({self.solution.runtime:6.3f}s)")

                except Exception as e:
                    print(f"  {algo_name:20s}: FAILED - {e}")

            results[n_points] = size_results

        if output:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            with open(output, 'w', encoding='utf-8') as f:
                json.dump({str(k): v for k, v in results.items()}, f, indent=2)
            print(f"\nBenchmark results written to {output}")

        return results

def solver_overrides(algorithm: str, vertices: int = None):
    """Translate the shared --vertices flag into per-solver parameters.

    The flag means different things to different solvers, and passing
    ``n_vertices`` to the clustering solver used to raise a TypeError.
    """
    if not vertices:
        return {}
    if algorithm == 'association':
        return {'n_vertices': vertices, 'adaptive_vertices': False}
    if algorithm == 'clustering':
        return {'n_clusters': vertices}
    return {}


def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(description='Semi-Supervised TSP Visualizer')

    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: Config.SEED)')
    parser.add_argument('--refine', action='store_true',
                        help='Polish every tour with 2-opt before reporting')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Generate data command
    gen_parser = subparsers.add_parser('generate', help='Generate random test data')
    gen_parser.add_argument('--points', '-p', type=int, default=100,
                           help='Number of points to generate')
    gen_parser.add_argument('--output', '-o', type=str,
                           help='Output file to save points')
    
    # Solve command
    solve_parser = subparsers.add_parser('solve', help='Solve TSP')
    solve_parser.add_argument('algorithm', choices=Config.AVAILABLE_ALGORITHMS,
                             help='TSP algorithm to use')
    solve_parser.add_argument('--input', '-i', type=str,
                             help='Input file with points')
    solve_parser.add_argument('--points', '-p', type=int,
                             help='Generate random points instead')
    solve_parser.add_argument('--output', '-o', type=str,
                             help='Output file for solution')
    solve_parser.add_argument('--vertices', '-v', type=int, default=None,
                             help='Loop vertices (association) or clusters (clustering)')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare algorithms')
    compare_parser.add_argument('algorithms', nargs='+', 
                               choices=Config.AVAILABLE_ALGORITHMS,
                               help='Algorithms to compare')
    compare_parser.add_argument('--input', '-i', type=str,
                               help='Input file with points')
    compare_parser.add_argument('--points', '-p', type=int,
                               help='Generate random points instead')
    compare_parser.add_argument('--runs', '-r', type=int, default=1,
                               help='Number of runs per algorithm')
    
    # Animate command
    animate_parser = subparsers.add_parser('animate', help='Create animation')
    animate_parser.add_argument('algorithm', choices=Config.AVAILABLE_ALGORITHMS,
                               help='TSP algorithm to animate')
    animate_parser.add_argument('--input', '-i', type=str,
                               help='Input file with points')
    animate_parser.add_argument('--points', '-p', type=int,
                               help='Generate random points instead')
    animate_parser.add_argument('--output', '-o', type=str,
                               help='Output video file')
    animate_parser.add_argument('--vertices', '-v', type=int, default=None,
                               help='Loop vertices (association) or clusters (clustering)')

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    benchmark_parser.add_argument('algorithms', nargs='+',
                                 choices=Config.AVAILABLE_ALGORITHMS,
                                 help='Algorithms to benchmark')
    benchmark_parser.add_argument('--sizes', '-s', nargs='+', type=int,
                                 default=[50, 100, 200, 300],
                                 help='Problem sizes to test')
    benchmark_parser.add_argument('--output', '-o', type=str,
                                 help='Write benchmark results to a JSON file')

    # GUI command
    subparsers.add_parser('gui', help='Launch interactive GUI')

    return parser


def main(argv=None):
    """Main command-line interface.

    Args:
        argv: argument list to parse; defaults to ``sys.argv[1:]``. Accepting
            it explicitly lets ``main.py --mode cli`` forward its own
            leftover arguments instead of re-reading sys.argv, which used to
            make the two parsers fight over the same tokens.
    """
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    config = Config()
    config.validate()
    tsp = TSPCommandLine(config, seed=args.seed, refine=args.refine)

    try:
        if args.command == 'generate':
            tsp.generate_data(args.points)
            if args.output:
                os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump({'points': tsp.points.tolist()}, f, indent=2)
                print(f"Points saved to {args.output}")

        elif args.command == 'solve':
            if args.input:
                tsp.load_points_from_file(args.input)
            else:
                tsp.generate_data(args.points or 100)

            tsp.solve_tsp(args.algorithm,
                          **solver_overrides(args.algorithm, args.vertices))

            if args.output:
                tsp.export_solution(args.output)

        elif args.command == 'compare':
            if args.input:
                tsp.load_points_from_file(args.input)
            else:
                tsp.generate_data(args.points or 100)

            tsp.compare_algorithms(args.algorithms, args.runs)

        elif args.command == 'animate':
            if args.input:
                tsp.load_points_from_file(args.input)
            else:
                tsp.generate_data(args.points or 100)

            tsp.animate_solution(args.algorithm, args.output,
                                 **solver_overrides(args.algorithm, args.vertices))

        elif args.command == 'benchmark':
            tsp.benchmark_performance(args.sizes, args.algorithms,
                                      output=args.output)

        elif args.command == 'gui':
            from gui import main as gui_main
            return gui_main() or 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
