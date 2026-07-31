"""
Enhanced main application with improved structure and features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import time

import numpy as np

from backend import has_display, select_backend
from config import Config
from utils import (generate_clustered_points, Timer, tour_length,
                   export_data, init_circular_loop, smooth_loop,
                   compute_convergence_metric, adaptive_parameters, SpatialIndex)
from algorithms import get_algorithm


def algorithm_kwargs(config, algorithm_name, seed=None, **overrides):
    """Build solver keyword arguments from a config object.

    Keeping this in one place stops the CLI, the GUI and the simple mode from
    drifting apart on what a given algorithm is configured with.
    """
    kwargs = {'seed': seed}

    if algorithm_name == 'association':
        kwargs.update(
            n_vertices=config.N_VERTICES,
            max_iterations=config.STEPS,
            adaptive_vertices=getattr(config, 'ADAPTIVE_VERTEX_DENSITY', True),
            subdivision_threshold=getattr(config, 'SUBDIVISION_THRESHOLD', 0.05),
            smoothing_iterations=getattr(config, 'SMOOTHING_ITERATIONS', 2),
            initial_move_rate=config.INITIAL_MOVE_RATE,
            initial_smooth_rate=config.INITIAL_SMOOTH_RATE,
            min_move_rate=config.MIN_MOVE_RATE,
            min_smooth_rate=config.MIN_SMOOTH_RATE,
            min_vertices=config.MIN_VERTICES,
            max_vertices=config.MAX_VERTICES,
            vertex_density_factor=config.VERTEX_DENSITY_FACTOR,
        )
    elif algorithm_name == 'clustering':
        kwargs.update(
            n_clusters=config.K_CLUSTERS,
            n_interpolated_vertices=config.N_VERTICES,
        )
    elif algorithm_name == 'genetic':
        kwargs.update(
            population_size=config.GA_POPULATION_SIZE,
            generations=config.GA_GENERATIONS,
            mutation_rate=config.GA_MUTATION_RATE,
            elite_size=config.GA_ELITE_SIZE,
            tournament_size=config.GA_TOURNAMENT_SIZE,
        )
    elif algorithm_name == 'simulated_annealing':
        kwargs.update(
            initial_temp=config.SA_INITIAL_TEMP,
            cooling_rate=config.SA_COOLING_RATE,
            min_temp=config.SA_MIN_TEMP,
            iterations_per_temp=config.SA_ITERATIONS_PER_TEMP,
        )

    kwargs.update(overrides)
    return kwargs


class EnhancedTSPVisualizer:
    """Enhanced TSP visualizer with multiple modes and algorithms"""

    def __init__(self, mode='gui', seed=None):
        self.config = Config()
        self.config.validate()

        self.mode = mode
        self.seed = self.config.SEED if seed is None else seed
        self.points = None
        self.vertices = None
        self.algorithm = None
        self.solution = None
        self.tour_history = []
        self.metrics_history = {'distances': [], 'convergence': [], 'times': []}

        # Animation state
        self.current_iteration = 0
        self.previous_vertices = None
        self.start_time = None

    def generate_data(self, n_points=None):
        """Generate test data"""
        n_points = n_points or self.config.N_POINTS

        print(f"Generating {n_points} test points...")
        self.points = generate_clustered_points(
            n_points,
            self.config.N_CLUSTERS_DATA,
            self.config.CLUSTER_STD,
            self.config.UNIFORM_RATIO,
            seed=self.seed,
        )
        print(f"Generated {len(self.points)} points")
        return self.points

    def set_algorithm(self, algorithm_name, **kwargs):
        """Set the TSP algorithm"""
        print(f"Setting algorithm: {algorithm_name}")

        seed = kwargs.pop('seed', self.seed)
        self.algorithm = get_algorithm(
            algorithm_name,
            **algorithm_kwargs(self.config, algorithm_name, seed=seed, **kwargs)
        )
        print(f"Algorithm set: {self.algorithm.name}")
        return self.algorithm

    def solve_static(self, refine=False):
        """Solve the TSP and report the tour length over the data points.

        Returns:
            The length of the tour through every input point. For the
            loop-based solvers this is the tour induced by the fitted loop,
            not the (always shorter) length of the loop itself.
        """
        if self.algorithm is None or self.points is None or len(self.points) == 0:
            raise ValueError("Algorithm and points must be set")

        print(f"Solving TSP with {self.algorithm.name}...")

        self.solution = self.algorithm.evaluate(self.points, refine=refine)
        self.vertices = self.solution.loop

        print(f"Solve took {self.solution.runtime:.3f} seconds")
        print(f"Solution found - Tour length: {self.solution.length:.6f}")
        if self.algorithm.produces_loop:
            print(f"Fitted loop length: {self.solution.loop_length:.6f} "
                  f"({len(self.solution.loop)} vertices)")

        return self.solution.length


    def run_animation(self, save_video=None, show_plot=True):
        """Run animated visualization.

        Args:
            save_video: path to write an MP4 to instead of opening a window.
            show_plot: False runs the iterations headless, printing progress.
        """
        if self.algorithm is None or self.points is None or len(self.points) == 0:
            raise ValueError("Algorithm and points must be set")

        print(f"Starting animation with {self.algorithm.name}...")

        # Initialize vertices for iterative algorithms
        if self.algorithm.name == 'Association':
            self.vertices = init_circular_loop(
                self.config.N_VERTICES, seed=self.seed
            )
        else:
            # For non-iterative algorithms, solve once and animate the result
            self.vertices = self.algorithm.solve(self.points)

        # Reset state
        self.current_iteration = 0
        self.tour_history = []
        self.metrics_history = {'distances': [], 'convergence': [], 'times': []}
        self.previous_vertices = None
        self.start_time = time.time()

        if not show_plot and not save_video:
            # Run the iterations without drawing anything.
            for frame in range(self.config.STEPS):
                self._update_frame(frame)
                if frame % 50 == 0:
                    distance = tour_length(self.vertices) if self.vertices is not None else 0
                    print(f"Frame {frame}: Loop length = {distance:.6f}")
            return None

        # Writing a video never needs a window; a live animation does.
        select_backend(interactive=save_video is None)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, axes = self._setup_visualization()

        anim = FuncAnimation(
            fig, self._update_frame,
            frames=self.config.STEPS,
            interval=self.config.INTERVAL_MS,
            blit=False,
            repeat=False
        )

        if save_video:
            print(f"Saving animation to {save_video}...")
            parent = os.path.dirname(os.path.abspath(save_video))
            os.makedirs(parent, exist_ok=True)
            try:
                anim.save(save_video, fps=self.config.VIDEO_FPS,
                          dpi=self.config.VIDEO_DPI)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not write {save_video}. Saving MP4 needs ffmpeg on PATH; "
                    f"use a .gif extension to fall back to the bundled Pillow writer. "
                    f"Original error: {exc}"
                ) from exc
            print("Animation saved!")
        elif has_display():
            plt.tight_layout()
            plt.show()
        else:
            print("No interactive display available; "
                  "pass --save-video to write the animation to a file instead.")

        return anim

    def _setup_visualization(self):
        """Setup the visualization plots"""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 6))

        # Main TSP plot
        ax1 = plt.subplot(1, 3, 1)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.set_title(f'Semi-Supervised TSP - {self.algorithm.name}', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Tour length plot
        ax2 = plt.subplot(1, 3, 2)
        ax2.set_title('Tour Length Over Time')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Tour Length')
        ax2.grid(True, alpha=0.3)
        
        # Convergence plot
        ax3 = plt.subplot(1, 3, 3)
        ax3.set_title('Convergence Metric')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Vertex Movement')
        ax3.grid(True, alpha=0.3)
        
        self.axes = [ax1, ax2, ax3]
        return fig, self.axes
        
    def _update_frame(self, frame):
        """Update animation frame"""
        self.current_iteration = frame
        
        # Update algorithm state
        if self.algorithm.name == 'Association':
            self._update_association_algorithm()
        elif frame == 0:  # For static algorithms, only compute once
            if self.vertices is None:
                self.vertices = self.algorithm.solve(self.points.copy())
                
        # Record metrics
        if self.vertices is not None:
            # Tour length
            distance = tour_length(self.vertices)
            self.metrics_history['distances'].append(distance)
            
            # Convergence
            convergence = compute_convergence_metric(self.vertices, self.previous_vertices)
            self.metrics_history['convergence'].append(convergence)
            
            # Time
            elapsed_time = time.time() - self.start_time
            self.metrics_history['times'].append(elapsed_time)
            
            # Store tour
            self.tour_history.append(self.vertices.copy())
            self.previous_vertices = self.vertices.copy()
            
            # Check for convergence
            if len(self.metrics_history['convergence']) >= self.config.CONVERGENCE_WINDOW:
                recent_conv = self.metrics_history['convergence'][-self.config.CONVERGENCE_WINDOW:]
                if np.mean(recent_conv) < self.config.CONVERGENCE_THRESHOLD:
                    print(f"Converged at iteration {frame}")
                    
        # Update visualization if axes exist
        if hasattr(self, 'axes'):
            self._update_plots()
            
        return []
        
    def _update_association_algorithm(self):
        """Update association algorithm step"""
        if self.vertices is None or self.points is None:
            return
            
        # Get adaptive parameters
        move_rate, smooth_rate = adaptive_parameters(
            self.current_iteration, self.config.STEPS,
            self.config.INITIAL_MOVE_RATE, self.config.INITIAL_SMOOTH_RATE,
            self.config.MIN_MOVE_RATE, self.config.MIN_SMOOTH_RATE
        )
        
        # Assign points to nearest vertices
        if len(self.points) > 0:
            spatial_index = SpatialIndex(self.vertices)
            _, nearest_indices = spatial_index.query_nearest(self.points)
            
            # Update vertices toward centroids
            new_vertices = self.vertices.copy()
            for v in range(len(self.vertices)):
                assigned_points = self.points[nearest_indices == v]
                if len(assigned_points) > 0:
                    centroid = np.mean(assigned_points, axis=0)
                    new_vertices[v] = (self.vertices[v] * (1 - move_rate) + 
                                     centroid * move_rate)
                                     
            # Apply smoothing
            self.vertices = smooth_loop(new_vertices, smooth_rate)
            
    def _update_plots(self):
        """Update all visualization plots"""
        if not getattr(self, 'axes', None):
            return

        # Clear all axes
        for ax in self.axes:
            ax.clear()
            
        # Main plot
        ax1 = self.axes[0]
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        
        # Plot points
        if self.points is not None:
            ax1.scatter(self.points[:, 0], self.points[:, 1], 
                       c='red', s=self.config.POINT_SIZE, alpha=self.config.ALPHA,
                       label=f'Points ({len(self.points)})')
                       
        # Plot tour
        if self.vertices is not None and len(self.vertices) > 0:
            # Close the loop
            tour_x = np.append(self.vertices[:, 0], self.vertices[0, 0])
            tour_y = np.append(self.vertices[:, 1], self.vertices[0, 1])
            
            ax1.plot(tour_x, tour_y, 'b-', linewidth=self.config.LINE_WIDTH,
                    label=f'Tour (length: {tour_length(self.vertices):.6f})')
                    
            # Plot vertices
            ax1.scatter(self.vertices[:, 0], self.vertices[:, 1],
                       c='blue', s=self.config.POINT_SIZE//2, alpha=0.8,
                       label=f'Vertices ({len(self.vertices)})')
                       
        ax1.set_title(f'{self.algorithm.name} - Iteration {self.current_iteration}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tour length plot
        ax2 = self.axes[1]
        if self.metrics_history['distances']:
            iterations = range(len(self.metrics_history['distances']))
            ax2.plot(iterations, self.metrics_history['distances'], 'g-', linewidth=2)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Tour Length')
            ax2.set_title('Tour Length Over Time')
            ax2.grid(True, alpha=0.3)
            
        # Convergence plot
        ax3 = self.axes[2]
        if self.metrics_history['convergence']:
            iterations = range(len(self.metrics_history['convergence']))
            ax3.plot(iterations, self.metrics_history['convergence'], 'r-', linewidth=2)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Vertex Movement')
            ax3.set_title('Convergence Metric')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
            
    def export_results(self, filename=None):
        """Export current results"""
        if filename is None:
            timestamp = int(time.time())
            filename = os.path.join(self.config.DEFAULT_EXPORT_DIR,
                                    f"tsp_result_{timestamp}.json")

        if self.solution is None:
            print("No results to export")
            return None

        export_data(self.solution.loop, self.points, filename,
                    self.solution.length, tour=self.solution.tour,
                    algorithm=self.solution.algorithm)
        return filename

    def compare_algorithms(self, algorithms, runs=1, refine=False):
        """Compare multiple algorithms on the same set of points.

        Every algorithm is scored on the length of the tour it produces over
        the input points, so loop-based and permutation solvers are measured
        on the same quantity. Spread across runs reflects each solver's own
        randomness; the point set is held fixed.
        """
        if self.points is None or len(self.points) == 0:
            raise ValueError("Points must be generated first")

        results = {}
        print(f"\nComparing {len(algorithms)} algorithms on {len(self.points)} points...")
        print("=" * 72)

        for algo_name in algorithms:
            print(f"\nTesting {algo_name}...")
            distances = []
            times = []

            for run in range(runs):
                try:
                    # Vary the stream per run so repeated runs of a stochastic
                    # solver actually explore, instead of reporting zero spread.
                    self.set_algorithm(algo_name, seed=self.seed + run)
                    distance = self.solve_static(refine=refine)

                    distances.append(distance)
                    times.append(self.solution.runtime)

                    if runs > 1:
                        print(f"  Run {run+1}: {distance:.6f} ({self.solution.runtime:.3f}s)")

                except Exception as e:
                    print(f"  Run {run+1}: FAILED - {e}")

            if distances:
                results[algo_name] = {
                    'avg_distance': float(np.mean(distances)),
                    'std_distance': float(np.std(distances)),
                    'best_distance': float(np.min(distances)),
                    'avg_time': float(np.mean(times)),
                    'runs': len(distances),
                }

                print(f"  Average: {results[algo_name]['avg_distance']:.6f} "
                      f"± {results[algo_name]['std_distance']:.6f} "
                      f"({results[algo_name]['avg_time']:.3f}s)")

        print("\n" + "=" * 72)
        print("COMPARISON SUMMARY (tour length over all input points)")
        print("=" * 72)

        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_distance'])
        for i, (algo_name, result) in enumerate(sorted_results):
            print(f"{i+1:2d}. {algo_name:20s} | "
                  f"Distance: {result['avg_distance']:8.6f} ± {result['std_distance']:8.6f} | "
                  f"Best: {result['best_distance']:8.6f} | "
                  f"Time: {result['avg_time']:6.3f}s")

        if not results:
            print("No algorithm completed successfully.")

        return results

def plot_solution(visualizer, save_plot=None, show=True):
    """Draw the points and the solved tour, optionally saving to a file"""
    select_backend(interactive=save_plot is None and show)
    import matplotlib.pyplot as plt

    solution = visualizer.solution
    fig = plt.figure(figsize=(10, 8))

    plt.scatter(visualizer.points[:, 0], visualizer.points[:, 1],
                c='red', s=20, alpha=0.7, label=f'Points ({len(visualizer.points)})')

    if solution is not None:
        ordered = solution.tour_points
        tour_x = np.append(ordered[:, 0], ordered[0, 0])
        tour_y = np.append(ordered[:, 1], ordered[0, 1])
        plt.plot(tour_x, tour_y, 'b-', linewidth=2,
                 label=f'Tour (length: {solution.length:.6f})')

        if visualizer.algorithm.produces_loop:
            loop = solution.loop
            loop_x = np.append(loop[:, 0], loop[0, 0])
            loop_y = np.append(loop[:, 1], loop[0, 1])
            plt.plot(loop_x, loop_y, 'g--', linewidth=1, alpha=0.6,
                     label=f'Fitted loop ({solution.loop_length:.6f})')

    plt.title(f'TSP Solution - {visualizer.algorithm.name}')
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_plot:
        os.makedirs(os.path.dirname(os.path.abspath(save_plot)), exist_ok=True)
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_plot}")
        plt.close(fig)
    elif show and has_display():
        plt.show()
    else:
        print("No interactive display available; use --save-plot to write a PNG.")
        plt.close(fig)


def build_parser():
    """Create the argument parser for the simple/GUI entry point"""
    parser = argparse.ArgumentParser(
        description='Semi-Supervised TSP Visualizer',
        epilog='For batch work and benchmarking use cli.py (or --mode cli).')

    parser.add_argument('--mode', choices=['gui', 'cli', 'simple'], default='simple',
                        help='Run mode (default: simple)')
    parser.add_argument('--algorithm', '-a', choices=Config.AVAILABLE_ALGORITHMS,
                        default='association', help='Algorithm to use')
    parser.add_argument('--points', '-p', type=int, default=200,
                        help='Number of points')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: Config.SEED)')
    parser.add_argument('--refine', action='store_true',
                        help='Polish the resulting tour with 2-opt')
    parser.add_argument('--animate', action='store_true',
                        help='Show animation')
    parser.add_argument('--compare', nargs='+', choices=Config.AVAILABLE_ALGORITHMS,
                        help='Compare multiple algorithms')
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Runs per algorithm when comparing')
    parser.add_argument('--save-video', type=str,
                        help='Save animation as video')
    parser.add_argument('--save-plot', type=str,
                        help='Save the static solution plot to an image file')
    parser.add_argument('--export', type=str,
                        help='Export results to a JSON file')
    parser.add_argument('--no-plot', action='store_true',
                        help='Run without showing plots')

    return parser


def split_cli_delegation(argv):
    """Detect `--mode cli` and return the arguments meant for cli.py.

    Both parsers define flags such as ``--points``, so letting this parser run
    first would swallow arguments intended for the CLI subcommand. Delegation
    is therefore decided by inspecting the raw argument list.

    Returns:
        The remaining arguments when CLI mode was requested, otherwise None.
    """
    argv = list(argv)

    for index, token in enumerate(argv):
        if token == '--mode' and index + 1 < len(argv) and argv[index + 1] == 'cli':
            return argv[:index] + argv[index + 2:]
        if token == '--mode=cli':
            return argv[:index] + argv[index + 1:]

    return None


def main(argv=None):
    """Main entry point"""
    if argv is None:
        argv = sys.argv[1:]

    delegated = split_cli_delegation(argv)
    if delegated is not None:
        from cli import main as cli_main
        return cli_main(delegated)

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == 'gui':
            from gui import main as gui_main
            return gui_main() or 0

        visualizer = EnhancedTSPVisualizer(mode=args.mode, seed=args.seed)
        visualizer.generate_data(args.points)

        if args.compare:
            visualizer.compare_algorithms(args.compare, runs=args.runs,
                                          refine=args.refine)
            return 0

        visualizer.set_algorithm(args.algorithm)

        if args.animate:
            visualizer.run_animation(save_video=args.save_video,
                                     show_plot=not args.no_plot)
            # The animation drives the loop directly, so produce a scored
            # solution as well for export and reporting.
            visualizer.solve_static(refine=args.refine)
        else:
            visualizer.solve_static(refine=args.refine)
            if not args.no_plot or args.save_plot:
                plot_solution(visualizer, save_plot=args.save_plot,
                              show=not args.no_plot)

        if args.export:
            visualizer.export_results(args.export)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
