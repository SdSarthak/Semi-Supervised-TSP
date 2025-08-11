"""
Command-line interface for Semi-Supervised TSP Visualizer
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json
import time
import os

from config import Config
from utils import (generate_clustered_points, tour_length, Timer, 
                   export_data, load_data)
from algorithms import get_algorithm

class TSPCommandLine:
    """Command-line interface for TSP visualization"""
    
    def __init__(self, config: Config):
        self.config = config
        self.algorithm = None
        self.points = None
        self.vertices = None
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
            self.config.UNIFORM_RATIO
        )
        print(f"Generated {len(self.points)} points")
        
    def load_points_from_file(self, filename: str):
        """Load points from file"""
        print(f"Loading points from {filename}...")
        
        if filename.endswith('.json'):
            with open(filename, 'r') as f:
                data = json.load(f)
                self.points = np.array(data['points'])
        elif filename.endswith('.csv'):
            self.points = np.loadtxt(filename, delimiter=',')
        else:
            # Try loading as simple text file
            self.points = np.loadtxt(filename)
            
        print(f"Loaded {len(self.points)} points")
        
    def solve_tsp(self, algorithm_name: str, **kwargs):
        """Solve TSP using specified algorithm"""
        if self.points is None:
            raise ValueError("No points loaded. Generate or load data first.")
            
        print(f"Solving TSP with {algorithm_name} algorithm...")
        
        with Timer(f"{algorithm_name} TSP"):
            self.algorithm = get_algorithm(algorithm_name, **kwargs)
            self.vertices = self.algorithm.solve(self.points)
            
        distance = tour_length(self.vertices)
        print(f"Tour length: {distance:.6f}")
        print(f"Number of vertices: {len(self.vertices)}")
        
        return distance
        
    def compare_algorithms(self, algorithms: list, runs: int = 1):
        """Compare multiple algorithms"""
        if self.points is None:
            raise ValueError("No points loaded. Generate or load data first.")
            
        results = {}
        
        print(f"\nComparing {len(algorithms)} algorithms with {runs} runs each...")
        print("-" * 80)
        
        for algo_name in algorithms:
            print(f"\nTesting {algo_name}...")
            distances = []
            times = []
            
            for run in range(runs):
                start_time = time.time()
                
                try:
                    distance = self.solve_tsp(algo_name)
                    elapsed = time.time() - start_time
                    
                    distances.append(distance)
                    times.append(elapsed)
                    
                    if runs > 1:
                        print(f"  Run {run+1}: {distance:.6f} ({elapsed:.3f}s)")
                        
                except Exception as e:
                    print(f"  Run {run+1}: FAILED - {e}")
                    continue
                    
            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                avg_time = np.mean(times)
                
                results[algo_name] = {
                    'avg_distance': avg_distance,
                    'std_distance': std_distance,
                    'avg_time': avg_time,
                    'distances': distances,
                    'times': times
                }
                
                print(f"  Average: {avg_distance:.6f} ± {std_distance:.6f}")
                print(f"  Time: {avg_time:.3f}s")
                
        # Print summary
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_distance'])
        
        for i, (algo_name, result) in enumerate(sorted_results):
            print(f"{i+1:2d}. {algo_name:20s} | "
                  f"Distance: {result['avg_distance']:8.6f} ± {result['std_distance']:8.6f} | "
                  f"Time: {result['avg_time']:6.3f}s")
                  
        return results
        
    def animate_solution(self, algorithm_name: str, save_video: str = None, **kwargs):
        """Create animated visualization of algorithm"""
        if self.points is None:
            raise ValueError("No points loaded. Generate or load data first.")
            
        print(f"Creating animation for {algorithm_name}...")
        
        # Setup algorithm
        self.algorithm = get_algorithm(algorithm_name, **kwargs)
        
        # Initialize
        if algorithm_name == 'association':
            from utils import init_circular_loop
            self.vertices = init_circular_loop(kwargs.get('n_vertices', 50))
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
        points_scatter = ax1.scatter(self.points[:, 0], self.points[:, 1], 
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
            anim.save(save_video, fps=30, dpi=150)
            print(f"Animation saved to {save_video}")
        else:
            plt.show()
            
        return anim
        
    def _update_association_step(self, iteration):
        """Update association algorithm step"""
        from utils import SpatialIndex, smooth_loop, adaptive_parameters
        
        # Get adaptive parameters
        move_rate, smooth_rate = adaptive_parameters(
            iteration, self.config.STEPS,
            self.config.INITIAL_MOVE_RATE,
            self.config.INITIAL_SMOOTH_RATE
        )
        
        # Assign points to vertices
        if len(self.points) > 0:
            spatial_index = SpatialIndex(self.vertices)
            _, nearest_indices = spatial_index.query_nearest(self.points)
            
            # Update vertices
            new_vertices = self.vertices.copy()
            for v in range(len(self.vertices)):
                assigned_points = self.points[nearest_indices == v]
                if len(assigned_points) > 0:
                    centroid = np.mean(assigned_points, axis=0)
                    new_vertices[v] = (self.vertices[v] * (1 - move_rate) + 
                                     centroid * move_rate)
                                     
            # Smooth
            self.vertices = smooth_loop(new_vertices, smooth_rate)
            
    def export_solution(self, filename: str):
        """Export current solution"""
        if self.vertices is None:
            raise ValueError("No solution to export")
            
        distance = tour_length(self.vertices)
        export_data(self.vertices, self.points, filename, distance)
        print(f"Solution exported to {filename}")
        
    def benchmark_performance(self, n_points_list: list, algorithms: list):
        """Benchmark algorithms with different problem sizes"""
        results = {}
        
        print(f"\nBenchmarking {len(algorithms)} algorithms with problem sizes: {n_points_list}")
        print("="*100)
        
        for n_points in n_points_list:
            print(f"\nProblem size: {n_points} points")
            print("-" * 50)
            
            # Generate data for this size
            self.generate_data(n_points)
            
            size_results = {}
            
            for algo_name in algorithms:
                try:
                    start_time = time.time()
                    distance = self.solve_tsp(algo_name)
                    elapsed = time.time() - start_time
                    
                    size_results[algo_name] = {
                        'distance': distance,
                        'time': elapsed
                    }
                    
                    print(f"  {algo_name:20s}: {distance:8.6f} ({elapsed:6.3f}s)")
                    
                except Exception as e:
                    print(f"  {algo_name:20s}: FAILED - {e}")
                    
            results[n_points] = size_results
            
        return results

def create_parser():
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(description='Semi-Supervised TSP Visualizer')
    
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
    solve_parser.add_argument('--vertices', '-v', type=int, default=50,
                             help='Number of vertices (for association/clustering)')
    
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
    animate_parser.add_argument('--vertices', '-v', type=int, default=50,
                               help='Number of vertices')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark performance')
    benchmark_parser.add_argument('algorithms', nargs='+',
                                 choices=Config.AVAILABLE_ALGORITHMS,
                                 help='Algorithms to benchmark')
    benchmark_parser.add_argument('--sizes', '-s', nargs='+', type=int,
                                 default=[50, 100, 200, 300],
                                 help='Problem sizes to test')
    
    # GUI command
    gui_parser = subparsers.add_parser('gui', help='Launch interactive GUI')
    
    return parser

def main():
    """Main command-line interface"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
        
    config = Config()
    tsp = TSPCommandLine(config)
    
    try:
        if args.command == 'generate':
            tsp.generate_data(args.points)
            if args.output:
                data = {'points': tsp.points.tolist()}
                with open(args.output, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Points saved to {args.output}")
                
        elif args.command == 'solve':
            # Load or generate data
            if args.input:
                tsp.load_points_from_file(args.input)
            else:
                tsp.generate_data(args.points or 100)
                
            # Solve
            kwargs = {}
            if args.algorithm in ['association', 'clustering']:
                kwargs['n_vertices'] = args.vertices
                
            tsp.solve_tsp(args.algorithm, **kwargs)
            
            # Export if requested
            if args.output:
                tsp.export_solution(args.output)
                
        elif args.command == 'compare':
            # Load or generate data
            if args.input:
                tsp.load_points_from_file(args.input)
            else:
                tsp.generate_data(args.points or 100)
                
            # Compare
            tsp.compare_algorithms(args.algorithms, args.runs)
            
        elif args.command == 'animate':
            # Load or generate data
            if args.input:
                tsp.load_points_from_file(args.input)
            else:
                tsp.generate_data(args.points or 100)
                
            # Animate
            kwargs = {}
            if args.algorithm in ['association', 'clustering']:
                kwargs['n_vertices'] = args.vertices
                
            tsp.animate_solution(args.algorithm, args.output, **kwargs)
            
        elif args.command == 'benchmark':
            tsp.benchmark_performance(args.sizes, args.algorithms)
            
        elif args.command == 'gui':
            from gui import main as gui_main
            gui_main()
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
