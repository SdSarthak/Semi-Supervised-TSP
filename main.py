"""
Enhanced main application with improved structure and features
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better GUI support
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse
import time

from config import Config
from utils import (generate_clustered_points, Timer, tour_length,
                   export_data, init_circular_loop, smooth_loop,
                   compute_convergence_metric, adaptive_parameters, SpatialIndex)
from algorithms import get_algorithm

class EnhancedTSPVisualizer:
    """Enhanced TSP visualizer with multiple modes and algorithms"""
    
    def __init__(self, mode='gui'):
        self.config = Config()
        self.config.validate()
        
        self.mode = mode
        self.points = None
        self.vertices = None
        self.algorithm = None
        self.tour_history = []
        self.metrics_history = {'distances': [], 'convergence': [], 'times': []}
        
        # Animation state
        self.current_iteration = 0
        self.previous_vertices = None
        self.start_time = None
        
        np.random.seed(self.config.SEED)
        
    def generate_data(self, n_points=None):
        """Generate test data"""
        n_points = n_points or self.config.N_POINTS
        
        print(f"Generating {n_points} test points...")
        self.points = generate_clustered_points(
            n_points,
            self.config.N_CLUSTERS_DATA,
            self.config.CLUSTER_STD,
            self.config.UNIFORM_RATIO
        )
        print(f"Generated {len(self.points)} points")
        
    def set_algorithm(self, algorithm_name, **kwargs):
        """Set the TSP algorithm"""
        print(f"Setting algorithm: {algorithm_name}")
        
        # Set default parameters based on algorithm with enhanced vertex counts
        if algorithm_name == 'association':
            kwargs.setdefault('n_vertices', self.config.N_VERTICES)
            kwargs.setdefault('max_iterations', self.config.STEPS)
            kwargs.setdefault('adaptive_vertices', getattr(self.config, 'ADAPTIVE_VERTEX_DENSITY', True))
            kwargs.setdefault('subdivision_threshold', getattr(self.config, 'SUBDIVISION_THRESHOLD', 0.05))
        elif algorithm_name == 'clustering':
            kwargs.setdefault('n_clusters', self.config.K_CLUSTERS)
            kwargs.setdefault('n_interpolated_vertices', self.config.N_VERTICES)
        elif algorithm_name == 'genetic':
            kwargs.setdefault('population_size', self.config.GA_POPULATION_SIZE)
            kwargs.setdefault('generations', self.config.GA_GENERATIONS)
        elif algorithm_name == 'simulated_annealing':
            kwargs.setdefault('initial_temp', self.config.SA_INITIAL_TEMP)
            kwargs.setdefault('cooling_rate', self.config.SA_COOLING_RATE)
            
        self.algorithm = get_algorithm(algorithm_name, **kwargs)
        print(f"Algorithm set: {self.algorithm.name}")
        if hasattr(self.algorithm, 'n_vertices'):
            print(f"Using {self.algorithm.n_vertices} vertices for finer resolution")
        
    def solve_static(self):
        """Solve TSP with static algorithm (non-iterative)"""
        if self.algorithm is None or self.points is None or len(self.points) == 0:
            raise ValueError("Algorithm and points must be set")
            
        print(f"Solving TSP with {self.algorithm.name}...")
        
        with Timer(f"{self.algorithm.name} solve"):
            self.vertices = self.algorithm.solve(self.points.copy())
            
        distance = tour_length(self.vertices)
        print(f"Solution found - Tour length: {distance:.6f}")
        print(f"Number of vertices: {len(self.vertices)}")
        
        return distance
        
    def run_animation(self, save_video=None, show_plot=True):
        """Run animated visualization"""
        if self.algorithm is None or self.points is None or len(self.points) == 0:
            raise ValueError("Algorithm and points must be set")
            
        print(f"Starting animation with {self.algorithm.name}...")
        
        # Initialize vertices for iterative algorithms
        if self.algorithm.name in ['Association']:
            self.vertices = init_circular_loop(
                getattr(self.algorithm, 'n_vertices', self.config.N_VERTICES)
            )
        else:
            # For non-iterative algorithms, solve once and animate the result
            self.vertices = self.algorithm.solve(self.points.copy())
            
        # Reset state
        self.current_iteration = 0
        self.tour_history = []
        self.metrics_history = {'distances': [], 'convergence': [], 'times': []}
        self.previous_vertices = None
        self.start_time = time.time()
        
        # Setup visualization
        if show_plot:
            fig, axes = self._setup_visualization()
            
            # Create animation
            anim = FuncAnimation(
                fig, self._update_frame,
                frames=self.config.STEPS,
                interval=self.config.INTERVAL_MS,
                blit=False,
                repeat=False
            )
            
            # Save or show
            if save_video:
                print(f"Saving animation to {save_video}...")
                anim.save(save_video, fps=self.config.VIDEO_FPS, 
                         dpi=self.config.VIDEO_DPI)
                print(f"Animation saved!")
            else:
                plt.tight_layout()
                plt.show()
                
            return anim
        else:
            # Run without visualization
            for frame in range(self.config.STEPS):
                self._update_frame(frame)
                if frame % 50 == 0:
                    distance = tour_length(self.vertices) if self.vertices is not None else 0
                    print(f"Frame {frame}: Tour length = {distance:.6f}")
                    
    def _setup_visualization(self):
        """Setup the visualization plots"""
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
            filename = f"tsp_result_{timestamp}.json"
            
        if self.vertices is not None:
            distance = tour_length(self.vertices)
            export_data(self.vertices, self.points, filename, distance)
            print(f"Results exported to {filename}")
        else:
            print("No results to export")
            
    def compare_algorithms(self, algorithms, runs=1):
        """Compare multiple algorithms"""
        if self.points is None or len(self.points) == 0:
            raise ValueError("Points must be generated first")
            
        results = {}
        print(f"\nComparing {len(algorithms)} algorithms...")
        print("=" * 60)
        
        for algo_name in algorithms:
            print(f"\nTesting {algo_name}...")
            distances = []
            times = []
            
            for run in range(runs):
                try:
                    self.set_algorithm(algo_name)
                    
                    start_time = time.time()
                    distance = self.solve_static()
                    elapsed = time.time() - start_time
                    
                    distances.append(distance)
                    times.append(elapsed)
                    
                    if runs > 1:
                        print(f"  Run {run+1}: {distance:.6f} ({elapsed:.3f}s)")
                        
                except Exception as e:
                    print(f"  Run {run+1}: FAILED - {e}")
                    
            if distances:
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                avg_time = np.mean(times)
                
                results[algo_name] = {
                    'avg_distance': avg_distance,
                    'std_distance': std_distance,
                    'avg_time': avg_time
                }
                
                print(f"  Average: {avg_distance:.6f} ± {std_distance:.6f} ({avg_time:.3f}s)")
                
        # Print comparison summary
        print("\n" + "=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_distance'])
        for i, (algo_name, result) in enumerate(sorted_results):
            print(f"{i+1:2d}. {algo_name:20s} | "
                  f"Distance: {result['avg_distance']:8.6f} ± {result['std_distance']:8.6f} | "
                  f"Time: {result['avg_time']:6.3f}s")
                  
        return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Enhanced Semi-Supervised TSP Visualizer')
    
    parser.add_argument('--mode', choices=['gui', 'cli', 'simple'], default='simple',
                       help='Run mode (default: simple)')
    parser.add_argument('--algorithm', '-a', choices=Config.AVAILABLE_ALGORITHMS,
                       default='association', help='Algorithm to use')
    parser.add_argument('--points', '-p', type=int, default=200,
                       help='Number of points')
    parser.add_argument('--animate', action='store_true',
                       help='Show animation')
    parser.add_argument('--compare', nargs='+', choices=Config.AVAILABLE_ALGORITHMS,
                       help='Compare multiple algorithms')
    parser.add_argument('--save-video', type=str,
                       help='Save animation as video')
    parser.add_argument('--export', type=str,
                       help='Export results to file')
    parser.add_argument('--no-plot', action='store_true',
                       help='Run without showing plots')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'gui':
            # Launch GUI
            from gui import main as gui_main
            gui_main()
            
        elif args.mode == 'cli':
            # Launch CLI
            from cli import main as cli_main
            cli_main()
            
        else:
            # Simple mode
            visualizer = EnhancedTSPVisualizer()
            
            # Generate data
            visualizer.generate_data(args.points)
            
            if args.compare:
                # Compare algorithms
                visualizer.compare_algorithms(args.compare)
            else:
                # Single algorithm
                visualizer.set_algorithm(args.algorithm)
                
                if args.animate:
                    # Run animation
                    visualizer.run_animation(
                        save_video=args.save_video,
                        show_plot=not args.no_plot
                    )
                else:
                    # Static solve
                    distance = visualizer.solve_static()
                    
                    if not args.no_plot:
                        # Show static plot
                        plt.figure(figsize=(10, 8))
                        plt.scatter(visualizer.points[:, 0], visualizer.points[:, 1],
                                  c='red', s=20, alpha=0.7, label='Points')
                        
                        if visualizer.vertices is not None:
                            tour_x = np.append(visualizer.vertices[:, 0], visualizer.vertices[0, 0])
                            tour_y = np.append(visualizer.vertices[:, 1], visualizer.vertices[0, 1])
                            plt.plot(tour_x, tour_y, 'b-', linewidth=2, 
                                   label=f'Tour (length: {distance:.6f})')
                                   
                        plt.title(f'TSP Solution - {visualizer.algorithm.name}')
                        plt.legend()
                        plt.axis('equal')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.show()
                        
                # Export if requested
                if args.export:
                    visualizer.export_results(args.export)
                    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
