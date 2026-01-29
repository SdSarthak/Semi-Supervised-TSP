"""
Interactive GUI for Semi-Supervised TSP Visualizer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable, Dict, Any
import threading
import queue
import time

from config import Config
from utils import (Timer, generate_clustered_points, tour_length, 
                   compute_convergence_metric, adaptive_parameters, export_data)
from algorithms import get_algorithm

class TSPVisualizer:
    """Main TSP visualization class with interactive controls"""
    
    def __init__(self):
        self.config = Config()
        self.config.validate()
        
        # Data
        self.points = None
        self.vertices = None
        self.tour_history = []
        self.convergence_history = []
        self.distance_history = []
        
        # Animation state
        self.animation = None
        self.is_running = False
        self.is_paused = False
        self.current_iteration = 0
        self.algorithm = None
        
        # GUI elements
        self.fig = None
        self.ax_main = None
        self.ax_metrics = None
        self.line_tour = None
        self.scatter_points = None
        self.scatter_vertices = None
        
        # Control widgets
        self.sliders = {}
        self.buttons = {}
        self.radio_buttons = None
        self.check_buttons = None
        
        # Metrics
        self.previous_vertices = None
        self.convergence_values = []
        
        self._setup_gui()
        self._generate_initial_data()
        
    def _setup_gui(self):
        """Setup the main GUI layout"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        
        # Main plot
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=2)
        self.ax_main.set_xlim(0, 1)
        self.ax_main.set_ylim(0, 1)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Semi-Supervised TSP Visualizer', fontsize=14, fontweight='bold')
        
        # Metrics plot
        self.ax_metrics = plt.subplot2grid((3, 4), (0, 2), colspan=2, rowspan=1)
        self.ax_metrics.set_title('Convergence Metrics', fontsize=12)
        self.ax_metrics.set_xlabel('Iteration')
        self.ax_metrics.set_ylabel('Tour Length / Convergence')
        
        # Algorithm selection
        ax_algo = plt.subplot2grid((3, 4), (1, 2), colspan=1, rowspan=1)
        self.radio_buttons = RadioButtons(ax_algo, 
                                        ['Association', 'Clustering', 'Nearest Neighbor', 
                                         '2-Opt', 'Genetic', 'Simulated Annealing'])
        self.radio_buttons.on_clicked(self._on_algorithm_change)
        
        # Control options
        ax_options = plt.subplot2grid((3, 4), (1, 3), colspan=1, rowspan=1)
        self.check_buttons = CheckButtons(ax_options, 
                                        ['Show Vertices', 'Adaptive Params', 'Auto Export'],
                                        [True, True, False])
        
        # Control sliders
        slider_height = 0.03
        slider_left = 0.1
        slider_width = 0.3
        
        # Move rate slider
        ax_move = plt.axes([slider_left, 0.25, slider_width, slider_height])
        self.sliders['move_rate'] = Slider(ax_move, 'Move Rate', 0.01, 1.0, 
                                          valinit=self.config.INITIAL_MOVE_RATE)
        
        # Smooth rate slider
        ax_smooth = plt.axes([slider_left, 0.20, slider_width, slider_height])
        self.sliders['smooth_rate'] = Slider(ax_smooth, 'Smooth Rate', 0.01, 1.0, 
                                           valinit=self.config.INITIAL_SMOOTH_RATE)
        
        # Animation speed slider
        ax_speed = plt.axes([slider_left, 0.15, slider_width, slider_height])
        self.sliders['speed'] = Slider(ax_speed, 'Speed (ms)', 10, 200, 
                                     valinit=self.config.INTERVAL_MS)
        
        # Number of vertices slider
        ax_vertices = plt.axes([slider_left, 0.10, slider_width, slider_height])
        self.sliders['n_vertices'] = Slider(ax_vertices, 'N Vertices', 10, 200, 
                                          valinit=self.config.N_VERTICES, valfmt='%d')
        
        # Control buttons
        button_width = 0.08
        button_height = 0.04
        button_left = 0.55
        
        # Start/Pause button
        ax_start = plt.axes([button_left, 0.25, button_width, button_height])
        self.buttons['start_pause'] = Button(ax_start, 'Start')
        self.buttons['start_pause'].on_clicked(self._on_start_pause)
        
        # Reset button
        ax_reset = plt.axes([button_left, 0.20, button_width, button_height])
        self.buttons['reset'] = Button(ax_reset, 'Reset')
        self.buttons['reset'].on_clicked(self._on_reset)
        
        # Generate data button
        ax_generate = plt.axes([button_left, 0.15, button_width, button_height])
        self.buttons['generate'] = Button(ax_generate, 'New Data')
        self.buttons['generate'].on_clicked(self._on_generate_data)
        
        # Export button
        ax_export = plt.axes([button_left, 0.10, button_width, button_height])
        self.buttons['export'] = Button(ax_export, 'Export')
        self.buttons['export'].on_clicked(self._on_export)
        
        # Load button
        ax_load = plt.axes([button_left + 0.09, 0.25, button_width, button_height])
        self.buttons['load'] = Button(ax_load, 'Load')
        self.buttons['load'].on_clicked(self._on_load)
        
        # Save video button
        ax_video = plt.axes([button_left + 0.09, 0.20, button_width, button_height])
        self.buttons['save_video'] = Button(ax_video, 'Save Video')
        self.buttons['save_video'].on_clicked(self._on_save_video)
        
        plt.tight_layout()
        
    def _generate_initial_data(self):
        """Generate initial random data"""
        self.points = generate_clustered_points(
            self.config.N_POINTS,
            self.config.N_CLUSTERS_DATA,
            self.config.CLUSTER_STD,
            self.config.UNIFORM_RATIO
        )
        
        # Initialize algorithm
        self._on_algorithm_change('Association')
        
        # Plot initial data
        self._update_plot()
        
    def _on_algorithm_change(self, algorithm_name: str):
        """Handle algorithm selection change"""
        algo_map = {
            'Association': 'association',
            'Clustering': 'clustering',
            'Nearest Neighbor': 'nearest_neighbor',
            '2-Opt': 'two_opt',
            'Genetic': 'genetic',
            'Simulated Annealing': 'simulated_annealing'
        }
        
        algo_key = algo_map.get(algorithm_name, 'association')
        
        try:
            if algo_key in ['association', 'clustering']:
                kwargs = {'n_vertices': int(self.sliders['n_vertices'].val)}
                if algo_key == 'clustering':
                    kwargs = {'n_clusters': int(self.sliders['n_vertices'].val)}
            else:
                kwargs = {}
                
            self.algorithm = get_algorithm(algo_key, **kwargs)
            print(f"Algorithm changed to: {self.algorithm.name}")
            
            # Reset for new algorithm
            self._reset_state()
            
        except Exception as e:
            print(f"Error changing algorithm: {e}")
            
    def _on_start_pause(self, event):
        """Handle start/pause button click"""
        if not self.is_running:
            self._start_animation()
        else:
            self._pause_animation()
            
    def _on_reset(self, event):
        """Handle reset button click"""
        self._stop_animation()
        self._reset_state()
        self._update_plot()
        
    def _on_generate_data(self, event):
        """Handle generate new data button click"""
        self._stop_animation()
        self._generate_initial_data()
        
    def _on_export(self, event):
        """Handle export button click"""
        if self.vertices is not None:
            try:
                import os
                os.makedirs(self.config.DEFAULT_EXPORT_DIR, exist_ok=True)
                
                timestamp = int(time.time())
                filename = f"{self.config.DEFAULT_EXPORT_DIR}/tsp_solution_{timestamp}.json"
                
                tour_length_val = tour_length(self.vertices)
                export_data(self.vertices, self.points, filename, tour_length_val)
                
                messagebox.showinfo("Export Successful", f"Data exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
                
    def _on_load(self, event):
        """Handle load button click"""
        try:
            filename = filedialog.askopenfilename(
                title="Load TSP Data",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                from utils import load_data
                vertices, points, tour_length_val = load_data(filename)
                
                self.vertices = vertices
                self.points = points
                
                self._stop_animation()
                self._reset_state()
                self._update_plot()
                
                messagebox.showinfo("Load Successful", 
                                  f"Loaded data with tour length: {tour_length_val:.3f}")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load: {e}")
            
    def _on_save_video(self, event):
        """Handle save video button click"""
        if len(self.tour_history) > 10:  # Need some history to make video
            try:
                filename = filedialog.asksaveasfilename(
                    title="Save Animation Video",
                    defaultextension=".mp4",
                    filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")]
                )
                
                if filename:
                    self._save_animation_video(filename)
                    messagebox.showinfo("Video Saved", f"Animation saved to {filename}")
                    
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save video: {e}")
        else:
            messagebox.showwarning("No Data", "Run animation first to generate video data")
            
    def _start_animation(self):
        """Start the animation"""
        if self.algorithm is None:
            return
            
        self.is_running = True
        self.is_paused = False
        self.buttons['start_pause'].label.set_text('Pause')
        
        self.animation = FuncAnimation(
            self.fig, self._update_frame,
            frames=self.config.STEPS,
            interval=int(self.sliders['speed'].val),
            blit=False,
            repeat=False
        )
        
        plt.draw()
        
    def _pause_animation(self):
        """Pause/resume the animation"""
        if self.is_paused:
            self.is_paused = False
            self.buttons['start_pause'].label.set_text('Pause')
            if self.animation:
                self.animation.resume()
        else:
            self.is_paused = True
            self.buttons['start_pause'].label.set_text('Resume')
            if self.animation:
                self.animation.pause()
                
    def _stop_animation(self):
        """Stop the animation"""
        self.is_running = False
        self.is_paused = False
        self.buttons['start_pause'].label.set_text('Start')
        
        if self.animation:
            self.animation.event_source.stop()
            self.animation = None
            
    def _reset_state(self):
        """Reset the algorithm state"""
        self.current_iteration = 0
        self.tour_history = []
        self.convergence_history = []
        self.distance_history = []
        self.previous_vertices = None
        self.convergence_values = []
        
        # Initialize vertices based on algorithm
        if self.algorithm and hasattr(self.algorithm, 'solve'):
            if self.algorithm.name in ['Association']:
                from utils import init_circular_loop
                n_verts = int(self.sliders['n_vertices'].val)
                self.vertices = init_circular_loop(n_verts)
            else:
                # For other algorithms, start with a simple solution
                self.vertices = self.algorithm.solve(self.points)
                
    def _update_frame(self, frame):
        """Update animation frame"""
        if self.is_paused or not self.is_running:
            return
            
        self.current_iteration = frame
        
        try:
            # Update algorithm
            if self.algorithm.name == 'Association':
                self._update_association_algorithm()
            elif self.algorithm.name == 'K-means Clustering':
                if frame == 0:  # Only solve once for clustering
                    self.vertices = self.algorithm.solve(self.points)
            else:
                if frame % 10 == 0:  # Update less frequently for heavy algorithms
                    self.vertices = self.algorithm.solve(self.points)
                    
            # Record metrics
            if self.vertices is not None:
                current_length = tour_length(self.vertices)
                self.distance_history.append(current_length)
                
                convergence = compute_convergence_metric(self.vertices, self.previous_vertices)
                self.convergence_values.append(convergence)
                self.convergence_history.append(convergence)
                
                self.tour_history.append(self.vertices.copy())
                self.previous_vertices = self.vertices.copy()
                
            # Update visualization
            self._update_plot()
            self._update_metrics_plot()
            
            # Check convergence
            if len(self.convergence_values) >= self.config.CONVERGENCE_WINDOW:
                recent_convergence = np.mean(self.convergence_values[-self.config.CONVERGENCE_WINDOW:])
                if recent_convergence < self.config.CONVERGENCE_THRESHOLD:
                    print(f"Converged at iteration {frame}")
                    self._stop_animation()
                    
        except Exception as e:
            print(f"Error in frame update: {e}")
            self._stop_animation()
            
        return self.line_tour, self.scatter_points
        
    def _update_association_algorithm(self):
        """Update vertices using association algorithm"""
        if self.vertices is None or self.points is None:
            return
            
        # Get current parameters
        if self.check_buttons.get_status()[1]:  # Adaptive parameters
            move_rate, smooth_rate = adaptive_parameters(
                self.current_iteration, self.config.STEPS,
                self.config.INITIAL_MOVE_RATE, self.config.INITIAL_SMOOTH_RATE
            )
        else:
            move_rate = self.sliders['move_rate'].val
            smooth_rate = self.sliders['smooth_rate'].val
            
        # Assign points to nearest vertices
        from utils import SpatialIndex, smooth_loop
        
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
            
    def _update_plot(self):
        """Update the main plot"""
        self.ax_main.clear()
        self.ax_main.set_xlim(0, 1)
        self.ax_main.set_ylim(0, 1)
        self.ax_main.set_aspect('equal')
        
        # Plot points
        if self.points is not None:
            self.scatter_points = self.ax_main.scatter(
                self.points[:, 0], self.points[:, 1],
                s=self.config.POINT_SIZE, c='red', alpha=self.config.ALPHA,
                label=f'Points ({len(self.points)})'
            )
            
        # Plot tour
        if self.vertices is not None:
            # Close the loop
            tour_x = np.append(self.vertices[:, 0], self.vertices[0, 0])
            tour_y = np.append(self.vertices[:, 1], self.vertices[0, 1])
            
            self.line_tour, = self.ax_main.plot(
                tour_x, tour_y, 'b-',
                linewidth=self.config.LINE_WIDTH,
                label=f'Tour (length: {tour_length(self.vertices):.3f})'
            )
            
            # Plot vertices if enabled
            if self.check_buttons.get_status()[0]:  # Show vertices
                self.scatter_vertices = self.ax_main.scatter(
                    self.vertices[:, 0], self.vertices[:, 1],
                    s=self.config.POINT_SIZE//2, c='blue', alpha=0.8,
                    label=f'Vertices ({len(self.vertices)})'
                )
                
        # Update title and legend
        title = f'Semi-Supervised TSP - {self.algorithm.name if self.algorithm else "No Algorithm"}'
        if self.is_running:
            title += f' (Iteration: {self.current_iteration})'
            
        self.ax_main.set_title(title, fontsize=14, fontweight='bold')
        self.ax_main.legend(loc='upper right')
        self.ax_main.grid(True, alpha=0.3)
        
    def _update_metrics_plot(self):
        """Update the metrics plot"""
        if not self.distance_history:
            return
            
        self.ax_metrics.clear()
        
        iterations = range(len(self.distance_history))
        
        # Plot tour length
        self.ax_metrics.plot(iterations, self.distance_history, 'b-', 
                           label='Tour Length', linewidth=2)
        
        # Plot convergence (scaled)
        if self.convergence_history:
            max_dist = max(self.distance_history) if self.distance_history else 1
            scaled_conv = [c * max_dist * 10 for c in self.convergence_history]
            self.ax_metrics.plot(iterations, scaled_conv, 'r--', 
                               label='Convergence (scaled)', alpha=0.7)
            
        self.ax_metrics.set_xlabel('Iteration')
        self.ax_metrics.set_ylabel('Value')
        self.ax_metrics.set_title('Performance Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)
        
    def _save_animation_video(self, filename: str):
        """Save animation as video"""
        # Create animation from history
        if not self.tour_history:
            raise ValueError("No animation history to save")
            
        fig_temp, ax_temp = plt.subplots(figsize=(10, 8))
        
        def animate_frame(frame):
            ax_temp.clear()
            ax_temp.set_xlim(0, 1)
            ax_temp.set_ylim(0, 1)
            ax_temp.set_aspect('equal')
            
            # Plot points
            ax_temp.scatter(self.points[:, 0], self.points[:, 1], 
                          s=20, c='red', alpha=0.7)
            
            # Plot tour
            vertices = self.tour_history[frame]
            tour_x = np.append(vertices[:, 0], vertices[0, 0])
            tour_y = np.append(vertices[:, 1], vertices[0, 1])
            ax_temp.plot(tour_x, tour_y, 'b-', linewidth=2)
            
            ax_temp.set_title(f'TSP Evolution - Frame {frame}')
            
        anim = FuncAnimation(fig_temp, animate_frame, frames=len(self.tour_history),
                           interval=100, blit=False)
        
        anim.save(filename, fps=self.config.VIDEO_FPS, dpi=self.config.VIDEO_DPI)
        plt.close(fig_temp)
        
    def run(self):
        """Run the visualizer"""
        plt.show()

def main():
    """Main function to run the interactive TSP visualizer"""
    visualizer = TSPVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
