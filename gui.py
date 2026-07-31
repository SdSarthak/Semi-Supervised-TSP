"""
Interactive GUI for Semi-Supervised TSP Visualizer
"""

import os
import time
from typing import Optional

import numpy as np

from backend import has_display, select_backend

# Binding the backend before pyplot is imported keeps this module importable
# on a machine with no display: it degrades to the file-writing Agg backend
# instead of raising at import time.
select_backend(interactive=True)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons

try:
    from tkinter import filedialog, messagebox
    HAS_TKINTER = True
except ImportError:  # pragma: no cover - depends on the Python build
    filedialog = None
    messagebox = None
    HAS_TKINTER = False

from config import Config
from main import algorithm_kwargs
from utils import (generate_clustered_points, tour_length,
                   compute_convergence_metric, adaptive_parameters, export_data)
from algorithms import get_algorithm


def _notify(kind: str, title: str, message: str) -> None:
    """Show a dialog when tkinter is present, otherwise print.

    tkinter is only used for the file and message dialogs, so a Python build
    without it should lose the dialogs, not the whole GUI.
    """
    if HAS_TKINTER:
        getattr(messagebox, f'show{kind}')(title, message)
    else:
        print(f"[{title}] {message}")


def _ask_open_path(title: str):
    """Prompt for a file to open, or return None when tkinter is unavailable"""
    if not HAS_TKINTER:
        print(f"[{title}] tkinter is not available; use cli.py to load files.")
        return None
    return filedialog.askopenfilename(
        title=title,
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")])


def _ask_save_path(title: str, default_extension: str, filetypes):
    """Prompt for a save destination, or return None when tkinter is unavailable"""
    if not HAS_TKINTER:
        print(f"[{title}] tkinter is not available; use cli.py to write files.")
        return None
    return filedialog.asksaveasfilename(
        title=title, defaultextension=default_extension, filetypes=filetypes)

class TSPVisualizer:
    """Main TSP visualization class with interactive controls"""

    def __init__(self, seed: Optional[int] = None):
        self.config = Config()
        self.config.validate()
        self.seed = self.config.SEED if seed is None else seed
        self.solution = None

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
        # A fresh stream each time, so the New Data button actually produces
        # new data instead of redrawing the same seeded point set.
        self.points = generate_clustered_points(
            self.config.N_POINTS,
            self.config.N_CLUSTERS_DATA,
            self.config.CLUSTER_STD,
            self.config.UNIFORM_RATIO,
            seed=self.seed,
        )
        self.seed += 1

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
            slider_value = int(self.sliders['n_vertices'].val)
            if algo_key == 'association':
                overrides = {'n_vertices': slider_value, 'adaptive_vertices': False}
            elif algo_key == 'clustering':
                overrides = {'n_clusters': slider_value}
            else:
                overrides = {}

            self.algorithm = get_algorithm(
                algo_key,
                **algorithm_kwargs(self.config, algo_key, seed=self.seed, **overrides)
            )
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
        if self.vertices is None:
            _notify('warning', "Nothing to Export", "Solve or run the animation first.")
            return

        try:
            os.makedirs(self.config.DEFAULT_EXPORT_DIR, exist_ok=True)

            timestamp = int(time.time())
            filename = os.path.join(self.config.DEFAULT_EXPORT_DIR,
                                    f"tsp_solution_{timestamp}.json")

            tour = None
            length = tour_length(self.vertices)
            if self.algorithm is not None and self.points is not None:
                # Export the tour over the real points, not just the loop.
                from utils import order_points_along_loop, tour_length_of_indices
                if self.algorithm.produces_loop and len(self.points) > 2:
                    tour = order_points_along_loop(self.points, self.vertices)
                    length = tour_length_of_indices(self.points, tour)

            export_data(self.vertices, self.points, filename, length,
                        tour=tour,
                        algorithm=self.algorithm.name if self.algorithm else None)

            _notify('info', "Export Successful", f"Data exported to {filename}")

        except Exception as e:
            _notify('error', "Export Error", f"Failed to export: {e}")

    def _on_load(self, event):
        """Handle load button click"""
        try:
            filename = _ask_open_path("Load TSP Data")

            if filename:
                from utils import load_data
                vertices, points, tour_length_val = load_data(filename)

                self.vertices = vertices
                self.points = points

                self._stop_animation()
                self._reset_state()
                self._update_plot()

                _notify('info', "Load Successful",
                        f"Loaded data with tour length: {tour_length_val:.3f}")

        except Exception as e:
            _notify('error', "Load Error", f"Failed to load: {e}")

    def _on_save_video(self, event):
        """Handle save video button click"""
        if len(self.tour_history) <= 10:  # Need some history to make video
            _notify('warning', "No Data", "Run animation first to generate video data")
            return

        try:
            filename = _ask_save_path(
                "Save Animation Video", ".mp4",
                [("MP4 files", "*.mp4"), ("GIF files", "*.gif"), ("All files", "*.*")])

            if filename:
                self._save_animation_video(filename)
                _notify('info', "Video Saved", f"Animation saved to {filename}")

        except Exception as e:
            _notify('error', "Save Error", f"Failed to save video: {e}")
            
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
        if self.algorithm is not None and self.points is not None:
            if self.algorithm.name == 'Association':
                from utils import init_circular_loop
                n_verts = int(self.sliders['n_vertices'].val)
                self.vertices = init_circular_loop(n_verts, seed=self.seed)
            else:
                # For other algorithms, start with a simple solution
                self.vertices = self.algorithm.solve(self.points)

    def _update_frame(self, frame):
        """Update animation frame"""
        if self.is_paused or not self.is_running:
            return

        self.current_iteration = frame

        try:
            # Only the association solver is genuinely iterative. Re-running
            # a whole genetic or annealing solve every few frames burnt
            # seconds per frame and made the window unresponsive, so the
            # non-iterative solvers are computed once and then just displayed.
            if self.algorithm.name == 'Association':
                self._update_association_algorithm()
            elif frame == 0 or self.vertices is None:
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

            # Accumulate every vertex's catchment in a single pass rather than
            # rescanning the assignment array once per vertex.
            sums = np.zeros_like(self.vertices)
            counts = np.bincount(nearest_indices,
                                 minlength=len(self.vertices)).astype(float)
            np.add.at(sums, nearest_indices, self.points)

            assigned = counts > 0
            new_vertices = self.vertices.copy()
            centroids = sums[assigned] / counts[assigned, None]
            new_vertices[assigned] = (self.vertices[assigned] * (1 - move_rate)
                                      + centroids * move_rate)

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

        try:
            anim.save(filename, fps=self.config.VIDEO_FPS, dpi=self.config.VIDEO_DPI)
        except Exception as exc:
            raise RuntimeError(
                f"Could not write {filename}. Saving MP4 needs ffmpeg on PATH; "
                f"choose a .gif filename to use the bundled Pillow writer instead. "
                f"Original error: {exc}"
            ) from exc
        finally:
            plt.close(fig_temp)

    def run(self):
        """Run the visualizer"""
        if not has_display():
            print("No interactive display is available, so the GUI cannot open.")
            print("Use 'python cli.py --help' for the batch interface, or set "
                  "MPLBACKEND to a GUI backend if one is installed.")
            return 1
        plt.show()
        return 0


def main():
    """Main function to run the interactive TSP visualizer"""
    if not has_display():
        print("No interactive display is available, so the GUI cannot open.")
        print("Use 'python cli.py --help' for the batch interface.")
        return 1
    visualizer = TSPVisualizer()
    return visualizer.run()


if __name__ == "__main__":
    raise SystemExit(main())
