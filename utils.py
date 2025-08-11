"""
Utility functions for Semi-Supervised TSP Visualizer
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import time

class Timer:
    """Simple timer context manager"""
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        print(f"{self.name} took {elapsed:.3f} seconds")

class SpatialIndex:
    """Spatial indexing for efficient nearest neighbor queries"""
    
    def __init__(self, points: np.ndarray):
        self.points = points
        self.tree = cKDTree(points)
        
    def query_nearest(self, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find nearest neighbors for query points"""
        distances, indices = self.tree.query(query_points)
        return distances, indices
        
    def query_k_nearest(self, query_points: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for query points"""
        distances, indices = self.tree.query(query_points, k=k)
        return distances, indices

def euclidean_distance_matrix(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """Compute euclidean distance matrix between two sets of points"""
    return cdist(points1, points2, metric='euclidean')

def tour_length(tour: np.ndarray) -> float:
    """Calculate total length of a tour"""
    if len(tour) < 2:
        return 0.0
    
    # Add first point at the end to close the loop
    closed_tour = np.vstack([tour, tour[0]])
    distances = np.sqrt(np.sum(np.diff(closed_tour, axis=0)**2, axis=1))
    return np.sum(distances)

def smooth_loop(vertices: np.ndarray, alpha: float, iterations: int = 1) -> np.ndarray:
    """Apply Laplacian smoothing to a closed loop with multiple iterations"""
    if len(vertices) < 3:
        return vertices
    
    smoothed_vertices = vertices.copy()
    
    for _ in range(iterations):
        prev_vertices = np.roll(smoothed_vertices, 1, axis=0)
        next_vertices = np.roll(smoothed_vertices, -1, axis=0)
        laplacian = (prev_vertices + next_vertices) / 2 - smoothed_vertices
        smoothed_vertices = smoothed_vertices + alpha * laplacian
        
    return smoothed_vertices

def adaptive_smooth_loop(vertices: np.ndarray, alpha: float, curvature_weight: float = 0.3) -> np.ndarray:
    """Apply adaptive Laplacian smoothing based on local curvature"""
    if len(vertices) < 3:
        return vertices
    
    # Calculate local curvature
    prev_vertices = np.roll(vertices, 1, axis=0)
    next_vertices = np.roll(vertices, -1, axis=0)
    
    # Vectors to neighbors
    v1 = prev_vertices - vertices
    v2 = next_vertices - vertices
    
    # Calculate curvature (angle between vectors)
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)
    
    # Avoid division by zero
    v1_norm = np.maximum(v1_norm, 1e-8)
    v2_norm = np.maximum(v2_norm, 1e-8)
    
    # Normalized vectors
    v1_unit = v1 / v1_norm[:, np.newaxis]
    v2_unit = v2 / v2_norm[:, np.newaxis]
    
    # Dot product for angle calculation
    dot_product = np.sum(v1_unit * v2_unit, axis=1)
    dot_product = np.clip(dot_product, -1, 1)
    
    # Curvature as (1 - cos(angle)) - higher for sharper turns
    curvature = 1 - dot_product
    
    # Adaptive smoothing strength based on curvature
    adaptive_alpha = alpha * (1 + curvature_weight * curvature)
    
    # Apply smoothing
    laplacian = (prev_vertices + next_vertices) / 2 - vertices
    smoothed_vertices = vertices + adaptive_alpha[:, np.newaxis] * laplacian
    
    return smoothed_vertices

def resample_curve(curve: np.ndarray, n_points: int) -> np.ndarray:
    """Resample a curve to have n_points uniformly distributed"""
    if len(curve) < 2:
        return curve
        
    # Calculate cumulative distances
    distances = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
    cumulative_distances = np.concatenate([[0], np.cumsum(distances)])
    
    # Create uniform sampling points
    total_length = cumulative_distances[-1]
    if total_length == 0:
        return np.tile(curve[0], (n_points, 1))
        
    uniform_distances = np.linspace(0, total_length, n_points, endpoint=False)
    
    # Interpolate
    resampled = []
    for target_dist in uniform_distances:
        # Find segment
        idx = np.searchsorted(cumulative_distances, target_dist) - 1
        idx = max(0, min(idx, len(curve) - 2))
        
        # Interpolate within segment
        local_dist = target_dist - cumulative_distances[idx]
        segment_length = distances[idx] if idx < len(distances) else 1e-9
        t = local_dist / max(segment_length, 1e-9)
        t = np.clip(t, 0, 1)
        
        point = curve[idx] * (1 - t) + curve[(idx + 1) % len(curve)] * t
        resampled.append(point)
    
    return np.array(resampled)

def generate_clustered_points(n_points: int, n_clusters: int = 5, 
                            cluster_std: float = 0.05, 
                            uniform_ratio: float = 0.6) -> np.ndarray:
    """Generate a mix of clustered and uniform random points"""
    np.random.seed(42)  # For reproducibility
    
    n_uniform = int(n_points * uniform_ratio)
    n_clustered = n_points - n_uniform
    
    points = []
    
    # Generate clustered points
    if n_clustered > 0:
        points_per_cluster = n_clustered // n_clusters
        remainder = n_clustered % n_clusters
        
        for i in range(n_clusters):
            n_cluster_points = points_per_cluster + (1 if i < remainder else 0)
            if n_cluster_points > 0:
                # Random cluster center
                center = np.random.uniform(0.2, 0.8, size=2)
                cluster_points = center + cluster_std * np.random.randn(n_cluster_points, 2)
                points.append(cluster_points)
    
    # Generate uniform points
    if n_uniform > 0:
        uniform_points = np.random.rand(n_uniform, 2)
        points.append(uniform_points)
    
    # Combine and clip to valid range
    all_points = np.vstack(points) if points else np.empty((0, 2))
    all_points = np.clip(all_points, 0.02, 0.98)
    
    return all_points

def init_circular_loop(n_vertices: int, center: Tuple[float, float] = (0.5, 0.5), 
                      radius: float = 0.35, noise_std: float = 0.02) -> np.ndarray:
    """Initialize a noisy circular loop"""
    angles = np.linspace(0, 2*np.pi, n_vertices, endpoint=False)
    r = radius + noise_std * np.random.randn(n_vertices)
    
    xs = center[0] + r * np.cos(angles)
    ys = center[1] + r * np.sin(angles)
    vertices = np.column_stack([xs, ys])
    
    # Add random jitter
    vertices += noise_std * np.random.randn(n_vertices, 2)
    vertices = np.clip(vertices, 0.02, 0.98)
    
    return vertices

def calculate_adaptive_vertex_count(data_points: np.ndarray, min_vertices: int = 60, 
                                   max_vertices: int = 300, density_factor: float = 0.4) -> int:
    """Calculate optimal number of vertices based on data point density"""
    n_points = len(data_points)
    
    # Base number of vertices proportional to data points
    adaptive_count = int(n_points * density_factor)
    
    # Add extra vertices for complex data distributions
    if n_points > 10:
        # Calculate data spread
        std_x = np.std(data_points[:, 0])
        std_y = np.std(data_points[:, 1])
        complexity_factor = (std_x + std_y) * 2  # Higher spread = more complexity
        
        adaptive_count += int(complexity_factor * 50)
    
    # Clamp to min/max bounds
    return max(min_vertices, min(adaptive_count, max_vertices))

def subdivide_vertices(vertices: np.ndarray, subdivision_threshold: float = 0.05) -> np.ndarray:
    """Subdivide edges that are too long to maintain fine resolution"""
    if len(vertices) < 3:
        return vertices
    
    new_vertices = []
    n = len(vertices)
    
    for i in range(n):
        current = vertices[i]
        next_vertex = vertices[(i + 1) % n]
        
        # Always add current vertex
        new_vertices.append(current)
        
        # Check if edge is too long
        edge_length = np.linalg.norm(next_vertex - current)
        
        if edge_length > subdivision_threshold:
            # Calculate number of subdivisions needed
            num_subdivisions = int(np.ceil(edge_length / subdivision_threshold)) - 1
            
            # Add intermediate vertices
            for j in range(1, num_subdivisions + 1):
                t = j / (num_subdivisions + 1)
                intermediate = current * (1 - t) + next_vertex * t
                new_vertices.append(intermediate)
    
    return np.array(new_vertices)

def remove_redundant_vertices(vertices: np.ndarray, min_distance: float = 0.01) -> np.ndarray:
    """Remove vertices that are too close together"""
    if len(vertices) < 4:  # Keep minimum for a loop
        return vertices
    
    filtered_vertices = [vertices[0]]  # Always keep first vertex
    
    for i in range(1, len(vertices)):
        current = vertices[i]
        last_kept = filtered_vertices[-1]
        
        # Check distance to last kept vertex
        if np.linalg.norm(current - last_kept) >= min_distance:
            filtered_vertices.append(current)
    
    # Ensure we have enough vertices for a proper loop
    if len(filtered_vertices) < 3:
        return vertices
    
    return np.array(filtered_vertices)

def optimize_vertex_distribution(vertices: np.ndarray, target_spacing: Optional[float] = None) -> np.ndarray:
    """Redistribute vertices along the loop for more uniform spacing"""
    if len(vertices) < 3:
        return vertices
    
    # Calculate current edge lengths
    n = len(vertices)
    edges = np.array([vertices[(i + 1) % n] - vertices[i] for i in range(n)])
    edge_lengths = np.linalg.norm(edges, axis=1)
    total_length = np.sum(edge_lengths)
    
    if total_length == 0:
        return vertices
    
    # If no target spacing provided, use average
    if target_spacing is None:
        target_spacing = total_length / n
    
    # Calculate cumulative distances
    cumulative_distances = np.concatenate([[0], np.cumsum(edge_lengths)])
    
    # Create new uniform sampling points
    new_positions = np.linspace(0, total_length, n, endpoint=False)
    
    # Interpolate new vertex positions
    new_vertices = []
    for pos in new_positions:
        # Find which edge this position falls on
        edge_idx = np.searchsorted(cumulative_distances[1:], pos)
        edge_idx = min(edge_idx, n - 1)
        
        # Interpolate within the edge
        if edge_idx == 0:
            t = pos / max(edge_lengths[0], 1e-8)
        else:
            local_pos = pos - cumulative_distances[edge_idx]
            t = local_pos / max(edge_lengths[edge_idx], 1e-8)
        
        t = np.clip(t, 0, 1)
        
        start_vertex = vertices[edge_idx]
        end_vertex = vertices[(edge_idx + 1) % n]
        
        new_vertex = start_vertex * (1 - t) + end_vertex * t
        new_vertices.append(new_vertex)
    
    return np.array(new_vertices)

def compute_convergence_metric(current_vertices: np.ndarray, 
                             previous_vertices: np.ndarray) -> float:
    """Compute convergence metric based on vertex movement"""
    if previous_vertices is None or len(current_vertices) != len(previous_vertices):
        return float('inf')
    
    movement = np.sqrt(np.sum((current_vertices - previous_vertices)**2, axis=1))
    return np.mean(movement)

def adaptive_parameters(iteration: int, total_iterations: int, 
                       initial_move_rate: float, initial_smooth_rate: float,
                       min_move_rate: float = 0.01, min_smooth_rate: float = 0.1) -> Tuple[float, float]:
    """Compute adaptive parameters that decrease over time"""
    progress = iteration / max(total_iterations, 1)
    
    # Exponential decay
    move_rate = max(min_move_rate, initial_move_rate * np.exp(-3 * progress))
    smooth_rate = max(min_smooth_rate, initial_smooth_rate * np.exp(-2 * progress))
    
    return move_rate, smooth_rate

def export_data(vertices: np.ndarray, points: np.ndarray, 
                filename: str, tour_length_val: float) -> None:
    """Export tour data to file"""
    import json
    import os
    
    # Create exports directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    data = {
        'vertices': vertices.tolist(),
        'points': points.tolist(),
        'tour_length': tour_length_val,
        'n_vertices': len(vertices),
        'n_points': len(points),
        'timestamp': time.time()
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data exported to {filename}")

def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load tour data from file"""
    import json
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    vertices = np.array(data['vertices'])
    points = np.array(data['points'])
    tour_length_val = data['tour_length']
    
    return vertices, points, tour_length_val
