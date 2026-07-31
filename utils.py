"""
Utility functions for Semi-Supervised TSP Visualizer
"""

import json
import os
import time
import warnings
from typing import Tuple, Optional, Sequence, Union

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

# Points are generated inside the unit square with this much margin, so a
# tour never sits exactly on the plot border.
COORD_MIN = 0.02
COORD_MAX = 0.98

SeedLike = Optional[Union[int, np.random.Generator]]


def as_generator(seed: SeedLike = None) -> np.random.Generator:
    """Coerce a seed (or an existing Generator) into a numpy Generator.

    Using a local generator keeps the helpers below free of global
    ``np.random`` side effects, so seeding one call never silently pins the
    randomness of unrelated code that runs afterwards.
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


class Timer:
    """Simple timer context manager"""
    def __init__(self, name: str = "Operation", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.verbose:
            print(f"{self.name} took {self.elapsed:.3f} seconds")

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


def validate_points(points, name: str = "points") -> np.ndarray:
    """Coerce input to an ``(n, 2)`` float array, rejecting unusable data.

    NaN and infinite coordinates do not raise anywhere downstream: the KD-tree
    returns an out-of-range neighbour index, ``np.argmin`` picks the first NaN
    instead of the nearest point, and every reported tour length comes back as
    NaN. The result looks like a solution but is meaningless, and ``json.dump``
    then writes a literal ``NaN`` that strict JSON parsers reject. Catching it
    at the boundary turns a silent wrong answer into an actionable error.

    Raises:
        ValueError: if the input is not a rectangular ``(n, 2)`` array of
            finite numbers.
    """
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a rectangular array of numeric (x, y) pairs: {exc}"
        ) from exc

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"{name} must be an (n, 2) array of 2-D coordinates, got shape {array.shape}"
        )

    if array.size:
        finite = np.isfinite(array)
        if not finite.all():
            bad = np.flatnonzero(~finite.all(axis=1))
            preview = ', '.join(str(int(i)) for i in bad[:5])
            more = '' if len(bad) <= 5 else f" (and {len(bad) - 5} more)"
            raise ValueError(
                f"{name} contains {len(bad)} row(s) with NaN or infinite "
                f"coordinates at index {preview}{more}; clean the input first"
            )

    return array


def data_bounds(points: np.ndarray, margin: float = 0.05):
    """Bounding box and characteristic scale of a point set.

    The elastic-loop solver was written against the unit square and hard-coded
    a ``[0.02, 0.98]`` clip, a radius of ``0.35`` and edge thresholds in
    absolute units. Fed real coordinates -- TSPLIB instances, latitude and
    longitude, anything not pre-normalised -- those constants either collapse
    the loop into a corner or make every edge look short. Deriving them from
    the data makes the solver behave the same at any coordinate scale.

    Args:
        points: ``(n, 2)`` array of data points.
        margin: padding around the bounding box, as a fraction of the scale.

    Returns:
        ``(lo, hi, scale)`` where ``lo``/``hi`` are ``(2,)`` padded corners and
        ``scale`` is the longer side of the unpadded box (1.0 for degenerate
        input, so callers never divide by zero).
    """
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return np.zeros(2), np.ones(2), 1.0

    lo = points.min(axis=0)
    hi = points.max(axis=0)
    scale = float(np.max(hi - lo))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    pad = float(margin) * scale
    return lo - pad, hi + pad, scale


def attract_vertices(points: np.ndarray, vertices: np.ndarray,
                     move_rate: float) -> np.ndarray:
    """Pull every loop vertex toward the centroid of the points nearest to it.

    This is the one step shared by the association solver and by all three
    animated front ends, which each kept their own copy. Catchments are
    accumulated in a single scatter-add rather than by scanning the assignment
    array once per vertex, which is what the animation loop used to do.

    Args:
        points: ``(n, 2)`` data points.
        vertices: ``(m, 2)`` loop vertices.
        move_rate: fraction of the way to move each vertex toward its centroid.

    Returns:
        A new ``(m, 2)`` array; ``vertices`` is never modified in place.
    """
    points = np.asarray(points, dtype=float)
    vertices = np.asarray(vertices, dtype=float)

    if len(points) == 0 or len(vertices) == 0:
        return vertices.copy()

    _, nearest = SpatialIndex(vertices).query_nearest(points)
    nearest = np.asarray(nearest, dtype=int)

    sums = np.zeros_like(vertices)
    counts = np.bincount(nearest, minlength=len(vertices)).astype(float)
    np.add.at(sums, nearest, points)

    assigned = counts > 0
    moved = vertices.copy()
    centroids = sums[assigned] / counts[assigned, None]
    moved[assigned] = (vertices[assigned] * (1.0 - move_rate)
                       + centroids * move_rate)
    return moved

def tour_length(tour: np.ndarray) -> float:
    """Calculate total length of a closed tour given as an array of points"""
    tour = np.asarray(tour, dtype=float)
    if tour.size == 0 or len(tour) < 2:
        return 0.0

    # Add first point at the end to close the loop
    closed_tour = np.vstack([tour, tour[0]])
    distances = np.sqrt(np.sum(np.diff(closed_tour, axis=0)**2, axis=1))
    return float(np.sum(distances))


def tour_length_of_indices(points: np.ndarray, tour: Sequence[int]) -> float:
    """Length of a closed tour expressed as an ordering of point indices"""
    points = np.asarray(points, dtype=float)
    order = np.asarray(tour, dtype=int)
    if order.size < 2:
        return 0.0
    return tour_length(points[order])


def is_valid_tour(tour: Sequence[int], n_points: int) -> bool:
    """True when ``tour`` visits every index in ``range(n_points)`` exactly once"""
    order = np.asarray(tour, dtype=int)
    if order.shape != (n_points,):
        return False
    return bool(np.array_equal(np.sort(order), np.arange(n_points)))


def project_points_onto_loop(points: np.ndarray, loop: np.ndarray,
                             chunk_size: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """Project points onto a closed polyline.

    For every point this finds the closest location on the closed loop and
    returns how far along the loop that location sits (arc length from the
    first vertex) together with the distance from the point to the loop.

    Args:
        points: ``(n, 2)`` array of data points.
        loop: ``(m, 2)`` array of loop vertices; the loop is implicitly closed.
        chunk_size: number of points handled per vectorised block, which caps
            peak memory at roughly ``chunk_size * m`` floats.

    Returns:
        ``(arc_positions, distances)``, both of shape ``(n,)``.
    """
    points = np.asarray(points, dtype=float)
    loop = np.asarray(loop, dtype=float)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be an (n, 2) array")
    if loop.ndim != 2 or loop.shape[1] != 2:
        raise ValueError("loop must be an (m, 2) array")
    if len(loop) < 2:
        raise ValueError("loop must contain at least two vertices")
    if len(points) == 0:
        return np.empty(0), np.empty(0)

    starts = loop
    segments = np.roll(loop, -1, axis=0) - loop
    segment_lengths = np.linalg.norm(segments, axis=1)
    segment_lengths_sq = np.maximum(segment_lengths ** 2, 1e-18)

    # Arc length at the start of each segment.
    arc_offsets = np.concatenate([[0.0], np.cumsum(segment_lengths)[:-1]])

    arc_positions = np.empty(len(points))
    distances = np.empty(len(points))

    for begin in range(0, len(points), max(chunk_size, 1)):
        block = points[begin:begin + max(chunk_size, 1)]
        offsets = block[:, None, :] - starts[None, :, :]          # (b, m, 2)
        t = np.einsum('bmk,mk->bm', offsets, segments) / segment_lengths_sq
        np.clip(t, 0.0, 1.0, out=t)
        closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
        block_distances = np.linalg.norm(block[:, None, :] - closest, axis=2)

        best = np.argmin(block_distances, axis=1)
        rows = np.arange(len(block))
        arc_positions[begin:begin + len(block)] = (
            arc_offsets[best] + t[rows, best] * segment_lengths[best]
        )
        distances[begin:begin + len(block)] = block_distances[rows, best]

    return arc_positions, distances


def order_points_along_loop(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Turn a continuous loop into a genuine TSP tour over the data points.

    The loop produced by the association or clustering solvers is a smooth
    curve threading the data, not a permutation of the points. Ordering the
    points by where they project onto that curve converts the curve into a
    tour that visits every input point exactly once, which is what makes the
    loop-based solvers comparable with the permutation solvers.

    Points that project to the same spot are broken apart by distance to the
    loop so the ordering is deterministic.

    Returns:
        ``(n,)`` array of point indices in visiting order.
    """
    points = np.asarray(points, dtype=float)
    if len(points) <= 2:
        return np.arange(len(points))

    arc_positions, distances = project_points_onto_loop(points, loop)
    return np.lexsort((distances, arc_positions))


def two_opt_refine(points: np.ndarray, tour: Sequence[int],
                   max_passes: int = 50, min_gain: float = 1e-12) -> np.ndarray:
    """Improve a closed tour with 2-opt segment reversals.

    Each candidate move is scored from the four endpoint distances it changes
    rather than by re-measuring the whole tour, which keeps a full sweep at
    ``O(n^2)`` instead of ``O(n^3)``.

    Args:
        points: ``(n, 2)`` array of point coordinates.
        tour: ordering of point indices.
        max_passes: maximum number of improvement sweeps.
        min_gain: smallest improvement worth applying, guarding against
            float noise driving an endless loop.

    Returns:
        ``(n,)`` array with the (possibly improved) ordering.
    """
    points = np.asarray(points, dtype=float)
    order = np.asarray(tour, dtype=int).copy()
    n = len(order)
    if n < 4:
        return order

    improved = True
    passes = 0

    while improved and passes < max_passes:
        improved = False
        passes += 1

        for i in range(n - 2):
            a = order[i]
            b = order[i + 1]
            # Reversing order[i+1:j+1] replaces edges (a, b) and (c, d)
            # with (a, c) and (b, d).
            j = np.arange(i + 2, n if i > 0 else n - 1)
            if j.size == 0:
                continue

            c = order[j]
            d = order[(j + 1) % n]

            removed = (np.linalg.norm(points[a] - points[b])
                       + np.linalg.norm(points[c] - points[d], axis=1))
            added = (np.linalg.norm(points[a] - points[c], axis=1)
                     + np.linalg.norm(points[b] - points[d], axis=1))
            gains = removed - added

            best = int(np.argmax(gains))
            if gains[best] > min_gain:
                cut = int(j[best])
                order[i + 1:cut + 1] = order[i + 1:cut + 1][::-1]
                improved = True

    return order

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
                              uniform_ratio: float = 0.6,
                              seed: SeedLike = None) -> np.ndarray:
    """Generate a mix of clustered and uniform random points.

    Args:
        n_points: total number of points to produce.
        n_clusters: number of Gaussian blobs to draw the clustered share from.
        cluster_std: standard deviation of each blob.
        uniform_ratio: fraction of points drawn uniformly instead of clustered.
        seed: seed or Generator for reproducible output. ``None`` gives fresh
            randomness without touching the global numpy RNG.
    """
    if n_points < 0:
        raise ValueError("n_points must be non-negative")
    if not 0.0 <= uniform_ratio <= 1.0:
        raise ValueError("uniform_ratio must be in [0, 1]")
    if n_points == 0:
        return np.empty((0, 2))

    rng = as_generator(seed)

    n_uniform = int(n_points * uniform_ratio)
    n_clustered = n_points - n_uniform

    points = []

    # Generate clustered points
    if n_clustered > 0:
        n_clusters = max(1, min(n_clusters, n_clustered))
        points_per_cluster = n_clustered // n_clusters
        remainder = n_clustered % n_clusters

        for i in range(n_clusters):
            n_cluster_points = points_per_cluster + (1 if i < remainder else 0)
            if n_cluster_points > 0:
                # Random cluster center
                center = rng.uniform(0.2, 0.8, size=2)
                cluster_points = center + cluster_std * rng.standard_normal((n_cluster_points, 2))
                points.append(cluster_points)

    # Generate uniform points
    if n_uniform > 0:
        points.append(rng.random((n_uniform, 2)))

    # Combine and clip to valid range
    all_points = np.vstack(points) if points else np.empty((0, 2))
    return np.clip(all_points, COORD_MIN, COORD_MAX)


def init_circular_loop(n_vertices: int, center: Tuple[float, float] = (0.5, 0.5),
                       radius: float = 0.35, noise_std: float = 0.02,
                       seed: SeedLike = None, bounds=None) -> np.ndarray:
    """Initialize a noisy circular loop.

    Args:
        n_vertices: number of loop vertices.
        center: loop centre, in the same units as the data.
        radius: nominal loop radius.
        noise_std: magnitude of the radial and positional jitter.
        seed: seed or Generator for reproducible output.
        bounds: optional ``(lo, hi)`` pair of ``(2,)`` corners to clip to.
            Defaults to the unit square; pass the data's own bounding box when
            the coordinates are not normalised, otherwise the whole loop is
            clipped into a corner of the unit square (see :func:`data_bounds`).
    """
    if n_vertices < 3:
        raise ValueError("a loop needs at least three vertices")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be a positive, finite number")

    rng = as_generator(seed)

    angles = np.linspace(0, 2 * np.pi, n_vertices, endpoint=False)
    r = radius + noise_std * rng.standard_normal(n_vertices)

    xs = center[0] + r * np.cos(angles)
    ys = center[1] + r * np.sin(angles)
    vertices = np.column_stack([xs, ys])

    # Add random jitter
    vertices += noise_std * rng.standard_normal((n_vertices, 2))

    if bounds is None:
        return np.clip(vertices, COORD_MIN, COORD_MAX)

    lo, hi = bounds
    return np.clip(vertices, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))

def calculate_adaptive_vertex_count(data_points: np.ndarray, min_vertices: int = 60,
                                   max_vertices: int = 300, density_factor: float = 0.4) -> int:
    """Calculate optimal number of vertices based on data point density.

    The spread term is normalised by the data's own extent. Taken in absolute
    units it grew with the coordinate scale, so the same point cloud expressed
    in metres instead of kilometres asked for a different loop resolution.
    """
    data_points = np.asarray(data_points, dtype=float)
    n_points = len(data_points)

    # Base number of vertices proportional to data points
    adaptive_count = int(n_points * density_factor)

    # Add extra vertices for complex data distributions
    if n_points > 10:
        extent = float(np.max(data_points.max(axis=0) - data_points.min(axis=0)))
        if not np.isfinite(extent) or extent <= 0.0:
            extent = 1.0
        std_x = np.std(data_points[:, 0]) / extent
        std_y = np.std(data_points[:, 1]) / extent
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
                filename: str, tour_length_val: float,
                tour: Optional[Sequence[int]] = None,
                algorithm: Optional[str] = None) -> str:
    """Export tour data to a JSON file.

    Args:
        vertices: loop or tour vertices in visiting order.
        points: the data points the tour was built for.
        filename: destination path; parent directories are created as needed.
        tour_length_val: length of the exported tour.
        tour: optional ordering of point indices, when the solver produced one.
        algorithm: optional name of the solver that produced the result.

    Returns:
        The path written to.
    """
    # A bare filename has no directory component, and os.makedirs("") raises.
    parent = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent, exist_ok=True)

    data = {
        'vertices': np.asarray(vertices, dtype=float).tolist(),
        'points': np.asarray(points, dtype=float).tolist(),
        'tour_length': float(tour_length_val),
        'n_vertices': len(vertices),
        'n_points': len(points),
        'timestamp': time.time(),
    }
    if tour is not None:
        data['tour'] = [int(i) for i in tour]
    if algorithm is not None:
        data['algorithm'] = algorithm

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Data exported to {filename}")
    return filename


def load_data(filename: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load tour data previously written by :func:`export_data`"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{filename} does not contain a TSP solution object")

    missing = [key for key in ('vertices', 'points') if key not in data]
    if missing:
        raise ValueError(f"{filename} is missing required field(s): {', '.join(missing)}")

    vertices = np.asarray(data['vertices'], dtype=float)
    points = np.asarray(data['points'], dtype=float)
    tour_length_val = float(data.get('tour_length', tour_length(vertices)))

    return vertices, points, tour_length_val


def load_points(filename: str) -> np.ndarray:
    """Load a set of 2-D points from a JSON, CSV or whitespace-delimited file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file is empty, is not two columns of coordinates,
            or contains NaN/infinite values.
    """
    extension = os.path.splitext(filename)[1].lower()

    if extension == '.json':
        with open(filename, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{filename} is not valid JSON: {exc}") from exc
        if isinstance(data, dict):
            if 'points' not in data:
                raise ValueError(f"{filename} has no 'points' field")
            data = data['points']
        points = np.asarray(data, dtype=float) if data else np.empty((0, 2))
    else:
        delimiter = ',' if extension == '.csv' else None
        with warnings.catch_warnings():
            # An empty file is reported below with a clearer message than
            # numpy's "input contained no data" warning.
            warnings.filterwarnings('ignore', message='loadtxt: input contained no data')
            points = np.loadtxt(filename, delimiter=delimiter, dtype=float, ndmin=2)

    points = np.atleast_2d(points)
    if points.size == 0:
        raise ValueError(f"{filename} contains no points")

    return validate_points(points, name=filename)
