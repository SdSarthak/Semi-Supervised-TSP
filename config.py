"""
Configuration settings for Semi-Supervised TSP Visualizer
"""

import os


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment, falling back on bad input"""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class Config:
    """Configuration class for TSP visualizer settings"""

    # Random seed for reproducibility. Override with TSP_SEED.
    SEED = _env_int('TSP_SEED', 42)

    # Data generation
    N_POINTS = 40
    N_CLUSTERS_DATA = 5
    CLUSTER_STD = 0.05
    UNIFORM_RATIO = 0.6

    # Algorithm parameters
    N_VERTICES = 120  # Loop resolution for the association solver
    K_CLUSTERS = 40
    INITIAL_MOVE_RATE = 0.25
    INITIAL_SMOOTH_RATE = 0.4
    # Smoothing has to decay away for the loop to sharpen onto the data. A
    # floor of 0.1 kept it taut on the convex hull, so interior points were
    # never threaded and the induced tour lost to plain greedy search.
    MIN_MOVE_RATE = 0.05
    MIN_SMOOTH_RATE = 0.0

    # Advanced loop parameters for better curves and bends
    ADAPTIVE_VERTEX_DENSITY = True  # Size the loop from the data instead of N_VERTICES
    MIN_VERTICES = 60  # Minimum number of vertices
    MAX_VERTICES = 300  # Maximum number of vertices
    VERTEX_DENSITY_FACTOR = 0.4  # Vertices per data point ratio
    SUBDIVISION_THRESHOLD = 0.05  # Distance threshold for vertex subdivision
    SMOOTHING_ITERATIONS = 2  # Number of smoothing iterations per step

    # Animation settings
    STEPS = 600
    INTERVAL_MS = 50
    FIGSIZE = (12, 8)

    # Convergence detection
    CONVERGENCE_THRESHOLD = 1e-6
    CONVERGENCE_WINDOW = 10

    # Performance settings
    USE_SPATIAL_INDEX = True
    BATCH_SIZE = 50

    # Visualization
    POINT_SIZE = 20
    LINE_WIDTH = 2
    ALPHA = 0.7

    # Export settings. Override the directory with TSP_EXPORT_DIR.
    DEFAULT_EXPORT_DIR = os.environ.get('TSP_EXPORT_DIR', './exports')
    VIDEO_FPS = 30
    VIDEO_DPI = 150

    # TSP algorithms
    AVAILABLE_ALGORITHMS = [
        'nearest_neighbor',
        'two_opt',
        'genetic',
        'simulated_annealing',
        'association',
        'clustering'
    ]

    # Genetic algorithm parameters
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 50
    GA_MUTATION_RATE = 0.02
    GA_ELITE_SIZE = 20
    GA_TOURNAMENT_SIZE = 3

    # Simulated annealing parameters. None lets the solver calibrate the
    # temperature to the scale of the data, which is what you want unless you
    # are deliberately reproducing a specific schedule.
    SA_INITIAL_TEMP = None
    SA_COOLING_RATE = 0.995
    SA_MIN_TEMP = None
    SA_ITERATIONS_PER_TEMP = None

    @classmethod
    def validate(cls):
        """Validate configuration parameters.

        Raises:
            ValueError: if any setting is out of range.
        """
        # Explicit raises rather than asserts: assertions are stripped under
        # ``python -O``, which would silently disable validation entirely.
        checks = [
            (cls.N_POINTS > 0, "N_POINTS must be positive"),
            (0 < cls.INITIAL_MOVE_RATE <= 1, "INITIAL_MOVE_RATE must be in (0, 1]"),
            (0 < cls.INITIAL_SMOOTH_RATE <= 1, "INITIAL_SMOOTH_RATE must be in (0, 1]"),
            (cls.STEPS > 0, "STEPS must be positive"),
            (cls.N_VERTICES >= 3, "N_VERTICES must be at least 3"),
            (cls.MIN_VERTICES >= 3, "MIN_VERTICES must be at least 3"),
            (cls.MAX_VERTICES >= cls.MIN_VERTICES,
             "MAX_VERTICES must be at least MIN_VERTICES"),
            (0 <= cls.UNIFORM_RATIO <= 1, "UNIFORM_RATIO must be in [0, 1]"),
            (0 < cls.GA_MUTATION_RATE <= 1, "GA_MUTATION_RATE must be in (0, 1]"),
            (0 < cls.GA_ELITE_SIZE < cls.GA_POPULATION_SIZE,
             "GA_ELITE_SIZE must be smaller than GA_POPULATION_SIZE"),
            (0 < cls.SA_COOLING_RATE < 1, "SA_COOLING_RATE must be in (0, 1)"),
            (cls.CONVERGENCE_WINDOW > 0, "CONVERGENCE_WINDOW must be positive"),
        ]

        for ok, message in checks:
            if not ok:
                raise ValueError(message)

        return True

    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith('_') and not callable(v)
                and not isinstance(v, (classmethod, staticmethod))}
