"""
Configuration settings for Semi-Supervised TSP Visualizer
"""

import numpy as np

class Config:
    """Configuration class for TSP visualizer settings"""
    
    # Random seed for reproducibility
    SEED = 42
    
    # Data generation
    N_POINTS = 40
    N_CLUSTERS_DATA = 5
    CLUSTER_STD = 0.05
    UNIFORM_RATIO = 0.6
    
    # Algorithm parameters
    N_VERTICES = 120  # Increased for finer loop resolution
    K_CLUSTERS = 40
    INITIAL_MOVE_RATE = 0.25
    INITIAL_SMOOTH_RATE = 0.4
    MIN_MOVE_RATE = 0.01
    MIN_SMOOTH_RATE = 0.1
    
    # Advanced loop parameters for better curves and bends
    ADAPTIVE_VERTEX_DENSITY = True  # Enable adaptive vertex density
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
    
    # Export settings
    DEFAULT_EXPORT_DIR = "./exports"
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
    
    # Simulated annealing parameters
    SA_INITIAL_TEMP = 1000
    SA_COOLING_RATE = 0.995
    SA_MIN_TEMP = 1e-8
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        assert cls.N_POINTS > 0, "Number of points must be positive"
        assert 0 < cls.INITIAL_MOVE_RATE <= 1, "Move rate must be in (0,1]"
        assert 0 < cls.INITIAL_SMOOTH_RATE <= 1, "Smooth rate must be in (0,1]"
        assert cls.STEPS > 0, "Number of steps must be positive"
        
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {k: v for k, v in cls.__dict__.items() 
                if not k.startswith('_') and not callable(v)}
