# Semi-Supervised TSP Visualizer

An advanced, interactive visualization tool for solving the Traveling Salesman Problem (TSP) using multiple algorithms including semi-supervised approaches.

## 🚀 Features

### Multiple TSP Algorithms
- **Association-based**: Iterative vertex-to-point assignment with smoothing
- **K-means Clustering**: Cluster points then solve TSP on cluster centers  
- **Nearest Neighbor**: Classic greedy TSP heuristic
- **2-Opt**: Local search optimization
- **Genetic Algorithm**: Evolutionary approach with crossover and mutation
- **Simulated Annealing**: Probabilistic optimization with cooling schedule

### Interactive Interfaces
- **GUI Mode**: Real-time interactive controls with sliders and buttons
- **CLI Mode**: Command-line interface for batch processing and automation
- **Simple Mode**: Quick visualization and comparison

### Advanced Features
- **Real-time Visualization**: Watch algorithms evolve in real-time
- **Performance Metrics**: Track convergence, tour length, and execution time
- **Algorithm Comparison**: Side-by-side performance analysis
- **Data Export/Import**: Save and load solutions in JSON format
- **Video Export**: Save animations as MP4 videos
- **Adaptive Parameters**: Algorithm parameters that adjust during execution
- **Convergence Detection**: Automatic stopping when solution stabilizes

### Performance Optimizations
- **Spatial Indexing**: Efficient nearest neighbor queries using k-d trees
- **Batch Processing**: Optimized for large datasets
- **Memory Management**: Efficient array operations with NumPy
- **Parallel-Ready**: Modular design for future parallelization

## 📦 Installation

### Requirements
```bash
pip install numpy matplotlib scikit-learn scipy
```

### Optional Dependencies
```bash
# For video export
pip install ffmpeg-python

# For enhanced GUI
pip install tkinter
```

### Quick Start
```bash
# Clone or download the project
git clone <repository-url>
cd Semi-Supervised-TSP

# Run with default settings
python main.py

# Launch interactive GUI
python main.py --mode gui

# Compare algorithms
python main.py --compare association nearest_neighbor genetic --points 100
```

## 🎮 Usage Examples

### Interactive GUI
```bash
python main.py --mode gui
```
- Real-time parameter adjustment with sliders
- Algorithm switching with radio buttons
- Start/pause/reset controls
- Export solutions and videos

### Command Line Interface
```bash
# Solve with specific algorithm
python cli.py solve association --points 200 --output solution.json

# Compare multiple algorithms
python cli.py compare nearest_neighbor two_opt genetic --points 100 --runs 5

# Create animation video
python cli.py animate association --points 150 --output animation.mp4

# Benchmark performance
python cli.py benchmark nearest_neighbor two_opt --sizes 50 100 200 300
```

### Simple Visualization
```bash
# Quick solve and display
python main.py --algorithm genetic --points 150

# Animated solution
python main.py --algorithm association --points 200 --animate

# Save animation video
python main.py --algorithm association --animate --save-video tsp_evolution.mp4

# Compare algorithms without GUI
python main.py --compare association clustering nearest_neighbor --points 100
```

## 🔧 Configuration

### Config File (`config.py`)
Customize algorithm parameters, visualization settings, and performance options:

```python
class Config:
    # Data generation
    N_POINTS = 300
    N_CLUSTERS_DATA = 5
    
    # Algorithm parameters  
    N_VERTICES = 80
    INITIAL_MOVE_RATE = 0.25
    INITIAL_SMOOTH_RATE = 0.4
    
    # Animation settings
    STEPS = 600
    INTERVAL_MS = 50
    
    # Convergence detection
    CONVERGENCE_THRESHOLD = 1e-6
    CONVERGENCE_WINDOW = 10
```

### Algorithm-Specific Parameters

#### Association Algorithm
- `n_vertices`: Number of loop vertices (default: 80)
- `max_iterations`: Maximum iterations (default: 600)
- Move rate and smoothing adapt automatically over time

#### Genetic Algorithm
- `population_size`: Population size (default: 100)
- `generations`: Number of generations (default: 50)
- `mutation_rate`: Mutation probability (default: 0.02)
- `elite_size`: Number of elite individuals (default: 20)

#### Simulated Annealing
- `initial_temp`: Starting temperature (default: 1000)
- `cooling_rate`: Temperature decay rate (default: 0.995)
- `min_temp`: Minimum temperature (default: 1e-8)

## 📊 Algorithm Details

### Association-Based Algorithm
1. Initialize circular loop of vertices
2. Assign each data point to nearest vertex
3. Move vertices toward centroid of assigned points
4. Apply Laplacian smoothing to maintain loop structure
5. Repeat until convergence

### K-means Clustering Approach
1. Perform k-means clustering on data points
2. Extract cluster centers
3. Solve TSP on cluster centers using nearest neighbor
4. Optional: Interpolate more vertices along the path

### Optimization Algorithms
- **2-Opt**: Iteratively improve tour by swapping edge pairs
- **Genetic**: Evolve population of tours using crossover and mutation
- **Simulated Annealing**: Accept worse solutions probabilistically with cooling

## 📈 Performance Analysis

### Benchmarking
```bash
# Test scaling with problem size
python cli.py benchmark nearest_neighbor two_opt genetic --sizes 50 100 200 500

# Compare solution quality
python cli.py compare association clustering nearest_neighbor two_opt --runs 10
```

### Metrics Tracked
- **Tour Length**: Total distance of the solution
- **Convergence Rate**: How quickly algorithm stabilizes
- **Execution Time**: Algorithm runtime
- **Solution Quality**: Comparison across algorithms

### Typical Performance
| Algorithm | Speed | Quality | Use Case |
|-----------|-------|---------|----------|
| Nearest Neighbor | Very Fast | Poor | Quick approximation |
| Association | Fast | Good | Smooth visualization |
| K-means Clustering | Fast | Good | Data with natural clusters |
| 2-Opt | Medium | Better | Improved solutions |
| Genetic | Slow | Good | Global optimization |
| Simulated Annealing | Slow | Good | Avoiding local minima |

## 🎨 Visualization Features

### Real-time Animation
- Watch algorithm evolution step-by-step
- Adjustable animation speed
- Pause/resume functionality
- Progress indicators

### Multiple Plot Types
- **Main Plot**: Points, tour, and vertices
- **Metrics Plot**: Tour length over time  
- **Convergence Plot**: Algorithm convergence rate

### Export Options
- **Static Images**: PNG, PDF, SVG formats
- **Animated Videos**: MP4 with customizable quality
- **Data Export**: JSON format with complete solution data

## 🧪 Testing

### Run Test Suite
```bash
python test_suite.py
```

### Test Categories
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end algorithm testing
- **Performance Tests**: Benchmarking and scaling tests
- **Data Validation**: Import/export functionality

### Coverage
- Configuration validation
- Utility functions
- All TSP algorithms
- Data import/export
- Spatial indexing
- Error handling

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python test_suite.py

# Check code style
flake8 *.py

# Type checking
mypy *.py
```

### Adding New Algorithms
1. Inherit from `TSPAlgorithm` base class
2. Implement `solve()` method
3. Add to algorithm factory in `algorithms.py`
4. Update configuration and documentation
5. Add tests in `test_suite.py`

### Code Structure
```
├── config.py          # Configuration settings
├── utils.py           # Utility functions  
├── algorithms.py      # TSP algorithm implementations
├── gui.py            # Interactive GUI interface
├── cli.py            # Command-line interface
├── main.py           # Main application entry point
├── test_suite.py     # Comprehensive test suite
└── README.md         # This file
```

## 📝 License

This project is open source. Feel free to use, modify, and distribute.

## 🔗 References

- Traveling Salesman Problem: [Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
- Semi-supervised Learning: [Scholarpedia](http://www.scholarpedia.org/article/Semi-supervised_learning)
- Genetic Algorithms for TSP: [Tutorial](https://towardsdatascience.com/evolution-of-a-salesman-a-complete-genetic-algorithm-tutorial-for-python-6fe5d2b3ca35)
- Simulated Annealing: [Optimization Methods](https://optimization.mccormick.northwestern.edu/index.php/Simulated_annealing)

## 📞 Support

For questions, issues, or contributions:
1. Check existing issues in the repository
2. Create a new issue with detailed description  
3. For urgent matters, contact the development team

---

**Happy Optimizing! 🎯**
