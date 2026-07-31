"""
Demonstration script for the Semi-Supervised TSP Visualizer.

Runs a full comparison across every solver, then fits the association loop to
a point set and writes an annotated plot showing both the fitted loop and the
tour it induces. Writes files only; never opens a window.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from backend import select_backend

# The demo only ever saves files, so bind the non-interactive writer up front.
select_backend(interactive=False)

import matplotlib.pyplot as plt

from config import Config
from main import EnhancedTSPVisualizer

OUTPUT_PLOT = 'demo_tsp_solution.png'
OUTPUT_JSON = 'demo_solution.json'


def demo_algorithm_comparison(n_points=100, seed=42):
    """Compare every solver on one point set and report the ranking"""
    print("=" * 80)
    print("SEMI-SUPERVISED TSP VISUALIZER DEMO")
    print("=" * 80)

    visualizer = EnhancedTSPVisualizer(mode='simple', seed=seed)

    print("\n1. Generating test data...")
    visualizer.generate_data(n_points)
    print(f"   Generated {len(visualizer.points)} points with clustering structure")

    print("\n2. Comparing TSP algorithms...")
    print("   Every solver is scored on the length of the tour through all")
    print("   input points, so the loop-based solvers are measured on the")
    print("   tour their loop induces rather than on the loop itself.")

    results = visualizer.compare_algorithms(Config.AVAILABLE_ALGORITHMS, runs=1)

    if results:
        print("\n3. Algorithm performance summary:")
        print("-" * 60)
        ranked = sorted(results.items(), key=lambda item: item[1]['avg_distance'])
        for position, (name, result) in enumerate(ranked, 1):
            print(f"   {position}. {name:20s} - Distance: {result['avg_distance']:7.4f}, "
                  f"Time: {result['avg_time']:6.3f}s")
    else:
        print("\n3. No algorithm completed successfully.")

    return visualizer


def demo_association_plot(visualizer):
    """Fit the association loop and save a plot of the loop and its tour"""
    print("\n4. Demonstrating the association algorithm...")

    # Use the configured defaults so the plot shows what the solver actually
    # does out of the box.
    visualizer.set_algorithm('association')
    distance = visualizer.solve_static()
    solution = visualizer.solution

    print(f"   Tour length over all points: {distance:.6f}")
    print(f"   Fitted loop length:          {solution.loop_length:.6f}")
    print(f"   Loop vertices:               {len(solution.loop)}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for ax in axes:
        ax.scatter(visualizer.points[:, 0], visualizer.points[:, 1],
                   c='red', s=30, alpha=0.7, label='Data points', zorder=3)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    loop = solution.loop
    loop_x = np.append(loop[:, 0], loop[0, 0])
    loop_y = np.append(loop[:, 1], loop[0, 1])
    axes[0].plot(loop_x, loop_y, 'g-', linewidth=2,
                 label=f'Fitted loop (length {solution.loop_length:.4f})')
    axes[0].scatter(loop[:, 0], loop[:, 1], c='green', s=12, alpha=0.8,
                    label=f'Loop vertices ({len(loop)})')
    axes[0].set_title('Step 1: elastic loop fitted to the data')
    axes[0].legend(loc='upper right', fontsize=8)

    ordered = solution.tour_points
    tour_x = np.append(ordered[:, 0], ordered[0, 0])
    tour_y = np.append(ordered[:, 1], ordered[0, 1])
    axes[1].plot(tour_x, tour_y, 'b-', linewidth=1.5,
                 label=f'Induced tour (length {solution.length:.4f})')
    axes[1].set_title('Step 2: points ordered along the loop')
    axes[1].legend(loc='upper right', fontsize=8)

    fig.suptitle('Semi-Supervised TSP - Association Algorithm', fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_PLOT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   Solution plot saved as: {OUTPUT_PLOT}")

    print("\n5. Exporting results...")
    visualizer.export_results(OUTPUT_JSON)


def demo_usage_examples():
    """Show usage examples"""
    print("\nUSAGE EXAMPLES:")
    print("-" * 30)

    examples = [
        ("Simple visualization", "python main.py --algorithm association --points 100"),
        ("Algorithm comparison", "python main.py --compare nearest_neighbor two_opt genetic"),
        ("Polish with 2-opt", "python main.py --algorithm association --refine"),
        ("Animation with video", "python main.py --animate --save-video evolution.mp4"),
        ("CLI solve", "python cli.py solve genetic --points 200 --output solution.json"),
        ("CLI benchmark", "python cli.py benchmark association clustering --sizes 50 100 200"),
        ("Interactive GUI", "python main.py --mode gui"),
        ("Run tests", "python test_suite.py"),
    ]

    for purpose, command in examples:
        print(f"  {purpose:22s}: {command}")


def main():
    demo_usage_examples()
    visualizer = demo_algorithm_comparison()
    demo_association_plot(visualizer)

    print("\n" + "=" * 80)
    print("DEMO COMPLETED")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
