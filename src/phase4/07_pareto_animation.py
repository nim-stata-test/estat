#!/usr/bin/env python3
"""
Animated visualization of Pareto optimization evolution.

Shows how the Pareto front evolves over generations, with solutions
appearing, becoming Pareto-optimal, and being dominated.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from datetime import datetime

# Import epsilon-dominance functions from optimization script
from importlib import import_module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

# Import EPSILON and epsilon_nondominated_sort from optimization script
_opt_module = import_module('04_pareto_optimization')
EPSILON = _opt_module.EPSILON
epsilon_nondominated_sort = _opt_module.epsilon_nondominated_sort

# Paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "phase4"
ARCHIVE_PATH = OUTPUT_DIR / "pareto_archive.json"


def load_history():
    """Load optimization history from archive."""
    with open(ARCHIVE_PATH) as f:
        data = json.load(f)
    return data.get('optimization_history', {}), data.get('metadata', {})


def prepare_animation_data(history, dominated_threshold=3):
    """Prepare data for animation frames.

    Solutions dominated for `dominated_threshold` consecutive generations
    are hidden from the visualization (but remain in the archive).

    Uses ε-dominance to determine Pareto status (same as final output).
    """
    generations = history.get('generations', [])
    all_solutions = history.get('all_solutions', [])

    # Create lookup for solution data by hash
    solution_lookup = {}
    for sol in all_solutions:
        sol_hash = sol['hash']
        solution_lookup[sol_hash] = {
            'variables': sol['variables'],
            'objectives': sol['objectives'],
            'first_gen': sol['first_gen'],
            'pareto_gens': set(sol['pareto_generations'])
        }

    frames = []
    cumulative_hashes = []  # Use list to preserve order for indexing
    cumulative_set = set()  # For fast lookup
    dominated_streak = {}  # Track consecutive dominated generations per solution

    for gen_data in generations:
        gen = gen_data['generation']

        # Track which solutions are new this generation
        new_hashes = set()
        for sol in gen_data['solutions']:
            sol_hash = sol['hash']
            if sol_hash not in cumulative_set:
                new_hashes.add(sol_hash)
                cumulative_hashes.append(sol_hash)
                cumulative_set.add(sol_hash)
                dominated_streak[sol_hash] = 0

        # Build objective matrix for epsilon-dominance calculation
        # F matrix: [neg_mean_temp, grid_kwh, cost_chf] (all minimized)
        F = np.array([
            [-solution_lookup[h]['objectives']['mean_temp'],
             solution_lookup[h]['objectives']['grid_kwh'],
             solution_lookup[h]['objectives']['cost_chf']]
            for h in cumulative_hashes
        ])

        # Compute ε-Pareto indices using epsilon-dominance
        epsilon_pareto_indices = set(epsilon_nondominated_sort(F, EPSILON))

        # Build frame data from all cumulative solutions
        frame = {
            'generation': gen,
            'solutions': [],
            'n_pareto': len(epsilon_pareto_indices),
            'cumulative_total': len(cumulative_hashes)  # Total unique solutions evaluated so far
        }

        for i, sol_hash in enumerate(cumulative_hashes):
            lookup = solution_lookup.get(sol_hash)
            if lookup is None:
                continue

            is_pareto = i in epsilon_pareto_indices  # Use ε-dominance status
            is_new = sol_hash in new_hashes

            # Update dominated streak
            if is_pareto:
                dominated_streak[sol_hash] = 0
            else:
                dominated_streak[sol_hash] = dominated_streak.get(sol_hash, 0) + 1

            # Skip solutions dominated for too long
            if dominated_streak[sol_hash] >= dominated_threshold:
                continue

            frame['solutions'].append({
                'mean_temp': lookup['objectives']['mean_temp'],
                'grid_kwh': lookup['objectives']['grid_kwh'],
                'cost_chf': lookup['objectives']['cost_chf'],
                'is_pareto': is_pareto,
                'is_new': is_new
            })

        frames.append(frame)

    return frames


def create_animation(frames, metadata, output_path):
    """Create animated visualization of Pareto front evolution."""

    # Set up figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Pareto Optimization Evolution', fontsize=14, fontweight='bold')

    # Collect all data for axis limits
    all_temps = []
    all_grids = []
    all_costs = []
    for frame in frames:
        for sol in frame['solutions']:
            all_temps.append(sol['mean_temp'])
            all_grids.append(sol['grid_kwh'])
            all_costs.append(sol['cost_chf'])

    temp_range = (min(all_temps) - 0.5, max(all_temps) + 0.5)
    grid_range = (min(all_grids) - 20, max(all_grids) + 20)
    cost_range = (min(all_costs) - 5, max(all_costs) + 5)

    # Initialize scatter plots
    scatter_plots = {}

    # Subplot 1: Temperature vs Grid (top-left)
    ax1 = axes[0, 0]
    ax1.set_xlabel('Mean Temperature (°C)')
    ax1.set_ylabel('Grid Import (kWh)')
    ax1.set_xlim(temp_range)
    ax1.set_ylim(grid_range)
    ax1.set_title('Temperature vs Grid Import')
    scatter_plots['temp_grid_dom'] = ax1.scatter([], [], c='lightgray', s=20, alpha=0.5, label='Dominated')
    scatter_plots['temp_grid_pareto'] = ax1.scatter([], [], c='#2ecc71', s=100, alpha=0.9, label='ε-Pareto')
    scatter_plots['temp_grid_new'] = ax1.scatter([], [], c='#e74c3c', s=25, marker='*', alpha=1.0, label='New this gen')
    ax1.legend(loc='upper right', fontsize=8)

    # Subplot 2: Temperature vs Cost (top-right)
    ax2 = axes[0, 1]
    ax2.set_xlabel('Mean Temperature (°C)')
    ax2.set_ylabel('Net Cost (CHF)')
    ax2.set_xlim(temp_range)
    ax2.set_ylim(cost_range)
    ax2.set_title('Temperature vs Cost')
    scatter_plots['temp_cost_dom'] = ax2.scatter([], [], c='lightgray', s=20, alpha=0.5)
    scatter_plots['temp_cost_pareto'] = ax2.scatter([], [], c='#2ecc71', s=100, alpha=0.9)
    scatter_plots['temp_cost_new'] = ax2.scatter([], [], c='#e74c3c', s=25, marker='*', alpha=1.0)

    # Subplot 3: Grid vs Cost (bottom-left)
    ax3 = axes[1, 0]
    ax3.set_xlabel('Grid Import (kWh)')
    ax3.set_ylabel('Net Cost (CHF)')
    ax3.set_xlim(grid_range)
    ax3.set_ylim(cost_range)
    ax3.set_title('Grid Import vs Cost')
    scatter_plots['grid_cost_dom'] = ax3.scatter([], [], c='lightgray', s=20, alpha=0.5)
    scatter_plots['grid_cost_pareto'] = ax3.scatter([], [], c='#2ecc71', s=100, alpha=0.9)
    scatter_plots['grid_cost_new'] = ax3.scatter([], [], c='#e74c3c', s=25, marker='*', alpha=1.0)

    # Subplot 4: Progress metrics (bottom-right)
    ax4 = axes[1, 1]
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Count')
    ax4.set_xlim(0, len(frames) + 1)
    ax4.set_yscale('log')
    max_total = max(f.get('cumulative_total', len(f['solutions'])) for f in frames)
    ax4.set_ylim(1, max_total * 1.5)
    ax4.set_title('Optimization Progress')
    line_total, = ax4.plot([], [], 'b-', linewidth=2, label='Total evaluated')
    line_pareto, = ax4.plot([], [], 'g-', linewidth=2, label='Pareto-optimal')
    ax4.legend(loc='upper left', fontsize=8)
    ax4.grid(True, alpha=0.3, which='both')

    # Generation text
    gen_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, fontweight='bold')

    # Progress tracking
    progress_gens = []
    progress_total = []
    progress_pareto = []

    def init():
        """Initialize animation."""
        for key, scatter in scatter_plots.items():
            scatter.set_offsets(np.empty((0, 2)))
        line_total.set_data([], [])
        line_pareto.set_data([], [])
        gen_text.set_text('')
        return list(scatter_plots.values()) + [line_total, line_pareto, gen_text]

    def update(frame_idx):
        """Update animation frame."""
        frame = frames[frame_idx]
        gen = frame['generation']

        # Separate solutions by status
        dominated = {'temp': [], 'grid': [], 'cost': []}
        pareto = {'temp': [], 'grid': [], 'cost': []}
        new = {'temp': [], 'grid': [], 'cost': []}

        for sol in frame['solutions']:
            target = new if sol['is_new'] else (pareto if sol['is_pareto'] else dominated)
            target['temp'].append(sol['mean_temp'])
            target['grid'].append(sol['grid_kwh'])
            target['cost'].append(sol['cost_chf'])

        # Update scatter plots
        # Temperature vs Grid
        if dominated['temp']:
            scatter_plots['temp_grid_dom'].set_offsets(np.c_[dominated['temp'], dominated['grid']])
        else:
            scatter_plots['temp_grid_dom'].set_offsets(np.empty((0, 2)))
        if pareto['temp']:
            scatter_plots['temp_grid_pareto'].set_offsets(np.c_[pareto['temp'], pareto['grid']])
        else:
            scatter_plots['temp_grid_pareto'].set_offsets(np.empty((0, 2)))
        if new['temp']:
            scatter_plots['temp_grid_new'].set_offsets(np.c_[new['temp'], new['grid']])
        else:
            scatter_plots['temp_grid_new'].set_offsets(np.empty((0, 2)))

        # Temperature vs Cost
        if dominated['temp']:
            scatter_plots['temp_cost_dom'].set_offsets(np.c_[dominated['temp'], dominated['cost']])
        else:
            scatter_plots['temp_cost_dom'].set_offsets(np.empty((0, 2)))
        if pareto['temp']:
            scatter_plots['temp_cost_pareto'].set_offsets(np.c_[pareto['temp'], pareto['cost']])
        else:
            scatter_plots['temp_cost_pareto'].set_offsets(np.empty((0, 2)))
        if new['temp']:
            scatter_plots['temp_cost_new'].set_offsets(np.c_[new['temp'], new['cost']])
        else:
            scatter_plots['temp_cost_new'].set_offsets(np.empty((0, 2)))

        # Grid vs Cost
        if dominated['grid']:
            scatter_plots['grid_cost_dom'].set_offsets(np.c_[dominated['grid'], dominated['cost']])
        else:
            scatter_plots['grid_cost_dom'].set_offsets(np.empty((0, 2)))
        if pareto['grid']:
            scatter_plots['grid_cost_pareto'].set_offsets(np.c_[pareto['grid'], pareto['cost']])
        else:
            scatter_plots['grid_cost_pareto'].set_offsets(np.empty((0, 2)))
        if new['grid']:
            scatter_plots['grid_cost_new'].set_offsets(np.c_[new['grid'], new['cost']])
        else:
            scatter_plots['grid_cost_new'].set_offsets(np.empty((0, 2)))

        # Update progress plot
        progress_gens.append(gen)
        cumulative_total = frame.get('cumulative_total', len(frame['solutions']))
        progress_total.append(cumulative_total)
        progress_pareto.append(frame['n_pareto'])

        line_total.set_data(progress_gens, progress_total)
        line_pareto.set_data(progress_gens, progress_pareto)

        # Update generation text
        n_new = len(new['temp'])
        gen_text.set_text(f'Generation {gen}/{len(frames)} | '
                         f'Total evaluated: {cumulative_total} | '
                         f'Pareto: {frame["n_pareto"]} | '
                         f'New: {n_new}')

        return list(scatter_plots.values()) + [line_total, line_pareto, gen_text]

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        init_func=init, blit=True,
        interval=500, repeat=True
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    # Save animation
    print(f"Saving animation to {output_path}...")
    writer = PillowWriter(fps=2)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"Animation saved: {output_path}")

    # Also save final frame as static image
    static_path = output_path.with_name('fig4.06_pareto_evolution.png')
    update(len(frames) - 1)  # Update to final frame
    fig.savefig(static_path, dpi=150, bbox_inches='tight')
    print(f"Static image saved: {static_path}")

    plt.close()

    return anim


def create_3d_animation(frames, metadata, output_path):
    """Create 3D animated visualization."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Collect all data for axis limits
    all_temps = []
    all_grids = []
    all_costs = []
    for frame in frames:
        for sol in frame['solutions']:
            all_temps.append(sol['mean_temp'])
            all_grids.append(sol['grid_kwh'])
            all_costs.append(sol['cost_chf'])

    ax.set_xlabel('Mean Temp (°C)', fontsize=10)
    ax.set_ylabel('Grid (kWh)', fontsize=10)
    ax.set_zlabel('Cost (CHF)', fontsize=10)
    ax.set_xlim(min(all_temps) - 0.5, max(all_temps) + 0.5)
    ax.set_ylim(min(all_grids) - 20, max(all_grids) + 20)
    ax.set_zlim(min(all_costs) - 5, max(all_costs) + 5)

    # Initialize scatter plots
    scatter_dom = ax.scatter([], [], [], c='lightgray', s=15, alpha=0.4, label='Dominated')
    scatter_pareto = ax.scatter([], [], [], c='#2ecc71', s=80, alpha=0.9, label='ε-Pareto')
    scatter_new = ax.scatter([], [], [], c='#e74c3c', s=20, marker='*', alpha=1.0, label='New')

    ax.legend(loc='upper left', fontsize=9)
    title = ax.set_title('', fontsize=12, fontweight='bold')

    def update(frame_idx):
        """Update 3D animation frame."""
        frame = frames[frame_idx]
        gen = frame['generation']

        # Separate solutions
        dominated = {'temp': [], 'grid': [], 'cost': []}
        pareto = {'temp': [], 'grid': [], 'cost': []}
        new = {'temp': [], 'grid': [], 'cost': []}

        for sol in frame['solutions']:
            target = new if sol['is_new'] else (pareto if sol['is_pareto'] else dominated)
            target['temp'].append(sol['mean_temp'])
            target['grid'].append(sol['grid_kwh'])
            target['cost'].append(sol['cost_chf'])

        # Update scatter data
        scatter_dom._offsets3d = (dominated['temp'], dominated['grid'], dominated['cost'])
        scatter_pareto._offsets3d = (pareto['temp'], pareto['grid'], pareto['cost'])
        scatter_new._offsets3d = (new['temp'], new['grid'], new['cost'])

        # Rotate view slightly each frame
        ax.view_init(elev=20, azim=30 + frame_idx * 3)

        cumulative_total = frame.get('cumulative_total', len(frame['solutions']))
        title.set_text(f'Generation {gen}/{len(frames)} | Pareto: {frame["n_pareto"]} | Total evaluated: {cumulative_total}')

        return scatter_dom, scatter_pareto, scatter_new, title

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        interval=500, repeat=True
    )

    plt.tight_layout()

    # Save animation
    print(f"Saving 3D animation to {output_path}...")
    writer = PillowWriter(fps=2)
    anim.save(output_path, writer=writer, dpi=100)
    print(f"3D Animation saved: {output_path}")

    plt.close()


def convert_to_mp4(gif_path, mp4_path, fps=2):
    """Convert GIF to MP4 for PowerPoint compatibility.

    Uses ffmpeg to create an H.264 encoded MP4 that works in PowerPoint.
    """
    import subprocess

    print(f"Converting {gif_path.name} to MP4...")

    # ffmpeg command for PowerPoint-compatible MP4
    # -y: overwrite output
    # -i: input file
    # -movflags faststart: optimize for streaming/PowerPoint
    # -pix_fmt yuv420p: compatible pixel format
    # -vf scale: ensure even dimensions (required for H.264)
    # -c:v libx264: H.264 codec (widely compatible)
    # -crf 23: quality (lower = better, 18-28 is reasonable)
    cmd = [
        'ffmpeg', '-y',
        '-i', str(gif_path),
        '-movflags', 'faststart',
        '-pix_fmt', 'yuv420p',
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264',
        '-crf', '23',
        '-r', str(fps),
        str(mp4_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Created: {mp4_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ ffmpeg failed: {e.stderr}")
        return False
    except FileNotFoundError:
        print("  ✗ ffmpeg not found. Install with: brew install ffmpeg")
        return False


def main():
    """Generate animated visualization of Pareto optimization."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Pareto optimization animations')
    parser.add_argument('--mp4-only', action='store_true',
                       help='Only convert existing GIFs to MP4 (skip GIF generation)')
    parser.add_argument('--fps', type=int, default=2,
                       help='Frames per second for MP4 (default: 2)')
    args = parser.parse_args()

    output_2d = OUTPUT_DIR / "pareto_evolution.gif"
    output_3d = OUTPUT_DIR / "pareto_evolution_3d.gif"
    mp4_2d = OUTPUT_DIR / "pareto_evolution.mp4"
    mp4_3d = OUTPUT_DIR / "pareto_evolution_3d.mp4"

    if not args.mp4_only:
        print("Loading optimization history...")
        history, metadata = load_history()

        if not history:
            print("ERROR: No optimization history found in archive.")
            print("Run optimization with history tracking first:")
            print("  python src/phase4/04_pareto_optimization.py --fresh -g 20")
            return

        print(f"Found {history['summary']['unique_solutions']} solutions across "
              f"{history['summary']['total_generations']} generations")

        print("\nPreparing animation data...")
        frames = prepare_animation_data(history)
        print(f"Prepared {len(frames)} animation frames")

        # Create 2D animation (2x2 subplots)
        create_animation(frames, metadata, output_2d)

        # Create 3D animation
        create_3d_animation(frames, metadata, output_3d)

    # Convert GIFs to MP4 for PowerPoint
    print("\nConverting to PowerPoint-compatible MP4...")
    if output_2d.exists():
        convert_to_mp4(output_2d, mp4_2d, fps=args.fps)
    if output_3d.exists():
        convert_to_mp4(output_3d, mp4_3d, fps=args.fps)

    print("\n✓ Animation complete!")
    print(f"  2D animation: {output_2d}")
    print(f"  3D animation: {output_3d}")
    print(f"  2D video (PowerPoint): {mp4_2d}")
    print(f"  3D video (PowerPoint): {mp4_3d}")


if __name__ == "__main__":
    main()
