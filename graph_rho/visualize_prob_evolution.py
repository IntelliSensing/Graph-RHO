"""
Probability Distribution Evolution Visualization for RHO.

This script visualizes how the model's predicted probability distribution evolves
across RHO iterations, demonstrating the advantage of adaptive thresholding
over static thresholding.

Visualization: Ridgeline Plot (Joy Plot)
- X-axis: Predicted Probability (0 to 1)
- Y-axis: RHO Iteration Steps (stacked distributions)
- Dashed line: Static threshold at 0.5
- Solid line: Adaptive threshold (top k% quantile)
"""

import argparse
import pickle
import copy
import collections
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from graph_rho.config import DATASET_CONFIG, DEVICE_CONFIG, GNN_CONFIG, TEST_CONFIG
from graph_rho.hetero_gnn_model import HeteroGNNModel
from graph_rho.gnn_data_loader import RunningNormalization
from graph_rho.utils.device_utils import get_device
from graph_rho.utils.lrho_paths import ensure_lrho_makespan_on_path
from graph_rho.utils.path_utils import get_analysis_dir, get_data_dir, get_model_dir

ensure_lrho_makespan_on_path()

from flexible_jss_main_common import flexible_jss, sort_by_task_order
from flexible_jss_main import exec_step, select_jss_data, select_window
from flexible_jss_data import get_rollout_data
from flexible_jss_data_common import FlexibleJSSDataset, get_dataloader

try:
    from flexible_jss_learn import FlexibleJSSNet
    import flexible_jss_learn as _flexible_jss_learn

    LRHO_MODEL_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not load L-RHO model: {e}")
    LRHO_MODEL_AVAILABLE = False


class ProbabilityCollector:
    """Collects probability distributions at each RHO iteration."""

    def __init__(self):
        self.iterations = []  # List of dicts containing step info

    def add_iteration(self, step_idx: int, probabilities: np.ndarray,
                      num_tasks: int, num_fix_static: int, num_fix_adaptive: int,
                      adaptive_threshold: float):
        """
        Record probability distribution for one RHO iteration.

        Args:
            step_idx: RHO step number
            probabilities: Array of predicted probabilities for overlapping tasks
            num_tasks: Total number of overlapping tasks
            num_fix_static: Number of tasks fixed with static threshold (0.5)
            num_fix_adaptive: Number of tasks fixed with adaptive threshold
            adaptive_threshold: The adaptive threshold value used
        """
        self.iterations.append({
            'step': step_idx,
            'probabilities': probabilities.copy(),
            'num_tasks': num_tasks,
            'num_fix_static': num_fix_static,
            'num_fix_adaptive': num_fix_adaptive,
            'adaptive_threshold': adaptive_threshold,
            'mean_prob': float(np.mean(probabilities)) if len(probabilities) > 0 else 0.0,
            'std_prob': float(np.std(probabilities)) if len(probabilities) > 0 else 0.0,
        })

    def get_all_probabilities(self) -> List[np.ndarray]:
        """Return list of probability arrays for each iteration."""
        return [it['probabilities'] for it in self.iterations]

    def get_adaptive_thresholds(self) -> List[float]:
        """Return list of adaptive thresholds."""
        return [it['adaptive_threshold'] for it in self.iterations]

    def summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("Probability Distribution Evolution Summary")
        print("=" * 60)
        for it in self.iterations:
            print(f"Step {it['step']:2d}: n={it['num_tasks']:3d}, "
                  f"mean={it['mean_prob']:.3f}, std={it['std_prob']:.3f}, "
                  f"fix_static={it['num_fix_static']:3d}, "
                  f"fix_adaptive={it['num_fix_adaptive']:3d}, "
                  f"adaptive_th={it['adaptive_threshold']:.3f}")


def compute_adaptive_threshold(probs: np.ndarray, target_ratio: float = 0.5) -> float:
    """
    Compute adaptive threshold based on target fix ratio.

    Args:
        probs: Array of probabilities
        target_ratio: Target ratio of tasks to fix (e.g., 0.5 means fix top 50%)

    Returns:
        Threshold value
    """
    if len(probs) == 0:
        return 0.5

    # Sort probabilities in descending order
    sorted_probs = np.sort(probs)[::-1]

    # Find threshold that fixes approximately target_ratio of tasks
    target_count = int(len(probs) * target_ratio)
    target_count = max(1, min(target_count, len(probs)))  # Clamp to valid range

    # Threshold is the probability at the target_count position
    threshold = sorted_probs[target_count - 1]

    return float(threshold)


def get_model_probabilities(model, data, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get probability predictions from model.

    Args:
        model: Trained model
        data: Input data for model
        device: Device to run on

    Returns:
        Tuple of predicted probabilities and the matching task-label indices
    """
    dataset = FlexibleJSSDataset([data])
    loader = get_dataloader(dataset, batch_size=1, shuffle=False,
                            follow_batch=['x_tasks', 'x_machines'],
                            num_workers=0, pin_memory=False)
    loader_data = next(iter(loader))

    with torch.no_grad():
        output = model(loader_data.to(device))

    # Handle multi-task learning output
    if isinstance(output, tuple):
        output = output[0]  # Use only the fix prediction

    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(output).cpu().reshape(-1).numpy()
    task_label_idx = loader_data.task_label_idx.cpu().numpy()

    return probs, task_label_idx


def rolling_horizon_with_prob_collection(
    n_machines: int,
    n_jobs: int,
    jobs_data: dict,
    window: int,
    step: int,
    time_limit: int,
    stop_search_time: int,
    model,
    device,
    target_ratio: float = 0.5,
    static_threshold: float = 0.5
) -> Tuple[ProbabilityCollector, float]:
    """
    Run rolling horizon optimization while collecting probability distributions.

    This is a modified version of rolling_horizon that collects model predictions
    at each iteration for visualization purposes.

    Args:
        n_machines: Number of machines
        n_jobs: Number of jobs
        jobs_data: Job data dictionary
        window: RHO window size
        step: RHO step size
        time_limit: Solver time limit
        stop_search_time: Solver stop search time
        model: Trained model
        device: Device for model inference
        target_ratio: Target ratio for adaptive threshold
        static_threshold: Static threshold value (default 0.5)

    Returns:
        collector: ProbabilityCollector with all iteration data
        makespan: Final makespan
    """
    collector = ProbabilityCollector()

    sorted_task_indices = sort_by_task_order(jobs_data)

    # Tracking solutions
    jobs_solution = {}
    machines_assignment = collections.defaultdict(list)
    jobs_solution_sel = {}
    machines_assignment_sel = {}

    # Tracking stats
    num_tasks = len(sorted_task_indices)

    # Information required by the solving process
    machines_start_time = [0 for _ in range(n_machines)]
    jobs_start_time = [0 for _ in range(n_jobs)]
    prev_sel_task_indices = []
    start_loc = 0

    iteration = 0

    print(f"\n{'='*60}")
    print(f"Running RHO with probability collection")
    print(f"Window: {window}, Step: {step}")
    print(f"Static threshold: {static_threshold}, Target ratio: {target_ratio}")
    print(f"{'='*60}\n")

    while len(jobs_solution) < num_tasks:
        sel_task_indices, sel_i_task_loc = select_window(
            start_loc, num_tasks, sorted_task_indices, jobs_solution, window)
        jobs_data_sel = select_jss_data(sel_task_indices, jobs_data)

        if len(jobs_solution) > 0:
            # Get overlapping tasks
            overlapping_task_indices = [task for task in prev_sel_task_indices
                                        if task in sel_task_indices and task in jobs_solution_sel]

            if len(overlapping_task_indices) > 0:
                # Get overlapping solutions
                overlapping_jobs_solution_all = {
                    (job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                    for job_id, task_id in overlapping_task_indices
                    if (job_id, task_id) in jobs_solution_sel
                }
                overlapping_machines_assignment_all = {
                    machine: [task for task in machines_assignment_sel[machine]
                              if (task.job, task.index) in overlapping_task_indices]
                    for machine in machines_assignment_sel
                }

                # Get model prediction data
                data = get_rollout_data(
                    jobs_data_sel, n_machines, machines_start_time,
                    jobs_start_time, overlapping_task_indices,
                    overlapping_jobs_solution_all, overlapping_machines_assignment_all
                )

                # Get probability predictions
                probs, task_label_idx = get_model_probabilities(model, data, device)

                # Compute adaptive threshold
                adaptive_th = compute_adaptive_threshold(probs, target_ratio)

                # Count fixes with different thresholds
                num_fix_static = np.sum(probs >= static_threshold)
                num_fix_adaptive = np.sum(probs >= adaptive_th)

                # Record this iteration
                collector.add_iteration(
                    step_idx=iteration,
                    probabilities=probs,
                    num_tasks=len(overlapping_task_indices),
                    num_fix_static=int(num_fix_static),
                    num_fix_adaptive=int(num_fix_adaptive),
                    adaptive_threshold=adaptive_th
                )

                print(f"Iteration {iteration}: {len(overlapping_task_indices)} overlapping tasks, "
                      f"mean_prob={np.mean(probs):.3f}, "
                      f"static_fix={num_fix_static}, adaptive_fix={num_fix_adaptive}")

                # Use static threshold for actual solving (can be changed)
                tasks_to_fix = {
                    data['tasks_tuple'][task_idx]
                    for keep, task_idx in zip(probs >= static_threshold, task_label_idx)
                    if keep
                }

                overlapping_jobs_solution = {
                    (job_id, task_id): jobs_solution_sel[(job_id, task_id)]
                    for job_id, task_id in overlapping_task_indices
                    if (job_id, task_id) in jobs_solution_sel and (job_id, task_id) in tasks_to_fix
                }

                # Apply hard filtering
                jobs_data_sel = select_jss_data(sel_task_indices, jobs_data_sel,
                                               jobs_solution=overlapping_jobs_solution)
            else:
                overlapping_jobs_solution = {}
        else:
            overlapping_jobs_solution = {}

        # Solve subproblem
        jobs_solution_sel, machines_assignment_sel, solve_time_sel, subproblem_obj = flexible_jss(
            jobs_data_sel, n_machines, time_limit=time_limit, stop_search_time=stop_search_time,
            machines_start_time=machines_start_time, jobs_start_time=jobs_start_time,
            do_warm_start=False, jobs_solution_warm_start=overlapping_jobs_solution
        )

        # Execute step
        start_loc, jobs_solution, machines_assignment, machines_start_time, jobs_start_time = \
            exec_step(sorted_task_indices, sel_task_indices, sel_i_task_loc, num_tasks,
                      jobs_solution, jobs_solution_sel, machines_assignment, machines_assignment_sel,
                      machines_start_time, jobs_start_time, n_machines, step)

        prev_sel_task_indices = copy.deepcopy(sel_task_indices)
        iteration += 1

        print(f"Progress: {len(jobs_solution)}/{num_tasks} tasks completed\n")

    # Calculate makespan
    makespan = max([jobs_solution[(job_id, task_id)][2] + jobs_solution[(job_id, task_id)][3]
                    for job_id, task_id in sorted_task_indices
                    if (job_id, task_id) in jobs_solution])

    return collector, makespan


def plot_ridgeline(collector: ProbabilityCollector,
                   output_path: str,
                   static_threshold: float = 0.5,
                   figsize: Tuple[int, int] = (16, 12),
                   title: str = "Probability Distribution Evolution across RHO Iterations"):
    """
    Create a ridgeline plot showing probability distribution evolution.
    Professional design suitable for top-tier venues (NeurIPS, ICML, Nature, Science).
    """
    iterations = collector.iterations
    n_iterations = len(iterations)

    if n_iterations == 0:
        print("No iterations to plot!")
        return

    # Set Times New Roman font for academic style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    plt.rcParams['mathtext.fontset'] = 'stix'  # Math font compatible with Times

    # Create figure with two subplots: main ridgeline + adaptive threshold bar
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 3, width_ratios=[3.5, 0.1, 1.5], wspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_adaptive = fig.add_subplot(gs[2])  # Don't share Y axis to preserve labels

    # ===== Professional color scheme for top-tier venues =====
    # Option 1: Viridis-inspired (perceptually uniform, colorblind friendly)
    # Option 2: Blue-Purple gradient (common in Nature/Science)
    # Option 3: Single-hue gradient (clean, professional)

    # Using a refined blue-to-purple gradient (Nature/Science style)
    # with slight desaturation for a more academic look
    def get_academic_color(i, total):
        """Generate colors suitable for academic publications.
        Uses a blue -> teal -> purple gradient with controlled saturation.
        """
        if total <= 1:
            return (0.267, 0.447, 0.769)  # Default blue

        t = i / (total - 1)

        # Define key colors (RGB, values from academic papers)
        # Start: Deep blue, Middle: Teal, End: Purple
        c1 = np.array([0.192, 0.353, 0.608])  # Deep blue (#314A9B)
        c2 = np.array([0.235, 0.557, 0.557])  # Teal (#3C8E8E)
        c3 = np.array([0.545, 0.318, 0.584])  # Purple (#8B5195)

        # Smooth interpolation through the three colors
        if t < 0.5:
            t2 = t * 2
            rgb = c1 * (1 - t2) + c2 * t2
        else:
            t2 = (t - 0.5) * 2
            rgb = c2 * (1 - t2) + c3 * t2

        return tuple(rgb)

    # Alternative: Use matplotlib's cubehelix or viridis
    # from matplotlib import cm
    # cmap = cm.get_cmap('viridis', n_iterations)
    # colors = [cmap(i / (n_iterations - 1))[:3] for i in range(n_iterations)]

    colors = [get_academic_color(i, n_iterations) for i in range(n_iterations)]

    # Parameters for ridgeline - MORE vertical spacing
    max_iterations_to_show = 10  # Only show first 10 iterations
    overlap = 0.0  # No overlap for clear separation
    scale = 1.2  # Increased distribution height (was 0.85)
    vertical_spacing = 1.5  # Increased vertical spacing (was 1.0)
    x_range = np.linspace(0, 1, 200)

    # Filter to first max_iterations_to_show valid iterations
    valid_iterations = []
    for iteration in iterations:
        probs = iteration['probabilities']
        if len(probs) >= 2:
            try:
                kde = stats.gaussian_kde(probs, bw_method='scott')
                valid_iterations.append(iteration)
            except Exception:
                continue
        if len(valid_iterations) >= max_iterations_to_show:
            break

    n_to_plot = len(valid_iterations)

    # Track data for plots
    adaptive_ths = []
    y_positions = []
    iter_labels = []

    # Plot each iteration's distribution (reversed order: top to bottom = increasing iter)
    for plot_idx, iteration in enumerate(valid_iterations):
        probs = iteration['probabilities']

        # Compute KDE
        kde = stats.gaussian_kde(probs, bw_method='scott')
        density = kde(x_range)

        # Normalize density
        density = density / density.max() * scale

        # Y baseline - reversed so iter 1 is at top, iter N is at bottom
        y_base = (n_to_plot - 1 - plot_idx) * vertical_spacing
        y_positions.append(y_base + scale / 2)  # Center position for markers
        iter_labels.append(f"Iter {iteration['step']}")

        # Get color from original index
        color_idx = plot_idx
        color = colors[color_idx] if color_idx < len(colors) else colors[-1]

        # Fill the distribution with academic color scheme
        ax_main.fill_between(x_range, y_base, y_base + density,
                             alpha=0.85, color=color,
                             edgecolor=tuple(min(1, c * 0.7) for c in color),
                             linewidth=0.8)

        # Record data
        adaptive_ths.append(iteration['adaptive_threshold'])

    # Reverse lists to match top-to-bottom order for plotting
    y_positions = y_positions[::-1]
    iter_labels = iter_labels[::-1]
    adaptive_ths = adaptive_ths[::-1]

    # Calculate plot bounds
    y_min = -0.8
    y_max = (n_to_plot - 1) * vertical_spacing + scale + 1.0

    # Draw static threshold line (bright red for high visibility)
    ax_main.axvline(x=static_threshold, color='#E63946', linestyle='--', linewidth=3,
                    label=f'Static tau = {static_threshold}', zorder=10)

    # Draw adaptive threshold line (bright orange/yellow for contrast)
    if len(adaptive_ths) > 1:
        ax_main.plot(adaptive_ths, y_positions, color='#F77F00', linestyle='-', linewidth=3,
                     marker='o', markersize=7, markerfacecolor='#FCBF49', markeredgecolor='#F77F00',
                     markeredgewidth=2, label='Adaptive tau', zorder=11)

    # Shade the "Fix" region with subtle coloring
    ax_main.axvspan(static_threshold, 1.0, alpha=0.06, color='#E63946', zorder=0)
    ax_main.text(0.75, y_max - 0.5, 'Fix Region\n(Static)', fontsize=14,
                 ha='center', va='top', color='#E63946', alpha=0.85, fontweight='medium')

    # Main plot formatting
    ax_main.set_xlim(-0.02, 1.02)
    ax_main.set_ylim(y_min, y_max)
    ax_main.set_xlabel('Predicted Probability', fontsize=18, fontweight='semibold')
    ax_main.set_ylabel('RHO Iteration (Time ->)', fontsize=18, fontweight='semibold')

    # Y-axis ticks showing iteration numbers
    ax_main.set_yticks(y_positions)
    ax_main.set_yticklabels(iter_labels)
    ax_main.tick_params(axis='y', labelsize=14, left=True, labelleft=True)

    # Add ellipsis indicator if there are more iterations
    total_valid_iterations = sum(1 for it in iterations if len(it['probabilities']) >= 2)
    if total_valid_iterations > max_iterations_to_show:
        ax_main.text(-0.02, y_min + 0.5, f'... ({total_valid_iterations - max_iterations_to_show} more)',
                    fontsize=11, ha='left', va='top', color='#666666', style='italic')

    # X-axis tick label size
    ax_main.tick_params(axis='x', labelsize=14)

    # Legend with refined style
    legend = ax_main.legend(loc='upper left', fontsize=14, framealpha=0.95,
                           edgecolor='#CCCCCC', fancybox=False)
    legend.get_frame().set_linewidth(0.5)

    # Subtle grid lines
    ax_main.axvline(x=0.25, color='#AAAAAA', linestyle=':', alpha=0.4, linewidth=0.8, zorder=0)
    ax_main.axvline(x=0.75, color='#AAAAAA', linestyle=':', alpha=0.4, linewidth=0.8, zorder=0)
    ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])

    # Remove spines for cleaner academic look
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.spines['left'].set_linewidth(0.8)
    ax_main.spines['bottom'].set_linewidth(0.8)

    # ===== Adaptive threshold panel (expanded) =====
    y_bar = np.array(y_positions)

    # Bar height matched to vertical spacing
    bar_height = vertical_spacing * 0.5

    # Use same colors as ridgeline for visual consistency (reversed to match)
    for i, (y, th) in enumerate(zip(y_bar, adaptive_ths)):
        # Color index matches the iteration order (iter_labels are already reversed)
        color_idx = n_to_plot - 1 - i
        color = colors[color_idx] if color_idx < len(colors) else colors[-1]
        ax_adaptive.barh(y, th, bar_height, color=color, alpha=0.85,
                        edgecolor=tuple(min(1, c * 0.7) for c in color), linewidth=0.8)

    # Static threshold reference line (matching bright red)
    ax_adaptive.axvline(x=static_threshold, color='#E63946', linestyle='--', linewidth=2.5,
                       label=f'Static tau={static_threshold}')

    # Compute and show range for emphasis
    th_min, th_max = min(adaptive_ths), max(adaptive_ths)
    th_range = th_max - th_min

    ax_adaptive.set_xlabel('Adaptive Threshold tau', fontsize=16, fontweight='semibold')

    # Fine-grained x-ticks to show variation better
    x_padding = 0.08
    x_min = max(0, th_min - x_padding)
    x_max = min(1, th_max + x_padding)
    ax_adaptive.set_xlim(x_min, x_max)

    # More precise tick marks
    tick_interval = 0.05
    ticks = np.arange(np.floor(x_min * 20) / 20, np.ceil(x_max * 20) / 20 + 0.01, tick_interval)
    ax_adaptive.set_xticks(ticks)
    ax_adaptive.tick_params(axis='x', labelsize=12, rotation=45)

    # Add text showing the range (positioned below x-axis label)
    ax_adaptive.text(0.5, -0.18, f'Range: {th_min:.2f} - {th_max:.2f} (delta={th_range:.2f})',
                    transform=ax_adaptive.transAxes, ha='center', va='top',
                    fontsize=11, color='#333333', fontweight='medium')

    # Sync Y axis range with main plot
    ax_adaptive.set_ylim(y_min, y_max)
    ax_adaptive.set_yticks([])
    ax_adaptive.spines['top'].set_visible(False)
    ax_adaptive.spines['right'].set_visible(False)
    ax_adaptive.spines['left'].set_visible(False)
    ax_adaptive.spines['bottom'].set_linewidth(0.8)
    legend2 = ax_adaptive.legend(loc='upper right', fontsize=12, framealpha=0.95,
                                edgecolor='#CCCCCC', fancybox=False)
    legend2.get_frame().set_linewidth(0.5)
    ax_adaptive.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.6)

    # Title with academic style
    fig.suptitle('Probability Distribution Evolution: Static vs Adaptive Thresholding',
                 fontsize=18, fontweight='bold', y=0.98)

    # Subtitle with key insight
    fig.text(0.5, 0.95,
             'Static threshold causes over-fixing when distributions shift; '
             'adaptive threshold maintains consistent fix ratio',
             ha='center', fontsize=13, style='italic', color='#555555')

    # Adjust layout - use subplots_adjust instead of tight_layout for more control
    plt.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.08, wspace=0.08)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\nRidgeline plot saved to: {output_path}")


def plot_threshold_comparison(collector: ProbabilityCollector,
                              output_path: str,
                              static_threshold: float = 0.5):
    """
    Create a comparison plot showing static vs adaptive threshold effectiveness.
    Improved design with cleaner aesthetics and consistent styling.
    """
    iterations = collector.iterations
    n_iterations = len(iterations)

    if n_iterations == 0:
        print("No iterations to plot!")
        return

    # Set consistent style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')

    steps = [it['step'] for it in iterations]
    num_tasks = [it['num_tasks'] for it in iterations]
    fix_static = [it['num_fix_static'] for it in iterations]
    fix_adaptive = [it['num_fix_adaptive'] for it in iterations]
    adaptive_ths = [it['adaptive_threshold'] for it in iterations]
    mean_probs = [it['mean_prob'] for it in iterations]
    std_probs = [it['std_prob'] for it in iterations]

    # Colors (consistent with ridgeline plot)
    color_static = '#E74C3C'  # Red
    color_adaptive = '#2980B9'  # Blue
    color_target = '#27AE60'  # Green
    color_purple = '#8E44AD'  # Purple

    # ===== Plot 1: Fix Ratio Evolution (most important) =====
    ax1 = axes[0, 0]
    ratio_static = [f/n if n > 0 else 0 for f, n in zip(fix_static, num_tasks)]
    ratio_adaptive = [f/n if n > 0 else 0 for f, n in zip(fix_adaptive, num_tasks)]

    # Shade regions above/below target
    ax1.fill_between(steps, ratio_static, 0.5, alpha=0.15, color=color_static,
                     where=[r > 0.5 for r in ratio_static])
    ax1.fill_between(steps, ratio_static, 0.5, alpha=0.15, color=color_static,
                     where=[r <= 0.5 for r in ratio_static])

    # Plot lines with markers
    ax1.plot(steps, ratio_static, color=color_static, linewidth=2.5, marker='o',
             markersize=8, markerfacecolor='white', markeredgewidth=2,
             label=f'Static (tau={static_threshold})')
    ax1.plot(steps, ratio_adaptive, color=color_adaptive, linewidth=2.5, marker='s',
             markersize=8, markerfacecolor='white', markeredgewidth=2,
             label='Adaptive')
    ax1.axhline(y=0.5, color=color_target, linestyle='--', linewidth=2.5,
                alpha=0.8, label='Target 50%', zorder=0)

    ax1.set_xlabel('RHO Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Fix Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Fix Ratio: Static vs Adaptive', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, linestyle=':', zorder=0)

    # ===== Plot 2: Number of fixed tasks =====
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(len(steps))

    # Use consistent bar colors
    bars1 = ax2.bar(x - width/2, fix_static, width, label='Static',
                    color=color_static, alpha=0.85, edgecolor='white', linewidth=0.5)
    bars2 = ax2.bar(x + width/2, fix_adaptive, width, label='Adaptive',
                    color=color_adaptive, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.plot(x, num_tasks, 'k-', marker='D', markersize=6, linewidth=2,
             markerfacecolor='#333333', label='Total Overlapping')

    ax2.set_xlabel('RHO Iteration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Tasks', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Fixed Task Count', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x[::2] if len(x) > 10 else x)
    ax2.set_xticklabels([steps[i] for i in (range(0, len(steps), 2) if len(x) > 10 else range(len(steps)))])
    ax2.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle=':', axis='y')

    # ===== Plot 3: Threshold vs Probability Distribution =====
    ax3 = axes[1, 0]

    # Fill probability mean +/- std
    ax3.fill_between(steps,
                     [max(0, m - s) for m, s in zip(mean_probs, std_probs)],
                     [min(1, m + s) for m, s in zip(mean_probs, std_probs)],
                     alpha=0.25, color=color_target, label='Prob. Mean +/- Std')
    ax3.plot(steps, mean_probs, color=color_target, linewidth=2, linestyle='-', alpha=0.8)

    # Plot threshold lines
    ax3.plot(steps, adaptive_ths, color=color_adaptive, linewidth=2.5, marker='o',
             markersize=8, markerfacecolor='white', markeredgewidth=2,
             label='Adaptive Threshold')
    ax3.axhline(y=static_threshold, color=color_static, linestyle='--', linewidth=2.5,
                label=f'Static Threshold ({static_threshold})')

    ax3.set_xlabel('RHO Iteration', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Threshold / Probability', fontsize=12, fontweight='bold')
    ax3.set_title('(c) Threshold vs Probability Distribution', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='best', fontsize=10, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle=':')
    ax3.set_ylim(0, 1)

    # ===== Plot 4: Overall Probability Histogram =====
    ax4 = axes[1, 1]
    all_probs = np.concatenate([it['probabilities'] for it in iterations if len(it['probabilities']) > 0])

    # Use nicer histogram style
    n, bins, patches = ax4.hist(all_probs, bins=40, density=True, alpha=0.75,
                                 color=color_purple, edgecolor='white', linewidth=0.5)

    # Threshold lines
    ax4.axvline(x=static_threshold, color=color_static, linestyle='--', linewidth=2.5,
                label=f'Static tau = {static_threshold}')
    mean_adaptive = np.mean(adaptive_ths)
    ax4.axvline(x=mean_adaptive, color=color_adaptive, linestyle='-', linewidth=2.5,
                label=f'Avg Adaptive tau = {mean_adaptive:.2f}')

    ax4.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('(d) Overall Probability Distribution', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle=':')
    ax4.set_xlim(0, 1)

    # Main title
    fig.suptitle('Static vs Adaptive Threshold: Comparative Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Comparison plot saved to: {output_path}")


def load_gnn_model(model_dir: Path, model_name: str, checkpoint_type: str, device):
    """Load trained GNN model."""
    # Load normalizers
    normalizer_dir = model_dir / "input_normalizer"
    x_tasks_norm = RunningNormalization(GNN_CONFIG['input_task_dim'])
    x_machines_norm = RunningNormalization(GNN_CONFIG['input_machine_dim'])

    normalizer_loaded = False
    if (normalizer_dir / "normalizer_tasks.pkl").exists():
        x_tasks_norm.load(str(normalizer_dir / "normalizer_tasks.pkl"))
        x_machines_norm.load(str(normalizer_dir / "normalizer_machines.pkl"))
        normalizer_loaded = True
        print(f"Loaded normalizers from {normalizer_dir}")

    # Find checkpoint
    if checkpoint_type == "best":
        candidates = list(model_dir.glob(f"{model_name}_best*.pth"))
        if candidates:
            checkpoint_path = sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]
        else:
            checkpoint_path = None
    else:
        checkpoint_path = model_dir / f"{model_name}_{checkpoint_type}.pth"
        if not checkpoint_path.exists():
            checkpoint_path = None

    if checkpoint_path is None:
        print(f"Error: No checkpoint found for {model_name}")
        return None

    print(f"Loading GNN model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint.get('config', GNN_CONFIG)
    state_dict = checkpoint['model_state_dict']
    has_critical_path = any('critical_path_head' in k for k in state_dict.keys())

    model = HeteroGNNModel(
        input_task_dim=config.get('input_task_dim', GNN_CONFIG['input_task_dim']),
        input_machine_dim=config.get('input_machine_dim', GNN_CONFIG['input_machine_dim']),
        hidden_dim=config.get('hidden_dim', GNN_CONFIG['hidden_dim']),
        num_layers=config.get('num_gnn_layers', GNN_CONFIG['num_gnn_layers']),
        gnn_type=config.get('gnn_type', GNN_CONFIG['gnn_type']),
        num_heads=config.get('num_attention_heads', GNN_CONFIG['num_attention_heads']),
        dropout=config.get('dropout', GNN_CONFIG['dropout']),
        x_tasks_norm=x_tasks_norm if normalizer_loaded else None,
        x_machines_norm=x_machines_norm if normalizer_loaded else None,
        use_critical_path_head=has_critical_path
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"GNN Model loaded successfully")
    return model


def load_lrho_model(model_path: Path, device):
    """Load trained L-RHO MLP model (FlexibleJSSNet)."""
    if not LRHO_MODEL_AVAILABLE:
        print("Error: L-RHO model (FlexibleJSSNet) not available. Check imports.")
        return None

    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return None

    print(f"Loading L-RHO MLP model from {model_path}")

    # Set global constants needed by FlexibleJSSNet
    # These are normally set by get_params_and_fn() in flexible_jss_learn.py
    _flexible_jss_learn.INPUT_TASK_DIM = 15
    _flexible_jss_learn.INPUT_MACHINE_DIM = 11
    _flexible_jss_learn.TASK_FEAT_IDX = 8
    _flexible_jss_learn.IN_PREV_IDX = 7

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model (FlexibleJSSNet from flexible_jss_learn.py)
    model = FlexibleJSSNet().to(device)

    # Load state dict - L-RHO saves the model directly, not wrapped in a dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        # Checkpoint is the state_dict itself
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"L-RHO MLP Model loaded successfully")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Visualize probability distribution evolution across RHO iterations")

    # Instance parameters
    parser.add_argument("--num_jobs", type=int, default=DATASET_CONFIG["num_jobs"])
    parser.add_argument("--num_machines", type=int, default=DATASET_CONFIG["num_machines"])
    parser.add_argument("--num_ops_per_job", type=int, default=DATASET_CONFIG["num_ops_per_job"])
    parser.add_argument("--instance_type", type=str, default=DATASET_CONFIG["instance_type"])

    # RHO parameters
    parser.add_argument("--window", type=int, default=DATASET_CONFIG["window"])
    parser.add_argument("--step", type=int, default=DATASET_CONFIG["step"])
    parser.add_argument("--time_limit", type=int, default=DATASET_CONFIG["time_limit"])
    parser.add_argument("--stop_search_time", type=int, default=DATASET_CONFIG["stop_search_time"])

    # Model parameters
    parser.add_argument("--model_type", type=str, default="gnn", choices=["gnn", "lrho"],
                        help="Model type: 'gnn' (HeteroGNN) or 'lrho' (L-RHO MLP)")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Name of the trained model (for GNN)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Full path to model checkpoint (for L-RHO)")
    parser.add_argument("--checkpoint", type=str, default="best",
                        help="Checkpoint to load: 'best' or epoch number (for GNN)")

    # Threshold parameters
    parser.add_argument("--static_threshold", type=float, default=0.5)
    parser.add_argument("--target_ratio", type=float, default=TEST_CONFIG["adaptive_target_ratio"],
                        help="Target ratio for adaptive threshold")

    # Instance selection
    parser.add_argument("--instance_idx", type=int, default=500,
                        help="Index of test instance to use")

    # Output
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots")

    args = parser.parse_args()

    # Setup device
    device = get_device(DEVICE_CONFIG['prefer'])
    print(f"Using device: {device}")

    # Setup paths
    data_dir = get_data_dir()
    model_root = get_model_dir()
    # model_dir is only needed for GNN models
    if args.model_type == "gnn" and args.model_name:
        model_dir = model_root / args.model_name
    else:
        model_dir = None

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = get_analysis_dir() / "prob_evolution"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Instance path - try multiple possible file patterns
    instance_name = f"j{args.num_jobs}-m{args.num_machines}-t{args.num_ops_per_job}_{args.instance_type}"
    instance_dir = data_dir / "instance" / instance_name

    # Try different file naming patterns
    possible_paths = [
        instance_dir / f"data_{args.instance_idx}.pkl",
        instance_dir / f"instance_{args.instance_idx}.pkl",
        instance_dir / f"{args.instance_idx}.pkl",
    ]

    instance_path = None
    for p in possible_paths:
        if p.exists():
            instance_path = p
            break

    if instance_path is None:
        instance_path = possible_paths[0]  # For error message

    print(f"\n{'='*60}")
    print("Probability Distribution Evolution Visualization")
    print(f"{'='*60}")
    print(f"Instance: {instance_name}")
    print(f"Instance index: {args.instance_idx}")
    print(f"Model type: {args.model_type}")
    if args.model_type == "gnn":
        print(f"Model name: {args.model_name}")
    else:
        print(f"Model path: {args.model_path}")
    print(f"Static threshold: {args.static_threshold}")
    print(f"Target ratio (adaptive): {args.target_ratio}")

    # Load instance
    if instance_path is None or not instance_path.exists():
        print(f"Error: Instance not found. Tried paths:")
        for p in possible_paths:
            print(f"  - {p}")
        return

    with open(instance_path, 'rb') as f:
        instance_data = pickle.load(f)

    if isinstance(instance_data, (tuple, list)):
        jobs_data, n_machines, n_jobs = instance_data[:3]
    else:
        jobs_data = instance_data['jobs_data']
        n_machines = instance_data['n_machines']
        n_jobs = instance_data['n_jobs']

    print(f"Loaded instance: {n_jobs} jobs, {n_machines} machines")
    print(f"Instance path: {instance_path}")

    # Load model based on type
    if args.model_type == "gnn":
        if args.model_name is None:
            print("Error: --model_name is required for GNN model")
            return
        model = load_gnn_model(model_dir, args.model_name, args.checkpoint, device)
    else:  # lrho
        if args.model_path is None:
            # Try default path
            default_model_dir = data_dir / "model" / f"{instance_name}-w{args.window}-s{args.step}-t{args.time_limit}-st{args.stop_search_time}"
            default_model_path = default_model_dir / "model_pw0.5" / "model_pw0.5_180.pth"
            if default_model_path.exists():
                args.model_path = str(default_model_path)
                print(f"Using default L-RHO model path: {args.model_path}")
            else:
                print("Error: --model_path is required for L-RHO model, or default path not found")
                print(f"Tried: {default_model_path}")
                return
        model = load_lrho_model(Path(args.model_path), device)

    if model is None:
        return

    # Run RHO with probability collection
    collector, makespan = rolling_horizon_with_prob_collection(
        n_machines=n_machines,
        n_jobs=n_jobs,
        jobs_data=jobs_data,
        window=args.window,
        step=args.step,
        time_limit=args.time_limit,
        stop_search_time=args.stop_search_time,
        model=model,
        device=device,
        target_ratio=args.target_ratio,
        static_threshold=args.static_threshold
    )

    print(f"\nFinal makespan: {makespan}")
    collector.summary()

    # Generate visualizations
    timestamp = f"{args.instance_idx}"

    # Ridgeline plot
    ridgeline_path = output_dir / f"ridgeline_{instance_name}_{timestamp}.png"
    plot_ridgeline(
        collector,
        str(ridgeline_path),
        static_threshold=args.static_threshold,
        title=f"Probability Distribution Evolution\n{instance_name}, Instance {args.instance_idx}"
    )

    # Comparison plot
    comparison_path = output_dir / f"comparison_{instance_name}_{timestamp}.png"
    plot_threshold_comparison(
        collector,
        str(comparison_path),
        static_threshold=args.static_threshold
    )

    # Save raw data
    data_path = output_dir / f"prob_data_{instance_name}_{timestamp}.pkl"
    with open(data_path, 'wb') as f:
        pickle.dump({
            'iterations': collector.iterations,
            'makespan': makespan,
            'config': vars(args)
        }, f)
    print(f"Raw data saved to: {data_path}")

    print(f"\n{'='*60}")
    print("Visualization complete!")
    print(f"{'='*60}")


def regenerate_plots_from_data(data_path: str, output_dir: str = None, static_threshold: float = 0.5):
    """
    Regenerate plots from saved pickle data without rerunning RHO.

    Args:
        data_path: Path to the saved prob_data_*.pkl file
        output_dir: Output directory for plots (default: same as data file)
        static_threshold: Static threshold value for comparison
    """
    data_path = Path(data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        saved_data = pickle.load(f)

    # Recreate collector from saved data
    collector = ProbabilityCollector()
    collector.iterations = saved_data['iterations']

    # Output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = data_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get base name for output files
    base_name = data_path.stem.replace('prob_data_', '')

    # Generate plots
    ridgeline_path = out_dir / f"ridgeline_{base_name}.png"
    plot_ridgeline(collector, str(ridgeline_path), static_threshold=static_threshold)

    comparison_path = out_dir / f"comparison_{base_name}.png"
    plot_threshold_comparison(collector, str(comparison_path), static_threshold=static_threshold)

    print(f"\nPlots regenerated successfully!")
    print(f"  Ridgeline: {ridgeline_path}")
    print(f"  Comparison: {comparison_path}")


if __name__ == "__main__":
    import sys

    # Check if running in regenerate mode
    if len(sys.argv) >= 2 and sys.argv[1] == '--regenerate':
        # Regenerate mode: python visualize_prob_evolution.py --regenerate <data_path> [--output_dir <dir>]
        parser = argparse.ArgumentParser(description="Regenerate plots from saved data")
        parser.add_argument("--regenerate", action="store_true")
        parser.add_argument("data_path", type=str, help="Path to saved prob_data_*.pkl file")
        parser.add_argument("--output_dir", type=str, default=None)
        parser.add_argument("--static_threshold", type=float, default=0.5)
        args = parser.parse_args()

        regenerate_plots_from_data(args.data_path, args.output_dir, args.static_threshold)
    else:
        main()
