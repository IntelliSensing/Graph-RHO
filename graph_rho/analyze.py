"""Result analysis for Graph-RHO test runs."""
from __future__ import annotations

import argparse
import csv
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from graph_rho.config import DATASET_CONFIG
from graph_rho.utils.path_utils import get_analysis_dir, get_test_results_dir


def load_results(results_path: Path):
    with open(results_path, "rb") as handle:
        return pickle.load(handle)


def print_results_info(results: dict):
    print("\n" + "=" * 60)
    print("Results Metadata")
    print("=" * 60)
    cfg = results.get("dataset_config", {})
    if cfg:
        print(
            f"Instance family: j{cfg.get('num_jobs', '?')}-m{cfg.get('num_machines', '?')}-"
            f"t{cfg.get('num_ops_per_job', '?')}_{cfg.get('instance_type', 'mix')}"
        )
        print(f"Window/step: {cfg.get('window', '?')}/{cfg.get('step', '?')}")
        print(f"Time limit / stop-search: {cfg.get('time_limit', '?')} / {cfg.get('stop_search_time', '?')}")
    opt = results.get("optimization", {})
    if opt:
        print(f"Solver optimization: {opt.get('use_solver_optimization', False)}")
        print(f"Adaptive threshold: {opt.get('use_adaptive_threshold', False)}")
        if opt.get("use_adaptive_threshold"):
            print(f"Adaptive target ratio: {opt.get('adaptive_target_ratio', 0.6)}")
    args = results.get("test_args", {})
    if args:
        print(f"Model: {args.get('model_name', '?')}")
        print(f"Checkpoint: {args.get('load_model_epoch', '?')}")
        print(f"Threshold: {args.get('model_th', '?')}")
        print(f"Test range: [{args.get('test_start', '?')}, {args.get('test_end', '?')})")


def compute_statistics(results: dict):
    stats = {}
    for method in ("model", "default"):
        rows = results.get(method, [])
        if not rows:
            continue
        makespans = [row["makespan"] for row in rows]
        times = [row["wall_time"] for row in rows]
        stats[method] = {
            "count": len(rows),
            "makespan_mean": np.mean(makespans),
            "makespan_std": np.std(makespans),
            "makespan_min": np.min(makespans),
            "makespan_max": np.max(makespans),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "instances": [row["instance"] for row in rows],
            "makespans": makespans,
            "times": times,
        }
    if "model" in stats and "default" in stats:
        model_dict = {row["instance"]: row for row in results["model"]}
        default_dict = {row["instance"]: row for row in results["default"]}
        common = sorted(set(model_dict) & set(default_dict))
        improvements = [
            (default_dict[idx]["makespan"] - model_dict[idx]["makespan"]) / default_dict[idx]["makespan"] * 100
            for idx in common
            if default_dict[idx]["makespan"] != 0
        ]
        stats["comparison"] = {
            "common_instances": len(common),
            "improvement_mean": np.mean(improvements) if improvements else 0.0,
            "improvement_std": np.std(improvements) if improvements else 0.0,
            "win_rate": (sum(1 for imp in improvements if imp > 0) / len(improvements) * 100) if improvements else 0.0,
            "improvements": improvements,
        }
    return stats


def print_statistics(stats: dict):
    print("\n" + "=" * 60)
    print("Statistics Summary")
    print("=" * 60)
    for method in ("model", "default"):
        summary = stats.get(method)
        if not summary:
            continue
        print(f"\n{method.upper()} ({summary['count']} instances):")
        print(f"  Makespan: {summary['makespan_mean']:.2f} +/- {summary['makespan_std']:.2f}")
        print(f"  Range: [{summary['makespan_min']:.2f}, {summary['makespan_max']:.2f}]")
        print(f"  Time: {summary['time_mean']:.2f}s +/- {summary['time_std']:.2f}s")
    comparison = stats.get("comparison")
    if comparison:
        print(f"\nCOMPARISON ({comparison['common_instances']} common instances):")
        print(f"  Avg improvement: {comparison['improvement_mean']:.2f}% +/- {comparison['improvement_std']:.2f}%")
        print(f"  Win rate: {comparison['win_rate']:.1f}%")


def plot_comparison(results: dict, output_dir: Path, prefix: str = ""):
    output_dir.mkdir(parents=True, exist_ok=True)
    if not results.get("model") or not results.get("default"):
        print("Need both model and default results for comparison plots")
        return

    model_dict = {row["instance"]: row for row in results["model"]}
    default_dict = {row["instance"]: row for row in results["default"]}
    common = sorted(set(model_dict) & set(default_dict))
    if not common:
        print("No common instances found")
        return

    model_makespans = [model_dict[idx]["makespan"] for idx in common]
    default_makespans = [default_dict[idx]["makespan"] for idx in common]
    model_times = [model_dict[idx]["wall_time"] for idx in common]
    default_times = [default_dict[idx]["wall_time"] for idx in common]
    improvements = [
        (default_value - model_value) / default_value * 100 if default_value != 0 else 0.0
        for model_value, default_value in zip(model_makespans, default_makespans)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    x = np.arange(len(common))
    width = 0.35
    axes[0].bar(x - width / 2, default_makespans, width, label="Default RHO", alpha=0.8, color="steelblue")
    axes[0].bar(x + width / 2, model_makespans, width, label="Graph-RHO", alpha=0.8, color="coral")
    axes[0].set_xlabel("Instance Index")
    axes[0].set_ylabel("Makespan")
    axes[0].set_title("Per-instance Makespan Comparison")
    axes[0].set_xticks(x[:: max(1, len(x) // 20)])
    axes[0].set_xticklabels([common[idx] for idx in range(0, len(common), max(1, len(x) // 20))], rotation=45)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(x, improvements, color=["green" if imp > 0 else "red" for imp in improvements], alpha=0.7)
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].axhline(y=np.mean(improvements), color="blue", linestyle="--", linewidth=2, label=f"Mean: {np.mean(improvements):.2f}%")
    axes[1].set_xlabel("Instance Index")
    axes[1].set_ylabel("Improvement (%)")
    axes[1].set_title("Makespan Improvement (Positive = Graph-RHO Better)")
    axes[1].set_xticks(x[:: max(1, len(x) // 20)])
    axes[1].set_xticklabels([common[idx] for idx in range(0, len(common), max(1, len(x) // 20))], rotation=45)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{prefix}per_instance_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    box = axes[0].boxplot([default_makespans, model_makespans], labels=["Default RHO", "Graph-RHO"], patch_artist=True)
    box["boxes"][0].set_facecolor("steelblue")
    box["boxes"][1].set_facecolor("coral")
    for patch in box["boxes"]:
        patch.set_alpha(0.7)
    axes[0].set_ylabel("Makespan")
    axes[0].set_title("Makespan Distribution")
    axes[0].grid(axis="y", alpha=0.3)

    box = axes[1].boxplot([default_times, model_times], labels=["Default RHO", "Graph-RHO"], patch_artist=True)
    box["boxes"][0].set_facecolor("steelblue")
    box["boxes"][1].set_facecolor("coral")
    for patch in box["boxes"]:
        patch.set_alpha(0.7)
    axes[1].set_ylabel("Time (s)")
    axes[1].set_title("Solve-time Distribution")
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = output_dir / f"{prefix}distribution_comparison.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def save_csv(results: dict, output_path: Path):
    model_dict = {row["instance"]: row for row in results.get("model", [])}
    default_dict = {row["instance"]: row for row in results.get("default", [])}
    all_instances = sorted(set(model_dict) | set(default_dict))
    with open(output_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Instance", "GraphRHO_Makespan", "GraphRHO_Time", "Default_Makespan", "Default_Time", "Improvement_%"])
        for idx in all_instances:
            row = [idx]
            if idx in model_dict:
                row.extend([model_dict[idx]["makespan"], f"{model_dict[idx]['wall_time']:.2f}"])
            else:
                row.extend(["", ""])
            if idx in default_dict:
                row.extend([default_dict[idx]["makespan"], f"{default_dict[idx]['wall_time']:.2f}"])
            else:
                row.extend(["", ""])
            if idx in model_dict and idx in default_dict and default_dict[idx]["makespan"] != 0:
                improvement = (default_dict[idx]["makespan"] - model_dict[idx]["makespan"]) / default_dict[idx]["makespan"] * 100
                row.append(f"{improvement:.2f}")
            else:
                row.append("")
            writer.writerow(row)
    print(f"Saved: {output_path}")


def list_results_files(results_dir: Path):
    candidates = sorted(results_dir.glob("results_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print(f"No results found in {results_dir}")
        return []
    print("\n" + "=" * 80)
    print(f"Available Results Files in: {results_dir}")
    print("=" * 80)
    info = []
    for idx, path in enumerate(candidates):
        try:
            data = load_results(path)
            opt = data.get("optimization", {})
            test_args = data.get("test_args", {})
            mode = []
            if opt.get("use_solver_optimization"):
                mode.append("solver-opt")
            if opt.get("use_adaptive_threshold"):
                mode.append(f"adaptive({opt.get('adaptive_target_ratio', 0.6)})")
            mode_str = ", ".join(mode) if mode else "standard"
            test_range = f"[{test_args.get('test_start', '?')}, {test_args.get('test_end', '?')})"
        except Exception as exc:
            mode_str = f"unreadable ({exc})"
            test_range = "?"
        print(f"\n[{idx}] {path.name}")
        print(f"    Test range: {test_range}")
        print(f"    Mode: {mode_str}")
        print(f"    Modified: {datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        info.append({"index": idx, "path": path})
    return info


def main():
    parser = argparse.ArgumentParser(description="Analyze Graph-RHO test results")
    parser.add_argument("--num_jobs", type=int, default=DATASET_CONFIG["num_jobs"])
    parser.add_argument("--num_machines", type=int, default=DATASET_CONFIG["num_machines"])
    parser.add_argument("--num_ops_per_job", type=int, default=DATASET_CONFIG["num_ops_per_job"])
    parser.add_argument("--instance_type", type=str, default=DATASET_CONFIG["instance_type"])
    parser.add_argument("--window", type=int, default=DATASET_CONFIG["window"])
    parser.add_argument("--step", type=int, default=DATASET_CONFIG["step"])
    parser.add_argument("--time_limit", type=int, default=DATASET_CONFIG["time_limit"])
    parser.add_argument("--stop_search_time", type=int, default=DATASET_CONFIG["stop_search_time"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--results_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_csv", action="store_true")
    parser.add_argument("--no_plots", action="store_true")
    parser.add_argument("--show_metadata", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    results_dir = get_test_results_dir() / args.model_name
    if args.list:
        list_results_files(results_dir)
        return

    if args.results_file is not None:
        results_path = Path(args.results_file)
        if not results_path.is_absolute():
            results_path = results_dir / args.results_file
    else:
        candidates = sorted(results_dir.glob("results_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"No results found in {results_dir}")
            return
        results_path = candidates[0]

    print(f"Loading results from {results_path}")
    results = load_results(results_path)
    if args.show_metadata:
        print_results_info(results)
    stats = compute_statistics(results)
    print_statistics(stats)

    output_dir = Path(args.output_dir) if args.output_dir else get_analysis_dir() / args.model_name / results_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_plots:
        plot_comparison(results, output_dir)
    if args.save_csv:
        save_csv(results, output_dir / "results.csv")


if __name__ == "__main__":
    main()
