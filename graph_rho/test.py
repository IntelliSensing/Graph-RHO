"""Testing entry point for Graph-RHO."""
from __future__ import annotations

import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from graph_rho.config import DATASET_CONFIG, DEVICE_CONFIG, GNN_CONFIG, TEST_CONFIG
from graph_rho.hetero_gnn_model import HeteroGNNModel
from graph_rho.gnn_data_loader import RunningNormalization
from graph_rho.utils.device_utils import get_device
from graph_rho.utils.lrho_paths import ensure_lrho_makespan_on_path
from graph_rho.utils.path_utils import get_data_dir, get_model_dir, get_test_results_dir


def find_checkpoint(model_dir: Path, model_name: str, checkpoint_type: str = "best") -> Path | None:
    if checkpoint_type in {"best", "last"}:
        direct = model_dir / f"{model_name}_{checkpoint_type}.pth"
        if direct.exists():
            return direct
        candidates = sorted(model_dir.glob(f"{model_name}_{checkpoint_type}*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0]
    if checkpoint_type.isdigit():
        path = model_dir / f"{model_name}_{checkpoint_type}.pth"
        if path.exists():
            return path
    candidates = sorted(model_dir.glob(f"{model_name}*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_model(model_dir: Path, model_name: str, checkpoint_type: str, device):
    normalizer_dir = model_dir / "input_normalizer"
    x_tasks_norm = RunningNormalization(GNN_CONFIG["input_task_dim"])
    x_machines_norm = RunningNormalization(GNN_CONFIG["input_machine_dim"])
    if (normalizer_dir / "normalizer_tasks.pkl").exists():
        x_tasks_norm.load(str(normalizer_dir / "normalizer_tasks.pkl"))
        x_machines_norm.load(str(normalizer_dir / "normalizer_machines.pkl"))
        normalizer_loaded = True
        print(f"Loaded normalizers from {normalizer_dir}")
    else:
        normalizer_loaded = False
        print(f"Warning: normalizers not found at {normalizer_dir}")

    checkpoint_path = find_checkpoint(model_dir, model_name, checkpoint_type)
    if checkpoint_path is None:
        print(f"Error: no checkpoint found for {model_name} ({checkpoint_type})")
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", GNN_CONFIG)
    state_dict = checkpoint["model_state_dict"]
    has_critical_path = any("critical_path_head" in key for key in state_dict)

    model = HeteroGNNModel(
        input_task_dim=config.get("input_task_dim", GNN_CONFIG["input_task_dim"]),
        input_machine_dim=config.get("input_machine_dim", GNN_CONFIG["input_machine_dim"]),
        hidden_dim=config.get("hidden_dim", GNN_CONFIG["hidden_dim"]),
        num_layers=config.get("num_gnn_layers", GNN_CONFIG["num_gnn_layers"]),
        gnn_type=config.get("gnn_type", GNN_CONFIG["gnn_type"]),
        num_heads=config.get("num_attention_heads", GNN_CONFIG["num_attention_heads"]),
        dropout=config.get("dropout", GNN_CONFIG["dropout"]),
        x_tasks_norm=x_tasks_norm if normalizer_loaded else None,
        x_machines_norm=x_machines_norm if normalizer_loaded else None,
        use_critical_path_head=has_critical_path,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


class GNNRolloutWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, data):
        if hasattr(data, "to"):
            data = data.to(self.device)
        with torch.no_grad():
            return self.model(data)

    def parameters(self):
        return self.model.parameters()

    def eval(self):
        self.model.eval()
        return self


def run_gnn_rollout(model, index_start, index_end, data_dir, model_th=0.5,
                    config=None, run_default=True, device="cpu", verbose=False,
                    use_solver_optimization=False, use_adaptive_threshold=False,
                    adaptive_target_ratio=0.6):
    config = dict(DATASET_CONFIG if config is None else config)
    ensure_lrho_makespan_on_path()
    from flexible_jss_main import rolling_horizon

    optimized_available = False
    optimized_rolling_horizon = None
    if use_solver_optimization or use_adaptive_threshold:
        try:
            from graph_rho.optimized_rollout import optimized_rolling_horizon as _optimized_rolling_horizon

            optimized_available = True
            optimized_rolling_horizon = _optimized_rolling_horizon
        except Exception as exc:
            print(f"Warning: optimized rollout unavailable, falling back to standard rollout: {exc}")

    instance_name = (
        f"j{config['num_jobs']}-m{config['num_machines']}-t{config['num_ops_per_job']}_"
        f"{config.get('instance_type', 'mix')}"
    )
    instance_dir = data_dir / "instance" / instance_name
    if not instance_dir.exists():
        print(f"ERROR: instance directory not found: {instance_dir}")
        return {"model": [], "default": [], "metric": "makespan"}

    results = {"model": [], "default": [], "metric": "makespan"}
    model_wrapper = GNNRolloutWrapper(model, device)

    for idx in tqdm(range(index_start, index_end), desc="Testing instances", unit="instance"):
        instance_path = None
        for candidate in (
            instance_dir / f"data_{idx}.pkl",
            instance_dir / f"instance_{idx}.pkl",
            instance_dir / f"{idx}.pkl",
        ):
            if candidate.exists():
                instance_path = candidate
                break
        if instance_path is None:
            print(f"Skipping missing instance {idx}")
            continue

        try:
            with open(instance_path, "rb") as handle:
                jobs_data, n_machines, n_jobs = pickle.load(handle)
        except Exception as exc:
            print(f"Error loading {instance_path}: {exc}")
            continue

        use_optimized = (use_solver_optimization or use_adaptive_threshold) and optimized_available
        try:
            start = time.time()
            if use_optimized:
                result = optimized_rolling_horizon(
                    n_machines=n_machines,
                    n_jobs=n_jobs,
                    jobs_data=jobs_data,
                    window=config["window"],
                    step=config["step"],
                    time_limit=config["time_limit"],
                    stop_search_time=config["stop_search_time"],
                    action_type="model",
                    model=model_wrapper,
                    model_th=model_th,
                    model_decode_strategy="argmax",
                    n_cpus=1,
                    use_solver_optimization=use_solver_optimization,
                    use_adaptive_threshold=use_adaptive_threshold,
                    adaptive_target_ratio=adaptive_target_ratio,
                    verbose=verbose,
                )
            else:
                result = rolling_horizon(
                    n_machines=n_machines,
                    n_jobs=n_jobs,
                    jobs_data=jobs_data,
                    window=config["window"],
                    step=config["step"],
                    time_limit=config["time_limit"],
                    stop_search_time=config["stop_search_time"],
                    action_type="model",
                    model=model_wrapper,
                    model_th=model_th,
                    model_decode_strategy="argmax",
                    n_cpus=1,
                )
            wall_time = time.time() - start
            _, _, _, avg_solve_time, makespan = result
            results["model"].append(
                {
                    "instance": idx,
                    "makespan": makespan,
                    "wall_time": wall_time,
                    "avg_solve_time": avg_solve_time,
                    "use_optimization": use_solver_optimization,
                    "use_adaptive_threshold": use_adaptive_threshold,
                }
            )
        except Exception as exc:
            print(f"Error running model on instance {idx}: {exc}")
            continue

        if not run_default:
            continue
        try:
            start = time.time()
            if use_solver_optimization and optimized_available:
                result = optimized_rolling_horizon(
                    n_machines=n_machines,
                    n_jobs=n_jobs,
                    jobs_data=jobs_data,
                    window=config["window"],
                    step=config["step"],
                    time_limit=config["time_limit"],
                    stop_search_time=config["stop_search_time"],
                    action_type="default",
                    n_cpus=1,
                    use_solver_optimization=True,
                    verbose=verbose,
                )
            else:
                result = rolling_horizon(
                    n_machines=n_machines,
                    n_jobs=n_jobs,
                    jobs_data=jobs_data,
                    window=config["window"],
                    step=config["step"],
                    time_limit=config["time_limit"],
                    stop_search_time=config["stop_search_time"],
                    action_type="default",
                    n_cpus=1,
                )
            wall_time = time.time() - start
            _, _, _, avg_solve_time, makespan = result
            results["default"].append(
                {
                    "instance": idx,
                    "makespan": makespan,
                    "wall_time": wall_time,
                    "avg_solve_time": avg_solve_time,
                    "use_optimization": use_solver_optimization,
                }
            )
        except Exception as exc:
            print(f"Error running default RHO on instance {idx}: {exc}")

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    if results["model"]:
        model_makespans = [row["makespan"] for row in results["model"]]
        model_times = [row["wall_time"] for row in results["model"]]
        print(f"Model instances: {len(results['model'])}")
        print(f"  Avg makespan: {np.mean(model_makespans):.2f} +/- {np.std(model_makespans):.2f}")
        print(f"  Avg wall time: {np.mean(model_times):.2f}s")
    if results["default"]:
        default_makespans = [row["makespan"] for row in results["default"]]
        default_times = [row["wall_time"] for row in results["default"]]
        print(f"Default instances: {len(results['default'])}")
        print(f"  Avg makespan: {np.mean(default_makespans):.2f} +/- {np.std(default_makespans):.2f}")
        print(f"  Avg wall time: {np.mean(default_times):.2f}s")
    if results["model"] and results["default"]:
        model_dict = {row["instance"]: row for row in results["model"]}
        default_dict = {row["instance"]: row for row in results["default"]}
        common = sorted(set(model_dict) & set(default_dict))
        improvements = [
            (default_dict[idx]["makespan"] - model_dict[idx]["makespan"]) / default_dict[idx]["makespan"] * 100
            for idx in common
            if default_dict[idx]["makespan"] != 0
        ]
        if improvements:
            print(f"Comparison on {len(common)} common instances:")
            print(f"  Avg improvement: {np.mean(improvements):.2f}%")
            print(f"  Win rate: {sum(1 for imp in improvements if imp > 0) / len(improvements) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test Graph-RHO on makespan instances")
    parser.add_argument("--num_jobs", type=int, default=DATASET_CONFIG["num_jobs"])
    parser.add_argument("--num_machines", type=int, default=DATASET_CONFIG["num_machines"])
    parser.add_argument("--num_ops_per_job", type=int, default=DATASET_CONFIG["num_ops_per_job"])
    parser.add_argument("--instance_type", type=str, default=DATASET_CONFIG["instance_type"])
    parser.add_argument("--window", type=int, default=DATASET_CONFIG["window"])
    parser.add_argument("--step", type=int, default=DATASET_CONFIG["step"])
    parser.add_argument("--time_limit", type=int, default=DATASET_CONFIG["time_limit"])
    parser.add_argument("--stop_search_time", type=int, default=DATASET_CONFIG["stop_search_time"])
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--load_model_epoch", type=str, default="best")
    parser.add_argument("--model_th", type=float, default=TEST_CONFIG["model_th"])
    parser.add_argument("--test_start", type=int, default=DATASET_CONFIG["test_start"])
    parser.add_argument("--test_end", type=int, default=DATASET_CONFIG["test_end"])
    parser.add_argument("--run_default", action="store_true", default=TEST_CONFIG["run_default"])
    parser.add_argument("--no_default", action="store_true")
    parser.add_argument("--verbose", action="store_true", default=TEST_CONFIG["verbose"])
    parser.add_argument("--use_solver_optimization", action="store_true", default=TEST_CONFIG["use_solver_optimization"])
    parser.add_argument("--use_adaptive_threshold", action="store_true", default=TEST_CONFIG["use_adaptive_threshold"])
    parser.add_argument("--adaptive_target_ratio", type=float, default=TEST_CONFIG["adaptive_target_ratio"])
    args = parser.parse_args()

    if args.no_default:
        args.run_default = False

    dataset_config = {
        "num_jobs": args.num_jobs,
        "num_machines": args.num_machines,
        "num_ops_per_job": args.num_ops_per_job,
        "instance_type": args.instance_type,
        "window": args.window,
        "step": args.step,
        "time_limit": args.time_limit,
        "stop_search_time": args.stop_search_time,
    }

    device = get_device(DEVICE_CONFIG["prefer"])
    print(f"Using device: {device}")

    data_dir = get_data_dir()
    model_dir = get_model_dir() / args.model_name
    results_root = get_test_results_dir() / args.model_name
    results_root.mkdir(parents=True, exist_ok=True)

    model = load_model(model_dir, args.model_name, args.load_model_epoch, device)
    if model is None:
        return

    results = run_gnn_rollout(
        model=model,
        index_start=args.test_start,
        index_end=args.test_end,
        data_dir=data_dir,
        model_th=args.model_th,
        config=dataset_config,
        run_default=args.run_default,
        device=device,
        verbose=args.verbose,
        use_solver_optimization=args.use_solver_optimization,
        use_adaptive_threshold=args.use_adaptive_threshold,
        adaptive_target_ratio=args.adaptive_target_ratio,
    )
    print_summary(results)

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    results["dataset_config"] = dataset_config
    results["test_args"] = vars(args)
    results["optimization"] = {
        "use_solver_optimization": args.use_solver_optimization,
        "use_adaptive_threshold": args.use_adaptive_threshold,
        "adaptive_target_ratio": args.adaptive_target_ratio,
    }
    instance_name = (
        f"j{dataset_config['num_jobs']}-m{dataset_config['num_machines']}-"
        f"t{dataset_config['num_ops_per_job']}_{dataset_config['instance_type']}"
    )
    suffix = ""
    if args.use_solver_optimization:
        suffix += "_opt"
    if args.use_adaptive_threshold:
        suffix += f"_adapt{args.adaptive_target_ratio}"
    results_path = results_root / f"results_{instance_name}_{args.test_start}_{args.test_end}{suffix}_{timestamp}.pkl"
    with open(results_path, "wb") as handle:
        pickle.dump(results, handle)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
