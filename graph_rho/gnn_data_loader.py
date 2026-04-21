"""Data loading utilities for Graph-RHO."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_rho.config import DATASET_CONFIG
from graph_rho.utils.lrho_paths import ensure_lrho_makespan_on_path


_PICKLE_READY = False


def setup_pickle_compatibility() -> None:
    """Ensure the upstream L-RHO makespan modules are importable for pickle loading."""
    global _PICKLE_READY
    if _PICKLE_READY:
        return
    ensure_lrho_makespan_on_path()
    import flexible_jss_data_common  # noqa: F401

    _PICKLE_READY = True


class FlexibleJSSData(Data):
    """PyG data object with batching rules matching L-RHO graph tensors."""

    def __init__(
        self,
        x_tasks,
        x_machines,
        overlap_machine_task_edge_idx,
        overlap_machine_task_edge_val,
        other_machine_task_edge_idx,
        other_machine_task_edge_val,
        task_precedence_edge_idx,
        task_precedence_edge_val,
        task_solution_edge_idx,
        task_solution_edge_val,
        task_label_idx,
        task_label,
        task_critical_idx=None,
        task_critical_label=None,
    ):
        super().__init__()
        self.x_tasks = x_tasks
        self.x_machines = x_machines
        self.overlap_machine_task_edge_idx = overlap_machine_task_edge_idx
        self.overlap_machine_task_edge_val = overlap_machine_task_edge_val
        self.other_machine_task_edge_idx = other_machine_task_edge_idx
        self.other_machine_task_edge_val = other_machine_task_edge_val
        self.task_precedence_edge_idx = task_precedence_edge_idx
        self.task_precedence_edge_val = task_precedence_edge_val
        self.task_solution_edge_idx = task_solution_edge_idx
        self.task_solution_edge_val = task_solution_edge_val
        self.task_label_idx = task_label_idx
        self.task_label = task_label
        self.task_critical_idx = (
            task_critical_idx
            if task_critical_idx is not None
            else torch.tensor([], dtype=torch.long)
        )
        self.task_critical_label = (
            task_critical_label
            if task_critical_label is not None
            else torch.tensor([], dtype=torch.float32)
        )

    def __inc__(self, key, value, *args, **kwargs):
        if key in {
            "x_tasks",
            "x_machines",
            "overlap_machine_task_edge_val",
            "other_machine_task_edge_val",
            "task_precedence_edge_val",
            "task_solution_edge_val",
            "task_label",
            "task_critical_label",
        }:
            return 0
        if key in {"task_precedence_edge_idx", "task_solution_edge_idx"}:
            return torch.tensor([[self.x_tasks.size(0)], [self.x_tasks.size(0)]])
        if key in {"task_label_idx", "task_critical_idx"}:
            return self.x_tasks.size(0)
        if key in {"overlap_machine_task_edge_idx", "other_machine_task_edge_idx"}:
            return torch.tensor([[self.x_machines.size(0)], [self.x_tasks.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in {
            "x_tasks",
            "x_machines",
            "overlap_machine_task_edge_val",
            "other_machine_task_edge_val",
            "task_precedence_edge_val",
            "task_solution_edge_val",
            "task_label_idx",
            "task_label",
            "task_critical_idx",
            "task_critical_label",
        }:
            return 0
        if key in {
            "task_precedence_edge_idx",
            "task_solution_edge_idx",
            "overlap_machine_task_edge_idx",
            "other_machine_task_edge_idx",
        }:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


class GNNFlexibleDataset(Dataset):
    """Dataset wrapper for serialized L-RHO training samples."""

    def __init__(self, data_list: List[Dict]):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        raw_data = self.data_list[idx]["data"]
        task_critical_idx = getattr(raw_data, "task_critical_idx", None)
        task_critical_label = getattr(raw_data, "task_critical_label", None)
        if task_critical_idx is None or len(task_critical_idx) == 0:
            task_critical_idx = torch.tensor(raw_data.task_label_idx, dtype=torch.long)
            task_critical_label = torch.zeros(len(raw_data.task_label_idx), dtype=torch.float32)
        else:
            task_critical_idx = torch.tensor(task_critical_idx, dtype=torch.long)
            task_critical_label = torch.tensor(task_critical_label, dtype=torch.float32)

        return FlexibleJSSData(
            x_tasks=torch.tensor(raw_data.x_tasks, dtype=torch.float32),
            x_machines=torch.tensor(raw_data.x_machines, dtype=torch.float32),
            overlap_machine_task_edge_idx=torch.tensor(
                raw_data.overlap_machine_task_edge_idx, dtype=torch.long
            ),
            overlap_machine_task_edge_val=torch.tensor(
                raw_data.overlap_machine_task_edge_val, dtype=torch.float32
            ),
            other_machine_task_edge_idx=torch.tensor(
                raw_data.other_machine_task_edge_idx, dtype=torch.long
            ),
            other_machine_task_edge_val=torch.tensor(
                raw_data.other_machine_task_edge_val, dtype=torch.float32
            ),
            task_precedence_edge_idx=torch.tensor(
                raw_data.task_precedence_edge_idx, dtype=torch.long
            ),
            task_precedence_edge_val=torch.tensor(
                raw_data.task_precedence_edge_val, dtype=torch.float32
            ),
            task_solution_edge_idx=torch.tensor(
                raw_data.task_solution_edge_idx, dtype=torch.long
            ),
            task_solution_edge_val=torch.tensor(
                raw_data.task_solution_edge_val, dtype=torch.float32
            ),
            task_label_idx=torch.tensor(raw_data.task_label_idx, dtype=torch.long),
            task_label=torch.tensor(raw_data.task_label, dtype=torch.float32),
            task_critical_idx=task_critical_idx,
            task_critical_label=task_critical_label,
        )


def get_dataloader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=["x_tasks", "x_machines"],
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def build_dataset_name(config: Dict) -> str:
    return (
        f"j{config['num_jobs']}-m{config['num_machines']}-t{config['num_ops_per_job']}_"
        f"{config.get('instance_type', 'mix')}"
        f"-w{config['window']}-s{config['step']}-t{config['time_limit']}-st{config['stop_search_time']}"
    )


def load_data(data_dir: Path, start_idx: int, end_idx: int, config: Optional[Dict] = None) -> List[Dict]:
    config = dict(DATASET_CONFIG if config is None else config)
    setup_pickle_compatibility()

    dataset_name = build_dataset_name(config)
    possible_paths = [
        data_dir / "train_data" / dataset_name,
        data_dir / dataset_name / "training_data",
        data_dir / dataset_name,
    ]

    data_path = None
    file_prefix = None
    for candidate in possible_paths:
        if not candidate.exists():
            continue
        for prefix in ("data_idx", "train_", "data_"):
            if list(candidate.glob(f"{prefix}*.pkl")):
                data_path = candidate
                file_prefix = prefix
                break
        if data_path is not None:
            break

    if data_path is None:
        tried = "\n".join(f"  - {path}" for path in possible_paths)
        print(f"WARNING: could not find training data directory.\n{tried}")
        return []

    data_list = []
    for idx in range(start_idx, end_idx):
        possible_names = [
            f"{file_prefix}{idx}.pkl" if file_prefix is not None else None,
            f"train_{idx}.pkl",
            f"data_idx{idx}.pkl",
            f"data_idx_{idx}.pkl",
            f"data_{idx}.pkl",
        ]
        file_path = None
        for name in possible_names:
            if not name:
                continue
            candidate = data_path / name
            if candidate.exists():
                file_path = candidate
                break
        if file_path is None:
            continue
        try:
            with open(file_path, "rb") as handle:
                instance_data = pickle.load(handle)
        except Exception as exc:
            print(f"Error loading {file_path}: {exc}")
            continue
        if isinstance(instance_data, list):
            for item in instance_data:
                item["instance_idx"] = idx
            data_list.extend(instance_data)
        else:
            instance_data["instance_idx"] = idx
            data_list.append(instance_data)

    print(
        f"Loaded {len(data_list)} samples from {data_path} for indices [{start_idx}, {end_idx})."
    )
    return data_list


class RunningNormalization:
    """Running mean/std estimator used by the original release."""

    def __init__(self, dim):
        self.dim = dim
        self.mean = torch.zeros(dim)
        self.var = torch.ones(dim)
        self.count = 0
        self.std = torch.ones(dim)

    def update(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        if self.count == 0:
            self.mean = batch_mean
            self.var = batch_var
        else:
            delta = batch_mean - self.mean
            total_count = self.count + batch_count
            self.mean = self.mean + delta * batch_count / total_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
            self.var = m2 / total_count
        self.count += batch_count
        self.std = torch.sqrt(self.var + 1e-8)

    def normalize(self, x):
        device = x.device if isinstance(x, torch.Tensor) else "cpu"
        return (x - self.mean.to(device)) / self.std.to(device)

    def save(self, path):
        torch.save(
            {
                "mean": self.mean,
                "var": self.var,
                "std": self.std,
                "count": self.count,
                "dim": self.dim,
            },
            path,
        )

    def load(self, path):
        state = torch.load(path, map_location="cpu")
        self.mean = state["mean"]
        self.var = state["var"]
        self.std = state["std"]
        self.count = state["count"]
        self.dim = state["dim"]


def compute_normalizers(data_list: List[Dict]):
    if not data_list:
        raise ValueError("Cannot compute normalizers from an empty data list.")
    sample = data_list[0]["data"]
    x_tasks = np.array(sample.x_tasks)
    x_machines = np.array(sample.x_machines)
    x_tasks_norm = RunningNormalization(x_tasks.shape[1])
    x_machines_norm = RunningNormalization(x_machines.shape[1])
    for item in data_list:
        raw = item["data"]
        x_tasks_norm.update(np.array(raw.x_tasks))
        x_machines_norm.update(np.array(raw.x_machines))
    return x_tasks_norm, x_machines_norm
