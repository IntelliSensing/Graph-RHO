"""Configuration for Graph-RHO."""
from graph_rho.utils.path_utils import get_data_dir, get_outputs_dir

DATA_DIR = get_data_dir()
OUTPUTS_DIR = get_outputs_dir()

DATASET_CONFIG = {
    "num_jobs": 20,
    "num_machines": 10,
    "num_ops_per_job": 30,
    "instance_type": "mix",  # Upstream L-RHO makespan naming convention.
    "window": 80,
    "step": 30,
    "time_limit": 60,
    "stop_search_time": 3,
    "train_start": 0,
    "train_end": 450,
    "val_start": 450,
    "val_end": 470,
    "test_start": 500,
    "test_end": 600,
}

GNN_CONFIG = {
    "input_task_dim": 15,
    "input_machine_dim": 11,
    "hidden_dim": 64,
    "num_gnn_layers": 2,
    "gnn_type": "gat",
    "num_attention_heads": 4,
    "dropout": 0.1,
    "machine_task_edge_dim": 2,
    "task_precedence_edge_dim": 1,
    "task_solution_edge_dim": 1,
    "use_global_aggr": True,
    "aggr_type": "mean",
    "num_epochs": 200,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "weight_decay": 1e-5,
    "pos_weight": 0.5,
    "model_th": 0.5,
    "use_critical_path": True,
    "critical_weight": 0.5,
    "critical_pos_weight": 1.0,
    "save_every": 10,
    "eval_every": 10,
}

TEST_CONFIG = {
    "model_th": 0.5,
    "run_default": False,
    "verbose": True,
    "use_solver_optimization": True,
    "use_adaptive_threshold": True,
    "adaptive_target_ratio": 0.6,
    "adaptive_min_threshold": 0.3,
}

DEVICE_CONFIG = {
    "prefer": "auto",
}
