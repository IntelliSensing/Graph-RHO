"""Path helpers for the standalone Graph-RHO release."""
from pathlib import Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    return ensure_dir(get_repo_root() / "data")


def get_outputs_dir() -> Path:
    return ensure_dir(get_repo_root() / "outputs")


def get_model_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "model")


def get_log_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "logs")


def get_test_results_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "test_results")


def get_analysis_dir() -> Path:
    return ensure_dir(get_outputs_dir() / "analysis")
