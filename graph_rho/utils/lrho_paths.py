"""Helpers for locating the external L-RHO dependency."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from graph_rho.utils.path_utils import get_repo_root


_ENV_VAR = "GRAPH_RHO_LRHO_ROOT"


def candidate_lrho_roots() -> list[Path]:
    repo_root = get_repo_root()
    candidates = []
    env_root = os.environ.get(_ENV_VAR)
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append((repo_root / "third_party" / "l-rho").resolve())
    candidates.append((repo_root.parent / "l-rho").resolve())
    seen = []
    for candidate in candidates:
        if candidate not in seen:
            seen.append(candidate)
    return seen


def resolve_lrho_root(required: bool = True) -> Path | None:
    for root in candidate_lrho_roots():
        if (root / "makespan" / "flexible_jss_main.py").exists():
            return root
    if required:
        message = [
            "Could not locate the external L-RHO checkout.",
            f"Set {_ENV_VAR} or place L-RHO under third_party/l-rho or ../l-rho.",
            "Looked in:",
        ]
        message.extend(f"  - {path}" for path in candidate_lrho_roots())
        raise FileNotFoundError("\n".join(message))
    return None


def ensure_lrho_makespan_on_path() -> Path:
    root = resolve_lrho_root(required=True)
    makespan_dir = root / "makespan"
    for path in (str(root), str(makespan_dir)):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return makespan_dir
