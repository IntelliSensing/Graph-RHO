#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: bash scripts/prepare_lrho.sh /path/to/l-rho

Applies Graph-RHO's makespan patch to an existing L-RHO checkout.
USAGE
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
patch_file="$repo_root/patches/lrho_makespan.patch"

lrho_root=${1:-${GRAPH_RHO_LRHO_ROOT:-}}
if [[ -z "$lrho_root" ]]; then
  usage
  exit 1
fi

lrho_root=$(cd "$lrho_root" && pwd)
if [[ ! -f "$lrho_root/makespan/flexible_jss_main.py" ]]; then
  echo "Error: $lrho_root does not look like an L-RHO checkout." >&2
  exit 1
fi

if [[ ! -f "$patch_file" ]]; then
  echo "Error: patch file not found at $patch_file" >&2
  exit 1
fi

if patch -p0 --dry-run --forward -d "$lrho_root" < "$patch_file" >/dev/null 2>&1; then
  patch -p0 --forward -d "$lrho_root" < "$patch_file"
  echo "Applied Graph-RHO patch to $lrho_root"
  exit 0
fi

if [[ -f "$lrho_root/makespan/critical_path_utils.py" ]] && \
   grep -q "get_critical_label" "$lrho_root/makespan/flexible_jss_data_common.py"; then
  echo "Patch already appears to be applied in $lrho_root"
  exit 0
fi

echo "Patch could not be applied cleanly." >&2
echo "Check whether your L-RHO checkout differs from the version used for Graph-RHO." >&2
exit 2
