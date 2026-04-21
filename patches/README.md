# Graph-RHO patches

`lrho_makespan.patch` is the minimal patch that Graph-RHO requires on top of the upstream `l-rho/makespan` code.

It adds or changes the following pieces:

- `makespan/critical_path_utils.py`: computes critical-path labels from rollout solutions.
- `makespan/flexible_jss_data.py`: exports critical-path labels together with rollout training samples.
- `makespan/flexible_jss_data_common.py`: extends the serialized graph data structure and rollout decoding utilities.

Apply the patch with:

```bash
bash scripts/prepare_lrho.sh /path/to/l-rho
```
