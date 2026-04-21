# Graph-RHO

Developed by IntelliSensing.

Graph-RHO is a graph-based rolling-horizon scheduler for makespan-oriented flexible job shop scheduling (FJSP). This repository contains the paper-aligned implementation of Graph-RHO, including the heterogeneous graph neural network, the critical-path auxiliary task, and the adaptive-threshold rollout strategy used during inference.

Graph-RHO builds on the rolling-horizon formulation introduced by L-RHO and extends it with a graph representation tailored to the structure of FJSP subproblems.

## What is included

This repository contains the code needed to reproduce the Graph-RHO method:

- heterogeneous GNN encoder and prediction heads;
- training code for Graph-RHO;
- rollout evaluation code for Graph-RHO and default rolling horizon;
- analysis scripts for saved test results;
- probability-evolution visualization for static vs adaptive thresholding;
- an optimized CP-SAT backend used during Graph-RHO inference;
- a minimal patch for the upstream `l-rho/makespan` code path.

This release is intentionally scoped to the final Graph-RHO method. Development branches and unrelated experimental variants are not included.

## Method overview

Graph-RHO combines three core ideas:

1. `Mgnn`: a heterogeneous graph neural network that models tasks, machines, precedence edges, and solution-order edges;
2. `Mcpa`: an auxiliary critical-path prediction task that improves the learned rollout policy;
3. `Mthr`: an adaptive thresholding strategy that avoids over-fixing overlap tasks when probability distributions shift across rolling-horizon steps.

The current release is makespan-only.

## Repository layout

```text
Graph-RHO/
|-- graph_rho/              # main package
|-- patches/                # minimal patch on top of upstream L-RHO
|-- scripts/                # helper scripts
|-- third_party/            # optional location for external dependencies
|-- data/                   # instances and serialized training data
`-- outputs/                # checkpoints, logs, results, plots
```

## Dependency on L-RHO

Graph-RHO depends on the upstream makespan version of L-RHO for instance generation, rollout state construction, and the base rolling-horizon scheduling formulation.

This repository does not vendor the full `l-rho/` codebase. Instead, Graph-RHO expects an external L-RHO checkout in one of the following locations:

1. `GRAPH_RHO_LRHO_ROOT`
2. `Graph-RHO/third_party/l-rho`
3. `../l-rho`

After obtaining the upstream L-RHO repository, apply the Graph-RHO patch:

```bash
bash scripts/prepare_lrho.sh /path/to/l-rho
```

The patch adds the critical-path label utilities and the data-collection changes needed by Graph-RHO.

## Installation

Create a clean Python environment and install the repository requirements:

```bash
conda create -n graph-rho python=3.10
conda activate graph-rho
pip install -r requirements.txt
```

PyTorch and PyTorch Geometric should be installed separately for your CPU/CUDA platform before running Graph-RHO.

## Data preparation

Graph-RHO expects all data under `data/`.

### 1. Generate makespan instances

From the patched `l-rho/makespan` directory:

```bash
python gen_instance.py \
  --data_dir /path/to/Graph-RHO/data \
  --n_j 20 \
  --n_m 10 \
  --op_per_job 30 \
  --n_data 600 \
  --data_suffix mix
```

This creates instances under `data/instance/j20-m10-t30_mix/`.

### 2. Collect Graph-RHO training samples

Run the patched upstream collector for each instance index:

```bash
python flexible_jss_main.py \
  --script_action collect_data \
  --jss_data_dir /path/to/Graph-RHO/data/instance/j20-m10-t30_mix \
  --train_data_dir /path/to/Graph-RHO/data/train_data/j20-m10-t30_mix-w80-s30-t60-st3 \
  --stats_dir /path/to/Graph-RHO/outputs/collector_stats \
  --data_idx 0 \
  --window 80 \
  --step 30 \
  --time_limit 60 \
  --stop_search_time 3 \
  --oracle_time_limit 60 \
  --oracle_stop_search_time 3
```

Repeat `--data_idx` over the required train/validation/test range.

## Training

Train Graph-RHO with the default paper-aligned configuration:

```bash
python -m graph_rho.train --model_name graph_rho_main
```

Default settings include:

- hidden dimension `64`
- `2` GNN layers
- `4` attention heads
- dropout `0.1`
- batch size `64`
- learning rate `1e-4`
- `200` epochs
- critical-path loss weight `0.5`
- rollout window/step `80/30`
- adaptive target ratio `0.6`

Checkpoints are stored under `outputs/model/<model_name>/`, and TensorBoard logs are stored under `outputs/logs/<model_name>/`.

## Evaluation

Run Graph-RHO rollout evaluation on the makespan benchmark:

```bash
python -m graph_rho.test \
  --model_name graph_rho_main \
  --load_model_epoch best \
  --test_start 500 \
  --test_end 600 \
  --run_default \
  --use_solver_optimization \
  --use_adaptive_threshold
```

Saved results are written to `outputs/test_results/<model_name>/`.

## Analysis

Generate aggregate statistics and comparison plots from saved test results:

```bash
python -m graph_rho.analyze --model_name graph_rho_main
```

To list available result files first:

```bash
python -m graph_rho.analyze --model_name graph_rho_main --list
```

## Probability-distribution visualization

Generate the probability-evolution plots used to compare static and adaptive thresholding:

```bash
python -m graph_rho.visualize_prob_evolution \
  --model_type gnn \
  --model_name graph_rho_main \
  --instance_idx 500
```

You can also compare against a trained L-RHO MLP checkpoint:

```bash
python -m graph_rho.visualize_prob_evolution \
  --model_type lrho \
  --model_path /path/to/lrho_checkpoint.pth \
  --instance_idx 500
```

Plots are saved under `outputs/analysis/prob_evolution/` by default.

## Notes

- This release focuses on the makespan setting only.
- Large datasets, trained checkpoints, and generated plots are excluded from version control.

## License

This repository is released under the MIT License. See `LICENSE`.

Graph-RHO depends on the external L-RHO codebase, which is not vendored in
this repository. Please review the upstream L-RHO repository terms separately
before redistributing any patched upstream code or a combined release.

## Acknowledgments

Graph-RHO was developed by IntelliSensing.

Graph-RHO is built on top of the open-source L-RHO codebase. We thank the authors of L-RHO for releasing their implementation and making this line of work easier to build on, compare against, and extend.
