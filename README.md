# DAG-Split Reproduction

This workspace contains a reproducible simulation scaffold for multi-user MEC
DNN offloading, a DAG `s-t` Min-Cut reformulation of the pairwise partition
subproblem, and a slot-wise online re-optimization extension.

## Model set

The paper-facing experiments keep five models:

- `AlexNet`
- `VGG19`
- `GoogLeNet`
- `DenseNet121`
- `ResNet101`

The profiling layer may still support additional torchvision models, but the
main experiments and paper text are built around this five-model pool.

## What is implemented

- PyTorch/torchvision profiling for the five retained models.
- A DAG `s-t` Min-Cut partition solver for a fixed MD-ES pair.
- Chain-topology, greedy-cut, random matching, and random-prefix partition
  baselines.
- Hungarian matching over the pairwise cost matrix for the multi-user,
  multi-slot setting.
- A slot-wise online extension that rebuilds the pairwise Min-Cut matrix under
  time-varying bandwidth, active users, and server-side waiting time.

## Experiment structure

The current repo is organized around two static modules and one online
extension.

1. `Single-pair partition`

```text
fixed MD-ES pair -> DAG Min-Cut -> pairwise optimal cost c_ij
```

2. `Unified five-model multi-user matching`

```text
pairwise cost matrix C=[c_ij] -> Hungarian matching -> system-wide assignment
```

3. `Online slot-wise re-optimization`

```text
observe slot state -> rebuild c_ij(t) -> Hungarian matching -> update queue wait
```

## Communication model

The current system uses a fixed analytical compression ratio with a selective
rule:

- if the transmitted tensor is the raw model input, it is uploaded unchanged;
- if the transmitted tensor is an intermediate feature, its uploaded bytes are
  divided by `compression_ratio`.

Formally,

```text
D_up = D_input                  for server-only
D_up = D_feature / rho          for intermediate offloading
```

The latency-energy model is

```text
T = T_local + T_upload + T_wait + T_server + T_down
E = E_local + E_upload
C = alpha * T + beta * E
```

This version does **not** model compression time, decompression time, or
compression-side energy. It only reduces the uploaded intermediate-feature
volume. The default setting is:

```text
compression_ratio = 4.0
```

## Run

Use the Python interpreter requested in `AGENTS.md`:

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe -B run_experiments.py --output outputs_mixed5_ratio4_midonly --random-repeats 30
```

For a quick smoke run:

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe -B run_experiments.py --output outputs_mixed5_ratio4_midonly_smoke --random-repeats 2
```

If a different compression ratio is needed:

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe -B run_experiments.py --output outputs_custom --random-repeats 30 --compression-ratio 2.0
```

## Main outputs

The formal artifact set is under `outputs_mixed5_ratio4_midonly/`.

Single-pair artifacts:

- `outputs_mixed5_ratio4_midonly/single_pair_summary.csv`
- `outputs_mixed5_ratio4_midonly/single_pair_runtime.csv`
- `outputs_mixed5_ratio4_midonly/single_pair_cost.png`
- `outputs_mixed5_ratio4_midonly/single_pair_topology_stress.png`
- `outputs_mixed5_ratio4_midonly/single_pair_bandwidth_sweep.csv`
- `outputs_mixed5_ratio4_midonly/single_pair_local_resource_sweep.csv`
- `outputs_mixed5_ratio4_midonly/single_pair_bar_metrics.csv`
- `outputs_mixed5_ratio4_midonly/single_pair_gain_heatmap.csv`
- `outputs_mixed5_ratio4_midonly/single_pair_bandwidth_lines.png`
- `outputs_mixed5_ratio4_midonly/single_pair_local_gflops_lines.png`
- `outputs_mixed5_ratio4_midonly/single_pair_model_bars.png`
- `outputs_mixed5_ratio4_midonly/single_pair_gain_heatmaps.png`

Unified five-model multi-user artifacts:

- `outputs_mixed5_ratio4_midonly/summary.csv`
- `outputs_mixed5_ratio4_midonly/random_repeats.csv`
- `outputs_mixed5_ratio4_midonly/mixed5_*.csv`
- `outputs_mixed5_ratio4_midonly/mixed5_main_cost.png`
- `outputs_mixed5_ratio4_midonly/mixed5_bandwidth_cost.png`
- `outputs_mixed5_ratio4_midonly/mixed5_bandwidth_match_focus.png`
- `outputs_mixed5_ratio4_midonly/mixed5_beta_cost.png`

Support artifacts:

- `outputs_mixed5_ratio4_midonly/model_topology.csv`
- `outputs_mixed5_ratio4_midonly/algorithm_runtime.csv`
- `outputs_mixed5_ratio4_midonly/algorithm_runtime.png`
- `outputs_mixed5_ratio4_midonly/scalability_runtime.csv`
- `outputs_mixed5_ratio4_midonly/scalability_runtime.png`

Online extension artifacts:

- `outputs_mixed5_ratio4_midonly/online_summary.csv`
- `outputs_mixed5_ratio4_midonly/online_timeseries.csv`
- `outputs_mixed5_ratio4_midonly/online_policy_switches.csv`
- `outputs_mixed5_ratio4_midonly/online_cost_over_time.png`
- `outputs_mixed5_ratio4_midonly/online_queue_over_time.png`

## Current evidence highlights

Results in `outputs_mixed5_ratio4_midonly/` with `compression_ratio=4.0`:

- `Unified five-model mixed scenario`:
  - `MinCut+BMatch` vs `Only-Local`: `94.89%` lower total cost
  - `MinCut+BMatch` vs `Only-Server`: `28.01%` lower total cost
  - `MinCut+BMatch` vs `MinCut+RMatch`: `8.91%` lower total cost
  - `MinCut+BMatch` vs `Rpartition+RMatch`: `91.64%` lower total cost
- `Standard single-pair scenario`:
  - `MinCut`, `ChainTopo`, and `GreedyCut` tie on average cost
  - `AlexNet -> features_2`
  - `VGG19 -> server-only`
  - `GoogLeNet -> maxpool1`
  - `DenseNet121 -> features_pool0`
  - `ResNet101 -> maxpool`
- `GoogLeNet branch-stress scenario`:
  - `MinCut` beats `ChainTopo` and `GreedyCut` by `5.75%`
  - the selected cut is the non-prefix multi-node set
    `inception4a_branch1_conv, inception4a_branch2_0_conv, inception4a_branch3_0_conv, inception4a_branch4_1_conv`
- `Online composite non-stationary scenario`:
  - `Online MinCut+BMatch` vs `Static MinCut+BMatch`: `12.73%` lower average cost
  - `Online MinCut+BMatch` vs `Online Only-Server`: `29.09%` lower average cost
  - `Online MinCut+BMatch` lowers average queue wait from `0.9203 s` to `0.5052 s`

## Paper drafts

- `paper_draft_cn.md`
- `paper_infocom_en.tex`
- `paper_submission_checklist.md`
