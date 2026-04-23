# Submission Checklist

This checklist matches the current five-model unified scenario and the online
extension artifact set under `outputs_mixed5_ratio4_midonly/`.

## What is already solid

- The paper story is split into:
  1. fixed MD-ES pairwise optimal partitioning,
  2. unified five-model multi-user assignment over the pairwise cost matrix,
  3. slot-wise online re-optimization under non-stationary conditions.
- The retained experimental models are exactly:
  - `AlexNet`
  - `VGG19`
  - `GoogLeNet`
  - `DenseNet121`
  - `ResNet101`
- The communication model is selective:
  - `compression_ratio = 4.0`
  - raw input upload is not compressed,
  - intermediate-feature upload is divided by the fixed ratio,
  - compression/decompression time and compression-side energy are not modeled.
- The multi-user summary keeps only:
  - `Only-Local`
  - `Only-Server`
  - `MinCut+BMatch`
  - `MinCut+RMatch`
  - `Rpartition+RMatch`
- The single-pair module keeps:
  - `Only-Local`
  - `Only-Server`
  - `ChainTopo`
  - `GreedyCut`
  - `MinCut`
- The online extension evaluates:
  - time-varying bandwidth traces,
  - burst-load traces,
  - composite non-stationary traces.

## Claims that are currently supported

- For a fixed MD-ES pair, the DAG Min-Cut graph yields the globally optimal
  partition cost under the current latency-energy objective.
- Under the independent `server_slot` abstraction, the multi-user problem
  decomposes into pairwise optimal partitioning plus Hungarian matching.
- The current communication model is analytical and restricted:
  - `D_up = D_input` for `server-only`,
  - `D_up = D_feature / rho` for intermediate offloading,
  - no compression or decompression latency is counted,
  - no compression-side power term is counted.
- In the standard single-pair setting, `MinCut` is never worse than
  `ChainTopo` and `GreedyCut` on the five retained models.
- In the standard single-pair setting, the retained models produce:
  - `AlexNet -> features_2`
  - `VGG19 -> server-only`
  - `GoogLeNet -> maxpool1`
  - `DenseNet121 -> features_pool0`
  - `ResNet101 -> maxpool`
- In the GoogLeNet branch-stress scenario, `MinCut` beats `ChainTopo` and
  `GreedyCut` by `5.75%` and selects a non-prefix multi-node cut.
- In the unified five-model mixed scenario, `MinCut+BMatch` reduces total cost by:
  - `94.89%` vs `Only-Local`
  - `28.01%` vs `Only-Server`
  - `8.91%` vs `MinCut+RMatch`
  - `91.64%` vs `Rpartition+RMatch`
- In the online composite non-stationary scenario, `Online MinCut+BMatch`
  reduces average cost by `12.73%` over `Static MinCut+BMatch` and by
  `29.09%` over `Online Only-Server`.
- In the online bandwidth-trace scenario, `Online MinCut+BMatch` reduces
  average cost by `20.08%` over `Static MinCut+BMatch` and by `40.80%` over
  `Online Only-Server`.
- In the online burst-load scenario, `Online MinCut+BMatch` reduces average
  cost by `20.93%` over `Static MinCut+BMatch` and by `18.69%` over
  `Online Only-Server`.
- Under a constant one-slot state, the online `MinCut+BMatch` result collapses
  exactly to the static `MinCut+BMatch` result.

## Claims that are not currently supported

- Do not write that the current compression model validates real hardware
  compression cost.
- Do not write that the current compression model is lossless or near-lossless.
- Do not write that `MinCut` is always better than `ChainTopo` or `GreedyCut`.
- Do not write that the method has been validated on a real hardware MEC testbed.
- Do not equate `server_slot` with a physical server without qualification.
- Do not write that the multi-user result is globally optimal under shared
  server queues or shared bandwidth coupling.
- Do not write that the online extension is globally optimal over an unknown
  future trajectory.
- Do not write regret, competitive-ratio, or Lyapunov-style guarantees.

## Remaining gaps before submission

1. Replace placeholder bibliography entries with full BibTeX metadata.
2. Fill author names, affiliations, and funding statements.
3. Move the LaTeX file to the exact target-conference template.
4. Unify figure fonts, line widths, and caption style for the final PDF.
5. If stronger communication-model claims are needed, add a separate study for
   realistic compression cost instead of extrapolating from the current fixed-ratio model.
6. If stronger theory is needed for the online part, add queue-coupled
   analysis instead of extrapolating from the current slot-wise controller.
