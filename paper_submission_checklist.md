# Submission Checklist

This checklist matches the current static two-module story and the five-model unified scenario under `outputs_mixed5_yolo_ratio4_midonly/`.

## What is already solid

- The paper story is split into:
  1. fixed MD-ES pairwise optimal partitioning and cost evaluation,
  2. unified five-model multi-user assignment over the pairwise cost matrix.
- The retained experimental models are exactly:
  - `AlexNet`
  - `TinyYOLOv2`
  - `GoogLeNet`
  - `DenseNet121`
  - `ResNet101`
- The communication model is selective:
  - `compression_ratio = 4.0`
  - raw input upload is not compressed,
  - intermediate-feature upload is divided by the fixed ratio,
  - compression/decompression time and compression-side energy are not modeled.
- The single-pair module keeps:
  - `Only-Local`
  - `Only-Server`
  - `ChainTopo`
  - `GreedyCut`
  - `MinCut`
- The multi-user summary keeps only:
  - `Only-Local`
  - `Only-Server`
  - `MinCut+BMatch`
  - `MinCut+RMatch`
  - `Rpartition+RMatch`

## Claims that are currently supported

- For a fixed MD-ES pair, the DAG Min-Cut graph yields the globally optimal partition cost under the current latency-energy objective.
- Under the independent `server_slot` abstraction, the static multi-user problem decomposes into pairwise optimal partitioning plus Hungarian matching.
- The current communication model is analytical and restricted:
  - `D_up = D_input` for `server-only`,
  - `D_up = D_feature / rho` for intermediate offloading,
  - no compression or decompression latency is counted,
  - no compression-side power term is counted.
- In the standard single-pair setting, `MinCut` is never worse than `ChainTopo` and `GreedyCut` on the five retained models.
- In the standard single-pair setting, the retained models produce:
  - `AlexNet -> features_2`
  - `TinyYOLOv2 -> features_7`
  - `GoogLeNet -> maxpool1`
  - `DenseNet121 -> features_pool0`
  - `ResNet101 -> maxpool`
- In the GoogLeNet branch-stress scenario, `MinCut` beats `ChainTopo` and `GreedyCut` by `5.75%` and selects a non-prefix multi-node cut.
- In the unified five-model mixed scenario, `MinCut+BMatch` reduces total cost by:
  - `80.24%` vs `Only-Local`
  - `60.35%` vs `Only-Server`
  - `4.74%` vs `MinCut+RMatch`
  - `72.37%` vs `Rpartition+RMatch`

## Claims that are not currently supported

- Do not write that the current compression model validates real hardware compression cost.
- Do not write that the current compression model is lossless or near-lossless.
- Do not write that `MinCut` is always better than `ChainTopo` or `GreedyCut`.
- Do not write that the method has been validated on a real hardware MEC testbed.
- Do not equate `server_slot` with a physical server without qualification.
- Do not write that the multi-user result is globally optimal under shared server queues or shared bandwidth coupling.
- Do not claim dynamic re-optimization, regret, competitive-ratio, or Lyapunov-style guarantees; that part is no longer in project scope.

## Remaining gaps before submission

1. Replace placeholder bibliography entries with full BibTeX metadata.
2. Fill author names, affiliations, and funding statements.
3. Move the LaTeX file to the exact target-conference template if needed.
4. Unify figure fonts, line widths, and caption style for the final PDF.
5. If stronger communication-model claims are needed, add a separate study for realistic compression cost instead of extrapolating from the current fixed-ratio model.
