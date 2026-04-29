# DAG-Split 复现实验

本项目提供一个可复现的多用户 MEC 场景下 DNN 分割与卸载仿真框架，包括单个 MD-ES 对的 DAG `s-t`
Min-Cut 图建模、多用户场景下的 Hungarian 匹配，以及面向在线动态场景的时隙级重优化扩展。

## 模型集合

面向论文主实验的模型集合包含以下五个模型：

- `AlexNet`
- `TinyYOLOv2`
- `GoogLeNet`
- `DenseNet121`
- `ResNet101`

其中，`TinyYOLOv2` 是项目内置的 profile-only 检测模型，只用于分区与卸载代价建模，不从外部下载权重，也不进行检测精度评估。profiling 层仍保留 VGG 及其他 torchvision 分类模型的历史兼容入口，但主实验与论文叙事均围绕上述五模型展开。

## 已实现功能

- 基于 PyTorch/torchvision 的模型 profiling，以及本地 `TinyYOLOv2` profile 模型。
- 针对固定 MD-ES 对的 DAG `s-t` Min-Cut 分区求解器。
- `ChainTopo`、`GreedyCut`、随机匹配与随机前缀分区等基线方法。
- 多用户、多 server slot 场景下，基于 pairwise cost matrix 的 Hungarian 匹配。
- 在线动态扩展：在带宽、活跃用户和服务端等待时间变化时，按时隙重建 pairwise Min-Cut 代价矩阵并重新匹配。

## 实验结构

当前项目包含两个静态实验模块和一个在线动态扩展。

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

## 默认混合场景参数

论文主实验使用参考 effectiveness experiment 中的 Test1 风格 MEC 参数设置：

- MD 数量与 ES slot 数量：`5` 和 `5`
- MD 计算资源（GFLOPS）：`1.5, 3, 5, 7, 9.5`
- ES 计算资源（GFLOPS）：`22, 30, 35, 40, 48`
- 本地计算功率（W）：`1.2, 1.5, 2, 2.5, 2.8`
- 上行传输功率（W）：`1`
- 主对比实验上行带宽：`1 MB/s`
- 带宽敏感性实验取值（MB/s）：`0.5, 1, 2, 3, 4, 5`

## 通信模型

当前系统采用固定解析压缩比，并使用选择性压缩规则：

- 如果上传的是原始模型输入，则不压缩；
- 如果上传的是中间特征张量，则上传字节数除以 `compression_ratio`。

形式化表示为：

```text
D_up = D_input                  for server-only
D_up = D_feature / rho          for intermediate offloading
```

时延、能耗和总代价模型为：

```text
T = T_local + T_upload + T_wait + T_server + T_down
E = E_local + E_upload
C = alpha * T + beta * E
```

当前版本**不建模**压缩时间、解压缩时间或压缩侧能耗，只通过压缩比降低中间特征的上传数据量。默认设置为：

```text
compression_ratio = 4.0
```

## 在线动态场景实现

在线扩展采用观测驱动的时隙级重优化机制。每个 `OnlineSlotState` 记录当前时隙编号、活跃 MD 集合、上/下行带宽，以及每个 server slot 的外部负载因子。

项目保留三类在线轨迹：

- `online_bandwidth_trace`：所有 MD 保持活跃，上行带宽从 `0.5 MB/s` 变化到 `1 MB/s`，再变化到 `4 MB/s`。
- `online_burst_load`：活跃 MD 集合发生变化，中间阶段部分 server slot 出现突发外部负载。
- `online_composite_nonstationary`：活跃用户、带宽和外部负载同时发生变化。

每个时隙会重新计算有效边缘算力：

```text
effective_es_gflops = base_es_gflops / (1 + external_load_factor)
```

`Online MinCut+BMatch` 在每个时隙使用当前带宽、当前有效边缘算力和当前队列等待时间，重建 pairwise Min-Cut 代价矩阵 `c_ij(t)`，然后对活跃 MD 行执行 Hungarian 匹配。`Static MinCut+BMatch` 只根据平均带宽和平均负载生成一次静态分区/匹配计划，但每个时隙都会在真实当前状态下重新评估该固定计划的时延、能耗与队列等待成本。因此，静态基线会承受当前队列、带宽和服务端算力变化带来的实际代价，但不会改变原先的分区和匹配决策。

队列等待时间按 server slot 和方法分别维护。如果某个被分配任务包含云端执行部分，则其 `server_time` 会加入该 slot 的需求量。下一时隙等待时间更新为：

```text
next_wait = max(0, old_wait + assigned_server_demand - slot_duration)
```

在线实验输出三个视角的 CSV：

- `online_timeseries.csv`：记录每个时隙的 cost、latency、energy、queue wait、活跃 MD 数量、带宽和外部负载。
- `online_policy_switches.csv`：记录每个 MD 的 partition、server 和 combined policy 切换率。
- `online_summary.csv`：记录每个在线场景的平均 cost、平均 queue wait 和策略切换摘要。

需要注意的是，当前在线机制是观测驱动的时隙级重优化，不预测未来轨迹，也不提供 regret、competitive ratio 或 Lyapunov 理论保证。

## 运行方式

使用 `AGENTS.md` 中指定的 Python 解释器：

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe -B run_experiments.py --output outputs_mixed5_yolo_ratio4_midonly --random-repeats 30
```

快速 smoke run：

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe -B run_experiments.py --output outputs_mixed5_yolo_ratio4_midonly_smoke --random-repeats 2
```

如需更换压缩比：

```powershell
D:\Program4Python\anaconda\envs\pytorch\python.exe -B run_experiments.py --output outputs_custom --random-repeats 30 --compression-ratio 2.0
```

## 主要输出

正式实验结果位于 `outputs_mixed5_yolo_ratio4_midonly/`。

单 MD-ES 对实验输出：

- `outputs_mixed5_yolo_ratio4_midonly/single_pair_summary.csv`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_runtime.csv`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_cost.png`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_topology_stress.png`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_bandwidth_sweep.csv`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_local_resource_sweep.csv`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_bar_metrics.csv`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_gain_heatmap.csv`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_bandwidth_lines.png`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_local_gflops_lines.png`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_model_bars.png`
- `outputs_mixed5_yolo_ratio4_midonly/single_pair_gain_heatmaps.png`

五模型多用户统一场景输出：

- `outputs_mixed5_yolo_ratio4_midonly/summary.csv`
- `outputs_mixed5_yolo_ratio4_midonly/random_repeats.csv`
- `outputs_mixed5_yolo_ratio4_midonly/mixed5_*.csv`
- `outputs_mixed5_yolo_ratio4_midonly/mixed5_main_cost.png`
- `outputs_mixed5_yolo_ratio4_midonly/mixed5_bandwidth_cost.png`
- `outputs_mixed5_yolo_ratio4_midonly/mixed5_bandwidth_match_focus.png`
- `outputs_mixed5_yolo_ratio4_midonly/mixed5_beta_cost.png`

辅助分析输出：

- `outputs_mixed5_yolo_ratio4_midonly/model_topology.csv`
- `outputs_mixed5_yolo_ratio4_midonly/algorithm_runtime.csv`
- `outputs_mixed5_yolo_ratio4_midonly/algorithm_runtime.png`
- `outputs_mixed5_yolo_ratio4_midonly/scalability_runtime.csv`
- `outputs_mixed5_yolo_ratio4_midonly/scalability_runtime.png`

在线动态实验输出：

- `outputs_mixed5_yolo_ratio4_midonly/online_summary.csv`
- `outputs_mixed5_yolo_ratio4_midonly/online_timeseries.csv`
- `outputs_mixed5_yolo_ratio4_midonly/online_policy_switches.csv`
- `outputs_mixed5_yolo_ratio4_midonly/online_cost_over_time.png`
- `outputs_mixed5_yolo_ratio4_midonly/online_queue_over_time.png`

## 当前结果摘要

以下结果来自 `outputs_mixed5_yolo_ratio4_midonly/`，实验设置为 `compression_ratio=4.0`，并已使用五模型 TinyYOLOv2 版本重新运行：

- `Unified five-model mixed scenario`：
  - `MinCut+BMatch` 相比 `Only-Local` 总成本降低 `80.24%`
  - `MinCut+BMatch` 相比 `Only-Server` 总成本降低 `60.35%`
  - `MinCut+BMatch` 相比 `MinCut+RMatch` 总成本降低 `4.74%`
  - `MinCut+BMatch` 相比 `Rpartition+RMatch` 总成本降低 `72.37%`
- `Standard single-pair scenario`：
  - `MinCut`、`ChainTopo` 和 `GreedyCut` 的平均 cost 持平
  - `AlexNet -> features_2`
  - `TinyYOLOv2 -> features_7`
  - `GoogLeNet -> maxpool1`
  - `DenseNet121 -> features_pool0`
  - `ResNet101 -> maxpool`
- `GoogLeNet branch-stress scenario`：
  - `MinCut` 相比 `ChainTopo` 和 `GreedyCut` 降低 `5.75%`
  - 选中的 cut 是非前缀多节点集合：
    `inception4a_branch1_conv, inception4a_branch2_0_conv, inception4a_branch3_0_conv, inception4a_branch4_1_conv`
- `Online composite non-stationary scenario`：
  - `Online MinCut+BMatch` 相比 `Static MinCut+BMatch` 平均 cost 降低 `6.28%`
  - `Online MinCut+BMatch` 相比 `Online Only-Server` 平均 cost 降低 `63.85%`
  - `Online MinCut+BMatch` 将平均 queue wait 从 `0.0311 s` 降低到 `0.0167 s`

## 论文支持材料

- `paper_submission_checklist.md`
