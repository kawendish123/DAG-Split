# 面向多用户移动边缘推理的 DAG 最小割分区与全局匹配

## 摘要

移动边缘推理中的协同 DNN 执行同时受到终端算力、无线传输和终端能耗约束。面对链式模型与复杂 DAG 模型并存的多用户场景，仅依靠端点执行策略无法稳定兼顾时延与能耗。本文将该问题组织为两个静态模块和一个在线扩展。模块一针对固定移动设备与边缘资源槽位对，构造 DAG `s-t` 最小割图，直接求解分区子问题的最优代价 `c_ij`；模块二将所有 `c_ij` 组成代价矩阵 `C=[c_ij]`，再通过 Hungarian 算法求解全局匹配。通信部分采用选择性固定压缩率模型：`server-only` 时原始输入按原始大小上传，中间切分时仅对跨边界中间特征按固定 `4x` 比例缩减上传字节数，不计压缩、解压时间和压缩功耗。实验覆盖 `AlexNet`、`VGG19`、`GoogLeNet`、`DenseNet121` 和 `ResNet101` 五个模型。模块一结果表明，`MinCut` 在标准单对场景下相对 `Only-Local` 和 `Only-Server` 的平均代价分别下降 `92.20%` 和 `33.19%`，并在复杂 DAG/残差模型上出现稳定的真实中间切分。统一五模型多用户场景中，`MinCut+BMatch` 相比 `Only-Local`、`Only-Server` 和 `Rpartition+RMatch` 的总成本分别下降 `94.89%`、`28.01%` 和 `91.64%`。在复合非平稳在线场景中，时隙级重优化相对 `Static MinCut+BMatch` 和 `Online Only-Server` 的平均成本分别下降 `12.73%` 和 `29.09%`，平均排队等待时间由 `0.9203 s` 降至 `0.5052 s`。结果说明，“单对最优分区 + 全局匹配 + 时隙级重优化”是一条结构清晰且可验证的系统优化路径。

## 1. 引言

协同推理真正要解决的问题不是“是否卸载”，而是“在哪里切分模型”以及“多个用户如何共享边缘资源”。对 AlexNet 和 VGG 这类顺序模型，前缀切分足以描述主要执行路径；但对 GoogLeNet 的 Inception 分支、DenseNet121 的密集连接和 ResNet101 的深层残差结构，单前缀切分无法完整表达真实的数据依赖关系。若固定 MD-ES 对上的分区代价本身不够精确，系统级匹配得到的也只是建立在粗糙代价矩阵上的局部最优。

为此，本文将问题拆成两个逻辑清晰的静态模块，再在其上叠加在线扩展：

1. 固定 MD-ES 对的最优分区：对每个移动设备 `i` 和边缘槽位 `j` 求出全局最优分区代价 `c_ij`。
2. 多用户全局匹配：将 `c_ij` 组成代价矩阵后，再求系统级最优分配。

在此基础上，本文进一步加入时隙级在线扩展：环境状态变化时，重复执行“pairwise Min-Cut + Hungarian matching”，从而形成观测驱动的滚动重优化框架。

本文的核心贡献概括如下：

1. 将固定 MD-ES 对上的分区子问题表述为 DAG 最小割问题，使复杂拓扑模型的分区决策可以直接由统一图模型求解。
2. 将多用户系统写成“逐对最优分区 + 全局线性匹配”的两阶段结构，并明确该分解成立的条件边界。
3. 通过带宽扫描、本地算力扫描、三联指标柱状图和 gain 热力图系统展示 `MinCut` 的有效区域，再在统一五模型多用户场景和在线非平稳场景中验证系统级收益。

从全文结构看，接下来的章节依次回答四个问题。相关工作部分说明本章与已有协同推理、MEC 调度和 DNN 分区研究之间的关系；系统模型部分给出两阶段问题定义和成立条件；方法部分说明固定 MD-ES 对上的 DAG 最小割如何编码原始分区代价；实验与结果部分分别从单对有效性、多用户匹配收益和在线扩展三个层面给出证据。

## 2. 相关工作

### 2.1 协同 DNN 推理与模型分区

协同 DNN 推理的核心目标是在终端与边缘之间分担计算，以缓解端侧算力不足和能耗过高的问题。早期代表性工作大多围绕链式 CNN 展开，将模型看作顺序层堆叠结构，再通过枚举单一切分点比较本地执行、部分卸载和全边缘执行的代价。这类建模在 AlexNet、VGG 等顺序模型上是自然的，也为后续 MEC 卸载研究提供了基本的时延和能耗分析框架。

然而，随着实际推理模型从纯链式结构演化到多分支和多汇合结构，单前缀切分的表达能力逐渐成为瓶颈。GoogLeNet 的 Inception 模块、DenseNet 的跨层密集连接以及深层 ResNet 中的大量跳连，都意味着“切在第几层之后”并不能完整描述局部执行和边缘执行之间的真实边界。因此，若仍用链式枚举来刻画复杂模型，分区问题本身就会被过度简化。

### 2.2 多用户 MEC 调度与匹配

与单用户协同推理不同，多用户 MEC 场景不仅要决定每个任务如何切分，还要处理多个终端竞争边缘资源的问题。已有研究通常会引入资源分配、任务调度、二分匹配或整数规划等机制，以降低系统总时延、总能耗或二者的加权和。在线场景下，还需要进一步考虑时变带宽、动态队列和活跃用户集合变化带来的额外耦合。

这类研究的一个共同特点是：系统级调度质量高度依赖于 pairwise 代价矩阵的准确性。如果固定 MD-ES 对的代价只是启发式近似，那么后续无论采用 Hungarian 匹配、贪心匹配还是更复杂的调度算法，系统最优性都难以成立。因此，系统级优化的前提不是“先有一个匹配器”，而是“先有一个可信的 pairwise 代价求解器”。

### 2.3 本文的位置

基于上述观察，本文不把研究重点放在重新发明多用户调度器，而是将问题重心前移到固定 MD-ES 对的分区代价建模。本文的立场是：只要能够在复杂 DAG 模型上准确求得 `c_ij`，那么多用户阶段就可以在明确假设下退化为标准线性分配问题。换言之，本文强调的是“先把子问题求准，再讨论系统级组合优化”，这也是全文采用“单对最优分区 + 全局匹配 + 在线扩展”这一组织方式的原因。

## 3. 系统模型与问题分解

### 3.1 异构 MEC 场景

设移动设备集合为 `M={1,...,m}`，边缘资源槽位集合为 `S={1,...,n}`。设备 `i` 的本地算力和功率分别记为 `g_i^L` 和 `P_i^L`，上行发射功率记为 `P_i^U`；槽位 `j` 的边缘算力记为 `g_j^E`。本文采用 `server_slot` 抽象，即每个槽位被视为一个已经吸收容量约束的独立可分配单元。该抽象的直接好处是：多用户竞争首先被压缩为“哪个任务占用哪个槽位”的分配问题，而不必在静态主线上额外引入更复杂的共享队列耦合。

每个 DNN 被表示为有向无环图 `G=(V,E)`。对任意节点 `v in V`，其计算量由 profiling 阶段得到，记为 `f_v`。于是固定设备与槽位后，本地与边缘执行时间可统一写为：

```text
t_v^L = f_v / g_i^L
t_v^E = f_v / g_j^E
```

若某个分区使得部分节点在本地执行、其余节点在边缘执行，则通信开销由所有 `local-to-cloud` 边界共同决定。通信部分采用选择性固定压缩率模型：

```text
D_up = D_input                 for server-only
D_up = D_feature / rho         for intermediate offloading
```

其中默认 `rho = 4.0`。当前系统不计压缩时间、解压时间和压缩侧能耗。

该建模的含义是：全边缘执行时直接上传原始输入，不假设输入也能通过相同机制获得同等压缩收益；而发生中间切分时，仅对边界中间特征使用固定压缩率，从而在不引入额外硬件测量参数的前提下反映“中间特征可通过近无损量化降低传输量”的趋势。

### 3.2 模块一：固定 MD-ES 对的分区子问题

对固定设备 `i` 和槽位 `j`，设可行分区集合为 `P_ij`，则分区子问题为：

```text
c_ij = min_{P in P_ij} (alpha * T_ij(P) + beta * E_ij(P))
```

其中总时延为：

```text
T_ij(P) = T_local + T_upload + T_wait + T_server + T_down
```

移动端能耗为：

```text
E_ij(P) = E_local + E_upload = P_i^L * T_local + P_i^U * T_upload
```

服务器端能耗不计入目标函数，目标聚焦于移动端侧的时延-能耗权衡。

更具体地，若分区 `P` 对应的本地节点集合为 `V_L`，边缘节点集合为 `V_E`，传输边界节点集合为 `B(P)`，则有：

```text
T_local = sum_{v in V_L} t_v^L
T_server = sum_{v in V_E} t_v^E
T_upload = sum_{u in B(P)} D_u^up / B_up
T_down = D_out / B_down
```

其中 `D_u^up` 在 `u = input` 时取原始输入大小，在 `u` 为中间特征时取压缩后的特征大小。这样定义之后，固定 MD-ES 对上的分区代价就是一个完全参数化、可比较的解析量。

### 3.3 模块二：多用户全局匹配

当所有 `c_ij` 已知后，多用户全局匹配问题可写为：

```text
min_X sum_i sum_j c_ij x_ij
```

满足：

```text
sum_j x_ij = 1
sum_i x_ij <= 1
x_ij in {0,1}
```

这是标准线性分配问题，可由 Hungarian 算法求解。

### 3.4 分解成立条件

**命题 1**：在 `server_slot` 相互独立、不同 MD-ES 对之间不存在共享带宽或共享排队耦合的前提下，原始多用户问题可以精确分解为“逐对最优分区 + 全局线性匹配”。

需要强调的是，这里的“全局最优匹配”仅在当前 `server_slot` 抽象下成立。如果未来回到真实共享服务器和共享队列模型，则 `c_ij` 将依赖全局分配状态，需要改用更一般的整数规划或最小费用流建模。

## 4. 固定 MD-ES 对的 DAG 最小割分区

### 4.1 图构造

本文使用 `torch.fx` 将 PyTorch 模型追踪为算子级 DAG，并将其转化为 `s-t` 最小割图：

1. 源点侧表示本地执行，汇点侧表示边缘执行。
2. 对每个节点 `v` 增加两条一元代价边：
   - `source -> v`，容量为边缘执行代价；
   - `v -> sink`，容量为本地执行代价。
3. 对每个可能发生 `local-to-cloud` 传输的源节点，引入辅助传输节点，使同一个中间特征即使被多个云端后继消费，也只计一次上传代价。
4. 对依赖关系加入无穷大反向约束边，禁止 `cloud-to-local` 回流。

### 4.2 等价性

**定理 1**：对固定 MD-ES 对和固定带宽条件，上述构造得到的 `s-t` 最小割与原始分区子问题等价。

证明思路如下：

1. 任意合法分区都对应一个源点侧/汇点侧划分。
2. 节点放在本地还是边缘的代价，由一元边准确编码。
3. 中间特征上传代价通过辅助传输节点准确编码且只计一次。
4. 非法的 `cloud-to-local` 回流一定切到无穷大边，因此不会成为最优解。

上述四点共同说明，任意合法分区和一组有限代价割之间存在一一对应关系。换言之，最小割值不是某种松弛下界，而是原始分区子问题目标值在图上的精确重写。因此，对固定状态下的单对问题而言，`MinCut` 给出的不是“较好的切分”，而是当前建模前提下的最优切分。

### 4.3 复杂度

设单个模型 DAG 的节点数和边数分别为 `|V|` 和 `|E|`，则整体复杂度为：

```text
O(|M| * |S| * MinCut(V,E) + Hungarian(max(|M|, |S|)^3))
```

前一项对应 pairwise 代价矩阵构造，后一项对应系统级匹配。

在工程上，这一复杂度也给出一个直接结论：当模型规模或用户数扩大时，主要成本并不在 Hungarian 匹配，而在于为所有 MD-ES 对构造并求解 pairwise 图。这一点将在后续运行时间实验中得到验证。

## 5. 实验设计

### 5.1 模型集合

本文最终保留五个模型：

- AlexNet
- VGG19
- GoogLeNet
- DenseNet121
- ResNet101

根据 [model_topology.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/model_topology.csv)：

- AlexNet：21 个节点，0 个汇合节点，1.43 GFLOPs
- VGG19：45 个节点，0 个汇合节点，39.28 GFLOPs
- GoogLeNet：196 个节点，9 个汇合节点，3.01 GFLOPs
- DenseNet121：430 个节点，58 个汇合节点，5.72 GFLOPs
- ResNet101：344 个节点，33 个汇合节点，15.66 GFLOPs

这五个模型分别覆盖轻量链式、重链式、多分支 DAG、密集连接 DAG 和深层残差 DAG。

### 5.2 单对分区场景

标准单对场景统一使用：

- `md_gflops = 3.0`
- `es_gflops = 35.0`
- `local_power = 2.0 W`
- `upload_power = 1.0 W`
- 上行带宽 `1 MB/s`
- 下行带宽 `10 MB/s`

围绕该标准场景，模块一进一步展开四组参数化证据：

- 带宽折线图：固定 `md_gflops = 3.0`，扫描 `bandwidth_up = 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5 MB/s`
- 本地算力折线图：固定 `bandwidth_up = 1 MB/s`，扫描 `md_gflops = 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0`
- 三联柱状图：在标准单对点上比较五个模型的 `latency`、`energy` 和 `objective cost`
- gain 热力图：在“带宽 × 本地算力”平面上可视化 `MinCut` 相对更优端点方案的归一化收益

GoogLeNet 压力场景使用：

- `md_gflops = 1.0`
- `es_gflops = 5.0`
- 上行带宽 `0.05 MB/s`

单对对照方法为：

- `Only-Local`
- `Only-Server`
- `ChainTopo`
- `GreedyCut`
- `MinCut`

模块一的四类图均只保留 `Only-Local`、`Only-Server` 和 `MinCut` 三种方法，目的是把问题严格限定在“固定 MD-ES 对上的分区有效性”上，而不让匹配策略和随机性因素干扰判断。`ChainTopo` 与 `GreedyCut` 仅在标准单对统计表和 GoogLeNet 压力案例中保留，用于补充拓扑表达能力的比较。

### 5.3 评价指标

本章使用三类指标评估系统行为。

第一类是分区目标值，即 `objective cost = alpha * latency + beta * energy`。这是全文的主指标，用于统一衡量时延与移动端能耗。

第二类是分解指标，即 `latency` 与 `energy`。它们分别回答“收益来自更低时延还是更低能耗”，用于解释 `MinCut` 相对端点方案的来源。

第三类是归一化收益指标 `gain`，定义为：

```text
gain = (min(Only-Local, Only-Server) - MinCut) / min(Only-Local, Only-Server)
```

该指标用于热力图。若 `gain = 0`，说明 `MinCut` 已退化为更优端点方案；若 `gain > 0`，说明中间切分相对端点方案带来了额外收益。相比直接显示原始 cost，`gain` 更适合跨模型比较“MinCut 真正有价值的参数区域”。

### 5.4 统一五模型多用户场景

多用户实验统一采用 `mixed5` 场景。该场景包含 5 个移动设备和 5 个边缘槽位：

| MD | 模型 | 本地 GFLOPS | 本地功率(W) |
| --- | --- | ---: | ---: |
| 0 | AlexNet | 4.0 | 1.4 |
| 1 | VGG19 | 1.8 | 1.8 |
| 2 | GoogLeNet | 2.8 | 2.0 |
| 3 | DenseNet121 | 2.2 | 2.2 |
| 4 | ResNet101 | 1.6 | 2.6 |

边缘槽位算力为：

```text
(20.0, 24.0, 28.0, 32.0, 36.0) GFLOPS
```

多用户对照方法为：

- `Only-Local`
- `Only-Server`
- `MinCut+BMatch`
- `MinCut+RMatch`
- `Rpartition+RMatch`

此外还执行：

- 带宽敏感性扫描：`0.1, 0.2, 0.5, 1, 2, 3, 4, 5 MB/s`
- `beta` 敏感性扫描：`0, 0.5, 1, 2, 4`

这里采用统一五模型场景，而不再人为拆成两套主场景，目的是让链式模型、重链式模型和复杂 DAG 模型同时出现在一个资源竞争环境中，从而更真实地观察统一代价矩阵上的系统行为。

### 5.5 在线扩展

在线扩展采用时隙级滚动重优化。每个时间槽开始时，系统观测：

- 当前活跃移动设备集合
- 当前上/下行带宽
- 每个 `server_slot` 的等待时间和外部负载

随后重新构造当前时隙的 pairwise cost matrix，再执行 Hungarian 匹配。在线对照方法为：

- `Online Only-Local`
- `Online Only-Server`
- `Static MinCut+BMatch`
- `Online MinCut+BMatch`

在线场景包括：

- 时变带宽
- 突发负载
- 复合非平稳

## 6. 实验结果

### 6.1 模块一：带宽折线图

[single_pair_bandwidth_lines.png](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_bandwidth_lines.png) 和 [single_pair_bandwidth_sweep.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_bandwidth_sweep.csv) 展示了固定 `md_gflops = 3.0`、扫描上行带宽时三种端点方法与 `MinCut` 的关系。

整体规律很清楚：

- `AlexNet` 在 `0.05 MB/s` 时退化为 `local-only`，从 `0.1 MB/s` 开始稳定切在 `features_2`，在 `1 MB/s` 时相对更优端点方案的收益达到 `77.55%`
- `VGG19` 在整个带宽区间内都保持 `server-only`，说明其最优策略主要由重计算量主导
- `GoogLeNet` 在 `0.05` 和 `0.1 MB/s` 时更接近本地执行，在 `0.5` 到 `3 MB/s` 之间稳定切在 `maxpool1`，到 `5 MB/s` 时退化为 `server-only`
- `DenseNet121` 从 `0.1 MB/s` 开始切在 `features_pool0`，到 `5 MB/s` 时退化为 `server-only`
- `ResNet101` 在低到中等带宽区间内稳定切在 `maxpool`，到高带宽时再退化为 `server-only`

这组结果说明，`MinCut` 并不会固定偏向某一种端点，而是会随着通信条件变化在 `local-only`、中间切分和 `server-only` 三种策略之间迁移。从机理上看，低带宽区域更强调上传量压缩，中高带宽区域更强调边缘高算力，因此最优切分点会在同一模型内部发生移动。不同模型迁移拐点不同，则说明最优分区既受外部网络条件影响，也受模型内部降采样位置和拓扑结构影响。

### 6.2 模块一：本地算力折线图

[single_pair_local_gflops_lines.png](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_local_gflops_lines.png) 和 [single_pair_local_resource_sweep.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_local_resource_sweep.csv) 固定 `bandwidth_up = 1 MB/s`，扫描本地算力。

该图反映出两个稳定现象：

- `Only-Server` 不随 `md_gflops` 变化，而 `Only-Local` 随本地算力提升明显下降
- `MinCut` 的收益区间具有明显模型差异

具体看，`VGG19` 在 `1.0` 到 `6.0 GFLOPS` 全区间内始终保持 `server-only`，对应 gain 始终为 `0`；`AlexNet` 始终切在 `features_2`，gain 由 `53.87%` 增长到 `72.48%`；`GoogLeNet`、`DenseNet121` 和 `ResNet101` 的切分点分别稳定在 `maxpool1`、`features_pool0` 和 `maxpool`，gain 随本地算力增加持续上升。

这一结果说明，本地算力增强并不会自动消除中间切分的价值。相反，在“本地先执行一个能显著缩减特征规模的子图、再把后缀重计算交给边缘”的模式下，本地算力越高，这一前缀执行所需付出的额外代价越低，中间切分反而会在一段区间内变得更有吸引力。

### 6.3 模块一：三联柱状图

[single_pair_model_bars.png](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_model_bars.png) 与 [single_pair_bar_metrics.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_bar_metrics.csv) 从 `latency`、`energy` 和 `objective cost` 三个维度比较标准单对点。

在标准场景下，平均 pairwise cost 为：

- `Only-Local`: `13.0189`
- `Only-Server`: `1.5208`
- `MinCut`: `1.0161`

因此，`MinCut` 相对 `Only-Local` 下降 `92.20%`，相对 `Only-Server` 下降 `33.19%`。分模型看，`VGG19` 的 `MinCut` 与 `Only-Server` 完全重合，说明对重链式模型而言全边缘执行已是最优；而 `GoogLeNet`、`DenseNet121` 和 `ResNet101` 的 `MinCut` 都同时压低了 `latency` 与 `energy`。例如：

- `GoogLeNet`：`Only-Server = 1.2347`，`MinCut = 0.7006`
- `DenseNet121`：`Only-Server = 1.3121`，`MinCut = 0.7787`
- `ResNet101`：`Only-Server = 1.5962`，`MinCut = 1.0628`

当前标准单对场景下，`MinCut` 的切分标签为：

- `AlexNet -> features_2`
- `VGG19 -> server-only`
- `GoogLeNet -> maxpool1`
- `DenseNet121 -> features_pool0`
- `ResNet101 -> maxpool`

这说明在“输入不压缩、仅中间特征压缩”的通信口径下，复杂 DAG/残差模型已经能够在标准场景里出现真实中间切分。进一步从三联柱状图看，`MinCut` 的收益来源并不完全一致：`AlexNet` 更依赖上传量下降带来的通信收益，而 `GoogLeNet`、`DenseNet121` 和 `ResNet101` 同时在 `latency` 与 `energy` 两个维度上获益，说明它们的最优切分本质上是计算与通信的联合平衡。

### 6.4 模块一：gain 热力图

[single_pair_gain_heatmaps.png](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_gain_heatmaps.png) 与 [single_pair_gain_heatmap.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/single_pair_gain_heatmap.csv) 将 `MinCut` 相对更优端点方案的归一化收益映射到“带宽 × 本地算力”平面。

热力图揭示了更完整的有效区域：

- `AlexNet` 的最大 gain 为 `0.8010`
- `GoogLeNet` 的最大 gain 为 `0.5762`
- `DenseNet121` 的最大 gain 为 `0.6096`
- `ResNet101` 的最大 gain 为 `0.6389`
- `VGG19` 的最大 gain 仅为 `0.0848`

其中，`VGG19` 的热力图几乎整体贴近零值区域，而 `GoogLeNet`、`DenseNet121` 和 `ResNet101` 在中等带宽与中高本地算力区域呈现连续高增益带。`AlexNet` 则在中等带宽附近出现峰值区，说明小模型的最优切分更容易受到通信条件变化影响。

与只报告单个标准点相比，热力图直接给出了 `MinCut` 的有效区域边界：哪些模型在较宽参数区间内稳定受益，哪些模型仅在局部条件下出现收益，哪些模型几乎总是退化为端点策略。就本文而言，这组图比单一均值更能说明 `MinCut` 的适用范围。

### 6.5 模块一：GoogLeNet 分支压力场景

在极低带宽的 GoogLeNet 压力场景中：

- `Only-Local`: `9.0224`
- `Only-Server`: `23.5706`
- `ChainTopo`: `8.9252`
- `GreedyCut`: `8.9252`
- `MinCut`: `8.4123`

因此：

- `MinCut` 相对 `ChainTopo`/`GreedyCut` 下降 `5.75%`
- 相对 `Only-Local` 下降 `6.76%`

更关键的是，`MinCut` 选择的切分不再是前缀单点，而是四个 Inception 分支节点组成的非前缀多节点割：

```text
inception4a_branch1_conv,
inception4a_branch2_0_conv,
inception4a_branch3_0_conv,
inception4a_branch4_1_conv
```

这组结果是当前系统中最直接的拓扑表达能力证据。前述带宽折线图、本地算力折线图和热力图说明 `MinCut` 在数值上具备收益，而该压力场景进一步说明，这种收益确实来自 DAG 图结构的表达能力，而不仅仅来自目标函数重新加权。

### 6.6 模块二：统一五模型多用户主结果

统一五模型混合场景 `mixed5` 的主结果如下：

| 方法 | total cost | total latency | total energy |
| --- | ---: | ---: | ---: |
| Only-Local | 108.7339 | 35.6403 | 73.0936 |
| Only-Server | 7.7255 | 4.8545 | 2.8711 |
| MinCut+BMatch | 5.5616 | 3.5217 | 2.0399 |
| MinCut+RMatch | 6.1054 +/- 0.3213 | 4.0655 | 2.0399 |
| Rpartition+RMatch | 66.5244 +/- 22.1488 | 23.2297 | 43.2947 |

对应收益为：

- `MinCut+BMatch` 相对 `Only-Local` 降低 `94.89%`
- 相对 `Only-Server` 降低 `28.01%`
- 相对 `MinCut+RMatch` 降低 `8.91%`
- 相对 `Rpartition+RMatch` 降低 `91.64%`

这组结果说明：固定 MD-ES 对上的精确分区建模能够优于纯本地和纯边缘执行；在 pairwise 代价已足够准确的前提下，系统级匹配仍然带来稳定额外收益。尤其是 `MinCut+BMatch` 相对 `MinCut+RMatch` 仍能降低 `8.91%`，说明模块二并不是模块一之后的简单附属步骤，而是系统级收益不可缺少的一部分。

### 6.7 带宽与能耗权重敏感性

在统一五模型混合场景中，`MinCut+BMatch` 随上行带宽提升呈现明显下降趋势：

- `0.1 MB/s`：`25.8757`
- `0.2 MB/s`：`15.1050`
- `0.5 MB/s`：`7.9474`
- `1.0 MB/s`：`5.5616`
- `5.0 MB/s`：`2.9974`

作为对比，`Only-Server` 在低带宽下明显更差：

- `0.1 MB/s`：`59.4052`
- `0.2 MB/s`：`30.6943`
- `1.0 MB/s`：`7.7255`
- `5.0 MB/s`：`3.1318`

这说明在“输入不压缩、仅中间特征压缩”的口径下，分区策略对上行带宽变化仍然保持较强适应性。

`beta` 从 `0` 扫到 `4` 时，`MinCut+BMatch` 的代价曲线保持平滑：

- `beta=0`：`3.5217`
- `beta=0.5`：`4.5416`
- `beta=1`：`5.5616`
- `beta=2`：`7.6014`
- `beta=4`：`11.6812`

这说明当前框架在不同 latency-energy 权重下具有一致的建模行为。换言之，系统不会因为目标函数偏向时延或偏向能耗就失去可解释性，而是能够通过同一套分区与匹配框架平滑响应权重变化。

### 6.8 在线扩展：时隙级重优化

[online_summary.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/online_summary.csv) 给出的结果如下。

时变带宽场景：

- `Online MinCut+BMatch` 相对 `Static MinCut+BMatch` 下降 `20.08%`
- 相对 `Online Only-Server` 下降 `40.80%`
- 平均等待时间由 `0.8644 s` 降至 `0.4367 s`

突发负载场景：

- `Online MinCut+BMatch` 相对 `Static MinCut+BMatch` 下降 `20.93%`
- 相对 `Online Only-Server` 下降 `18.69%`
- 平均等待时间由 `0.9824 s` 降至 `0.3826 s`

复合非平稳场景：

- `Online MinCut+BMatch` 平均成本 `8.9200`
- `Static MinCut+BMatch` 平均成本 `10.2206`
- `Online Only-Server` 平均成本 `12.5794`

因此，在线方法分别相对静态策略和纯边缘在线策略下降 `12.73%` 和 `29.09%`，平均等待时间由 `0.9203 s` 降至 `0.5052 s`。

这些结果说明，当前在线扩展可以被准确描述为：在每个时间槽基于当前观测状态，重复执行“pairwise Min-Cut + Hungarian”的滚动重优化框架。更重要的是，在线收益不仅表现为目标值下降，还表现为等待时间和积压水平的同步收缩，这说明在线模块确实改善了动态环境下的资源竞争状态，而不是只在静态目标函数上做局部修补。

### 6.9 运行时间与可扩展性

[algorithm_runtime.csv](D:/MyProgram/DAG-Split/outputs_mixed5_ratio4_midonly/algorithm_runtime.csv) 显示，复杂模型下 `MinCut` 的求解开销仍明显低于 `ChainTopo` 与 `GreedyCut`：

- AlexNet：`0.70 ms` vs `0.65 ms` vs `0.74 ms`
- VGG19：`1.11 ms` vs `2.43 ms` vs `2.92 ms`
- GoogLeNet：`7.71 ms` vs `43.85 ms` vs `115.89 ms`
- DenseNet121：`28.02 ms` vs `311.03 ms` vs `349.37 ms`
- ResNet101：`53.26 ms` vs `126.04 ms` vs `158.77 ms`

在 `50 MD x 50 slot` 的可扩展性实验中：

- 代价矩阵构造：`32747.86 ms`
- Hungarian 匹配：`1.40 ms`

因此当前系统的工程瓶颈明确在 pairwise 代价矩阵构造，而不在匹配阶段。这一观察也给出后续工程优化方向：如果需要进一步提升在线吞吐或扩大用户规模，应优先考虑图结构复用、pairwise 代价缓存和更高效的最小割实现，而不是首先替换匹配算法。

## 7. 讨论与局限

本文仍然是仿真驱动的算法论文，而非真实 MEC 硬件部署论文。需要明确以下边界：

1. 当前“全局最优匹配”只在独立 `server_slot` 抽象下成立。
2. 当前在线扩展是观测驱动的时隙级重优化，不主张对未知未来轨迹的全局最优，也不提供 regret、competitive ratio 或 Lyapunov 保证。
3. 当前通信模型只采用选择性固定压缩率，不建模压缩时间、解压时间和压缩功耗，因此所有结论都针对该简化通信模型成立。
4. 当前版本的 DAG 优势既体现在统一建模能力上，也在 GoogLeNet 压力场景中体现为真实的非前缀多节点数值收益，但不应夸大为“对所有场景都严格优于链式基线”。

## 8. 结论

本文将多用户 DNN 卸载问题组织为两个静态模块加一个在线扩展：固定 MD-ES 对的 DAG 最小割分区、多用户全局匹配，以及时隙级在线重优化。模块一的带宽折线图、本地算力折线图、三联柱状图和 gain 热力图共同表明，`MinCut` 的收益既依赖模型拓扑，也依赖通信与本地资源条件；在标准单对场景中，`GoogLeNet`、`DenseNet121` 和 `ResNet101` 已出现稳定的真实中间切分，而 GoogLeNet 分支压力场景进一步证明了非前缀多节点割的结构优势。模块二说明，当逐对分区代价足够准确时，系统级匹配仍可进一步降低总成本。在线扩展进一步说明，同一 `pairwise + matching` 分解可以继续用于非平稳带宽与排队环境。整体上，当前版本形成了一条完整且自洽的论文主线：精确单对分区、清晰全局匹配，以及可验证的在线自适应扩展。

## 参考文献占位

1. 多用户 MEC 环境下的协同 DNN 分区与任务卸载研究。
2. Neurosurgeon.
3. Dynamic adaptive DNN partitioning / DNN surgery.
4. Collaborative intelligence / Edgent.
5. DAG 神经网络分区与图优化相关研究。
