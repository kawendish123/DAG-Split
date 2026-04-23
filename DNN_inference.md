# Dynamic Adaptive DNN Surgery for Inference Acceleration on the Edge

Chuang $\mathrm { H u ^ { * } }$ , Wei $\mathrm { B a o ^ { \dagger } }$ , Dan Wang∗, Fengming Liu‡ ∗Department of Computing, The Hong Kong Polytechnic University †School of Computer Science, The University of Sydney 

‡Department of Electronic and Information Engineering, The Hong Kong Polytechnic University 

Abstract—Recent advances in deep neural networks (DNNs) have substantially improved the accuracy and speed of a variety of intelligent applications. Nevertheless, one obstacle is that DNN inference imposes heavy computation burden to end devices, but offloading inference tasks to the cloud causes transmission of a large volume of data. Motivated by the fact that the data size of some intermediate DNN layers is significantly smaller than that of raw input data, we design the DNN surgery, which allows partitioned DNN processed at both the edge and cloud while limiting the data transmission. The challenge is twofold: (1) Network dynamics substantially influence the performance of DNN partition, and (2) State-of-the-art DNNs are characterized by a directed acyclic graph (DAG) rather than a chain so that partition is greatly complicated. In order to solve the issues, we design a Dynamic Adaptive DNN Surgery (DADS) scheme, which optimally partitions the DNN under different network condition. Under the lightly loaded condition, DNN Surgery Light (DSL) is developed, which minimizes the overall delay to process one frame. The minimization problem is equivalent to a min-cut problem so that a globally optimal solution is derived. In the heavily loaded condition, DNN Surgery Heavy (DSH) is developed, with the objective to maximize throughput. However, the problem is NP-hard so that DSH resorts an approximation method to achieve an approximation ratio of 3. Real-world prototype based on selfdriving car video dataset is implemented, showing that compared with executing entire the DNN on the edge and cloud, DADS can improve latency up to 6.45 and 8.08 times respectively, and improve throughput up to 8.31 and 14.01 times respectively. 

# I. INTRODUCTION

Recent advances in deep neural networks (DNN) have substantially improve the accuracy and speed of computer vision and video analytics, which creates new avenues for a new generation of smart applications. The maturity of cloud computing, equipped with powerful hardware such as TPU and GPU, becomes a typical choice for such kind computation intensive DNN tasks. For example, in a self-driving car application, cameras continuously monitor and stream surrounding scene to servers, which then conduct video analytic and feed back control signals to pedals and steering wheels. In an augmented reality application, a smart glass continuously records its current view and streams the information to the cloud servers, while the cloud servers perform object recognition and send back contextual augmentation labels, to be seamlessly displayed overlaying the actual scenery. 

One obstacle to realizing smart applications is the large amount of data volume of video streaming. For example, Google’s self-driving car can generate up to 750 megabytes of sensor data per second [1], but the average uplink rate of 4G, fastest existing solution, is only 5.85Mbps [2]. The data rate is substantially decreased when the user is fast moving 

or the network is heavily loaded. In order to avoid the effect of network and put the computing at the proximity of data source, edge computing emerges. As a network-free approach, it provides anywhere and anytime available computing resources. For example, AWS DeepLens camera can run deep convolutional neural networks (CNNs) to analyze visual imagery [3]. Nevertheless, edge computer themselves are limited by their computing capacity and energy constraints, which cannot fully replace cloud computing. 

From Fig. 1, we observe that, for the DNN, the amount of some intermediate results (the output of intermediate layers) are significantly smaller than that of raw input data. For example, the input data size of tiny YOLOv2 [4] is 0.95MB, while the output data size of intermediate layer max5 is 0.08MB with a reduction of $93 \%$ . This provides the opportunity for us to take the advantages of the powerful computation capacity of the cloud computing and the proximity of the edge computing. More specifically, we can compute a part of DNN on the edge side, transfer a small number of intermediate results to the cloud, and compute the left part on the cloud side. The partition of DNN constitutes a tradeoff between computation and transmission. As shown in Fig. 2, partition at different layers will cause different computation time and transmission time. So, an optimal partition is desirable. 

Unfortunately, the decision on how to split the DNN layers heavily depends on the network conditions. In a LTE network, the throughput can decrease by 10.33 times during peak hours [5], and this value could reach 18.65 for a WiFi hotspot [6]. Under a high-throughput network condition, computing delay dominates and it is more desirable to offload the DNNs as early as possible. However, if the network condition degrades severely, we should prudently determine the DNN cut so to decrease the volume of data transmission. For example, Fig. 3 shows that when the network capacity is as high as 18Mbps, the optimal cut is at input layer and the overall processing delay is 0.59s. However, when the network capacity is lowered to 4Mbps, cutting at input layer is no longer valid as the communication delay increases substantially. Under this scenario, cutting at max5 is optimal, with a delay reduction of $62 \%$ . 

Another challenge in the partition is that the recent advances of DNN show that DNNs are no longer limited to a chain topology, DAG topologies gain popularity. For example, GoogleNet [7] and ResNet [8], the champion of ImageNet Challenge 2014 and 2015 respectively, are DAGs. Obviously, partitioning DAG instead of chain involves much more complicated graph theoretic analysis, which may lead to NP-hardness 

![](images/0f236cfc29b335421d324fde1bf723aca522610dc39ca60b3f6ed14defa4f910.jpg)



Fig. 1: The output data size of each layer of YOLOv2.


![](images/a3d6bec18fce320b5ec0bcae022bb9cb8462a924afdb295115613590bff812e1.jpg)



Fig. 2: Latency constitution when partition at the different layers of tiny YOLOv2. Bandwidth is 4Mbps.


![](images/48ff15ab338f27ba2792a6371e61c4f5ef6fe8ad40ceabdef990cffc7b1a7afb.jpg)



Fig. 3: The latency of partition at different layers of YOLOv2 as a function of bandwidth.


in performance optimization. 

To this end, in this paper, we investigate the DNN partition problem, in order to find the optimal DNN partitioning in an integrated edge and cloud computing environment with dynamic network conditions. We design a Dynamic Adaptive DNN Surgery (DADS) scheme, which optimally partitions the DNN network by continually monitoring the network condition. The key design of DADS is as follows. DADS keeps monitoring the network condition and determines if the system is operated in the lightly loaded condition or heavily loaded condition. Under the lightly loaded condition, DNN Surgery Light (DSL) is developed, which minimizes the overall delay to process one frame. In this part, in order to solve the delay minimization problem, we convert the original problem to an equivalent mincut problem so that the globally optimal solution can be found. In the heavily loaded condition, DNN Surgery Heavy (DSH) is developed, which maximizes the throughput, i.e. the number of frames can be handled per unit time. However, we prove such optimization problem is NP-hard, which cannot be solved within polynomial computational complexity. DSH resorts an approximation approach, which achieves an approximation ratio of 3. 

Finally, we develop a real-world testbed to validate our proposed DADS scheme. The testbed is based on the selfdriving car video dataset and real traces of wireless network. We test 5 DNN models. We observe that compared with executing entire DNNs on the cloud and on the edge, DADS can reduce execution latency up to 6.45 times and 8.08 times respectively, and improve throughput up to 8.31 times and 14.01 times respectively. 

# II. AN EDGE-CLOUD DNN INFERENCE (ECDI) MODEL

# A. Background

Video analytics is the core to realize a wide range of exciting applications ranging from surveillance and self-driving cars, to personal digital assistants and automatic drone controls. The current state-of-the-art approach is to use a deep neural network (DNN) where the video frames are processed by a well-trained constitutional neural network (CNN) or recurrent neural network (RNN). Video analytics use DNNs to extract features from input frames of the video and classify the objects in the frames into one of the predefined classes. 

DNN network consists of quite a few layers which can be organized in a directed acyclic graph (DAG). Fig. 4 shows a 7-layer DNN model. Inference for video is performed with a DNN using a feed-forward algorithm that operates on each 

frame separately. The algorithm begins at the input layer and progressively moves forward layer by layer. Each layer receives the output of prior layers as the input, performs a series of computation on the input data to get the output, and feeds its output to the successor layers. This process terminates once the computation of output layer is finished. 

The video is generated at the edge side and the frames of the video are fed into the DNN as input. The computation of each layer in DNN can be performed at the edge or at the cloud. Computing layers at edge devices does not require to transmit data to the cloud but incurs more computation due to resourceconstrained device. Computing layers at the cloud leads to less computation but incurs transmission latency for transmitting data from edge devices to the cloud. 

# B. The ECDI Model

In this subsection. We formally present the ECDI model. 

1) Video Frame: A video consists of a sequence of frames (pictures) to be processed, with a sampling rate $Q$ frames/second. Each sampled frame is fed to a predetermined DNN for inference. Please note that the sampling rate is not the frame rate of the video. It indicates how many frames/pictures are processed each unit time [9]. 

2) DNN as a Graph: A DNN is modeled as a directed acyclic graph (DAG). Each vertex represents one layer of the neural network. A layer is indivisible and must be processed on either the edge side or the cloud side. We add an virtual entry vertex and an exit vertex to represent the starting point and the ending point of DNN respectively. The links1 represent communication and dependency among layers. 

Let $\mathcal { G } = ( \mathcal { V } \bigcup \{ e , c \} , \mathcal { L } )$ denote the DAG of DNN, where $\nu =$ $\{ v _ { 1 } , v _ { 2 } , \cdots , v _ { n } \}$ is the set of vertices representing the layers of the DNN (specially, $v _ { 1 }$ and $v _ { n }$ represent the input layer and output layer respectively). e and $c$ denote virtual entry and exit vertices (to facilitate the subsequent analysis). $\mathcal { L }$ is the set of links. A link $( v _ { i } , v _ { j } ) \in { \mathcal { L } }$ represents that $v _ { i }$ has to be processed before $v _ { j }$ , and $v _ { i }$ feeds its output to $v _ { j }$ . Fig. 6 shows the DAG of the pure inception v4 network [10] in Fig. 5. 

Since each layer can be processed on either the edge or cloud side, its processing time depends on where it is processed (i.e. on the edge or on the cloud). Let $t _ { i } ^ { e }$ and $t _ { i } ^ { c }$ be the time needed to process $v _ { i }$ one edge and cloud respectively. Let $d _ { i }$ and $t _ { i } ^ { t }$ denote the output data size and the transmission time of $v _ { i }$ . We 

1Please note that to avoid misunderstanding, throughout this paper, we use the term “link” to represent “edge of a graph.” This is because “edge” in this paper has already represented “edge computing.” 

![](images/4d682aa8ff87efdd22267aa54aea75801a3951272f5044287a01821974a40137.jpg)



Fig. 4: A 7-layer DNN model classifies frames of video.


![](images/47aaf8b4df0be65ee673b31e64105e0a4b2171f262d1fcada3b9320a9b0cc613.jpg)



Fig. 5: The inception v4 network represented in layer form.


![](images/f5645f71ffb98550b2c7231762a3a0265467690a60efe648edeac5b59679119e.jpg)



Fig. 6: Graph representation of inception v4 network.


define $\mathbf { D } _ { t } = \{ d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { n } \}$ . Let $B$ be the network bandwidth, we have $\begin{array} { r l r } { t _ { i } ^ { t } } & { { } = } & { \frac { d _ { i } } { B } } \end{array}$ . Please note that $B$ can be dynamically changed and we need to adapt such changes. We define ${ \bf F } _ { e } = { \bf \Lambda }$ $\{ t _ { 1 } ^ { e } , t _ { 2 } ^ { e } , \cdots , t _ { n } ^ { e } \}$ , $\mathbf { F } _ { c } = \{ t _ { 1 } ^ { c } , t _ { 2 } ^ { c } , \cdot \cdot \cdot , t _ { n } ^ { c } \}$ , $\mathbf { F } _ { t } = \{ t _ { 1 } ^ { t } , t _ { 2 } ^ { t } , \cdot \cdot \cdot , t _ { n } ^ { t } \}$ . They denote the three key delays: processing delay at the edge, transmission delay, and processing delay at the cloud of each layer. 

3) DNN Partitioning: Our objective is to partition DNN into two parts so the one part is processed at the edge and the other is processed at the cloud. Mathematically, we should find a set of vertices $\nu _ { S }$ as a subset of $\nu$ such that removing $\nu _ { S }$ causes that the rest of $\mathcal { G }$ becomes two disconnected components. One component contains e, denoted by $\gamma _ { E } ^ { \prime }$ and the other component contains $c$ , denoted by $\nu _ { C }$ . $\nu _ { S }$ is the cut so that all downstreaming layers are processed at the cloud. $\mathcal { V } _ { E } ^ { \prime }$ and $\nu _ { S }$ are processed at the edge and $\nu _ { C }$ are processed at the cloud. We define $\mathcal { V } _ { E } = \mathcal { V } _ { E } ^ { \prime } \bigcup \mathcal { V } _ { S }$ . The output data of vertices in $\nu _ { S }$ will be transmitted from the edge side to the cloud. $\nu _ { E }$ , including $\mathcal { V } _ { E } ^ { \prime }$ and $\nu _ { S }$ will generate processing delay at the edge. $\nu _ { S }$ will generate transmission delay. $\nu _ { C }$ will generate processing delay at the cloud. Our aim is to determine best cut $\nu _ { S }$ so that the overall delay is minimized. 

As shown in Fig. 6, we cut at $\mathcal { V } _ { S } = \{ v _ { 3 } , v _ { 5 } , v _ { 9 } , v _ { 1 2 } \}$ so that the $\mathcal { V } _ { E } ^ { \prime } = \{ e , v _ { 1 } , v _ { 2 } , v _ { 4 } \}$ , ${ \mathcal V } _ { E } = \{ e , v _ { 1 } , v _ { 2 } , v _ { 3 } , v _ { 4 } , v _ { 5 } , v _ { 9 } , v _ { 1 2 } \} { } _ { }$ , and ${ { \mathcal V } _ { C } } = \left\{ { { v } _ { 6 } } , { { v } _ { 7 } } , { { v } _ { 8 } } , { { v } _ { 1 0 } } , { { v } _ { 1 1 } } , { { v } _ { 1 3 } } , { { c } } \right\}$ . The overall delay is the processing delay of $\nu _ { E }$ on the edge and $\nu _ { C }$ on the cloud plus the communication delay of the output data of layer in $\nu _ { S }$ . 

4) Delay Components: Once the partition is made, each frame is processed at the edge, and then sent from the edge to the cloud, and then processed at the cloud. Since there are multiple frames to be processed, we assume that the three stages are conducted in pipeline. In order words, when frame 1 is being processed at the cloud, frame 2 can be transmitted and frame 3 can be processed at the edge. 

The delays of the three stages are characterized as follows. In the edge-computing stage 

$$
T _ {e} = \sum_ {v _ {i} \in \mathcal {V} _ {E}} t _ {i} ^ {e}. \tag {1}
$$

In the cloud-computing stage 

$$
T _ {c} = \sum_ {v _ {i} \in \mathcal {V} _ {C}} t _ {i} ^ {c}. \tag {2}
$$

In the communication stage 

$$
T _ {t} = \sum_ {v _ {i} \in \mathcal {V} _ {S}} t _ {i} ^ {t}. \tag {3}
$$

For each frame, $T _ { e }$ , $T _ { c }$ , and $T _ { t }$ are spent for each stage. Frames are processed in pipeline every $\textstyle { \frac { 1 } { Q } }$ . As a consequence, the Gantt chart (scheduling chart) of frames can be shown in Fig. 8. $T _ { e }$ , $T _ { c }$ , and $T _ { l }$ cannot exceed $\textstyle { \frac { 1 } { Q } }$ . Otherwise, the incoming rate is greater than the completion rate, leading to system congestion. Our aim is to smartly partition the DNN so that the overall delay to process frames is minimized and the system is not congested. 

# C. Parameter Estimation for ECDI

In this subsection, we discuss how to derive the input parameters. The first class of parameters is called DNN profile, including DNN topology $\mathcal { G }$ , processing delays of each layer at the edge and the cloud $\mathbf { F } _ { e }$ , $\mathbf { F } _ { c }$ , data size of each layer $\mathbf { D } _ { t }$ . These parameters can be well derived in advance. $\mathcal { G }$ and $\mathbf { D } _ { t }$ can be directly derived given the DNN definition. $\mathbf { F } _ { e }$ and $\mathbf { F } _ { c }$ can be measured beforehand. For example, we derive $\mathbf { D } _ { t }$ of tiny YOLOv2 model and measure $\mathbf { F } _ { e }$ of tiny YOLOv2 model processed on Raspberry Pi 3 model B and Ali Cloud respectively. We show the results in Fig. 1 and Fig. 7 respectively. 

The value $B$ is dynamic and should be measured during the process of DNN inference. This can be realized by a method similar to HTTP DASH [11]. We use the tool “ping” at edge to send two different size data consecutively to the cloud, and measure the response times. The bandwidth equals to the ratio between the difference of data size and the difference of response times. 

The value $Q$ is user-specific. The user lets the system know $Q$ when the inference starts. The system does nothing unless $Q$ is too large for the system to handle (See Section III-D). 

# III. ECDI PARTITIONING OPTIMIZATION

# A. The Impact of DNN Inference Workloads

Our first objective is to minimize the overall delay to process each frame. This is true under the light workload: for each stage, the current frame is completed before the next frame arrives. Mathematically $\begin{array} { r } { \operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \} < \frac { 1 } { Q } } \end{array}$ so that the Gantt chart is shown as the bottom one of Fig. 8. In this case, we just need to complete every frame as soon as possible, i.e., minimize $T _ { c } + T _ { t } + T _ { e }$ . 

However, if the system is heavily loaded, minimizing $T _ { e } + T _ { t } +$ $T _ { c }$ may lead to system congestion as $\begin{array} { r } { \operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \} \ge \frac { 1 } { Q } } \end{array}$ . For example, in Fig. 8 (top), $\begin{array} { r } { T _ { e } > \frac { 1 } { Q } } \end{array}$ so that the next frame arrives before the current frame is completed at the edge. Therefore, under this situation, we need to maximize the throughput of the system, i.e. how many frames at most the system can handle 

![](images/c4cf6d60c4488cab28e68fe5e3406c8e96314b410195a26a3e151b7d7887ca90.jpg)



Fig. 7: The computation latency of YOLOv2’s layers on the edge (top) and cloud (bottom) respectively.


![](images/e99130b566313cfec42b03921dee6bf0ba85fe867afe35c965c7857d45434743.jpg)



Fig. 8: Gantt charts for three stages.


![](images/9a8661d215dc1c250eded6ae4871cab2e8c742b8116f685450a2a0052c688646.jpg)



Fig. 9: Illustration of conversion to the minimum s-t cut problem.


per unit time. Our objective is to minimize $\operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \}$ as the system throughput is 1max T ,T ,T . For presentation $\frac { 1 } { \operatorname* { m a x } \{ { T _ { e } } , { T _ { t } } , { T _ { c } } \} }$ convenience, $\operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \}$ is referred to as the max stage time. 

Please note that in Section III-D, we will further discuss how to judge if the system is lightly loaded or heavily loaded. There, we also need to consider that if the sampling rate is greater than 1min max{Te,Tt,Tc} so that the system will be congested $\frac { 1 } { \operatorname* { m i n } \operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \} }$ eventually. The system has to force the sender/user to reduce sampling rate. 

# B. The Light Workload Partitioning Algorithm

In this subsection, we study Edge Cloud DNN Inference for Light Workload (ECDI-L) problem. Our goal is to minimize the overall delay of one frame, under a given the network condition $B$ . In summary, we have the following optimization problem: 

Problem 1. (ECDI-L) Given , $\left[ \mathbf { F } _ { e } , \mathbf { F } _ { c } , \mathbf { D } _ { t } \right]$ , and $B$ , determine $\nu _ { E }$ , $\nu _ { S }$ and $\nu _ { C }$ , to minimize $T _ { i n f } = T _ { e } + T _ { t } + T _ { c }$ . 

Proposition 1. Problem ECDI-L can be solved in polynomial time. 

One challenge to solve ECDI-L problem directly is that each vertex in $\mathcal { G }$ contains three delay values $\begin{array} { r } { t _ { i } ^ { e } , t _ { i } ^ { c } , t _ { i } ^ { t } = \frac { { { \dot { d } } } _ { i } } { B } } \end{array}$ . The delay that contributes to the overall delay depends on where the vertex is processed. To this end, we construct a new graph $\mathcal { G } ^ { \prime }$ so that each edge only captures a single delay value. By doing so, we convert ECDI-L problem to the minimum weighted s-t cut problem of $\mathcal { G } ^ { \prime }$ . 

We first illustrate how to construct $\mathcal { G } ^ { \prime }$ based on $\mathcal { G }$ . 

a) Cloud Computing Delay: Based on $\mathcal { G }$ , we add links between $e$ and each vertex $v \in \mathcal V$ , referred to as “red links,” to capture the cloud-computing delay of $v$ . 

b) Edge Computing Delay: Similarly, we add links between vertex $v \in \mathcal V$ and $c$ , referred to as “blue links,” to capture the edge-computing delay of $v$ . 

c) Communication Delay: All the other links correspond to communication delays. A link from $v$ to $u$ should capture the communication delay of $v$ . However, this is insufficient as one vertex may have multiple successors and its communication delay is counted multiple times. For example, $v _ { 1 }$ in Fig. 6 has 4 outgoing links but the communication delay of $v _ { 1 }$ has to be counted at most once. To this end, we introduce axillary vertices into graph $\mathcal { G } ^ { \prime }$ . That is, for any vertex $v _ { k } \in \mathcal V$ whose outdegree is greater than one, we add an auxiliary vertex $v _ { k } ^ { \prime }$ and link $( v _ { k } , v _ { k } ^ { \prime } )$ . The links from $v _ { k }$ to successors of $v _ { k }$ are now re-placed from 

$v _ { k } ^ { \prime }$ to successors of $v _ { k }$ . For example, a 4-layer DNN is shown in Fig. 9(a). The outdegree of vertex $v _ { 1 }$ is greater than one, we thus add an auxiliary vertex $v _ { 1 } ^ { \prime }$ and link $( v _ { 1 } , v _ { 1 } ^ { \prime } )$ shown in Fig. 9(b). The links $( v _ { 1 } , v _ { 2 } )$ and $( v _ { 1 } , v _ { 3 } )$ are re-placed by links $( v _ { 1 } ^ { \prime } , v _ { 2 } )$ and $( v _ { 1 } ^ { \prime } , v _ { 3 } )$ respectively. We define $\gamma _ { D }$ to be the set of axillary vertices. 

Now, without considering $e$ and $c$ , if a vertex $v$ has one successor, the link starting from $v$ corresponds to its communication delay, which is referred to as “black link.” If $v$ has multiple successors, then all the links starting from $v$ are referred to as “dashed links” and should not be considered since the communication delay has already been considered from $v$ to $v ^ { \prime }$ . 

Links are assigned costs. The costs assigned to red, blue, black links are cloud-computing, edge-computing, and communication delays. Dashed links are assigned infinity. 

$$
c \left(v _ {i}, v _ {j}\right) = \left\{ \begin{array}{l l} t _ {i} ^ {e}, & \text {i f} v _ {i} \in \mathcal {V}, v _ {j} = c. \\ t _ {i} ^ {t}, & \text {i f} v _ {i} \in \mathcal {V}, v _ {j} \in \mathcal {V} \bigcup \mathcal {V} _ {D}. \\ t _ {i} ^ {c}, & \text {i f} v _ {i} = e, v _ {j} \in \mathcal {V}. \\ + \infty , & \text {o t h e r s .} \end{array} \right. \tag {4}
$$

At this stage, we can convert ECDI-L problem to the minimum weighted s–t cut problem of $\mathcal { G } ^ { \prime }$ . 

A cut is a partition of the vertices of a DAG into two disjoint subsets. The s–t cut of $\mathcal { G } ^ { \prime }$ is a cut that requires source $s$ and sink $t$ to be in different subsets, and its cut-set only consists of links going from the source’s side to sink’s side. The value of a cut is defined as the sum of the cost of each link in the cut-set. Problem ECDI-L is equivalent to the minimum $e { - } c$ cut of $\mathcal { G } ^ { \prime }$ . If cutting on link from $e$ to $v _ { i } \in \mathcal V$ (red link shown in Fig. 9(b)), then $v _ { i }$ will be processed on the cloud, i.e $v _ { i } \in \mathcal V _ { C }$ . If cutting on link from $v _ { j } \in \mathcal V$ to $c$ (blue link show in Fig. 9(b)), then $v _ { j }$ will be processed on the edge, i.e. $v _ { j } \in \mathcal { V } _ { E }$ . If cutting on link from $v _ { i } \in \mathcal V$ to $v _ { j } \in \mathcal { V } \bigcup \mathcal { V } _ { D }$ (black link show in Fig. 9(b)), then the data of $v _ { i }$ will be transmitted to the cloud, i.e $v _ { i } \in \mathcal V _ { S }$ . It is impossible to cut on link from $v _ { i } \in \mathcal V _ { D }$ to $v _ { j } \in \mathcal V$ (dashed links), because otherwise it will lead to infinite cost (but finite cost exists). The total cost of cut on red links equals to cloud computation time $T _ { c }$ . The total cost of cut on blue links equals to edge computation time $T _ { e }$ . The total cost of cut on black links equals to transmission time without network latency $T _ { t }$ . If the $e { - } c$ cut of $\mathcal { G } ^ { \prime }$ is minimum, then the inference latency on a single frame is minimum. For example, in Fig 9(b), the cut is at $( e , v _ { 2 } )$ , $( e , v _ { 3 } )$ , $( e , v _ { 4 } )$ , $( v _ { 1 } , v _ { 1 } ^ { \prime } )$ and $( v _ { 1 } , c )$ . $v _ { 1 }$ is processed at the edge so that $t _ { 1 } ^ { e }$ is counted in the blue link. $v _ { 2 }$ , $v _ { 3 }$ and $v _ { 4 }$ are processed at the cloud so that $t _ { 2 } ^ { c } t _ { 3 } ^ { c }$ and $t _ { 4 } ^ { c }$ are counted in the 

red links. The communication delay $t _ { 1 } ^ { t }$ is counted in the black link. 

We develop DNN Surgery Light (denoted as DSL) algorithm for ECDI-L problem. The overall algorithm DSL() is shown in Algorithm 1. The algorithm first calls compute-net() to compute $\mathbf { F } _ { t }$ . Then it calls graph-construct() (line 2) to construct $\mathcal { G } ^ { \prime }$ based on $\mathcal { G }$ with the computation complexity of $O ( n + m )$ , where $n$ is the number of layers $| \nu |$ , $m$ is the number of links $| { \mathcal { L } } |$ , and then it calls min-cut() (line 3) to find minimum $e { - } c$ cut of $\mathcal { G } ^ { \prime }$ which outputs the partition strategy (i.e. $\nu _ { E } , \nu _ { S }$ and $\nu _ { C }$ ). Boykov’s algorithm [12] is used in min-cut() to solve the minimum e–c cut problem with the computational complexity of $\mathcal { O } ( ( m + n ) n ^ { 2 } )$ . DSL() is a polynomial-time algorithm with the computational complexity of $O ( ( m + n ) n ^ { 2 } )$ . 


Algorithm 1: DSL Algorithm DSL().


Input: $\mathcal{G},\mathbf{F}_e,\mathbf{F}_c\mathbf{D}_t,B$ Output: $\nu_{E},\nu_{S},\nu_{C},T_{e},T_{t},T_{c}$ 1 $\mathbf{F}_t\gets$ compute-net $(\mathbf{D}_t,B)$ .   
2 $\mathcal{G}^{\prime}\gets$ graph-construct $(\mathcal{G},\mathbf{F}_e,\mathbf{F}_c,\mathbf{F}_t)$ 3 $[\nu_{E},\nu_{S},\nu_{C},T_{e},T_{t},T_{c}]\gets$ min-cut $(\mathcal{G}^{\prime})$ 4 return $\nu_{E},\nu_{S},\nu_{C},T_{e},T_{t},T_{c}$ 

# C. The Heavy Workload Partitioning Algorithms

As discussed in Section III-A, we formulate the Edge Cloud DNN Inference for Heavy Workload (ECDI-H) problem, to minimize $\operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \}$ . The decision variables are $\scriptstyle \gamma _ { E }$ , $\nu _ { S }$ and $\nu _ { C }$ . In summary, we have the following optimization problem: 

Problem 2. (ECDI-H) Given $\mathcal { G }$ , $[ \mathbf { F } _ { e } , \mathbf { F } _ { c } , \mathbf { D } _ { t } ] _ { : }$ , and $B$ , determine $\nu _ { E }$ , $\nu _ { S }$ and $\nu _ { C }$ , to minimize $\operatorname* { m a x } \{ T _ { e } , T _ { t } , T _ { c } \}$ (i.e. maximize throughput). 

ECDI-H Problem is NP-hard. We provide the sketch of the proof. We prove it by reducing from the smallest component of the most balanced minimum st-vertex cut problem (MBMVC-SC), which is known to be NP-complete [13]. We consider the following MBMVC-SC problem on $\mathcal { G } _ { A } = ( \mathcal { V } _ { A } \bigcup \{ s , t \} , \mathcal { L } _ { A } )$ , the goal is to find a vertex cut set $\nu _ { C }$ to partition the graph into two disjoint components $( \nu _ { 1 } , \nu _ { 2 } )$ , for which $s \in \mathcal { V } _ { 1 }$ , $t \in \nu _ { 2 }$ and the largest components among $\{ \mathcal { V } _ { 1 } , \mathcal { V } _ { 2 } , \mathcal { V } _ { C } \}$ is minimum. Any instance of the above problem is equivalent to an instance in ECDI-H problem. Due to page limitation, detailed explanations are omitted. 

ECDI-H is NP-hard. It is unrealistic to find a globally optimal solution within polynomial time. We design DNN Surgery Heavy (denoted as DSH) algorithm which achieves a locally optimal solution. In addition, its approximation ratio is 3. 

The rationale to develop DSH is as follows. We modify $\mathcal { G } ^ { \prime }$ by changing the costs of links as follow: 

$$
c \left(v _ {i}, v _ {j}\right) = \left\{ \begin{array}{l l} \alpha t _ {i} ^ {e}, & \text {i f} \quad v _ {i} \in \mathcal {V}, v _ {j} = c. \\ \beta t _ {i} ^ {t}, & \text {i f} \quad v _ {i} \in \mathcal {V}, v _ {j} \in \mathcal {V} \bigcup \mathcal {V} _ {D}. \\ \gamma t _ {i} ^ {c}, & \text {i f} \quad v _ {i} = e, v _ {j} \in \mathcal {V}. \\ + \infty , & \text {o t h e r s .} \end{array} \right. \tag {5}
$$

Here $\alpha , \beta$ and $\gamma$ are non-negative variables. The approach is to run DSL() with several different $\alpha , \beta$ and $\gamma$ values. By this way, a solution is generated to optimize ECDI-L with a specific $\alpha , \beta , \gamma$ tuple. Then we test if this solution is also good enough for ECDI-H. If it is better than all existing solutions, it is regarded as a new solution to ECDI-H. We repeat the above procedure for a wide range of $\alpha , \beta , \gamma$ tuples. 

Here, the result of DSL() is determined by the ratio of the three parameters, instead of their absolute values. Therefore, we can fix one of the three, for example, $\beta = 1$ , and only vary the other two. Thus, we have a two-dimensional search space for $\alpha$ and $\gamma$ . We first search in the two-dimensional plane with a coarse granularity to find the best solution. Then we use a finer granularity search in the neighborhood of the best solution for further improvement. We repeat the steps until the improved performance is smaller than a threshold . 

The overall algorithm DSH() is shown in Algorithm 2. A function search() (line 11–19) is designed to search for the best solution in a given space $\mathbf { S } \triangleq [ \alpha _ { l } , \gamma _ { l } , \alpha _ { h } , \gamma _ { h } ]$ , meaning that $\alpha _ { l } \le \alpha \le \alpha _ { h } , \gamma _ { l } \le \gamma \le \gamma _ { h }$ , and a granularity $\delta$ (line 13–14), i.e. the step size of changing $\alpha$ and $\gamma$ is $\delta$ each time. For each $\alpha$ and $\gamma$ , search() calls DSL() to compute the vertex cut and calls max-time() to compute the $\operatorname* { m a x } [ T _ { c } , T _ { e } , T _ { t } ]$ . Lines 17– 18 guarantee $\operatorname* { m a x } [ T _ { c } , T _ { e } , T _ { t } ]$ derived is non-increasing. 

The overall algorithm first initializes the search granularity $\delta$ to be 1 (line 2) and the search space large enough (line 3–4). It calls search() (line 8) to search on the given space S with a granularity $\delta$ , and returns the best $\alpha$ and $\gamma$ found currently. Then DSH() narrows down the search space S (line 8) to the neighborhood of the best $\alpha$ and $\gamma$ for the current iteration, and adjusts $\delta$ to a finer granularity (line 9). Such space S and granularity $\delta$ is returned to search(). The termination condition for the loop is that the improved performance is smaller than a threshold $\epsilon$ (line 5). Finally, it returns the vertex cut with the best-found performance (line 10). Obviously, we can achieve a local optimal result with respect to the neighborhood of the final $\alpha$ and $\gamma$ . 

Theorem 1. The approximation ratio of the algorithm DSH for ECDI-H is 3. 

Proof. Let the max stage time of DHL be $t _ { D S H }$ . Let the optimal max stage time of ECDI-H be $t ^ { * }$ . We prove $\frac { t _ { D S H } } { t ^ { * } } \leq 3$ . Let $T ^ { * }$ denote the minimum inference latency for one frame. Let $T _ { o }$ denote the inference latency of a single frame when achieving the optimal max stage time. We have $T ^ { * } \leq T _ { o }$ . Because there are three stages, we have $T _ { o } \leq 3 t ^ { * }$ , thus $T ^ { * } \le 3 t _ { o }$ . 

As shown in Algorithm 2, when $\delta = 1$ , Search() will calls DSL() using $\alpha = 1$ and $\gamma = 1$ as the parameter. When $\alpha = 1$ and $\gamma = 1$ , DSL() achieves the minimum inference time $T ^ { * }$ for one frame. Let $t _ { 1 }$ , $t _ { 2 }$ and $t _ { 3 }$ be the edge computation time, the transmission time and the cloud computation time respectively when achieving the minimum inference time. We have $T ^ { * } =$ $t _ { 1 } + t _ { 2 } + t _ { 3 }$ . DSH() guarantees the searched max stage time is non-increasing, thus $t _ { m } ~ \le ~ \operatorname* { m a x } \{ t _ { 1 } , t _ { 2 } , t _ { 3 } \}$ , combined with $T ^ { * } = t _ { 1 } + t _ { 2 } + t _ { 3 }$ , we have $t _ { m } \leq T _ { m i n }$ . As $t _ { m } \leq T ^ { * }$ and $T ^ { * } \leq 3 t ^ { * }$ , we prove $\frac { t _ { D S H } } { t ^ { * } } \leq 3$ . □ 


Algorithm 2: DSH Algorithm DSH().


Input: $\mathcal{G},\mathbf{F}_e,\mathbf{F}_c,\mathbf{D}_t,B,\epsilon ,K$ Output: $\nu_{E},\nu_{S},\nu_{C},T_{max}$ 1 $\mathbf{F}_t\gets$ compute-net $(\mathbf{D}_t,B)$ .   
2 $T_{max}\gets +\infty ;T_{max}^{\prime}\gets 0;\delta \gets 1$ 3 $\alpha_l\gets 0;\gamma_l\gets 0;\alpha_u\gets \frac{\sum(\mathbf{F}_e)}{\min(\mathbf{F}_t)};\gamma_u\gets \frac{\sum(\mathbf{F}_c)}{\min(\mathbf{F}_t)};$ 4 S $\leftarrow [\alpha_l,\gamma_l,\alpha_u,\gamma_u]$ 5 while $|T_{max}^{\prime} - T_{max}|\geq \epsilon$ do   
6 $T_{max}^{\prime}\gets T_{max};$ 7 [α,γ,VE,Vc,Tmax]←Search (S,δ,Tmax);   
8 $\alpha_l\gets \alpha -\delta ;\alpha_u\gets \alpha +\delta ;\gamma_l\gets \gamma -\delta ;\gamma_u\gets \gamma +\delta ;$ 9 $\delta \gets \delta /K$ 10 return V,E,Vs,Vc,Tmax;   
11 function Search([αl,γl,αu,γu],δ,Tmax)   
12 $T_{max}\gets +\infty$ 13 for $\alpha \gets \alpha_l;\alpha \leq \alpha_u;\alpha \gets \alpha +\delta$ do   
14 for γ←γl;γ≤γu;γ←γ+δ do   
15 [VE,Vs,Vc,Te,Tt,Tc]←DSL (G,αFe,γFc,Dt,B)   
16 $T_{max}\gets$ max-time $(T_e,T_t,T_c)$ 17 if $T_{max}\leq T_{max}^{*}$ then   
18 $\begin{array}{r}\lfloor \alpha^{*}\gets \alpha ;\gamma^{*}\gets \gamma ;T_{max}^{*}\gets T_{max}; \end{array}$ 19 return α*,γ*,VE,VS,Vc,Tmax; 

# D. The Dynamic Partitioning Algorithm

We now consider network dynamics. In practice, the network status $B$ varies. This will affect the workload mode selection and the partition decision dynamically. We design Dynamic Adaptive DNN Surgery (DADS) scheme to adapt network dynamics. 

It is shown in Algorithm 3. monitor-task() monitors whether the video is active (line 2). This can be realized by tool “iperf.” Detailed implementation can be found in Section IV. The real-time network bandwidth is derived by monitor-net() (line 3). Then DSL() is called to compute the partition strategy (line 4). In this case, if it satisfies the sampling rate $\textstyle { \frac { 1 } { Q } }$ , i.e.ma $\begin{array} { r } { \mathrm { x } \{ T _ { e } , T _ { t } , T _ { c } ) < \frac { 1 } { Q } } \end{array}$ , we can confirm that the system is in the light workload mode and the partition by DSL is accepted. 

Otherwise, the system is in the heavy workload mode and calls DSH() to adjust the partition strategy to minimize the max delay (line 6). However, if the completing rate is still smaller than the sampling rate, it means that the sampling rate if too large so that even DSH() still cannot satisfy the sampling rate. The system will be congested. It calls the user to decreases the sampling rate (line 7–8). 

# IV. IMPLEMENTATION

We implement a DADS prototype system. We use the Raspberry Pi 3 model B as the edge device, integrated with a Logitech BRIO camera. We rent a server in Cloud Ali with 8 cores of $2 . 5 \ \mathrm { G H z }$ and a total memory of 128 GB. We employ WiFi as the communication link between the edge device and the cloud. The wired link from the edge router and the cloud is sufficiently large. We implement our client-server interface using GRPC, an open source flexible remote procedure call (RPC) interface for inter-process communication. 

The edge device. The duty of the edge device is to 1) extract video from the camera and to sample frames from video, 2) 


Algorithm 3: DADS Algorithm DADS()


1 while true do  
2 if monitor-task() == true then  
3 B ← monitor-net();  
4 [V_E, V_S, V_C, T_e, T_t, T_c] ← DSL(G, F_e, F_c, D_t, B);  
5 if max{Te, T_t, T_c} > 1/Q then  
6 [V_E, V_S, V_C, T_max] ← DSH(G, F_e, F_c, D_t, B, \epsilon, K);  
7 if T_max > 1/Q then  
8 inform-decrease();  
9 execute (V_E, V_S, V_C); 

make partition decision, 3) process the layers allocated to the edge device, and 4) inform the cloud the partition decision and transfer the intermediate results to the cloud. 

For video extraction, we extract videos from camera logitech BRIO using the provided API video_capture(). The camera transfers the captured video to Raspberry Pi through the USB-to-serial cable. 

For partition decision making, we implement a process that monitors the generated frame by the camera, and runs DADS scheme. DADS requires to estimate the real-time network bandwidth. We use the command “iperf” provided by the operation system Raspbian on Raspberry Pi. This command feeds back the real-time network bandwidth between the Raspberry Pi and the cloud. 

For processing allocated layers on the edge, we install a modified instance of Caffe and store a full DNN model on the edge device. The challenge is to control Caffe to stop execution at partitioned layers (e.g., $\nu _ { S }$ ). In Caffe, there is a “prototxt” file recording the DNN structure. Layers are processed according to this file. To solve the challenge, we modify the model structure file “prototxt” by inserting a “stop layer” after each partitioned layer. The instance of Caffe will stop processing at the desired places. 

For the intermediate results and partition decision transmission, the edge device calls the RPC function receiveRPC() provided by the cloud to transmit the data to the cloud. 

The cloud. The duty of the cloud is to execute the DNN layers allocated to the cloud. There are two jobs: 1) to receive the partition decision and the intermediate results from the edge device, and 2) to execute the layers allocated to the cloud. 

For the first job, we expose an API receiveRPC() to the edge device. After completing processing layers allocated to the edge, the edge device calls this RPC function to transmit the intermediate results packed with the partition decision to the cloud. 

For the second job, we implement a modified instance of Caffe and store a full DNN model. The challenge is to execute only the layers allocated to the cloud. To this end, after receiving the partition decision and intermediate results, the layers allocated to the edge are deleted before the marked place in “prototxt,” and the intermediate results are forwarded to the corresponding layers as input. By this way, only layers allocated to the cloud will be executed. 

# V. PERFORMANCE EVALUATION

We evaluate the DADS prototype (Section IV) using real-trace driven simulations. 

# A. Setup

Video Datasets. We employ the publicly available BDD100K self-driving dataset. The videos of this dataset are obtained from the camera on the self-driving car. Each video is about 40 seconds long and is viewed in 720p at 30 FPS. 

Workload Setting. We divide the inference task into low workload mode and heavy workload mode. Accordingly, We transform the video into different sampling rates to produce different workload. We set a low sampling rate to 0.1 frame per second when evaluating light workload mode, and 20 frames per second for heavy workload mode. The default resolution is 224p. Each inference task consists of processing 100 frames using the given DNN benchmarks. 

Communication Network Parameters. To model the communication between edge and cloud, we used the average uplink rate of mobile Internet for different wireless networks, i.e. CAT1, 3G, 4G and WiFi as shown in Table I. 

DNN Benchmarks. DADS can make partition not only on chain topology DNN but also on the DAG topology. We evaluate the performance of DADS for both topologies. For the chain topology, NiN, tiny YOlOv2 and VGG16, are well-known models used as benchmarks in this evaluation shown in Fig. 10. For the DAG topology, we employ AlexNet and ResNet-18 as the benchmarks shown in Fig. 11. 

Evaluation Criteria: We compare DADS against Edge-Only (i.e. executing the entire DNN on the edge), Cloud-Only (i.e. executing the entire DNN on the cloud), and a variant Neurosurgeon which is a partition strategy for chain-topology DNN. To evaluation Neurosurgeon’s performance for DAG, we consider a variant Neurosurgeon, which first employs topological sorting method to transform the DAG topology to the chain topology, and then uses the original partition method. We use the Edge-Only method as the baseline, i.e. the performance is normalized to Edge-Only method. 

We evaluate the latency and throughput of DADS compared with Edge-Only, Cloud-Only and Neurosurgeon in Section V-B. We also evaluate the impact of different types of wireless network to DADS, and the impact of bandwidth on the selection of workload mode in Section V-C. 

# B. Performance Comparison

We first compare our DADS with Edge-Only, Cloud-Only and Neurosurgeon under light workload mode and heavy workload mode across the 5 DNN benchmarks in Fig. 12–14. The results are normalized to Edge-Only method. We see that DADS achieves a higher latency speedup and throughput gain compared with other methods. 

Comparing DADS with Edge-Only and Cloud-Only: DADS has a latency speedup of 1.91–6.45 times, 1.35–8.08 times compared with Edge-Only and Cloud-Only methods respectively under the light workload mode shown in the bottom graph of Fig. 12. DADS has a throughput gain of 3.45–8.31 

times, 1.46–11.13 times compared with Edge-Only and Cloud-Only methods respectively under the light workload mode shown in the upper graph of Fig. 12. This is because, Edge-Only method executes the entire DNN on the edge side, it avoids data transmission and ignores the weak computation capacity of edge side. Cloud-Only method ignores the effect of the transmission time. DADS considers both computation and transmission, and it makes a good tradeoff between them. 

From Fig. 16, we can see that, for the heavy workload mode, DADS outperforms Edge-Only and Cloud-Only 1.66– 5.19 times and 1.07–6.92 times respectively in latency reduction, and DADS outperforms Edge-Only and Cloud-Only 4.34–9.14 times and 1.46–14.10 times respectively in throughput gain. This further confirms that DADS significantly outperforms Edge-Only and Cloud-Only methods. 

Comparing DADS with Neurosurgeon: Neurosurgeon can automatically partition DNN between the edge device and cloud at granularity of neural network layers, but it is only effective for chain topology. 

From Fig. 14, we can see that, for the chain topology models, DADS and Neurosurgeon have the similar performance in latency and throughput for the light workload. While for the heavy workload, Neurosurgeon has a latency reduction of $1 6 . 2 8 \%$ and $1 3 . 6 4 \%$ than that of DADS for YOLOv2 and VGG16, however the throughput gain of DADS is 1.26 times and 1.27 times than that of Neurosurgeon under these two DNN models. This is because, for the heavy workload, the higher throughput is prior for DADS. We also can see that, for the heavy workload and NiN model, the latency and the throughput of Neurosurgeon and DADS are both the same. This is because for NiN model, DADS achieves the minimum max stage time when the latency is minimum. 

For the DAG topology, we can observe that DADS outperforms Neurosurgeon significantly. For DAG topology models, DADS has a latency speedup $6 6 \% - 8 6 \%$ and throughput gain of $76 \% - 8 7 \%$ compared with Neurosurgeon. This observation validates the usefulness of DADS for DAG topology. 

# C. Network Variation

In this section, we evaluate how transmission network affects the performance of DADS using ResNet18 model. The sampling rate is 1 frame per second. 

The Impact of Transmission Network Type: We first evaluate the performance of DADS, Edge-Only and Cloud-Only for ResNet18 model when using Cat1, 3G, 4G and WiFi as the communication network. 

In Figs. 15–16, we show the latency speedup and the throughput gain achieved by DADS and Cloud-Only normalized to Edge-Only when using Cat1, 3G, 4G and WiFi for light and heavy workload respectively. 

Shown in Fig. 15, when the workload is light and the edge device communicates with the cloud through Cat1, DADS achieves 1.46 times latency reduction and 2.03 times throughput gain compared with Edge-Only. When the network changes to 3G, 4G and 5G the latency reduction and the throughput gain becomes more significant: 4.14 times and 8.3 times for 3G, 7.23 times and 9.78 times for 4G, 8.32 times and 9.31 times for 

![](images/b73ee27393871b4a1772488012b5529a0e6ccd18a7a1081ff67a2ae9a97835b0.jpg)



Fig. 10: The chain-topology DNN models.


![](images/eb40db0ffd5797f9b75271e7cf66a3e9e41fb63fdd7710214022649e635b1a0d.jpg)



Fig. 11: The DAG-topology DNN models.


![](images/2d481d2c0c0a45d42eaffdfd2da62e31495f886f5e4c0e5754b04b93e1f0b104.jpg)



Fig. 12: Latency speedup and throughput gain achieved by DADS under light workload mode.



TABLE I: DNN Benchmark Specifications


<table><tr><td></td><td>CAT1</td><td>3G</td><td>4G</td><td>WiFi</td></tr><tr><td>Uplink rate (Mpbs)</td><td>0.13</td><td>1.1</td><td>5.85</td><td>18.88</td></tr><tr><td colspan="5">TABLE II: DNN Benchmark Specifications</td></tr></table>

<table><tr><td>Type</td><td colspan="3">Chain</td><td colspan="2">DAG</td></tr><tr><td>Model</td><td>NiN</td><td>YOLOv2</td><td>VGG16</td><td>Alexnet</td><td>ResNet18</td></tr><tr><td>Layers</td><td>9</td><td>17</td><td>24</td><td>23</td><td>20</td></tr></table>

WiFi respectively. When the communication link provides more bandwidth, DADS pushes larger portions of layers to the cloud to achieve better performance. We can also see that, compared with Cloud-Only, DADS achieves latency reduction of $64 \%$ for CAT1, $26 \%$ for 3G and $7 \%$ for 4G receptively, and throughput gain of $73 \%$ for CAT1, $45 \%$ for 3G and $4 \%$ for 4G. For WiFi, the performance of Cloud-Only is good enough, it has the same performance with DADS. 

Edge-Only is only good for low data rate. Cloud-Only is only good for high data rate, DADS can be adaptive to a wide range of network setting. 

The Impact of Bandwidth on Workload Mode Selection: In Fig. 17, we show the workload mode switch of DADS under different network bandwidth. We can see that when the available bandwidth is smaller than 1.51Mbps, DADS works at heavy workload mode, and the achieved latency speedup and throughput gain increase compared with Edge-Only. When the bandwidth is greater than 1.51Mbps, DADS works at light workload mode. 

We also evaluate DADS’s resilience to real-world measured wireless network variations. In Fig. 18, the top graph shows measured wireless bandwidth over a period of time. The bottom graph shows the latency speedup of DADS normalized to Edge-Only for ResNet18 model. We can see that DADS adjusts the partition strategy according to the bandwidth variance successfully. For example, when the bandwidth drops from 3.41Mbps to 2.15Mbps, DADS changes the partition from conv2 layer to conv3 layer. DADS changes the partition from conv3 layer to conv7 layer when bandwidth is smaller than 1.72Mbps. 

# VI. RELATED WORK

Modification of DNN Models. In order to realize inference acceleration, one category of related work investigated how to modify DNN models for speedup. For example, Microsoft and Google developed small-scale DNNs for speech recognition on mobile platforms by sacrificing the high prediction 

accuracy [14]. MCDNN [15] proposed generating alternative DNN models to trade off accuracy and performance/energy and choosing to execute either in the cloud or on the mobile. [16] proposed deep models that are much smaller than normal and to be run on phones. [17] allowed to use a pool of DNNs and the most effective one is selected to use at runtime. Our proposed DADS does not modify DNNs. It employs full-scale deep models without sacrificing accuracy. 

Computation Offloading. Research efforts focusing on offloading computation from the resource-constrained mobile to the powerful cloud will reduce inference time. Neurosurgeon [18] explored a computation offloading method for DNNs between the mobile device and the cloud server at layer granularity. However, Neurosurgeon is not applicable for the computation partition performed by DADS for a number of reasons: 1) Neurosurgeon only handle chain-topology DNNs that are much easier to process. 2) Neurosurgeon can only handle one inference task, without considering a sequence of tasks. Needless to say the adaptation to network condition realized by DADS. MAUI’s [19] is an offloading framework that can determine where to execute functions (edge or cloud) of a program. However, it is not designed specifically for DNN partitioning as the communication data volume between functions is small. [20] proposed DDNN, a distributed deep neural network architecture that is distributed across computing hierarchies, consisting of the cloud, the edge and end devices. DDNN aims at reducing the communication data size among devices for the given DNN. DADS differs as it handles dynamic network condition to reduce the inference latency (communication and computing latency) rather than communication overhead only. 

Hardware Acceleration. Different from the scope of this paper. Hardware specialization is another method for inference acceleration. [21] proposed DeepBurning, an automation tool to generate FPGA-based accelerators for DNN models. Vanhoucke et al. [22] used fixed point arithmetic and SSSE3/SSE4 instructions on $\mathbf { \boldsymbol { x } } 8 6$ machines to reduce the inference latency. DeepX [23] explored the opportunities to use mobile GPUs to enable real-time deep learning inferences. DADS investigates intelligent collaboration between the edge device and cloud for inference optimization and can be jointly applied with specialized hardware. 

# VII. CONCLUSION

In this paper, we study DNN inference acceleration through collaborative edge-cloud computation. We propose Dynamic Adaptive DNN surgery (DADS) scheme that can partition 

![](images/8042963a31754b3c318e77261926466c302b104191214bc91cfee6287fdb130f.jpg)



Fig. 13: Latency speedup and throughput gain achieved by DADS under heavy workload mode.


![](images/b737f32ab78e4decda9be23cfcfaee39a909903c459097ae8cf2c5b112af5661.jpg)



Fig. 16: Latency speedup and throughput gain achieved by DADS of different networks under heavy workload mode.


![](images/aff4f1fa6546d05ce904ce25c6f33843d1ee404890feafe7dbc6ecea34e8404e.jpg)



Fig. 14: Latency and throughput speedup achieved by DADS vs. Neurosurgeon under light and heavy workload modes.


![](images/b9c32092a043563589417592edcb54629064da01d5775c220784a4b2093f5ed8.jpg)



Fig. 17: Latency speedup and throughput gain achieved by DADS as a function of bandwidth.


![](images/95d4ff4b715f2e12c5c6afecb95e6a9dee2fdc72ab5d0e4a90f72dc1cd720f12.jpg)



Fig. 15: Latency speedup and throughput gain achieved by DADS of different networks under light workload mode.


![](images/207dd708b39d9e5d909190d12bf1bfa672f04c96f3e78214ddea133f686eb5df.jpg)



Fig. 18: The impact of network variance on DADS partition decision using Edge-Only as the baseline.


DNN inference between the edge device and the cloud at the granularity of neural network layers, according to the dynamic network status. We present a comprehensive study of the partition problem under the lightly loaded condition and the heavily loaded condition. We also develop an optimal solution to the lightly loaded condition by converting it to min-cut problem, and design a 3-approximation ratio algorithm under the heavily loaded condition as the problem is NP-hard. We then implement a fully functioning system. Evaluations show that DADS can effectively improve latency and throughput in an order compared with executing the entire DNN on the edge or on the cloud. 

# VIII. ACKNOWLEDGMENTS

We would like to acknowledge the support of the Hong Kong Polytechnic University under Grant PolyU G-YBQE and Innovation Technology Fund (ITF)-UICP-MGJR UIM/363. This work was also supported by the University of Sydney DVC Research/Bridging Support Grant. 

# REFERENCES



[1] V. Va, T. Shimizu, G. Bansal, R. W. Heath Jr et al., “Millimeter wave vehicular communications: A survey,” in Now Publishers Journal on Foundations and Trends in Networking, vol. 10, no. 1, pp. 1–113, 2016. 





[2] “State of Mobile Networks: USA.” https://opensignal.com/reports/2017/ 08/usa/state-of-the-mobile-network, accessed June, 2018. 





[3] “AWS DeepLens,” https://aws.amazon.com/deeplens, accessed June, 2018. 





[4] J. Redmon and A. Farhadi, “YOLO9000: Better, Faster, Stronger,” arXiv preprint arXiv:1612.08242, 2016. 





[5] D. Raca, J. J. Quinlan, A. H. Zahran, and C. J. Sreenan, “Beyond throughput: a 4G LTE dataset with channel and context metrics,” in Proc. ACM MMSys’18, Amsterdam, The Netherlands, Jun. 2018. 





[6] M. Franceschinis, M. Mellia, M. Meo, and M. Munafo, “Measuring TCP over WiFi: A real case,” in 1st workshop on Wireless Network Measurements, Riva Del Garda, Italy, Sep. 2005. 





[7] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proc. IEEE CVPR’15, Boston, MA, Jun. 2015. 





[8] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proc. IEEE CVPR’16, Las Vegas, Nevada, Jul. 2016. 





[9] H. Zhang, G. Ananthanarayanan, P. Bodik, M. Philipose, P. Bahl, and M. J. Freedman, “Live Video Analytics at Scale with Approximation and Delay-Tolerance.” in Proc. USENIX NSDI’17, Boston, MA. 





[10] C. Szegedy, S. Ioffe, V. Vanhoucke, and A. A. Alemi, “Inception-v4, inception-resnet and the impact of residual connections on learning.” in Proc. AAAI’17, San Francisco, CA, Feb. 2017. 





[11] T. Stockhammer, “Dynamic adaptive streaming over HTTP: standards and design principles,” in Proc. ACM MMSYS ’11, Santa Clara, CA, Feb. 2011. 





[12] Y. Boykov and V. Kolmogorov, “An experimental comparison of mincut/max- flow algorithms for energy minimization in vision,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 9, pp. 1124–1137, 2004. 





[13] P. Bonsma, “Most balanced minimum cuts,” Discrete Applied Mathematics, vol. 158, no. 4, pp. 261–276, 2010. 





[14] P. Aleksic, M. Ghodsi, A. Michaely, C. Allauzen, B. Hall, D. Rybach, and P. Moreno, “Bringing contextual information to google speech recognition,” in Proc. INTERSPEECH’15, Dresden, Germany, Sep. 2015. 





[15] S. Han, H. Shen, M. Philipose, S. Agarwal, A. Wolman, and A. Krishnamurthy, “MCDNN: An approximation-based execution framework for deep stream processing under resource constraints,” in Proc. ACM MobiSys’16, Singapore, Jun. 2016. 





[16] E. Variani, X. Lei, E. McDermott, I. L. Moreno, and J. Gonzalez-Dominguez, “Deep neural networks for small footprint text-dependent speaker verification,” in Proc. IEEE ICASSP’14, Florence, Italy, May 2014. 





[17] B. Taylor, V. S. Marco, W. Wolff, Y. Elkhatib, and Z. Wang, “Adaptive selection of deep learning models on embedded systems,” arXiv preprint arXiv:1805.04252, 2018. 





[18] Y. Kang, J. Hauswald, C. Gao, A. Rovinski, T. Mudge, J. Mars, and L. Tang, “Neurosurgeon: Collaborative intelligence between the cloud and mobile edge,” in Proc. ACM ASPLOS’17, Xi’an, China, Apr. 2017. 





[19] E. Cuervo, A. Balasubramanian, D.-k. Cho, A. Wolman, S. Saroiu, R. Chandra, and P. Bahl, “Maui: making smartphones last longer with code offload,” in Proc. ACM MobiSys’10, San Francisco, CA, Jun. 2010. 





[20] S. Teerapittayanon, B. McDanel, and H. Kung, “Distributed deep neural networks over the cloud, the edge and end devices,” in Proc. IEEE ICDCS’17, Atlanta, GA, Jun. 2017. 





[21] Y. Wang, J. Xu, Y. Han, H. Li, and X. Li, “Deepburning: automatic generation of fpga-based learning accelerators for the neural network family,” in Proc. ACM DAC’16, Austin, TX, Jun. 2016. 





[22] V. Vanhoucke, A. Senior, and M. Z. Mao, “Improving the speed of neural networks on cpus,” in Proc. NIPS’11, Granada, Spain, Jan. 2011. 





[23] N. D. Lane, S. Bhattacharya, P. Georgiev, C. Forlivesi, L. Jiao, L. Qendro, and F. Kawsar, “DeepX: A software accelerator for low-power deep learning inference on mobile devices,” in Proc. IEEE IPSN’16, Vienna, Austria, Apr. 2016. 

