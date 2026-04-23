from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .flow import min_cut
from .profiling import DNNProfile, LayerProfile


MB = 1024 * 1024
INF = 1e15


@dataclass(frozen=True)
class CostWeights:
    alpha: float = 1.0
    beta: float = 1.0


@dataclass(frozen=True)
class PartitionResult:
    method: str
    model_name: str
    cost: float
    latency: float
    energy: float
    local_time: float
    upload_time: float
    wait_time: float
    server_time: float
    down_time: float
    local_nodes: Tuple[str, ...]
    cloud_nodes: Tuple[str, ...]
    transmitted_nodes: Tuple[str, ...]

    @property
    def partition_label(self) -> str:
        if not self.cloud_nodes:
            return "local-only"
        if self.transmitted_nodes == ("input",):
            return "server-only"
        return ",".join(self.transmitted_nodes)


@dataclass(frozen=True)
class AssignmentResult:
    total_cost: float
    total_latency: float
    total_energy: float
    md_to_server: Tuple[int, ...]
    selected: Tuple[PartitionResult, ...]


def gflops_to_flops(value: float) -> float:
    return float(value) * 1e9


def mbps_to_bps(value_mb_s: float) -> float:
    return float(value_mb_s) * MB


def _layer_time(layer: LayerProfile, gflops: float) -> float:
    return layer.flops / gflops_to_flops(gflops) if gflops > 0 else INF


def _comm_time(num_bytes: float, bandwidth_mb_s: float) -> float:
    return num_bytes / mbps_to_bps(bandwidth_mb_s) if bandwidth_mb_s > 0 else INF


def _comm_bytes(profile: DNNProfile, node_id: str) -> float:
    if node_id == "input":
        return profile.input_bytes
    layer = profile.layer_by_id[node_id]
    return layer.output_bytes


def _compressed_upload_bytes(profile: DNNProfile, node_id: str, compression_ratio: float) -> float:
    raw_bytes = _comm_bytes(profile, node_id)
    # Raw input upload is left unchanged; only intermediate features use the
    # fixed analytical compression ratio.
    if node_id == "input":
        return raw_bytes
    ratio = max(float(compression_ratio), 1.0)
    return raw_bytes / ratio


def _successors(edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    succ: Dict[str, List[str]] = {}
    for src, dst in edges:
        succ.setdefault(src, []).append(dst)
    return succ


def _predecessors(edges: Iterable[Tuple[str, str]]) -> Dict[str, List[str]]:
    pred: Dict[str, List[str]] = {}
    for src, dst in edges:
        pred.setdefault(dst, []).append(src)
    return pred


def _evaluate_assignment(
    profile: DNNProfile,
    local_ids: Iterable[str],
    cloud_ids: Iterable[str],
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float,
    weights: CostWeights,
    method: str,
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    layer_by_id = profile.layer_by_id
    local_set = set(local_ids)
    cloud_set = set(cloud_ids)
    succ = _successors(profile.edges)

    transmitted: List[str] = []
    for src, dsts in succ.items():
        if src == "input":
            src_is_local = True
        else:
            src_is_local = src in local_set
        if src_is_local and any(dst in cloud_set for dst in dsts):
            transmitted.append(src)

    local_time = sum(_layer_time(layer_by_id[node_id], md_gflops) for node_id in local_set)
    server_time = sum(_layer_time(layer_by_id[node_id], es_gflops) for node_id in cloud_set)
    upload_time = sum(_comm_time(_compressed_upload_bytes(profile, node_id, compression_ratio), bandwidth_up_mb_s) for node_id in transmitted)
    wait_time = float(server_wait_time_s) if cloud_set else 0.0
    down_time = _comm_time(profile.output_bytes, bandwidth_down_mb_s) if cloud_set else 0.0
    latency = local_time + upload_time + wait_time + server_time + down_time
    energy = local_power_w * local_time + upload_power_w * upload_time
    cost = weights.alpha * latency + weights.beta * energy

    ordered = profile.layer_ids
    return PartitionResult(
        method=method,
        model_name=profile.model_name,
        cost=cost,
        latency=latency,
        energy=energy,
        local_time=local_time,
        upload_time=upload_time,
        wait_time=wait_time,
        server_time=server_time,
        down_time=down_time,
        local_nodes=tuple(node_id for node_id in ordered if node_id in local_set),
        cloud_nodes=tuple(node_id for node_id in ordered if node_id in cloud_set),
        transmitted_nodes=tuple(transmitted),
    )


def evaluate_partition(
    profile: DNNProfile,
    local_ids: Iterable[str],
    cloud_ids: Iterable[str],
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    method: str = "re-evaluate",
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    return _evaluate_assignment(
        profile,
        local_ids,
        cloud_ids,
        md_gflops,
        es_gflops,
        local_power_w,
        upload_power_w,
        bandwidth_up_mb_s,
        bandwidth_down_mb_s,
        weights,
        method,
        server_wait_time_s=server_wait_time_s,
        compression_ratio=compression_ratio,
    )


def min_cut_partition(
    profile: DNNProfile,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    """Solve the MD-ES partition subproblem as a closed DAG s-t min-cut.

    Source side means local execution on the mobile device. Sink side means
    server execution. Infinite reverse dependency arcs disallow cloud-to-local
    transitions, which keeps the cut physically executable.
    """

    actual_ids = list(profile.layer_ids)
    layer_by_id = profile.layer_by_id
    node_ids = ["input"] + actual_ids
    index = {node_id: i for i, node_id in enumerate(node_ids)}
    edges: List[Tuple[int, int, float]] = []
    source = len(node_ids)
    sink = source + 1
    next_idx = sink + 1

    def add_node(node_id: str) -> int:
        nonlocal next_idx
        index[node_id] = next_idx
        next_idx += 1
        return index[node_id]

    for node_id in node_ids:
        if node_id == "input":
            cloud_cost = INF
            local_cost = 0.0
        else:
            layer = layer_by_id[node_id]
            local_time = _layer_time(layer, md_gflops)
            cloud_time = _layer_time(layer, es_gflops)
            local_cost = weights.alpha * local_time + weights.beta * local_power_w * local_time
            cloud_cost = weights.alpha * cloud_time
            if node_id == profile.output_node_id:
                cloud_cost += weights.alpha * (_comm_time(profile.output_bytes, bandwidth_down_mb_s) + server_wait_time_s)
        edges.append((source, index[node_id], cloud_cost))
        edges.append((index[node_id], sink, local_cost))

    succ = _successors(profile.edges)
    for src, dsts in succ.items():
        aux_id = f"{src}__tx"
        aux_idx = add_node(aux_id)
        tx_time = _comm_time(_compressed_upload_bytes(profile, src, compression_ratio), bandwidth_up_mb_s)
        tx_cost = weights.alpha * tx_time + weights.beta * (upload_power_w * tx_time)
        edges.append((index[src], aux_idx, tx_cost))
        for dst in dsts:
            edges.append((aux_idx, index[dst], INF))
            edges.append((index[dst], index[src], INF))

    value, reachable = min_cut(next_idx, edges, source, sink)
    del value
    local_ids = [node_id for node_id in actual_ids if reachable[index[node_id]]]
    cloud_ids = [node_id for node_id in actual_ids if not reachable[index[node_id]]]
    return _evaluate_assignment(
        profile,
        local_ids,
        cloud_ids,
        md_gflops,
        es_gflops,
        local_power_w,
        upload_power_w,
        bandwidth_up_mb_s,
        bandwidth_down_mb_s,
        weights,
        method="min-cut",
        server_wait_time_s=server_wait_time_s,
        compression_ratio=compression_ratio,
    )


def chain_partition(
    profile: DNNProfile,
    cut_index: int,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    method: str = "chain",
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    ids = list(profile.layer_ids)
    cut_index = max(0, min(int(cut_index), len(ids)))
    local_ids = ids[:cut_index]
    cloud_ids = ids[cut_index:]

    return _evaluate_assignment(
        profile,
        local_ids,
        cloud_ids,
        md_gflops,
        es_gflops,
        local_power_w,
        upload_power_w,
        bandwidth_up_mb_s,
        bandwidth_down_mb_s,
        weights,
        method,
        server_wait_time_s=server_wait_time_s,
        compression_ratio=compression_ratio,
    )


def best_chain_topo_partition(
    profile: DNNProfile,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    """Best single prefix cut over the profile's FX topological order."""

    best: Optional[PartitionResult] = None
    for cut_index in range(len(profile.layer_ids) + 1):
        result = chain_partition(
            profile,
            cut_index,
            md_gflops=md_gflops,
            es_gflops=es_gflops,
            local_power_w=local_power_w,
            upload_power_w=upload_power_w,
            bandwidth_up_mb_s=bandwidth_up_mb_s,
            bandwidth_down_mb_s=bandwidth_down_mb_s,
            weights=weights,
            method="chain-topo",
            server_wait_time_s=server_wait_time_s,
            compression_ratio=compression_ratio,
        )
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        raise RuntimeError(f"could not find chain-topology partition for {profile.model_name}")
    return best


def greedy_cut_partition(
    profile: DNNProfile,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    """Greedily build a DAG-closed local set and keep the best observed cut."""
    ordered = tuple(profile.layer_ids)
    all_ids = set(ordered)
    pred = _predecessors(profile.edges)
    local_set: set[str] = set()
    remaining = set(ordered)

    best = _evaluate_assignment(
        profile,
        local_set,
        all_ids,
        md_gflops,
        es_gflops,
        local_power_w,
        upload_power_w,
        bandwidth_up_mb_s,
        bandwidth_down_mb_s,
        weights,
        method="greedy-cut",
        server_wait_time_s=server_wait_time_s,
        compression_ratio=compression_ratio,
    )

    while remaining:
        ready = [
            node_id
            for node_id in ordered
            if node_id in remaining and all(parent == "input" or parent in local_set for parent in pred.get(node_id, ()))
        ]
        if not ready:
            break

        candidate_result: Optional[PartitionResult] = None
        candidate_node: Optional[str] = None
        for node_id in ready:
            trial_local = local_set | {node_id}
            trial_cloud = all_ids - trial_local
            result = _evaluate_assignment(
                profile,
                trial_local,
                trial_cloud,
                md_gflops,
                es_gflops,
                local_power_w,
                upload_power_w,
                bandwidth_up_mb_s,
                bandwidth_down_mb_s,
                weights,
                method="greedy-cut",
                server_wait_time_s=server_wait_time_s,
                compression_ratio=compression_ratio,
            )
            if candidate_result is None or result.cost < candidate_result.cost:
                candidate_result = result
                candidate_node = node_id

        if candidate_node is None or candidate_result is None:
            break
        local_set.add(candidate_node)
        remaining.remove(candidate_node)
        if candidate_result.cost < best.cost:
            best = candidate_result

    return best


def random_cut_index(profile: DNNProfile, rng: random.Random) -> int:
    n = len(profile.layers)
    if n < 2:
        return n
    return rng.randint(1, n - 1)


def random_partition(
    profile: DNNProfile,
    rng: random.Random,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    server_wait_time_s: float = 0.0,
    compression_ratio: float = 1.0,
) -> PartitionResult:
    return chain_partition(
        profile,
        cut_index=random_cut_index(profile, rng),
        md_gflops=md_gflops,
        es_gflops=es_gflops,
        local_power_w=local_power_w,
        upload_power_w=upload_power_w,
        bandwidth_up_mb_s=bandwidth_up_mb_s,
        bandwidth_down_mb_s=bandwidth_down_mb_s,
        weights=weights,
        method="random-partition",
        server_wait_time_s=server_wait_time_s,
        compression_ratio=compression_ratio,
    )


def build_result_matrix(
    profiles: Mapping[str, DNNProfile],
    md_models: Sequence[str],
    md_gflops: Sequence[float],
    es_gflops: Sequence[float],
    local_powers: Sequence[float],
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    method: str,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    server_wait_times_s: Optional[Sequence[float]] = None,
    compression_ratio: float = 1.0,
) -> List[List[PartitionResult]]:
    wait_times = list(server_wait_times_s) if server_wait_times_s is not None else [0.0] * len(es_gflops)
    if len(wait_times) != len(es_gflops):
        raise ValueError("server_wait_times_s must match the number of server slots")
    matrix: List[List[PartitionResult]] = []
    for i, model_name in enumerate(md_models):
        profile = profiles[model_name.lower()]
        row: List[PartitionResult] = []
        for server_idx, server_gflops in enumerate(es_gflops):
            server_wait_time_s = float(wait_times[server_idx])
            if method == "min-cut":
                row.append(
                    min_cut_partition(
                        profile,
                        md_gflops=md_gflops[i],
                        es_gflops=server_gflops,
                        local_power_w=local_powers[i],
                        upload_power_w=upload_power_w,
                        bandwidth_up_mb_s=bandwidth_up_mb_s,
                        bandwidth_down_mb_s=bandwidth_down_mb_s,
                        weights=weights,
                        server_wait_time_s=server_wait_time_s,
                        compression_ratio=compression_ratio,
                    )
                )
            elif method == "server-only":
                row.append(
                    chain_partition(
                        profile,
                        0,
                        md_gflops=md_gflops[i],
                        es_gflops=server_gflops,
                        local_power_w=local_powers[i],
                        upload_power_w=upload_power_w,
                        bandwidth_up_mb_s=bandwidth_up_mb_s,
                        bandwidth_down_mb_s=bandwidth_down_mb_s,
                        weights=weights,
                        method="server-only",
                        server_wait_time_s=server_wait_time_s,
                        compression_ratio=compression_ratio,
                    )
                )
            elif method == "chain-topo":
                row.append(
                    best_chain_topo_partition(
                        profile,
                        md_gflops=md_gflops[i],
                        es_gflops=server_gflops,
                        local_power_w=local_powers[i],
                        upload_power_w=upload_power_w,
                        bandwidth_up_mb_s=bandwidth_up_mb_s,
                        bandwidth_down_mb_s=bandwidth_down_mb_s,
                        weights=weights,
                        server_wait_time_s=server_wait_time_s,
                        compression_ratio=compression_ratio,
                    )
                )
            elif method == "greedy-cut":
                row.append(
                    greedy_cut_partition(
                        profile,
                        md_gflops=md_gflops[i],
                        es_gflops=server_gflops,
                        local_power_w=local_powers[i],
                        upload_power_w=upload_power_w,
                        bandwidth_up_mb_s=bandwidth_up_mb_s,
                        bandwidth_down_mb_s=bandwidth_down_mb_s,
                        weights=weights,
                        server_wait_time_s=server_wait_time_s,
                        compression_ratio=compression_ratio,
                    )
                )
            else:
                raise ValueError(f"unsupported matrix method: {method}")
        matrix.append(row)
    return matrix


def build_random_partition_matrix(
    profiles: Mapping[str, DNNProfile],
    md_models: Sequence[str],
    md_gflops: Sequence[float],
    es_gflops: Sequence[float],
    local_powers: Sequence[float],
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    rng: random.Random,
    bandwidth_down_mb_s: float = 10.0,
    weights: CostWeights = CostWeights(),
    server_wait_times_s: Optional[Sequence[float]] = None,
    compression_ratio: float = 1.0,
) -> List[List[PartitionResult]]:
    wait_times = list(server_wait_times_s) if server_wait_times_s is not None else [0.0] * len(es_gflops)
    if len(wait_times) != len(es_gflops):
        raise ValueError("server_wait_times_s must match the number of server slots")
    matrix: List[List[PartitionResult]] = []
    for i, model_name in enumerate(md_models):
        profile = profiles[model_name.lower()]
        cut_index = random_cut_index(profile, rng)
        row: List[PartitionResult] = []
        for server_idx, server_gflops in enumerate(es_gflops):
            row.append(
                chain_partition(
                    profile,
                    cut_index=cut_index,
                    md_gflops=md_gflops[i],
                    es_gflops=server_gflops,
                    local_power_w=local_powers[i],
                    upload_power_w=upload_power_w,
                    bandwidth_up_mb_s=bandwidth_up_mb_s,
                    bandwidth_down_mb_s=bandwidth_down_mb_s,
                    weights=weights,
                    method="random-partition",
                    server_wait_time_s=float(wait_times[server_idx]),
                    compression_ratio=compression_ratio,
                )
            )
        matrix.append(row)
    return matrix


def assign_min_cost(result_matrix: Sequence[Sequence[PartitionResult]]) -> AssignmentResult:
    cost = np.array([[item.cost for item in row] for row in result_matrix], dtype=float)
    rows, cols = linear_sum_assignment(cost)
    selected_by_md: List[Optional[PartitionResult]] = [None] * len(result_matrix)
    md_to_server = [-1] * len(result_matrix)
    for row, col in zip(rows, cols):
        selected_by_md[row] = result_matrix[row][col]
        md_to_server[row] = int(col)
    selected = tuple(item for item in selected_by_md if item is not None)
    return AssignmentResult(
        total_cost=sum(item.cost for item in selected),
        total_latency=sum(item.latency for item in selected),
        total_energy=sum(item.energy for item in selected),
        md_to_server=tuple(md_to_server),
        selected=selected,
    )


def assign_random(result_matrix: Sequence[Sequence[PartitionResult]], seed: int = 7) -> AssignmentResult:
    rng = random.Random(seed)
    cols = list(range(len(result_matrix[0])))
    rng.shuffle(cols)
    cols = cols[: len(result_matrix)]
    selected = tuple(result_matrix[i][cols[i]] for i in range(len(result_matrix)))
    return AssignmentResult(
        total_cost=sum(item.cost for item in selected),
        total_latency=sum(item.latency for item in selected),
        total_energy=sum(item.energy for item in selected),
        md_to_server=tuple(cols),
        selected=selected,
    )


def local_only_total(
    profiles: Mapping[str, DNNProfile],
    md_models: Sequence[str],
    md_gflops: Sequence[float],
    local_powers: Sequence[float],
    weights: CostWeights = CostWeights(),
    compression_ratio: float = 1.0,
) -> AssignmentResult:
    selected: List[PartitionResult] = []
    for i, model_name in enumerate(md_models):
        profile = profiles[model_name.lower()]
        result = chain_partition(
            profile,
            len(profile.layers),
            md_gflops=md_gflops[i],
            es_gflops=1.0,
            local_power_w=local_powers[i],
            upload_power_w=0.0,
            bandwidth_up_mb_s=1.0,
            weights=weights,
            method="local-only",
            compression_ratio=compression_ratio,
        )
        selected.append(result)
    return AssignmentResult(
        total_cost=sum(item.cost for item in selected),
        total_latency=sum(item.latency for item in selected),
        total_energy=sum(item.energy for item in selected),
        md_to_server=tuple(-1 for _ in selected),
        selected=tuple(selected),
    )


def write_assignment_csv(path: Path, assignment: AssignmentResult, md_models: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "md",
                "model",
                "server_slot",
                "method",
                "partition",
                "cost",
                "latency",
                "energy",
                "local_time",
                "upload_time",
                "wait_time",
                "server_time",
                "down_time",
            ]
        )
        for i, result in enumerate(assignment.selected):
            writer.writerow(
                [
                    i,
                    md_models[i],
                    assignment.md_to_server[i],
                    result.method,
                    result.partition_label,
                    result.cost,
                    result.latency,
                    result.energy,
                    result.local_time,
                    result.upload_time,
                    result.wait_time,
                    result.server_time,
                    result.down_time,
                ]
            )
