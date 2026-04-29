from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .partition import (
    AssignmentResult,
    CostWeights,
    PartitionResult,
    assign_min_cost,
    assign_random,
    best_chain_topo_partition,
    build_random_partition_matrix,
    build_result_matrix,
    chain_partition,
    evaluate_partition,
    greedy_cut_partition,
    local_only_total,
    min_cut_partition,
    write_assignment_csv,
)
from .profiling import DNNProfile, build_profiles, canonical_model_name


MAIN_MODELS = ("alexnet", "tiny_yolov2", "googlenet", "densenet121", "resnet101")
SINGLE_PAIR_MODELS = MAIN_MODELS
DEFAULT_COMPRESSION_RATIO = 4.0
BANDWIDTHS = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0)
BETAS = (0, 0.5, 1, 2, 4)
MODULE1_BANDWIDTHS = BANDWIDTHS
MODULE1_LOCAL_GFLOPS = (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
SINGLE_PAIR_METHOD_ORDER = (
    "Only-Local",
    "Only-Server",
    "ChainTopo",
    "GreedyCut",
    "MinCut",
)
SINGLE_PAIR_FOCUS_METHOD_ORDER = (
    "Only-Local",
    "Only-Server",
    "MinCut",
)
MULTIUSER_METHOD_ORDER = (
    "Only-Local",
    "Only-Server",
    "MinCut+BMatch",
    "MinCut+RMatch",
    "Rpartition+RMatch",
)
ONLINE_METHOD_ORDER = (
    "Online Only-Local",
    "Online Only-Server",
    "Static MinCut+BMatch",
    "Online MinCut+BMatch",
)
MODULE1_METHOD_COLORS = {
    "Only-Local": "#1f77b4",
    "Only-Server": "#ff7f0e",
    "MinCut": "#2ca02c",
}
MODEL_DISPLAY_NAMES = {
    "alexnet": "AlexNet",
    "tiny_yolov2": "TinyYOLOv2",
    "googlenet": "GoogLeNet",
    "densenet121": "DenseNet121",
    "resnet101": "ResNet101",
}


@dataclass(frozen=True)
class Scenario:
    name: str
    md_models: Sequence[str]
    md_gflops: Sequence[float]
    es_slots_gflops: Sequence[float]
    local_powers: Sequence[float]
    upload_power_w: float = 1.0


@dataclass(frozen=True)
class OnlineSlotState:
    slot: int
    active_md_ids: Tuple[int, ...]
    bandwidth_up_mb_s: float
    bandwidth_down_mb_s: float
    external_load_factors: Tuple[float, ...]


@dataclass(frozen=True)
class OnlineScenario:
    name: str
    base_scenario: Scenario
    slot_duration_s: float
    slots: Tuple[OnlineSlotState, ...]


def mixed5_scenario() -> Scenario:
    return Scenario(
        name="mixed5",
        md_models=MAIN_MODELS,
        md_gflops=(1.5, 3.0, 5.0, 7.0, 9.5),
        es_slots_gflops=(22.0, 30.0, 35.0, 40.0, 48.0),
        local_powers=(1.2, 1.5, 2.0, 2.5, 2.8),
        upload_power_w=1.0,
    )


def googlenet_stress_scenario() -> Scenario:
    return Scenario(
        name="googlenet4_stress",
        md_models=("googlenet", "googlenet", "googlenet", "googlenet"),
        md_gflops=(1.0, 1.0, 1.0, 1.0),
        es_slots_gflops=(3.0, 4.0, 5.0, 6.0),
        local_powers=(2.0, 2.0, 2.0, 2.0),
    )


def _all_active_ids(scenario: Scenario) -> Tuple[int, ...]:
    return tuple(range(len(scenario.md_models)))


def online_bandwidth_trace_scenario() -> OnlineScenario:
    base = mixed5_scenario()
    es_count = len(base.es_slots_gflops)
    active = _all_active_ids(base)
    slots = tuple(
        OnlineSlotState(
            slot=slot,
            active_md_ids=active,
            bandwidth_up_mb_s=bandwidth,
            bandwidth_down_mb_s=10.0,
            external_load_factors=(0.0,) * es_count,
        )
        for slot, bandwidth in enumerate((0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0))
    )
    return OnlineScenario(name="online_bandwidth_trace", base_scenario=base, slot_duration_s=0.25, slots=slots)


def online_burst_load_scenario() -> OnlineScenario:
    base = mixed5_scenario()
    es_count = len(base.es_slots_gflops)
    active_groups = (
        (0, 1, 2),
        (0, 1, 2),
        (0, 1, 2),
        _all_active_ids(base),
        _all_active_ids(base),
        _all_active_ids(base),
        (2, 3, 4),
        (2, 3, 4),
        (2, 3, 4),
    )
    slots: List[OnlineSlotState] = []
    for slot, active_ids in enumerate(active_groups):
        if slot < 3:
            load_factors = (0.0,) * es_count
        elif slot < 6:
            load_factors = (1.5, 1.0) + (0.0,) * (es_count - 2)
        else:
            load_factors = (0.5, 0.0) + (0.0,) * (es_count - 2)
        slots.append(
            OnlineSlotState(
                slot=slot,
                active_md_ids=tuple(active_ids),
                bandwidth_up_mb_s=1.0,
                bandwidth_down_mb_s=10.0,
                external_load_factors=tuple(load_factors),
            )
        )
    return OnlineScenario(name="online_burst_load", base_scenario=base, slot_duration_s=0.25, slots=tuple(slots))


def online_composite_nonstationary_scenario() -> OnlineScenario:
    base = mixed5_scenario()
    es_count = len(base.es_slots_gflops)
    phases = (
        ((0, 1, 2), 0.5, (1.5, 0.5) + (0.0,) * (es_count - 2)),
        ((0, 1, 2), 0.5, (1.5, 0.5) + (0.0,) * (es_count - 2)),
        ((0, 1, 2), 0.5, (1.5, 0.5) + (0.0,) * (es_count - 2)),
        (_all_active_ids(base), 1.0, (1.0, 1.0, 0.4) + (0.0,) * (es_count - 3)),
        (_all_active_ids(base), 1.0, (1.0, 1.0, 0.4) + (0.0,) * (es_count - 3)),
        (_all_active_ids(base), 1.0, (1.0, 1.0, 0.4) + (0.0,) * (es_count - 3)),
        ((1, 2, 3, 4), 4.0, (0.2, 0.3, 0.8) + (0.0,) * (es_count - 3)),
        ((1, 2, 3, 4), 4.0, (0.2, 0.3, 0.8) + (0.0,) * (es_count - 3)),
        ((1, 2, 3, 4), 4.0, (0.2, 0.3, 0.8) + (0.0,) * (es_count - 3)),
    )
    slots = tuple(
        OnlineSlotState(
            slot=slot,
            active_md_ids=tuple(active_ids),
            bandwidth_up_mb_s=bandwidth_up_mb_s,
            bandwidth_down_mb_s=10.0,
            external_load_factors=tuple(load_factors),
        )
        for slot, (active_ids, bandwidth_up_mb_s, load_factors) in enumerate(phases)
    )
    return OnlineScenario(name="online_composite_nonstationary", base_scenario=base, slot_duration_s=0.25, slots=slots)


def required_models(scenarios: Iterable[Scenario]) -> List[str]:
    return sorted({canonical_model_name(model) for scenario in scenarios for model in scenario.md_models})


def standard_single_pair_kwargs() -> dict:
    return {
        "md_gflops": 3.0,
        "es_gflops": 35.0,
        "local_power_w": 2.0,
        "upload_power_w": 1.0,
        "bandwidth_up_mb_s": 1.0,
        "bandwidth_down_mb_s": 10.0,
    }


def googlenet_stress_pair_kwargs() -> dict:
    return {
        "md_gflops": 1.0,
        "es_gflops": 5.0,
        "local_power_w": 2.0,
        "upload_power_w": 1.0,
        "bandwidth_up_mb_s": 0.05,
        "bandwidth_down_mb_s": 10.0,
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def _summary_row(
    experiment: str,
    scenario: str,
    bandwidth: float,
    beta: float,
    compression_ratio: float,
    method: str,
    assignments: Sequence[AssignmentResult],
) -> dict:
    costs = [item.total_cost for item in assignments]
    latencies = [item.total_latency for item in assignments]
    energies = [item.total_energy for item in assignments]
    cost_mean, cost_std = _mean_std(costs)
    latency_mean, latency_std = _mean_std(latencies)
    energy_mean, energy_std = _mean_std(energies)
    return {
        "experiment": experiment,
        "scenario": scenario,
        "bandwidth_mb_s": bandwidth,
        "beta": beta,
        "compression_ratio": compression_ratio,
        "method": method,
        "repeat_count": len(assignments),
        "total_cost": cost_mean,
        "total_cost_std": cost_std,
        "total_latency": latency_mean,
        "total_latency_std": latency_std,
        "total_energy": energy_mean,
        "total_energy_std": energy_std,
    }


def _repeat_rows(
    experiment: str,
    scenario: str,
    bandwidth: float,
    beta: float,
    compression_ratio: float,
    method: str,
    assignments: Sequence[AssignmentResult],
) -> List[dict]:
    return [
        {
            "experiment": experiment,
            "scenario": scenario,
            "bandwidth_mb_s": bandwidth,
            "beta": beta,
            "compression_ratio": compression_ratio,
            "method": method,
            "repeat": i,
            "total_cost": assignment.total_cost,
            "total_latency": assignment.total_latency,
            "total_energy": assignment.total_energy,
        }
        for i, assignment in enumerate(assignments)
    ]


def run_multiuser_methods(
    scenario: Scenario,
    profiles: Mapping[str, DNNProfile],
    bandwidth: float,
    weights: CostWeights,
    random_repeats: int,
    seed_base: int,
    compression_ratio: float,
) -> Dict[str, List[AssignmentResult]]:
    if len(scenario.es_slots_gflops) < len(scenario.md_models):
        raise ValueError(
            f"scenario {scenario.name} has {len(scenario.md_models)} MDs but only {len(scenario.es_slots_gflops)} server slots"
        )

    mincut_matrix = build_result_matrix(
        profiles,
        scenario.md_models,
        scenario.md_gflops,
        scenario.es_slots_gflops,
        scenario.local_powers,
        scenario.upload_power_w,
        bandwidth,
        method="min-cut",
        weights=weights,
        compression_ratio=compression_ratio,
    )
    server_matrix = build_result_matrix(
        profiles,
        scenario.md_models,
        scenario.md_gflops,
        scenario.es_slots_gflops,
        scenario.local_powers,
        scenario.upload_power_w,
        bandwidth,
        method="server-only",
        weights=weights,
        compression_ratio=compression_ratio,
    )

    results: Dict[str, List[AssignmentResult]] = {
        "Only-Local": [
            local_only_total(
                profiles,
                scenario.md_models,
                scenario.md_gflops,
                scenario.local_powers,
                weights,
                compression_ratio=compression_ratio,
            )
        ],
        "Only-Server": [assign_min_cost(server_matrix)],
        "MinCut+BMatch": [assign_min_cost(mincut_matrix)],
        "MinCut+RMatch": [],
        "Rpartition+RMatch": [],
    }

    for repeat in range(random_repeats):
        seed = seed_base + repeat
        results["MinCut+RMatch"].append(assign_random(mincut_matrix, seed=seed))
        random_matrix = build_random_partition_matrix(
            profiles,
            scenario.md_models,
            scenario.md_gflops,
            scenario.es_slots_gflops,
            scenario.local_powers,
            scenario.upload_power_w,
            bandwidth,
            rng=random.Random(seed),
            weights=weights,
            compression_ratio=compression_ratio,
        )
        results["Rpartition+RMatch"].append(assign_random(random_matrix, seed=seed + 10000))
    return results


def run_single_pair_methods(
    profile: DNNProfile,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    bandwidth_up_mb_s: float,
    bandwidth_down_mb_s: float,
    weights: CostWeights,
    compression_ratio: float,
) -> tuple[Dict[str, PartitionResult], List[dict]]:
    common_kwargs = {
        "profile": profile,
        "md_gflops": md_gflops,
        "es_gflops": es_gflops,
        "local_power_w": local_power_w,
        "upload_power_w": upload_power_w,
        "bandwidth_up_mb_s": bandwidth_up_mb_s,
        "bandwidth_down_mb_s": bandwidth_down_mb_s,
        "weights": weights,
        "compression_ratio": compression_ratio,
    }

    def local_only() -> PartitionResult:
        return chain_partition(
            profile,
            len(profile.layers),
            md_gflops=md_gflops,
            es_gflops=es_gflops,
            local_power_w=local_power_w,
            upload_power_w=upload_power_w,
            bandwidth_up_mb_s=bandwidth_up_mb_s,
            bandwidth_down_mb_s=bandwidth_down_mb_s,
            weights=weights,
            method="local-only",
            compression_ratio=compression_ratio,
        )

    def server_only() -> PartitionResult:
        return chain_partition(
            profile,
            0,
            md_gflops=md_gflops,
            es_gflops=es_gflops,
            local_power_w=local_power_w,
            upload_power_w=upload_power_w,
            bandwidth_up_mb_s=bandwidth_up_mb_s,
            bandwidth_down_mb_s=bandwidth_down_mb_s,
            weights=weights,
            method="server-only",
            compression_ratio=compression_ratio,
        )

    runners = {
        "Only-Local": local_only,
        "Only-Server": server_only,
        "ChainTopo": lambda: best_chain_topo_partition(**common_kwargs),
        "GreedyCut": lambda: greedy_cut_partition(**common_kwargs),
        "MinCut": lambda: min_cut_partition(**common_kwargs),
    }

    results: Dict[str, PartitionResult] = {}
    runtime_rows: List[dict] = []
    for method in SINGLE_PAIR_METHOD_ORDER:
        func = runners[method]
        runtime_mean, runtime_std = _average_runtime_ms(func, repeats=3)
        result = func()
        results[method] = result
        runtime_rows.append(
            {
                "model": profile.model_name,
                "method": method,
                "nodes": len(profile.layers),
                "edges": len(profile.edges),
                "runtime_ms": runtime_mean,
                "runtime_ms_std": runtime_std,
                "cost": result.cost,
                "latency": result.latency,
                "energy": result.energy,
                "partition": result.partition_label,
                "transmitted_node_count": len(result.transmitted_nodes),
            }
        )
    return results, runtime_rows


def _effective_server_gflops(
    base_es_gflops: Sequence[float],
    external_load_factors: Sequence[float],
) -> Tuple[float, ...]:
    if len(base_es_gflops) != len(external_load_factors):
        raise ValueError("external_load_factors must match the number of server slots")
    return tuple(max(base / (1.0 + max(load, 0.0)), 1e-6) for base, load in zip(base_es_gflops, external_load_factors))


def _select_active_rows(
    result_matrix: Sequence[Sequence[PartitionResult]],
    active_md_ids: Sequence[int],
) -> List[List[PartitionResult]]:
    return [list(result_matrix[md_idx]) for md_idx in active_md_ids]


def _assignment_from_selected(md_to_server: Sequence[int], selected: Sequence[PartitionResult]) -> AssignmentResult:
    selected_tuple = tuple(selected)
    return AssignmentResult(
        total_cost=sum(item.cost for item in selected_tuple),
        total_latency=sum(item.latency for item in selected_tuple),
        total_energy=sum(item.energy for item in selected_tuple),
        md_to_server=tuple(int(slot) for slot in md_to_server),
        selected=selected_tuple,
    )


def _online_local_only_assignment(
    scenario: Scenario,
    active_md_ids: Sequence[int],
    profiles: Mapping[str, DNNProfile],
    slot_state: OnlineSlotState,
    weights: CostWeights,
    compression_ratio: float,
) -> AssignmentResult:
    selected: List[PartitionResult] = []
    for md_idx in active_md_ids:
        profile = profiles[scenario.md_models[md_idx].lower()]
        selected.append(
            evaluate_partition(
                profile,
                local_ids=profile.layer_ids,
                cloud_ids=(),
                md_gflops=scenario.md_gflops[md_idx],
                es_gflops=1.0,
                local_power_w=scenario.local_powers[md_idx],
                upload_power_w=scenario.upload_power_w,
                bandwidth_up_mb_s=slot_state.bandwidth_up_mb_s,
                bandwidth_down_mb_s=slot_state.bandwidth_down_mb_s,
                weights=weights,
                method="online-local-only",
                compression_ratio=compression_ratio,
            )
        )
    return _assignment_from_selected(tuple(-1 for _ in selected), selected)


def _re_evaluate_planned_assignment(
    scenario: Scenario,
    active_md_ids: Sequence[int],
    planned_assignment: AssignmentResult,
    profiles: Mapping[str, DNNProfile],
    effective_es_gflops: Sequence[float],
    queue_wait_times_s: Sequence[float],
    slot_state: OnlineSlotState,
    weights: CostWeights,
    method: str,
    compression_ratio: float,
) -> AssignmentResult:
    selected: List[PartitionResult] = []
    md_to_server: List[int] = []
    for local_idx, md_idx in enumerate(active_md_ids):
        planned = planned_assignment.selected[local_idx]
        server_slot = int(planned_assignment.md_to_server[local_idx])
        profile = profiles[scenario.md_models[md_idx].lower()]
        wait_time = float(queue_wait_times_s[server_slot]) if planned.cloud_nodes else 0.0
        selected.append(
            evaluate_partition(
                profile,
                local_ids=planned.local_nodes,
                cloud_ids=planned.cloud_nodes,
                md_gflops=scenario.md_gflops[md_idx],
                es_gflops=effective_es_gflops[server_slot] if server_slot >= 0 else 1.0,
                local_power_w=scenario.local_powers[md_idx],
                upload_power_w=scenario.upload_power_w,
                bandwidth_up_mb_s=slot_state.bandwidth_up_mb_s,
                bandwidth_down_mb_s=slot_state.bandwidth_down_mb_s,
                weights=weights,
                method=method,
                server_wait_time_s=wait_time,
                compression_ratio=compression_ratio,
            )
        )
        md_to_server.append(server_slot)
    return _assignment_from_selected(md_to_server, selected)


def _update_queue_waits(
    queue_wait_times_s: Sequence[float],
    assignment: AssignmentResult,
    slot_duration_s: float,
) -> Tuple[float, ...]:
    demand_by_slot: Dict[int, float] = {slot: 0.0 for slot in range(len(queue_wait_times_s))}
    for server_slot, result in zip(assignment.md_to_server, assignment.selected):
        if server_slot >= 0 and result.cloud_nodes:
            demand_by_slot[server_slot] += result.server_time
    return tuple(
        max(0.0, float(queue_wait_times_s[slot]) + demand_by_slot[slot] - slot_duration_s)
        for slot in range(len(queue_wait_times_s))
    )


def _record_policy_switch(
    stats: dict,
    partition_label: str,
    server_slot: int,
) -> None:
    stats["active_slots"] += 1
    if stats["last_partition"] is not None:
        partition_changed = partition_label != stats["last_partition"]
        server_changed = server_slot != stats["last_server"]
        if partition_changed:
            stats["partition_switches"] += 1
        if server_changed:
            stats["server_switches"] += 1
        if partition_changed or server_changed:
            stats["policy_switches"] += 1
    stats["last_partition"] = partition_label
    stats["last_server"] = server_slot


def run_online_scenario(
    online_scenario: OnlineScenario,
    profiles: Mapping[str, DNNProfile],
    weights: CostWeights,
    compression_ratio: float,
) -> tuple[List[dict], List[dict]]:
    scenario = online_scenario.base_scenario
    avg_up = mean(slot.bandwidth_up_mb_s for slot in online_scenario.slots)
    avg_down = mean(slot.bandwidth_down_mb_s for slot in online_scenario.slots)
    avg_load = tuple(
        mean(slot.external_load_factors[slot_idx] for slot in online_scenario.slots)
        for slot_idx in range(len(scenario.es_slots_gflops))
    )
    static_es_gflops = _effective_server_gflops(scenario.es_slots_gflops, avg_load)
    static_matrix = build_result_matrix(
        profiles,
        scenario.md_models,
        scenario.md_gflops,
        static_es_gflops,
        scenario.local_powers,
        scenario.upload_power_w,
        avg_up,
        method="min-cut",
        bandwidth_down_mb_s=avg_down,
        weights=weights,
        server_wait_times_s=[0.0] * len(scenario.es_slots_gflops),
        compression_ratio=compression_ratio,
    )

    queue_by_method: Dict[str, Tuple[float, ...]] = {
        method: tuple(0.0 for _ in scenario.es_slots_gflops) for method in ONLINE_METHOD_ORDER
    }
    switch_stats: Dict[str, List[dict]] = {
        method: [
            {
                "active_slots": 0,
                "partition_switches": 0,
                "server_switches": 0,
                "policy_switches": 0,
                "last_partition": None,
                "last_server": None,
            }
            for _ in scenario.md_models
        ]
        for method in ONLINE_METHOD_ORDER
    }
    timeseries_rows: List[dict] = []

    for slot_state in online_scenario.slots:
        active_md_ids = tuple(slot_state.active_md_ids)
        effective_es_gflops = _effective_server_gflops(scenario.es_slots_gflops, slot_state.external_load_factors)
        for method in ONLINE_METHOD_ORDER:
            current_queue = queue_by_method[method]
            if method == "Online Only-Local":
                assignment = _online_local_only_assignment(
                    scenario,
                    active_md_ids,
                    profiles,
                    slot_state,
                    weights,
                    compression_ratio,
                )
            elif method == "Online Only-Server":
                matrix = build_result_matrix(
                    profiles,
                    scenario.md_models,
                    scenario.md_gflops,
                    effective_es_gflops,
                    scenario.local_powers,
                    scenario.upload_power_w,
                    slot_state.bandwidth_up_mb_s,
                    method="server-only",
                    bandwidth_down_mb_s=slot_state.bandwidth_down_mb_s,
                    weights=weights,
                    server_wait_times_s=current_queue,
                    compression_ratio=compression_ratio,
                )
                assignment = assign_min_cost(_select_active_rows(matrix, active_md_ids))
            elif method == "Static MinCut+BMatch":
                planned = assign_min_cost(_select_active_rows(static_matrix, active_md_ids))
                assignment = _re_evaluate_planned_assignment(
                    scenario,
                    active_md_ids,
                    planned,
                    profiles,
                    effective_es_gflops,
                    current_queue,
                    slot_state,
                    weights,
                    method="static-mincut-bmatch",
                    compression_ratio=compression_ratio,
                )
            elif method == "Online MinCut+BMatch":
                matrix = build_result_matrix(
                    profiles,
                    scenario.md_models,
                    scenario.md_gflops,
                    effective_es_gflops,
                    scenario.local_powers,
                    scenario.upload_power_w,
                    slot_state.bandwidth_up_mb_s,
                    method="min-cut",
                    bandwidth_down_mb_s=slot_state.bandwidth_down_mb_s,
                    weights=weights,
                    server_wait_times_s=current_queue,
                    compression_ratio=compression_ratio,
                )
                assignment = assign_min_cost(_select_active_rows(matrix, active_md_ids))
            else:
                raise ValueError(f"unsupported online method: {method}")

            avg_task_queue_wait = (sum(result.wait_time for result in assignment.selected) / len(assignment.selected)) if assignment.selected else 0.0
            avg_queue_backlog = mean(current_queue) if current_queue else 0.0
            max_queue_backlog = max(current_queue) if current_queue else 0.0
            timeseries_rows.append(
                {
                    "experiment": "online_extension",
                    "scenario": online_scenario.name,
                    "slot": slot_state.slot,
                    "method": method,
                    "compression_ratio": compression_ratio,
                    "active_md_count": len(active_md_ids),
                    "bandwidth_up_mb_s": slot_state.bandwidth_up_mb_s,
                    "bandwidth_down_mb_s": slot_state.bandwidth_down_mb_s,
                    "total_cost": assignment.total_cost,
                    "total_latency": assignment.total_latency,
                    "total_energy": assignment.total_energy,
                    "avg_task_queue_wait": avg_task_queue_wait,
                    "avg_queue_backlog": avg_queue_backlog,
                    "max_queue_backlog": max_queue_backlog,
                    "mean_external_load": mean(slot_state.external_load_factors) if slot_state.external_load_factors else 0.0,
                }
            )
            for local_idx, md_idx in enumerate(active_md_ids):
                result = assignment.selected[local_idx]
                _record_policy_switch(
                    switch_stats[method][md_idx],
                    result.partition_label,
                    int(assignment.md_to_server[local_idx]),
                )
            queue_by_method[method] = _update_queue_waits(current_queue, assignment, online_scenario.slot_duration_s)

    policy_rows: List[dict] = []
    for method, md_stats in switch_stats.items():
        for md_idx, stats in enumerate(md_stats):
            active_slots = int(stats["active_slots"])
            if active_slots == 0:
                continue
            transitions = max(active_slots - 1, 1)
            policy_rows.append(
                {
                    "experiment": "online_extension",
                    "scenario": online_scenario.name,
                    "method": method,
                    "md": md_idx,
                    "active_slots": active_slots,
                    "partition_switches": int(stats["partition_switches"]),
                    "server_switches": int(stats["server_switches"]),
                    "policy_switches": int(stats["policy_switches"]),
                    "policy_switch_rate": float(stats["policy_switches"]) / transitions,
                }
            )
    return timeseries_rows, policy_rows


def _single_pair_summary_row(
    experiment: str,
    scenario: str,
    model: str,
    bandwidth: float,
    beta: float,
    compression_ratio: float,
    md_gflops: float,
    es_gflops: float,
    local_power_w: float,
    upload_power_w: float,
    method: str,
    result: "PartitionResult",
    runtime_ms: float,
    runtime_ms_std: float,
) -> dict:
    return {
        "experiment": experiment,
        "scenario": scenario,
        "model": model,
        "bandwidth_mb_s": bandwidth,
        "beta": beta,
        "compression_ratio": compression_ratio,
        "method": method,
        "md_gflops": md_gflops,
        "es_gflops": es_gflops,
        "local_power_w": local_power_w,
        "upload_power_w": upload_power_w,
        "cost": result.cost,
        "latency": result.latency,
        "energy": result.energy,
        "local_time": result.local_time,
        "upload_time": result.upload_time,
        "wait_time": result.wait_time,
        "server_time": result.server_time,
        "down_time": result.down_time,
        "partition": result.partition_label,
        "transmitted_node_count": len(result.transmitted_nodes),
        "local_node_count": len(result.local_nodes),
        "cloud_node_count": len(result.cloud_nodes),
        "runtime_ms": runtime_ms,
        "runtime_ms_std": runtime_ms_std,
    }


def _single_pair_focus_scan_row(
    model: str,
    method: str,
    x_field: str,
    x_value: float,
    result: PartitionResult,
) -> dict:
    return {
        "model": model,
        x_field: x_value,
        "method": method,
        "cost": result.cost,
        "latency": result.latency,
        "energy": result.energy,
        "partition": result.partition_label,
    }


def _single_pair_bar_metric_row(model: str, method: str, result: PartitionResult) -> dict:
    return {
        "model": model,
        "method": method,
        "cost": result.cost,
        "latency": result.latency,
        "energy": result.energy,
    }


def _single_pair_gain_row(model: str, bandwidth_mb_s: float, md_gflops: float, results: Mapping[str, PartitionResult]) -> dict:
    endpoint_method, endpoint_result = min(
        (
            ("Only-Local", results["Only-Local"]),
            ("Only-Server", results["Only-Server"]),
        ),
        key=lambda item: item[1].cost,
    )
    mincut_result = results["MinCut"]
    endpoint_cost = endpoint_result.cost
    gain = 0.0 if endpoint_cost <= 0 else (endpoint_cost - mincut_result.cost) / endpoint_cost
    if gain < -1e-9:
        raise ValueError(
            f"negative gain detected for {model} at bandwidth={bandwidth_mb_s}, md_gflops={md_gflops}: "
            f"endpoint={endpoint_cost}, mincut={mincut_result.cost}"
        )
    return {
        "model": model,
        "bandwidth_mb_s": bandwidth_mb_s,
        "md_gflops": md_gflops,
        "gain": max(gain, 0.0),
        "mincut_cost": mincut_result.cost,
        "endpoint_cost": endpoint_cost,
        "best_endpoint": endpoint_method,
        "partition": mincut_result.partition_label,
    }


def write_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "bandwidth_mb_s",
        "beta",
        "compression_ratio",
        "method",
        "repeat_count",
        "total_cost",
        "total_cost_std",
        "total_latency",
        "total_latency_std",
        "total_energy",
        "total_energy_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_pair_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "model",
        "bandwidth_mb_s",
        "beta",
        "compression_ratio",
        "method",
        "md_gflops",
        "es_gflops",
        "local_power_w",
        "upload_power_w",
        "cost",
        "latency",
        "energy",
        "local_time",
        "upload_time",
        "wait_time",
        "server_time",
        "down_time",
        "partition",
        "transmitted_node_count",
        "local_node_count",
        "cloud_node_count",
        "runtime_ms",
        "runtime_ms_std",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_pair_runtime_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "model",
        "method",
        "nodes",
        "edges",
        "runtime_ms",
        "runtime_ms_std",
        "cost",
        "latency",
        "energy",
        "partition",
        "transmitted_node_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_pair_bandwidth_sweep_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "bandwidth_mb_s", "method", "cost", "latency", "energy", "partition"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_pair_local_resource_sweep_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "md_gflops", "method", "cost", "latency", "energy", "partition"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_pair_bar_metrics_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "method", "cost", "latency", "energy"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_single_pair_gain_heatmap_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "bandwidth_mb_s",
        "md_gflops",
        "gain",
        "mincut_cost",
        "endpoint_cost",
        "best_endpoint",
        "partition",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_random_repeats_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "bandwidth_mb_s",
        "beta",
        "compression_ratio",
        "method",
        "repeat",
        "total_cost",
        "total_latency",
        "total_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _online_summary_rows(
    timeseries_rows: Sequence[dict],
    policy_rows: Sequence[dict],
    compression_ratio: float,
) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in timeseries_rows:
        grouped.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)
    policy_by_key: Dict[Tuple[str, str], List[dict]] = {}
    for row in policy_rows:
        policy_by_key.setdefault((str(row["scenario"]), str(row["method"])), []).append(row)

    summary_rows: List[dict] = []
    for (scenario, method), rows in grouped.items():
        costs = [float(row["total_cost"]) for row in rows]
        latencies = [float(row["total_latency"]) for row in rows]
        energies = [float(row["total_energy"]) for row in rows]
        queue_waits = [float(row["avg_task_queue_wait"]) for row in rows]
        queue_backlogs = [float(row["avg_queue_backlog"]) for row in rows]
        active_counts = [float(row["active_md_count"]) for row in rows]
        policy_for_key = policy_by_key.get((scenario, method), [])
        switch_rates = [float(row["policy_switch_rate"]) for row in policy_for_key]
        summary_rows.append(
            {
                "experiment": "online_extension",
                "scenario": scenario,
                "method": method,
                "compression_ratio": compression_ratio,
                "slots": len(rows),
                "avg_active_md_count": mean(active_counts) if active_counts else 0.0,
                "avg_total_cost": mean(costs) if costs else 0.0,
                "std_total_cost": pstdev(costs) if len(costs) > 1 else 0.0,
                "avg_total_latency": mean(latencies) if latencies else 0.0,
                "avg_total_energy": mean(energies) if energies else 0.0,
                "avg_task_queue_wait": mean(queue_waits) if queue_waits else 0.0,
                "avg_queue_backlog": mean(queue_backlogs) if queue_backlogs else 0.0,
                "max_queue_backlog": max(float(row["max_queue_backlog"]) for row in rows) if rows else 0.0,
                "policy_switch_rate": mean(switch_rates) if switch_rates else 0.0,
            }
        )
    return summary_rows


def write_online_summary_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "method",
        "compression_ratio",
        "slots",
        "avg_active_md_count",
        "avg_total_cost",
        "std_total_cost",
        "avg_total_latency",
        "avg_total_energy",
        "avg_task_queue_wait",
        "avg_queue_backlog",
        "max_queue_backlog",
        "policy_switch_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_online_timeseries_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "slot",
        "method",
        "compression_ratio",
        "active_md_count",
        "bandwidth_up_mb_s",
        "bandwidth_down_mb_s",
        "total_cost",
        "total_latency",
        "total_energy",
        "avg_task_queue_wait",
        "avg_queue_backlog",
        "max_queue_backlog",
        "mean_external_load",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_online_policy_switches_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "scenario",
        "method",
        "md",
        "active_slots",
        "partition_switches",
        "server_switches",
        "policy_switches",
        "policy_switch_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_model_topology(path: Path, profiles: Mapping[str, DNNProfile]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "nodes", "edges", "merge_nodes", "branch_nodes", "partitionable_nodes", "total_gflops", "input_mb", "output_mb"])
        for name, profile in sorted(profiles.items()):
            indegree: Dict[str, int] = {node_id: 0 for node_id in profile.layer_ids}
            outdegree: Dict[str, int] = {node_id: 0 for node_id in profile.layer_ids}
            for src, dst in profile.edges:
                if dst in indegree:
                    indegree[dst] += 1
                if src in outdegree:
                    outdegree[src] += 1
            writer.writerow(
                [
                    name,
                    len(profile.layers),
                    len(profile.edges),
                    sum(1 for value in indegree.values() if value > 1),
                    sum(1 for value in outdegree.values() if value > 1),
                    sum(1 for layer in profile.layers if layer.partitionable),
                    sum(layer.flops for layer in profile.layers) / 1e9,
                    profile.input_bytes / (1024 * 1024),
                    profile.output_bytes / (1024 * 1024),
                ]
            )


def _average_runtime_ms(func, repeats: int = 3) -> Tuple[float, float]:
    values: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        values.append((time.perf_counter() - start) * 1000.0)
    runtime_mean, runtime_std = _mean_std(values)
    return runtime_mean, (runtime_std if repeats > 1 else 0.0)


def write_algorithm_runtime_csv(
    path: Path,
    profiles: Mapping[str, DNNProfile],
    weights: CostWeights,
    compression_ratio: float,
) -> List[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    method_funcs = {
        "MinCut": min_cut_partition,
        "ChainTopo": best_chain_topo_partition,
        "GreedyCut": greedy_cut_partition,
    }
    for model_name, profile in sorted(profiles.items()):
        for method, func in method_funcs.items():
            def run_once(func=func, profile=profile):
                return func(
                    profile,
                    md_gflops=3.0,
                    es_gflops=35.0,
                    local_power_w=2.0,
                    upload_power_w=1.0,
                    bandwidth_up_mb_s=1.0,
                    weights=weights,
                    compression_ratio=compression_ratio,
                )

            runtime_mean, runtime_std = _average_runtime_ms(run_once, repeats=3)
            result = run_once()
            rows.append(
                {
                    "model": model_name,
                    "method": method,
                    "nodes": len(profile.layers),
                    "edges": len(profile.edges),
                    "runtime_ms": runtime_mean,
                    "runtime_ms_std": runtime_std,
                    "cost": result.cost,
                    "partition": result.partition_label,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "method", "nodes", "edges", "runtime_ms", "runtime_ms_std", "cost", "partition"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _synthetic_scenario(md_count: int, es_count: int) -> Scenario:
    model_cycle = MAIN_MODELS
    md_models = tuple(model_cycle[i % len(model_cycle)] for i in range(md_count))
    md_gflops = tuple(1.2 + (i % 10) * 0.8 for i in range(md_count))
    local_powers = tuple(1.2 + (i % 10) * 0.18 for i in range(md_count))
    es_slots = tuple(20.0 + (i % 10) * 3.5 for i in range(es_count))
    return Scenario(
        name=f"scale_m{md_count}_s{es_count}",
        md_models=md_models,
        md_gflops=md_gflops,
        es_slots_gflops=es_slots,
        local_powers=local_powers,
    )


def write_scalability_runtime_csv(
    path: Path,
    profiles: Mapping[str, DNNProfile],
    weights: CostWeights,
    compression_ratio: float,
) -> List[dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for md_count, es_count in ((5, 5), (10, 10), (10, 20), (20, 20), (20, 50), (50, 50)):
        scenario = _synthetic_scenario(md_count, es_count)
        start = time.perf_counter()
        matrix = build_result_matrix(
            profiles,
            scenario.md_models,
            scenario.md_gflops,
            scenario.es_slots_gflops,
            scenario.local_powers,
            scenario.upload_power_w,
            1.0,
            method="min-cut",
            weights=weights,
            compression_ratio=compression_ratio,
        )
        matrix_ms = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        assignment = assign_min_cost(matrix)
        matching_ms = (time.perf_counter() - start) * 1000.0
        rows.append(
            {
                "md_count": md_count,
                "es_slots": es_count,
                "matrix_pairs": md_count * es_count,
                "matrix_build_ms": matrix_ms,
                "hungarian_ms": matching_ms,
                "total_runtime_ms": matrix_ms + matching_ms,
                "total_cost": assignment.total_cost,
            }
        )

    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["md_count", "es_slots", "matrix_pairs", "matrix_build_ms", "hungarian_ms", "total_runtime_ms", "total_cost"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot_lines(
    path: Path,
    rows: Sequence[dict],
    experiment: str,
    scenario: str,
    x_field: str,
    xlabel: str,
    title: str,
    method_order: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    for method in method_order:
        selected = [row for row in rows if row["experiment"] == experiment and row["scenario"] == scenario and row["method"] == method]
        selected = sorted(selected, key=lambda row: float(row[x_field]))
        if not selected:
            continue
        xs = [row[x_field] for row in selected]
        ys = [row["total_cost"] for row in selected]
        yerr = [row["total_cost_std"] for row in selected]
        if any(value > 0 for value in yerr):
            plt.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=method)
        else:
            plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel(xlabel)
    plt.ylabel("total cost")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_online_over_time(
    path: Path,
    rows: Sequence[dict],
    scenario: str,
    y_field: str,
    ylabel: str,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.4, 4.8))
    for method in ONLINE_METHOD_ORDER:
        selected = [row for row in rows if row["scenario"] == scenario and row["method"] == method]
        selected = sorted(selected, key=lambda row: int(row["slot"]))
        if not selected:
            continue
        xs = [int(row["slot"]) for row in selected]
        ys = [float(row[y_field]) for row in selected]
        plt.plot(xs, ys, marker="o", label=method)
    plt.xlabel("time slot")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_bars(path: Path, rows: Sequence[dict], experiment: str, scenario: str, title: str, method_order: Sequence[str]) -> None:
    selected = [row for row in rows if row["experiment"] == experiment and row["scenario"] == scenario]
    labels: List[str] = []
    values: List[float] = []
    errors: List[float] = []
    for method in method_order:
        hit = next((row for row in selected if row["method"] == method), None)
        if hit is not None:
            labels.append(method)
            values.append(hit["total_cost"])
            errors.append(hit["total_cost_std"])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.bar(labels, values, yerr=errors, capsize=3)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("total cost")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_single_pair_models(path: Path, rows: Sequence[dict], experiment: str, scenario: str, title: str) -> None:
    models = [model for model in SINGLE_PAIR_MODELS if any(row["experiment"] == experiment and row["scenario"] == scenario and row["model"] == model for row in rows)]
    width = 0.15
    x_positions = list(range(len(models)))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10.8, 5.2))
    for offset, method in enumerate(SINGLE_PAIR_METHOD_ORDER):
        ys = []
        for model in models:
            hit = next(
                (
                    row
                    for row in rows
                    if row["experiment"] == experiment and row["scenario"] == scenario and row["model"] == model and row["method"] == method
                ),
                None,
            )
            ys.append(float(hit["cost"]) if hit else 0.0)
        xs = [x + (offset - 2) * width for x in x_positions]
        plt.bar(xs, ys, width=width, label=method)
    plt.xticks(x_positions, models, rotation=25, ha="right")
    plt.ylabel("pairwise cost")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_single_pair_line_grid(
    path: Path,
    rows: Sequence[dict],
    x_field: str,
    xlabel: str,
    title: str,
) -> None:
    models = [model for model in SINGLE_PAIR_MODELS if any(str(row["model"]) == model for row in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4), sharey=False)
    axes_list = list(axes.flat)
    for idx, model in enumerate(models):
        ax = axes_list[idx]
        selected = [row for row in rows if str(row["model"]) == model]
        for method in SINGLE_PAIR_FOCUS_METHOD_ORDER:
            method_rows = [row for row in selected if str(row["method"]) == method]
            method_rows = sorted(method_rows, key=lambda row: float(row[x_field]))
            xs = [float(row[x_field]) for row in method_rows]
            ys = [float(row["cost"]) for row in method_rows]
            ax.plot(
                xs,
                ys,
                marker="o",
                color=MODULE1_METHOD_COLORS[method],
                label=method,
            )
        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model))
        ax.set_xlabel(xlabel)
        ax.set_ylabel("objective cost")
        ax.grid(True, alpha=0.25)
    for ax in axes_list[len(models) :]:
        ax.axis("off")
    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_single_pair_metric_triptych(path: Path, rows: Sequence[dict], title: str) -> None:
    models = [model for model in SINGLE_PAIR_MODELS if any(str(row["model"]) == model for row in rows)]
    metrics = (
        ("latency", "latency"),
        ("energy", "energy"),
        ("cost", "objective cost"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    width = 0.22
    x_positions = list(range(len(models)))
    for ax, (field, ylabel) in zip(axes, metrics):
        for offset, method in enumerate(SINGLE_PAIR_FOCUS_METHOD_ORDER):
            ys = []
            for model in models:
                hit = next(
                    (row for row in rows if str(row["model"]) == model and str(row["method"]) == method),
                    None,
                )
                ys.append(float(hit[field]) if hit else 0.0)
            xs = [x + (offset - 1) * width for x in x_positions]
            ax.bar(xs, ys, width=width, color=MODULE1_METHOD_COLORS[method], label=method)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([MODEL_DISPLAY_NAMES.get(model, model) for model in models], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, y=0.985)
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_single_pair_gain_heatmaps(path: Path, rows: Sequence[dict], title: str) -> None:
    models = [model for model in SINGLE_PAIR_MODELS if any(str(row["model"]) == model for row in rows)]
    bandwidths = sorted({float(row["bandwidth_mb_s"]) for row in rows})
    md_gflops_values = sorted({float(row["md_gflops"]) for row in rows})
    max_gain = max(float(row["gain"]) for row in rows) if rows else 1.0
    norm = Normalize(vmin=0.0, vmax=max(max_gain, 1e-9))

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), sharex=False, sharey=False)
    axes_list = list(axes.flat)
    last_im = None
    for idx, model in enumerate(models):
        ax = axes_list[idx]
        model_rows = [row for row in rows if str(row["model"]) == model]
        gain_matrix: List[List[float]] = []
        for md_gflops in md_gflops_values:
            gain_matrix.append(
                [
                    float(
                        next(
                            row["gain"]
                            for row in model_rows
                            if float(row["md_gflops"]) == md_gflops and float(row["bandwidth_mb_s"]) == bandwidth
                        )
                    )
                    for bandwidth in bandwidths
                ]
            )
        last_im = ax.imshow(gain_matrix, aspect="auto", origin="lower", cmap="YlGn", norm=norm)
        ax.set_title(MODEL_DISPLAY_NAMES.get(model, model))
        ax.set_xticks(range(len(bandwidths)))
        ax.set_xticklabels([str(bandwidth) for bandwidth in bandwidths], rotation=25, ha="right")
        ax.set_yticks(range(len(md_gflops_values)))
        ax.set_yticklabels([str(value) for value in md_gflops_values])
        ax.set_xlabel("uplink bandwidth (MB/s)")
        ax.set_ylabel("local GFLOPS")
    for ax in axes_list[len(models) :]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 0.94, 0.95))
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_list[: len(models)], shrink=0.92, label="gain", pad=0.02)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_runtime_by_model(path: Path, rows: Sequence[dict]) -> None:
    models = [model for model in SINGLE_PAIR_MODELS if any(str(row["model"]) == model and row["method"] == "MinCut" for row in rows)]
    methods = ["MinCut", "ChainTopo", "GreedyCut"]
    width = 0.24
    x_positions = list(range(len(models)))
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9.2, 5.0))
    for offset, method in enumerate(methods):
        ys = []
        yerr = []
        for model in models:
            hit = next((row for row in rows if row["model"] == model and row["method"] == method), None)
            ys.append(float(hit["runtime_ms"]) if hit else 0.0)
            yerr.append(float(hit["runtime_ms_std"]) if hit else 0.0)
        xs = [x + (offset - 1) * width for x in x_positions]
        plt.bar(xs, ys, width=width, yerr=yerr, capsize=2, label=method)
    plt.xticks(x_positions, models, rotation=25, ha="right")
    plt.ylabel("runtime (ms)")
    plt.title("Partition algorithm runtime by model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_scalability_runtime(path: Path, rows: Sequence[dict]) -> None:
    labels = [f"{row['md_count']}x{row['es_slots']}" for row in rows]
    ys = [float(row["total_runtime_ms"]) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(labels, ys, marker="o")
    plt.xlabel("MD count x ES slots")
    plt.ylabel("runtime (ms)")
    plt.title("MinCut+BMatch scalability")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _append_results(
    summary_rows: List[dict],
    repeat_rows: List[dict],
    experiment: str,
    scenario: Scenario,
    bandwidth: float,
    weights: CostWeights,
    compression_ratio: float,
    results: Mapping[str, Sequence[AssignmentResult]],
    method_order: Sequence[str],
) -> None:
    for method in method_order:
        assignments = list(results[method])
        summary_rows.append(
            _summary_row(
                experiment,
                scenario.name,
                bandwidth,
                weights.beta,
                compression_ratio,
                method,
                assignments,
            )
        )
        if len(assignments) > 1:
            repeat_rows.extend(
                _repeat_rows(
                    experiment,
                    scenario.name,
                    bandwidth,
                    weights.beta,
                    compression_ratio,
                    method,
                    assignments,
                )
            )


def _cleanup_stale_outputs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The desktop sandbox allows overwriting files in-place but may deny deletes.
    # Keep reruns robust by relying on explicit overwrites and recommending a new
    # output directory when a perfectly clean artifact set is required.


def run_all(
    output_dir: Path,
    weights: CostWeights,
    random_repeats: int = 30,
    compression_ratio: float = DEFAULT_COMPRESSION_RATIO,
) -> None:
    scenarios = [mixed5_scenario(), googlenet_stress_scenario()]
    model_names = sorted(set(required_models(scenarios)) | set(MAIN_MODELS))
    profiles = build_profiles(model_names)
    _cleanup_stale_outputs(output_dir)
    write_model_topology(output_dir / "model_topology.csv", profiles)

    single_pair_rows: List[dict] = []
    single_pair_runtime_rows: List[dict] = []
    single_pair_bandwidth_rows: List[dict] = []
    single_pair_local_rows: List[dict] = []
    single_pair_bar_rows: List[dict] = []
    single_pair_gain_rows: List[dict] = []
    standard_pair = standard_single_pair_kwargs()

    single_pair_cache: Dict[Tuple[str, float, float], Tuple[Dict[str, PartitionResult], List[dict]]] = {}

    def cached_single_pair(model_name: str, md_gflops: float, bandwidth_up_mb_s: float) -> tuple[Dict[str, PartitionResult], List[dict]]:
        key = (model_name, float(md_gflops), float(bandwidth_up_mb_s))
        cached = single_pair_cache.get(key)
        if cached is None:
            cached = run_single_pair_methods(
                profiles[model_name],
                md_gflops=md_gflops,
                es_gflops=standard_pair["es_gflops"],
                local_power_w=standard_pair["local_power_w"],
                upload_power_w=standard_pair["upload_power_w"],
                bandwidth_up_mb_s=bandwidth_up_mb_s,
                bandwidth_down_mb_s=standard_pair["bandwidth_down_mb_s"],
                weights=weights,
                compression_ratio=compression_ratio,
            )
            single_pair_cache[key] = cached
        return cached

    for model_name in SINGLE_PAIR_MODELS:
        results, runtime_rows = cached_single_pair(model_name, standard_pair["md_gflops"], standard_pair["bandwidth_up_mb_s"])
        runtime_by_method = {row["method"]: row for row in runtime_rows}
        for method in SINGLE_PAIR_METHOD_ORDER:
            single_pair_rows.append(
                _single_pair_summary_row(
                    experiment="single_pair_partition",
                    scenario="standard_pair",
                    model=model_name,
                    bandwidth=standard_pair["bandwidth_up_mb_s"],
                    beta=weights.beta,
                    compression_ratio=compression_ratio,
                    md_gflops=standard_pair["md_gflops"],
                    es_gflops=standard_pair["es_gflops"],
                    local_power_w=standard_pair["local_power_w"],
                    upload_power_w=standard_pair["upload_power_w"],
                    method=method,
                    result=results[method],
                    runtime_ms=runtime_by_method[method]["runtime_ms"],
                    runtime_ms_std=runtime_by_method[method]["runtime_ms_std"],
                )
            )
        for row in runtime_rows:
            single_pair_runtime_rows.append({"experiment": "single_pair_partition", "scenario": "standard_pair", **row})
        for method in SINGLE_PAIR_FOCUS_METHOD_ORDER:
            single_pair_bar_rows.append(_single_pair_bar_metric_row(model_name, method, results[method]))

        for bandwidth in MODULE1_BANDWIDTHS:
            bandwidth_results, _ = cached_single_pair(model_name, standard_pair["md_gflops"], bandwidth)
            for method in SINGLE_PAIR_FOCUS_METHOD_ORDER:
                single_pair_bandwidth_rows.append(
                    _single_pair_focus_scan_row(
                        model_name,
                        method,
                        "bandwidth_mb_s",
                        bandwidth,
                        bandwidth_results[method],
                    )
                )

        for md_gflops in MODULE1_LOCAL_GFLOPS:
            local_results, _ = cached_single_pair(model_name, md_gflops, standard_pair["bandwidth_up_mb_s"])
            for method in SINGLE_PAIR_FOCUS_METHOD_ORDER:
                single_pair_local_rows.append(
                    _single_pair_focus_scan_row(
                        model_name,
                        method,
                        "md_gflops",
                        md_gflops,
                        local_results[method],
                    )
                )

        for md_gflops in MODULE1_LOCAL_GFLOPS:
            for bandwidth in MODULE1_BANDWIDTHS:
                gain_results, _ = cached_single_pair(model_name, md_gflops, bandwidth)
                single_pair_gain_rows.append(_single_pair_gain_row(model_name, bandwidth, md_gflops, gain_results))

    stress_pair = googlenet_stress_pair_kwargs()
    stress_results, stress_runtime_rows = run_single_pair_methods(
        profiles["googlenet"],
        **stress_pair,
        weights=weights,
        compression_ratio=compression_ratio,
    )
    stress_runtime_by_method = {row["method"]: row for row in stress_runtime_rows}
    for method in SINGLE_PAIR_METHOD_ORDER:
        single_pair_rows.append(
            _single_pair_summary_row(
                experiment="single_pair_topology_stress",
                scenario="googlenet_pair_stress",
                model="googlenet",
                bandwidth=stress_pair["bandwidth_up_mb_s"],
                beta=weights.beta,
                compression_ratio=compression_ratio,
                md_gflops=stress_pair["md_gflops"],
                es_gflops=stress_pair["es_gflops"],
                local_power_w=stress_pair["local_power_w"],
                upload_power_w=stress_pair["upload_power_w"],
                method=method,
                result=stress_results[method],
                runtime_ms=stress_runtime_by_method[method]["runtime_ms"],
                runtime_ms_std=stress_runtime_by_method[method]["runtime_ms_std"],
            )
        )
    for row in stress_runtime_rows:
        single_pair_runtime_rows.append({"experiment": "single_pair_topology_stress", "scenario": "googlenet_pair_stress", **row})

    write_single_pair_summary_csv(output_dir / "single_pair_summary.csv", single_pair_rows)
    write_single_pair_runtime_csv(output_dir / "single_pair_runtime.csv", single_pair_runtime_rows)
    write_single_pair_bandwidth_sweep_csv(output_dir / "single_pair_bandwidth_sweep.csv", single_pair_bandwidth_rows)
    write_single_pair_local_resource_sweep_csv(output_dir / "single_pair_local_resource_sweep.csv", single_pair_local_rows)
    write_single_pair_bar_metrics_csv(output_dir / "single_pair_bar_metrics.csv", single_pair_bar_rows)
    write_single_pair_gain_heatmap_csv(output_dir / "single_pair_gain_heatmap.csv", single_pair_gain_rows)

    summary_rows: List[dict] = []
    repeat_rows: List[dict] = []

    mixed = mixed5_scenario()
    mixed_results = run_multiuser_methods(
        mixed,
        profiles,
        1.0,
        weights,
        random_repeats,
        seed_base=1100,
        compression_ratio=compression_ratio,
    )
    _append_results(
        summary_rows,
        repeat_rows,
        "main_comparison",
        mixed,
        1.0,
        weights,
        compression_ratio,
        mixed_results,
        MULTIUSER_METHOD_ORDER,
    )
    for method in MULTIUSER_METHOD_ORDER:
        safe_method = method.replace("+", "_").replace("-", "_")
        write_assignment_csv(output_dir / f"mixed5_{safe_method}.csv", mixed_results[method][0], mixed.md_models)

    for bandwidth_idx, bandwidth in enumerate(BANDWIDTHS):
        results = run_multiuser_methods(
            mixed,
            profiles,
            bandwidth,
            weights,
            random_repeats,
            seed_base=3000 + bandwidth_idx * 100,
            compression_ratio=compression_ratio,
        )
        _append_results(
            summary_rows,
            repeat_rows,
            "bandwidth_sensitivity",
            mixed,
            bandwidth,
            weights,
            compression_ratio,
            results,
            MULTIUSER_METHOD_ORDER,
        )

    for beta_idx, beta in enumerate(BETAS):
        beta_weights = CostWeights(alpha=weights.alpha, beta=beta)
        results = run_multiuser_methods(
            mixed,
            profiles,
            1.0,
            beta_weights,
            random_repeats,
            seed_base=7000 + beta_idx * 100,
            compression_ratio=compression_ratio,
        )
        _append_results(
            summary_rows,
            repeat_rows,
            "beta_sensitivity",
            mixed,
            1.0,
            beta_weights,
            compression_ratio,
            results,
            MULTIUSER_METHOD_ORDER,
        )

    write_summary_csv(output_dir / "summary.csv", summary_rows)
    write_random_repeats_csv(output_dir / "random_repeats.csv", repeat_rows)
    runtime_rows = write_algorithm_runtime_csv(
        output_dir / "algorithm_runtime.csv",
        profiles,
        weights,
        compression_ratio=compression_ratio,
    )
    scalability_rows = write_scalability_runtime_csv(
        output_dir / "scalability_runtime.csv",
        profiles,
        weights,
        compression_ratio=compression_ratio,
    )
    online_timeseries_rows: List[dict] = []
    online_policy_rows: List[dict] = []
    for online_scenario in (
        online_bandwidth_trace_scenario(),
        online_burst_load_scenario(),
        online_composite_nonstationary_scenario(),
    ):
        scenario_timeseries_rows, scenario_policy_rows = run_online_scenario(
            online_scenario,
            profiles,
            weights,
            compression_ratio,
        )
        online_timeseries_rows.extend(scenario_timeseries_rows)
        online_policy_rows.extend(scenario_policy_rows)
    online_summary_rows = _online_summary_rows(
        online_timeseries_rows,
        online_policy_rows,
        compression_ratio=compression_ratio,
    )
    write_online_summary_csv(output_dir / "online_summary.csv", online_summary_rows)
    write_online_timeseries_csv(output_dir / "online_timeseries.csv", online_timeseries_rows)
    write_online_policy_switches_csv(output_dir / "online_policy_switches.csv", online_policy_rows)

    _plot_single_pair_models(output_dir / "single_pair_cost.png", single_pair_rows, "single_pair_partition", "standard_pair", "Single-pair partition comparison")
    _plot_single_pair_models(
        output_dir / "single_pair_topology_stress.png",
        single_pair_rows,
        "single_pair_topology_stress",
        "googlenet_pair_stress",
        "Single-pair topology stress on GoogLeNet",
    )
    _plot_single_pair_line_grid(
        output_dir / "single_pair_bandwidth_lines.png",
        single_pair_bandwidth_rows,
        "bandwidth_mb_s",
        "uplink bandwidth (MB/s)",
        "Single-pair bandwidth sweep by model",
    )
    _plot_single_pair_line_grid(
        output_dir / "single_pair_local_gflops_lines.png",
        single_pair_local_rows,
        "md_gflops",
        "local GFLOPS",
        "Single-pair local-resource sweep by model",
    )
    _plot_single_pair_metric_triptych(
        output_dir / "single_pair_model_bars.png",
        single_pair_bar_rows,
        "Single-pair latency / energy / objective cost by model",
    )
    _plot_single_pair_gain_heatmaps(
        output_dir / "single_pair_gain_heatmaps.png",
        single_pair_gain_rows,
        "Single-pair normalized MinCut gain over endpoint baselines",
    )
    _plot_bars(
        output_dir / "mixed5_main_cost.png",
        summary_rows,
        "main_comparison",
        "mixed5",
        "Unified five-model comparison",
        MULTIUSER_METHOD_ORDER,
    )
    _plot_lines(
        output_dir / "mixed5_bandwidth_cost.png",
        summary_rows,
        "bandwidth_sensitivity",
        "mixed5",
        "bandwidth_mb_s",
        "uplink bandwidth (MB/s)",
        "Bandwidth sensitivity: unified five-model scenario",
        MULTIUSER_METHOD_ORDER,
    )
    _plot_lines(
        output_dir / "mixed5_bandwidth_match_focus.png",
        summary_rows,
        "bandwidth_sensitivity",
        "mixed5",
        "bandwidth_mb_s",
        "uplink bandwidth (MB/s)",
        "Bandwidth sensitivity: matching-focused comparison",
        ("Only-Server", "MinCut+BMatch", "MinCut+RMatch"),
    )
    _plot_lines(
        output_dir / "mixed5_beta_cost.png",
        summary_rows,
        "beta_sensitivity",
        "mixed5",
        "beta",
        "energy weight beta",
        "Energy-weight sensitivity: unified five-model scenario",
        MULTIUSER_METHOD_ORDER,
    )
    _plot_runtime_by_model(output_dir / "algorithm_runtime.png", runtime_rows)
    _plot_scalability_runtime(output_dir / "scalability_runtime.png", scalability_rows)
    _plot_online_over_time(
        output_dir / "online_cost_over_time.png",
        online_timeseries_rows,
        "online_composite_nonstationary",
        "total_cost",
        "total cost",
        "Online cost over time: composite non-stationary scenario",
    )
    _plot_online_over_time(
        output_dir / "online_queue_over_time.png",
        online_timeseries_rows,
        "online_composite_nonstationary",
        "avg_task_queue_wait",
        "avg queue wait (s)",
        "Online queue wait over time: composite non-stationary scenario",
    )

    print(f"wrote results to {output_dir} (compression_ratio={compression_ratio})")
    for row in summary_rows:
        if row["experiment"] == "main_comparison":
            print(
                f"{row['experiment']:20s} {row['scenario']:8s} {row['method']:20s} "
                f"cost={row['total_cost']:.6f}+/-{row['total_cost_std']:.6f} "
                f"latency={row['total_latency']:.6f} energy={row['total_energy']:.6f}"
            )

    for row in online_summary_rows:
        if row["scenario"] == "online_composite_nonstationary":
            print(
                f"{row['scenario']:20s} {row['method']:20s} "
                f"avg_cost={row['avg_total_cost']:.6f} avg_wait={row['avg_task_queue_wait']:.6f} "
                f"switch_rate={row['policy_switch_rate']:.6f}"
            )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Reproduce and optimize multi-user DNN partitioning experiments.")
    parser.add_argument("--output", default="outputs_mixed5_yolo_ratio4_midonly", help="directory for CSV and plots")
    parser.add_argument("--alpha", type=float, default=1.0, help="latency weight")
    parser.add_argument("--beta", type=float, default=1.0, help="energy weight")
    parser.add_argument("--random-repeats", type=int, default=30, help="repeat count for random matching and random partition baselines")
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=DEFAULT_COMPRESSION_RATIO,
        help="fixed upload compression ratio applied only to transmitted intermediate features",
    )
    args = parser.parse_args(argv)
    run_all(
        Path(args.output),
        CostWeights(alpha=args.alpha, beta=args.beta),
        random_repeats=args.random_repeats,
        compression_ratio=args.compression_ratio,
    )


if __name__ == "__main__":
    main()
