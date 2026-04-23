from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision
from torch.fx import symbolic_trace
from torch.fx.node import Node
from torch.fx.passes.shape_prop import ShapeProp


@dataclass(frozen=True)
class LayerProfile:
    node_id: str
    name: str
    kind: str
    flops: float
    output_bytes: float
    partitionable: bool


@dataclass(frozen=True)
class DNNProfile:
    model_name: str
    input_bytes: float
    layers: Tuple[LayerProfile, ...]
    edges: Tuple[Tuple[str, str], ...]
    output_node_id: str
    output_bytes: float

    @property
    def layer_ids(self) -> Tuple[str, ...]:
        return tuple(layer.node_id for layer in self.layers)

    @property
    def layer_by_id(self) -> Dict[str, LayerProfile]:
        return {layer.node_id: layer for layer in self.layers}


MODEL_BUILDERS = {
    "alexnet": lambda: torchvision.models.alexnet(weights=None),
    "vgg13": lambda: torchvision.models.vgg13(weights=None),
    "vgg19": lambda: torchvision.models.vgg19(weights=None),
    "resnet18": lambda: torchvision.models.resnet18(weights=None),
    "resnet50": lambda: torchvision.models.resnet50(weights=None),
    "resnet101": lambda: torchvision.models.resnet101(weights=None),
    "googlenet": lambda: torchvision.models.googlenet(weights=None, aux_logits=False, init_weights=False),
    "densenet121": lambda: torchvision.models.densenet121(weights=None),
    "mobilenet_v2": lambda: torchvision.models.mobilenet_v2(weights=None),
}


PARTITIONABLE_MODULES = (
    nn.Conv2d,
    nn.Linear,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
)


def canonical_model_name(name: str) -> str:
    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "alexnet": "alexnet",
        "vgg13": "vgg13",
        "vgg19": "vgg19",
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "googlenet": "googlenet",
        "densenet121": "densenet121",
        "mobilenetv2": "mobilenet_v2",
        "mobilenet2": "mobilenet_v2",
    }
    if key not in aliases:
        raise ValueError(f"unsupported model: {name}")
    return aliases[key]


def _shape_numel(shape: Sequence[int]) -> int:
    out = 1
    for dim in shape:
        if dim is None:
            return 0
        out *= int(dim)
    return int(out)


def _tensor_numel_from_meta(meta) -> int:
    if meta is None:
        return 0
    if hasattr(meta, "shape"):
        return _shape_numel(meta.shape)
    if isinstance(meta, (tuple, list)):
        return sum(_tensor_numel_from_meta(item) for item in meta)
    return 0


def _tensor_bytes_from_node(node: Node) -> float:
    tensor_meta = node.meta.get("tensor_meta")
    return float(_tensor_numel_from_meta(tensor_meta) * 4)


def _first_input_node(node: Node) -> Optional[Node]:
    return next(iter(node.all_input_nodes), None)


def _pool_kernel_size(module: nn.Module) -> int:
    kernel_size = getattr(module, "kernel_size", 1)
    if isinstance(kernel_size, tuple):
        return math.prod(kernel_size)
    return int(kernel_size)


def _module_flops(module: nn.Module, node: Node) -> float:
    out_meta = node.meta.get("tensor_meta")
    out_numel = _tensor_numel_from_meta(out_meta)
    input_node = _first_input_node(node)
    input_numel = _tensor_numel_from_meta(input_node.meta.get("tensor_meta")) if input_node else 0

    if isinstance(module, nn.Conv2d):
        out_shape = list(out_meta.shape)
        if len(out_shape) < 4:
            return 0.0
        batch, out_channels, out_h, out_w = [int(x) for x in out_shape[:4]]
        kernel_ops = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        return float(batch * out_channels * out_h * out_w * 2 * kernel_ops)
    if isinstance(module, nn.Linear):
        return float(2 * module.in_features * module.out_features)
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return float(2 * out_numel)
    if isinstance(module, (nn.ReLU, nn.ReLU6, nn.Sigmoid, nn.Tanh)):
        return float(out_numel)
    if isinstance(module, (nn.MaxPool2d, nn.AvgPool2d)):
        return float(out_numel * _pool_kernel_size(module))
    if isinstance(module, nn.AdaptiveAvgPool2d):
        return float(input_numel)
    return 0.0


def _function_kind(target) -> str:
    if target is torch.flatten:
        return "Flatten"
    if target is operator.add or str(target) == "<built-in function add>":
        return "Add"
    return getattr(target, "__name__", str(target))


def _function_flops(kind: str, node: Node) -> float:
    if kind == "Add":
        return float(_tensor_numel_from_meta(node.meta.get("tensor_meta")))
    return 0.0


def _node_profile(node: Node, modules: Dict[str, nn.Module]) -> LayerProfile:
    output_bytes = _tensor_bytes_from_node(node)
    if node.op == "call_module":
        module = modules[str(node.target)]
        kind = module.__class__.__name__
        partitionable = isinstance(module, PARTITIONABLE_MODULES)
        flops = _module_flops(module, node)
        name = str(node.target)
    elif node.op == "call_function":
        kind = _function_kind(node.target)
        partitionable = kind == "Add"
        flops = _function_flops(kind, node)
        name = str(node.target)
    elif node.op == "call_method":
        kind = str(node.target)
        partitionable = False
        flops = 0.0
        name = str(node.target)
    else:
        kind = node.op
        partitionable = False
        flops = 0.0
        name = str(node.target)
    return LayerProfile(
        node_id=node.name,
        name=name,
        kind=kind,
        flops=flops,
        output_bytes=output_bytes,
        partitionable=partitionable,
    )


@lru_cache(maxsize=None)
def build_profile(model_name: str, input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> DNNProfile:
    name = canonical_model_name(model_name)
    model = MODEL_BUILDERS[name]().eval()
    graph_module = symbolic_trace(model)
    ShapeProp(graph_module).propagate(torch.zeros(*input_shape))
    modules = dict(graph_module.named_modules())

    raw_nodes = [node for node in graph_module.graph.nodes if node.op not in {"placeholder", "output"}]
    raw_profiles = {node.name: _node_profile(node, modules) for node in raw_nodes}
    skipped_ids = {node.name for node in raw_nodes if raw_profiles[node.name].kind == "Flatten"}
    fx_nodes = [node for node in raw_nodes if node.name not in skipped_ids]
    layers = tuple(raw_profiles[node.name] for node in fx_nodes)
    layer_ids = {node.name for node in fx_nodes}

    def resolved_sources(node: Node) -> List[str]:
        if node.op == "placeholder":
            return ["input"]
        if node.name in layer_ids:
            return [node.name]
        if node.name in skipped_ids:
            out: List[str] = []
            for parent in node.all_input_nodes:
                out.extend(resolved_sources(parent))
            return out
        return []

    edges: List[Tuple[str, str]] = []
    for node in fx_nodes:
        for src in node.all_input_nodes:
            for resolved in resolved_sources(src):
                edges.append((resolved, node.name))

    output_node = next(node for node in graph_module.graph.nodes if node.op == "output")
    output_inputs = list(output_node.all_input_nodes)
    if not output_inputs:
        raise RuntimeError(f"could not identify output node for {model_name}")
    resolved_output = resolved_sources(output_inputs[0])
    if not resolved_output:
        raise RuntimeError(f"could not resolve output node for {model_name}")
    output_node_id = resolved_output[0]
    output_bytes = _tensor_bytes_from_node(output_inputs[0])
    input_bytes = float(math.prod(input_shape) * 4)

    return DNNProfile(
        model_name=name,
        input_bytes=input_bytes,
        layers=layers,
        edges=tuple(dict.fromkeys(edges)),
        output_node_id=output_node_id,
        output_bytes=output_bytes,
    )


def build_profiles(names: Iterable[str]) -> Dict[str, DNNProfile]:
    return {canonical_model_name(name): build_profile(name) for name in names}
