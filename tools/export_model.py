"""Validate and export supported PyTorch models in Cinder format version 1.

validate_model() returns independent CPU float32 parameter snapshots without
file I/O. export_model() validates first, then atomically replaces a destination
with the complete binary file. Neither executes forward() or changes the source.
See docs/EXPORTER.md for supported configurations, ownership, and file handling.
"""

from dataclasses import dataclass
from enum import IntEnum
import os
from pathlib import Path
import struct
import tempfile
from typing import BinaryIO

import torch
from torch import nn
from torch.nn.modules import module as module_hooks


# File-format constants, independent of C runtime enum values or native sizes.
FORMAT_VERSION = 1
MAX_LAYERS = 1024
MAX_VECTOR_SIZE = 65536
MAX_PARAMETER_BYTES = 64 * 1024 * 1024
FILE_HEADER_BYTES = 16
LAYER_HEADER_BYTES = 16
_PARAMETER_CHUNK_WORDS = 4096


class LayerType(IntEnum):
    DENSE = 1
    RELU = 2
    SOFTMAX = 3


class ModelValidationError(ValueError):
    """The supplied model cannot be represented by Cinder format version 1."""


@dataclass(frozen=True, slots=True, eq=False)
class ValidatedLayer:
    """Fixed metadata and independently owned parameter tensors, if Dense.

    Tensor contents are mutable Python objects; consumers must treat them as
    read-only. Frozen metadata does not make PyTorch tensors immutable.
    """

    type_id: LayerType
    input_size: int
    output_size: int
    weights: torch.Tensor | None = None
    bias: torch.Tensor | None = None

    @property
    def parameter_count(self) -> int:
        if self.type_id == LayerType.DENSE:
            return self.input_size * self.output_size + self.output_size
        return 0

    @property
    def payload_bytes(self) -> int:
        return 4 * self.parameter_count


@dataclass(frozen=True, slots=True, eq=False)
class ValidatedModel:
    """Snapshot in execution order; no references to the source modules."""

    input_size: int
    layers: tuple[ValidatedLayer, ...]

    @property
    def output_size(self) -> int:
        return self.layers[-1].output_size

    @property
    def parameter_count(self) -> int:
        return sum(layer.parameter_count for layer in self.layers)

    @property
    def parameter_bytes(self) -> int:
        return 4 * self.parameter_count

    @property
    def file_bytes(self) -> int:
        return (FILE_HEADER_BYTES + LAYER_HEADER_BYTES * len(self.layers)
                + self.parameter_bytes)


def _dimension(value: object, context: str) -> int:
    # bool is an int subclass, but is not a meaningful model dimension.
    if type(value) is not int or not 1 <= value <= MAX_VECTOR_SIZE:
        raise ModelValidationError(
            f"{context}: expected an integer in 1..{MAX_VECTOR_SIZE}, got {value!r}"
        )
    return value


def _standard_behavior(module: nn.Module, context: str) -> None:
    if module._forward_hooks or module._forward_pre_hooks:
        raise ModelValidationError(f"{context}: forward hooks are not supported")
    if any(name in vars(module) for name in ("forward", "_call_impl")):
        raise ModelValidationError(f"{context}: overridden execution is not supported")
    if getattr(module, "_compiled_call_impl", None) is not None:
        raise ModelValidationError(f"{context}: compiled execution is not supported")
    if module._buffers:
        raise ModelValidationError(f"{context}: registered buffers are not supported")


def _parameter(value: object, shape: tuple[int, ...], context: str) -> torch.Tensor:
    # Tensor subclasses may change operations; lazy and parametrized weights
    # must not be evaluated or silently materialized by validation.
    if type(value) not in (torch.Tensor, nn.Parameter):
        raise ModelValidationError(f"{context}: expected a plain Tensor or Parameter")
    if value.device.type != "cpu":
        raise ModelValidationError(f"{context}: expected CPU storage, got {value.device}")
    if value.dtype != torch.float32:
        raise ModelValidationError(f"{context}: expected torch.float32, got {value.dtype}")
    if value.layout != torch.strided or value.is_nested:
        raise ModelValidationError(f"{context}: expected an ordinary strided tensor")
    if tuple(value.shape) != shape:
        raise ModelValidationError(
            f"{context}: expected shape {shape}, got {tuple(value.shape)}"
        )
    return value


def _snapshot(value: torch.Tensor, context: str) -> torch.Tensor:
    # clone is required even for an already contiguous CPU tensor: detach alone
    # shares storage. Check the copied values that the writer will later use.
    result = value.detach().clone(memory_format=torch.contiguous_format)
    if not bool(torch.isfinite(result).all()):
        raise ModelValidationError(f"{context}: every parameter must be finite")
    return result


def validate_model(
    model: nn.Sequential, *, input_size: int | None = None
) -> ValidatedModel:
    """Validate a flat Sequential and extract an independent parameter snapshot.

    Accept exact nn.Linear (with bias), nn.ReLU, and nn.Softmax(dim=-1)
    classes. Parameters must already be strided CPU float32 tensors. Infer the
    input size from the first Linear, or require it for activation-only models.
    An explicit size is always checked against every connection.

    Metadata for the entire model is checked before parameter copies. Repeated
    module occurrences remain separate records with separate owned copies.
    Both training modes are accepted for these mode-independent operations.
    Never call forward(), eval(), cpu(), or float() on the supplied model.
    The caller must not mutate it during this call. See docs/EXPORTER.md.

    Unsupported inputs raise ModelValidationError. Allocation/system failures
    propagate normally; no partial snapshot is returned and no files are used.
    """
    if type(model) is not nn.Sequential:
        raise ModelValidationError("model: expected an exact torch.nn.Sequential")
    if module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks:
        raise ModelValidationError("model: global forward hooks are not supported")
    _standard_behavior(model, "model")
    if model._parameters:
        raise ModelValidationError("model: parameters must belong to Linear layers")
    count = len(model)
    if not 1 <= count <= MAX_LAYERS:
        raise ModelValidationError(f"model: expected 1..{MAX_LAYERS} layers, got {count}")
    if input_size is not None:
        input_size = _dimension(input_size, "model input_size")

    # children()/named_children() deduplicate shared modules; Sequential forward
    # executes every registered occurrence. Preserve the same ordered entries.
    entries = list(model._modules.items())
    descriptions = []
    parameter_bytes = 0
    for index, (name, layer) in enumerate(entries):
        context = f"layer {index} ({name!r})"
        if type(layer) not in (nn.Linear, nn.ReLU, nn.Softmax):
            raise ModelValidationError(
                f"{context}: unsupported module {type(layer).__name__}; "
                "expected Linear, ReLU, or Softmax"
            )
        _standard_behavior(layer, context)
        if layer._modules:
            raise ModelValidationError(f"{context}: nested modules are not supported")

        if type(layer) is nn.Linear:
            ins = _dimension(getattr(layer, "in_features", None), f"{context} in_features")
            outs = _dimension(getattr(layer, "out_features", None), f"{context} out_features")
            # Python integers do not wrap. Bound the total before copying or
            # scanning tensors, even when their metadata describes huge views.
            size = 4 * (ins * outs + outs)
            if size > MAX_PARAMETER_BYTES - parameter_bytes:
                raise ModelValidationError(
                    f"{context}: parameters exceed the {MAX_PARAMETER_BYTES}-byte model limit"
                )
            parameter_bytes += size
            if set(layer._parameters) != {"weight", "bias"}:
                raise ModelValidationError(f"{context}: expected only weight and bias parameters")
            if layer.bias is None:
                raise ModelValidationError(f"{context}: Linear bias is required")
            weight = _parameter(layer.weight, (outs, ins), f"{context} weight")
            bias = _parameter(layer.bias, (outs,), f"{context} bias")
            descriptions.append((context, LayerType.DENSE, ins, outs, weight, bias))
        else:
            if layer._parameters:
                raise ModelValidationError(f"{context}: activation parameters are not supported")
            if type(layer) is nn.ReLU:
                if type(getattr(layer, "inplace", None)) is not bool:
                    raise ModelValidationError(f"{context}: ReLU inplace must be bool")
                kind = LayerType.RELU
            else:
                dim = getattr(layer, "dim", None)
                if type(dim) is not int or dim != -1:
                    raise ModelValidationError(f"{context}: Softmax requires dim=-1")
                kind = LayerType.SOFTMAX
            descriptions.append((context, kind, None, None, None, None))

    if input_size is None:
        input_size = next((ins for _, kind, ins, _, _, _ in descriptions
                           if kind == LayerType.DENSE), None)
        if input_size is None:
            raise ModelValidationError("model: activation-only models require input_size")

    # Resolve dimension-preserving activations and every connection before
    # taking copies, including activations before the first Dense layer.
    resolved = []
    current = input_size
    for context, kind, ins, outs, weight, bias in descriptions:
        if kind == LayerType.DENSE:
            if ins != current:
                raise ModelValidationError(
                    f"{context}: expects {ins} inputs, preceding vector has {current}"
                )
        else:
            ins = outs = current
        resolved.append((context, kind, ins, outs, weight, bias))
        current = outs

    layers = []
    for context, kind, ins, outs, weight, bias in resolved:
        if kind == LayerType.DENSE:
            weight = _snapshot(weight, f"{context} weight")
            bias = _snapshot(bias, f"{context} bias")
        layers.append(ValidatedLayer(kind, ins, outs, weight, bias))
    return ValidatedModel(input_size, tuple(layers))


def _write_exact(stream: BinaryIO, data: bytes) -> None:
    if stream.write(data) != len(data):
        raise OSError("incomplete write while exporting Cinder model")


def _write_parameters(stream: BinaryIO, values: torch.Tensor) -> None:
    # Reinterpret binary32 bits, rather than converting float values. Packing
    # those words explicitly little-endian preserves signed zero/subnormals and
    # does not depend on native byte order or require NumPy.
    words = values.view(torch.int32).reshape(-1)
    for start in range(0, words.numel(), _PARAMETER_CHUNK_WORDS):
        chunk = words[start:start + _PARAMETER_CHUNK_WORDS].tolist()
        _write_exact(stream, struct.pack(f"<{len(chunk)}i", *chunk))


def _write_snapshot(stream: BinaryIO, snapshot: ValidatedModel) -> None:
    # Private: only consume a fresh snapshot owned by export_model().
    _write_exact(stream, struct.pack("<4sIII", b"CNDR", FORMAT_VERSION,
                                     snapshot.input_size, len(snapshot.layers)))
    for layer in snapshot.layers:
        _write_exact(stream, struct.pack("<IIII", layer.type_id, layer.input_size,
                                         layer.output_size, layer.payload_bytes))
        if layer.type_id == LayerType.DENSE:
            _write_parameters(stream, layer.weights)
            _write_parameters(stream, layer.bias)


def export_model(
    model: nn.Sequential, path: str | os.PathLike[str], *, input_size: int | None = None
) -> None:
    """Validate and write a complete .cnd v1 file, replacing path on success.

    The parent directory must exist. Validation finishes before file creation.
    Write a private temporary file in the destination directory, verify length,
    flush/fsync/close it, then publish with os.replace(). Errors propagate and
    temporary files are removed on ordinary failures; an existing destination
    is preserved unless replacement succeeds. A destination symlink itself is
    replaced, not followed. This is not a power-loss durability guarantee.

    The path does not need a .cnd suffix. No checkpoint loading or forward pass
    occurs. The caller must not mutate the model during validation.
    """
    snapshot = validate_model(model, input_size=input_size)
    destination = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".cinder-", suffix=".tmp",
                                         dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            _write_snapshot(stream, snapshot)
            if stream.tell() != snapshot.file_bytes:
                raise OSError("incorrect byte count while exporting Cinder model")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
