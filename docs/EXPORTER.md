# Python exporter — model validation

[`tools/export_model.py`](../tools/export_model.py) implements the validation
and parameter-extraction step for the [`.cnd` version 1 contract](MODEL_FORMAT.md).
It returns an in-memory snapshot. Binary writing, checkpoint loading, and an
exporter command-line interface are later work; `export_model(model, path)` is
not implemented yet.

## Public function and result

```python
import torch
from torch import nn
from tools.export_model import validate_model, ModelValidationError

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=-1),
)

snapshot = validate_model(model)  # Or validate_model(model, input_size=2).
print(snapshot.input_size, snapshot.output_size)  # 2 2
print(snapshot.parameter_count, snapshot.parameter_bytes)  # 22 88
print(snapshot.file_bytes)  # 168: expected serialized size, no file written.

for layer in snapshot.layers:
    print(layer.type_id.name, layer.input_size, layer.output_size)
```

The signature is `validate_model(model, *, input_size=None) -> ValidatedModel`.
`ModelValidationError`, a `ValueError` subclass, reports unsupported input with
model or zero-based layer index/name context. For example, a `3 -> 4` connection
mismatch reports that the next layer expects four inputs but the preceding
vector has three. Allocation and system failures propagate normally; no partial
snapshot is returned. The function takes no filename and performs no file I/O.

`ValidatedModel` exposes `input_size`, derived `output_size`, an ordered tuple
of `layers`, and derived parameter count, parameter bytes, and expected file
bytes. Each `ValidatedLayer` exposes:

| Field/property | Meaning |
| --- | --- |
| `type_id` | `LayerType.DENSE = 1`, `RELU = 2`, or `SOFTMAX = 3`, matching the file specification. |
| `input_size`, `output_size` | Positive float-element counts. |
| `weights` | Dense-only CPU float32 tensor shaped `[output_size, input_size]`; otherwise `None`. |
| `bias` | Dense-only CPU float32 tensor shaped `[output_size]`; otherwise `None`. |
| `parameter_count`, `payload_bytes` | Derived parameter totals; zero for activations. |

The records have frozen metadata, and the layer collection is a tuple. PyTorch
tensors themselves remain mutable, so consumers must treat snapshot tensors as
read-only. These objects are validation results, not a separate public input
format for constructing arbitrary models. The future writer will validate a
source model and consume its fresh snapshot.

## Accepted structure and configurations

The root must be an exact, nonempty `torch.nn.Sequential`. It must contain a
flat sequence of these exact built-in classes:

| PyTorch module | Cinder mapping | Requirements |
| --- | --- | --- |
| `nn.Linear` | Dense | Mandatory bias, valid `in_features`/`out_features`, matching parameter shapes. |
| `nn.ReLU` | ReLU | Boolean `inplace`; both `False` and `True` are accepted. |
| `nn.Softmax` | Softmax | Explicit integer `dim=-1`; no inferred or alternative axis. |

The exporter targets a single feature vector. ReLU's in-place setting changes
storage behavior in Python, not its elementwise calculation, so both settings
map to the same Cinder operation. Cinder retains its own input/output ownership
rules. Training and evaluation modes are both accepted for these operations,
which have the same inference calculation in either mode; validation does not
call `eval()` or change any training flags.

Nested containers, custom subclasses, unsupported layers (including `Dropout`
even in evaluation mode), bias-free Linear, lazy or parametrized modules,
compiled execution, instance-level execution overrides, and forward hooks are
rejected. Forward hooks on either a module or the process-wide module registry
can change the calculation, so neither is silently ignored. Pruned modules with
forward hooks are also rejected. Materializing a plain supported model belongs
to caller preparation, outside this step.

Registered buffers, extra parameters, and children attached to leaf modules
are rejected. This accepts a deliberately small set of standard PyTorch objects;
it is not a sandbox for arbitrary Python or a detector for arbitrary monkey
patches to PyTorch classes. Use unmodified built-in classes, and do not mutate
the source or its execution hooks during validation. Hook inspection uses
PyTorch's internal registries and is covered by tests with the pinned release.

Execution order is taken from every registered Sequential entry. Reusing the
same module object in multiple positions is supported: every occurrence gets
its own record and independent parameter copies. Generic `children()` traversal
would deduplicate shared modules and incorrectly change the network.

## Dimensions, parameter representation, and limits

If `input_size` is supplied, it must be a Python integer in `1..65536`, excluding
booleans. If omitted, it is inferred from the first Linear's `in_features`.
Any leading activations preserve that size. An activation-only sequence must
receive an explicit size because it contains no dimensional parameter metadata:

```python
snapshot = validate_model(nn.Sequential(nn.ReLU(), nn.Softmax(dim=-1)), input_size=4)
```

Every Linear input must match the current vector size; its output becomes the
next size. ReLU and Softmax preserve it. Explicit input sizes are checked rather
than overriding a layer's dimensions. A final Softmax is optional; a one-layer
Dense, activation-only model, or compatible sequence with intermediate Softmax
is valid.

Dense weights must have exactly `[out_features, in_features]` shape, and bias
must have exactly `[out_features]` shape. Both must already be ordinary strided
CPU `torch.float32` tensors or Parameters. Sparse/nested tensors, tensor
subclasses, non-CPU devices (including meta), and other dtypes are rejected.
The caller can explicitly prepare a suitable model before validation; no
implicit device transfer, precision conversion, reshaping, or bias synthesis
occurs.

Noncontiguous strided parameters are accepted. Their copies are contiguous in
the existing logical row-major order, preserving `[output][input]` layout.
Every copied value must be finite; signed zeros, subnormals, and finite extremes
retain their float32 encodings. There is no inference trial to prove that all
future inputs avoid arithmetic overflow.

Limits match the format: at most 1,024 layers, 65,536 values per vector, and
64 MiB of total weights/biases. Repeated Dense occurrences count separately.
Python integer arithmetic does not wrap; the validator still checks the
computed total against the format cap before copying or scanning parameter
values. Individual dimensions within bounds can still describe an oversized
matrix. Counts and file size are derived from validated dimensions.

## Validation order and ownership

1. Check the root, execution hooks/state, layer count, and optional input size.
2. Inspect every layer's supported type/configuration, dimensions, parameter
   shapes/device/dtype/layout, and cumulative parameter size.
3. Infer or confirm the model input size and validate every connection.
4. Copy Dense parameters with `detach().clone(memory_format=torch.contiguous_format)`
   and check the copied values for finiteness. Return the complete snapshot.

This avoids copying earlier weights when a later layer already has invalid
metadata. Finiteness is checked on the copy that a future writer would consume.
Diagnostics identify a violation; when several exist, callers should not rely
on one particular error being reported before every other category.

`detach()` alone shares storage. The explicit clone ensures that later source
parameter updates cannot change a successful snapshot. Snapshot changes also
cannot affect the source. Validation preserves source tensors, gradients,
`requires_grad`, training flags, and module configuration, and never executes
the model's forward methods. The returned tensors have no autograd history or
references to the original modules. Repeated calls create independent results;
a failed validation requires no reset before retry.

The parameter-byte cap does not describe peak memory: source parameters,
snapshots, and temporary validation storage may coexist. Ordinary Python
ownership releases abandoned snapshots; allocation errors are not converted
into misleading model-format errors.

## Dependencies and running tests

Python 3.11+ and PyTorch are optional for Cinder's ordinary C build. The tested
PyTorch version is pinned in
[`tools/requirements-exporter.txt`](../tools/requirements-exporter.txt).
From the project root on macOS:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements-exporter.txt
.venv/bin/python -m unittest discover -s tests -p test_export_model.py -v
```

On Linux, install the same requirement from the official CPU wheel index:

```sh
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -r tools/requirements-exporter.txt
```

NumPy is not required by validation or these tests. The tests read the existing
C fixture arrays with Python's standard library. To regenerate references as
well, install the separate `tests/requirements-numpy.txt` dependency.

The opt-in CTest integration is:

```sh
cmake -S . -B build-exporter -DCMAKE_BUILD_TYPE=Debug \
  -DCINDER_ENABLE_EXPORTER_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-exporter --parallel
ctest --test-dir build-exporter --output-on-failure
```

CMake requires Python 3.11+, an importable PyTorch, and `BUILD_TESTING=ON` when
this option is enabled. It does not install dependencies. With the option off,
the C build and its saved-fixture tests require no Python or PyTorch. The Linux
CI build enables exporter validation alongside fresh NumPy references; the
macOS C job and Linux C sanitizer job retain their saved-fixture configuration.

## Coverage

[`tests/test_export_model.py`](../tests/test_export_model.py) exercises real
PyTorch modules, not replacements for the framework. Coverage includes the
fixed network and all ten saved NumPy inputs at all four stages, exact extracted
parameters, activation-first/only models, scalar and repeated layers, snapshot
independence, noncontiguous storage, finite edge values, invalid configurations,
connections, hooks, resource boundaries, and failure/retry behavior. It also
blocks file/checkpoint APIs and forward calls while validation runs.

## Verification — October 24, 2026

Local checks on macOS on October 23–24 used Python 3.12.14, PyTorch 2.14.1,
and NumPy 2.3.5 for fresh reference generation. The final combined suite passed
on October 24:

| Check | Observed result |
| --- | --- |
| Exporter Python suite | All 33 unittest methods passed, including rejection cases and retries. |
| Extracted tiny-network calculations | All 40 stage vectors / 120 values matched the saved NumPy references, with matching shapes, finite values, and `atol = rtol = 1e-5`. |
| Fresh C-only Debug build | Built successfully; all 19 CTest tests passed with Python package discovery disabled. |
| Debug build with both optional Python suites | Built successfully; all 22 CTest tests passed: 19 ordinary tests, two fresh NumPy comparisons, and one exporter suite containing the 33 Python tests. |
| Exporter option with testing disabled | Configuration rejected with the required `BUILD_TESTING=ON` diagnostic. |
| Exporter option with a Python environment lacking PyTorch | Configuration rejected with the dependency-install diagnostic. |

To reproduce the combined configuration after installing both requirement files:

```sh
cmake -S . -B build-exporter -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
  -DCINDER_ENABLE_NUMPY_TESTS=ON -DCINDER_ENABLE_EXPORTER_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-exporter --parallel
ctest --test-dir build-exporter --output-on-failure --no-tests=error
```

The Linux workflow is configured to run these additions, but hosted results
are pending a future push. No C runtime implementation changed in this step;
the sanitizer/leak results in [the M2 report](M2_VERIFICATION.md) belong to that
earlier verified revision, not a new sanitizer run for the Python exporter.

## Next step and references

The next scoped step is binary writing and reproducible `.cnd` fixtures using
this validated representation. These tests do not claim export/load round-trip
coverage; the writer and C loader are still to be implemented.

The supported semantics were checked against the official PyTorch documentation
for [Sequential](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sequential.html),
[Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html),
[ReLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html), and
[Softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softmax.html).
