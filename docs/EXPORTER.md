# Python exporter — validation and binary writing

[`tools/export_model.py`](../tools/export_model.py) validates supported models
and writes the [`.cnd` version 1 format](MODEL_FORMAT.md).
`validate_model()` returns an in-memory snapshot; `export_model()` validates a
source model and writes its complete binary representation. Checkpoint loading,
a general exporter CLI, and the C loader are not implemented.

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
format for constructing arbitrary models. The writer validates a source model
and consumes its fresh, privately owned snapshot.

## Binary export API and encoding

```python
from tools.export_model import export_model

export_model(model, "model.cnd")
# An explicit input size is checked, including for activation-only models:
export_model(model, "model.cnd", input_size=2)
```

The signature is `export_model(model, path, *, input_size=None) -> None`.
`path` is a string or string-valued path-like object. The parent directory must
exist; `.cnd` is conventional but the suffix is not enforced. Successful export
returns `None`. Validation errors retain `ModelValidationError`; file and
allocation errors propagate as their original exceptions.

After all validation and parameter copying succeeds, the writer emits:

1. `struct.pack("<4sIII", b"CNDR", 1, input_size, layer_count)` — the 16-byte header.
2. `struct.pack("<IIII", type_id, input_size, output_size, payload_bytes)` for
   each layer — the 16-byte layer record header.
3. For each Dense, row-major weights followed by biases. Activations have no
   payload. The final record ends the file with no footer or extra bytes.

The `<` prefix selects little-endian fixed-width fields without alignment
padding. Parameter tensors are viewed as 32-bit integer bit patterns, then
packed little-endian in chunks of at most 4,096 words / 16 KiB. This preserves
signed zeros, subnormals, and finite extremes without converting through Python
floating-point arithmetic or depending on the host's byte order. NumPy is not
required. Each encoded write is at most 16 KiB; temporary Python integer lists
add their own overhead, and the validated snapshot still occupies its own storage.

Every write checks its byte count. Before publishing, the writer also verifies
that the total length equals the size derived from the validated records. File
bytes contain no path, model name, clock value, or random data. The same ordered
dimensions and parameter bits produce identical output bytes.

## Destination ownership and failures

Validation finishes before any output file is created. The exporter then
creates a private `.cinder-*.tmp` file in the destination directory, writes it,
checks its length, flushes Python buffering, calls `fsync`, and closes it.
Only then does `os.replace` publish it at the requested path. Creating the
temporary file beside the destination keeps replacement on the same filesystem.

Existing destinations are overwritten only on successful replacement. Validation,
encoding, allocation, write, flush, sync, close, or replacement failures leave
the original destination intact (or absent if it did not exist). Ordinary
exception paths close and remove the temporary file. If the operating system
also refuses cleanup, that cleanup error propagates and the temporary file may
remain; forced process termination can likewise leave an unpublished temporary.
No parent directory is created by this API.

Replacement replaces a destination symlink itself, leaving its target untouched.
The new file has the temporary file's permissions; existing permissions and
other inode metadata are not preserved. On the supported macOS/Linux systems,
temporary files start readable/writable by their owner only. Concurrent exports
to one path publish complete files, with the last successful replacement winning.
The directory is not synced, so this is atomic publication, not a guarantee
against power loss. These are filesystem rules, separate from the `.cnd` bytes.

## Saved tiny-network fixtures

[`tests/fixtures/tiny/tiny.cnd`](../tests/fixtures/tiny/tiny.cnd) is the fixed
`2 -> 4 -> 2` model: 168 bytes, four records, and 22 parameters / 88 payload bytes.
Its SHA-256 is:

```text
3f6f2942084b334e2c9e8b59a1b0e702057554f4253473ae86ffd7cde7d86fa5
```

| Record | Header offset | Payload offset | Payload bytes | Next offset |
| --- | ---: | ---: | ---: | ---: |
| Dense `2 -> 4` | 16 | 32 | 48 | 80 |
| ReLU `4 -> 4` | 80 | 96 | 0 | 96 |
| Dense `4 -> 2` | 96 | 112 | 40 | 152 |
| Softmax `2 -> 2` | 152 | 168 | 0 | 168 / EOF |

The directory also holds ten `inputs/<case>.f32` files, each exactly eight bytes,
and [`references.json`](../tests/fixtures/tiny/references.json). The JSON names
the cases and files, records the model hash, dimensions and comparison tolerance,
and preserves all four expected stage vectors per case as exact hexadecimal
float strings. Read a stage value with `float.fromhex(value)` and convert to
float32. Shape and finite-value checks precede `atol = rtol = 1e-5` comparisons.
The hash is external fixture metadata, not an embedded format checksum.

The generator reads existing saved NumPy arrays from `numpy_fixtures.h`.
Parameters construct a plain PyTorch model and pass through the public exporter;
reference outputs are copied from NumPy fixtures, never recomputed by the
exporter. The original C arrays remain available for future C loader tests.

```sh
.venv/bin/python tests/generate_cnd_fixtures.py
.venv/bin/python tests/generate_cnd_fixtures.py --check
xxd -g 1 tests/fixtures/tiny/tiny.cnd
```

Generation overwrites the twelve known fixture files. `--output <directory>`
chooses another destination; `--check` generates in temporary storage and
compares all file names and bytes without modifying the saved directory. Drift,
missing files, and extra files fail the check. Regeneration is tested twice
against the saved set, including JSON and every input file. CTest writes only
to temporary directories; it never silently refreshes the saved fixtures.

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
metadata. Finiteness is checked on the copy that the writer consumes.
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
.venv/bin/python -m unittest discover -s tests -p 'test_export_*.py' -v
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
CI build enables validation and binary export tests alongside fresh NumPy references; the
macOS C job and Linux C sanitizer job retain their saved-fixture configuration.

## Coverage

[`tests/test_export_model.py`](../tests/test_export_model.py) exercises real
PyTorch modules, not replacements for the framework. Coverage includes the
fixed network and all ten saved NumPy inputs at all four stages, exact extracted
parameters, activation-first/only models, scalar and repeated layers, snapshot
independence, noncontiguous storage, finite edge values, invalid configurations,
connections, hooks, resource boundaries, and failure/retry behavior. It also
blocks file/checkpoint APIs and forward calls while validation runs.

[`tests/test_export_binary.py`](../tests/test_export_binary.py) checks the exact
168 bytes against an independent literal, the specification's hex dump, and the
saved binary. A test-only decoder checks headers, offsets, parameters, and exact
EOF before running all ten inputs against saved NumPy stages. Other cases cover
activation-only/shared layers, noncontiguous storage, bit-preserving finite edge
values, chunk boundaries, deterministic regeneration, and replacement semantics.
Fault injection covers every tiny-network write boundary, allocation failures
during writing, short writes, temporary creation, flush, sync, close, incorrect
total lengths, and replacement. Original destinations and temporary cleanup are
checked after failures, followed by a successful retry.

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

## Binary-export verification — October 24, 2026

On the same macOS Python 3.12.14 / PyTorch 2.14.1 environment:

- All 17 binary-export unittest methods passed, alongside the existing 33
  validation methods: 50 Python tests in total.
- The Debug build with fresh NumPy references and both exporter suites passed
  **23/23 CTest checks** (19 ordinary C tests, two fresh-reference tests, and
  two Python suites).
- The exported file matched all 168 bytes of the specification and the saved
  fixture, including every header/parameter offset and strict EOF. `xxd -g 1`
  inspection showed the same bytes; SHA-256 matched the recorded hash above.
- All 40 vectors / 120 stage values calculated from independently decoded
  parameters matched the saved NumPy references at `atol = rtol = 1e-5`, with
  matching shapes and finite values. All ten raw inputs and all stored expected
  outputs preserved their original binary32 representations exactly.
- Two complete regenerations matched all twelve saved files byte for byte.
  The standalone `--check` command passed; a deliberately changed model was
  detected without modifying that changed fixture.
- Failure injection confirmed unchanged/absent destinations, closed resources,
  temporary cleanup, and successful retry under the tested filesystem conditions.

These changes have not been pushed, so hosted macOS/Linux CI remains pending.
The existing Linux exporter option automatically registers both Python suites.
No C loader or new sanitizer/leak result is claimed. Big-endian execution and
crash/power-loss behavior were not tested; explicit little-endian encoding and
the filesystem limitations above define the intended behavior.

## Next step and references

The next scoped step is defining the C loader API, statuses, file handling, and
ownership/cleanup contract. Binary export and independent Python inspection are
implemented; these tests do not claim a C export/load/inference round trip.

The supported semantics were checked against the official PyTorch documentation
for [Sequential](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Sequential.html),
[Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html),
[ReLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ReLU.html), and
[Softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Softmax.html).
Binary encoding and replacement use Python's documented
[struct formats](https://docs.python.org/3/library/struct.html) and
[os.replace](https://docs.python.org/3/library/os.html#os.replace), with
[Tensor.view(dtype)](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.view.html)
for bit reinterpretation.
