# Model inspection

**Implemented October 31, 2026; included in v0.1.1, publication pending.** The
published v0.1.0 tag does not contain `info`. The current source reports
`Cinder 0.1.1`; the owner will create the tag and publish the release separately.

## Command

```text
cinder info <model.cnd>
```

Exactly one nonempty path is required. Check argument count before path
emptiness. Names are case-sensitive; paths are literal, including `-` or
`--help`. Relative paths use the working directory, suffixes are descriptive,
and normal host path/symlink resolution applies. Quote paths containing spaces.
There is no input file, inference, preprocessing, or timing operation.

Build and try the committed examples from the repository root:

```sh
cmake -S . -B build-info -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-info --parallel
ctest --test-dir build-info --output-on-failure
./build-info/cinder info tests/fixtures/tiny/tiny.cnd
./build-info/cinder info examples/mnist/mnist.cnd
```

Exact tiny-model output:

```text
Model: tiny.cnd
Input: 2
Output: 2
Layers: 4
  0: Dense 2 -> 4 (parameters: 12)
  1: ReLU 4 -> 4 (parameters: 0)
  2: Dense 4 -> 2 (parameters: 10)
  3: Softmax 2 -> 2 (parameters: 0)
Parameters: 22
```

`Input` and `Output` are vector lengths. Layers appear in execution order with
zero-based indices and their input/output sizes. Dense counts every weight and
bias slot, including zero values: `I * O + O`. The tiny model therefore has
`2 * 4 + 4 = 12` and `4 * 2 + 2 = 10` parameters. ReLU/Softmax count zero.
The committed MNIST model has 100480 + 1290 = **101770** parameters.
These are element counts, not file sizes, nonzero counts, or peak memory usage.

The report uses decimal integers without grouping, two spaces before layer
indices, and LF after every line, with no blank lines or tabs. It displays the
basename after the last `/`, escaping quotes and backslashes; control and
non-ASCII bytes use uppercase `\xHH`, matching `run` and `bench`. Paths do not
inject additional report lines. There is no JSON output or class-label metadata.

## Public metadata API

Declared in [`model.h`](../include/cinder/model.h), implemented in
[`model.c`](../src/model.c):

```c
typedef struct {
    size_t input_size, output_size, layer_count, parameter_count;
} cinder_model_info_t;

typedef struct {
    cinder_layer_type_t type;
    size_t input_size, output_size, parameter_count;
} cinder_layer_info_t;

cinder_status_t cinder_model_get_info(
    const cinder_model_t *model, cinder_model_info_t *info, cinder_error_t *error);
cinder_status_t cinder_model_get_layer_info(
    const cinder_model_t *model, size_t index,
    cinder_layer_info_t *info, cinder_error_t *error);
```

Both return value copies, expose no layer/parameter/buffer pointers, allocate
nothing, and perform no inference or I/O. Copies survive model destruction.
Whole-model inspection is O(layer count); one layer is O(1). Products, bias
addition, and aggregation use checked `size_t` arithmetic. Metadata inspection
does not scan parameter values; loading/construction validates the model first.

NULL model or output returns `INVALID_ARGUMENT`; index >= layer count returns
`OUT_OF_BOUNDS`; unrepresentable counts return `OVERFLOW`; an unknown layer
type returns `UNSUPPORTED`. Output is unchanged on every failure. Optional
error storage follows the existing bounded-message contract and is cleared
on success. Model lifetime, pointer validity, and intact immutable metadata
remain caller preconditions. Writable metadata/error storage must be disjoint
from each other and the model. Readers with separate output/error storage can
overlap inference, but must not race destruction or mutation.

## Ownership and failures

[`cli.c`](../src/cli.c) loads once with `cinder_model_load`, inspects summary and
layers, and stores the layer records in one checked allocation. It completes
inspection before stdout output, frees the model, prints the snapshot, and
frees that snapshot on every path. This keeps model-owned pointers private and
releases parameters/work buffers before reporting. There is no input or
prediction-buffer allocation. Normal model loading still constructs both
reusable work buffers: this is not a streaming or low-memory file inspector.

All existing [loader](LOADER.md) validation, resource limits, file handling,
and cleanup guarantees apply, including finite parameters and exact EOF.
One-layer and activation-only models are valid; no final Softmax is required.

| Exit | Meaning |
| --- | --- |
| 0 | Inspection, printing, and stdout flush succeeded; stderr is empty. |
| 1 | Loading, inspection, allocation, or output failed. |
| 2 | Invalid command or arguments; no file opened or model allocated. |

Usage reasons are `info requires exactly one file path.` and
`model path must not be empty.` The shared usage block is in [CLI.md](CLI.md).
Loader errors use `cinder: model "<escaped-path>" [STATUS]: <detail>`.
Metadata/snapshot errors use `cinder: inspection [STATUS]: <detail>`.
Snapshot allocation failure says `Could not allocate model metadata.`;
snapshot size overflow says `Model metadata byte count is not representable.`
All diagnostics end with LF and use existing bounded/escaped detail formatting.

Failures before printing leave stdout empty. A failed write or flush returns 1
and attempts `cinder: output [IO]: Could not write inspection output.` on
stderr; stdout may contain a partial report. Stderr failure cannot turn an
error into success. As with the existing commands, OS signals such as SIGPIPE
can terminate the process instead of returning a normal CLI exit status.

## Verification

The ordinary C suite uses committed tiny/MNIST files without Python or downloads.
`model_info` checks non-square counts, zero-valued parameters, activation-only
models, invalid arguments/indices, error details, and independent value copies.
The model allocation audit additionally checks zero allocations/frees/forwards
during inspection and synthetic arithmetic overflow/unknown-type failures.

CLI audits check exact reports and argument precedence; real subprocess tests
cover tiny, MNIST, single Dense layers, escaped paths, missing/malformed files,
unsupported versions, and non-finite parameters. The shared CLI acceptance
audit injects failures at every loader/constructor/snapshot allocation and
report write, plus flush and file-close failures, checks full cleanup, and
retries successfully. It also asserts no input reads, inference, or clock use,
including when the benchmark clock backend is unavailable.

### Actual local results — October 31, 2026

Verified the working tree based on `98dea44` on macOS 27.0.1 arm64 with Apple
Clang 21.0.0. All three builds enabled `-Wall -Wextra -Wpedantic` and produced
no compiler warnings.

| Configuration/check | Result |
| --- | --- |
| Debug, ordinary C suite (`build-info-debug`) | 39/39 passed. |
| Release with NumPy, exporter, and offline MNIST tests (`build-info-release`) | 48/48 passed, including all nine optional Python tests. |
| Debug ASan + UBSan (`build-info-sanitize`) | 39/39 passed; no sanitizer findings. |
| Native `leaks --atExit`: Release `info` on MNIST | Exit 0; zero leaks / zero leaked bytes. |
| Native `leaks --atExit`: controlled CLI acceptance audit | Exit 0; zero leaks / zero leaked bytes. |

The complete CLI audit passes **1,222 invocations** with the controlled clock
backend and **129** with the unavailable backend. For each backend, inspection
faults all **22 allocation points** for both tiny and MNIST, all **15 tiny / 16
MNIST report-write calls**, plus flush and close failures. Each call retains
zero tracked blocks, bytes, or files. Inspection makes zero input-reader,
inference, layer-forward, or clock calls. Metadata access itself makes zero
allocation/free calls; loading and the CLI snapshot intentionally allocate.

The Release build used the existing `.venv`, with
`CINDER_ENABLE_NUMPY_TESTS`, `CINDER_ENABLE_EXPORTER_TESTS`, and
`CINDER_ENABLE_MNIST_TESTS` all ON. The sanitizer build enabled
`CINDER_ENABLE_ASAN` and `CINDER_ENABLE_UBSAN`, with
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. LeakSanitizer is not supported
on this macOS configuration; the native leak checks above are separate.

Linux and hosted macOS/Linux CI were not run for this implementation review.
The existing workflow discovers the new CTest case without a workflow edit.
These results preceded the v0.1.1 version update. The v0.1.0 publication and
its historical verification remain unchanged.

### v0.1.1 version preparation — October 31, 2026

After committing the implementation as `c082f17`, update CMake's project
version to `0.1.1`. The executable, CLI audit, benchmark audit builds, and CTest
version expectation all derive their version from that one value. No separate
hard-coded test version needs changing.

Rebuild Debug, Release, and ASan/UBSan and rerun the affected command checks:
`cli_version`, `cli_audit`, `cli_process`, `benchmark_controlled`,
`benchmark_unavailable`, and `model_info`. **All six pass in each configuration**,
with no compiler warnings. Each executable prints exactly `Cinder 0.1.1\n`
with exit 0 and empty stderr. A negative control using `EXPECTED_VERSION=0.1.0`
fails as expected. The earlier full-suite and leak results above remain the
implementation evidence; this version-only follow-up reruns the relevant checks.

The README, current command guides, roadmap, and release notes identify v0.1.1
as prepared with publication pending. The published v0.1.0 notes are unchanged.
No push, tag creation, release publication, or hosted CI run was performed here.
