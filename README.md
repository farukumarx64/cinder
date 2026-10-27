# Cinder

A tiny neural-network inference runtime in C, built to understand how pretrained
models become forward-pass computations over arrays in memory.

Cinder loads sequential float32 models and executes Dense, ReLU, and Softmax
layers on the CPU, with results checked against a Python reference.
Training happens outside the runtime.

## Current status

**M0 through M4 are complete; M4 closed on October 27, 2026.** Tensor allocation,
cleanup, basic access, matrix-vector multiplication, bias addition, ReLU, and stable Softmax are
implemented and tested. The math operations match deterministic NumPy float32
references within the documented tolerance.
Phase 0 is complete, including the MIT license and verified development setup.
Phase 2 includes Dense creation/cleanup with owned parameter copies and a
tested, allocation-free forward pass. The common Dense/ReLU/Softmax layer
interface now has implemented constructors, metadata validation, cleanup, and
allocation-free dispatch. Model construction, connection validation, size
accessors, cleanup, and two reusable working buffers are implemented. Sequential
inference through `cinder_model_run` is implemented: Dense switches buffers,
activations run in place, and caller output changes only after full success.
Every stage of the complete example network now matches deterministic NumPy
float32 fixtures. The Phase 2 lifecycle audit passes locally and in macOS/Linux
CI, including failure recovery and caller-managed concurrency. See
[the verification report](docs/M2_VERIFICATION.md) for the exact revision,
results, and limitations. Phase 3's [`.cnd` version 1 contract](docs/MODEL_FORMAT.md)
is now defined, including exact byte layouts, limits, input files, and a worked
168-byte tiny-network example. [Exporter model validation](docs/EXPORTER.md)
checks supported PyTorch sequences and extracts independently owned CPU float32
parameter snapshots. Binary export now writes the version 1 format, with a saved
168-byte tiny model, ten inputs, and expected stage outputs verified against NumPy.
The [C loader](docs/LOADER.md) now validates `.cnd` files and constructs executable
models through the existing layer/model constructors. Tests cover successful
loads, saved tiny-network probabilities, malformed files, exact resource limits,
and forced allocation/file failures. The [complete round trip](docs/ROUND_TRIP.md)
now compares every saved stage from both committed fixtures and fresh Python
exports. [Loaded-model lifecycle verification](docs/M3_VERIFICATION.md) also
passes locally and in all three hosted CI jobs, closing M3. Phase 4's
[reproducible MNIST reference model](examples/mnist/README.md) is now trained:
**97.48% accuracy on the official 10,000-image test split**, with saved parameters
and [measured results](examples/mnist/RESULTS.md). Its trained
[`mnist.cnd`](examples/mnist/mnist.cnd) includes inference Softmax and passes
[deterministic export checks](examples/mnist/EXPORT.md). Saved
[reference fixtures](examples/mnist/FIXTURES.md) now cover test indices `0..99`
and every intermediate stage. [C inference validation](examples/mnist/VALIDATION.md)
passes all 27,600 saved stage values with `atol = rtol = 1e-5` and 100/100
prediction agreement; C and Python each classify 99/100 subset images correctly.
The [`cinder run` command](docs/CLI.md) is implemented with exact-length finite
float32 input reading, ranked output, contextual errors, and cleanup.
Start with the [offline MNIST demonstration](examples/mnist/DEMO.md).
[MNIST lifecycle verification](examples/mnist/LIFECYCLE.md) now passes locally and
in all five hosted CI jobs,
including failure recovery, independent owners, poisoned workspaces, and all
construction allocation failures. Hosted verification completed October 27, 2026.
The CLI supports `--version` and `run`. The new CLI passes local Debug 38/38,
C-only Release 29/29, ASan/UBSan 29/29, and native leak checks. All five hosted
jobs passed for `7aa4bda`: Linux Debug 38/38; macOS Debug/Release, Linux Release,
and Linux sanitizer/leak checks 29/29 each. The
[demonstration report](examples/mnist/DEMO.md#hosted-verification-and-m4-closure)
records the exact revision, results, GCC warnings, and remaining limitations.

- C11 compilation with `-Wall`, `-Wextra`, and `-Wpedantic` is configured.
- A local Git repository on `main` tracks source and documentation. Generated
  build files are ignored.
- The macOS build was verified on Apple Silicon with Apple Clang 21.0.0 and
  CMake 4.4.4. GitHub Actions also verified builds and tests on macOS 15 and
  Ubuntu 24.04.
- CTest provides the version-command smoke test, two tensor allocation tests,
  and tests for tensor access, matrix-vector multiplication, bias addition,
  ReLU, Softmax, and NumPy reference fixtures, plus Dense and common layer tests.
- Opt-in AddressSanitizer and UndefinedBehaviorSanitizer builds passed the
  smoke test on macOS, both separately and together.
- The public GitHub repository is [farukumarx64/cinder](https://github.com/farukumarx64/cinder).
  The local `main` branch tracks `origin/main` over SSH.
- GitHub Actions passed all three jobs: macOS, Linux, and Linux with ASan,
  UBSan, and leak detection enabled.
- The documented macOS setup was verified from a fresh GitHub clone of
  `1190c0e` on October 8, 2026. The ordinary build, sanitizer build, smoke
  tests, native leak check, and build without tests all passed.
- The [tensor contract](docs/TENSOR.md) defines the representation, ownership,
  indexing, and error behavior. Creation and cleanup are implemented, including
  size validation, zero initialization, and allocation failure handling.
- Checked `fill`, `get`, and `set` support row-major indexing, reject invalid
  indices and non-finite values, and preserve data or outputs on failure.
- `cinder_tensor_matvec` computes `y = Wx` with exact input-length and minimum
  output-capacity checks. It rejects non-finite sources and running sums,
  requires separate output storage, and allocates no memory.
- `cinder_tensor_add_bias` adds one bias per output in place, completing
  `y = Wx + b` after matvec. It checks equal, nonzero vector lengths, finite
  sources and results, and does not allocate memory.
- `cinder_tensor_relu` replaces negative finite values with zero in place.
  It preserves positive values and existing zeros, validates the full vector
  before writing, and leaves it unchanged on any failure.
- `cinder_tensor_softmax` subtracts the maximum score, exponentiates, and
  normalizes in place without allocating. It uses double intermediates with
  float32 storage, accepts underflow to zero, and rejects empty/non-finite
  inputs before changing the vector.
- At Phase 1 completion on October 16, 2026, all 9 ordinary Debug tests passed
  locally. Release and combined ASan/UBSan builds with fresh NumPy generation
  passed all 10 tests.
  Reference tests compared 79 vectors / 341 results with the unchanged
  `atol = rtol = 1e-5` rule. The largest absolute error was `1.1920929e-7`.
  See [the reference test guide](docs/NUMPY_TESTS.md) for coverage and limits.
- The Softmax and NumPy reference executables reported zero native leaks on
  October 16, as did access, matrix-vector, bias, and ReLU tests on October 15 and the allocation
  tests on October 9.
- The [Dense-layer contract](docs/DENSE.md) specifies dimensions, row-major
  weights, a bias vector, owned parameter copies, public functions, buffer
  overlap rules, and validation versus arithmetic failure behavior.
- `cinder_dense_create` validates dimensions, allocation sizes, exact parameter
  counts, and finite sources before allocating. It copies parameters into
  owned tensors; `cinder_dense_free` releases the layer and both tensors.
  Tests force failure at all five allocation points and verify full cleanup.
- After adding Dense creation/cleanup on October 16, all 11 ordinary Debug
  tests passed. Release and ASan/UBSan builds with generated NumPy references
  enabled passed 12/12. Both new Dense executables reported zero native leaks;
  a build without tests also passed.
- `cinder_dense_forward` computes `y = Wx + b` through the existing matvec and
  bias functions. It validates lengths/capacity and all sources before writes,
  preserves input/parameters, and leaves spare output capacity untouched.
  Validation failures preserve the output; arithmetic failures can leave partial
  results that must be discarded. No allocation or free occurs during execution.
- After adding Dense forward execution on October 16, Debug passed 12/12 tests;
  Release and ASan/UBSan with generated tensor NumPy references enabled passed
  13/13. Dense matches all 11 saved NumPy matrix/bias fixtures. The forward and
  extended allocation-failure executables reported zero native leaks; a build
  without tests passed.
- The [common layer contract](docs/LAYER.md) defines a type tag, shared
  input/output dimensions, an owned Dense-only payload, constructors, metadata
  validation, cleanup, and forward execution. Activations support exact in-place
  input/output aliasing or separate buffers; Dense needs separate buffers. Activations must
  preserve vector length. These functions are implemented in `src/layer.c`.
- Common dispatch reuses the existing Dense, ReLU, and Softmax functions.
  Activations validate all input values before writing, copy separate input into
  output when needed, and preserve dimensions and spare capacity. Each activation
  owns one wrapper and no parameters or working buffers.
- After adding common layer execution on October 21, Debug passed 14/14 tests;
  Release and combined ASan/UBSan with generated tensor NumPy references enabled
  passed 15/15. Layer dispatch matches 35 saved NumPy reference vectors. Tests
  force every constructor allocation failure and verify allocation-free calls.
  Both new executables reported zero native leaks; a build without tests and
  the linked documentation example passed.
- The [sequential-model contract](docs/MODEL.md) and [public declarations](include/cinder/model.h)
  define an opaque, nonempty, ordered model. Creation validates connections and
  transfers layer ownership only on success, clearing the supplied array slots.
  Headers and examples passed C11 syntax checks on October 22, 2026, before implementation.
- `cinder_model_create` now validates all layer metadata, duplicate pointers, and
  adjacent dimensions before allocating. It owns a private pointer list and two
  working buffers sized to the largest layer input/output vector. The example
  `2 -> 4 -> 2` model has two four-float buffers. Size getters expose external
  dimensions; `cinder_model_free` releases all adopted layers and model storage.
- On October 22, model construction/cleanup passed the full Debug suite (16/16)
  and Release plus combined ASan/UBSan suites with generated tensor NumPy
  references enabled (17/17 each). Tests force all four model allocation failures,
  verify rollback and ownership, and check private buffer sizes. Both new tests
  reported zero native leaks; a build without tests and the linked construction
  example passed.
- `cinder_model_run` validates pointers, lengths, capacity, and finite input,
  copies input to private storage, and dispatches layers in order. Dense uses
  the other working buffer; ReLU/Softmax update the current vector in place.
  The first layer failure stops execution with its original diagnostic. Only
  full success copies the logical result to caller output; spare capacity and
  input remain untouched. Repeated runs need no reset, including after failure,
  and allocate/free nothing. Calls on one instance must be serialized.
- On October 22, sequential inference passed Debug (17/17), Release and combined
  ASan/UBSan with generated tensor NumPy references enabled (18/18 each).
  Model-run, model-storage, and layer tests reported zero native leaks. The
  build without tests and linked model/layer documentation examples passed.
- The complete `2 -> 4 -> 2` network now matches independent float32 NumPy
  references for ten fixed inputs. First Dense, ReLU, final logits, and Softmax
  are checked through both common layers and sequential models, using shared
  shape/finite checks and the unchanged `atol = rtol = 1e-5` rule. Saved fixtures
  retain parameters, inputs, and every expected stage for the future file round trip.
- On October 23, Debug passed 18/18 tests; Release and combined ASan/UBSan with
  fresh NumPy generation passed 20/20. Each new network test compared 80 vectors
  / 240 values; the maximum absolute error was `5.96046448e-8`. Regeneration was
  identical locally and preserved all existing tensor arrays. The network test
  reported zero native leaks, and a build without tests passed.
- On October 23, the lifecycle audit passed Debug (19/19), Release and combined
  ASan/UBSan with fresh references (21/21 each). It exercises 4,120
  success/failure/recovery cycles, independent models, and externally serialized
  shared-model threads. Poisoned buffers reproduce baseline results; inference
  performs zero allocation/free attempts. Both lifecycle and model-storage
  executables reported zero native leaks. See [the M2 report](docs/M2_VERIFICATION.md).
- Hosted CI for `f5424a4`, including the NumPy and lifecycle additions, passed
  19/19 tests on macOS, 21/21 on Linux with fresh NumPy references, and 19/19
  under Linux ASan/UBSan/leak detection. Job statuses and test logs in
  [run 39984963408](https://github.com/farukumarx64/cinder/actions/runs/39984963408)
  were inspected on October 23, completing M2.
- The loader API `cinder_model_load(path, &model, &error)` is implemented and
  declared in `model.h`; it returns an independently owned model only after full file
  validation and successful close. On failure it leaves the output NULL and
  cleans up staging and partial layers. Four new statuses distinguish I/O,
  format, unsupported version/platform, and resource-limit failures. Successful-load
  and parser tests are included; see [the loader guide](docs/LOADER.md).
- On October 24, loader implementation passed 24/24 Debug tests with NumPy and
  exporter checks; C-only Release and ASan/UBSan passed 20/20 each. The loader test
  initially covered 206 rejection cases and 50 repeated NumPy output comparisons,
  and reported zero native leaks. The subsequent rejection/cleanup audit expands
  this to 709 cases, forces all 21 tiny-model allocations and 29 file-failure
  scenarios, and checks cleanup after 1,562 rejected calls. Debug passed 25/25
  tests; C-only Release and ASan/UBSan passed 21/21 each. Both loader executables
  reported zero native leaks; hosted verification subsequently passed at M3 closure. See the dated results
  and platform limitations in the [loader guide](docs/LOADER.md#rejection-and-cleanup-audit--october-24-2026).
- On October 25, the complete export/load/inference comparison passed all four
  stages for ten inputs: 40 vectors / 120 values per saved or fresh-export run.
  Dense/ReLU/logits matched exactly locally; maximum Softmax error was
  `5.96046448e-8` with `atol = rtol = 1e-5`. The default C test uses committed
  files without Python; the exporter option adds live export and nine deliberate
  fixture-change rejection checks. See [round-trip verification](docs/ROUND_TRIP.md).
- On October 25, loaded lifecycle tests passed 502 successful loads, 100 rejected
  loads, and 14,160 inference calls. The tracked audit additionally verifies
  poisoned workspaces, no inference allocation/free attempts, unchanged parameters,
  and recovery from all 21 allocation failures while another model remains alive.
  Debug and ASan/UBSan passed 28/28 tests; C-only Release passed 23/23. Both native
  leak checks reported zero leaks. Hosted CI for `5ad7dc4` passed 23/23 macOS,
  28/28 Linux, and 23/23 Linux sanitizer/leak tests, closing M3. See
  [M3 verification](docs/M3_VERIFICATION.md) for the exact revision, run, and limitations.
- The external MNIST example implements CPU float32 `784 -> 128 -> 10` training,
  logits with CrossEntropyLoss, shared row-major `/255` preprocessing, fixed
  seeds/settings, checked IDX data loading, and checkpoint/metadata saving.
  The fixed five-epoch run trained on 60,000 official training images and scored
  9,748/10,000 (97.48%) on the held-out test split. Fresh-process reload and repeat
  training checks reproduced the results. Training-script tests still use synthetic
  data; see [the measured results and saved checkpoint](examples/mnist/RESULTS.md).
- The saved checkpoint exports through the existing writer as a 407,160-byte
  Dense/ReLU/Dense/Softmax model. Independent processes produce identical bytes,
  and every stored parameter bit matches the checkpoint. See
  [the MNIST export guide](examples/mnist/EXPORT.md).
- Deterministic MNIST fixtures preserve 100 test images' indices, labels, raw
  preprocessed inputs, and every Dense/ReLU/logits/probability vector. All 108
  files regenerate identically in the recorded environment. Offline tests use
  the saved selected pixels and checkpoint, without downloads. See
  [the fixture guide](examples/mnist/FIXTURES.md).
- Every saved MNIST stage passes C comparisons. Maximum absolute errors are
  `6.19888306e-6` (Dense1), `5.7220459e-6` (ReLU/logits), and `3.57627869e-7`
  (probabilities). Prediction agreement is 100/100; classification accuracy on
  that fixed subset is 99/100 for both runtimes, separate from Python's 97.48%
  full-test result. Debug passed 33/33 tests; C-only Release and ASan/UBSan each
  passed 24/24. Native leaks reported zero. See [the validation report](examples/mnist/VALIDATION.md)
  for fresh-export checks, 28 rejection controls, reproduction, and hosted results.
- MNIST lifecycle tests exercise 2,100 public-API and 1,384 instrumented inference
  calls. All 21 allocation points fail cleanly with a surviving model and a
  successful retry. Workspaces are independent, inference cycles make zero
  allocation/free attempts, and parameters stay unchanged. Local Debug passed
  35/35 tests; C-only Release and ASan/UBSan passed 26/26 each; both native leak
  checks reported zero. See [the lifecycle report](examples/mnist/LIFECYCLE.md)
  for evidence: Linux Debug passes 35/35 hosted tests; macOS Debug/Release,
  Linux Release, and Linux sanitizers each pass 26/26. Native hosted macOS leak
  checking did not complete; local native checks and hosted Linux LeakSanitizer pass.
- The `info` and `bench` commands remain planned features.
- The [benchmark contract](docs/BENCHMARK.md), defined October 27, specifies
  required `--iterations N` (1–1,000,000), ten fixed warm-ups, per-call monotonic
  inference timing, checked statistics, exact reporting, and failure behavior.
  The command is not implemented and no performance results have been measured.
- [Shared CLI preparation](docs/BENCHMARK.md#preparation-implementation-and-verification)
  now loads the model once, reads input once, allocates output, and owns cleanup.
  `run` uses it with unchanged behavior. Tiny/MNIST tests cover reuse without
  file access, all 23 allocation failures per fixture, and recovery. Local Debug
  passes 39/39 tests; C-only Release and ASan/UBSan pass 30/30 each; native setup
  and actual-command leak checks report zero. These changes have not run in
  hosted CI yet.
- The [monotonic timer](docs/BENCHMARK.md#monotonic-timer-implementation-and-verification)
  now uses `CLOCK_MONOTONIC_RAW` on macOS/Linux, with checked resolution,
  timestamps, integer elapsed times, failure recovery, and zero-duration
  rejection. This Mac reports 42 ns resolution. Local Debug passes 42/42 tests;
  C-only Release and ASan/UBSan pass 33/33 each; native timer leak checks report
  zero. Linux execution and hosted CI for this step remain unverified. The
  warm-up and measured inference loop are still pending.
- The [`run` contract](docs/CLI.md) now defines two positional file paths, a
  zero-based prediction and up to three ranked raw scores, lower-index tie
  breaking, exit statuses `0/1/2`, and contextual errors. It accepts valid models
  with or without Softmax. The implementation and three new C tests now pass
  locally and in all five CI jobs; see [the demonstration report](examples/mnist/DEMO.md).

See [the roadmap](ROADMAP.md#current-progress) for the current checklist and
next scoped task.

## Prerequisites

- CMake 3.16 or newer.
- A C11-capable compiler, such as Apple Clang, Clang, or GCC.
- A build tool supported by CMake, such as Make or Ninja. The verified macOS
  setup uses Make from the Xcode Command Line Tools.
- Git for version control.

Python, NumPy, and PyTorch are optional for the C build and its saved reference tests.
Regenerating references requires Python 3.11+ and NumPy; the tested dependency
is pinned in `tests/requirements-numpy.txt`.
Exporter validation requires Python 3.11+ and PyTorch, pinned separately in
`tools/requirements-exporter.txt`; see [the exporter guide](docs/EXPORTER.md).
Softmax requires `double` to have wider range and precision than `float`,
checked at compile time. CMake links the system math library (`m`) on Unix
for exponentiation; no extra third-party library is needed.

## Build and run

From the project root:

```sh
cmake -S . -B build
cmake --build build
./build/cinder --version
```

The first command configures the project and generates build files in `build/`.
The second compiles and links the executable. The third prints:

```text
Cinder 0.1.0-dev
```

The version number comes from CMake's project version, with `-dev` appended for
the development build. Run the saved MNIST model without Python or downloads:

```sh
./build/cinder run examples/mnist/mnist.cnd tests/fixtures/mnist/inputs/00000.f32
```

This predicts digit 7 and prints the three highest scores. See the
[demonstration](examples/mnist/DEMO.md) for actual output and measured results.
The [command contract](docs/CLI.md) defines input layout, class mapping, ranking,
and errors. Normal exit statuses are `0` for success, `1` for operational failure,
and `2` for invalid usage. Errors go to stderr; failures before printing leave
stdout empty.

Keep generated files in a separate build directory. The ignore rules cover
`build/`, `build-*/`, and `cmake-build-*/`, along with common generated CMake files.

## Run the tests

CTest ships with CMake. Tests are enabled by default through `BUILD_TESTING`.
Configure and build the project before running the tests:

```sh
cmake -S . -B build
cmake --build build
(cd build && ctest --output-on-failure)
```

The `cli_version` test runs `cinder --version` and requires exit status `0`,
exactly `Cinder 0.1.0-dev` followed by a newline on standard output, and empty
standard error. The expected version follows the CMake project version.
Failures show diagnostics; the test has a ten-second timeout.

The runtime tests cover tensor math and memory, layers, model ownership, and inference:

- `tensor_allocation` checks dimensions, zero initialization, writable storage,
  independent tensors, invalid arguments, both size-overflow cases, repeated
  creation/cleanup, and optional error details against the `cinder_core` library.
- `tensor_allocation_failures` forces the first and second allocations to fail,
  checks partial cleanup and NULL outputs, and confirms validation happens
  before allocation. It also checks that access, matvec, bias addition, ReLU,
  and Softmax do not allocate.
  It compiles the same implementation with allocator
  substitutions confined to the test; production has no failure switches.
- `tensor_access` checks fills across several shapes, reads and writes against
  known row-major storage, NULL pointers, boundary and huge indices, NaN and
  infinities, unchanged data/outputs on error, and optional error details.
  Finite extremes, subnormal values, and signed zero are also exercised.
- `tensor_matvec` checks a hand-calculated non-square product, decimal values,
  identity/zero cases, single-row/column shapes, input lengths, output capacity,
  unchanged sources and spare output elements, non-finite values, and arithmetic
  overflow. It uses `atol = 1e-5` and `rtol = 1e-5` for numeric comparisons.
- `tensor_bias` checks in-place addition, vector lengths, byte-count overflow,
  non-finite sources and sums, failure preservation, partial updates on
  arithmetic overflow, unchanged bias and spare output storage, and error
  details. Integration cases verify `Wx + b` and a zero input whose result is
  exactly the bias. It uses the same numerical tolerance rule as matvec.
- `tensor_relu` checks negative/zero/positive values, finite extremes, signed
  zeros, subnormal values, invalid counts and pointers, non-finite values,
  unchanged data on failure, and untouched spare capacity. It also checks
  repeated application and the full `ReLU(Wx + b)` sequence. Expected values
  are compared exactly because ReLU introduces no rounding arithmetic.
- `tensor_softmax` checks known probabilities, large score offsets, equal and
  extreme values, tiny probabilities and allowed underflow, and a long vector
  with many small contributions to its sum. It also checks empty/non-finite
  input rejection, unchanged arrays on validation failure, optional errors,
  spare capacity, and composed `Softmax(ReLU(Wx + b))` results. Expected values
  use independent constants, with tighter checks for tiny probabilities.
- `tensor_numpy_fixtures` compares each math operation and composed intermediate
  results with saved NumPy float32 references. It checks shapes and finiteness
  before applying `atol = rtol = 1e-5`, and reports the failing fixture, stage,
  index, values, and error limit. Coverage includes 11 matrix/bias fixtures and
  12 standalone activation fixtures, for 79 result vectors / 341 values.
- `dense_allocation` checks shape/count validation, owned parameter copies,
  source independence, and cleanup. `dense_allocation_failures` forces each of
  five constructor allocations to fail, tracks partial cleanup, and verifies
  zero allocations/frees during successful and failing forward calls.
- `dense_forward` checks the `2 -> 4` worked example, repeated inputs, source
  preservation, exact/spare capacity, validation and arithmetic failures,
  recovery, and permitted storage arrangements. It also compares the layer's
  output against all 11 saved NumPy matrix/bias fixtures (41 output values).
- `layer_interface` checks Dense/ReLU/Softmax dispatch, constructor ownership,
  metadata validation, unchanged activation dimensions and sources, exact/spare
  capacity, rejection before writes, diagnostics, recovery, repeated calls,
  and in-place activation execution.
  It compares 12 saved NumPy fixtures through each activation and 11 through
  Dense. `layer_allocation_failures` forces all six Dense-wrapper allocation
  failures and each activation's one allocation failure, tracks cleanup, and
  verifies that validation and forward execution allocate/free nothing.
- `model_allocation` checks construction validation, connection mismatches,
  ownership transfer, dimension getters, source-list lifetime, and independent
  models. `model_allocation_failures` forces all four model allocations to fail,
  verifies caller layers survive rollback, checks ordered adoption and exact
  private buffer sizes, and tracks complete cleanup with no new allocations.
  Its inference checks observe buffer choices and logical counts, poison spare
  tails, inject partial-write failures at each layer, verify immediate stopping,
  and check zero allocations/frees through success, failure, and recovery.
- `model_run` checks hand-calculated network prefixes and probabilities,
  consecutive Dense layers, activation-first/only models, single layers,
  shrinking/widening vectors, input/parameter preservation, exact/spare capacity,
  validation order, independent models, and repeated execution. Real arithmetic
  overflow at early/middle/late layers preserves caller output and diagnostics.
- `model_numpy_fixtures` compares all four stages of the fixed `2 -> 4 -> 2`
  network against NumPy for ten deterministic inputs. It checks a direct layer
  sequence and models ending at each stage, including complete inference,
  while preserving inputs/parameters and spare capacity. The tensor and model
  reference tests share `numpy_compare.h` for shape, finiteness, tolerance,
  failure diagnostics, and measured error statistics.
- `model_lifecycle` runs the full network through 4,120 success/failure/recovery
  cycles, compares results with saved NumPy fixtures, and checks independent
  models before/after destruction. On macOS/Linux it also tests concurrent
  independent models and externally locked shared-model calls. POSIX Threads
  is a dependency of this test only, not of `cinder_core`.
- `model_round_trip_fixtures` loads the full binary model and three exported
  prefixes, reads the ten raw input files, and compares every stage with saved
  NumPy references through the public model API. It checks shapes, finiteness,
  tolerance, unchanged inputs, and spare output capacity without needing Python.
- `model_loaded_lifecycle` repeats the lifecycle checks with file-loaded models,
  adds repeated and concurrent load/run/free cycles, and verifies recovery after
  failed loads and continued use after deleting the source file. The tracked
  `loader_failures` test also checks loaded-model buffer/parameter independence,
  zero inference allocation/free attempts, and construction rollback while an
  existing owner remains usable.

Most runtime tests have ten-second timeouts; the two loader boundary/failure
suites allow 60 seconds for their large fixtures. Checks stay active in Release
builds, where ordinary C `assert` checks could be disabled by `NDEBUG`.

To configure a build without tests, pass `-DBUILD_TESTING=OFF` to CMake.

### Generate fresh NumPy references

The default build uses the saved fixture header without running Python. To
also compare against newly generated references, use Python 3.11 or newer
(for example, `python3.12`) and a local virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements-numpy.txt
cmake -S . -B build-numpy \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCINDER_ENABLE_NUMPY_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-numpy
ctest --test-dir build-numpy --output-on-failure
```

This adds `tensor_numpy_generated` and `model_numpy_generated`, using one
generated header inside the build directory. The saved fixtures stay unchanged. CMake fails
clearly if NumPy is missing or testing is disabled. See
[NumPy reference verification](docs/NUMPY_TESTS.md) for the formulas, fixture
generation command, reproducibility policy, and recorded results.

### Validate and export Python models

`tools.export_model.validate_model(model, *, input_size=None)` checks a flat
`torch.nn.Sequential` of biased Linear, ReLU, and `Softmax(dim=-1)` layers and
returns independent contiguous CPU float32 parameter copies. It validates
dimensions, connections, finite values, supported configurations, and format
limits before any file-writing stage. It performs no model execution or file I/O.
See [the exporter guide](docs/EXPORTER.md) for the keyword-only input size,
supported subset, result ownership, and diagnostics.

`tools.export_model.export_model(model, path, *, input_size=None)` validates the
source model and writes the complete `.cnd` file. It writes a temporary file in
the destination directory, verifies its length, flushes/syncs/closes it, then
replaces the destination. Failed validation or writing preserves an existing
destination. The parent directory must already exist. See the exporter guide
for replacement, permissions, and cleanup details.

```python
from tools.export_model import export_model

export_model(model, "model.cnd")
```

Using Python 3.11 or newer, set up the optional environment on macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r tools/requirements-exporter.txt
.venv/bin/python -m unittest discover -s tests -p 'test_export_*.py' -v
```

On Linux, use the official CPU-wheel index for that installation:

```sh
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -r tools/requirements-exporter.txt
```

To include the Python suite in CTest:

```sh
cmake -S . -B build-exporter -DCMAKE_BUILD_TYPE=Debug \
  -DCINDER_ENABLE_EXPORTER_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-exporter --parallel
ctest --test-dir build-exporter --output-on-failure
```

The option defaults to `OFF`. CMake requires an importable PyTorch and enabled
testing when it is selected, and does not install Python dependencies. The
exporter tests consume saved reference arrays without requiring NumPy.

The option also registers `model_round_trip_exported`, which exports fresh
models/inputs into a temporary directory and invokes the same C stage comparator.
Both round-trip paths use the saved NumPy answers. See [the guide](docs/ROUND_TRIP.md).

The independent `CINDER_ENABLE_MNIST_TESTS=ON` option adds the offline
`mnist_training`, `mnist_export`, `mnist_fixtures`, and `mnist_inference_exported`
suites using the same
PyTorch dependency. They test synthetic training, checkpoint handling, export,
and every saved image/stage reference using the selected pixels and checkpoint.
The fresh-export test runs the C comparator and corruption controls against
saved answers. The ordinary `mnist_inference_fixtures` C test also runs with
all Python options OFF. No dataset is downloaded.
Both options default to OFF. See
[MNIST setup and test commands](examples/mnist/README.md).

The fixed binary model, three stage prefixes, ten raw float32 input files, and
exact expected stage outputs are saved under [`tests/fixtures/tiny`](tests/fixtures/tiny).
Regenerate them explicitly or check for drift without changing saved files:

```sh
.venv/bin/python tests/generate_cnd_fixtures.py
.venv/bin/python tests/generate_cnd_fixtures.py --check
xxd -g 1 tests/fixtures/tiny/tiny.cnd
```

## Sanitizer builds

Sanitizers add checks that run while the program executes. Use a GCC or Clang
toolchain with its sanitizer runtime libraries installed. Both options default
to `OFF` and can be enabled independently:

| CMake option | Purpose |
| --- | --- |
| `CINDER_ENABLE_ASAN` | Detect memory errors such as out-of-bounds access and use-after-free. |
| `CINDER_ENABLE_UBSAN` | Detect undefined behavior such as signed integer overflow and invalid pointer alignment. |

For development, enable both in a separate Debug build:

```sh
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCINDER_ENABLE_ASAN=ON \
  -DCINDER_ENABLE_UBSAN=ON
cmake --build build-sanitize
(cd build-sanitize && ctest --output-on-failure)
```

Shared CMake settings apply warning and sanitizer flags to the core library,
CLI, and all test executables, including sanitizer flags at executable link
time. Sanitizer builds retain frame pointers for clearer stack traces. UBSan
uses `-fno-sanitize-recover=undefined` so a detected error terminates the program
with a failure instead of continuing. CTest reports any failing test.

CMake remembers these options per build directory. Use `build/` for ordinary
builds and `build-sanitize/` for sanitizer builds; explicitly set either option
to `OFF` when disabling it in an existing build directory. Benchmark with
sanitizers disabled because their checks add runtime overhead.

The current tests exercise the version command, tensor math and memory,
Dense/common layer execution, model construction/cleanup, and sequential inference.
NumPy reference comparisons cover tensor math and every stage of the tiny network.
Passing the current tests is not evidence that future code is free of memory errors.

See the upstream [AddressSanitizer documentation](https://clang.llvm.org/docs/AddressSanitizer.html)
and [UndefinedBehaviorSanitizer documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
for the supported checks and runtime options.

### Leak checks

Leak detection depends on the toolchain. On the verified Apple Silicon setup
with Apple Clang 21.0.0, setting `ASAN_OPTIONS=detect_leaks=1` reports that leak
detection is unsupported. The default macOS sanitizer command above does not
force that setting.

On macOS, run the native `leaks` tool against the ordinary build:

```sh
leaks --atExit -- ./build/cinder --version
leaks --atExit -- ./build/test_tensor
leaks --atExit -- ./build/test_tensor_alloc_failures
leaks --atExit -- ./build/test_tensor_access
leaks --atExit -- ./build/test_tensor_matvec
leaks --atExit -- ./build/test_tensor_bias
leaks --atExit -- ./build/test_tensor_relu
leaks --atExit -- ./build/test_tensor_softmax
leaks --atExit -- ./build/test_tensor_numpy
```

These commands reported zero leaks for the version command and tensor test
executables during their respective verification steps. The tool needs
permission to inspect the process it launches; a restrictive sandbox can
block it. An inspection error is not a successful leak check.

On Linux with a LeakSanitizer-capable toolchain, explicitly enable leak checks
when running the sanitizer build's tests:

```sh
(cd build-sanitize && ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure)
```

The Linux sanitizer CI job passed all 19 tests using saved NumPy references,
with leak detection enabled at M2 closure; see [the recorded results](docs/M2_VERIFICATION.md).
See the
[LeakSanitizer documentation](https://clang.llvm.org/docs/LeakSanitizer.html).

## Continuous integration

[The CI workflow](.github/workflows/ci.yml) runs on pushes to `main`, pull
requests, and manual dispatch from the repository's Actions tab.

| Job | Runner | Checks |
| --- | --- | --- |
| Linux Debug | Ubuntu 24.04 | Configuration, compilation, and CTest, including optional NumPy, exporter, and offline MNIST suites. |
| Linux Release | Ubuntu 24.04 | Release configuration, compilation, and ordinary CTest tests using committed fixtures. |
| macOS Debug | macOS 15 | Debug configuration, compilation, and ordinary CTest tests. |
| macOS Release | macOS 15 | Release configuration, compilation, and ordinary CTest tests. |
| Linux sanitizers | Ubuntu 24.04 with Clang | Debug build with ASan and UBSan, then CTest with leak detection enabled. |

Each job starts from a fresh checkout, prints its toolchain versions, and fails
if configuration, compilation, or testing fails. CTest also fails if no tests
are registered. Jobs have a ten-minute limit, and newer runs replace older runs
for the same branch or pull request.

All jobs test the saved NumPy and MNIST fixtures. The Linux Debug job additionally
creates a Python virtual environment, installs the pinned NumPy and CPU PyTorch
dependencies, and enables `CINDER_ENABLE_NUMPY_TESTS`,
`CINDER_ENABLE_EXPORTER_TESTS`, and `CINDER_ENABLE_MNIST_TESTS`. Ordinary C tests
require no Python or MNIST download. The preceding numerical-validation commit
passed [all three previous jobs](https://github.com/farukumarx64/cinder/actions/runs/40318184929).
The expanded matrix and lifecycle tests passed
[all five jobs for `bd6cbd9`](https://github.com/farukumarx64/cinder/actions/runs/40417063563)
on October 27, 2026; see [the report](examples/mnist/LIFECYCLE.md).

The workflow uses read-only repository permissions and a pinned checkout action.
Check results and failure logs in [GitHub Actions](https://github.com/farukumarx64/cinder/actions/workflows/ci.yml).
All three jobs passed in the [first hosted run](https://github.com/farukumarx64/cinder/actions/runs/37680825782)
on October 7, 2026.
LeakSanitizer is enabled in the Linux sanitizer job. Native macOS
`leaks --atExit` passed locally on both MNIST lifecycle executables. Its hosted
macOS step did not complete and was removed; the [lifecycle report](examples/mnist/LIFECYCLE.md)
records that limitation and local reproduction commands.

## Repository layout

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Core library and executable targets, shared compiler/sanitizer settings, and CTest registration. |
| `src/cli.c` | Version/run dispatch, model execution, ranking, escaped output, diagnostics, and cleanup. |
| `src/cli_session.c`, `src/cli_session.h` | CLI-private reusable model/input/output preparation, ownership, failure context, and cleanup. |
| `src/timer.c`, `src/timer.h` | CLI-private raw monotonic clock, checked resolution/timestamps/elapsed nanoseconds, and start/stop state. |
| `src/input.c`, `src/input.h` | CLI-private exact-length finite little-endian float32 reader. |
| `tests/test_input.c`, `tests/test_cli.c`, `tests/test_cli_process.c` | Reader/CLI failure audits and actual command processes against saved fixtures. |
| `tests/test_cli_setup.c` | Real shared preparation, repeated inference without file access or allocations, forced failures, and cleanup/recovery. |
| `tests/test_timer.c`, `tests/test_timer_unavailable.c`, `tests/test_timer_native.c` | Controlled clock boundary/failure checks, unavailable-backend rejection, and native clock smoke test. |
| `examples/mnist/DEMO.md` | Offline demonstration, reproduction commands, actual output, measured results, limitations, and M4 verification gate. |
| `src/tensor.c` | Tensor allocation, cleanup, checked access, matrix-vector multiplication, bias addition, ReLU, Softmax, and error reporting. |
| `src/dense.c` | Dense creation, parameter copying, cleanup, and forward execution through tensor kernels. |
| `src/layer.c` | Common constructors, metadata validation, cleanup, and Dense/ReLU/Softmax dispatch. |
| `src/model.c` | Private model storage, ordered adoption, connection validation, reusable buffers, sequential inference, size getters, and cleanup. |
| `src/loader.c` | Checked `.cnd` decoding, file handling, layer/model construction, and partial-load cleanup. |
| `include/cinder/tensor.h` | Tensor representation and function declarations. |
| `include/cinder/dense.h` | Dense representation and implemented create/free/forward API declarations. |
| `include/cinder/layer.h` | Common layer tags, dimensions, payload, and implemented API declarations. |
| `include/cinder/model.h` | Opaque model with implemented create/load/size/run/free APIs. |
| `include/cinder/error.h` | Status codes and caller-owned error details. |
| `docs/TENSOR.md` | Tensor layout, ownership, validation, and failure contract. |
| `docs/DENSE.md` | Dense parameter shapes, ownership, public API, overlap, failures, and worked example. |
| `docs/LAYER.md` | Common layer representation, ownership, dimension validation, dispatch, and buffer/failure contract. |
| `docs/MODEL.md` | Model ownership transfer, connections, dimensions, buffers, inference failures, and concurrency contract. |
| `docs/LOADER.md` | C loader API, file/status rules, constructor ownership, cleanup, implementation results, and remaining checks. |
| `docs/MODEL_FORMAT.md` | `.cnd` version 1 byte layout, layer IDs, limits, validation, input-file contract, and worked example. |
| `docs/CLI.md` | Implemented `cinder run` syntax, score formatting, class mapping, exit statuses, diagnostics, cleanup, and implementation acceptance cases. |
| `docs/BENCHMARK.md` | Planned benchmark contract, acceptance cases, and implemented shared preparation/timer with verification results. |
| `docs/EXPORTER.md` | Python model validation, binary writing, file replacement, fixtures, dependencies, and tests. |
| `docs/NUMPY_TESTS.md` | Reference calculations, fixtures, tolerances, regeneration, and verification results. |
| `docs/ROUND_TRIP.md` | Saved and freshly exported model-prefix comparisons through C loading/inference, dependencies, and measured results. |
| `docs/M2_VERIFICATION.md` | Lifecycle/allocation audit, local and hosted evidence, M2 closure, and limitations. |
| `docs/M3_VERIFICATION.md` | Loaded-model lifecycle, allocation/recovery audit, local/hosted evidence, M3 closure, and limitations. |
| `tests/check_version.cmake` | Checks the version command's exit status and output. |
| `tests/test_tensor.c` | Tests the core library's allocation and cleanup API. |
| `tests/test_dense.c` | Tests Dense shapes, validation, copied parameters, and independent ownership/lifetimes. |
| `tests/test_dense_alloc_failures.c` | Forces all five Dense allocation failures and checks cleanup, recovery, and allocation-free forward calls. |
| `tests/test_dense_forward.c` | Tests Dense results, saved NumPy references, validation, partial failures, and repeated execution. |
| `tests/test_layer.c` | Tests common dispatch, metadata, ownership, source/output preservation, errors, and saved NumPy references. |
| `tests/test_layer_alloc_failures.c` | Forces wrapper/owned-data allocation failures and checks cleanup, recovery, and allocation-free execution. |
| `tests/test_model.c` | Tests model construction, connection validation, ownership, dimension getters, and cleanup. |
| `tests/test_model_alloc_failures.c` | Checks allocation rollback, ownership, buffer capacities/reuse, immediate failure stopping, and allocation-free inference. |
| `tests/test_model_run.c` | Tests ordered results, validation, output preservation, arithmetic failures, recovery, and independent models. |
| `tests/test_model_load.c` | Tests saved/generated binary loads, reference outputs, independent owners, truncation, malformed fields, and recovery. |
| `tests/test_loader_failures.c` | Reuses loader cases with allocation/stream tracking, forced allocation/stdio failures, overflow guards, exact parameter bits, and retry checks. |
| `tests/test_tensor_access.c` | Tests row-major access, bounds, finite values, and failure behavior. |
| `tests/test_tensor_matvec.c` | Tests matrix-vector products, dimensions, output capacity, and numeric failures. |
| `tests/test_tensor_bias.c` | Tests bias addition, vector lengths, numeric failures, and composed `Wx + b` results. |
| `tests/test_tensor_relu.c` | Tests ReLU values, validation, failure preservation, and composed `ReLU(Wx + b)` results. |
| `tests/test_tensor_softmax.c` | Tests stable probabilities, extreme/tiny values, validation, and composed math results. |
| `tests/test_tensor_numpy.c` | Compares Cinder's math with saved or freshly generated NumPy fixtures. |
| `tests/test_model_numpy.c` | Compares all tiny-network stages through layers and model prefixes against NumPy. |
| `tests/test_model_round_trip.c` | Loads exported model prefixes and raw inputs, then compares all stages against saved NumPy outputs. |
| `tests/check_model_round_trip.py` | Exports fresh fixtures, invokes the C comparator, and verifies rejection of nine deliberate fixture changes. |
| `tests/test_model_lifecycle.c` | Exercises constructed/loaded models, repeated/concurrent loading, recovery, source-file independence, and caller-serialized model use. |
| `tests/numpy_compare.h` | Shared shape/finite checks, tolerance rule, diagnostics, and statistics for reference tests. |
| `tests/generate_numpy_fixtures.py` | Generates independent float32 NumPy references with fixed inputs and seed. |
| `tests/fixtures/numpy_fixtures.h` | Saved input arrays and NumPy expected results, tracked in Git. |
| `tests/requirements-numpy.txt` | Pinned optional NumPy dependency for reference generation. |
| `tests/test_export_model.py` | Real PyTorch validation, extraction, rejection, resource-boundary, and saved-reference tests. |
| `tests/test_export_binary.py` | Exact encoding, independent decoding, reference comparisons, regeneration, and I/O failure tests. |
| `tests/cnd_fixture_support.py` | Shared saved NumPy array reader and fixed PyTorch network construction. |
| `tests/generate_cnd_fixtures.py` | Reproduces or checks the tiny binary model, inputs, and saved expected stages. |
| `tests/fixtures/tiny/` | The 168-byte full model, three model prefixes, ten eight-byte inputs, and a reference JSON manifest. |
| `tools/export_model.py` | Validates supported PyTorch models and exports complete `.cnd` version 1 files. |
| `tools/requirements-exporter.txt` | Pinned optional PyTorch dependency for exporter validation. |
| `examples/mnist/train.py` | Seeded CPU training, held-out evaluation, and checkpoint/settings records for the external MNIST example. |
| `examples/mnist/data.py` | Checked original IDX archives, explicit optional downloads, and shared float32 row-major preprocessing. |
| `examples/mnist/requirements.txt` | Reuses the exporter's pinned PyTorch dependency. |
| `examples/mnist/README.md` | Architecture, settings, reproduction/reload commands, offline verification, and remaining MNIST steps. |
| `examples/mnist/RESULTS.md` | Measured full-data training/test results, split isolation, repeatability evidence, and reproduction commands. |
| `examples/mnist/reference/` | Preserved trained checkpoint, exact configuration/metrics, training log, and artifact verification record. |
| `examples/mnist/export.py` | Restores the checkpoint, appends inference Softmax, and exports or checks exact `.cnd` bytes through the existing writer. |
| `examples/mnist/mnist.cnd` | Trained 407,160-byte model with four layers and ten output probabilities. |
| `examples/mnist/EXPORT.md` | Export commands, byte layout, provenance, checks, and remaining scope. |
| `examples/mnist/generate_fixtures.py` | Generates/checks the fixed 100-image test fixture set from the original test archives and saved checkpoint. |
| `examples/mnist/FIXTURES.md` | Selection policy, binary layouts, regeneration commands, measured results, and verification scope. |
| `examples/mnist/VALIDATION.md` | C-versus-Python stage errors, prediction agreement, separate classification accuracy, reproduction, and verification scope. |
| `examples/mnist/LIFECYCLE.md` | Larger-model failure recovery, ownership, allocation audit, concurrency, local verification, and CI evidence. |
| `tests/fixtures/mnist/` | Indices, labels, original selected pixels, 100 raw inputs, four stage arrays, and a provenance/hash manifest. |
| `tests/test_mnist_training.py` | Offline synthetic training, preprocessing/IDX checks, independent-process repeatability, and checkpoint/CLI failures. |
| `tests/test_mnist_export.py` | Exact trained parameter bits, evaluation/Softmax behavior, independent-process export, and checkpoint/CLI rejection checks. |
| `tests/test_mnist_fixtures.py` | Offline preprocessing, stage comparison, regeneration, source selection, corruption detection, and cleanup checks. |
| `tests/test_mnist_inference.c` | Python-independent comparison of all 100 saved MNIST inputs and four stages through the C loader/model APIs. |
| `tests/check_mnist_inference.py` | Fresh checkpoint export, prefix identity, 28 C rejection controls, and accuracy/agreement separation. |
| `tests/test_mnist_lifecycle.c` | C-only repeated inference, failed-call/load recovery, independent owners, and supported concurrency. |
| `tests/mnist_lifecycle_audit.h` | Test-only allocation/I/O tracking, partial-write injection, and compilation of the real ownership chain. |
| `tests/test_tensor_alloc_failures.c` | Forces allocation failures and tracks cleanup with test-only allocator substitutions. |
| `.github/workflows/ci.yml` | macOS/Linux build jobs and the Linux sanitizer job. |
| `ROADMAP.md` | Scope, implementation decisions, milestones, and progress. |
| `LICENSE` | MIT license and copyright notice. |
| `.gitignore` | Generated build files and macOS metadata exclusions. |

## Development workflow

Work on one scoped task at a time. Define its completion check, make the change,
verify the relevant behavior, and update the roadmap with the result and next
task. Record unresolved issues before moving on.

The roadmap is the source of truth for progress. Its future command examples
and benchmark numbers describe targets, not existing capabilities or measured
performance. Phases 0 and 1 are complete locally. Phase 2's Dense-layer contract,
creation, cleanup, and forward pass are complete. The common layer interface
and its constructors, validator, cleanup, and dispatch are implemented. The
sequential-model contract is defined, and construction, connection validation,
size accessors, cleanup, reusable buffers, and sequential inference are implemented.
The complete example network matches NumPy at every stage, and lifecycle checks
pass locally and in hosted macOS/Linux CI. M2 closed on October 23, 2026.
The Phase 3 `.cnd` format contract, Python validation, and binary export are
implemented, with reproducible tiny-network fixtures.
The C loader API is implemented with successful-load, parser resource-boundary,
and forced-failure cleanup tests. Every saved stage now passes through the full
export/load/inference path. Loaded-model lifecycle, local sanitizer/leak checks,
and all three hosted jobs pass for `5ad7dc4`, closing M3 on October 25, 2026.
Phase 4's Python reference model is trained and saved, with 97.48% official test
accuracy and verified reload/repeatability results.
The trained `mnist.cnd` includes inference Softmax and regenerates identically.
Saved fixtures now cover 100 fixed test inputs and all four stage outputs.
The `cinder run` command and exact-length input reader are implemented and
verified locally and in all five hosted jobs. The
[demonstration report](examples/mnist/DEMO.md) consolidates reproduction commands,
results, limitations, and M4 closure on October 27, 2026.
The [Phase 5 benchmark contract](docs/BENCHMARK.md) is defined, and reusable CLI
model/input setup, cleanup, and checked macOS/Linux monotonic timing are
implemented with the tested `run` behavior preserved. Benchmark dispatch,
warm-ups, the measured loop, statistics, and reporting remain pending.
Next: implement the ten warm-up calls and measured inference loop.

The v0.1.0 scope excludes training, GPU execution, convolutional layers,
quantization, SIMD, multithreading, and arbitrary computation graphs.

## License

Cinder is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Faruk Umar.
