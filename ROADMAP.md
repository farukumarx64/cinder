# Cinder

**A Tiny Neural-Network Inference Runtime in C**

Cinder is a lightweight neural-network inference runtime written in C.

The goal of `v0.1.0` is simple:

> Take a neural network trained elsewhere, load its weights in pure C, perform forward inference on the CPU, and produce results matching a trusted Python implementation.

Cinder is **not** intended to replace PyTorch, TensorFlow, ONNX Runtime, or other full machine-learning frameworks.

The purpose of the project is to understand what happens underneath those abstractions.

This plan was revised on October 7, 2026 to clarify implementation contracts and introduce verification earlier. The project began in an empty folder; the checklist below records actual progress, while the later milestones remain targets.

The daily dates are planning targets. Preserve the milestone order, the October 25 feature freeze, and the October 31 release target; adjust individual workdays to actual progress. M0 was completed on October 8, 2026; Current Progress tracks the remaining work.

All predictions, accuracy percentages, and benchmark timings shown in this document are illustrative until measured.

## Current Progress

Phase 0 (M0) is complete as of October 8, 2026, including the MIT license and
verified development setup. Phase 1 (M1) was completed locally on October 16,
2026: the tensor contract, allocation/cleanup, access, matrix-vector
multiplication, bias addition, ReLU, and stable Softmax are implemented and
tested. The math operations match deterministic NumPy float32 references
within the documented tolerance; hosted macOS/Linux verification passed at M2
closure on October 23.
Phase 2 has a documented Dense-layer contract and implemented creation, cleanup,
and allocation-free forward execution. Dense owns independent parameter copies
and matches the saved NumPy matrix/bias references. The common layer interface
now implements constructors, metadata validation, cleanup, and allocation-free
Dense/ReLU/Softmax dispatch. Activation layers preserve dimensions and separate
sources, support exact in-place aliasing, reject invalid inputs before writing,
and leave spare output capacity untouched.
The sequential-model contract is defined: fixed nonempty order, ownership
transfer only on successful construction, validated connections, private reusable
buffers, unchanged caller output on failure, and serialized runs per instance.
Construction, connection validation, size accessors, cleanup, and two reusable
working buffers are implemented. Construction adopts layers only after all four
model allocations succeed. Sequential inference through `cinder_model_run` is
implemented with separate Dense buffers, in-place activations, unchanged caller
output on failure, and no execution allocations/frees. Independent NumPy fixtures
now verify every stage of the full example network for ten fixed inputs through
both common layers and model execution. The lifecycle audit passes locally,
including poisoned-buffer independence, allocation counts, construction retries,
and valid threaded caller use. **M2 is complete as of October 23, 2026:** hosted
CI for `f5424a4` passed 19/19 macOS tests, 21/21 Linux tests with fresh NumPy,
and 19/19 Linux sanitizer/leak tests, including the lifecycle additions.
See [the dated verification report](docs/M2_VERIFICATION.md) for evidence and
limitations. Phase 3's [`.cnd` version 1 contract](docs/MODEL_FORMAT.md) was defined
on October 23: fixed header/record layouts, little-endian binary32 parameters,
limits, strict EOF, input preprocessing, and the worked tiny-network bytes.
**Next scoped step: implement exporter model validation before binary writing.**
Implement and verify one scoped step at a time.

### Phase 0 checklist

- [x] Install CMake and verify the existing C compiler.
- [x] Configure the C11 executable with `-Wall`, `-Wextra`, and `-Wpedantic`; build and run the minimal entry point.
- [x] Initialize Git on `main` and add ignore rules for generated build files and macOS metadata.
- [x] Add the basic README describing the current setup, prerequisites, build commands, and development workflow.
- [x] Choose and add the project license. The owner selected the [MIT License](LICENSE); copyright (c) 2026 Faruk Umar.
- [x] Implement `cinder --version`; verify its output, successful exit status, and rejection of missing, unknown, or extra arguments.
- [x] Add the CTest version-command smoke test; verify exact output, exit status, empty standard error, and failure detection.
- [x] Configure opt-in ASan and UBSan builds; verify each separately and together, confirm error detection with temporary probes, and document platform-specific leak checks. Native macOS `leaks` reports zero leaks for `--version`; Apple Clang LeakSanitizer is unavailable on the verified setup.
- [x] Set up the public `farukumarx64/cinder` GitHub repository; `origin` uses SSH and local `main` tracks `origin/main`.
- [x] Add and verify macOS and Linux CI, including Linux LeakSanitizer checks. All three jobs passed in [the first hosted run](https://github.com/farukumarx64/cinder/actions/runs/37680825782): macOS 15, Ubuntu 24.04, and Ubuntu 24.04 with Clang ASan, UBSan, and leak detection enabled.
- [x] Verify the documented setup from a fresh checkout before closing M0. The README commands passed on macOS on October 8, 2026; details below.

### Fresh-checkout verification — October 8, 2026

Cloned the public GitHub repository over HTTPS into a new temporary directory
at commit `1190c0ec06475524563c2ee2afa64f74ad9c02ae`, without copying existing
build directories or CMake caches. Verification used macOS 27.0.1 on Apple
Silicon, Apple Clang 21.0.0, CMake/CTest 4.4.4, GNU Make 3.81, and Git 2.54.0.

- The documented configure and build commands succeeded; `--version` printed `Cinder 0.1.0-dev`.
- The ordinary build passed the CTest smoke test (1/1).
- The Debug build with ASan and UBSan enabled passed the same test (1/1).
- Native macOS `leaks --atExit` exited successfully and reported zero leaks for the version command.
- A separate build with `BUILD_TESTING=OFF` compiled, printed its version, and registered zero tests as expected.
- Generated files were ignored, and the fresh clone's Git working tree remained clean.

This verifies the documented commands with the installed macOS toolchain.
Linux and hosted macOS coverage is recorded in the CI checklist item above.
The MIT license was added on October 8, 2026, completing the final Phase 0
checklist item and closing M0.

### Phase 1 checklist

- [x] Define the tensor contract: `rows`, `cols`, `data`, row-major layout, ownership, and error behavior. See [the contract](docs/TENSOR.md), [tensor declarations](include/cinder/tensor.h), and [error types](include/cinder/error.h).
- [x] Implement tensor creation and cleanup, including dimension/overflow checks, zero initialization, allocation failure cleanup, and CTest coverage with sanitizers applied to the new code and test targets. On October 9, 2026, all 3 tests passed locally on macOS in Debug, Release, and combined ASan/UBSan builds; both tensor executables reported zero native leaks. A build without tests also passed.
- [x] Implement checked tensor fill/get/set and their tests. On October 15, 2026, all 4 tests passed locally on macOS in Debug, Release, and combined ASan/UBSan builds. Access tests cover row-major indexing, bounds, non-finite values, failure preservation, and optional error details; allocator tracking confirms no access allocations. The new test executable reported zero native leaks, and a build without tests passed.
- [x] Implement matrix-vector multiplication; explain the loops and verify a non-square example calculated by hand. The API checks input length, output capacity, finite sources, and arithmetic results without allocating. Its contract documents separate output storage and partial output on arithmetic failure. On October 15, 2026, all 5 tests passed locally in Debug, Release, and combined ASan/UBSan builds; the matvec executable reported zero native leaks, and a build without tests passed. See [the worked example and contract](docs/TENSOR.md#matrix-vector-multiplication).
- [x] Implement bias addition and validate vector lengths. The in-place API checks equal nonzero lengths, byte-count overflow, finite sources, and sums without allocating. Integration tests verify `Wx + b`, a zero input, and untouched spare output capacity. On October 15, 2026, all 6 tests passed locally in Debug, Release, and combined ASan/UBSan builds; the bias executable reported zero native leaks, and a build without tests passed. See [the bias contract and worked example](docs/TENSOR.md#bias-addition).
- [x] Implement ReLU and test negative, zero, and positive values. The in-place API validates pointers, count, byte extent, and finite values before replacing negatives with positive zero. Tests cover signed zeros, extremes, subnormals, full preservation on failure, repeated application, no allocation, and `ReLU(Wx + b)`. On October 15, 2026, all 7 tests passed locally in Debug, Release, and combined ASan/UBSan builds; the ReLU executable reported zero native leaks, and a build without tests passed. See [the ReLU contract and worked example](docs/TENSOR.md#relu).
- [x] Implement stable Softmax and its empty/non-finite input handling. The in-place API subtracts the maximum before exponentiation, uses double intermediates with float32 storage, and allocates no memory. Tests cover known probabilities, offsets, equal/extreme scores, tiny probabilities, long-vector normalization, input rejection without mutation, and composed math. On October 16, 2026, all 8 tests passed locally in Debug, Release, and combined ASan/UBSan builds; the Softmax executable reported zero native leaks, and a build without tests passed. See [the Softmax contract and worked example](docs/TENSOR.md#softmax).
- [x] Validate the math operations against deterministic float32 NumPy fixtures using the documented tolerance rule. The saved and freshly generated references cover 11 matrix/bias fixtures and 12 activation fixtures, comparing 79 vectors / 341 results including intermediate outputs. Shapes and finiteness are checked first; `atol = rtol = 1e-5` is unchanged. On October 16, 2026, the ordinary Debug suite passed 9/9 tests; Release and ASan/UBSan builds with fresh generation passed 10/10. The largest absolute error was `1.1920929e-7`. Repeated generation was identical locally, missing-dependency/configuration guards passed, and the reference executable reported zero native leaks. See [NumPy reference verification](docs/NUMPY_TESTS.md). M1 was completed locally on October 16; hosted verification passed at M2 closure.

### Phase 2 checklist

- [x] Define the Dense-layer contract: positive input/output sizes, row-major `[output_size][input_size]` weights, an `[output_size][1]` bias tensor, owned copies of finite source parameters, create/free/forward declarations, exact input length, minimum output capacity, overlap preconditions, and validation versus arithmetic failure behavior. See [the contract](docs/DENSE.md) and [public declarations](include/cinder/dense.h). The header and worked example passed C11 syntax checks on October 16, 2026, before implementation began.
- [x] Implement Dense-layer creation and cleanup using tensor allocation, with validation before allocation and independently owned parameter copies. Tests cover invalid dimensions/counts, overflow, non-finite sources, source independence and permitted overlap, all five allocation-failure points, partial cleanup, optional errors, and recovery. On October 16, 2026, Debug passed 11/11 tests; Release and combined ASan/UBSan builds with generated NumPy references enabled passed 12/12. Both Dense executables reported zero native leaks, and a build without tests passed. See [implementation and verification](docs/DENSE.md#creation-and-cleanup-implementation).
- [x] Implement the Dense forward pass using the existing matvec and bias kernels. It checks input length, output capacity, and finite sources before writes, propagates kernel failures, and preserves input/parameters and spare capacity. Tests cover non-square shapes, known results, 11 saved NumPy fixtures (41 outputs), repeated inputs, validation preservation, arithmetic partial results/recovery, and zero allocation/free calls. On October 16, 2026, Debug passed 12/12 tests; Release and combined ASan/UBSan builds with generated tensor NumPy references enabled passed 13/13. Forward and extended allocation-failure executables reported zero native leaks; a build without tests passed. See [forward behavior](docs/DENSE.md#forward-execution).
- [x] Define the common layer interface and the Dense/ReLU/Softmax representation and execution contracts. The tagged structure exposes positive input/output sizes and a Dense-only owned payload; activations require equal sizes and no payload. Constructors, metadata validation, cleanup, separate-buffer forward execution, switch routes, and failure/overlap rules are declared in [layer.h](include/cinder/layer.h) and explained in [the contract](docs/LAYER.md). Header and example C11 syntax checks passed on October 16, 2026, before common layer implementation began.
- [x] Implement common layer constructors, metadata validation, cleanup, and switch dispatch for Dense/ReLU/Softmax, reusing the existing kernels. Tests cover tags, payload/dimension consistency, activation input validation before copying, preserved lengths/sources, untouched spare capacity, error propagation/recovery, and 35 saved NumPy vectors. Fault tests cover all six Dense-wrapper allocations and the single allocation for each activation, complete cleanup, and allocation-free validation/forward calls. On October 21, 2026, Debug passed 14/14 tests; Release and combined ASan/UBSan with generated tensor references enabled passed 15/15. Both new executables reported zero native leaks; a build without tests and the linked documentation example passed. See [implementation and verification](docs/LAYER.md#verification-and-next-step).
- [x] Define the sequential-model contract: opaque nonempty ordered models, layer ownership transfer only on construction success with cleared source slots, rejection of duplicate entries and mismatched connections, dimension getters, two private working buffers, final output copy only after success, unchanged output on every returned failure, repeated-call recovery, and externally serialized runs per instance. See [the contract](docs/MODEL.md) and [public declarations](include/cinder/model.h). Header and example C11 syntax checks passed on October 22, 2026, before model implementation began.
- [x] Implement model creation, connection validation, size accessors, cleanup, and two reusable buffers sized for the largest required vector. Validation precedes allocation; ownership transfers only after all four model allocations succeed. Tests cover ordered adoption, cleared source slots, unchanged failure slots/layers, all four forced allocation failures, complete cleanup, independent owners, and exact private buffer capacities (four floats each for the example network). On October 22, 2026, Debug passed 16/16 tests; Release and combined ASan/UBSan with generated tensor references enabled passed 17/17. Both new executables reported zero native leaks; a build without tests and the linked construction example passed. See [model verification](docs/MODEL.md#verification-and-next-scoped-steps).
- [x] Implement sequential inference: copy validated input into private storage, alternate buffers for Dense, run ReLU/Softmax in place through common dispatch, and publish caller output only after all layers succeed. Stop on the first failure and preserve its diagnostic; no allocations/frees occur during success, rejection, arithmetic failure, or recovery. Tests cover known results, differing vector sizes, single/activation-only/activation-first models, consecutive Dense layers, repeated calls, independent models, source/spare-capacity preservation, and failure at each layer position. On October 22, 2026, Debug passed 17/17 tests; Release and combined ASan/UBSan with generated tensor NumPy references enabled passed 18/18. See [model verification](docs/MODEL.md#verification-and-next-scoped-steps).
- [x] Compare every intermediate output and final probabilities of the fixed `2 -> 4 -> 2` network against NumPy for ten deterministic inputs. Saved and generated fixtures cover Dense1, ReLU, final logits, and Softmax through both common layers and sequential model prefixes. Shared shape/finite checks and `atol = rtol = 1e-5` remain unchanged. On October 23, 2026, Debug passed 18/18 tests; Release and combined ASan/UBSan with fresh references passed 20/20. Each network test compared 80 vectors / 240 values, with maximum absolute error `5.96046448e-8`. Regeneration was identical locally; the network executable reported zero native leaks, and a build without tests passed. Parameters, inputs, and stage outputs are retained for Phase 3. See [complete-network verification](docs/NUMPY_TESTS.md#complete-network-verification--october-23-2026).
- [x] Verify lifecycle behavior locally: repeated/different inputs, failed calls followed by success, independent owners, destruction of one model while another survives, poisoned private buffers, zero inference allocation/free attempts, and construction-failure cleanup/retry. The new test executes 4,120 success/failure/recovery cycles, including concurrent independent models and caller-serialized shared-model threads. On October 23, 2026, Debug passed 19/19; Release and combined ASan/UBSan with fresh NumPy references passed 21/21. Lifecycle and model-storage executables reported zero native leaks; the tests-disabled build passed. See [M2 verification](docs/M2_VERIFICATION.md).
- [x] Inspect hosted macOS/Linux and Linux sanitizer/leak results. [Run 39930054347](https://github.com/farukumarx64/cinder/actions/runs/39930054347) passed all three jobs for `f6480d5`, with 17/17 macOS, 18/18 Linux, and 17/17 Linux sanitizer tests. Runtime sources/headers are unchanged since that revision; newer test additions were not included in the run.
- [x] Publish and verify the NumPy and lifecycle additions on all three CI jobs. [Run 39984963408](https://github.com/farukumarx64/cinder/actions/runs/39984963408) passed for `f5424a4a3639cf5c357fccb6ab5ad8f6cf73477a`: 19/19 macOS, 21/21 Linux with fresh NumPy, and 19/19 Linux ASan/UBSan/leak tests. Job statuses, revision, and test logs were inspected. **M2 closed October 23, 2026.**

The counts in earlier checklist entries record their original local checks.
The complete suite, including those earlier features, passed the hosted run
above. Remaining platform/concurrency limits are recorded in the M2 report.

### Phase 3 checklist

- [x] Define the `.cnd` version 1 contract before implementing export/load: a 16-byte file header, 16-byte layer headers, stable IDs, exact Dense payload sizes, dimension connections, little-endian binary32 encoding, finite parameters, resource limits, strict EOF, filename-based display names, and the separate input/preprocessing contract. On October 23, the complete 168-byte example, offsets, 22 parameters, input bytes, and stage results were checked against the existing tiny-network fixtures with Python's standard library. See [the specification](docs/MODEL_FORMAT.md). No exporter or loader is implemented by this step.
- [ ] Implement exporter model validation: accept only supported sequential configurations, extract dimensions and finite float32 parameters, and reject unsupported or inconsistent models.
- [ ] Implement binary export and reproducible tiny-network fixtures according to the specification; verify exact bytes and preserve reference inputs/outputs.
- [ ] Define the C loader API, error statuses, file handling, and ownership/cleanup contract.
- [ ] Implement loading and model construction with successful-load and malformed-file tests alongside the parser.
- [ ] Complete malformed-file, size-limit, non-finite-data, and forced allocation-failure coverage; verify partial cleanup and recovery.
- [ ] Verify Python export -> `.cnd` -> C load -> inference against every saved tiny-network stage, using matching shapes, finite values, and `atol = rtol = 1e-5`.
- [ ] Exercise loaded-model lifecycle behavior, sanitizers, supported leak checks, and macOS/Linux CI; record actual results and close M3.

---

# 1. v0.1.0 Goal

By the end of October, Cinder should support a workflow like:

```bash
$ cinder info mnist.cnd
```

Output:

```text
Model: mnist.cnd
Input: 784

Layers:
  Dense 784 -> 128
  ReLU
  Dense 128 -> 10
  Softmax

Parameters: 101,770
```

Inference:

```bash
$ cinder run mnist.cnd digit.bin
```

Output:

```text
Prediction:

7    98.42%
1     0.71%
9     0.39%

Inference time: 0.84 ms
```

Benchmarking:

```bash
$ cinder bench mnist.cnd digit.bin --iterations 1000
```

Output:

```text
Iterations:       1000
Average:          0.79 ms
Min:              0.73 ms
Max:              1.12 ms
```

That is enough for a strong `v0.1.0`.

---

# 2. MVP Scope

## Required for v0.1.0

- [ ] Pure C runtime
- [ ] CPU inference
- [ ] `float32` tensors
- [x] Dense / fully connected layers (single-vector create/free/forward API)
- [x] ReLU activation
- [x] Softmax activation
- [x] Sequential neural networks (in-memory single-vector API)
- [ ] Model loading from disk
- [x] Custom Cinder model-format contract (version 1; exporter/loader implementation remains)
- [ ] Python model exporter
- [ ] CLI inference
- [ ] CLI model inspection
- [ ] Benchmark command
- [ ] Unit tests
- [ ] Python/reference comparison tests
- [ ] Memory-error testing from the first allocation code onward
- [ ] Automated macOS and Linux build/test checks
- [ ] Tiny exported-network reference test before MNIST
- [ ] MNIST demonstration
- [ ] macOS compilation
- [ ] Linux compilation
- [ ] GitHub release

---

# 3. Explicitly Out of Scope

Do **not** include the following in `v0.1.0`:

- training
- backpropagation
- autograd
- CUDA
- GPU inference
- convolutional layers
- transformers
- attention
- ONNX
- Python bindings
- HTTP serving
- networking
- INT8 quantization
- SIMD
- NEON
- AVX
- multithreading
- arbitrary computation graphs
- dynamic tensor shapes

These can become future versions.

The goal of `v0.1.0` is not feature completeness.

The goal is a **small, understandable, reliable inference runtime**.

---

## Implementation Decisions for v0.1.0

These decisions refine the existing scope without adding operators or a general framework:

- **Execution:** one sample per inference, fixed dimensions, and a sequential list of Dense, ReLU, and Softmax layers. No batching in v0.1.0.
- **Representation:** contiguous float32 vectors and matrices. Dense weights have shape `[output_size][input_size]` in row-major order. Keep the tensor abstraction small; strides, views, broadcasting, and general tensor machinery are unnecessary for this release.
- **API:** callers provide input length and output capacity. The API exposes model input/output sizes and returns status codes with useful error details. Ownership is explicit.
- **Formats:** document fixed-width fields, little-endian byte order, IEEE 754 binary32 values, weight layout, and input preprocessing before writing the serializer and loader. Never serialize native C structs directly.
- **Exporter:** accept a constructed Python `torch.nn.Sequential` containing only explicitly supported layer configurations. Reject unsupported configurations clearly. Importing arbitrary `.pth` files is outside v0.1.0.
- **Verification:** run unit tests, sanitizers, and macOS/Linux CI early. Compare intermediate outputs and final outputs against deterministic float32 reference fixtures with explicit absolute and relative tolerances.
- **First end-to-end checkpoint:** export the tiny `2 → 4 → 2` network, load it in C, and match its reference outputs before beginning MNIST integration.

The detailed binary layout is defined in [docs/MODEL_FORMAT.md](docs/MODEL_FORMAT.md).
The core math implementation follows the decisions above.

---

# 4. Repository Structure

```text
cinder/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
├── ROADMAP.md
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── include/
│   └── cinder/
│       ├── tensor.h
│       ├── ops.h
│       ├── model.h
│       ├── runtime.h
│       └── error.h
│
├── src/
│   ├── tensor.c
│   ├── ops.c
│   ├── model.c
│   ├── runtime.c
│   ├── error.c
│   └── cli.c
│
├── tests/
│   ├── check_version.cmake
│   ├── test_tensor.c
│   ├── test_ops.c
│   ├── test_model.c
│   ├── test_runtime.c
│   └── fixtures/  # tiny model, inputs, and per-layer reference outputs
│
├── tools/
│   └── export_model.py
│
├── examples/
│   └── mnist/
│       ├── train.py
│       ├── model.cnd
│       ├── sample.bin
│       └── README.md
│
└── docs/
    ├── MODEL_FORMAT.md
    └── ARCHITECTURE.md
```

Use **CMake** as the build system.

---

# 5. Public API

Keep the initial API small while making buffer sizes and failure details explicit.

The sequential-model interface is defined in [model.h](include/cinder/model.h)
and [the model contract](docs/MODEL.md). Construction, size accessors, sequential
inference, and cleanup are implemented:

```c
#include "cinder/layer.h"

typedef struct cinder_model cinder_model_t;

cinder_status_t cinder_model_create(
    cinder_layer_t **layers,
    size_t layer_count,
    cinder_model_t **out_model,
    cinder_error_t *error
);

size_t cinder_model_input_size(const cinder_model_t *model);
size_t cinder_model_output_size(const cinder_model_t *model);

cinder_status_t cinder_model_run(
    cinder_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);

void cinder_model_free(cinder_model_t *model);
```

Phase 3 will separately specify and implement file loading, using this proposed
entry point; it is not declared by the current model header:

```c
cinder_status_t cinder_model_load(
    const char *path,
    cinder_model_t **out_model,
    cinder_error_t *error
);
```

API requirements:

- `input_count` and `output_capacity` count float elements, not bytes.
- Require the exact model input count and enough output capacity before computation begins. As in any pointer-based C API, the caller must report the actual available buffer sizes.
- Return `CINDER_OK` on success and a specific status on failure. Define status codes and caller-owned error details in `error.h`; avoid relying on printed messages or global error state.
- Model construction sets `*out_model` to NULL before validation. Failure leaves all source slots/layers unchanged and caller-owned. Success adopts every layer, clears the source slots, and publishes the model. Empty lists, NULL/duplicate entries, invalid layer metadata, and incompatible adjacent dimensions are rejected before allocation or inference.
- Future model loading must set `*out_model` to NULL on failure and free any partially loaded state it owns.
- The model owns its layers, weights, biases, and reusable working buffers. The caller owns the input/output arrays; the runtime does not retain those pointers after a run.
- Input and output buffers must be separate from each other, error storage, and model-owned storage. Every layer writes into one of two private buffers; only after all layers succeed does inference copy the logical result to caller output. Every returned failure preserves the entire caller output, and spare capacity is always untouched.
- Inference on one instance must be sequential and non-reentrant because working buffers are shared. The caller must serialize whole calls with an external lock or use independently owned models. There is no internal lock or concurrency error status; destruction must not race any call.
- Input/output size getters expose immutable positive dimensions and return zero for NULL. Getters may run during inference but never race destruction. Repeated runs, including after failure, require no reset or allocation.
- `cinder_model_free(NULL)` is safe.

The runtime supports only sequential models in v0.1.0.

Internally:

```text
Input
  ↓
Dense
  ↓
ReLU
  ↓
Dense
  ↓
Softmax
  ↓
Output
```

No computational graph engine.

No scheduler.

No DAG.

No unnecessary abstractions.

---

# 6. Cinder Model Format

The [version 1 specification](docs/MODEL_FORMAT.md) defines the `.cnd` format.
It was completed on October 23, 2026, before exporter or loader implementation.
The specification is authoritative; this section summarizes its decisions.

```text
16-byte file header: CNDR | version | input_size | layer_count
    followed by exactly layer_count records:
16-byte layer header: type_id | input_size | output_size | payload_bytes
    followed immediately by exactly payload_bytes parameter bytes
EOF immediately after the final record
```

All numeric header fields are unsigned 32-bit little-endian values. `CNDR`
is four literal bytes; version is exactly `1`. The final record supplies the
model output size. Files contain no padding, end marker, names, or metadata.

- File IDs are Dense `1`, ReLU `2`, and Softmax `3`, explicitly mapped to runtime
  types. Unknown IDs and unsupported versions are rejected.
- Dense stores little-endian IEEE 754 binary32 row-major
  `[output_size][input_size]` weights, followed by `[output_size]` biases.
  Its exact payload length is `4 * (input_size * output_size + output_size)`.
- ReLU and Softmax have zero payload bytes and preserve vector length.
  Every connection and the header's input dimension must agree.
- All parameters must be finite. Signed zeros and subnormals are allowed.
- Inclusive limits are 1,024 layers, 65,536 elements per vector, and 64 MiB of
  total parameter bytes. Validate lengths, available bytes, checked arithmetic,
  and native allocation sizes before use; release partial state on failure.
- The file must end exactly after its last record. Any trailing byte is rejected.
- The future CLI displays the supplied path's basename, including its extension,
  such as `Model: mnist.cnd`. Descriptions and class labels live in example docs.

No native structs, `size_t`, pointers, or enum objects are written to disk. A
finite weight changed by a bit flip can still be structurally valid; version 1
has no checksum or authenticity mechanism.

The specification includes the complete 168-byte encoding of our fixed
`2 -> 4 -> 2` network, exact offsets, sample input bytes, and expected stages.
The exporter and loader must later reproduce and validate this fixture.

### Input File Contract

For `cinder run` and `cinder bench`, the separate input file contains exactly the
model's input count of little-endian IEEE 754 binary32 values, with no header.
Reject missing/extra bytes and non-finite inputs before inference.

The MNIST example uses 784 values: flatten a 28 x 28 image in row-major order,
convert pixels to float32, and divide by float32 `255.0` to obtain `[0, 1]`.
Training, reference inference, and sample generation use the same preprocessing.
The C runtime receives already-preprocessed values. The general input format
allows any finite values; MNIST supplies the normalization convention.

The [input-file specification](docs/MODEL_FORMAT.md#input-file-and-preprocessing-contract)
defines the exact layout. The future MNIST README must repeat its preprocessing
requirements, and example/export tooling must generate sample files.

---

# 7. October Development Roadmap

## Phase 0 — October 4–5

### Project Bootstrap

Create:

```text
repo
CMake
src/
include/
tests/
README
ROADMAP
LICENSE
```

Get this working:

```bash
cmake -S . -B build
cmake --build build
```

Then:

```bash
./build/cinder --version
```

Output:

```text
Cinder 0.1.0-dev
```

Enable warnings:

```text
-Wall
-Wextra
-Wpedantic
```

Also establish verification during bootstrap:

- Register a minimal smoke test with CTest and run it through `ctest --test-dir build --output-on-failure`.
- Add macOS and Linux CI jobs that configure, build, and run tests.
- Add a CMake option for AddressSanitizer and UndefinedBehaviorSanitizer in development builds, with a sanitizer CI job on a supported runner.
- Enable leak detection where supported and document which platform/check provides it.

Use sanitizer builds as soon as tensor allocations exist. Cross-platform builds and memory checks are part of development, not first attempted in the release week.

Refresh:

- pointers
- structs
- heap allocation
- `malloc`
- `calloc`
- `free`
- header files
- static memory
- dynamic memory
- row-major arrays

### Milestone

```text
M0 — Cinder builds, prints its version, and passes a smoke test with CI and sanitizer configuration in place
```

---

# 8. Phase 1 — October 6–9

## Tensor and Mathematical Operations

The minimal representation and API contract are now defined in
[`docs/TENSOR.md`](docs/TENSOR.md) and the public headers. Creation, cleanup,
checked fill/get/set, matrix-vector multiplication, bias addition, ReLU, and
Softmax are implemented and tested against deterministic NumPy references.
M1 is complete locally; see Current Progress and the reference verification guide.

For `v0.1.0`, two-dimensional tensors are enough.

Contract representation:

```c
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} cinder_tensor_t;
```

Implement:

```text
cinder_tensor_create
cinder_tensor_free
cinder_tensor_fill
cinder_tensor_get
cinder_tensor_set
cinder_tensor_matvec
cinder_tensor_add_bias
cinder_tensor_relu
cinder_tensor_softmax
```

Then implement the operations needed for inference:

```text
matrix × vector
bias addition
ReLU
Softmax
```

The central Dense operation is:

```text
y = Wx + b
```

---

## October 6

Implement tensor allocation and memory management.

Define ownership, reject invalid dimensions, check allocation-size overflow, and handle allocation failures. Test these behaviors under the sanitizer build as they are introduced.

Understand exactly:

```text
struct
   ↓
dimensions
   ↓
data pointer
   ↓
contiguous float array
```

---

## October 7

Implement matrix-vector multiplication.

Write or deeply understand the first implementation yourself.

Conceptually:

```text
weight matrix

[w11 w12 w13]
[w21 w22 w23]

×

input

[x1]
[x2]
[x3]

=

output

[y1]
[y2]
```

Do not treat this as magic generated by Codex.

You should understand exactly what every loop is calculating.

---

## October 8

Implement:

```text
ReLU
Softmax
```

ReLU:

```text
f(x) = max(0, x)
```

For Softmax, use a numerically stable implementation. Reject empty vectors. Define non-finite values as unsupported in v0.1.0 and report an error if computation produces a non-finite result.

Implemented in `cinder_tensor_softmax`: subtract and exponentiate in double,
store exponentials in the float32 array, accumulate those stored terms in
double, and divide in double before storing float32 probabilities. This avoids
overflow when subtracting opposite finite float extremes and improves the
normalization sum's precision without allocating scratch storage. Finite
underflow to zero is accepted; pointer, size, and input-value validation
failures preserve the entire array. See the contract for numeric-failure
behavior and caller storage requirements.

Instead of simply:

```c
exp(x)
```

subtract the largest logit before exponentiation.

Conceptually:

```text
softmax(x_i) =
exp(x_i - max(x))
-----------------
Σ exp(x_j - max(x))
```

---

## October 9

Test all operations against Python/NumPy.

Example:

```text
NumPy result
      ↓
compare
      ↓
Cinder result
```

Use a documented comparison rule:

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Start with `atol = 1e-5` and `rtol = 1e-5` for the small deterministic float32 fixtures. Confirm these thresholds against the supported operations and model sizes; record any justified changes rather than silently weakening a failing test.

Verify equal shapes and finite values before numerical comparison. Exercise negative values, zeros, non-square matrices, and large finite logits. Keep reference arrays and parameters explicitly float32.

For networks, compare each layer's output, including logits before Softmax, as well as final probabilities. A matching top prediction alone is insufficient.

NumPy documents this tolerance rule in [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

Implemented and verified locally on October 16, 2026. The independent NumPy
generator uses float32 operations, explicit examples, and PCG64 seed `20261016`.
Saved references run without Python; `CINDER_ENABLE_NUMPY_TESTS=ON` additionally
generates references inside the build directory. See
[the reference guide](docs/NUMPY_TESTS.md) for the fixtures, intermediate-output
checks, measured errors, regeneration commands, and remaining model-level work.

### Milestone

```text
M1 — Cinder can perform the mathematics required for neural inference
```

---

# 9. Phase 2 — October 10–14

## Layers and Runtime

The Dense-specific contract is defined in [docs/DENSE.md](docs/DENSE.md), with
the representation and create/free/forward declarations in
[include/cinder/dense.h](include/cinder/dense.h). Creation, cleanup, and forward
execution are implemented in [src/dense.c](src/dense.c). The common layer API
is specified in [docs/LAYER.md](docs/LAYER.md) and
[include/cinder/layer.h](include/cinder/layer.h), and implemented in
[src/layer.c](src/layer.c). The sequential-model contract is defined in
[docs/MODEL.md](docs/MODEL.md), with declarations in
[include/cinder/model.h](include/cinder/model.h). Model construction, connection
validation, size accessors, cleanup, reusable buffers, and sequential inference
are implemented in [src/model.c](src/model.c). Full-network NumPy verification
and lifecycle checks pass locally and in hosted macOS/Linux CI. M2 closed on
October 23, 2026. Follow the Phase 2 checklist in Current Progress for observed
results; the dated sections below retain the original planned milestone order.

The declared common layer tags are:

```c
typedef enum {
    CINDER_LAYER_INVALID = 0,
    CINDER_LAYER_DENSE,
    CINDER_LAYER_RELU,
    CINDER_LAYER_SOFTMAX
} cinder_layer_type_t;
```

Every layer exposes `input_size` and `output_size`. Dense may change length;
ReLU and Softmax must preserve it. After validating each layer's metadata,
model construction requires each output size to match the next input size.

Conceptually:

```text
model
│
├── layer 0
├── layer 1
├── layer 2
└── layer 3
```

Model execution calls the common API for each layer, handling status
and reusable buffers as specified by the model contract:

```c
cinder_layer_forward(layer, input, input_count, output, output_capacity, &error);
```

---

## October 10–11

Implement the Dense layer.

Each Dense layer needs:

```text
input size
output size
weights
biases
```

---

## October 12

Implement activation layers:

```text
ReLU
Softmax
```

---

## October 13

Implement sequential model execution.

Initially hard-code a tiny network:

```text
2 inputs
   ↓
Dense 2 → 4
   ↓
ReLU
   ↓
Dense 4 → 2
   ↓
Softmax
```

Feed deterministic inputs and known weights through it and verify every layer's output against Python. Keep these fixtures for the exporter/loader round trip in Phase 3. Per-layer inspection can stay inside tests; it does not need a new public tracing API.

---

## October 14

Focus on memory ownership.

For every pointer in the runtime, you should be able to answer:

> Who allocated this?

and:

> Who is responsible for freeing it?

Avoid unclear ownership.

As specified in the model contract, allocate two reusable working buffers during model construction, each sized to the largest input or output vector among all layers. Copy input into private storage, alternate buffers for Dense, and run activations in place. Copy to caller output only after all layers succeed. Keep ownership explicit and avoid allocation inside inference, including error paths. A general arena or memory planner can wait.

### Milestone

```text
M2 — A complete hard-coded neural network executes entirely in C
```

Completed October 23, 2026; see [M2 verification](docs/M2_VERIFICATION.md).

---

# 10. Phase 3 — October 15–18

## Model Serialization

The dates in this section are the original planning targets. The format
contract was completed on October 23, 2026; the Phase 3 checklist in Current
Progress records the actual status. Exporter model validation is next.

Replace the hard-coded application path with model loading, while retaining the tiny network as a test fixture.

Models should now come from:

```text
model.cnd
```

The process becomes:

```text
Python model
     ↓
Cinder exporter
     ↓
model.cnd
     ↓
Cinder loader
     ↓
runtime
```

---

## October 15

Write:

```text
docs/MODEL_FORMAT.md
```

Completed October 23, 2026: [the version 1 contract](docs/MODEL_FORMAT.md).
This specification must precede the exporter and parser.

Document the binary format clearly.

For example:

```text
Offset     Type        Meaning

0          char[4]     "CNDR"
4          uint32      version
8          uint32      input size
12         uint32      layer count
...
```

The specification resolves Section 6's format and input requirements, including
resource limits and a strict trailing-data policy. It is the authoritative
layout; the header table above is a short excerpt.

Future format changes must follow the specification's versioning policy and
update the specification, exporter, loader, and fixtures together.

---

## October 16

Build the Python serializer in `tools/export_model.py` with a small callable interface.

Example usage from Python after constructing or training a supported model:

```python
from tools.export_model import export_model

export_model(model, "model.cnd")
```

For v0.1.0, accept a flat `torch.nn.Sequential` with `Linear` layers with bias, `ReLU`, and `Softmax(dim=-1)` over the feature vector. Export on the CPU in float32, preserving the specified weight layout. Reject unsupported modules, dimensions, dtypes, or configurations with clear errors rather than approximating their behavior.

Checkpoint loading belongs in the example's Python code, which knows the architecture. A `.pth` file containing a `state_dict` is not a complete description of execution order or parameter-free activations; see [PyTorch's save/load example](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

Export the existing tiny `2 → 4 → 2` network first. MNIST integration comes after this path is verified.

---

## October 17

Implement the C model loader and its malformed-file tests together. Validate every field before using it for allocation, indexing, or payload reads; ensure partial-load failures release all allocated memory.

Conceptually:

```text
open file
   ↓
read CNDR magic
   ↓
validate version
   ↓
read model metadata
   ↓
allocate layers
   ↓
read parameters
   ↓
model ready
```

---

## October 18

Complete validation coverage and the first end-to-end checkpoint.

Test:

```text
invalid magic
unsupported version
truncated model
invalid dimensions
unknown layer types
missing data
inconsistent adjacent layer dimensions
size arithmetic overflow and resource limits
non-finite parameters
trailing data according to the format policy
cleanup after partial-load failures
```

Errors should be understandable.

Example:

```text
Cinder error: invalid model header
```

rather than:

```text
Segmentation fault
```

### Milestone

```text
M3 — The tiny network exports from Python, loads in C, and matches reference outputs layer by layer
```

Required end-to-end fixture:

```text
Known 2 → 4 → 2 sequential model + deterministic inputs
     ↓
Python reference outputs for every layer
     ↓
export_model → tiny.cnd → Cinder loader → runtime
     ↓
Matching shapes, finite values, and numerical agreement
```

Check in the tiny model, inputs, expected outputs, and a reproducible way to regenerate them. The C-only tests should run from fixtures without requiring Python or downloading MNIST.

Do not begin MNIST integration until this checkpoint passes. At this point Cinder has become a genuine inference runtime.

---

# 11. Phase 4 — October 19–22

## MNIST Demonstration

Create a simple MNIST classifier externally.

Architecture:

```text
784
 ↓
Dense 784 → 128
 ↓
ReLU
 ↓
Dense 128 → 10
 ↓
Softmax
```

Train using Python.

PyTorch is fine. Use the preprocessing contract from Section 6 and record the training seed and example dependency versions for reproducibility.

If training with `CrossEntropyLoss`, pass logits to the loss; append Softmax to the inference/export model afterward. Use evaluation mode for reference inference.

Training is not part of Cinder.

---

## October 19

Train the reference MNIST model.

---

## October 20

Export the trained model:

```text
PyTorch
   ↓
export_model.py
   ↓
mnist.cnd
```

---

## October 21

Run MNIST inference with Cinder.

Example:

```bash
cinder run mnist.cnd sample.bin
```

Output:

```text
Prediction:

7    98.42%
1     0.71%
9     0.39%
```

---

## October 22

Create reference/golden tests.

Take the same inputs and run:

```text
PyTorch
```

and:

```text
Cinder
```

Compare outputs.

Example:

```text
PyTorch:

7 = 0.984123

Cinder:

7 = 0.984121
```

Test against many images.

Ideally:

```text
100–1000 samples
```

and report maximum absolute error as well as pass/fail results under the documented absolute-plus-relative tolerance policy. Compare logits and final probabilities; include intermediate-layer comparisons in diagnostic tests.

Track classification accuracy on a named dataset split separately from runtime agreement. A runtime can match a poorly trained model exactly, so these measure different things.

### Milestone

```text
M4 — Cinder successfully performs real neural-network inference
```

At this point the MVP technically exists.

---

# 12. Phase 5 — October 23–25

## Benchmarking and Runtime Improvements

Add:

```bash
cinder bench
```

Example:

```bash
cinder bench mnist.cnd sample.bin --iterations 1000
```

Measure:

```text
iterations
average latency
minimum latency
maximum latency
throughput
```

Define the benchmark before interpreting results:

- Use an optimized Release build and record the CPU, operating system, compiler, and build options.
- Load the model and input and allocate working buffers before timing.
- Run warm-up iterations, then measure forward inference with a monotonic clock. Exclude file I/O, preprocessing, and output printing from the reported inference time.
- Validate the iteration count and ensure benchmark outputs are consumed so computation cannot be discarded by optimization.
- Report the iteration count, mean, minimum, maximum, and throughput. Treat noisy individual measurements cautiously.
- Confirm numerical correctness after any performance change.

Do not obsess over beating established runtimes.

Benchmark Cinder against **previous versions of Cinder**.

Example:

```text
Cinder naive runtime        1.41 ms
Cinder optimized buffers    0.94 ms
```

The important part is understanding why performance changed.

---

## Memory Reuse

Initially you may accidentally perform allocations during every layer:

```text
Dense
 ↓
malloc

ReLU
 ↓
malloc

Dense
 ↓
malloc
```

Verify that the reusable working buffers introduced with the runtime eliminate allocations during inference. Use an earlier allocation-heavy implementation for comparison only if one exists; do not introduce one just to create a benchmark.

For sequential networks, you may only need:

```text
Buffer A
Buffer B
```

Then:

```text
input
 ↓
Buffer A
 ↓
Buffer B
 ↓
Buffer A
 ↓
Buffer B
```

This avoids unnecessary allocations during inference.

---

# October 25 — FEATURE FREEZE

On October 25:

> No more features for `v0.1.0`.

If you suddenly think:

```text
What if Cinder supports ONNX?
```

Create:

```text
Issue: feat: ONNX importer
Milestone: v0.2.0+
```

Do not implement it.

The final week exists for making the existing runtime reliable.

---

# 13. Phase 6 — October 26–28

## Testing and Robustness

Extend the tests and sanitizer checks already running since Phase 0; this is the final hardening pass, not their first use.

Run:

```text
AddressSanitizer
UndefinedBehaviorSanitizer
Leak detection on a supported platform
```

Test:

```text
memory leaks
double frees
use-after-free
invalid models
zero dimensions
incorrect input sizes
large dimensions
corrupted files
allocation failures
unsupported versions
```

Cinder should fail gracefully. Release requires no known sanitizer findings or leaks in the exercised paths; a passing sanitizer run does not prove the absence of all memory bugs.

Bad:

```text
Segmentation fault
```

Good:

```text
Cinder error:
model expects 784 input values but received 256
```

---

# 14. Phase 7 — October 29–31

## Documentation and Release

The README should explain:

### What is Cinder?

> Cinder is a tiny neural-network inference runtime written in C, created to explore how neural networks execute beneath high-level machine-learning frameworks.

---

## Architecture

```text
                   ┌──────────────┐
model.cnd ───────→ │ Model Loader │
                   └──────┬───────┘
                          ↓
                   ┌──────────────┐
input ───────────→ │   Runtime    │
                   └──────┬───────┘
                          ↓
                   Dense → ReLU
                          ↓
                   Dense → Softmax
                          ↓
                       Output
```

Document:

```text
Installation
Building
CLI
Architecture
Model format
Examples
Benchmarks
Limitations
Roadmap
```

---

# 15. Release v0.1.0

When everything is ready:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Create the GitHub release:

```text
Cinder v0.1.0
```

The release should include:

```text
Basic FP32 neural-network inference
Dense layers
ReLU
Softmax
Cinder .cnd model format
Python exporter
MNIST example
CLI inference
Model inspection
Benchmarking
```

---

# 16. Definition of Done

Cinder `v0.1.0` is finished when:

- [x] Pure C in-memory inference runtime works
- [x] Builds on macOS
- [x] Builds on Linux
- [x] Tensor representation works
- [x] Matrix-vector multiplication works
- [x] Dense, ReLU, and Softmax work through common dispatch and sequential models
- [x] ReLU works
- [x] Softmax works
- [ ] `.cnd` models load from disk
- [ ] Python exporter works
- [ ] MNIST model executes successfully
- [ ] Tiny exported-network fixture matches each reference layer before MNIST integration
- [ ] Results satisfy documented absolute and relative tolerances with matching shapes and finite values
- [ ] Input layout and preprocessing are documented and identical in Python and C inputs
- [ ] Buffer lengths, output capacity, ownership, and error behavior are documented and checked
- [ ] `cinder run` works
- [ ] `cinder info` works
- [ ] `cinder bench` works
- [x] Unit tests pass for the implemented M2 scope
- [x] macOS and Linux CI build and test successfully for M2
- [x] Sanitizer and supported leak checks report no findings in the exercised M2 suite
- [ ] Loader rejects malformed structures, excessive sizes, and incomplete data without leaking memory
- [ ] Benchmarks document their environment and measure inference separately from loading and preprocessing
- [ ] Architecture is documented
- [x] Model format version 1 is documented
- [ ] README is complete
- [ ] GitHub `v0.1.0` release exists

M2 checks reflect the current implementation and must remain green as the
loader, exporter, CLI, and other release features are added.

Once these boxes are checked:

**Ship it.**

Do not delay the release because of features that belong in future versions.

---

# 17. Post-MVP Roadmap

A possible progression:

```text
v0.1
Basic MLP inference
        ↓
v0.2
Additional operators
        ↓
v0.3
INT8 quantization
        ↓
v0.4
SIMD / NEON / AVX
        ↓
v0.5
CNN inference
        ↓
v0.6
ONNX subset importer
        ↓
v0.7
Multithreading
        ↓
v1.0
Stable runtime and public API
```

There is no requirement to implement all of these.

Cinder should evolve according to what is interesting and educational.

---

# 18. Strong Post-MVP Directions

## Quantization

Add:

```text
FP32
 ↓
INT8
```

Then compare:

```text
model size
memory consumption
latency
accuracy
```

This turns Cinder into a useful systems experiment.

---

## SIMD

Investigate accelerating the core matrix operations.

Possible progression:

```text
naive scalar C
       ↓
loop optimizations
       ↓
ARM NEON
```

Since development is being done on Apple Silicon, ARM NEON would be a particularly interesting direction.

Benchmark every implementation.

---

## Memory Planning

v0.1.0 already sizes and reuses two working buffers for sequential execution. A post-MVP extension could support richer execution patterns by calculating workspace lifetimes and sharing storage across more intermediates.

Consider a reusable arena/workspace only when new model structures justify it.

Conceptually:

```text
Model load
   ↓
determine largest intermediate tensor
   ↓
allocate workspace
   ↓
reuse during inference
```

This moves Cinder closer to how serious inference runtimes approach memory management.

---

# 19. Using AI/Codex While Building Cinder

AI assistance is encouraged, but the core runtime should remain understandable.

You should personally understand:

```text
tensor representation
memory layout
matrix multiplication
Dense forward pass
ReLU
Softmax
binary serialization
binary deserialization
memory ownership
runtime execution loop
```

You do not need to manually type every character.

But you should be able to explain the code.

Use Codex heavily for:

```text
CMake
CI
tests
sanitizer setup
CLI parsing
error-handling boilerplate
documentation
refactoring
code review
cross-platform fixes
```

Use Codex collaboratively for the core runtime.

Avoid prompts equivalent to:

```text
Build an entire neural-network inference engine for me.
```

Instead, work feature by feature.

---

# 20. What Cinder Should Teach

The real objective of the project is not simply another GitHub repository.

By the end of October, you should understand this pipeline:

```text
trained neural network
        ↓
model serialization
        ↓
.cnd file
        ↓
binary bytes
        ↓
C structs
        ↓
weights loaded into memory
        ↓
input tensor
        ↓
matrix multiplication
        ↓
bias
        ↓
activation
        ↓
next layer
        ↓
logits
        ↓
softmax
        ↓
prediction
```

You should be able to explain what happens at every stage.

---

# 21. October Schedule Summary

```text
Oct 4–5
Project setup
        ↓
Oct 6–9
Tensor + math
        ↓
Oct 10–14
Layers + runtime
        ↓
Oct 15–18
Cinder model format + loader
        ↓
Oct 19–22
MNIST inference
        ↓
Oct 23–25
Benchmarking + optimization
        ↓
OCTOBER 25
FEATURE FREEZE
        ↓
Oct 26–28
Testing + sanitizers
        ↓
Oct 29–31
Documentation + release
        ↓
CINDER v0.1.0
```

Follow these milestones in order even if individual daily targets shift. M0, M1, and M2 are complete. The full tiny network matches NumPy, and the lifecycle audit passes locally and in hosted macOS/Linux CI. Revision `f5424a4`, including the NumPy/lifecycle tests, passed all three CI jobs, closing M2 on October 23, 2026. See the M2 verification report for actual results and limitations. Phase 3's `.cnd` model-format contract is defined; exporter model validation is the next scoped task.

The **October 25 deadline is the internal deadline**.

October 31 is the actual release deadline.

That gives Cinder almost a full week of buffer for bugs, life interruptions, documentation, and unexpected problems.

---

# Final Objective

By the end of the project, you should be able to say:

> **I built a neural-network inference runtime from scratch in C. It defines its own binary model format, loads pretrained neural networks, manages their parameters in memory, executes forward inference using custom tensor operations, benchmarks runtime performance, and produces outputs validated against a reference implementation.**

That is Cinder `v0.1.0`.
