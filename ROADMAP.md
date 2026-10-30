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
[Exporter model validation](docs/EXPORTER.md) was completed locally on October 24:
supported PyTorch sequences are checked and their parameters copied into an
independent snapshot. All 33 Python tests and the combined 22-test CTest suite
passed; hosted verification subsequently passed at M3 closure.
Binary export was also completed locally on October 24: the 168-byte tiny
model matches the format's byte example, and all twelve model/input/reference
files regenerate identically. The 50 Python tests and combined 23-test CTest
suite passed. The [C loader contract](docs/LOADER.md) is now defined: public
declaration, status mapping, seekable binary file handling, and cleanup using
existing layer/model constructors. Loading is now implemented with successful-load
and basic malformed-file tests; the saved tiny network produces the expected
NumPy probabilities. The rejection/cleanup audit now covers exact limits,
non-finite data, all tiny-model allocation points, and file-operation failures.
On October 25, the complete Python export -> `.cnd` -> C load -> inference path
also passed every saved stage, with a Python-independent committed-fixture test.
Loaded-model lifecycle, Debug/Release, sanitizer, and native leak checks also
pass locally. **M3 is complete as of October 25, 2026:** revision `5ad7dc4`
passed 23/23 macOS tests, 28/28 Linux tests, and 23/23 Linux sanitizer/leak tests
in [run 40192296107](https://github.com/farukumarx64/cinder/actions/runs/40192296107).
See [the M3 report](docs/M3_VERIFICATION.md) for evidence and limitations.
Phase 4's [reproducible external training script](examples/mnist/README.md) is
implemented: CPU float32 `784 -> 128 -> 10`, shared row-major `/255`
preprocessing, seeded training, logits/CrossEntropyLoss, and saved parameters
with settings/environment/data records. Training-script tests use synthetic data.
The fixed five-epoch reference run now achieves **97.48% official test accuracy
(9,748/10,000)** after training on the separate 60,000-image training split.
Parameters, configuration, metrics, and fresh-process reload/repeat evidence
are preserved in [the MNIST results](examples/mnist/RESULTS.md).
The saved checkpoint now exports with inference Softmax to a 407,160-byte
`mnist.cnd`, with every parameter bit preserved and identical fresh-process
exports. See [the export guide](examples/mnist/EXPORT.md).
The [reference fixtures](examples/mnist/FIXTURES.md) now preserve test indices
`0..99`, labels, inputs, and every stage output. All 108 files reproduce exactly
in the recorded environment; offline tests cover every selected image.
The [`cinder run` command](docs/CLI.md) now implements exact-length finite
little-endian float32 input reading, inference, ranking, contextual errors, and
cleanup. Local checks pass: Debug 38/38, C-only Release 29/29, ASan/UBSan 29/29,
122 real command cases, and zero native leaks. The [demonstration](examples/mnist/DEMO.md)
records reproduction commands, actual output, accuracy, agreement, and limitations.
**M4 is complete as of October 27, 2026:** all five jobs passed for `7aa4bda` in
[run 40487216629](https://github.com/farukumarx64/cinder/actions/runs/40487216629).
Linux Debug passed 38/38 tests; macOS Debug/Release, Linux Release, and Linux
ASan/UBSan/LeakSanitizer passed 29/29 each. The demonstration report records
actual numerical results, existing GCC warnings, and platform limitations.
The [Phase 5 benchmark contract](docs/BENCHMARK.md) was defined October 27:
required `--iterations N` (1–1,000,000), ten fixed warm-ups, monotonic timing of
one public model call per sample, checked nanosecond totals, microsecond latency,
throughput, exact reports/errors, and failure cleanup. Shared model/input/output
preparation, cleanup, checked `CLOCK_MONOTONIC_RAW` timing, ten untimed warm-ups,
and measured execution are now implemented. `cinder bench` argument dispatch
and statistics/reporting were completed October 28, followed by the local
[acceptance audit](docs/BENCHMARK.md#benchmark-acceptance-audit). The full command
passes 1,088 controlled invocations and two unavailable-clock cases, using real
tiny/MNIST ownership and saved references. Current local Debug passes 46/46;
C-only Release and ASan/UBSan pass 37/37 each; native acceptance leak checks
report zero. The twelve labeled benchmark tests impose no speed threshold.
Hosted run [40688919176](https://github.com/farukumarx64/cinder/actions/runs/40688919176)
passed all five jobs for the unchanged implementation at `cd461c6`: Linux Debug
46/46; macOS Debug/Release, Linux Release, and Linux ASan/UBSan/LeakSanitizer
37/37 each. See [M5 verification](docs/M5_VERIFICATION.md) for exact evidence.
The [first Release baseline](docs/benchmarks/2026-10-28-apple-m4/README.md) now
records `cd461c6` on Apple M4: five attempts/model, ten warm-ups, and 100,000
measured iterations. Four tiny reports and all five MNIST reports succeeded;
the failed tiny zero-interval attempt is preserved. Median trial means are
0.159771025 us and 105.936159 us respectively. Fresh Release passes 37/37 and
allocation audits remain clear. Tiny timing reliability limits interpretation.
**M5 is complete as of October 29, 2026.** An identical-executable control
retains all five attempts/model: four valid tiny trials and five MNIST trials,
plus another tiny zero-interval failure. Median trial means are 0.15861274 us
and 104.596974 us; movement without code changes supports no optimization
claim. Fresh local Debug 46/46, Release/ASan+UBSan 37/37 each, four native leak
checks, numerical comparisons, and allocation audits pass. The existing hosted
run verifies the exact unchanged code; this closure adds documentation/evidence
without a commit, push, or new CI run.
**Phase 6 release-checklist audit completed October 29, 2026.** The
[audit](docs/RELEASE_AUDIT.md) reconciles all ten previously unchecked release
requirements: seven satisfied, `cinder info` deferred beyond v0.1.0, and the
final README review and GitHub release still open. Earlier M2-only test/CI
entries now reference M5 evidence; final candidate verification remains a
separate gate. Version finalization and release notes were pending at that audit.
**GCC warning investigation completed locally October 29, 2026.** All four
groups reproduce with GCC 13.3; checked byte extents and explicit initialization
resolve them without suppressing warnings. Linux Debug/Release pass 37/37 each;
macOS Debug passes 46/46, Release and ASan/UBSan 37/37 each, all with `-Werror`
and zero build warnings. Three focused native leak checks report zero. See the
[causes, changes, and evidence](docs/GCC_WARNINGS.md). All five hosted jobs for
`fb9a655` subsequently passed after retrying a macOS runner-capacity failure.
**Tiny-model timing investigation completed locally October 29, 2026.** The
[report](docs/TIMING_INVESTIGATION.md) captures equal raw clock readings around
completed tiny inferences in 19 instrumented failures. Original/current CLI
executables also reproduce the zero-interval rejection. Preserve the per-call
contract and explicit failure; no runtime change or batching is justified.
Add deterministic 41/42 ns and late-zero tests, and correct native CLI smoke
expectations to allow the exact documented zero rejection without retries.
Release passes 37/37, macOS
Debug/ASan+UBSan and Linux Release pass 12/12 benchmark tests each, and the
expanded controlled audit reports zero native leaks. Subsequent M6 hosted verification below includes these tests.
**Robustness coverage review completed locally October 30, 2026.** The
[report](docs/ROBUSTNESS_REVIEW.md) maps existing malformed-file, allocation,
capacity, cleanup, and recovery tests. Add 162 multi-chunk loader failures with
successful recovery, 2,567 parameter bit comparisons, and protected caller
buffers at both page edges. No production defect was exposed. Fresh macOS
Debug passes 47/47; Release and ASan/UBSan pass 38/38 each; Linux GCC Debug/Release
pass 38/38 each, all warning-free under `-Werror`. Three native leak checks
report zero. Subsequent M6 hosted verification below includes these tests.
**M6 completed October 30, 2026.** [Final verification](docs/M6_VERIFICATION.md)
passes at `cace0dfb4898ad4f569e8ebd60df3f13089ee966`: fresh local Debug 47/47,
Release/ASan+UBSan and Linux GCC Debug/Release 38/38 each, zero compiler warnings,
and seven native leak checks reporting zero. All five hosted jobs pass on the
first attempt in [run 40989772451](https://github.com/farukumarx64/cinder/actions/runs/40989772451):
Linux Debug 47/47; macOS Debug/Release, Linux Release, and Linux sanitizer/leak
checks 38/38 each. Numerical agreement, lifecycle recovery, and allocation-free
inference remain verified. Record tiny-clock, caller-precondition, platform,
format, and fixture-coverage limits.
**Release README completed October 30, 2026.** The [README](README.md) now
leads with source installation, build/test commands, `run`/`bench`, architecture,
ownership, measured results, and limitations; detailed history stays in linked
reports. Its recipes pass in an isolated source snapshot: Debug/Release and
ASan+UBSan 38/38 each, optional Python checks 47/47, both model examples,
benchmark command, deterministic export check, and zero native leaks. See
[recipe evidence](docs/verification/2026-10-30-readme.txt). This does not replace
the final release-candidate checkout/CI gate. That review recorded `0.1.0-dev`.
**Version and release notes finalized October 30, 2026.** The executable and
CLI test builds now use CMake's `0.1.0` project version without the `-dev` suffix.
The [release notes](RELEASE_NOTES.md) document supported features, commands,
ownership, measured results, compatibility, and constraints. Fresh local
Debug/Release pass 38/38 each; both and a build without tests print exactly
`Cinder 0.1.0`. The smoke test accepts `0.1.0` and rejects the old development
expectation. See [verification](docs/verification/2026-10-30-version.txt).
Historical output records retain their original version strings.
**Next scoped step: verify the final release candidate from a fresh checkout.**
Follow the [remaining release gates](#remaining-release-gates).
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
- [x] Implement exporter model validation: accept supported flat PyTorch Sequential models, validate every connection and format limit, and extract independent contiguous CPU float32 parameter snapshots. Unsupported layers, configurations, hooks, shapes, dtypes/devices, and non-finite parameters produce contextual errors before any file-writing stage. On October 24, 2026, all 33 Python tests passed, including ten saved NumPy inputs at four stages (40 vectors / 120 values, `atol = rtol = 1e-5`). The C-only suite passed 19/19 tests; the combined suite with fresh NumPy references and exporter validation passed 22/22. The Linux CI job enables exporter tests; hosted results subsequently passed at M3 closure. See [the exporter guide and verification results](docs/EXPORTER.md#verification--october-24-2026). Binary writing is still separate work.
- [x] Implement binary export and reproducible tiny-network fixtures: `export_model(model, path, *, input_size=None)` validates before file creation, writes explicit little-endian headers and bit-preserved float32 parameters, and publishes only a complete flushed/synced/closed temporary file. On October 24, 2026, all 17 binary-export tests and 33 validation tests passed; combined CTest with fresh NumPy references passed 23/23. The saved 168-byte model matches the specification exactly, all ten inputs and 40 expected stage vectors are preserved, and two regenerations match all twelve fixture files byte for byte. Independently decoded parameters reproduce 120 NumPy stage values within `atol = rtol = 1e-5`. File/allocation failure injection verifies destination preservation, cleanup, and retry. See [binary-export verification](docs/EXPORTER.md#binary-export-verification--october-24-2026). Hosted CI and the C loader round trip were pending at this step and subsequently passed at M3 closure.
- [x] Define the C loader contract on October 24, 2026: declare `cinder_model_load` and append IO/FORMAT/UNSUPPORTED/LIMIT statuses without changing existing values. Specify seekable read-only binary file handling, full format/resource/connection checks, exact EOF and successful close before model creation, per-record Dense staging copied by existing constructors, all-or-nothing ownership, NULL output/full cleanup on failure, first-error preservation, and independent repeated/concurrent loads. Headers, declaration, and caller example passed C11 syntax checks with warnings as errors; status values/signature checks passed, and the rebuilt Debug C suite passed 19/19. See [the loader contract and verification](docs/LOADER.md#contract-verification--october-24-2026). Implementation and behavioral tests are the next step.
- [x] Implement loading and model construction: `cinder_model_load` checks host encoding, file/header/record bounds, dimensions/connections, exact payload sizes, finite parameters, and EOF/close before publishing a model constructed through existing layer/model APIs. Per-record staging and partial layers are cleaned up on failure. On October 24, 2026, the new C test passed successful tiny/non-square/activation-only/multi-block loads and 206 rejection cases with and without error details; 50 repeated final-probability vectors / 100 values matched NumPy (`atol = rtol = 1e-5`, maximum absolute error `5.96046448e-8`). Debug with NumPy/exporter suites passed 24/24 tests; C-only Release and combined ASan/UBSan passed 20/20 each. Native leaks reported zero; the linked caller example passed. See [loader implementation results and limitations](docs/LOADER.md#implementation-verification--october-24-2026). Exhaustive fault injection, full intermediate-stage round trip, and hosted CI remain separate steps.
- [x] Complete rejection and cleanup tests on October 24, 2026: 709 parser/argument/file rejection cases run with and without diagnostics, including all tiny-file truncations, all nine type-to-type connection mismatches, non-finite encodings at every parameter position, all one-byte suffixes, and real files at resource boundaries. The tracked audit verifies 1,562 rejected calls, forces every one of the tiny model's 21 allocations and 29 stdio-failure scenarios, checks NULL output/complete cleanup/first-error preservation, and retries successfully. Exact parameter bits and test-only lowered native-size overflow guards pass. Debug with NumPy/exporter checks passed 25/25; C-only Release and combined ASan/UBSan passed 21/21 each. Both loader executables reported zero native leaks. No production/API changes were required. See [the audit and limitations](docs/LOADER.md#rejection-and-cleanup-audit--october-24-2026); hosted CI subsequently passed at M3 closure.
- [x] Validate the complete round trip on October 25, 2026: export the full tiny model and three prefixes, load each through the public C API, read the ten raw input files, and compare all four stages with saved NumPy references. Both the committed-fixture C test and the optional fresh Python-export test compare 40 vectors / 120 values with shape/finite checks and `atol = rtol = 1e-5`. First Dense/ReLU/logits match exactly locally; maximum Softmax error is `5.96046448e-8`. Nine deliberately altered stage/input fixtures are rejected. All 15 fixture files regenerate identically; original model/input/reference values remain unchanged. C-only Release passed 22/22 tests; Debug and combined ASan/UBSan with NumPy/exporter options passed 27/27 each. See [round-trip verification and scope](docs/ROUND_TRIP.md); hosted verification subsequently passed at M3 closure.
- [x] Verify lifecycle behavior and close M3 on October 25, 2026. Local checks passed 502 successful loads, 100 rejected loads, 4,720 success/failure/recovery cycles (14,160 inference calls), concurrent independent loads/models, caller-serialized sharing, and continued execution after source deletion. The tracked audit adds 284 poisoned-buffer cycles (852 inference calls), zero inference allocation/free attempts, unchanged parameters, and all 21 allocation failures with/without diagnostics while another owner survives. Debug and ASan/UBSan passed 28/28; C-only Release passed 23/23; both native leak checks reported zero. Hosted revision `5ad7dc4`, including the round-trip and lifecycle additions, passed 23/23 macOS, 28/28 Linux, and 23/23 Linux sanitizer/leak tests in [run 40192296107](https://github.com/farukumarx64/cinder/actions/runs/40192296107). See [actual results and limitations](docs/M3_VERIFICATION.md).

Earlier checklist entries record the results at each implementation step. All
Phase 3 additions are included in the successful hosted revision above; M3 has
no outstanding verification requirement. Platform and concurrency limitations
remain documented in the M3 report.

### Phase 4 checklist

- [x] Create the reproducible Python training script on October 25, 2026: implement CPU float32 Dense(784 -> 128), ReLU, Dense(128 -> 10), logits with CrossEntropyLoss, exact row-major float32 division by 255, fixed seeds and a separate shuffle generator, deterministic algorithms, and one intra-op thread. Read/check original MNIST IDX archives, download only on explicit request, and save a final checkpoint plus settings, data hashes, dependency/environment details, and metrics in a new run directory. The 18 offline tests use synthetic data and include fresh-process parameter/result repeatability; the combined Debug suite passed 29/29, the C-only Release suite passed 23/23, and the independent MNIST CTest option passed. No real dataset was downloaded or trained in this step. See [the example and verification](examples/mnist/README.md).
- [x] Train and evaluate the reference model on real MNIST on October 25, 2026. The fixed seed `20261025`, five-epoch CPU run trained on 60,000 official training examples and achieved 97.48% accuracy (9,748/10,000) with mean cross-entropy loss `0.0826597822714597` on the held-out test split. No test examples were used for parameter updates or model/settings selection. Preserve the first run's checkpoint, exact configuration/metrics, log, and verification record under `examples/mnist/reference/`. Fresh-process reload reproduced the exact test metrics without changing parameters; another complete training run reproduced all 101,770 parameters bit for bit and all metadata exactly in the recorded macOS environment. See [actual results and limitations](examples/mnist/RESULTS.md).
- [x] Export the saved model with inference Softmax, verified October 26, 2026. The wrapper strictly restores finite CPU float32 parameters, appends `Softmax(dim=-1)`, switches the entire sequence to evaluation mode, and calls the existing exporter. The 407,160-byte `mnist.cnd` contains all 101,770 trained parameter values unchanged. Two fresh processes produce identical bytes; independent inspection checks every parameter bit, all layer headers, and exact EOF. Save the binary and export provenance record; provide a non-mutating `--check` command. All 11 export tests and the combined 30/30 Debug CTest suite pass. Fix the pre-existing training CLI test to use a temporary output directory so saved experiments cannot affect its missing-data assertion. See [verification and remaining scope](examples/mnist/EXPORT.md); see the completed C comparison and M4 closure entries below.
- [x] Generate deterministic reference fixtures on October 26, 2026 for official test indices `0..99`, in original order without prediction filtering. Preserve indices/labels, 100 individual 784-value raw inputs, Dense1/ReLU/logits/probabilities as row-major float32 arrays, and the selected original pixels for offline verification. The generator checks archive provenance and checkpoint/model identity, uses single-vector CPU float32 evaluation, checks shapes/finiteness/normalization, and records hashes and `atol = rtol = 1e-5` in the manifest. Two fresh CLI processes reproduce all 108 files / 548,481 bytes exactly; the non-mutating `--check` also passes. All 11 fixture tests and the combined 31/31 Debug CTest suite pass. The fixed subset includes its one misclassification (index 8, label 5, prediction 6); 99/100 is a regression-subset result, separate from full-test accuracy. See [layout, commands, evidence, and limitations](examples/mnist/FIXTURES.md). C-side comparison is now complete as recorded below.
- [x] Define the `cinder run <model.cnd> <input.bin>` contract on October 26, 2026. Specify exactly two nonempty literal file paths, one preprocessed vector, public loader/model API use, basename display, zero-based argmax prediction, and the top `min(3, output_size)` raw scores with lower-index ties and `%.9g` formatting. Define normal exits `0` success / `1` operational failure / `2` usage error; stderr templates, status tags, input-reader messages, failure precedence, delayed stdout, write-error handling, and ownership cleanup. Accept valid non-Softmax models without implicit activation or percentage claims. Align earlier illustrative roadmap output with this contract. Check worked display values against saved Python fixtures; no source/header/build change or new CLI implementation is part of this step. See [the contract and implemented acceptance coverage](docs/CLI.md).
- [x] Implement exact-length finite float32 little-endian input-file reading on October 27, 2026. The CLI-private reader validates representation, checked extent, measured length before allocation, finite values, exact reads/EOF, and close. Tests force allocation and every file-operation failure, preserve the first error, and verify one close attempt and zero retained allocations. See [implementation and tests](examples/mnist/DEMO.md#verification-of-the-cli-addition).
- [x] Implement `cinder run` on October 27, 2026 using the public model API, caller-owned input/output buffers, stable top-three ranking, escaped names/errors, and the documented `0/1/2` exits. Cleanup precedes output, and writes/flush are checked. Three new C tests cover controlled API/I/O/allocation/output failures plus 122 real executable cases, including all 100 MNIST inputs with matching predictions/top-three indices. Local Debug passes 38/38; C-only Release and ASan/UBSan pass 29/29 each; native leak checks report zero. See [the demonstration](examples/mnist/DEMO.md). All five hosted jobs subsequently passed as recorded in the closure entry.
- [x] Validate every saved MNIST stage against Python on October 26, 2026. The C-only test loads the saved model and byte-preserving prefixes, runs all 100 original inputs, and checks shapes, finite values, exact input/array lengths, preprocessing, caller buffers, and the unchanged `atol = rtol = 1e-5` rule. All 400 vectors / 27,600 values pass; maximum absolute errors are `6.19888306e-6` (Dense1), `5.7220459e-6` (ReLU/logits), and `3.57627869e-7` (probabilities). Runtime prediction agreement is 100/100; C and Python each classify 99/100 fixed-subset images correctly, separate from the prior 97.48% full-test Python result. The optional fresh-export test verifies prefix identities, rejects 28 corrupted cases, and proves that a valid label-only change affects accuracy without affecting agreement. Debug passes 33/33 tests; C-only Release and ASan/UBSan each pass 24/24; native leaks reports zero. Production code and saved answers are unchanged. See [measured results, commands, and hosted verification](examples/mnist/VALIDATION.md). The subsequent CLI implementation and its passing hosted verification are recorded in this checklist.
- [x] Verify larger-model lifecycle behavior and CI, completed October 27, 2026. Local verification passed October 26, 2026: 2,100 public-API inference calls and 1,384 instrumented calls cover repeated inputs, failure recovery, independent owners, deleted source files, caller-serialized shared access, poisoned 784-float workspaces, partial-write failures at all four layers, unchanged parameters, and all 21 allocation points failing with/without diagnostics. Debug passed 35/35 tests; C-only Release and ASan/UBSan passed 26/26 each; both native leak checks reported zero. Ordinary C tests use committed fixtures without Python or MNIST downloads. The workflow now covers Debug/Release on macOS/Linux and Linux sanitizer/leak checks. The first hosted macOS native leak step did not complete and was removed; native macOS checks passed locally. All five jobs passed for `bd6cbd9` in [run 40417063563](https://github.com/farukumarx64/cinder/actions/runs/40417063563): Linux Debug 35/35; macOS Debug/Release, Linux Release, and Linux ASan/UBSan/LeakSanitizer 26/26 each. See [the lifecycle report](examples/mnist/LIFECYCLE.md) for exact toolchains and the native hosted macOS leak-check limitation. The subsequent [demonstration report](examples/mnist/DEMO.md) tracks CLI implementation and M4 closure.
- [x] Document the demonstration and close M4 on October 27, 2026. [DEMO.md](examples/mnist/DEMO.md) records measured accuracy and agreement separately, all-stage numerical errors, offline build/run commands, Python reproduction, actual C output, local checks, and limitations. The saved export and all 108 fixture files regenerate identically. All five jobs for `7aa4bdaca26149a1746779817b8f42c6813a9bc6` passed in [run 40487216629](https://github.com/farukumarx64/cinder/actions/runs/40487216629): Linux Debug 38/38; macOS Debug/Release, Linux Release, and Linux ASan/UBSan/LeakSanitizer 29/29 each. Every job passed the 122 CLI cases and 27,600 stage comparisons with 100/100 prediction/top-three agreement; lifecycle and allocation-failure audits also passed. Record existing GCC warnings, native hosted macOS leak-check limitations, and unmeasured full-test C accuracy explicitly. M4 is complete; Phase 5 benchmarking remains separate.

### Phase 5 checklist

- [x] Define the benchmark command and measurement contract on October 27, 2026. [BENCHMARK.md](docs/BENCHMARK.md) specifies fixed argument order, checked decimal iterations in 1..1,000,000, ten untimed warm-ups, reuse of one model/input/output, monotonic per-call wall-clock timing including public API validation/copies, exclusion of setup/observation/printing, checked integer nanosecond accumulation, microsecond latency and throughput formulas, exact nine-line output, `0/1/2` exits, contextual failures/precedence/cleanup, and controlled-clock acceptance cases. Verify synthetic arithmetic and documentation consistency only; no implementation, benchmark timings, build, or new CI results are claimed.
- [x] Prepare reusable CLI model/input setup and cleanup on October 27, 2026. Private session helpers load once, read/close input once, allocate disjoint output, publish ownership only on success, and release/reset all resources. `run` retains its existing arguments/output/errors. Tiny/MNIST audits each reuse prepared buffers for 100 calls after source deletion, fail all 23 allocation points with/without details, check I/O/invalid-input failures and recovery, and retain zero blocks/bytes/streams. Local Debug passes 39/39; C-only Release and ASan/UBSan pass 30/30 each; native setup and actual MNIST command leak checks report zero. No hosted CI or timing result is claimed for this step. See [preparation verification](docs/BENCHMARK.md#preparation-implementation-and-verification).
- [x] Implement the monotonic timer on October 27, 2026. Private init/start/stop helpers use `CLOCK_MONOTONIC_RAW` on macOS/Linux, query positive resolution once, validate normalized timestamps and within/across-call ordering, subtract before checked uint64 conversion, and reject zero intervals without retrying or rounding. Tests cover controlled clock/read/representation/overflow failures, second borrowing, very large uptime, failure recovery, independent state, unavailable-backend rejection, and the real native clock. Local Debug passes 42/42; C-only Release and ASan/UBSan pass 33/33 each; native controlled/native timer leak checks report zero. Native resolution is 42 ns on this Mac. Linux execution and hosted CI are not claimed; Docker was paused. See [timer verification](docs/BENCHMARK.md#monotonic-timer-implementation-and-verification).
- [x] Implement ten warm-up calls and the measured loop on October 27, 2026. A private helper borrows the prepared session, initializes timing once, consumes all finite outputs through a local volatile sink outside timing, and accumulates checked raw count/total/minimum/maximum nanoseconds. It stops at the first failure, preserves inference-over-end-clock error precedence, and publishes no partial result. Controlled tests cover exact counts through 1,000,000 iterations, timing exclusion, every failure position, non-finite outputs, total overflow, and recovery. Real tiny/MNIST loops reuse buffers after source deletion with unchanged allocation/file counters and saved-reference agreement. Debug passes 43/43; C-only Release and ASan/UBSan pass 34/34 each; Release/ThinLTO focused audits pass 2/2. CLI dispatch/reporting and hosted CI remain pending; see [loop verification](docs/BENCHMARK.md#warm-up-and-measured-loop-implementation-and-verification).
- [x] Calculate/display statistics and connect `cinder bench` on October 28, 2026. Derive microsecond mean/minimum/maximum and inferences/s from unrounded integer measurements, reject zero divisors/inconsistent metadata, and require finite positive statistics. Validate the complete command before setup, release ownership before reporting, preserve contextual errors, and check every write/flush. Known arithmetic and formatting tests plus 11 new real-process benchmark cases pass; the process suite retains all 122 prediction cases. Debug passes 44/44; C-only Release and ASan/UBSan pass 35/35 each; native CLI/actual-command leak checks report zero. Hosted CI and a reproducible baseline remain pending; see [statistics/report verification](docs/BENCHMARK.md#statistics-and-cli-report-implementation-and-verification).
- [x] Complete benchmark acceptance tests locally on October 28, 2026. Add the full real CLI/runtime audit with controlled OS timestamps and a compiled unavailable-clock variant: 1,088 controlled and two unavailable invocations cover counts, warm-up/setup/observation exclusion, exact statistics, clock/inference/non-finite/output errors, first-error precedence, all 23 allocation points, file failures, complete cleanup, and recovery on tiny/MNIST. Successful inference retains saved-reference agreement. Label twelve ordinary C tests `benchmark`; the existing macOS/Linux jobs include them without a speed threshold or Python/download requirement. Debug passes 46/46; C-only Release and ASan/UBSan pass 37/37 each; both new native leak checks report zero. Hosted execution remains unverified for this step. See [acceptance evidence and coverage map](docs/BENCHMARK.md#benchmark-acceptance-audit).
- [x] Record the first Release baseline on October 28, 2026 at clean commit `cd461c696aea932e6ea53e4093008d6463c42fa9`: Apple M4, macOS 27.0.1, Apple Clang 21.0.0, `-O3 -DNDEBUG`, sanitizers/LTO off, ten warm-ups and 100,000 requested measured iterations per attempt. Preserve all five attempts/model: four tiny and five MNIST valid reports, plus tiny trial 1's zero-interval error at iteration 37,270. Median valid-trial means are 0.159771025 us (tiny) and 105.936159 us (MNIST); raw output, timestamps, source/binary/fixture identities, environment/power/load observations, commands, and limits are retained. Fresh Release passes 37/37 with numerical agreement and zero runtime allocation/free attempts in instrumented inference paths. This is a macOS baseline with a documented tiny-clock limitation, not an optimization or Linux result. See [baseline and evidence](docs/benchmarks/2026-10-28-apple-m4/README.md).
- [x] Evaluate improvements and close M5 on October 29, 2026. Preserve the runtime because no measured candidate justifies a change. The October 28 unchanged-executable control repeats five attempts/model with identical settings and hashes; retain the second tiny zero-interval failure and compare all valid results as variation, not speedup. Fresh local Debug passes 46/46; Release and ASan/UBSan pass 37/37 each; four native leak checks report zero. Numerical/shape/finite checks, 100/100 MNIST prediction agreement, and zero audited inference allocation/free attempts remain unchanged. All five hosted jobs passed for exact code revision `cd461c6` in [run 40688919176](https://github.com/farukumarx64/cinder/actions/runs/40688919176). Record GCC warnings, desktop/timing limitations, and the hosted native macOS leak-check limitation. See [M5 verification and evidence](docs/M5_VERIFICATION.md).

### Phase 6 checklist

- [x] Audit the remaining release checklist on October 29, 2026. Compare all ten unchecked requirements with source, registered tests, and saved M3–M5 evidence; correct seven stale entries, explicitly defer `cinder info`, and preserve open documentation/publication gates. Confirm the existing binary accepts `run`, rejects `info`, and reports `0.1.0-dev`; read-only GitHub checks report no releases or tags. See [the audit and scope decision](docs/RELEASE_AUDIT.md). This documentation step does not close M6.
- [x] Investigate the Linux GCC warnings locally on October 29, 2026. Reproduce all four groups with Ubuntu 24.04/GCC 13.3: size guards are always false for encoded counts on this host; metadata/output/result warnings reflect initialization facts not established by optimized analysis. Add shared checked byte extents with boundary tests, defined snapshots, NaN-poisoned test output, and an invalid initial benchmark result. No reachable uninitialized read or allocation overflow was found in these paths. Linux Debug/Release pass 37/37 each; macOS Debug 46/46, Release and ASan/UBSan 37/37 each, all warning-free under `-Werror`; three native leak checks report zero. Numerical/lifecycle/allocation checks pass. See [the investigation and limitations](docs/GCC_WARNINGS.md); all five subsequent hosted jobs passed at `fb9a655`, final M6 verification of the resulting revision is recorded below.
- [x] Investigate tiny-model timing reliability on October 29, 2026. Preserve all 72 diagnostic attempts: original/current/probe tiny runs reject 19/20, 20/20, and 19/20 times; all nine MNIST attempts pass. All 19 instrumented tiny rejections capture exactly equal native start/end timestamps with correct outputs and no published result. The native timebase is 125/3 ns, with reported resolution 42 ns and observed 41/42 ns increments. No timer arithmetic/loop defect was found; retain explicit zero rejection and the unchanged measurement contract. Add controlled 41/42 ns and failures at calls 9853, 37270, and 100000; verify immediate stopping, result/report preservation, cleanup, and recovery. Correct native CLI smoke expectations to accept only a valid report or the exact in-range zero-interval rejection, with no retries; a constant-clock control passes all 133 process cases including seven zero rejections. Release passes 37/37; macOS Debug/ASan+UBSan and Linux Release each pass 12/12 benchmark tests, with zero native leaks in the controlled audit. See [evidence, limits, and future batching requirements](docs/TIMING_INVESTIGATION.md). These test additions subsequently pass the final M6 hosted checks below.
- [x] Review the remaining robustness coverage locally on October 30, 2026. Map invalid models, dimensions, capacities, malformed files, allocation/I/O failures, cleanup, and recovery to existing tests. Add 162 failures/retries across 4 KiB decoder chunks, 2,567 parameter bit comparisons, and protected-page caller-buffer checks on macOS/Linux. No production defect was exposed. Fresh macOS Debug passes 47/47, Release/ASan+UBSan 38/38 each, Linux GCC Debug/Release 38/38 each, with zero build warnings under `-Werror`; three native leak checks report zero. Preserve caller-pointer/lifetime/concurrency preconditions and platform limits. See [coverage and evidence](docs/ROBUSTNESS_REVIEW.md). Hosted verification of these changes passes in the final M6 checks below.
- [x] Verify the resulting revision and close M6 on October 30, 2026. At `cace0df`, fresh local macOS Debug passes 47/47; Release/ASan+UBSan and Linux GCC Debug/Release pass 38/38 each, with zero build warnings under `-Werror`. Seven native leak checks report zero. All five jobs pass in [run 40989772451](https://github.com/farukumarx64/cinder/actions/runs/40989772451), attempt 1: Linux Debug 47/47; the other four 38/38, including Linux ASan/UBSan/LeakSanitizer. Preserve `atol = rtol = 1e-5`, all 27,600 MNIST stage comparisons, 100/100 prediction agreement, recovery and zero inference-allocation audits. Record actual results and accepted limitations in [M6 verification](docs/M6_VERIFICATION.md). Phase 7 release documentation is next; later candidate changes require verification.

---

# 1. v0.1.0 Goal

The v0.1.0 CLI scope, reconciled October 29, 2026, is:

```text
cinder --version
cinder run <model.cnd> <input.bin>
cinder bench <model.cnd> <input.bin> --iterations N
```

`cinder info` was part of the original plan but is explicitly deferred beyond
v0.1.0. It is not implemented and is not a release gate; see the
[scope decision](docs/RELEASE_AUDIT.md#scope-decision-defer-cinder-info-beyond-v010)
and the [post-MVP backlog](#17-post-mvp-roadmap). The export → load → inference
workflow and its Python validation remain the release goal.

Inference:

```bash
$ cinder run mnist.cnd digit.bin
```

Actual local output under the [CLI contract](docs/CLI.md) for saved test index 0
(October 27, 2026); see [reproduction commands](examples/mnist/DEMO.md):

```text
Model: mnist.cnd
Prediction: 7
Scores:
7	0.9978019
3	0.0018843736
2	0.000236767111
```

Benchmarking command, specified in [the contract](docs/BENCHMARK.md):

```bash
$ cinder bench mnist.cnd digit.bin --iterations 1000
```

Planned output format; placeholders below are not measured values:

```text
Model: mnist.cnd
Iterations: 1000
Warm-up iterations: 10
Clock: monotonic
Clock resolution (ns): <resolution>
Mean latency (us): <mean>
Minimum latency (us): <minimum>
Maximum latency (us): <maximum>
Throughput (inferences/s): <throughput>
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
- [x] Model loading from disk (C API; parser, exact-limit, and forced-failure cleanup tests pass locally)
- [x] Custom Cinder model-format contract (version 1; exporter and C loader implemented)
- [x] Python model exporter (validation and binary writing; tiny fixtures verified locally)
- [x] CLI inference
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

Phase 3's [loader guide](docs/LOADER.md) documents this implemented entry point
in the model header, including rejection/cleanup tests:

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
- Model loading sets `*out_model` to NULL on failure and frees all partial state it owns; implementation and cleanup verification are complete.
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
- The CLI displays the supplied path's basename, including its extension,
  such as `Model: mnist.cnd`. Descriptions and class labels live in example docs.
  The [run contract](docs/CLI.md) defines escaped display, raw top-three scores,
  zero-based class indices, errors, and process exit statuses.

No native structs, `size_t`, pointers, or enum objects are written to disk. A
finite weight changed by a bit flip can still be structurally valid; version 1
has no checksum or authenticity mechanism.

The specification includes the complete 168-byte encoding of our fixed
`2 -> 4 -> 2` network, exact offsets, sample input bytes, and expected stages.
The exporter reproduces this fixture exactly; the C loader validates and
executes it against the preserved reference cases.

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
defines the exact layout. The [MNIST demonstration](examples/mnist/DEMO.md)
repeats the preprocessing requirements and runs the saved sample files.

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
Cinder 0.1.0
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
contract was completed on October 23, 2026, and exporter validation and binary
writing on October 24; the Phase 3 checklist in Current Progress records the
actual status. The C loader API and ownership/cleanup contract were also defined
on October 24. Loading, parser rejection, exact-limit, and forced-failure cleanup
tests are implemented. The full intermediate-stage file round trip passed on
October 25. Loaded-model lifecycle, local verification, and all three hosted CI
jobs passed for `5ad7dc4`, closing M3 on October 25, 2026.

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

The validation API, `validate_model(model, *, input_size=None)`, and binary API,
`export_model(model, path, *, input_size=None)`, were implemented and verified
locally on October 24, 2026. See [the exporter guide](docs/EXPORTER.md) for
supported configurations, snapshot ownership, file failures, fixtures, and tests.

Usage from Python after constructing or training a supported model:

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

The API and ownership/error/file rules are defined in [docs/LOADER.md](docs/LOADER.md).
The declared loader must close the validated file before using the existing
model constructor to adopt the loader-owned layers and publish the result.

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

The October 19–22 dates below are the original planning targets. M3 closed on
October 25, 2026. The reproducible training script was implemented and verified
offline on October 25; real MNIST training/evaluation completed that same day
with 97.48% held-out test accuracy. Trained checkpoint export was verified on
October 26, and the first 100 real-image reference fixtures were generated and
verified the same day. Every saved stage now passes C comparison, with 100/100
prediction agreement and 99/100 subset classification accuracy in both runtimes;
see [the validation report](examples/mnist/VALIDATION.md). The `cinder run`
contract was defined October 26 and implemented October 27 with the input
reader. [DEMO.md](examples/mnist/DEMO.md) records actual output and local checks.
The Phase 4 checklist records actual progress. M4 closed October 27 after
all five hosted jobs passed for the new CLI; the original dates remain targets.

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

Actual local output under the [CLI contract](docs/CLI.md), captured October 27,
2026 for saved test index 0; see [the demonstration](examples/mnist/DEMO.md):

```text
Model: mnist.cnd
Prediction: 7
Scores:
7	0.9978019
3	0.0018843736
2	0.000236767111
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

October 23–25 was the original planning target. M4 closed October 27; the
benchmark contract, shared CLI preparation, checked monotonic timer, and
warm-up/measured execution were completed that day. CLI argument dispatch and
latency/throughput reporting and the local acceptance audit were completed
October 28. The first Release baseline was recorded that day, including a
failed tiny-network timing attempt and its limitations. Evaluation retained the
implementation after an unchanged-executable control. Numerical/lifecycle,
local Debug/Release, sanitizer/native leak checks, and all five hosted jobs
passed. **M5 closed October 29, 2026**; see [the closure report](docs/M5_VERIFICATION.md).
The [Phase 5 checklist](#phase-5-checklist) records the scoped steps.

The authoritative [command and measurement contract](docs/BENCHMARK.md) specifies:

```bash
cinder bench <model.cnd> <input.bin> --iterations N
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

Apply the contract before interpreting results:

- Use an optimized Release build and record the CPU, operating system, compiler, and build options.
- Load the model and input and allocate working buffers before timing.
- Require `--iterations N` in 1..1,000,000 and run ten fixed untimed warm-ups, then measure each public `cinder_model_run` call with a monotonic clock. Include its validation/copies; exclude loading, allocation, preprocessing, result observation, statistics, cleanup, and printing.
- Observe all output values outside each interval, preserve reusable buffers, and stop on failure without printing partial statistics.
- Keep checked uint64 nanosecond totals/minimum/maximum; reject zero/backward time and arithmetic overflow. Report microsecond latency and throughput as measured count divided by total measured time, with the clock resolution. Treat noisy individual measurements cautiously.
- Confirm numerical correctness after any performance change.

Do not obsess over beating established runtimes.

Benchmark Cinder against **previous versions of Cinder**.

Illustrative comparison only; these are not measured implementations or results:

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

October 26–28 was the original planning target. M5 closed October 29;
**M6 closed October 30** after [final verification](docs/M6_VERIFICATION.md)
passed locally and in all five hosted jobs at `cace0df`.
The [release-checklist audit](docs/RELEASE_AUDIT.md) is
complete and `cinder info` is deferred. The [GCC warning investigation](docs/GCC_WARNINGS.md)
also passes locally and in hosted CI. The [tiny timing investigation](docs/TIMING_INVESTIGATION.md)
retains explicit rejection for unresolved native clock intervals and adds local
regression coverage. The [robustness review](docs/ROBUSTNESS_REVIEW.md) now adds
multi-chunk loader faults and protected caller-buffer tests, all passing locally
and in hosted CI. Next is Phase 7 release documentation.
The [Phase 6 checklist](#phase-6-checklist) records these scoped steps. Change
behavior only for demonstrated defects or an explicitly revised contract.

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

The release README rewrite is complete October 30; its source-build workflow,
commands, architecture/ownership, measured results, and limitations are verified
as recorded in [the recipe evidence](docs/verification/2026-10-30-readme.txt).
Version finalization and [release notes](RELEASE_NOTES.md) are also complete.
Final candidate verification and authorized publication remain. M7 stays open.

The README covers the planned topics below:

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
Benchmarking
```

---

# 16. Definition of Done

Audited October 29, 2026; [the evidence table](docs/RELEASE_AUDIT.md#every-previously-unchecked-requirement)
accounts for each formerly unchecked entry. Checked capabilities have source,
test, or documentation evidence through M6. They do not replace the final
release-candidate gates below.

## Implemented and verified capabilities

- [x] Pure C in-memory inference runtime works
- [x] Builds on macOS
- [x] Builds on Linux
- [x] Tensor representation works
- [x] Matrix-vector multiplication works
- [x] Dense, ReLU, and Softmax work through common dispatch and sequential models
- [x] ReLU works
- [x] Softmax works
- [x] `.cnd` models load from disk through `cinder_model_load`; see [loader implementation and tests](docs/LOADER.md)
- [x] Python exporter works for the supported subset; tiny model and fixtures verified locally
- [x] MNIST model executes through the public C API and CLI on 100 saved real inputs
- [x] Tiny exported-network fixtures match every reference stage; [M3 round-trip verification](docs/ROUND_TRIP.md) preceded MNIST integration
- [x] Saved tiny-network and 100-image MNIST stages satisfy documented tolerances with matching shapes and finite values
- [x] [Input layout and preprocessing](docs/MODEL_FORMAT.md#input-file-and-preprocessing-contract) are documented; Python/C fixture checks confirm row-major float32 `/255` inputs
- [x] Buffer lengths, output capacity, ownership, and error behavior are [documented and tested](docs/MODEL.md#inference-validation-and-failure-behavior), subject to valid-pointer/storage/lifetime preconditions
- [x] `cinder run` works; [actual demonstration and CLI verification](examples/mnist/DEMO.md) pass
- [x] `cinder bench` passes local and macOS/Linux hosted checks; see M5 verification for the tiny-clock limitation
- [x] Tests pass through M6 at `cace0df`: local Debug 47/47; ordinary C Release/ASan+UBSan and Linux GCC Debug/Release 38/38 each; see [M6 verification](docs/M6_VERIFICATION.md)
- [x] All five macOS/Linux jobs pass for M6 revision `cace0df`; [exact CI evidence](docs/M6_VERIFICATION.md#hosted-ci)
- [x] M6 ASan/UBSan, seven supported native macOS leak checks, and hosted Linux LeakSanitizer report no findings in exercised paths; platform limits are documented
- [x] Loader rejection and cleanup pass malformed-file, resource-boundary, allocation/I/O-failure, and lifecycle tests; see [the audit](docs/LOADER.md#rejection-and-cleanup-audit--october-24-2026)
- [x] Benchmarks document their environment, preserve failed attempts, and measure inference separately from loading and preprocessing; see M5 verification
- [x] Architecture is documented across the [model](docs/MODEL.md), [layer](docs/LAYER.md), [Dense](docs/DENSE.md), [tensor](docs/TENSOR.md), and [loader](docs/LOADER.md) guides
- [x] Model format version 1 is documented

## Remaining release gates

- [x] Close M6 on October 30 after GCC warning triage, tiny-clock investigation, robustness review, and verification of `cace0df`; see [the final report](docs/M6_VERIFICATION.md).
- [x] Finish the release README on October 30: source installation, build/test commands, `run`/`bench`, architecture and memory ownership, measured results, and limitations, with history in linked reports. Isolated-snapshot recipes pass Debug/Release/ASan+UBSan 38/38 each, optional Python 47/47, model/benchmark/export examples, and a native leak check. That review recorded `0.1.0-dev`; version finalization below updates the current output. The final candidate checkout/CI gate remains open. See [evidence](docs/verification/2026-10-30-readme.txt).
- [x] Finalize release version reporting and notes on October 30. CMake uses project version `0.1.0` for the executable, version smoke expectation, CLI audit, and both benchmark audit builds. Fresh local Debug/Release pass 38/38 each; builds with and without tests print exactly `Cinder 0.1.0` with empty stderr. A negative control rejects the obsolete `0.1.0-dev` expectation. Current examples and the [release notes](RELEASE_NOTES.md) describe the release features/constraints; `.cnd` stays at format version 1 and historical reports remain intact. See [evidence](docs/verification/2026-10-30-version.txt). Final candidate verification and publication remain separate gates.
- [ ] Verify final documented commands from a fresh release-candidate checkout, and run final Debug/Release, numerical/lifecycle, sanitizer/leak, and macOS/Linux CI checks after the last code/build changes.
- [ ] Publish a `v0.1.0` tag and GitHub release identifying the verified revision after the user authorizes publication. Neither existed in the October 29 read-only audit.

`cinder info` is **deferred beyond v0.1.0**, not checked off as implemented.
See [the decision](docs/RELEASE_AUDIT.md#scope-decision-defer-cinder-info-beyond-v010)
and [backlog](#17-post-mvp-roadmap). Existing passing checks must remain green;
M6 is complete; M7 remains open.

Once these boxes are checked:

**Ship it.**

Do not delay the release because of features that belong in future versions.

---

# 17. Post-MVP Roadmap

**Deferred October 29, 2026: `cinder info <model.cnd>`.** Model inspection is
not part of v0.1.0 and has no committed delivery date. Define read-only metadata
access and the output/error contract before implementation; reuse the validated
loader and test reporting, counts, failures, and cleanup. See [the scope decision](docs/RELEASE_AUDIT.md#scope-decision-defer-cinder-info-beyond-v010).

The original proposed example is retained below as an **unimplemented future
illustration**, not current command output:

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

Follow these milestones in order even if individual daily targets shift.
**M0 through M5 are complete.** The dated checklists above preserve the steps
and their original verification results. M2 closed October 23 after tiny-network
NumPy/lifecycle checks and hosted CI; M3 closed October 25 after export/load,
rejection/cleanup, and round-trip verification. M4 closed October 27 after the
MNIST demonstration and CLI passed all five hosted jobs. Python full-test
accuracy is 97.48%; C/Python agreement is 100/100 saved inputs, separate from
99/100 classification accuracy on that fixed subset.

M5 closed October 29 after the baseline and unchanged-executable comparison,
fresh local numerical/lifecycle/Debug/Release/sanitizer/leak checks, and
verification of all five hosted jobs at `cd461c6`. No performance edit was
justified; tiny clock failures, GCC warnings, and platform-check limitations
are retained in [the M5 report](docs/M5_VERIFICATION.md). Phase 6's final
hardening review is now complete. Its release-checklist audit reconciles seven stale
requirements, inspection is explicitly deferred, and remaining
release gates are recorded. GCC warning triage and the tiny timing investigation
are complete; the latter preserves the explicit failure policy and records its
native-clock evidence and local regressions. The robustness review also passes
locally and in hosted CI, adding chunk-boundary faults and protected-buffer tests.
[M6 closed October 30](docs/M6_VERIFICATION.md) after fresh local checks and all
five hosted jobs passed at `cace0df`. Phase 7's release README is now complete,
with a verified supported workflow and architecture/ownership overview. Version
finalization and release notes are also complete. Next is final candidate
verification, followed by authorized publication. M7 remains open.

The **October 25 deadline is the internal deadline**.

October 31 is the actual release deadline.

That gives Cinder almost a full week of buffer for bugs, life interruptions, documentation, and unexpected problems.

---

# Final Objective

By the end of the project, you should be able to say:

> **I built a neural-network inference runtime from scratch in C. It defines its own binary model format, loads pretrained neural networks, manages their parameters in memory, executes forward inference using custom tensor operations, benchmarks runtime performance, and produces outputs validated against a reference implementation.**

That is Cinder `v0.1.0`.
