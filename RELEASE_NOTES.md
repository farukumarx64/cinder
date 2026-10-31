# Cinder v0.1.1 — Model Inspection

**Prepared October 31, 2026; publication pending.** The source reports
`Cinder 0.1.1`. The owner will create tag `v0.1.1` and publish the release after
pushing the candidate and checking macOS/Linux CI.

## Added

- **Model inspection:** `cinder info <model.cnd>` reports input/output sizes,
  the ordered Dense/ReLU/Softmax layers, per-layer parameter counts, and the
  total. It loads and validates the model without an input file or inference.
- **Read-only metadata API:** `cinder_model_get_info` and
  `cinder_model_get_layer_info` return value copies without allocations or
  exposing model-owned pointers. Parameter counts use checked arithmetic;
  failures preserve the caller's metadata output.
- **Tests and documentation:** exact reports for the tiny and MNIST models,
  malformed-file rejection, invalid arguments, allocation/output failures,
  cleanup, recovery, and an [inspection guide](docs/INFO.md).

## Usage

```sh
./build-release/cinder info examples/mnist/mnist.cnd
```

The saved MNIST model reports **784 inputs**, **10 outputs**, **4 layers**, and
**101770 parameters**. Dense counts include weights and biases; activation
layers contribute zero parameters.

## Compatibility and limitations

- `.cnd` remains at **format version 1**. Existing models require no conversion.
- Successful `run` and `bench` output and behavior are unchanged. Shared usage
  diagnostics now list `info`; `--version` prints `Cinder 0.1.1`.
- Inspection uses the normal loader and its validation/resource limits. It
  constructs the model, including its two work buffers; it is not a streaming
  or low-memory inspector. There is no JSON or class-label output.
- Build from source on macOS or Linux. No packaged binaries or CMake install
  target are added. CPU float32 sequential Dense/ReLU/Softmax remains the
  supported execution scope.

## Verification

Local implementation verification passed **39/39 Debug tests**, **48/48 Release
tests including optional Python comparisons**, and **39/39 ASan/UBSan tests**.
Both native macOS leak checks reported **zero leaks**. Fault injection covers
all 22 inspection allocation points for each example model, output failures,
and subsequent recovery. Ordinary C tests use committed fixtures without
Python or downloads.

See [the inspection report](docs/INFO.md#verification) for the dated results and
v0.1.1 version checks. Hosted macOS/Linux CI for this candidate is pending;
previous release CI is not evidence for this revision.

---

# Cinder v0.1.0

**Released October 30, 2026:** [Cinder v0.1.0 — Initial Release](https://github.com/farukumarx64/cinder/releases/tag/v0.1.0).
The published tag points to `d4e96db`; the executable reports `Cinder 0.1.0`.

Cinder is a small C11 CPU inference runtime for sequential float32 networks.
It makes model loading, array-based layer execution, and memory ownership
explicit. Training remains outside the C runtime.

## Supported features

- **Dense, ReLU, and stable Softmax.** Dense computes `y = Wx + b` with
  row-major weights. ReLU preserves vector length; Softmax subtracts the
  maximum and uses double intermediates with float32 storage.
- **Sequential models and a public C API.** Construct or load a validated
  ordered model, query input/output sizes, run repeated inference, and free it.
  Two model-owned work buffers are allocated during construction and reused
  without Cinder allocation/free calls during inference.
- **`.cnd` version 1 export and loading.** The Python exporter accepts the
  documented PyTorch `nn.Sequential` subset and writes deterministic
  little-endian binary32 parameters. The C loader validates records,
  connections, finite parameters, resource bounds, truncation, and trailing
  data before publishing a model. Failed loads release partial construction.
- **CLI inference.** `cinder run` reads one preprocessed vector and prints its
  predicted index and up to three ranked output scores. Class indices are
  zero-based; ties favor the smaller index. Scores are probabilities only
  when the model includes a final Softmax.
- **CLI benchmarking.** `cinder bench` loads once, runs ten untimed warm-ups,
  and reports measured iteration count, clock resolution, mean/minimum/maximum
  latency, and throughput. Timing brackets the public inference call,
  including validation and copies; loading, allocation, preprocessing, and
  printing are outside the intervals.
- **Reproducible examples and tests.** A tiny `2 → 4 → 2` model, a trained
  MNIST `784 → 128 → 10` model, deterministic inputs, and stage references are
  committed. Ordinary C tests need no Python or downloads; optional Python
  tests cover regeneration, export, and the full round trip.

## Commands and setup

```text
cinder --version
cinder run <model.cnd> <input.bin>
cinder bench <model.cnd> <input.bin> --iterations 1000
```

`--version` prints exactly `Cinder 0.1.0` followed by a newline. `bench` requires
an iteration count from 1 through 1,000,000 in decimal, in the position shown.
Exit status is `0` on success, `1` for operational/output failure, and `2` for
invalid usage; diagnostics go to stderr. These are the three supported commands.

Build from source on macOS or Linux with CMake 3.16+, a C11 compiler, and Make
or Ninja. There is no CMake install target or packaged binary. The C runtime
uses the system C/math libraries. Python 3.11+ with the pinned dependencies is
optional for training/export/reference work.

See the [README](README.md) for installation, build/test commands, actual CLI
output, sanitizer setup, and architecture. The [exporter guide](docs/EXPORTER.md)
defines the supported PyTorch configurations; arbitrary graphs or checkpoint
formats are not accepted automatically.

## Ownership and input contract

The model owns its layers and work buffers; each Dense owns copies of its
weights and biases. The caller owns the returned model and frees it with
`cinder_model_free`. Successful manual model construction adopts layers;
failed construction leaves them caller-owned. A loaded model no longer depends
on its source file.

The caller owns input, destination output, and optional error storage. Inference
copies the final output only after the whole sequence succeeds, preserving
caller output on every returned failure. Repeated calls need no reset.
Accessible buffer lengths, disjoint input/output ranges, intact owners, and
object lifetime remain caller obligations. Serialize calls on a shared model
or use independent instances; there is no internal lock.

For `D` input values, `input.bin` contains exactly `4 * D` bytes: finite,
little-endian IEEE binary32 floats with no header or trailing data. The CLI
does not decode images or preprocess values. MNIST uses 784 row-major float32
pixels divided by float32 `255.0`. The format stores no class-label or
preprocessing metadata. See the [model](docs/MODEL.md), [CLI](docs/CLI.md), and
[file-format](docs/MODEL_FORMAT.md) contracts.

## Recorded results and verification

| Measurement | Recorded result |
| --- | --- |
| Python MNIST accuracy, official held-out test split | **97.48%**, 9,748/10,000 |
| C/Python accuracy on the fixed 100-image subset | **99/100 each** |
| C/Python prediction agreement on that subset | **100/100** |
| MNIST stage comparison | **27,600 values pass** shape/finite checks and `atol = rtol = 1e-5` |

Agreement includes the shared incorrect prediction; it is distinct from
classification accuracy. Full 10,000-image C accuracy has not been measured.
The [demonstration](examples/mnist/DEMO.md) and [validation report](examples/mnist/VALIDATION.md)
record the saved models, data selection, and observed numerical errors.

[M6 verification](docs/M6_VERIFICATION.md) records five passing macOS/Linux CI
jobs at `cace0df`, including Linux ASan/UBSan/LeakSanitizer, plus local tests and
seven zero-leak native macOS checks. That evidence predates this version-string
change; it does not replace final verification of the release candidate.
Historical [Release benchmarks](docs/benchmarks/2026-10-28-apple-m4/README.md)
and [evaluation](docs/M5_VERIFICATION.md) retain their exact revisions,
environments, valid trials, and failures. No speedup is claimed by this release
version update.

For this version finalization, fresh local macOS Debug and Release builds pass
**38/38 C tests each**. Both builds and a Release build with testing disabled
print exactly `Cinder 0.1.0` with empty stderr. The existing smoke test accepts
the release version and rejects the former development-version expectation.
[Local evidence](docs/verification/2026-10-30-version.txt) identifies the tested
CMake contents; candidate sanitizer/leak and hosted checks were still pending
at that step.

Subsequent [candidate verification](docs/RELEASE_VERIFICATION.md) at `1e5a5f6`
passes all local Debug/Release, numerical/lifecycle, sanitizer, native leak,
and documented setup checks, including a fresh Python environment. All five
macOS/Linux CI jobs pass on the first attempt in
[run 41029277234](https://github.com/farukumarx64/cinder/actions/runs/41029277234),
including Linux ASan/UBSan/LeakSanitizer. The subsequent released commit
`d4e96db` also passes all five jobs on its first attempt in
[run 41066635400](https://github.com/farukumarx64/cinder/actions/runs/41066635400).
The [publication record](docs/RELEASE_VERIFICATION.md#publication-record) preserves
the tag, date, full commit SHA, and CI result for the released revision.

## Constraints and known limitations

- Sequential, single-vector CPU inference only. No C training, GPU backend,
  convolutional layers, quantization, explicit SIMD kernels, batched inference,
  or internal multithreading. `cinder info` is deferred beyond v0.1.0; there is
  no `--help` command, stdin convention, or general checkpoint-import CLI.
- `.cnd` v1 allows at most 1,024 layers, 65,536 values per vector, and 64 MiB of
  parameter payload. Peak memory also includes staging/copies, work buffers,
  and allocation overhead. Model/input files must be seekable and stable
  during reading. Structural validation does not supply a checksum, signature,
  or authenticity guarantee.
- Hosts must support the required binary32 representation and wider double
  intermediates. Finite inputs can overflow during arithmetic and return an
  error. The verified platforms and toolchains are recorded in the reports;
  no native 32-bit or additional platform certification is claimed.
- Tiny benchmarks can return an error when monotonic start/end readings are
  equal. No partial timing report is published and there is no retry, rounding,
  or batching workaround. More iterations cannot resolve a zero-length
  individual interval. See the [timing investigation](docs/TIMING_INVESTIGATION.md).
- Deterministic fixtures, injected failures, and sanitizers cover exercised
  behavior; they are not exhaustive fuzzing or a proof for every input.
  Native macOS leak checks passed locally; hosted native macOS leak checking
  remains unavailable. Hosted Linux LeakSanitizer supplies the hosted leak check.

## Version and publication

The executable and CLI test builds use CMake's `project(... VERSION 0.1.0)`
directly. This removes the former `-dev` display suffix without changing the
public C API, model math, `.cnd` format version **1**, or committed fixtures.
Existing `.cnd` v1 examples remain compatible. Historical reports retain the
version strings that their original executables printed.

The owner published `v0.1.0` at commit
[`d4e96dba8decb81df3c5cf32d17d0877fae6170e`](https://github.com/farukumarx64/cinder/commit/d4e96dba8decb81df3c5cf32d17d0877fae6170e)
on October 30, 2026, at 14:54:41 Africa/Lagos (UTC+01:00). All
[release gates](ROADMAP.md#remaining-release-gates) are complete, and M7 is closed.
Cinder is distributed under the [MIT License](LICENSE).
