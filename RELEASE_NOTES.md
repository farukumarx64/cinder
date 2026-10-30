# Cinder v0.1.0

**Prepared for release; final candidate verification and publication are
pending.** The executable now reports `Cinder 0.1.0`. This document does not
announce a published tag or release.

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
CMake contents; final candidate sanitizer/leak and hosted checks remain pending.

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

Final fresh-checkout tests and macOS/Linux CI on the release candidate remain
required before the owner authorizes its tag and publication. The
[release gates](ROADMAP.md#remaining-release-gates) track those remaining steps.
Cinder is distributed under the [MIT License](LICENSE).
