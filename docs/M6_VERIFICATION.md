# M6 final verification — October 30, 2026

**Phase 6 is complete, October 30, 2026.** Fresh local checks and all five
hosted CI jobs pass for the final hardening commit
`cace0dfb4898ad4f569e8ebd60df3f13089ee966`, pushed to `main` with the owner's
authorization. This report records the results after that verified revision.

This step adds the final evidence report and progress updates. It changes no
runtime source, public API, format, fixture, test, build option, or CI workflow.
All fresh local builds started from the clean committed tree before these
documentation edits. The original October 26–28 Phase 6 dates were planning
targets, not actual completion dates.

## What Phase 6 resolved

| Work | Result |
| --- | --- |
| [Release checklist audit](RELEASE_AUDIT.md) | Reconciled seven stale requirements, explicitly deferred `cinder info` beyond v0.1.0, and retained the release-documentation/publication gates. |
| [GCC warnings](GCC_WARNINGS.md) | Reproduced all four groups. Checked byte extents and explicit initialization resolved them without suppression; no reachable overflow or uninitialized read was established in the reported paths. Fresh local GCC and Clang builds pass with `-Werror`. |
| [Tiny-model timing](TIMING_INVESTIGATION.md) | Observed equal raw clock timestamps around completed inference. Preserve zero-interval failure and the per-call contract; deterministic tests cover late failure, cleanup, and recovery. No batching or retry workaround. |
| [Robustness coverage](ROBUSTNESS_REVIEW.md) | Added 162 failures/retries across parameter-decoding chunks, 2,567 parameter bit comparisons, and protected caller-buffer boundaries. No additional production defect was exposed. |

## Fresh local verification

[Local evidence](verification/2026-10-30-m6-local.txt) retains the revision,
environments, selected actual numerical/lifecycle output, and all seven native
leak reports. Full ignored build logs are in `build-m6-verification-20261030/`.

| Build | Toolchain | Result |
| --- | --- | --- |
| macOS Debug, optional Python checks enabled | Apple Clang 21.0.0 | **47/47**, 52.75 s |
| macOS Release | Apple Clang 21.0.0 | **38/38**, 22.08 s |
| macOS Debug + ASan + UBSan | Apple Clang 21.0.0 | **38/38**, 43.00 s; no findings |
| Linux Debug, C tests only | GCC 13.3.0 | **38/38**, 11.63 s |
| Linux Release, C tests only | GCC 13.3.0 | **38/38**, 5.75 s |

Every build uses C11, `-Wall -Wextra -Wpedantic -Werror` and produces **zero
compiler warnings**. Debug uses `-g`; Release uses `-O3 -DNDEBUG`, without
fast-math or LTO. Assertions and guard-page checks remain active in Release.
Suites include all twelve benchmark tests, with no speed threshold.

Local macOS is 27.0.1 (26A434), Apple M4 arm64, CMake 4.4.4. Optional checks use
Python 3.12.14, NumPy 2.3.5, and PyTorch 2.14.1 from the existing environment.
Linux is Ubuntu 24.04.5 amd64, GCC 13.3.0, CMake 3.28.3, in the existing offline
Docker image under emulation on the Mac. Source was mounted read-only and builds
used a separate output mount. These durations describe test execution, not
inference benchmarks or native Linux performance.

macOS sanitizer variables were `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`
and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Apple LeakSanitizer is
unavailable in this setup, so native leak checking is separate.

## Numerical agreement

All comparisons retain shape and finite-value checks, followed by
`abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)`. No tolerance was relaxed.
The following are actual maximum absolute errors from the fresh Release runs:

| Comparison | Values | macOS Clang | Linux GCC |
| --- | ---: | ---: | ---: |
| Tensor NumPy fixtures | 341 | `1.1920929e-7` | `1.1920929e-7` |
| Tiny export/load round trip, all four stages | 120 | `5.96046448e-8` | `5.96046448e-8` |
| MNIST first Dense | 12,800 | `6.19888306e-6` | `6.67572021e-6` |
| MNIST ReLU | 12,800 | `5.7220459e-6` | `4.76837158e-6` |
| MNIST final logits | 1,000 | `5.7220459e-6` | `5.7220459e-6` |
| MNIST probabilities | 1,000 | `3.57627869e-7` | `3.57627869e-7` |

All **400 MNIST stage vectors / 27,600 values** pass. C/Python prediction
agreement remains **100/100**, including the CLI prediction/top-three checks.
The tiny network's first Dense, ReLU, and logits are exact in these runs; its
listed error is in Softmax. Optional Python checks regenerate and validate the
export/reference paths; ordinary C tests use committed fixtures without Python
or MNIST downloads.

Runtime agreement is separate from classification accuracy. Both runtimes
classify **99/100** of the fixed subset correctly. The earlier Python full-test
accuracy remains **97.48%**; this verification does not retrain the model or
measure full-test C accuracy.

## Ownership, lifecycle, and leak checks

The full suites verify repeated calls, failed calls followed by successful
calls, poisoned working buffers, independent owners, and caller-serialized
shared-model execution. Selected actual results:

- Loader audit: **1,766 rejected calls** preserve NULL returned ownership,
  existing owners, closed streams, and zero newly retained blocks. The 162 new
  chunk-boundary failures each have a successful load/inference/free retry.
- MNIST audit: **300 cycles / 1,384 inference calls**, all 21 load allocation
  points failed twice, partial-write failures at all four layers, unchanged
  parameters, and zero final live blocks/bytes/streams.
- Loaded tiny-model lifecycle: **502 successful loads, 100 rejected loads,
  4,720 cycles / 14,160 calls**, including independent concurrent instances,
  deleted source files, and serialized shared access.
- CLI setup: both tiny and MNIST models run **100 repeated calls** and benchmark
  recovery after source deletion; each of the 23 allocation points fails with
  and without diagnostics, with complete cleanup.
- Controlled benchmark: **1,095 command invocations**; unavailable-clock
  variant: **2**. Late zero intervals stop without partial reports, then recover.

Instrumentation continues to show **zero Cinder allocation/free attempts or
loader I/O during the audited inference cycles**. This is not a claim that all
platform-library internals or the entire process allocate nothing.

Fresh `leaks --atExit` runs each exit 0 and report **0 leaks / 0 bytes** for:
loader failures; protected caller buffers; input-reader failures; CLI setup;
controlled benchmark failures; MNIST lifecycle audit; and the production
Release MNIST benchmark with 1,000 iterations. Native `leaks` emits its usual
restricted-process inspection notice but completes the analyses. Guard-page
mappings are separately checked with `munmap`; heap leak tools do not establish
mapping cleanup. The instrumented benchmark timings are not a new baseline.

## Hosted CI

[Run 40989772451](https://github.com/farukumarx64/cinder/actions/runs/40989772451)
was triggered by the authorized push of the exact tested commit `cace0df`.
All five jobs completed successfully on the first attempt, at 02:08:25 UTC
(03:08:25 Africa/Lagos). Each job's checkout log identifies the full SHA above;
its build/test output and conclusion were inspected.

| Hosted job | Compiler | Result |
| --- | --- | --- |
| [Linux Debug](https://github.com/farukumarx64/cinder/actions/runs/40989772451/job/123849076814), including optional Python checks | GCC 13.3.0 | **47/47** |
| [Linux Release](https://github.com/farukumarx64/cinder/actions/runs/40989772451/job/123849076577) | GCC 13.3.0 | **38/38** |
| [macOS Debug](https://github.com/farukumarx64/cinder/actions/runs/40989772451/job/123849077055) | Apple Clang 17.0.0 | **38/38** |
| [macOS Release](https://github.com/farukumarx64/cinder/actions/runs/40989772451/job/123849076889) | Apple Clang 17.0.0 | **38/38** |
| [Linux ASan + UBSan + LeakSanitizer](https://github.com/farukumarx64/cinder/actions/runs/40989772451/job/123849076843) | Clang 18.1.3 | **38/38** |

The workflow targets `ubuntu-24.04` and `macos-15`. Linux's sanitizer log confirms
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`; no findings occurred. All five
logs contain zero compiler warnings. Local builds additionally enforced
`-Werror`; the unchanged hosted workflow uses the normal warning flags.

Each job passed the new protected-buffer and decoder-chunk tests, numerical
comparisons, lifecycle/allocation audits, and controlled benchmark failures.
All 27,600 MNIST stage values and 100/100 prediction comparisons pass on both
platforms. Hosted numerical errors match the corresponding local platform's
figures above. Ordinary C jobs use committed fixtures; only Linux Debug installs
the optional Python dependencies, and it does not download MNIST.

[Job metadata](verification/2026-10-30-m6-ci.json) retains conclusions, steps,
revision evidence, toolchains, and test counts.
[Original log excerpts](verification/2026-10-30-m6-ci.txt) retain numerical,
lifecycle, sanitizer-environment, and robustness output. Complete downloaded
job logs are also retained in the ignored local verification directory.

## Reproduction

From the repository root, use fresh build directories:

```sh
cmake -S . -B build-m6-debug -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON -DCMAKE_C_FLAGS=-Werror
cmake --build build-m6-debug --parallel 4
ctest --test-dir build-m6-debug --output-on-failure --no-tests=error --verbose

cmake -S . -B build-m6-release -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON -DCMAKE_C_FLAGS=-Werror
cmake --build build-m6-release --parallel 4
ctest --test-dir build-m6-release --output-on-failure --no-tests=error --verbose

cmake -S . -B build-m6-sanitize -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON -DCMAKE_C_FLAGS=-Werror \
  -DCINDER_ENABLE_ASAN=ON -DCINDER_ENABLE_UBSAN=ON
cmake --build build-m6-sanitize --parallel 4
# macOS: use the separate native leak checks below.
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-m6-sanitize --output-on-failure --no-tests=error --verbose
```

Linux CI uses Clang and `detect_leaks=1` for combined ASan/UBSan/LeakSanitizer.
The offline GCC image names its compiler `/usr/bin/gcc-13`; pass
`-DCMAKE_C_COMPILER=/usr/bin/gcc-13`. To reproduce the optional Python suite,
use the existing documented Python environment and configure Debug with all
three `CINDER_ENABLE_NUMPY_TESTS`, `CINDER_ENABLE_EXPORTER_TESTS`, and
`CINDER_ENABLE_MNIST_TESTS` options `ON`, plus
`-DPython3_EXECUTABLE="$PWD/.venv/bin/python"`, then rebuild and run CTest.

On macOS, after CTest finishes so scratch files are not shared concurrently:

```sh
leaks --atExit -- ./build-m6-release/test_loader_failures \
  "$PWD/tests/fixtures/tiny/tiny.cnd" "$PWD/build-m6-release/loader-audit-files/model.bin"
leaks --atExit -- ./build-m6-release/test_buffer_boundaries
leaks --atExit -- ./build-m6-release/test_input "$PWD/build-m6-release/input-test-files/input.bin"
leaks --atExit -- ./build-m6-release/test_cli_setup "$PWD" "$PWD/build-m6-release/cli-setup-files"
leaks --atExit -- ./build-m6-release/test_benchmark_controlled "$PWD"
leaks --atExit -- ./build-m6-release/test_mnist_lifecycle_audit \
  "$PWD/examples/mnist/mnist.cnd" "$PWD/tests/fixtures/mnist" \
  "$PWD/build-m6-release/mnist-lifecycle-audit-files/model.bin"
leaks --atExit -- ./build-m6-release/cinder bench \
  examples/mnist/mnist.cnd tests/fixtures/mnist/inputs/00000.f32 --iterations 1000
```

## Remaining limitations and release handoff

- **Tiny timing:** equal native clock readings can reject a valid tiny-model
  inference. No fabricated minimum duration, retries, batching, or universal
  performance claim is introduced. A batching design needs a revised contract.
- **Memory/API preconditions:** valid pointers, accessible lengths, intact
  owners, permitted overlap, and serialized access to a shared model remain
  caller obligations. Deterministic faults and selected guard-page shapes are
  not exhaustive fuzzing or a proof against every memory/lifetime error.
- **Platforms:** this verification does not add native 32-bit or other-platform
  certification. Local Linux uses emulation. Hosted native macOS leak checking
  remains unavailable; local native checks complement hosted Linux leak checks.
- **Format:** `.cnd` v1 has structural/finite/resource validation but no checksum,
  signature, or authenticity guarantee. Supported layer types remain sequential
  Dense, ReLU, and Softmax within the documented limits.
- **Numerics and accuracy:** saved deterministic fixtures exercise numerical
  agreement. They do not establish correctness for every model or full-test
  classification accuracy in C.
- **Release scope:** `cinder info` is explicitly deferred. The version remains
  `0.1.0-dev`; the release README, version finalization, release notes, final
  fresh-checkout instructions, candidate checks, and authorized tag/publication
  belong to Phase 7.

**M6 is closed with these limitations recorded.** The next scoped work is the
release README and supported workflow/architecture overview. Later source,
build, or version changes require appropriate verification of that release
candidate; M6 evidence is tied to the revision above.
