# GCC warning investigation — October 29, 2026

**All four warning groups reproduced and resolved locally.** GCC 13.3 produced
10 diagnostics in Debug and 20 in Release before the changes. Fresh Debug and
Release builds now produce zero warnings with `-Werror`, and each passes all
37 ordinary C tests. No reachable uninitialized read or allocation overflow
was found in the reported paths. The changes clarify initialization and checked
size conversion while retaining validation, failure behavior, and optimization.

This completed the warning-investigation step. The later
[M6 verification](M6_VERIFICATION.md) closes the phase at `cace0df` with fresh
warning-free local and hosted builds. Earlier hosted
verification of `fb9a655` passed all five jobs in
[run 40915045986](https://github.com/farukumarx64/cinder/actions/runs/40915045986),
attempt 2. The first attempt's macOS jobs were cancelled before checkout because
GitHub could not allocate runners; all Linux jobs passed. The earlier
[M5 CI results](M5_VERIFICATION.md#hosted-ci-evidence-and-closure) describe the
earlier code and remain separate evidence.

## Reproduction and provenance

The before build used a clean `git archive` of
`27bd04cedcb6a541a622f2c1a302151080d54f19`. Its code and tests are unchanged from
the warning-producing hosted revision `cd461c6`. The after build used the
working tree with five source/test files changed. Their SHA-256 hashes,
original diagnostics, isolated compiler probes, test excerpts, and full native
leak reports are retained in [the evidence file](verification/2026-10-29-gcc-warnings.txt).

Linux reproduction used Docker with Ubuntu 24.04.5, GCC
`13.3.0-6ubuntu2~24.04.1`, CMake 3.28.3, and glibc development package
`2.39-0ubuntu8.9`. The container was `linux/amd64` running through emulation
on an Apple M4 Mac. Source was mounted read-only, builds had a separate output
mount, and verification containers had networking disabled. This reproduces
the GCC version and x86-64 environment; it is not another GitHub Actions run
or a Linux performance baseline.

Both original configurations retained C11, `-Wall -Wextra -Wpedantic`;
Release retained `-O3 -DNDEBUG`. Each loader size warning appears five times
because the loader is compiled both normally and inside fault-injection tests.
The Release build adds four metadata, three output-buffer, and three benchmark
result diagnostics. These are repeated diagnostics, not 20 distinct defects.

## Cause and disposition of each warning

### Loader size comparisons: valid guards with an unreachable branch on this host

The original comparisons in `cnd_dimension` and `cinder_model_load` promoted
32-bit encoded counts to `uint64_t` and compared them with
`SIZE_MAX / sizeof(element)`. A cast changes the type, not the maximum value
the original count can have. On this 64-bit host neither count can exceed the
native allocation limit. GCC correctly reports `-Wtype-limits` for an
always-false comparison; this does not indicate an allocation that can wrap.

[`src/loader.c`](../src/loader.c) now has one private `cnd_byte_extent` helper
for vector, layer-list, and Dense parameter-staging sizes. It accepts a
`uint64_t` count, which also accommodates derived Dense parameter counts,
checks `count <= SIZE_MAX / element_size`, and only then narrows and multiplies
to produce a `size_t` byte count. A zero element width is rejected before
division. Failure leaves the byte destination unchanged.

Layer-list and parameter allocations use that returned byte count directly.
The vector check uses the same conversion without allocating. Format limits,
validation order, overflow status/messages, and cleanup remain unchanged.
This consolidates the actual allocation-size calculation instead of removing
the protection or disguising a comparison with a cast.

### Model metadata snapshots: GCC could not connect two conditions

[`tests/test_model.c`](../tests/test_model.c) snapshots layer metadata, rejects
an invalid model, then verifies that neither the layer-pointer list nor the
layers changed. Previously only non-NULL slots received a snapshot. A snapshot
was read only for a non-NULL slot after the pointer-list equality check passed.
The check exits the process on failure, including in Release builds, so an
unchanged NULL slot could not reach a read of its uninitialized snapshot.

GCC's optimized analysis did not establish that relationship and warned about
four fields of `metadata[0]`. Each supplied slot now receives a defined
snapshot: a copy for a real layer, or an invalid zero-valued structure for a
NULL slot. All pointer equality and metadata-preservation assertions remain.
This makes the test's state explicit without weakening ownership checks.

### Ownership-failure output: an inlining warning attributed to ReLU

The trace through `tensor.c` came from
[`tests/test_model_alloc_failures.c`](../tests/test_model_alloc_failures.c).
After a failed model construction, the test executes its still-owned first
Dense layer into a local four-float destination. Dense writes the destination
before the test reads it. Although GCC's inlining trace points at ReLU,
dispatch selects Dense here. The general activation path also copies a
separate input to output before calling an in-place activation.

The test now starts all four destination values at `NAN`. A successful Dense
call must overwrite every poison value with the expected finite results
`[0.5, -3, 3.5, 5]`. Missing or partial output writes therefore fail explicitly.
The tensor kernels and dispatch are unchanged. This initialization is a test
oracle, not a new requirement on callers: the existing `model_run` tests also
continue using an uninitialized `exact_output` destination.

### Benchmark result: failure paths do not publish measurements

Compiling the original benchmark acceptance translation unit separately for
both clock backends isolates the three result warnings to the unavailable-clock
variant. The controlled-clock variant emits only the loader warnings.
`cinder_benchmark_execute` returns failure when clock setup is unavailable;
it assigns the complete result only immediately before returning `CINDER_OK`.
The CLI calls statistics and prints the result only after success. The test's
tracking wrapper preserves the returned status. Thus this path cannot consume
an unpublished result, although GCC's analysis warned that it might.

[`src/cli.c`](../src/cli.c) now initializes its local result to `{0}` before
execution. Zero iterations and durations are invalid measurements and are
already rejected by the statistics function. Initialization defines the local
state; it does not manufacture a valid fallback report. Status checks, error
precedence, output suppression on failure, and the execute function's
leave-result-unchanged-on-failure contract are preserved. Initialization occurs
before any measured inference interval.

GCC documents that `-Wmaybe-uninitialized` depends on optimization and can
warn when it cannot prove an uninitialized path unreachable; see the
[GCC 13.3 warning documentation](https://gcc.gnu.org/onlinedocs/gcc-13.3.0/gcc/Warning-Options.html).
The classifications above also rely on inspection of these specific control
flows and their tests, not just on the fact that a compiler can warn incorrectly.

## Regression coverage and actual results

The existing `loader_failures` test now exercises byte extents at zero, one,
the exact representable count, one past that count, and `UINT64_MAX` where it
exceeds the limit, for byte/float/layer-pointer widths. It also rejects a zero
width and verifies unchanged output on failure and zero allocation attempts.
These tests calculate sizes without attempting huge allocations. Existing
lowered-native-ceiling tests still exercise all three loader overflow sites,
including cleanup and retry with and without diagnostics.

All five fresh builds used `-Werror` in addition to the existing warning flags.
No warning was disabled, and no optimization level was reduced.

| Configuration | Build warnings | CTest result |
| --- | ---: | ---: |
| Linux GCC 13.3 Debug | 0 | **37/37** |
| Linux GCC 13.3 Release (`-O3`) | 0 | **37/37** |
| macOS Apple Clang 21 Debug, optional Python checks enabled | 0 | **46/46** |
| macOS Apple Clang 21 Release | 0 | **37/37** |
| macOS Apple Clang 21 Debug, ASan + UBSan | 0 | **37/37** |

The host is macOS 27.0.1 (26A434), arm64, CMake 4.4.4. Optional Debug checks
used the existing Python 3.12.14 environment with NumPy 2.3.5 and PyTorch
2.14.1. Ordinary C tests used committed fixtures without Python or downloads.
The controlled benchmark's 1,088 invocations, two unavailable-clock invocations,
133 CLI process cases, and model ownership/allocation checks pass. The loader
audit still passes all 21 forced allocation failures, 29 I/O-failure scenarios,
1,604 rejected calls, and the poisoned-buffer lifecycle/recovery checks.

Both platforms pass all 27,600 saved MNIST stage values with shape/finite
checks and `atol = rtol = 1e-5`; prediction agreement remains **100/100**.
Maximum absolute errors differ slightly across the two toolchains:

| MNIST stage | macOS (all three builds) | Linux (both builds) |
| --- | ---: | ---: |
| First Dense | `6.19888306e-6` | `6.67572021e-6` |
| ReLU | `5.7220459e-6` | `4.76837158e-6` |
| Logits | `5.7220459e-6` | `5.7220459e-6` |
| Probabilities | `3.57627869e-7` | `3.57627869e-7` |

These compare runtime outputs with saved Python references, not classification
accuracy. The saved subset remains 99/100 correctly classified by each runtime.
The existing allocation audits still report no inference allocation/free attempts.

Three focused native macOS `leaks --atExit` checks also pass: loader rejection/
cleanup, controlled benchmark acceptance, and unavailable-clock acceptance.
Each exits 0 and reports **0 leaks / 0 bytes**. The tool prints its existing
restricted-process memory-inspection notice, retained in the evidence. ASan
ran with `detect_leaks=0`; native checks provide separate leak evidence.

## Repeating the verification

In Ubuntu 24.04 with GCC 13 and CMake installed, run from the repository root:

```sh
for mode in Debug Release; do
    cmake -S . -B "build-gcc-$mode" -DCMAKE_BUILD_TYPE="$mode" \
        -DCMAKE_C_COMPILER=gcc-13 -DCMAKE_C_FLAGS=-Werror
    cmake --build "build-gcc-$mode" --parallel 4
    ctest --test-dir "build-gcc-$mode" --output-on-failure --no-tests=error
done
```

For the original warnings, build a checkout of `27bd04c` with the same commands
but omit `-DCMAKE_C_FLAGS=-Werror`. The local reproduction used a task-specific
Docker image built from `ubuntu:24.04` with `gcc-13 libc6-dev cmake make
ca-certificates` installed. Exact versions and commands for the two isolated
clock-backend probes are preserved in the evidence file. Full local build/test
logs and scripts remain in ignored `build-gcc-audit-20261029/`.

For native macOS checks, use the existing README build commands with
`-DCMAKE_C_FLAGS=-Werror`, enabling all three Python options for Debug and both
sanitizer options for the sanitizer build. After CTest completes, the focused
leak commands for the build directory used here are:

```sh
leaks --atExit -- ./build-gcc-audit-20261029/mac-Debug/test_loader_failures \
    "$PWD/tests/fixtures/tiny/tiny.cnd" \
    "$PWD/build-gcc-audit-20261029/mac-Debug/loader-audit-files/model.bin"
leaks --atExit -- ./build-gcc-audit-20261029/mac-Debug/test_benchmark_controlled "$PWD"
leaks --atExit -- ./build-gcc-audit-20261029/mac-Debug/test_benchmark_unavailable "$PWD"
```

## Limits and next step

This verifies GCC 13.3 x86-64 and Apple Clang 21 arm64; it does not promise that
every compiler/version emits no warnings. No native 32-bit build was run;
lowered test ceilings exercise guards but do not emulate a complete 32-bit ABI.
At initial local verification, hosted CI and Linux sanitizer/leak execution
were pending. The subsequent run above passed Linux Debug 46/46, Linux Release
and ASan/UBSan/leaks 37/37 each, and macOS Debug/Release 37/37 each. Later edits
are separately verified in the [final M6 report](M6_VERIFICATION.md).

The next scoped step at this point was the saved tiny-model zero-interval timing
failures. The subsequent [timing report](TIMING_INVESTIGATION.md) records that
investigation; robustness review and final M6 verification subsequently passed,
closing the phase on October 30. Phase 7 release documentation is next.
