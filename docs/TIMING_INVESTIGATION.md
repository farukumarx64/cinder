# Tiny-model timing investigation — October 29, 2026

**Decision: preserve explicit failure for zero intervals.** Native observations
now show equal `CLOCK_MONOTONIC_RAW` timestamps around completed tiny-model
inferences. Cinder correctly subtracts those readings to zero and rejects the
measurement. No arithmetic, unit-conversion, or loop-order defect was found
that justifies changing the runtime. This remains a limitation of measuring
each tiny inference separately on this machine. No batching, retry, rounding,
clock substitution, or benchmark-contract change was added.

The investigation step is complete locally. Remaining robustness review and
final M6 verification are still open.

## What the earlier failures established

The [original baseline](benchmarks/2026-10-28-apple-m4/README.md) rejected tiny
trial 1 at measured call **37,270**; the [unchanged-executable control](M5_VERIFICATION.md)
rejected call **9,853**. Both requested 100,000 measurements and published no
partial statistics. Positive tiny observations reached **41 ns**, alongside
a reported **42 ns** resolution. Neither collection retained raw timestamp
pairs, so those historical events cannot be reconstructed individually.

The timer was reviewed from clock read through publication:

1. Both readings use `CLOCK_MONOTONIC_RAW`, not its approximate variant.
2. Seconds/nanoseconds are checked for valid ranges and ordering.
3. Integer seconds/nanoseconds are subtracted, borrowing one second if needed,
   before checked conversion to `uint64_t` nanoseconds. Absolute uptime is never
   converted to floating point. For valid ordered timestamps, zero occurs only
   when both fields of the start and end are equal.
4. One real `cinder_model_run` call occurs between start and stop. Its result is
   consumed outside timing on success. Inference failure retains precedence
   over an end-clock failure.
5. A zero interval returns `CINDER_ERROR_IO`. The benchmark stops immediately
   and never publishes the partially accumulated result. The CLI releases its
   session, exits 1, and emits the existing diagnostic without a statistics report.

Existing tests already cover nanosecond borrowing, very large uptime, elapsed
overflow, backward time, clock failures, and timer recovery. The new tests
extend whole-command coverage to realistic late failure positions.

## Native observations

The [complete observation file](benchmarks/2026-10-29-timing/observations.json)
retains all **72 planned process attempts**, including every failure. The
[collector](benchmarks/2026-10-29-timing/collect.py) ran 20 trials per executable
on tiny with 100,000 requested calls, three per executable on MNIST with 1,000,
and three million-pair adjacent-clock collections. Each benchmark invocation
retained the normal ten untimed warm-ups. No attempt was replaced or retried.

| Executable | Tiny: valid / zero-interval failures | MNIST: valid / failures |
| --- | ---: | ---: |
| Original retained baseline, compiled at `cd461c6` | **1 / 19** | **3 / 0** |
| Current Release build, production code at `fb9a655` | **0 / 20** | **3 / 0** |
| Diagnostic executable using the current real loop/core | **1 / 19** | **3 / 0** |

The original executable's SHA-256 remains
`766f38969630db0335468d89658559660337ec24167ae23e8c2ecea03aed4e05`.
Both ordinary executables reject with exit 1 and empty stdout. Reproduction in
the identical original binary shows that the GCC warning fixes are not required
to trigger the problem. These short collections are not a comparison of failure
rates or speed: they differ from the earlier collections in system conditions,
and the diagnostic executable adds timestamp-recording overhead.

The [manual probe](benchmarks/2026-10-29-timing/probe.c) wraps only the native
clock read used by the real timer. It records the unchanged returned timestamp
before the timer interprets it, links the existing benchmark/core libraries,
and prints after execution stops. All **19** instrumented tiny failures captured
equal start/end pairs. For example, probe trial 2 recorded:

```text
Measured iteration: 60256
Start: seconds=1240256, nanoseconds=261973000
End:   seconds=1240256, nanoseconds=261973000
Clock reads: 120512 (exactly two per measured call)
Published result: iterations=0, total_ns=0
Output: [0.904650509, 0.0953494608]
```

The completed inference's probabilities agree with the saved reference even
though its timing is unresolved. All **70** printed probe output values across
the tiny/MNIST attempts pass shape, finite-value, and `atol = rtol = 1e-5`
checks; maximum absolute error is approximately `5.97e-8`. A rejected timing
does not mean inference failed or literally took no physical time.

The three adjacent-read collections contained **760,376**, **736,187**, and
**739,697** equal pairs out of one million each, with no backwards pairs.
Positive differences include 41 and 42 ns. These contain no inference and
are clock diagnostics, not latency estimates or overhead corrections.

The native Mach timebase query returned **125/3 ns per tick**, approximately
41.67 ns. Integer nanosecond timestamps from such ticks can differ by 41 or
42 ns, while a rounded-up resolution reports 42 ns. Apple's published
[clock implementation](https://github.com/apple-oss-distributions/Libc/blob/main/gen/clock_gettime.c)
uses Mach continuous time for the raw clock and rounds up the timebase ratio
for resolution. This is explanatory source evidence, not a claim that the
published file exactly matches the installed OS binary. The
[Linux clock manual](https://man7.org/linux/man-pages/man2/clock_gettime.2.html)
also explicitly permits identical consecutive monotonic readings.

The trace establishes the immediate cause: **the native clock did not resolve
a difference for those calls**. Quantization is consistent with the observed
timebase and timestamp increments. It does not establish each call's true
physical duration, CPU scheduling/frequency behavior, or why zeros were more
frequent in this collection than on October 28. No replacement clock was shown
to provide a more reliable measurement here.

## Environment and limits of the experiment

Collection started at **21:31:19 Africa/Lagos on October 29**, and the final
process started at 21:31:22 (UTC starts are retained in the JSON). Hardware:
Apple M4, Mac16,12, ten cores, 16 GiB memory; macOS 27.0.1 (26A434), arm64.
Apple Clang 21.0.0 and CMake 4.4.4 built the existing Release libraries with
`-O3 -DNDEBUG`, C11, `-Wall -Wextra -Wpedantic`, no sanitizers or LTO; the
current build and manual probe also used `-Werror`.

This was an active desktop, with unpinned cores and system-managed frequency.
No agent builds/tests ran during collection. Other applications were left
running; power, thermal state, and per-core frequency were not controlled.
The one-minute load average was about 1.74 before and after. The short duration
and repeated fixed inputs do not establish a general failure probability.
Binary and fixture hashes are recorded and remained unchanged.

The probe additionally compiled and ran with GCC 13.3 in the existing Ubuntu
24.04 x86-64 Docker environment. That environment is emulated on this Mac;
its observed clock values are retained only as a portability diagnostic,
not a native Linux timing baseline or evidence of independent hardware behavior.

## Added regression coverage and verification

[`test_benchmark_acceptance.c`](../tests/test_benchmark_acceptance.c) now checks:

- A controlled sequence of **41, 42, 42 ns** at reported resolution 42 ns
  retains the exact sum of 125 ns, minimum 41, maximum 42, and throughput
  24,000,000/s. It must not round the first value up or discard it.
- A zero at call **9,853**, **37,270**, or the final call **100,000** stops at
  that exact call after ten warm-ups. There are exactly two clock reads per
  measured inference, so no hidden retry, omitted call, or extra inference.
- Every unsuccessful execution leaves the caller's raw result unchanged,
  including after thousands of previously accumulated measurements.
- The CLI produces no partial stdout/statistics, preserves its diagnostic,
  releases all tracked allocations/files, and can run a fresh successful command.

These tests substitute deterministic timestamps and run the real tiny model.
CI does not have to reproduce native zero intervals or meet a speed threshold.
The controlled audit increases from 1,088 to **1,095 command invocations**;
the unavailable-clock audit remains two. No new CTest target or Python dependency
was added to ordinary tests. The probe/collector are manual investigation tools.

Review also found a test expectation that contradicted this policy:
[`test_cli_process.c`](../tests/test_cli_process.c) required every native
benchmark to exit successfully, including the smallest generated models.
Its seven normally successful benchmark cases now accept either the complete
validated report or exit 1 with empty stdout and the **exact** zero-interval
diagnostic for an iteration within the requested range. Other IO errors,
unavailable clocks, malformed/out-of-range indices, partial reports, crashes,
and unexpected exit codes still fail. No retry is performed; the test prints
how many reports and unresolved intervals occurred. Controlled tests continue
to require successful statistics for positive timestamps.

The error-message checks run even when the native clock produces no zero.
A separate [constant-clock control](benchmarks/2026-10-29-timing/zero_clock.c)
was linked into a test-only CLI executable to force the alternate process path:
all 133 process cases pass, with **0 valid benchmark reports / 7 exact zero
rejections**, while ordinary `run` output and 100/100 prediction agreement
remain correct. This control is synthetic and is not part of the native timing
collection or the shipped executable.

| Check | Result |
| --- | --- |
| macOS Release, complete ordinary C suite | **37/37** |
| macOS Debug, benchmark-labelled suite | **12/12** |
| macOS ASan + UBSan, benchmark-labelled suite | **12/12** |
| Linux GCC 13.3 Release, benchmark-labelled suite | **12/12** |
| Native macOS `leaks --atExit`, expanded controlled audit | **0 leaks / 0 bytes**, exit 0 |
| macOS Release process checks with a deliberately constant clock | **133 cases**, including **7 exact zero rejections** |

All builds retain `-Werror` and produce zero compiler warnings. Release also
passes the saved tiny round trip and all 27,600 MNIST stage comparisons with
100/100 prediction agreement. Tracked inference allocation/free counts remain
zero. macOS ASan uses `detect_leaks=0`; the separate native leak check retains
the tool's restricted-process notice. [Verification excerpts and hashes](benchmarks/2026-10-29-timing/verification.txt)
record the exact scope. All five hosted jobs passed for prior revision `fb9a655`
after a runner-capacity retry in [run 40915045986](https://github.com/farukumarx64/cinder/actions/runs/40915045986).
The new test additions have local verification; no new commit/push/CI run is
claimed for them.

## Reproduction

From the repository root on macOS, create a Release build, then build the
diagnostic against its existing libraries. The normal `cinder` executable is
not instrumented by these commands:

```sh
cmake -S . -B build-timing-release -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS=-Werror -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
cmake --build build-timing-release --parallel 4
cc -O3 -DNDEBUG -std=c11 -Wall -Wextra -Wpedantic -Werror \
    -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE \
    -DCINDER_HAVE_MONOTONIC_RAW_CLOCK=1 -Iinclude -Isrc \
    docs/benchmarks/2026-10-29-timing/probe.c src/cli_session.c src/input.c \
    build-timing-release/libcinder_cli_benchmark.a \
    build-timing-release/libcinder_core.a -lm -o build-timing-release/probe
./build-timing-release/probe clock
./build-timing-release/probe tests/fixtures/tiny/tiny.cnd \
    tests/fixtures/tiny/inputs/worked.f32 100000
ctest --test-dir build-timing-release -L benchmark --output-on-failure
```

To exercise the smoke-test error path without depending on machine speed,
compile a separate test-only CLI using the same flags and libraries, replace
`probe.c` with `zero_clock.c`, add `src/cli.c` and
`'-DCINDER_VERSION="0.1.0-dev"'`, and choose a separate output name such as
`build-timing-release/cinder-zero-clock`. Then run:

```sh
mkdir -p build-timing-release/zero-clock-process-files
./build-timing-release/test_cli_process \
    "$PWD/build-timing-release/cinder-zero-clock" "$PWD" \
    "$PWD/build-timing-release/zero-clock-process-files"
```

On Linux omit `-D_DARWIN_C_SOURCE`; older systems may also need `-lrt`, as the
normal CMake probe determines. For the full fixed collection, wait until all
builds/tests finish, then run the retained collector with the three executable
paths and a new output directory. The exact paths used here were:

```sh
python3 docs/benchmarks/2026-10-29-timing/collect.py \
    build-baseline-release-20261028/cinder \
    build-gcc-audit-20261029/mac-Release/cinder \
    build-timing-audit-20261029/probe build-timing-audit-20261029/collection
```

Use a new destination if that collection directory already exists. A fresh
checkout needs an original-revision executable built separately for the first
argument; byte identity with the retained historical binary is not promised.
New collection results must be kept independently of the saved observations.

## What a future batching proposal must specify

Measuring several calls between two clock reads would change what a sample
means. Before implementing that, define batch size/iteration semantics, warm-up
and tail-batch handling, where output consumption and loop work fall relative
to timing, and how inference/timing failures stop the batch. Define throughput
from the actual completed count and measured total. A batch's duration divided
by its call count is an average; it cannot supply the existing minimum and
maximum individual-call latency. Such a mode needs distinct reporting and tests,
and must retain explicit handling when even a batch interval is unresolved.

For the current command, increasing `--iterations` does not lengthen any one
measured interval. The accepted limitation is that tiny-model benchmarks can
exit 1 on this host. Ordinary `cinder run` inference is unaffected. The next
scoped step is the remaining robustness-coverage review, followed by M6's final
verification gate.
