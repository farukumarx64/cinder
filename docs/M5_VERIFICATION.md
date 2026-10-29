# M5 performance evaluation and closure — October 29, 2026

**Phase 5 is complete. No runtime optimization is justified by the measurements
collected so far.** The benchmark command, its failure handling, a recorded
Release baseline, numerical/lifecycle checks, and all five macOS/Linux CI jobs
are verified. The known tiny-model clock limitation remains explicit.

The baseline, repeat measurements, and local checks below ran on October 28;
this closure was documented on October 29, 2026 (Africa/Lagos). The original
October 23–25 Phase 5 dates were planning targets, not actual completion dates.

This step adds this evaluation, captured evidence, a repeat-collection script,
and progress updates. It changes no C source, public API, format, fixture,
build option, test, or CI workflow. In particular, it neither relaxes numerical
checks nor changes the per-call measurement contract to manufacture a speedup.

## Decision and alternatives considered

| Candidate | Evidence and decision |
| --- | --- |
| Remove inference allocations | Existing reusable workspaces already eliminate allocation/free attempts in the audited inference paths. There is no remaining per-call allocation to remove in those paths. |
| Rewrite or fuse Dense/activation operations | End-to-end timings do not attribute cost to individual functions. No profile or measured candidate demonstrates a benefit. Keep the tested implementation. |
| Remove validation, reorder arithmetic, or enable fast-math | These can change failure behavior or floating-point results. No measured benefit justifies that tradeoff; preserve the contracts and `atol = rtol = 1e-5`. |
| Add SIMD, threads, or new compiler tuning | No candidate comparison supports the additional complexity; SIMD and threading are also outside v0.1.0 scope. The baseline remains ordinary `-O3 -DNDEBUG`, without LTO or fast-math. |
| Retry, round up, or batch zero-length tiny timings | This would alter which calls are retained or what a sample measures. Preserve the explicit zero-interval failure and document the limitation. Any future batching design needs its own measurement contract. |

A no-change control is more informative here than an unmotivated code edit:
repeat the same executable and inputs to see how much the reported numbers
move without any optimization. This does not prove the runtime is optimal;
future work should begin with a reliable workload and profiling, then test one
specific candidate against the unchanged implementation.

## Original baseline versus unchanged-executable control

Both collections used Apple M4 / Mac16,12, arm64, 10 cores (4 Performance +
6 Efficiency), 16 GiB RAM, macOS 27.0.1 (26A434), Apple Clang 21.0.0, CMake 4.4.4,
and SDK 27.0. The effective flags were
`-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wpedantic`, with sanitizers and LTO off.
The [original report](benchmarks/2026-10-28-apple-m4/README.md) retains the full
build recipe and settings.

- Compiled source / hosted CI revision:
  `cd461c696aea932e6ea53e4093008d6463c42fa9`.
- Clean checkout during the control:
  `473aab15750ab436de18b9dd9f33a350f71c9e49`, which adds only the original
  baseline documentation. Source, tests, fixtures, CMake, and CI are unchanged.
- Both collections used the **identical 53,384-byte executable**, SHA-256
  `766f38969630db0335468d89658559660337ec24167ae23e8c2ecea03aed4e05`.
- The model/input SHA-256 identities also match. Tiny is `2 → 4 → 2`, input
  `[2, -1]`; MNIST is `784 → 128 → 10`, official test index 0, label 7.
- Five attempts per model, ten untimed warm-ups, 100,000 requested measured
  calls per attempt, one process per attempt; alternate tiny/MNIST and pause
  one second after each. No retry, replacement, or outlier trimming.
- `CLOCK_MONOTONIC_RAW`; all successful reports show 42 ns resolution.
  Loading, setup allocations, preprocessing, output observation, statistics,
  cleanup, and printing stay outside the inference intervals.

The comparison is **before/repeat with no code change**, not an optimized
before/after experiment. Each median below is the median of valid **trial
means**, not a median of individual inference samples or a pooled rate.

| Model | Original valid attempts | Repeat valid attempts | Original median trial mean (us) | Repeat median trial mean (us) | Observed change |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 4/5 | 4/5 | 0.159771025 | 0.15861274 | −0.725% |
| MNIST | 5/5 | 5/5 | 105.936159 | 104.596974 | −1.264% |

The repeat's tiny trial means range from 0.15011882 to 0.1671059 us;
MNIST ranges from 103.010277 to 105.258955 us. Median trial throughput is
6,308,846.1/s for valid tiny attempts and 9,560.50604/s for MNIST.
Throughput within each trial remains `iterations / measured inference seconds`;
cross-trial median throughput is only a descriptive summary.

The same executable became apparently faster in both comparisons. That is
observed run-to-run variation, not evidence of an implementation improvement.
The two small collections do not establish a confidence interval or a universal
noise threshold. MNIST's original/repeat ranges overlap; tiny is especially
limited by clock granularity and rejected attempts.

### Every repeat attempt

Latency is in microseconds; throughput is inferences per second.

| Model | Trial | Mean | Minimum | Maximum | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1 | **Failed: zero interval at iteration 9,853** | — | — | — |
| tiny | 2 | 0.15452888 | 0.083 | 25.833 | 6471282.26 |
| tiny | 3 | 0.15011882 | 0.083 | 25.833 | 6661389.96 |
| tiny | 4 | 0.1626966 | 0.083 | 4.167 | 6146409.94 |
| tiny | 5 | 0.1671059 | 0.083 | 30.084 | 5984229.16 |
| MNIST | 1 | 103.010277 | 97.125 | 293.916 | 9707.76924 |
| MNIST | 2 | 104.342408 | 97.125 | 559.833 | 9583.83098 |
| MNIST | 3 | 104.596974 | 97.125 | 314.166 | 9560.50604 |
| MNIST | 4 | 105.017964 | 97.167 | 313.75 | 9522.18044 |
| MNIST | 5 | 105.258955 | 97.166 | 372.459 | 9500.37928 |

The failed command exited 1 with empty stdout and the existing diagnostic:

```text
cinder: benchmark iteration 9853 [IO]: Measured interval is zero; clock resolution is insufficient.
```

The original failure at iteration 37,270 also remains in its report. Both
collections retain all ten attempts and publish no partial timing statistics
for a failed command. Their four valid tiny trials each meet the original
three-valid-trial reporting minimum; this does not make the clock limitation
disappear. Increasing the iteration count cannot repair a zero-length sample.
The cause of these intervals has not been established.

Repeat collection ran 22:28:15–22:29:16 Africa/Lagos on October 28
(21:28:15–21:29:16 UTC), on an active desktop. Builds, tests, and leak checks
finished before collection; other desktop applications remained open. Cores
were not pinned and frequency was system-managed. One-minute load averages
were 1.52 before and 1.45 afterward. `pmset` reported AC Power and Low Power
Mode 0, alongside a discharging battery at 56% → 55%; this mixed report does
not establish charging. Thermal/performance status was unavailable, so no
claim about absence of throttling is made. Inspected instrumentation environment
variables were absent. Per-trial load and original command output are retained.

Evidence:

- [control.json](benchmarks/2026-10-28-phase5-evaluation/control.json): source,
  executable/fixture identities, settings, environment, all attempts, summaries.
- [raw-trials.txt](benchmarks/2026-10-28-phase5-evaluation/raw-trials.txt): exact
  stdout/stderr, exit statuses, commands, and UTC timestamps for all ten attempts.
- [collect_control.py](benchmarks/2026-10-28-phase5-evaluation/collect_control.py):
  the script used for this collection; standard-library Python only. It checks
  a clean checkout, unchanged sources and fixture hashes, and an executable
  hash identical to the original baseline; it retains failed attempts.

To repeat this exact control with the retained baseline executable, use a clean
checkout containing these documents and a new ignored destination directory:

```sh
python3 docs/benchmarks/2026-10-28-phase5-evaluation/collect_control.py \
  build-baseline-release-20261028 \
  build-baseline-release-20261028/m5-repeat
```

It refuses to overwrite prior evidence or accept a different executable hash.
A future build on another path/toolchain need not reproduce the binary hash;
use the original report's build/collection recipe for a separately identified
baseline instead. Do not run builds, tests, or leak checks during collection.

## Numerical and lifecycle verification

The full suites were rerun before timing. Shape checks, finite-value checks,
and `abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)` all remain active.
The fresh Release results match the original baseline's errors:

| Comparison | Values checked | Maximum absolute error |
| --- | ---: | ---: |
| Tensor NumPy references | 341 | `1.1920929e-7` |
| Tiny export/load round trip, all four stages | 120 | `5.96046448e-8` (Softmax); earlier stages exact |
| MNIST Dense1 | 12,800 | `6.19888306e-6` |
| MNIST ReLU | 12,800 | `5.7220459e-6` |
| MNIST logits | 1,000 | `5.7220459e-6` |
| MNIST probabilities | 1,000 | `3.57627869e-7` |

All 400 MNIST stage vectors / 27,600 values passed. Prediction agreement with
saved Python remained **100/100**, and CLI prediction/top-three indices also
agreed on all 100 inputs. Classification remains a separate measurement:
C and Python each classify 99/100 fixed-subset images correctly; the previously
measured Python full-test accuracy is 97.48%, not a new full-test C result.
Debug's optional Python suite also regenerates and validates the reference path.
Ordinary C tests continue to use committed fixtures without Python or downloads.

The rerun includes existing instrumentation, rather than adding it to the
performance executable:

- `cli_setup` reuses each real tiny/MNIST model/input/output for 100 calls and
  benchmark success/failure/recovery after deleting source files. Allocation,
  free, and file-operation counters stay unchanged during execution; all 23
  setup allocation points are forced to fail with and without diagnostics.
- `benchmark_controlled` passes 1,088 real CLI invocations; the unavailable-clock
  variant passes two. Controlled timestamps verify statistics and exclusion of
  warm-up/setup/observation. Fault injection covers timing, inference, output,
  allocation, and file failures, with complete cleanup. Allocation/free attempts,
  including `free(NULL)`, are forbidden during execution.
- `mnist_lifecycle_audit` passes 300 cycles / 1,384 inference calls: poisoned
  workspaces, partial writes at all four layers, failure recovery, preserved
  parameters, all 21 load allocation points failing twice, and zero final live
  blocks/bytes/streams. Inference has zero allocation/free attempts or loader I/O.
- The loaded-model lifecycle suite passes 502 successful and 100 rejected loads,
  4,720 cycles / 14,160 calls, independent owners and caller-serialized threads.

These establish zero Cinder allocation/free attempts in the exercised inference
paths, not zero allocations in every platform-library internal or entire process.

## Local checks and supported leak detection

[local-verification.txt](benchmarks/2026-10-28-phase5-evaluation/local-verification.txt)
preserves suite summaries, selected actual numerical/lifecycle output for each
build, and all four native leak reports. Absolute checkout paths are normalized
to `<repo>` in that file.

| Check on Apple M4 / Apple Clang 21.0.0 | Actual result on October 28 |
| --- | --- |
| Debug, optional NumPy/exporter/offline MNIST tests enabled | **46/46**, 27.97 s |
| Release, ordinary C tests only | **37/37**, 3.18 s |
| Debug, ASan + UBSan, ordinary C tests | **37/37**, 21.33 s; no sanitizer findings |
| Native `leaks`: controlled benchmark audit | Exit 0; **0 leaks / 0 bytes** |
| Native `leaks`: unavailable-clock audit | Exit 0; **0 leaks / 0 bytes** |
| Native `leaks`: CLI setup/failure/recovery audit | Exit 0; **0 leaks / 0 bytes** |
| Native `leaks`: production Release MNIST `bench`, 1,000 iterations | Exit 0; **0 leaks / 0 bytes** |

Each full suite includes the twelve `benchmark`-label tests, with no machine
speed requirement. The existing build directories were rebuilt, then tested:

```sh
cmake --build build-mnist --parallel 4
ctest --test-dir build-mnist --output-on-failure
cmake --build build-baseline-release-20261028 --parallel 4
ctest --test-dir build-baseline-release-20261028 --output-on-failure
cmake --build build-mnist-sanitize --parallel 4
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build-mnist-sanitize --output-on-failure

leaks --atExit -- ./build-mnist/test_benchmark_controlled "$PWD"
leaks --atExit -- ./build-mnist/test_benchmark_unavailable "$PWD"
leaks --atExit -- ./build-mnist/test_cli_setup "$PWD" "$PWD/build-mnist/cli-setup-files"
leaks --atExit -- ./build-baseline-release-20261028/cinder bench \
  examples/mnist/mnist.cnd tests/fixtures/mnist/inputs/00000.f32 --iterations 1000
```

ASan leak detection is disabled for these macOS sanitizer runs; native `leaks`
is the separate supported local check. It reported its usual restricted-process
inspection notice, completed successfully, and reported zero leaked bytes. The
production run under `leaks` verifies cleanup only; its timings are not a baseline.
Do not run the setup audit concurrently with CTest using the same scratch path.

## Hosted CI evidence and closure

[Run 40688919176](https://github.com/farukumarx64/cinder/actions/runs/40688919176)
completed successfully on October 28 for the exact compiled-source revision
`cd461c696aea932e6ea53e4093008d6463c42fa9`. All five job statuses, checked-out
commits, toolchains, test summaries, numerical/lifecycle output, and sanitizer
environment were inspected. This is an existing published run, not a newly
triggered run for the evaluation documents.

| Job | Toolchain | Tests |
| --- | --- | ---: |
| [Linux Debug](https://github.com/farukumarx64/cinder/actions/runs/40688919176/job/122818396001), including optional Python checks | GCC 13.3.0 | **46/46** |
| [Linux Release](https://github.com/farukumarx64/cinder/actions/runs/40688919176/job/122818396032) | GCC 13.3.0 | **37/37** |
| [macOS Debug](https://github.com/farukumarx64/cinder/actions/runs/40688919176/job/122818395961) | Apple Clang 17.0.0 | **37/37** |
| [macOS Release](https://github.com/farukumarx64/cinder/actions/runs/40688919176/job/122818396025) | Apple Clang 17.0.0 | **37/37** |
| [Linux ASan + UBSan + LeakSanitizer](https://github.com/farukumarx64/cinder/actions/runs/40688919176/job/122818395574) | Clang 18.1.3 | **37/37** |

The sanitizer job uses `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`; no findings occurred.
The workflow targets `ubuntu-24.04` and `macos-15`. Each job includes all twelve
benchmark tests and numerical/lifecycle regressions. Ordinary C jobs do not
install Python or download MNIST; Linux Debug separately installs the optional
Python dependencies and runs offline fixtures/script tests.

[ci-jobs.json](benchmarks/2026-10-28-phase5-evaluation/ci-jobs.json) preserves job
IDs/statuses/steps; [ci-excerpts.txt](benchmarks/2026-10-28-phase5-evaluation/ci-excerpts.txt)
preserves selected original log lines, including warnings. Revision `473aab1`
adds only baseline documentation to `cd461c6`; the current evaluation likewise
changes no runtime/test/build/workflow files. Thus the published run verifies
the implementation being evaluated. No commit, push, or new CI dispatch was
needed for this closure.

## Limitations and Phase 6 handoff

- Tiny per-call measurements can fail at the clock's granularity; both recorded
  failures remain evidence. Do not claim precise tiny-model speedups until
  timing reliability is investigated. No unverified retry or batching workaround
  was introduced.
- This is one desktop, one repeated input per model, and two small collections.
  No cold-cache, varied-input, energy, percentile, or Linux performance baseline
  is claimed. Scheduler/power/thermal effects were not controlled.
- Hosted native macOS leak checking was previously removed after it did not
  complete. Local native checks and hosted Linux LeakSanitizer passed; do not
  describe this as a passing hosted macOS native leak test.
- Linux GCC logs still contain `-Wtype-limits` warnings at loader size guards
  and Release `-Wmaybe-uninitialized` warnings in source-including test paths
  (tensor output, benchmark result, and model metadata). They are preserved,
  not suppressed or treated as a warning-free build. Passing tests do not by
  themselves settle each warning; triage them during Phase 6.

All required Phase 5 correctness and verification gates pass with these
limitations recorded. **M5 is closed October 29, 2026.** Phase 6 is the final
hardening pass: reconcile the remaining robustness checklist, investigate the
compiler warnings and timing limitation, and address demonstrated defects one
scoped step at a time. It is not yet marked complete.

Subsequent Phase 6 work: the [October 29 GCC investigation](GCC_WARNINGS.md)
reproduces and resolves the warning groups locally, with checked byte extents,
explicit initialization, boundary coverage, and warning-free GCC/Clang builds.
Its evidence is separate from the historical M5 results above; at that point,
new hosted CI and the tiny-model timing investigation remained pending.

Subsequent timing follow-up: the [October 29 investigation](TIMING_INVESTIGATION.md)
captures equal native timestamp pairs around completed tiny inferences, retains
explicit rejection without a runtime change, and adds deterministic late-failure
coverage. This establishes the immediate cause of newly observed failures;
the historical trials above did not record their raw pairs and remain unchanged.
