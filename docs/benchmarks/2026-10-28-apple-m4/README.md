# Release baseline — Apple M4, October 28, 2026

This is the first recorded performance baseline for the existing inference
implementation. It measures the public inference API on one repeated input per
model, using the [benchmark contract](../../BENCHMARK.md). It is not a speedup
claim or an end-to-end application throughput measurement.

**Five attempts per model were retained: four tiny trials and all five MNIST
trials produced valid reports.** Tiny trial 1 failed at measured iteration
37,270 because its elapsed interval was zero. No partial statistics were
published, that attempt was not replaced, and no clock error was bypassed.
The remaining four tiny results satisfy the three-valid-trial reporting minimum,
but their variation and the timing failure limit conclusions about this model.

## Provenance and environment

| Item | Recorded value |
| --- | --- |
| Source commit | `cd461c696aea932e6ea53e4093008d6463c42fa9` |
| Source state | Clean before and after all trials; benchmark binary and fixture hashes unchanged |
| CPU | Apple M4, arm64; 10 physical/logical cores: 4 Performance + 6 Efficiency |
| Machine / memory | Mac16,12; 17,179,869,184 bytes (16 GiB) |
| OS | macOS 27.0.1, build 26A434 |
| Compiler | Apple Clang 21.0.0 (`clang-2100.3.34.2`), `/usr/bin/cc` |
| Compiler target | `arm64-apple-darwin27.0.0` |
| CMake / generator | 4.4.4 / Unix Makefiles |
| Selected SDK version | 27.0; CMake SDK, deployment-target, and architecture overrides unset |
| Build directory | `build-baseline-release-20261028` (fresh, out of source) |
| Effective C flags | `-O3 -DNDEBUG -std=c11 -Wall -Wextra -Wpedantic` |
| Sanitizers / LTO | ASan OFF, UBSan OFF, interprocedural optimization OFF |
| Other optimization settings | No explicit fast-math, native-architecture, or architecture-tuning flags |
| Testing options | BUILD_TESTING ON; optional NumPy/exporter/MNIST Python tests OFF |
| Version string | `0.1.0-dev` |
| Clock | `CLOCK_MONOTONIC_RAW`; successful reports show 42 ns resolution |
| Concurrency | One runtime thread, no core pinning, system-managed CPU frequency |
| Per attempt | 10 untimed warm-ups, then 100,000 requested measured iterations |
| Ordering | Tiny then MNIST, repeated five times; a fresh process/model/session per attempt |
| Trial spacing | One-second pause after each collected attempt, except the documented interruption after the first failure |
| Recording window | 19:50:14–19:53:03 Africa/Lagos (18:50:14–18:53:03 UTC) |

The initial collector stopped on tiny trial 1's nonzero exit. Collection resumed
with MNIST trial 1 at 19:52:02, preserving that failed attempt and the original
remaining trial order. All ten attempts and timestamps are in
[raw-trials.txt](raw-trials.txt); [baseline.json](baseline.json) preserves parsed
metrics, per-trial load averages, environment output, build options, effective
compile/link commands, executable identity, and source-tree identity. `<repo>`
in the recorded compile commands replaces the absolute checkout path.

This was an active desktop, with other applications left running. Builds and
tests finished before the timed collection; no agent build/test workload ran
concurrently with it. One-minute load averages were 4.47 before collection and
2.52 afterward. A separate pre-trial `top` snapshot at 19:48:02 reported 83.7%
CPU idle; this is not utilization measured throughout the trials.

`pmset` reported AC Power and Low Power Mode 0 at both ends, but its battery
line simultaneously reported discharging, from 81% to 80%. Preserve that mixed
report rather than claiming a verified charging state. Thermal/performance
status queries returned unavailable-data messages, so absence of thermal
throttling was not established. No power settings were changed. The inspected
DYLD/malloc-debugging/sanitizer environment variables were absent.

## Models and input identities

Both sequences execute Dense → ReLU → Dense → Softmax. Tiny is `2 → 4 → 2`
and uses `[2, -1]`; MNIST is `784 → 128 → 10` and uses official test index 0,
label 7. Inputs are the committed preprocessed float32 vectors. Image decoding
and preprocessing do not occur inside this command.

| Artifact | Repository path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| tiny model | `tests/fixtures/tiny/tiny.cnd` | 168 | `3f6f2942084b334e2c9e8b59a1b0e702057554f4253473ae86ffd7cde7d86fa5` |
| tiny input | `tests/fixtures/tiny/inputs/worked.f32` | 8 | `29f16f743de29d3d29fb51bb15c62b3592bdfcf27c18836a90e16e2b1063b01b` |
| mnist model | `examples/mnist/mnist.cnd` | 407,160 | `b0376dd2562f0a89d95cc4f9f1172005e293d0ab9c4a0c9c345fff2dd98fb8bd` |
| mnist input | `tests/fixtures/mnist/inputs/00000.f32` | 3,136 | `25c7cba0509b7d1e068a5793c8bbd5cd421f4b92792aae3484e3a8dbb218e73f` |

The measured executable is 53,384 bytes, SHA-256
`766f38969630db0335468d89658559660337ec24167ae23e8c2ecea03aed4e05`.
This identifies the binary used here; it does not promise byte-identical future
builds across paths, linker metadata, or toolchains.

## All results

Latency fields are microseconds; throughput is inferences per second. Values
below retain the CLI's reported precision. A row's mean/minimum/maximum describe
its measured calls. The failed attempt supplies no such statistics.

| Model | Trial | Mean (us) | Minimum (us) | Maximum (us) | Inferences/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1 | **Failed: zero interval at 37,270** | — | — | — |
| tiny | 2 | 0.15640399 | 0.083 | 22.333 | 6393698.78 |
| tiny | 3 | 0.09371174 | 0.041 | 11.833 | 10671021.6 |
| tiny | 4 | 0.16313806 | 0.083 | 24.292 | 6129777.44 |
| tiny | 5 | 0.17047407 | 0.083 | 27.958 | 5865994.75 |
| mnist | 1 | 104.349249 | 97.125 | 839.667 | 9583.20265 |
| mnist | 2 | 105.786189 | 97.125 | 1456.417 | 9453.02979 |
| mnist | 3 | 105.936159 | 97.125 | 3550 | 9439.64753 |
| mnist | 4 | 106.506521 | 97.041 | 360.333 | 9389.09645 |
| mnist | 5 | 106.561481 | 97.166 | 329.917 | 9384.25394 |

For the four valid tiny trials, the median **trial mean** is
**0.159771025 us**, with trial means ranging from 0.09371174 to 0.17047407 us.
Their throughput ranges from 5,865,994.75 to 10,671,021.6 inferences/s.
For MNIST, the median trial mean is **105.936159 us**, with trial means from
104.349249 to 106.561481 us. Its median trial throughput is
**9,439.64753 inferences/s**.

These are summaries across the displayed trial aggregates, not medians of all
individual inference latencies. Per-call samples are not stored by the runtime.
Each trial's throughput is computed from its unrounded total measured inference
time, `N / total_seconds`; it is not the average of per-call reciprocal latencies.
The JSON's cross-trial medians are descriptive summaries, not a pooled rate.

All valid trials retain their slowest observations, including the 3,550 us MNIST
maximum in trial 3. No slow samples or valid trials were trimmed. The nine valid
reports cover 400,000 tiny and 500,000 MNIST measured calls; the failed tiny
attempt is additional and has no published timing aggregate.

The tiny workload is close to this clock's granularity: the smallest positive
reported observation is 41 ns, alongside a nominal 42 ns resolution. The
contract retains positive observations below the reported resolution and
rejects zero without retry, rounding, substitution, or batching. The failed
trial therefore remains a real limitation of this per-call baseline. Increasing
the iteration count cannot repair an individual zero interval. The present data
does not establish the cause of its occurrence or support a precise tiny-model
speedup claim. Investigate timing reliability and repeat under comparable quiet
conditions before using tiny-model differences to justify an optimization.

## Numerical and allocation verification

The same fresh Release build passed **37/37 ordinary CTest tests** before the
trials, without Python or dataset downloads. [verification.txt](verification.txt)
preserves that run and the relevant numerical/ownership audit output.

- Tiny round-trip fixtures: all 40 stage vectors / 120 values passed. The maximum
  absolute error was `5.96046448e-8`, in Softmax; earlier stages matched exactly.
- MNIST: all 400 stage vectors / 27,600 values passed shape, finite-value, and
  `atol = rtol = 1e-5` checks. Maximum absolute errors were `6.19888306e-6`
  (Dense1), `5.7220459e-6` (ReLU/logits), and `3.57627869e-7` (probabilities).
  Saved-Python prediction agreement remained 100/100. Classification accuracy
  remains a separate measurement, not a benchmark metric.

**Cinder inference still makes zero allocation/free attempts in the audited
paths.** This is supported by source-level instrumentation, separately from the
uninstrumented executable used for performance:

| Audit | Evidence |
| --- | --- |
| [`cli_setup`](../../../tests/test_cli_setup.c) | For both shapes, 100 repeated real inferences plus native-clock benchmark success/failure/recovery reuse the same buffers after source-file deletion. Allocation, deallocation, and file counters do not change during these calls. Each benchmark sequence has 27 model calls: 26 successful calls and one rejected input. |
| [`benchmark_controlled`](../../../tests/test_benchmark_acceptance.c) | 1,088 complete command invocations use real model ownership and controlled clocks. Allocation/free operations, including `free(NULL)` attempts, are prohibited during execution. Every invocation ends with zero tracked blocks/bytes/streams. Both models have 23 setup allocation points, all tested for failure cleanup. |
| [`mnist_lifecycle_audit`](../../../tests/test_mnist_lifecycle.c) | 300 cycles / 1,384 inference calls, including failures/recovery, have zero allocation/free attempts or loader I/O during cycles; parameters remain unchanged and final ownership is empty. |

The model and CLI buffers are allocated during setup. Models own their layers,
parameters, and two reusable workspaces; the CLI session owns its input and
output. These allocations are allowed and excluded from inference timing.
The audits track Cinder's runtime allocations; they are not a claim that the
entire process or all platform-library internals never allocate. No allocation
instrumentation is added to the timed executable.

## Reproduce the build and collection

Use a clean checkout of the recorded commit. From that checkout's root:

```sh
git rev-parse HEAD
# Expected: cd461c696aea932e6ea53e4093008d6463c42fa9

cmake -S . -B build-baseline-release-20261028 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DCINDER_ENABLE_ASAN=OFF -DCINDER_ENABLE_UBSAN=OFF \
  -DCINDER_ENABLE_NUMPY_TESTS=OFF -DCINDER_ENABLE_EXPORTER_TESTS=OFF \
  -DCINDER_ENABLE_MNIST_TESTS=OFF \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DCMAKE_C_COMPILER=/usr/bin/cc -DCMAKE_C_FLAGS= \
  '-DCMAKE_C_FLAGS_RELEASE=-O3 -DNDEBUG' -DCMAKE_EXE_LINKER_FLAGS= \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build-baseline-release-20261028 --parallel 4
ctest --test-dir build-baseline-release-20261028 --output-on-failure
```

Wait until builds/tests finish before collecting performance. Record the current
hardware, OS, compiler, power/background conditions, commit, and file hashes;
new measurements describe that environment. The two exact benchmark commands
used here are:

```sh
./build-baseline-release-20261028/cinder bench \
  tests/fixtures/tiny/tiny.cnd tests/fixtures/tiny/inputs/worked.f32 \
  --iterations 100000
./build-baseline-release-20261028/cinder bench \
  examples/mnist/mnist.cnd tests/fixtures/mnist/inputs/00000.f32 \
  --iterations 100000
```

To retain five attempts per model, including errors, run the following loop.
Use a fresh output directory name for a new collection; `mkdir` deliberately
rejects an existing collection so earlier evidence is not overwritten.

```sh
mkdir build-baseline-release-20261028/trials-new && (
  for trial in 1 2 3 4 5; do
    for model_name in tiny mnist; do
      case "$model_name" in
        tiny)
          model_file=tests/fixtures/tiny/tiny.cnd
          input_file=tests/fixtures/tiny/inputs/worked.f32 ;;
        mnist)
          model_file=examples/mnist/mnist.cnd
          input_file=tests/fixtures/mnist/inputs/00000.f32 ;;
      esac
      record=build-baseline-release-20261028/trials-new/$model_name-$trial
      date -u '+%Y-%m-%dT%H:%M:%SZ' > "$record.started"
      if ./build-baseline-release-20261028/cinder bench \
          "$model_file" "$input_file" --iterations 100000 \
          > "$record.stdout" 2> "$record.stderr"; then
        result=0
      else
        result=$?
      fi
      printf '%s\n' "$result" > "$record.status"
      date -u '+%Y-%m-%dT%H:%M:%SZ' > "$record.finished"
      sleep 1
    done
  done
)
```

The loop does not retry failed calls or suppress failed trials. It collects
stdout/stderr/status independently; inspect every status before summarizing.
Timing includes model input validation/copies, layers, and final output copy.
Loading, allocation, preprocessing, warm-ups, output observation, statistics,
cleanup, and printing are excluded; clock-boundary overhead and preemption
within an interval are unavoidable. Values do not predict whole-command wall
time or dataset throughput.

This report records a macOS baseline only. No Linux performance baseline,
hosted CI result, optimization, or Phase 5 closure is claimed by this step.
Next: assess whether the evidence justifies any runtime change, accounting for
the tiny-model timing limitation, then perform like-for-like validation before
making improvement claims.
