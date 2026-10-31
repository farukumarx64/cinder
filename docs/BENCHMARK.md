# `cinder bench` command and measurement contract

**Defined October 27, 2026; `cinder bench` implemented October 28, 2026.**
The command now validates arguments, prepares one model/input/output session,
runs ten warm-ups and the measured calls, derives statistics, releases ownership,
and prints the report. The [preparation report](#preparation-implementation-and-verification),
[timer report](#monotonic-timer-implementation-and-verification),
[loop report](#warm-up-and-measured-loop-implementation-and-verification),
[statistics/report verification](#statistics-and-cli-report-implementation-and-verification), and
[acceptance audit](#benchmark-acceptance-audit)
record the implementation and checks. The synthetic examples below illustrate
arithmetic. The [first Release baseline](benchmarks/2026-10-28-apple-m4/README.md)
records actual results, including a tiny-network zero-interval failure.
The [M5 evaluation](M5_VERIFICATION.md) closes Phase 5 on October 29, 2026:
no optimization was justified, an unchanged-executable control is preserved,
and local checks plus all five hosted jobs passed for the unchanged code.
Implementation-step results below are historical; the M5 report supplies the
completed cross-platform verification and current limitations.

The [run contract](CLI.md), [input format](MODEL_FORMAT.md#input-file-and-preprocessing-contract),
and [model API](MODEL.md) remain authoritative for loading, input encoding,
ownership, numerical behavior, and per-model call serialization. Benchmarking
uses those same paths and public accessors.

## Invocation and argument rules

```text
cinder bench <model.cnd> <input.bin> --iterations N
```

Example invocation:

```sh
./build-release/cinder bench examples/mnist/mnist.cnd \
  tests/fixtures/mnist/inputs/00000.f32 --iterations 1000
```

- Exactly five arguments follow the executable: the case-sensitive command
  `bench`, two nonempty literal paths, the exact token `--iterations`, and `N`,
  in that order. The option is required; there is no implicit iteration count.
- The two path slots are always paths, even if named `-`, `--iterations`, or
  another leading-hyphen name. `-` does not mean stdin. Relative paths use the
  working directory, shell quoting handles spaces, and ordinary symlink
  resolution applies. Filename suffixes are descriptive, not validation rules.
- `N` is a nonempty string of ASCII digits `0..9`, interpreted in decimal.
  Leading zeros are accepted (`0001000` means 1000), then omitted in output.
  Reject signs, whitespace, decimal points, exponents, hexadecimal prefixes,
  non-ASCII digits, and any trailing characters.
- The inclusive range is **1 through 1,000,000 measured iterations**. Zero and
  larger values are usage errors. Check the range while parsing, before a
  multiply/add can overflow; never rely on a wrapped integer or unchecked
  conversion. Very long digit strings must be rejected or accepted by value
  under the same rule, without allocating a parsing buffer.
- Reject extra/repeated/reordered options, `--iterations=1000`, `--warmup`,
  `--help`, and `--`. No batching, duration-based stopping, automatic iteration
  selection, or output-file option is part of this command.
- Complete all argument validation before opening files, loading a model,
  allocating its buffers, querying a clock, or running inference.

The limit bounds the requested call count and keeps the counter portable; it
does not guarantee a short run for a large supported model. There is no automatic
wall-time cutoff. Signal termination is outside the normal cleanup/exit contract.

## Setup, ownership, and warm-up

Use the same model/input validation as `run`, including non-Softmax models,
positive dimensions, resource limits, checked allocation extents, and exact
input length. The input contains one already-preprocessed, finite little-endian
float32 vector. There is no image decoding or preprocessing in this command.

Before warm-up or timing:

1. Load the model once with `cinder_model_load` and obtain its dimensions through
   public accessors. A model failure takes precedence over any input failure.
2. Read and close the input through the existing private reader. Its exact
   length, finite values, clean EOF, and successful close are required.
3. Allocate one disjoint caller-owned output buffer with exactly the model's
   output capacity. The model already owns its layers, parameters, and two
   reusable working buffers. Initialize benchmark counters/statistics on the stack.
4. Establish an available monotonic elapsed-time clock and query its positive
   resolution. Fail if either is unavailable or invalid; do not fall back to
   calendar time or process CPU time. The timer uses
   `clock_gettime(CLOCK_MONOTONIC_RAW)` and queries the same clock with
   `clock_getres`; record this backend in the baseline report.

Run **exactly 10 warm-up calls**, sequentially on the same model, same input,
and same output buffer. Warm-up is fixed for this first contract and is not
user-configurable. Every call must succeed and its output must pass the
observation rule below. Warm-ups use no start/end timestamps and contribute
nothing to counts, minimum, maximum, total, mean, or throughput.

Only after all 10 warm-ups succeed do the `N` measured calls begin. Thus a
successful invocation performs exactly **N + 10 model calls**, with no extra
preflight or final inference. Warm-up aims to reduce first-call effects; ten
calls do not guarantee thermally stable or fully warmed execution on every host.

No model reload, workspace reset, input mutation, inference allocation/free,
or file I/O occurs between calls. All calls use one model sequentially; there
are no worker threads or simultaneous calls on the instance. Additional storage
is constant with respect to `N`; do not allocate an array of per-call timings.

## Timing boundary and clock rules

For measured iteration `i`, bracket exactly one public call:

```text
start = read_monotonic_clock()
status = cinder_model_run(model, input, D, output, M, error)
end = read_monotonic_clock()
validate status and timestamps
observe output
update timing statistics
```

This is measured **wall-clock latency of the public inference API**. It includes
that call's input validation, input copy into the model workspace, layer
execution, final output copy, and status/error handling. It is not a
matrix-kernel-only measurement. Operating-system preemption during the interval
is included. Process startup, file I/O, construction, allocation, preprocessing,
warm-up, between-call loop work, output observation, statistics, destruction,
and printing are excluded.

Use one monotonic clock for resolution and all timestamps. Calendar changes
must not make this clock run backwards. The backend's behavior across machine
suspend need not be identical on every platform; do not suspend the machine
during a recorded baseline trial. Clock reads and call-boundary bookkeeping
have unavoidable measurement overhead. Do not subtract a guessed clock cost,
discard slow iterations, trim outliers, clamp times, or silently time batches.

The timer interface must supply normalized, nonnegative seconds and nanoseconds
(`0 <= nanoseconds < 1,000,000,000`), or an equivalent checked representation.
Validate both timestamps. The first start has no prior sample; each subsequent
start must be at least the previous end, and each end must be at least its start.
Any backward transition fails the whole command.

Subtract timestamps before converting the interval to **unsigned 64-bit integer
nanoseconds**. Handle the nanosecond borrow and check multiplication/addition
for representability. Do not convert absolute uptime to float or subtract large
floating-point timestamps. Clock resolution is also checked, represented as a
positive unsigned 64-bit integer nanosecond count, and printed with the report.

Every measured interval must be **strictly positive**. If start and end are
equal, fail with a timing diagnostic and no statistics. Do not report infinite
throughput, invent a one-nanosecond result, omit that call, or retry it. Raising
`N` does not repair an individual zero-length measurement under this per-call
policy; a suitable timer or a separately specified future batching mode would
be needed. Positive quantized observations are retained as reported by the clock,
including positive differences smaller than its reported resolution. Resolution
is metadata, not a threshold used to round, discard, or retry a positive sample.
It and the printed decimal digits do not establish measurement accuracy.

The [October 29 timing investigation](TIMING_INVESTIGATION.md) captured equal
native timestamp pairs around completed tiny inferences and retains this policy.
Reported resolution is 42 ns on that Mac; positive 41/42 ns differences and
unresolved zero intervals were observed. The command can therefore fail for
very short work even when inference produces the correct result.

## Observe results outside the timer

After each successful warm-up or measured call, scan all `M` output values in
index order. Require each value to be finite and copy it to a dedicated
`volatile float` sink outside the timed interval. This makes every result
observable to the compiler without including formatting or a checksum in the
timing report. Do not make the model/input/output arrays volatile or add sink
writes inside `cinder_model_run`. The sink needs no allocation and is not printed.

Stop at the first non-finite output. The input and model parameters remain
unchanged, while model workspaces and the output buffer are reused as allowed
by the existing model contract. This finite-value check is not proof of numerical
correctness: existing NumPy/Python comparisons remain required for the baseline
and after any performance change. Benchmark output contains neither predictions
nor classification accuracy.

Observation and statistics happen outside each interval, but still affect the
overall process workload and cache state between calls. Report throughput as
the reciprocal of measured mean API latency, not end-to-end application throughput.

## Accumulation and numerical definitions

For `N` successful measured calls, let `d_i` be the positive integer elapsed
nanoseconds for call `i`, and `S = sum(d_i)`. Keep `S`, the minimum, and the
maximum as unsigned 64-bit integers. Reject a sum that would overflow before
adding it. Update the successful measured count only after status, timestamps,
output observation, and accumulation have succeeded. Warm-ups never update it.

Compute these final values in `double`, using the unrounded integer statistics:

| Report field | Definition / calculation order |
| --- | --- |
| Mean latency (us) | `((double)S / (double)N) / 1000.0` |
| Minimum latency (us) | `(double)min(d_i) / 1000.0` |
| Maximum latency (us) | `(double)max(d_i) / 1000.0` |
| Throughput (inferences/s) | `((double)N / (double)S) * 1000000000.0` |

Throughput is **N divided by total measured inference time**, not the arithmetic
mean of individual reciprocal latencies. Require all four results to be finite
and strictly positive before formatting. Integer accumulation avoids per-sample
floating-point summation drift; conversion to `double` may still round large
totals. Never calculate throughput from an already rounded display string.

Worked arithmetic only: for measured intervals `[1000, 2000, 3000]` nanoseconds,
`N = 3` and `S = 6000`. Mean is 2 us, minimum 1 us, maximum 3 us, and throughput
500,000 inferences/s. These synthetic values do not describe Cinder performance.

## Exact successful stdout

The report contains exactly these nine lines, in this order:

```text
Model: <escaped-basename>
Iterations: <N>
Warm-up iterations: 10
Clock: monotonic
Clock resolution (ns): <R>
Mean latency (us): <mean>
Minimum latency (us): <minimum>
Maximum latency (us): <maximum>
Throughput (inferences/s): <throughput>
```

- Basename selection and byte escaping exactly follow `run`: the component
  after the last `/` in the supplied model path, without symlink-name resolution;
  printable ASCII except quotes/backslashes passes through, quotes/backslashes
  are escaped, and other bytes use uppercase `\xHH` escapes.
- Counts and resolution use unsigned decimal integers without leading zeros.
  All four calculated fields use C `%.9g` under the `C` numeric locale. Scientific
  notation is permitted. Use the ASCII unit `us` for microseconds.
- Each line ends with LF, including the final line. No tabs, blank lines,
  trailing spaces, banners, progress messages, scores, percentiles, or timing
  of loading are added. Success leaves stderr empty.
- Complete all measured calls, validate statistics, and release model/input/
  output ownership before writing the first report byte. Retain only the scalar
  statistics and supplied path needed for display. Check every write and the
  final stdout flush.

Applying the synthetic arithmetic above with an assumed 1 ns clock resolution
would print the following **illustration, not an executed command's output**:

```text
Model: example.cnd
Iterations: 3
Warm-up iterations: 10
Clock: monotonic
Clock resolution (ns): 1
Mean latency (us): 2
Minimum latency (us): 1
Maximum latency (us): 3
Throughput (inferences/s): 500000
```

## Exit statuses and usage diagnostics

Normal exit statuses retain the CLI convention:

| Status | Meaning |
| ---: | --- |
| `0` | All setup, warm-up, measured calls, statistics, cleanup, report writes and stdout flush succeeded. |
| `1` | Operational failure: model/input/allocation/inference/output errors, unavailable or invalid clock, zero/backward time, or unrepresentable timing/statistics. |
| `2` | Invalid command syntax, paths, option, or iteration value. |

**All CLI usage errors** use this expanded block, including errors for `run`
and `--version`:

```text
cinder: usage: <reason>
Usage:
  cinder --version
  cinder run <model.cnd> <input.bin>
  cinder bench <model.cnd> <input.bin> --iterations N
  cinder info <model.cnd>
```

Every line ends with LF. The global reason for missing/unknown commands or extra
arguments to `--version` is `expected --version, run, bench, or info.` Existing
`run` argument reasons, successful output, and operational errors stay as defined
in [CLI.md](CLI.md). Benchmark usage was added October 28; the unreleased
[inspection command](INFO.md) adds `info` on October 31. Successful
`run`/`--version` output is unchanged.

For recognized `bench`, select the first applicable reason in this order:

| Condition | Exact reason |
| --- | --- |
| Argument count is not exactly six including the executable | `bench requires two file paths followed by --iterations N.` |
| Empty model path | `model path must not be empty.` |
| Empty input path | `input path must not be empty.` |
| The fourth argument after the executable is not `--iterations` | `bench requires --iterations after the two file paths.` |
| `N` is empty or contains any non-ASCII-digit byte | `iterations must contain only ASCII decimal digits.` |
| Decimal value is zero or exceeds 1,000,000 | `iterations must be between 1 and 1000000.` |

Check the complete lexical form before reporting a numeric range error. For
example, `999999999999x` receives the digit diagnostic; an equally long string
of nines receives the range diagnostic without integer overflow.

## Operational errors and failure order

Setup reuses the model/input/inference diagnostic templates, status-name mapping,
bounded detail handling, fallback `Operation failed.`, escaping, and exact
reader/allocation/dimension details from [CLI.md](CLI.md). The library stays
silent. Once execution begins, identify the phase and **one-based** call number:

```text
cinder: benchmark warm-up I [STATUS]: detail
cinder: benchmark iteration I [STATUS]: detail
cinder: benchmark [STATUS]: detail
cinder: output [IO]: Could not write benchmark output.
```

`I` is 1..10 for warm-up or 1..N for measured calls; it does not combine the two
phases. Output element indices `J` below remain zero-based. These tags reuse
existing library status names; this contract adds no public status enum or API.

| Failure | Context / tag | Exact CLI-generated detail |
| --- | --- | --- |
| No supported monotonic backend | benchmark / `UNSUPPORTED` | `No monotonic clock is available.` |
| Resolution query fails | benchmark / `IO` | `Could not query monotonic clock resolution.` |
| Resolution is nonpositive or malformed | benchmark / `IO` | `Monotonic clock resolution is invalid.` |
| Positive resolution cannot fit uint64 nanoseconds | benchmark / `OVERFLOW` | `Clock resolution nanoseconds are not representable.` |
| Start clock read fails | iteration I / `IO` | `Could not read start timestamp.` |
| End clock read fails | iteration I / `IO` | `Could not read end timestamp.` |
| Timestamp has negative seconds or unnormalized nanoseconds | iteration I / `IO` | `Monotonic clock returned an invalid timestamp.` |
| Start precedes prior end, or end precedes start | iteration I / `IO` | `Monotonic clock moved backwards.` |
| Interval cannot fit uint64 nanoseconds | iteration I / `OVERFLOW` | `Elapsed nanoseconds are not representable.` |
| Interval is zero | iteration I / `IO` | `Measured interval is zero; clock resolution is insufficient.` |
| Total would overflow | iteration I / `OVERFLOW` | `Total elapsed nanoseconds are not representable.` |
| Successful call returns a non-finite output | warm-up I or iteration I / `NONFINITE` | `Output value at index J is not finite.` |
| Final calculated statistic is non-finite or nonpositive | benchmark / `NONFINITE` | `Benchmark statistics must be finite and positive.` |

Propagate each model-call failure with its returned status and bounded library
detail in the corresponding phase/iteration template. No `errno` text is appended.

Failure order is argument validation → model loading/dimensions → input reading/
close → output allocation → clock availability/resolution → warm-ups → measured
calls → final statistics → report writes/flush. During a measured call:

1. Require a valid start timestamp and non-backward transition from the previous
   end before invoking the model.
2. Invoke the model once and immediately attempt the end clock read, even when
   the model returns failure. Preserve the model status/detail first: a model
   error wins over a simultaneous end-clock error. Never reuse the same error
   object in a way that overwrites that failure.
3. If the model succeeded, validate the end timestamp, ordering, and positive
   representable interval before inspecting output and accumulating statistics.
   A timing error therefore precedes an unexpected non-finite successful output.

Stop immediately on the first failure under these rules. Do not retry a call,
continue collecting samples, or print a summary for a successful prefix. Before
printing, ordinary failure leaves stdout empty and attempts one stderr diagnostic
without usage. Every failure releases acquired model/buffer ownership; each opened
file gets exactly one close attempt, with the earlier error preserved. No signal
handler, cancellation API, or recovery within a failed command is introduced.

Detected report-write or final-flush failure exits `1` and attempts the exact
output diagnostic. Stdout may already be partial and cannot be rolled back.
Failure to write stderr does not change failure into success.

## Baseline reporting and interpretation

Use an optimized **Release** build for published performance results. Debug and
sanitizer builds may execute `bench` for functional verification, but their times
must not be presented as the optimized baseline. The command does not guess or
enforce compiler settings from the executable name.

For every recorded baseline, preserve:

- The exact commit, command, model/input identifiers and SHA-256 hashes.
- CPU model/architecture, operating system, compiler/version, CMake version,
  build type and optimization/sanitizer options; clock backend and resolution.
- The fixed 10 warm-ups, requested measured count, and complete output.
- At least three separate invocations under comparable conditions, with all
  trial results retained and relevant background-load/power conditions noted.
- The accompanying numerical validation and allocation-count checks. Each
  inference must still allocate/free nothing; setup allocations are allowed.

Each invocation measures one repeated input on one loaded model, not a dataset
batch, cold loading, or real-world multi-user throughput. Compare the same model,
input, measurement contract, build settings, and host when evaluating changes.
Do not substitute illustrative roadmap timings for a measured baseline or claim
a speedup against an allocation-heavy implementation that never existed.

## Acceptance cases for implementation

1. Preserve version/run behavior except for the explicitly expanded usage
   text. Test exact bench syntax, literal unusual paths, leading zeros, counts
   1 and 1,000,000, zero, above-limit/huge strings, invalid bytes, and wrong option
   positions. Argument failures must cause no file/clock/model/allocation calls.
2. Verify one load/read/output allocation, successful input close before any
   inference, exactly 10 warm-ups followed by N measured calls, and 2N clock
   reads on success. Test the maximum count through parser or stubbed-loop
   checks, not one million expensive real MNIST runs in CI.
3. Inject a controlled monotonic clock: known intervals, second-boundary borrow,
   large uptime with tiny intervals, minimum/maximum ordering, zero/backward time
   within and between samples, resolution/read failures, invalid representations,
   interval/sum overflow, and finite positive final statistics. Check the worked
   `[1000, 2000, 3000]` arithmetic and exclude warm-up/setup/observation time.
4. Instrument ownership, calls, and allocation counts; force setup, warm-up,
   measured inference, output-finiteness, clock, and output-write/flush failures.
   Check one-based context, first-error precedence, no partial summary, and full
   cleanup. A later independent invocation must still succeed.
5. Verify exact nine-line formatting, escaping, numeric locale, counts/units,
   no unwanted prediction/progress text, and result consumption outside timing.
   Check real tiny/MNIST execution with small N and existing numerical references.
6. Keep ordinary tests independent of Python and downloads. Functional CI checks
   structure, counts, errors, and controlled-clock arithmetic; it must not impose
   a speed threshold or compare exact real-clock durations between machines.

## Preparation implementation and verification

Implemented October 27, 2026 in [`src/cli_session.h`](../src/cli_session.h) and
[`src/cli_session.c`](../src/cli_session.c). These are CLI-private helpers, not
new public runtime APIs. `run` now uses them, and the future `bench` command can
reuse the same preparation without duplicating loading or buffer ownership.

`cinder_cli_session_t` holds the loaded model, input/output pointers, and their
element counts. Initialize it with `{0}`. `cinder_cli_prepare` performs these
steps once, in the existing `run` failure order:

1. Call `cinder_model_load` and obtain positive input/output dimensions through
   the public accessors. The loaded model already owns its layers, parameters,
   and two reusable working buffers.
2. Call `cinder_input_read` for exactly the required input count. This reuses its
   byte-order, finite-value, exact-length, and file-close checks. Both files are
   closed before preparation succeeds.
3. Check `output_size * sizeof(float)` for overflow, then allocate one separate
   output buffer with exactly that capacity.
4. Publish the complete session to the caller. Preparation performs no inference,
   clock queries, or printing. The output is uninitialized until inference
   succeeds; no zero-fill is required by the model API.

All ownership stays in a temporary session until success. Any operational
failure releases partial allocations and leaves the caller's session empty.
`cinder_cli_failure_t` preserves the original status/detail plus the existing
`model`, `input`, or `inference` diagnostic context. Its path borrows the supplied
argument; cleanup does not free it or replace the first failure. Failure details
are optional. A NULL or already nonempty destination is rejected before I/O;
an existing session is preserved, so accidental double preparation cannot lose
its allocations.

`cinder_cli_release` frees the output, input, and model, then clears every field.
An empty session or NULL is safe, and a released object can be prepared again.
Do not shallow-copy an owning session. The model still owns its internal storage;
the session owns the separate input and destination output. Calls using the same
model must remain serialized, and release must wait for inference to finish.

The future benchmark will prepare once, query its clock, perform warm-ups and
measured calls using these same buffers, then release. The shared helper does
not start a timer or implement that loop. `run` still performs one inference,
checks all scores, ranks up to three, releases ownership, and prints the same
report. Arguments, usage, exit statuses, diagnostics, and formatting are unchanged.

[`tests/test_cli_setup.c`](../tests/test_cli_setup.c), registered as `cli_setup`,
instruments the real loader, reader, runtime, and preparation code with test-only
allocation and file counters. It uses committed tiny-network and MNIST fixtures:

- Each successful preparation calls the loader once and the input reader once,
  closes both files, performs no inference, and allocates the exact input/output
  capacities. The buffers are separate.
- After deleting temporary copies of both source files, each prepared model
  completes 100 repeated inferences. Outputs are overwritten with NaNs before
  each call, then checked for shape, finiteness, and agreement with saved NumPy
  probabilities at `atol = rtol = 1e-5`. Input contents and allocation,
  deallocation, and file-operation counters remain unchanged.
- Each fixture has 23 allocation points: 21 in loading/construction, one input
  allocation, and one output allocation. Every point is forced to fail with and
  without a diagnostic destination: 92 failure cases across the two fixtures.
  Each failure leaves zero retained blocks, bytes, and streams; a subsequent
  preparation and inference succeeds.
- Model/input open and close failures, malformed input length, non-finite input,
  first-error precedence, rejected double preparation, and repeated cleanup are
  covered. Existing `cli_audit` assertions now exercise the extracted helper;
  the 122 real `cli_process` cases still verify the public command behavior.

Local results on macOS arm64 with Apple Clang 21.0.0 and CMake 4.4.4:

| Check | Result |
| --- | --- |
| Debug with optional Python reference checks | 39/39 passed |
| Release, ordinary C tests only | 30/30 passed |
| Debug ASan + UBSan, ordinary C tests only | 30/30 passed; no findings |
| Native `leaks --atExit` on `test_cli_setup` and an actual MNIST `cinder run` | Both exited 0; 0 leaks / 0 leaked bytes |
| C11 syntax checks on changed/new C sources with `-Wall -Wextra -Wpedantic -Werror` | Passed |

Reproduce the focused checks after building a Debug tree (here `build-mnist`):

```sh
ctest --test-dir build-mnist --output-on-failure -R '^cli_'
leaks --atExit -- ./build-mnist/test_cli_setup "$PWD" "$PWD/build-mnist/cli-setup-files"
leaks --atExit -- ./build-mnist/cinder run examples/mnist/mnist.cnd \
  tests/fixtures/mnist/inputs/00000.f32
```

Run the manual setup audit separately from CTest when using the same scratch
directory. It creates and removes temporary fixture copies there. All ordinary
C tests use committed fixtures and require neither Python nor MNIST downloads.
ASan used `detect_leaks=0` on this Mac; native leak checking provides separate
evidence. The native tool reported its restricted-process inspection notice
while completing both zero-leak checks. No hosted macOS/Linux CI result is
claimed for these preparation changes; the existing matrix will run
the new test on a later authorized push. Timing and performance are unmeasured.

The next step at the time of this preparation report was the monotonic timer;
its implementation is recorded below.

## Monotonic timer implementation and verification

Implemented October 27, 2026 in [`src/timer.h`](../src/timer.h) and
[`src/timer.c`](../src/timer.c), as a private `cinder_cli_timer` build target.
It does not change the public runtime API or query a clock during `run` or
`--version`. No heap allocation, file handling, global measurement state, or
timer destruction is required.

The backend is **`CLOCK_MONOTONIC_RAW` on both macOS and Linux**, accessed through
`clock_gettime` and `clock_getres`. Both hosts document this raw clock as
unaffected by clock frequency/time adjustments; their suspend behavior may
differ. See the [Linux clock manual](https://man7.org/linux/man-pages/man3/clock_gettime.3.html)
and Apple's installed SDK manual (`man 3 clock_gettime`, macOS 10.12 and newer).
Do not mix timestamps from different clock IDs or suspend a baseline trial.

CMake checks that the raw clock ID and both functions compile and link. Timing
targets enable POSIX declarations, plus `_DARWIN_C_SOURCE` on macOS to expose
the raw clock extension. Linux retries the link probe with `librt` if needed.
If the backend cannot be compiled, initialization returns `UNSUPPORTED` with
`No monotonic clock is available.` There is no calendar/CPU-clock fallback.
The native test requires a working backend on the supported macOS/Linux hosts,
so an accidental feature-detection failure cannot silently pass their tests.

The interface has three operations:

| Function | Behavior |
| --- | --- |
| `cinder_timer_init` | Reset all state; query and validate resolution once; store positive `uint64_t resolution_ns`. Failure leaves the timer zeroed and unusable until successful initialization. |
| `cinder_timer_start` | Require an initialized, inactive timer; read and validate a start timestamp; reject a start before the last successful end. |
| `cinder_timer_stop` | Require an active measurement and output destination; read/validate its end; calculate a positive, representable elapsed nanosecond count; publish the result and remember the successful end. |

Timestamps retain separate signed 64-bit seconds and 32-bit nanoseconds after
checking the OS result: seconds must be nonnegative and representable;
nanoseconds must be in `0..999999999`. Absolute uptime is never converted to
floating point or multiplied into a potentially overflowing nanosecond count.
Instead, stop subtracts seconds/nanoseconds, borrows a second if necessary, and
checks the final multiply/add against `UINT64_MAX` before performing it.

For example, start `(3 s, 999999900 ns)` and end `(4 s, 100 ns)` produce exactly
`200 ns`. Even at `INT64_MAX` seconds, timestamps one nanosecond apart produce
`1 ns`. An interval of exactly `UINT64_MAX` ns is accepted; the next value is
rejected as `OVERFLOW`.

Failures use the existing status codes and the exact diagnostic details in the
table above: resolution/read failures, malformed timestamps, backwards time,
and zero intervals use `IO`; unrepresentable resolution/intervals use `OVERFLOW`.
Misuse of the private API, such as starting twice or stopping without a start,
uses `INVALID_ARGUMENT` before querying the clock. Optional error details are
cleared on success. The helper prints nothing; iteration context and exit status
belong to the later benchmark command.

A failed start leaves the timer unchanged. A valid stop attempt consumes the
active measurement even if the clock or duration fails, but leaves the caller's
elapsed destination and previous successful end unchanged. A new start can
recover for another independent use; the benchmark itself must stop on its
first failure. Reinitialization starts a fresh sequence. Separate timer objects
have independent state; calls on one object must be serialized. Output/error
destinations must not overlap timer state or one another.

When integrated, stop must be called even if the intervening inference fails,
using a separate error object to preserve model-error precedence. Start/end
reads enclose one public inference call; there is unavoidable clock and
call-boundary bookkeeping overhead. No warm-up, result observation, statistics,
or printing is performed by this helper. The measured loop is the next step.

The zero-duration rule is an explicit resolution failure, with no retry,
clamping, invented duration, batching, or dropped sample. A positive observation
below reported resolution is retained under the existing contract, not promoted
to a claim of corresponding accuracy. More iterations cannot rescue a zero
duration for an individual call.

Three ordinary C tests were added:

- `timer_checked` replaces only the OS clock calls, exercising the real timer
  code with deterministic values. It covers invalid/zero/overflowing resolution,
  failed reads, negative or unnormalized timestamps, within/across-call backwards
  transitions, equality to the prior end, second-boundary borrowing, large uptime,
  exact/overflowing uint64 limits, zero and positive sub-resolution intervals,
  API misuse, preserved failure outputs, recovery, and independent timers.
- `timer_unavailable` compiles the real unavailable-backend branch without
  POSIX clock support and verifies explicit rejection and reset state.
- `timer_native` links the actual library and reads the real host clock. Its
  32 pairs include one short sleep and adjacent reads; it accepts either a
  positive interval or the exact zero-interval rejection for each pair. This
  checks clock behavior without a speed threshold or an inference benchmark.

The final raw-clock probe on this Mac reported **42 ns resolution**, compared
with **1000 ns** for the regular `CLOCK_MONOTONIC` checked during development.
This motivated using the raw clock for short intervals. Adjacent raw reads can
still coincide: zero durations are explicitly rejected, including in the native
test. Neither resolution value is an inference latency or accuracy guarantee.

Final local checks on macOS 27.0.1 arm64, Apple Clang 21.0.0, CMake 4.4.4:

| Check | Result |
| --- | --- |
| Debug with optional Python reference checks | 42/42 passed |
| Release, ordinary C tests only | 33/33 passed |
| Debug ASan + UBSan, ordinary C tests only | 33/33 passed; no findings |
| Native `leaks --atExit` on controlled and native timer tests | Both exited 0; 0 leaks / 0 leaked bytes |
| C11 syntax checks, available/unavailable branches, `-Wall -Wextra -Wpedantic -Werror` | Passed |
| Fresh Release build with `BUILD_TESTING=OFF` and `--version` | Passed |

Reproduce the focused tests and native macOS leak checks:

```sh
cmake -S . -B build-timer -DCMAKE_BUILD_TYPE=Debug
cmake --build build-timer --parallel
ctest --test-dir build-timer --output-on-failure -V -R '^timer_'
leaks --atExit -- ./build-timer/test_timer
leaks --atExit -- ./build-timer/test_timer_native
```

All three timer tests are ordinary C tests and require neither Python nor data
downloads. The full suites include the unchanged `run`/version regression tests.
ASan used `detect_leaks=0` on this Mac; native leak checks are separate evidence
and completed with the tool's restricted-process inspection notice. A local
Linux container run was unavailable because Docker Desktop was manually paused;
it was left paused. No Linux execution or new hosted CI result is claimed for
this timer step. The existing macOS/Linux CI matrix will automatically include
these tests on a later authorized push. The legacy Linux `librt` fallback is
implemented but not exercised on this Mac. No inference performance is measured.

The next step at the time of the timer report was the warm-up and measured
inference loop; its implementation is recorded below.

## Warm-up and measured-loop implementation and verification

Implemented October 27, 2026 in [`src/benchmark.h`](../src/benchmark.h) and
[`src/benchmark.c`](../src/benchmark.c). The CLI-private function is:

```c
cinder_status_t cinder_benchmark_execute(
    const cinder_cli_session_t *session,
    uint32_t iterations,
    cinder_benchmark_result_t *result,
    cinder_benchmark_failure_t *failure);
```

The caller prepares the session once with `cinder_cli_prepare`, passes it to
this helper, and releases it afterwards on success or failure. The helper
borrows the model and input/output buffers without loading, allocating, freeing,
resetting workspaces, or printing. It validates required pointers, positive
dimensions matching the model's public accessors, and `iterations` in
`1..1,000,000` before querying the timer. The future command must still validate
argument syntax before preparing the session. Ownership and pointer validity
remain caller preconditions; destinations and session storage must be disjoint.

One timer initialization precedes exactly ten warm-ups. Each warm-up calls the
same public model API using the same buffers, then checks/consumes its output.
Warm-ups use no start/end timestamps and contribute no measured samples or
nanoseconds. Measured calls then execute in this order:

```text
start timer → model_run → stop timer → check errors
           → check/consume output → accumulate integer measurements
```

Success performs exactly `N + 10` model calls, `N` starts, `N` stops, and one
resolution query. There is no extra inference after the loop. A stack-local
`volatile float` sink receives each finite output element in index order after
every successful call. Input/output arrays remain ordinary float arrays. The
sink makes the writes observable without adding allocation, output formatting,
or a checksum to the measured interval. All additional storage is constant in N.
The finite scan is an integrity check, not a replacement for NumPy comparisons.

`cinder_benchmark_result_t` stores measured count, resolution, total, minimum,
and maximum integer nanoseconds. A temporary result is updated only after a
successful inference, valid timer interval, and finite output scan. Total
addition is checked before it can overflow. Only a completely successful run
publishes the result; any failure leaves the caller's previous result unchanged.
The session output is reusable scratch storage and can already contain a prior
completed call when a later call fails; it is not rolled back. Derived floating-
point latency/throughput and exact CLI report formatting remain the next step.

Failure details retain a typed phase (`SETUP`, `WARMUP`, or `MEASURED`) and a
one-based call index within that phase; setup and cleared success use index zero.
The optional diagnostic is cleared on success. Failure stops immediately without
retrying or publishing partial measurements. A measured inference failure still
gets its end-clock read; separate error objects preserve the inference error if
the end read also fails. A timer failure precedes a non-finite output error,
and output validation precedes a total-overflow error. Releasing the borrowed
session remains the caller's responsibility on every return path.

[`tests/test_benchmark.c`](../tests/test_benchmark.c), registered as
`benchmark_loop`, substitutes model/timer APIs and observes finite-value scans:

- Check exact call counts for 1, 3, and 1,000,000 measured iterations. The maximum
  uses a cheap simulated model rather than a million real MNIST executions.
- Inject large setup, warm-up, and observation costs; measured intervals
  `[1000, 2000, 3000]` still produce count 3, total 6000, minimum 1000, and
  maximum 3000 ns. Assert observation never occurs between start and stop.
- Fail inference at every warm-up and measured position; check stopping,
  one-based context, unchanged result storage, end-read behavior, and recovery.
- Fail setup/start/end clocks, combine inference/end failures and clock/non-finite
  failures, and check precedence. Inject NaN and both infinities at every output
  position in every warm-up/measured call. Check total overflow after a maximum
  interval, and accept a single interval exactly equal to `UINT64_MAX`.
- Reject invalid arguments before clock/model execution. Test both optional
  diagnostics paths. Direct allocations/frees, file access, or report-printing
  calls in the loop are redirected to unresolved names so they fail the test link.

The existing [`cli_setup` audit](../tests/test_cli_setup.c) now also exercises
the real loop, real timer, and real tiny/MNIST runtime after removing temporary
copies of the model/input files. For each model, it runs a successful N=1 loop,
an input-failure loop, then a successful N=5 loop using the same session. This is
27 model API calls, including one rejected warm-up; the 26 successful calls
execute all four layers. Counters confirm three timer initializations, six
starts/stops, unchanged allocation/deallocation and file-operation counts, one
original load/read, stable buffer pointers, and preserved input after recovery.
Final vectors match the saved NumPy probabilities with shape/finite checks and
`atol = rtol = 1e-5`. Existing construction-failure and cleanup checks still pass.

Local checks on macOS arm64, Apple Clang 21.0.0 and CMake 4.4.4:

| Check | Result |
| --- | --- |
| Debug with optional Python reference checks | 43/43 passed |
| Release, ordinary C tests only | 34/34 passed |
| Debug ASan + UBSan, ordinary C tests only | 34/34 passed; no findings |
| Release with ThinLTO (`-O3 -flto=thin`), loop and real setup audits | 2/2 passed |
| Native `leaks --atExit`, loop and real setup audits | Both exited 0; 0 leaks / 0 leaked bytes |
| C11 syntax checks for new/changed C sources, warnings as errors | Passed |

An inspection of the optimized production assembly also confirmed measured
start/model/stop calls remain inside the loop and each finite-element sink store
follows the stop call. This verifies the current compiler output; it is not a
performance baseline or a promise that every future compiler transformation is
identical. The normal `run` and version regression tests remain unchanged.

Reproduce focused checks after configuring/building a Debug tree:

```sh
ctest --test-dir build-mnist --output-on-failure -R '^(benchmark_loop|cli_setup)$'
leaks --atExit -- ./build-mnist/test_benchmark
leaks --atExit -- ./build-mnist/test_cli_setup "$PWD" "$PWD/build-mnist/cli-setup-files"
```

Run the manual setup audit separately from CTest when sharing its scratch
directory. ASan used `detect_leaks=0` on this Mac; the separate native checks
completed with zero leaks and the tool's restricted-process inspection notice.
Ordinary C tests need neither Python nor downloads. Linux execution
and hosted macOS/Linux CI are not claimed for these changes. No benchmark CLI
report or Release performance baseline is published by this step.

The next step at the time of this loop report was statistics and CLI dispatch;
its completed implementation is recorded below.

## Statistics and CLI report implementation and verification

**Completed locally October 28, 2026.** [`src/benchmark_stats.c`](../src/benchmark_stats.c)
adds `cinder_benchmark_statistics` and the private header adds
`cinder_benchmark_statistics_t`. The raw measurement object still holds exact
integer count/resolution/total/minimum/maximum; the new object holds four
`double` values: mean/minimum/maximum microseconds and inferences per second.
Derivation occurs after execution, outside every timed interval, without heap
allocation or I/O. It does not change the measurement loop or model arithmetic.

The helper checks the count and all durations for zero **before any division**.
It also checks the iteration limit, positive resolution, min/max ordering, and
aggregate bounds. The bounds use integer quotient/remainder rather than
`minimum * count` or `maximum * count`, which could overflow for large inputs.
Missing pointers or inconsistent metadata return `INVALID_ARGUMENT`; zero
counts/durations or invalid final statistics return `NONFINITE`. These metadata
checks are defensive checks for this private helper: a successful execution
already supplies consistent positive measurements. Output remains unchanged
on failure, the raw measurements are never modified, and optional error details
are cleared on success.

The formulas above are implemented in their specified order. Converting the
integer total to `double` before dividing preserves fractional means, and the
rate uses the unrounded total. For example, durations 1 and 9 us produce a
5 us mean and **200,000 inferences/s** (`2 / 0.000010`). Averaging their individual
rates would incorrectly give about 555,556/s. All four calculated values must
be finite and strictly positive before they are published. No epsilon,
substitute duration, retry, clamping, or rounded-text calculation is used.

[`src/cli.c`](../src/cli.c) now dispatches the documented fixed-order command.
It checks the complete ASCII-digit syntax before parsing the decimal value,
then checks the 1..1,000,000 bound before each multiply/add. All argument errors
precede file/clock access. Leading zeros are accepted, including arbitrarily
long runs of leading zeros; an excessive number followed by an invalid byte
receives the lexical diagnostic. No parsing buffer is allocated.

The command calls shared preparation once, the existing execution helper once,
and the statistics helper once. It releases the model/input/output session on
success and failure before any report or diagnostic. Execution failures retain
the warm-up/measured phase and one-based call number. Final statistics errors
use the unnumbered benchmark context. Successful output has exactly nine lines,
with escaped model basename, unsigned counts/resolution, C-locale `%.9g` values,
and final LF. Every write and the final flush is checked; a write failure returns
1 and attempts the benchmark output diagnostic. A partially written report
cannot be rolled back. Normal exits are 0 success, 1 operational error, and
2 usage error. The common usage block gains `bench`; existing successful
`run`/`--version` output and operational errors remain unchanged.

New and expanded ordinary C tests:

- `benchmark_statistics` checks known arithmetic, the unequal-duration rate
  example, fractional means from `[1,1,2]` ns, one iteration, 1,000,000 iterations,
  `UINT64_MAX` totals/resolution, zero denominators/durations, invalid metadata,
  optional diagnostics, and unchanged output on rejection. It does not require
  the clock, a model, Python, or downloads.
- `cli_audit` uses controlled execution results and the real statistics helper
  to check exact nine-line output, fractional/scientific formatting, uint64
  resolution printing, basename escaping, parsing bounds/precedence, setup and
  execution errors, bounded diagnostics, statistics failure, recovery, every
  report-write failure, and flush failure. Ownership counters require cleanup
  before the first output byte. The existing prediction tests remain in place.
- `cli_process` now passes **133 actual commands**, preserving the 122 existing
  cases and adding 11 benchmark cases. These cover real tiny/MNIST files,
  non-Softmax output, escaped filenames, leading zeros, argument rejection,
  missing files, arithmetic failure during warm-up, and recovery with valid
  input. The MNIST runs use saved indices 0, 8, and 99. Reports must have the
  exact structure, finite positive ordered statistics, and reciprocal mean/rate
  consistency within display rounding. No speed threshold is imposed.
- Existing controlled loop/timer checks and the real setup/reuse audit continue
  to check measured boundaries, zero/overflow rejection, failure precedence,
  buffer reuse, allocation/file counters, and numerical agreement.

Local macOS arm64 checks (Apple Clang 21.0.0, CMake 4.4.4):

| Check | Result |
| --- | --- |
| Debug with optional Python reference checks | 44/44 passed |
| Release, ordinary C tests only | 35/35 passed |
| Debug ASan + UBSan, ordinary C tests only | 35/35 passed; no findings |
| Native `leaks --atExit`, CLI audit and actual MNIST benchmark | Both exited 0; 0 leaks / 0 leaked bytes |
| New/changed C sources with C11 warnings as errors | Passed |

ASan used `detect_leaks=0` on macOS; native leak checks completed separately,
with the usual restricted-process inspection notice. The 100 saved MNIST
prediction/top-three comparisons still agree 100/100, with maximum displayed
score absolute error `3.57627869e-7`. All-stage comparison and lifecycle tests
also pass. Classification accuracy is not a benchmark statistic.

From the repository root, reproduce a report with:

```sh
cmake -S . -B build-mnist-c-only -DCMAKE_BUILD_TYPE=Release
cmake --build build-mnist-c-only
./build-mnist-c-only/cinder bench examples/mnist/mnist.cnd \
  tests/fixtures/mnist/inputs/00000.f32 --iterations 1000
```

Actual output from one local functional run on October 28:

```text
Model: mnist.cnd
Iterations: 1000
Warm-up iterations: 10
Clock: monotonic
Clock resolution (ns): 42
Mean latency (us): 112.477741
Minimum latency (us): 98.417
Maximum latency (us): 166.292
Throughput (inferences/s): 8890.64797
```

These are observed values from the current uncommitted implementation, not a
reproducible performance baseline or speed guarantee. Other verification work
was running concurrently. A baseline still needs isolated repeated trials,
exact revision/environment/build/model/input records, and the existing numerical
and allocation evidence. Linux execution and hosted macOS/Linux CI have not
been verified for this step; no commit or push was made.

The next step at the time of this statistics report was the consolidated
acceptance audit; its completed local results follow. Phase 5 remains open.

## Benchmark acceptance audit

**Completed locally October 28, 2026.** This step adds tests and documentation;
production source and the command/measurement contract are unchanged.
[`tests/test_benchmark_acceptance.c`](../tests/test_benchmark_acceptance.c) closes
the integration gap between the existing component tests: a clock or inference
failure now travels through the actual command, execution, and cleanup code in
one test, with real model ownership.

The audit compiles the real tensor, Dense, layer, model, loader, input reader,
CLI session, timer, benchmark loop, statistics, and CLI sources with test-only
wrappers. It reads the committed tiny/MNIST models and saved input/probability
fixtures. No PyTorch, NumPy installation, downloads, or production test hooks
are needed. The wrappers supply OS clock responses, requested layer/output
faults, allocation/file failures, and captured stream failures. Normal calls
still execute the real implementation. Test capture resources are separate
from the tracked production allocations and file handles.

Two ordinary CTest cases are registered on the supported raw-clock hosts:

- `benchmark_controlled`: **1,088 command invocations** on this 64-bit Mac,
  covering both tiny and MNIST. The real timer consumes deterministic timestamps;
  inference itself runs the actual loaded models. Every successful model call
  is checked against saved probabilities using shape/finite checks and
  `atol = rtol = 1e-5`, before any intentionally injected output corruption.
- `benchmark_unavailable`: compile the same sources with the clock backend
  disabled. For each model, setup succeeds, the real timer returns `UNSUPPORTED`,
  the command runs no inference, stdout stays empty, and cleanup releases all
  23 successful setup allocations. This exercises the compiled fallback rather
  than substituting a preassembled benchmark error.

### What the integrated audit proves

**Arguments and counts.** Reject representative missing/extra arguments, empty
paths, wrong option positions, zero/above-limit/huge counts, signs, whitespace,
fractions, exponents, hexadecimal text, and non-ASCII digits before any model
load, file open, allocation, clock query, or inference. Accept the maximum count
at the parser, then intentionally fail the first file open so CI does not run
one million real MNIST inferences. The existing cheap `benchmark_loop` audit
executes the actual loop at the maximum count. Successful integrated N=1 and
N=3 runs require exactly N+10 model calls, one resolution query, 2N clock reads,
one model load/input read, and 23 total setup allocations.

**Warm-up and timing exclusion.** The test clock charges one simulated second
for each setup allocation, each warm-up, and each observed output element,
plus 100 seconds for clock setup. Measured calls advance it by exactly 1,000,
2,000, and 3,000 ns. The complete command must still print the exact nine-line
report with mean/minimum/maximum **2/1/3 us** and **500,000 inferences/s**.
Timestamp counters require zero start/end reads during warm-up; the observer
checks output order and prohibits observation between start and end. Timestamp
values cross a second boundary. Separate runs retain a tiny elapsed duration
at very large uptime and accept positive intervals below the 42 ns simulated
resolution. Unsorted `[3000,1000,2000]` ns intervals must retain the same extrema,
catching accidental first/last-sample reporting. `[1,1,2]` ns must print a fractional mean and 750,000,000/s.
These numbers are synthetic correctness checks, not measured performance.

**Timing failures.** Fail the resolution query; supply zero/negative/malformed
resolution; and force resolution conversion overflow. Fail or corrupt every
start/end read across three measured calls. Exercise zero duration, backward
end timestamps, a start before the previous end, interval conversion overflow,
and accumulated-total overflow. The command must stop at the failing call,
retain the exact context/message, publish no summary, and clean up. Independent
successful invocations verify recovery after the clock-failure cases.

**Inference and error precedence.** Fail each of the four layers at each of
10 warm-ups and three measured calls, after deliberately writing part of its
private workspace. Check that subsequent layers/calls never run, caller output
is unchanged by that failed model call, the end timestamp is still attempted
for a measured inference failure, and the CLI identifies the correct one-based
warm-up/iteration. Inject NaN, positive infinity, and negative infinity at every
output position and call position. Simultaneous-failure cases require:

1. Layer failure before end-clock failure.
2. Invalid elapsed time before a non-finite output.
3. Non-finite output before accumulated-total overflow.

**Ownership and output failure.** Force each of the 23 real construction/input/
output allocation points to fail, and fail each model/input open or close.
Every invocation requires zero retained tracked blocks, bytes, and input/model
streams; each successful allocation has exactly one free. Model/input/output
addresses stay fixed across calls, input bytes stay unchanged, and allocations,
frees (including `free(NULL)` attempts), and tracked file operations are
prohibited during execution. All CLI writes require resources to have been
released first. Fail each stdout write, including partial multi-byte writes;
fail the final flush; and combine flush failure with unavailable stderr.
Commands still return failure, stop attempting report writes, and retain no
ownership. Before reporting, failures leave stdout empty; partial stdout after
a write failure remains permitted by the contract.

### Coverage map and CI policy

The integration audit supplements the existing focused tests rather than
replacing their more exhaustive boundary checks:

| Contract area | Ordinary CTest coverage |
| --- | --- |
| Exact syntax/reasons, precedence, literal paths, leading zeros, upper limit | `cli_audit`, `cli_process`, `benchmark_controlled` |
| Exact warm-up/measured counts through 1,000,000; observation outside timing | `benchmark_loop`, `benchmark_controlled` |
| Known statistics, fractional means, uint64 boundaries, zero denominators | `benchmark_statistics`, `cli_audit`, `benchmark_controlled` |
| Clock arithmetic/state, all timing failure categories, missing backend | `timer_checked`, `timer_unavailable`, `benchmark_controlled`, `benchmark_unavailable` |
| Layer/inference errors, non-finite output, first-error precedence, recovery | `benchmark_loop`, `benchmark_controlled`, `cli_setup` |
| Write/flush errors, bounded diagnostics, report formatting, cleanup | `cli_audit`, `benchmark_controlled` |
| Actual linked binary, real clock, tiny/MNIST numerical regressions | `cli_process`, `timer_native`, `cli_setup`, existing model/MNIST tests |

[`CMakeLists.txt`](../CMakeLists.txt) labels the twelve benchmark-related tests
`benchmark`, including the two new cases. The existing macOS/Linux Debug/Release
and Linux sanitizer CI jobs already run the full CTest suite, so registration
includes these cases without another workflow job. Tests assert counts,
structure, numerical correctness, error behavior, and controlled arithmetic;
there is **no maximum latency or minimum throughput requirement**. Native-clock
checks likewise do not impose a machine speed threshold. The new cases have
120-second hang timeouts, separate from any benchmark-result assertion.

Run the focused group after configuring and building:

```sh
ctest --test-dir build-mnist -L benchmark --output-on-failure
```

For a fresh ordinary C build, without optional Python dependencies:

```sh
cmake -S . -B build-benchmark-acceptance -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON
cmake --build build-benchmark-acceptance --parallel
ctest --test-dir build-benchmark-acceptance -L benchmark --output-on-failure
```

Local macOS arm64 results, Apple Clang 21.0.0 and CMake 4.4.4:

| Check | Result |
| --- | --- |
| Full Debug, including optional Python reference checks | 46/46 passed |
| Full Release, ordinary C tests | 37/37 passed |
| Full Debug ASan + UBSan, ordinary C tests | 37/37 passed; no findings |
| Final new-test rerun after adding precedence and unsorted-duration cases | 2/2 passed in each build |
| Native `leaks --atExit`, controlled/unavailable audits | Both exited 0; 0 leaks / 0 leaked bytes |
| Both compile-time backend variants, C11 warnings as errors | Passed |

The full suites include the twelve labeled tests and all existing numerical,
loader, and lifecycle regressions. ASan used `detect_leaks=0` on macOS; separate
native leak checks completed with the usual restricted-process inspection
notice. The native leak commands are:

```sh
leaks --atExit -- ./build-mnist/test_benchmark_controlled "$PWD"
leaks --atExit -- ./build-mnist/test_benchmark_unavailable "$PWD"
```

At the local acceptance step, Linux execution and hosted CI had not yet been
verified. Subsequent [M5 verification](M5_VERIFICATION.md#hosted-ci-evidence-and-closure)
confirms all five jobs passed for `cd461c6`, including these tests. Controlled
times establish correctness, not a speed improvement. The Release baseline
and completed evaluation follow.


## First Release baseline

The [October 28 Apple M4 report](benchmarks/2026-10-28-apple-m4/README.md) measures
clean commit `cd461c696aea932e6ea53e4093008d6463c42fa9` with a fresh Release build,
`-O3 -DNDEBUG`, sanitizers/LTO off, ten warm-ups, and 100,000 measured iterations
requested per attempt. It retains all five attempts per model: four valid tiny
reports and all five MNIST reports, plus tiny trial 1's zero-interval failure at
iteration 37,270. No failed attempt was replaced and no slow results were trimmed.

Median valid-trial means are 0.159771025 us for tiny and 105.936159 us for MNIST.
The report preserves every result, source/binary/fixture hashes, environment,
power/background-load observations, commands, and allocation/numerical evidence.
The same Release build passed 37/37 tests; audited inference paths make zero
allocation/free attempts. Tiny timing is close to the clock's granularity and
varies substantially, so it needs further timing investigation before drawing
optimization conclusions. This baseline alone did not close Phase 5 and does
not establish a Linux performance result.

## Evaluation and M5 closure

The [October 29 closure report](M5_VERIFICATION.md) retains the existing runtime
after evaluating the evidence. An identical-executable control on October 28
produced median valid-trial means of 0.15861274 us (tiny) and 104.596974 us
(MNIST), versus 0.159771025 us and 105.936159 us originally. These are changes
without an optimization, not speedup claims. All ten repeat attempts are saved,
including a tiny zero-interval failure at iteration 9,853.

Numerical/shape/finite checks and allocation/lifecycle audits passed again, as
did local Debug 46/46, Release and ASan/UBSan 37/37 each, and four native leak
checks. Existing hosted run 40688919176 passed all five macOS/Linux jobs at
exact compiled-source revision `cd461c6`; runtime, tests, CMake, and workflow
are unchanged. The report records raw evidence, reproduction, CI provenance,
GCC warnings, timing/environment limits, and the Phase 6 handoff. **M5 is closed.**
