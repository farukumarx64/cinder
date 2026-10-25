# M3 lifecycle verification — October 25, 2026

**Local verification is complete. M3 remains open until hosted macOS/Linux CI
passes a revision containing both the round-trip and loaded-lifecycle tests.**

This step adds tests, CMake test registration, and documentation. No runtime
source, public API, `.cnd` format, fixture values, or numerical tolerance changed.
The complete [export/load round trip](ROUND_TRIP.md) remains part of the suite.

## Repeated loading, execution, and destruction

[`test_model_lifecycle.c`](../tests/test_model_lifecycle.c) now supports a loaded
model mode, registered as `model_loaded_lifecycle`. The original
`model_lifecycle` still constructs its models directly and runs unchanged cases.
Both tests link the production `cinder_core`; neither needs Python.

The loaded mode performs **502 successful loads, 100 rejected loads, and 4,720
success/failure/recovery cycles (14,160 inference calls)**:

- 100 serial file cycles each load two independent owners. After inference,
  the scratch file is changed to an unknown final layer type, unsupported
  version, non-finite final Dense bias, or missing path. Rejection must return
  the expected status and clear the output pointer without harming an existing
  model. Each rejection category is exercised with and without diagnostics.
- The first owner still runs after rejection. After it is freed and the source
  file is removed, the second owner also runs. Restoring the file permits a
  fresh load using the same diagnostic object, followed by inference after
  deleting the file again. All three owners are destroyed in each cycle.
- Two synchronized worker threads independently load, run, and destroy the
  same immutable source model 100 times each. Each load has its own stream,
  model, caller buffers, and diagnostic storage.
- Two further loaded models run the existing 4,120 lifecycle cycles: interleaved
  inference, concurrent independent models, continued use after destroying the
  other owner, and caller-mutex-serialized access to one shared model. Models
  are destroyed only after workers join.

Each inference cycle performs a successful call, a failed call, and immediate
successful recovery with a different input. Failures rotate through wrong input
length, insufficient output capacity, non-finite input, and actual arithmetic
overflow from finite inputs. The failed call must preserve every byte of the
preceding caller output. Recovery reuses model/output/error storage without
clearing private buffers and must clear the previous diagnostic.

All ten saved inputs are used. Successful outputs are compared to saved NumPy
probabilities using shape/finite checks and `atol = rtol = 1e-5`; input values,
spare output capacity, and exposed dimensions are also checked. Test checks
stay active in Release. Scratch files live in a dedicated build directory,
separate from the loader parser and allocation audit tests.

## Loaded-model storage and fault injection

[`test_loader_failures.c`](../tests/test_loader_failures.c) already compiles the
real loader/model/layer/Dense/tensor sources with test-only allocation and stdio
substitutions. The new lifecycle audit adds:

- Two loaded models with distinct layers, Dense parameter storage, and all four
  working-buffer addresses. Each model owns two buffers of four floats.
- **284 success/failure/recovery cycles (852 inference calls)**. Both private
  buffers are filled with NaNs before initial success and before the failing
  call. Recovery runs directly on the failed state. Saved NumPy outputs must
  still match, unused input-buffer tails stay poisoned, and all learned
  parameter bytes remain identical to the saved arrays.
- A fifth failure kind: finite inputs that complete Dense1 and ReLU, then
  overflow in Dense2. Private inspection confirms the earlier stages completed.
  This checks recovery after a later failure as well as an early Dense failure.
- Allocation counters stay unchanged throughout inference, with the next
  allocation armed to fail. Free-attempt counters include `free(NULL)` and also
  stay unchanged. Tracked loader open/read/close counts do not change.
- Each of the **21 allocation points** is forced to fail again, with and without
  diagnostics, while another loaded model remains alive: **42 failures**. Every
  failure releases all new allocations and preserves the original owner's
  blocks and bytes. That owner runs successfully immediately afterward; a new
  load/run/free retry also succeeds. Final destruction leaves zero tracked
  blocks/bytes and no retained file stream.

These counters cover Cinder source calls, not internal allocations in libc,
PyTorch, or the pthread harness. Private inspection and fault injection remain
confined to tests; no runtime instrumentation or public tracing API was added.

The earlier rejection matrix still exercises 709 public rejection cases with
and without diagnostics, all tiny-model allocation points, and 29 injected
stdio scenarios. The expanded tracked suite now checks **1,604 rejected load
calls** for rollback, preserved existing owners, and released streams. See
[the loader audit](LOADER.md#rejection-and-cleanup-audit--october-24-2026) for
the original coverage and resource-boundary cases.

## Local results

Verified on macOS 27.0.1, arm64, Apple Clang 21.0.0, and CMake 4.4.4. Optional
checks used Python 3.12.14, NumPy 2.3.5, and PyTorch 2.14.1.

| Configuration/check | Actual result |
| --- | --- |
| Fresh Debug, optional NumPy/exporter suites enabled | 28/28 CTest tests passed |
| Fresh Release, both optional Python suites disabled | 23/23 passed; no Python discovery or execution |
| Fresh Debug, ASan + UBSan, both optional Python suites enabled | 28/28 passed; no sanitizer findings |
| Native `leaks --atExit` on loaded lifecycle mode | Exit 0; zero leaks / zero leaked bytes |
| Native `leaks --atExit` on the expanded loader audit | Exit 0; zero leaks / zero leaked bytes |
| Both changed C test sources, C11 warnings as errors | Passed `-Wall -Wextra -Wpedantic -Werror` syntax checks |
| Release with `BUILD_TESTING=OFF` | Built; zero tests registered; `Cinder 0.1.0-dev` printed |

The sanitizer run used `ASAN_OPTIONS=halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Native `leaks` emitted the
usual restricted-process memory-display notice but completed successfully.
Apple Clang's runtime on this Mac does not support LeakSanitizer; Linux CI is
configured to enable it.

Both saved and fresh-export round-trip tests also passed in Debug and sanitizer
builds. Each compared 40 stage vectors / 120 values; maximum absolute error was
`5.96046448e-8`, with all nine deliberate fixture corruptions rejected. The
Release suite ran the saved C-only version. Fixture regeneration and all
51 Python validation/export unittest methods passed within the optional suite.

To run the ordinary loaded-model checks:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure --no-tests=error
ctest --test-dir build -R '^(model_loaded_lifecycle|loader_failures)$' --output-on-failure -V
```

Use separate build directories for Release and sanitizers. The
[README](../README.md#sanitizer-builds) documents sanitizer options, and
[ROUND_TRIP.md](ROUND_TRIP.md#two-ctest-paths) documents the optional live exporter
path. C-only configuration remains the default.

## Hosted verification and milestone status

The latest inspected successful run is the earlier
[run 40140441755](https://github.com/farukumarx64/cinder/actions/runs/40140441755)
for `312801d64791047170c5c5d1cda06077a45bca36`. Its success is a baseline only:
it predates round-trip commit `cc0b77f` and this lifecycle addition.

The existing workflow automatically includes the new CTest registration in
the macOS 15, Ubuntu 24.04, and Linux ASan/UBSan/LeakSanitizer jobs. The ordinary
Linux job additionally runs fresh NumPy and Python export/load comparisons.
No workflow changes are needed. Publication has been authorized. A new hosted
run containing both additions must pass before M3 can close; the results will
be recorded here after inspecting that revision's jobs and logs.

## Remaining limitations

- One model permits only one active inference call. Shared access must be
  serialized by the caller, and destruction must wait for all users. Separate
  models may run independently. Concurrent loads assume an unchanged source
  file; mutation during loading remains outside the contract.
- ThreadSanitizer was not run. The valid synchronization patterns exercised
  here do not prove correctness under every thread schedule.
- Big-endian execution, a different native ABI, and unsupported host float
  representations were not exercised. Lowered size ceilings in loader tests
  check overflow branches, not a complete simulated platform.
- Arbitrary invalid pointers, dangling objects, and inaccessible advertised
  buffer extents remain caller preconditions.
- M3 covers the fixed tiny sequential network, `.cnd` export/loading, numerical
  comparisons, and lifecycle verification. MNIST and the inference CLI remain
  later roadmap work.
