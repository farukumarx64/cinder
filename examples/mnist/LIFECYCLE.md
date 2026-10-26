# MNIST lifecycle and CI verification

**Local checks passed October 26, 2026. Hosted verification of the new lifecycle
tests and expanded workflow is pending publication.** M4 remains open because
the input-file reader and `cinder run` implementation are still outstanding.

This step adds two CTest executables from
[`test_mnist_lifecycle.c`](../../tests/test_mnist_lifecycle.c), with test-only
instrumentation in [`mnist_lifecycle_audit.h`](../../tests/mnist_lifecycle_audit.h).
Production C sources, public APIs, the checkpoint, exported model, and all
reference fixtures remain unchanged.

## Public API lifecycle checks

`mnist_lifecycle` links the normal `cinder_core` library and uses the committed
407,160-byte model, all 100 saved 784-value inputs, and the saved ten-value
probability vectors. It performs **62 successful loads, 20 rejected loads,
500 success/repeat/failure/recovery cycles, and 2,100 inference calls**.

Each cycle:

1. Runs one saved input and compares with Python using matching shapes,
   finite-value checks, and `atol = rtol = 1e-5`.
2. Immediately repeats the same input and requires bit-identical C output.
3. Attempts an invalid call and requires the entire caller output, including
   guards and spare capacity, to remain unchanged.
4. Runs a different saved input successfully on the same model, without reset,
   comparing it with both Python and its previously recorded C result.

The failures cover input lengths `783` and `785` instead of `784`, output
capacity `9` instead of `10`, NaN and infinity in the last input element, and
finite extreme inputs causing actual Dense1 accumulation overflow. Error
objects are reused across failure and recovery, including calls with omitted
diagnostics. Success must clear earlier diagnostics. Every successful call
preserves its input and spare output capacity.

Before these cycles, the test records one C baseline for every image. These
baselines stay read-only afterward. Bit equality is required for repeated
execution within one executable; comparisons with Python retain the numerical
tolerance. The test does not require different platforms to produce identical
float bits. These are repeated runs of 100 unique images, not a larger accuracy
evaluation set; [accuracy and agreement remain separate](VALIDATION.md).

Two loaded models are exercised in an interleaved order and concurrently on
separate threads. After one is destroyed, the other continues to work. Another
thread pair shares one model under a caller-owned mutex. It never deliberately
violates the contract's one-active-call restriction.

Twenty load/recovery rounds use a private scratch file. Each loads two owners,
then corrupts either the final layer type, final record length, final Dense
bias (infinity), or format version. Rejection must clear the returned model
slot while previously loaded models remain usable. The source file is removed,
one owner is freed, and the surviving owner still runs correctly. Restoring the
file permits a new load/run/free cycle immediately.

## Allocation, ownership, and partial-write audit

`mnist_lifecycle_audit` compiles the same lifecycle source with the actual
runtime sources included under test-only allocation and loader-I/O wrappers.
It does not link a second runtime copy. This follows the existing ownership
audit approach and requires no production instrumentation or tracing API.
The tracked test is single-threaded; concurrency uses the normally linked test.

It performs **104 successful loads, 62 rejected loads, 300 cycles, and 1,384
inference calls**. In addition to the public failure cases, it verifies:

- Each model owns **two distinct buffers of 784 floats / 3,136 bytes each**.
  Capacity includes the input length, so it is larger than the 128-value hidden
  layer. Separate models also have distinct layer objects and parameter arrays.
- Both workspaces are filled with NaNs before each cycle and before its failed
  call. Recovery runs on the failed state without clearing it. Correct outputs
  and an untouched poisoned tail in the first Dense output buffer show that
  later runs do not depend on leftover values outside the logical vector.
- All **101,770 learned parameters** remain byte-identical to values decoded
  from the committed model before and after each cycle.
- Every inference cycle leaves allocation, free-attempt, and loader
  open/read/close counters unchanged. The next allocation is armed to fail.
  Even an unexpected `free(NULL)` changes the free-attempt counter and fails.
- A test-only dispatch wrapper injects a failure after a partial workspace
  write at each of the four layers. Dispatch stops at that layer, caller output
  stays unchanged, and the next valid call succeeds. These injected failures
  complement the real overflow case; they are not claims that valid ReLU or
  Softmax inputs naturally fail.
- Every one of the **21 allocation points** in loading/construction fails
  individually, with and without an error object: **42 forced failures**. Each
  failure releases all new allocations and closes its stream while preserving
  another live model's pointers, sizes, parameter bytes, and usability. Each
  failed load is followed by a successful load/run/free retry.
- Final destruction leaves **zero tracked blocks, zero tracked bytes, and no
  open loader stream**. Freed memory is poisoned to expose borrowed staging.

Counts cover allocation and I/O calls made by the Cinder sources under these
wrappers. They do not instrument libc, libm, pthread, or operating-system
internals. Native leak checking and sanitizers provide separate checks.

## Local verification

Environment: macOS 27.0.1 arm64, Apple Clang 21.0.0, CMake 4.4.4. Optional
Debug suites used Python 3.12.14, NumPy 2.3.5, and PyTorch 2.14.1.

| Check | Actual result |
| --- | --- |
| Full Debug suite, all optional Python suites enabled | **35/35 passed** |
| Full Release suite, all Python options OFF | **26/26 passed** |
| Full Debug ASan + UBSan suite, all Python options OFF | **26/26 passed**, no findings |
| Final lifecycle fixture-layout guards, both tests rebuilt in all three configurations | **2/2 passed in each** |
| Native macOS `leaks --atExit`, public lifecycle executable | **0 leaks / 0 leaked bytes** |
| Native macOS `leaks --atExit`, instrumented lifecycle executable | **0 leaks / 0 leaked bytes** |
| Both compilation modes with C11, `-Wall -Wextra -Wpedantic -Werror` | Passed |

Sanitizers used `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. This Mac does not support
LeakSanitizer. Native `leaks` ran outside the execution sandbox to inspect its
child processes; it emitted the standard restricted-process memory-display
notice and exited successfully for both programs.

## Hosted CI

The preceding numerical-validation revision
`42b55f43e37a7c9990d958459639d5b14552a826` passed all three existing jobs in
[run 40318184929](https://github.com/farukumarx64/cinder/actions/runs/40318184929).
That is baseline evidence, before these lifecycle additions.

The updated [workflow](../../.github/workflows/ci.yml) defines five jobs:

| Configuration | Tests and additional checks |
| --- | --- |
| Ubuntu 24.04 Debug | C tests plus optional NumPy/exporter/offline MNIST suites |
| Ubuntu 24.04 Release | Ordinary C tests, using committed fixtures |
| macOS 15 Debug | Ordinary C tests, using committed fixtures |
| macOS 15 Release | Ordinary C tests plus native leak checks for both MNIST lifecycle executables |
| Ubuntu 24.04 Clang sanitizers | Ordinary C tests under ASan + UBSan + LeakSanitizer |

Only Linux Debug installs Python dependencies. Ordinary C tests never download
MNIST, regenerate fixtures, train a model, or invoke Python. Verbose CTest logs
retain lifecycle counts and numerical-comparison details from passing tests.
Linux sanitizer settings keep `detect_leaks=1` and halt on the first finding.
**Results for the new revision remain pending until it is committed and pushed.**

## Reproduce and limits

From the repository root, an ordinary C-only build is sufficient:

```sh
cmake -S . -B build-mnist-lifecycle -DCMAKE_BUILD_TYPE=Debug
cmake --build build-mnist-lifecycle --parallel
ctest --test-dir build-mnist-lifecycle -R '^mnist_lifecycle' -V
```

Use a separate build directory with `-DCMAKE_BUILD_TYPE=Release` for Release,
or `-DCINDER_ENABLE_ASAN=ON -DCINDER_ENABLE_UBSAN=ON` for sanitizers.
Run the whole suite by removing the `-R` filter. The native leak commands are
shown in the workflow; scratch files stay in each build directory.

This verifies the committed model and exercised inputs, errors, allocation
points, and synchronization patterns. It does not prove every possible thread
schedule; ThreadSanitizer was not run. Same-model concurrency still requires
caller serialization, and destruction must wait for all users. Big-endian and
other unsupported host representations were not exercised. CLI file handling
and command-level cleanup remain part of the upcoming `cinder run` work.
