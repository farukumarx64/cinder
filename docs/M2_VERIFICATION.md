# M2 lifecycle verification — October 23, 2026

**M2 is complete as of October 23, 2026. Local verification and all three
hosted CI jobs passed with the NumPy network fixtures and lifecycle additions.**

The verified revision is `f5424a4a3639cf5c357fccb6ab5ad8f6cf73477a`, which includes
the NumPy commit `19dd4a1acac79e5e3a8a0b261e94614db01d7258`.
[GitHub Actions run 39984963408](https://github.com/farukumarx64/cinder/actions/runs/39984963408)
confirms the published revision. This verification adds tests, test-build
configuration, and documentation; it changes no runtime source or public API.

## Lifecycle behavior

[`test_model_lifecycle.c`](../tests/test_model_lifecycle.c) links the production
`cinder_core` library and uses the saved `2 -> 4 -> 2` NumPy network fixtures.
Each cycle performs three calls:

1. Successful inference, checked against the corresponding NumPy probabilities.
2. A failed call, checking that every byte of the preceding output is preserved.
3. Successful inference with a different input, reusing the same model, output
   array, and error object without resetting working buffers.

Failures rotate among wrong input length, insufficient output capacity,
non-finite input, and actual Dense arithmetic overflow from finite `FLT_MAX`
inputs. Successful recovery clears the diagnostic. All ten fixture inputs are
used with every failure kind. Comparison uses the shared shape/finite checks
and unchanged `atol = rtol = 1e-5`; input, spare output, and dimensions are checked.

The executable performs **4,120 cycles / 12,360 model calls**:

- 80 cycles interleaved across two separately constructed full models.
- 2,000 cycles across two threads, each owning a different model and using
  separate caller input/output/error storage.
- 40 cycles on the surviving model after the other model is destroyed.
- 2,000 cycles across two threads sharing one model, with a caller-owned mutex
  held around each cycle, serializing every complete model call.

A condition-variable gate starts each worker pair after both workers arrive.
Models are freed only after the threads join. There is no intentional same-model
data race. POSIX Threads is linked only to this test on the supported Unix
platforms (macOS/Linux); the runtime does not gain a threading dependency,
internal lock, or parallel layer execution.

## Working buffers, allocations, and failed construction

[`test_model_alloc_failures.c`](../tests/test_model_alloc_failures.c) compiles
the actual model/layer/Dense/tensor sources with test-only allocator and
dispatch substitutions. These hooks do not appear in the production library.

The existing storage checks now compare results **byte-for-byte with baseline
runs** after both private buffers have been filled with NaNs. This checks the
actual result, beyond probability range or normalization alone. Tests inject
partial-write failures at all four layer positions and immediately retry without
clearing the buffers; results must still equal the baseline. Unused buffer
tails remain poisoned, and observed counts confirm that shrinking vectors never
consume them. Dense alternates buffers; activations use exact in-place storage.

Allocation counters verify zero Cinder `malloc`/`calloc` calls during successful
inference, argument rejection, injected failure, real arithmetic failure, and
recovery. The next allocation is armed to fail. Free accounting now counts
**every attempted free, including `free(NULL)`**, in addition to successful
releases; inference attempts none. These counters measure the Cinder sources,
not allocations performed by the thread harness or system libraries.

Every one of the four model-construction allocations is forced to fail, with
and without optional error storage. Partial model storage is released, caller
layers remain owned and usable by the caller, and both abandonment and retry
paths finish with zero tracked live blocks. Retried construction is now followed
by real inference before destruction. Existing tensor, Dense, and common-layer
tests separately cover their allocation failures and nested cleanup.

Existing `model_run`, `model_allocation`, and NumPy tests also cover distinct
inputs, early/middle/late numerical errors, exact/spare capacity, independent
owners, all-or-nothing ownership transfer, layer ordering, unchanged parameters,
and the full network's intermediate/final results.

## Local results

Verified on macOS 27.0.1, arm64, Apple Clang 21.0.0, and CMake 4.4.4. Fresh
reference builds use the installed Python 3.12.14 / NumPy 2.3.5 environment.

| Check | Result |
| --- | --- |
| Debug, saved references | 19/19 CTest tests passed. |
| Release, saved and freshly generated references | 21/21 passed. |
| Debug, AddressSanitizer + UndefinedBehaviorSanitizer, fresh references | 21/21 passed, with UBSan configured to stop on errors. |
| Native `leaks --atExit -- ./build/test_model_lifecycle` | Exit 0; zero leaks / zero leaked bytes. |
| Native `leaks --atExit -- ./build/test_model_alloc_failures` | Exit 0; zero leaks / zero leaked bytes. |
| Release build with `BUILD_TESTING=OFF` | Built, registered zero tests, and printed `Cinder 0.1.0-dev`. |

All test assertions stay active in Release. No tolerance or runtime changes
were needed. The saved and generated network comparisons are described in
[NumPy verification](NUMPY_TESTS.md#complete-tiny-network).

To reproduce the ordinary build and lifecycle checks:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ctest --test-dir build -R 'model_(lifecycle|allocation_failures|run)' --output-on-failure
```

See [the README](../README.md#sanitizer-builds) for sanitizer configuration and
[the reference guide](NUMPY_TESTS.md#two-ways-to-run-the-same-comparison) for
fresh NumPy generation. On this Apple toolchain, native `leaks` is the supported
leak check; Linux CI uses LeakSanitizer. Native `leaks` reported its usual
restricted-process memory-display notice but completed successfully.

## Hosted results actually inspected

[GitHub Actions run 39984963408](https://github.com/farukumarx64/cinder/actions/runs/39984963408)
completed successfully on October 23, 2026, for
`f5424a4a3639cf5c357fccb6ab5ad8f6cf73477a`. Job statuses, checked-out commit,
and test logs were inspected:

| Job | Toolchain / checks | Actual result |
| --- | --- | --- |
| macOS 15 | Apple Clang 17.0.0, CMake 4.4.3 | 19/19 tests passed. |
| Ubuntu 24.04 | GCC 13.3.0, CMake 3.31.6, NumPy 2.3.5 fresh references | 21/21 passed. |
| Ubuntu 24.04 sanitizers | Clang 18.1.3, ASan + UBSan + LeakSanitizer | 19/19 passed. |

The sanitizer log confirms `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Configuration, compilation,
and testing succeeded in all three jobs, with no reported sanitizer or leak
findings. The logs explicitly show `model_lifecycle`, `model_allocation_failures`,
and `model_numpy_fixtures` passing on each platform; `model_numpy_generated`
also passes in the ordinary Linux job. The two extra Linux tests compare freshly
generated tensor and network references in addition to the saved fixtures.

The earlier successful run for `f6480d5` was only a baseline; the run above
includes both the NumPy and lifecycle additions and supplies the M2 closure
evidence. The existing CI workflow required no changes.

## Milestone closure and limitations

The lifecycle, allocation, cleanup, numerical-reference, sanitizer, supported
leak, and macOS/Linux checks required for M2 have passed. The next scoped step
is Phase 3's `.cnd` model-format contract in `docs/MODEL_FORMAT.md`, before
implementing the exporter or loader.

- One model still permits only one active call. Callers must serialize shared
  access and coordinate destruction; there is no internal concurrency detection.
  Separate models own separate parameters and workspaces.
- The thread tests exercise valid synchronization patterns under ASan/UBSan;
  ThreadSanitizer was not run, and these tests are not proof of every schedule.
- Pointer validity, advertised accessible buffer extents, forbidden overlap,
  and object lifetime remain caller preconditions. The runtime cannot validate
  arbitrary dangling pointers or repair externally corrupted objects.
- This milestone covers fixed-shape, single-vector in-memory inference. The
  `.cnd` format, exporter/loader round trip, MNIST, and inference CLI remain later
  work. Full-network NumPy fixtures are retained for that round trip.
