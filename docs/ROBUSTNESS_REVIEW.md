# Robustness coverage review — October 30, 2026

**The scoped review is complete locally. Two coverage gaps are filled; no
production defect was exposed.** Runtime code, public contracts, and the `.cnd`
format remain unchanged. This step left M6 open for final verification.
Subsequent [M6 verification](M6_VERIFICATION.md) passed all local checks and
five hosted jobs at `cace0df`, closing the phase on October 30.

This review starts at `ec2b69c` and tests the working-tree additions below.
[Verification excerpts and SHA-256 hashes](verification/2026-10-30-robustness.txt)
identify the tested CMake/test contents independently of a future commit.

## Existing coverage reviewed

| Requirement | Existing checks | Disposition |
| --- | --- | --- |
| Malformed model files | `test_model_load.c`: magic/version/type, dimensions, connections, payload lengths, truncation, trailing bytes, non-finite parameters, and resource boundaries. 709 public rejection cases, each with and without error details. | Retain. Small-file faults did not exercise failure across the parameter decoder's chunk boundary; add the cases below. |
| Allocation failure and partial construction | Tensor, Dense, layer, and model allocation audits; `test_loader_failures.c` forces all 21 tiny-load allocations, including staging, parameter copies, and model buffers. `test_cli_setup.c` covers setup ownership through model/input/output allocation. | Retain the existing ownership audits; also force every allocation in the new multi-chunk model. |
| File I/O and error precedence | Loader audit injects open/seek/tell/read/EOF/close faults, preserving the first failure if close also fails. `test_input.c` checks binary32 inputs, exact lengths, non-finite values, short reads, final EOF, allocation failure, and cleanup. | Add short-read faults after a successful decoding chunk; preserve existing error classifications. |
| Shapes, capacities, and overlap rules | Tensor, Dense, layer, and model tests check invalid lengths, insufficient capacity, spare-output sentinels, unchanged sources, and permitted in-place activations. | Sentinel values alone do not reliably reveal extra reads. Add protected caller buffers. |
| Cleanup and recovery | Loader checks no new live allocations, NULL output on failure, and closed streams. Model/lifecycle tests poison workspaces, keep independent models alive during failures, and retry after invalid or overflowing inference. | Extend failure/recovery coverage at the new boundaries; retain existing lifecycle and allocation-count checks. |
| CLI and benchmark failure paths | `test_cli.c`, `test_cli_setup.c`, process tests, and controlled benchmark tests cover arguments, output failures, timing/inference failures, and cleanup without speed requirements. | Existing focused coverage is retained and rerun in the full suite. |

This is a mapping of the exercised contract, not a claim that all possible
malformed inputs or implementation paths have been exhausted.

## Loader decoding boundaries

[`test_loader_failures.c`](../tests/test_loader_failures.c) now generates a
10,348-byte `2 → 513 → 2` model in its existing scratch directory. Its sequence
is Dense → ReLU → Dense → ReLU; all parameters and expected results are
deterministic. It requires no saved large fixture, Python, network, or MNIST data.

The loader decodes at most 4,096 bytes (1,024 floats) per parameter read:

| Payload | Starts at byte | Floats | Reads |
| --- | ---: | ---: | --- |
| First Dense: 1,026 weights + 513 biases | 32 | 1,539 | 1,024 + 515 floats |
| Second Dense: 1,026 weights + 2 biases | 6,220 | 1,028 | 1,024 + 4 floats |

The test compares all **2,567 decoded parameter bit patterns**, including
values not needed by the known-output calculation. This detects skipped,
shifted, or repeated parameter chunks. For input `[2, -1]`, inference must
produce `[3.75, 0]` after the final ReLU.

It adds **162 rejected calls**, all followed by successful loading, inference,
and destruction of the valid model:

- **80 non-finite cases:** positive/negative infinity, quiet NaN, and signaling
  NaN at indices 1,023, 1,024, 1,025, 1,026, and the last parameter of each
  Dense payload, with and without diagnostics. These span the chunk boundary,
  the transition from weights to biases, and the final element. When details
  are requested, the reported layer index and byte offset must be correct.
- **16 physical truncations:** just before, exactly at, and just after each
  4 KiB boundary, plus the last payload byte, in both diagnostic modes.
- **24 injected short reads:** fail each of the four parameter reads with an
  I/O error, EOF, or neither flag, in both modes. This reaches the decoding loop
  even when the file's initially measured length is sufficient. The recorded
  offset must include the bytes actually obtained; EOF maps to `FORMAT`, while
  a read error or unexplained short read maps to `IO`.
- **42 allocation failures:** fail each of the 21 construction allocations,
  in both modes. This includes partially copied Dense parameters and failure
  to allocate the final model's reusable buffers.

Every failure uses the existing allocation/stream audit: no published model,
no retained stream, no new live blocks, and balanced cleanup. Successful
recovery also confirms no allocations or frees occur during inference.
The complete loader audit now checks **1,766 rejected calls**, up from 1,604.
The original 709 public cases remain unchanged; these counts measure different
things because the audit includes diagnostic variants and injected faults.

## Protected caller buffers

[`test_buffer_boundaries.c`](../tests/test_buffer_boundaries.c) adds the
`buffer_boundaries` CTest on macOS and Linux. Each buffer uses three mapped
pages: the outer pages are inaccessible and the middle page holds the values.
The checks run twice, with vectors placed at opposite ends of the middle page.
An access crossing the adjacent protected edge fails the process, whether it
is a read or a write. This supplements the existing spare-capacity sentinels.

Inputs and constructor parameters are read-only. After constructing Dense
owners, the test revokes all access to the original parameters while keeping
their addresses reserved. Forward execution must therefore use the owned
copies; an accidentally retained caller pointer cannot be hidden by address
reuse. All mappings are explicitly released, with `munmap` results checked.

The test exercises:

- Matrix–vector multiplication, bias addition, ReLU, and Softmax with exact
  caller-buffer lengths.
- Standalone Dense, common Dense dispatch, and a complete Dense → ReLU model.
- Separate and exact in-place ReLU/Softmax layer execution.
- Actually smaller destination buffers, invalid input lengths, and a
  non-finite last input, with the promised output preservation.
- Finite inputs that overflow during model execution, unchanged caller output,
  and successful subsequent execution.

All advertised buffer extents are accessible. Objects come from the public
constructors; the test does not fabricate tensor metadata or violate overlap
preconditions. Its assertions remain active in Release builds. CMake keeps
the POSIX feature macros local to this test target; production compilation
and APIs are unaffected.

## Actual verification

All builds used C11, `-Wall -Wextra -Wpedantic -Werror`; each configured, built,
and tested successfully with **zero compiler warnings**.

| Environment | Result |
| --- | --- |
| macOS 27.0.1 arm64, Apple Clang 21, Debug | **47/47**, including nine optional Python/NumPy/exporter/offline MNIST checks |
| Same host, Release (`-O3 -DNDEBUG`) | **38/38**, C tests only |
| Same host, Debug + ASan + UBSan | **38/38**, no sanitizer report |
| Ubuntu 24.04.5 amd64, GCC 13.3, Debug | **38/38**, C tests only |
| Same Linux environment, Release | **38/38**, C tests only |
| Native macOS `leaks --atExit` | Loader failure audit, protected-buffer test, and input-reader audit each exit 0 and report **0 leaks / 0 bytes** |

Linux ran offline in the existing `cinder-gcc13-audit:20261029` Docker image,
under amd64 emulation on Apple M4; these are correctness checks, not native
Linux timing measurements. macOS used CMake 4.4.4; Linux used 3.28.3.
ASan used `detect_leaks=0:halt_on_error=1`; UBSan used
`halt_on_error=1:print_stacktrace=1`. Apple LeakSanitizer is unavailable in this
setup. Native `leaks` emitted its restricted-process inspection notice but
completed all three analyses successfully. Heap leak reports do not by
themselves prove mappings were released; the test checks each unmap directly.

The full suites retain numerical shape/finite/tolerance checks and lifecycle
audits. MNIST compares **27,600 saved stage values** under `atol = rtol = 1e-5`
and retains **100/100 C/Python prediction agreement**. This is runtime agreement,
not a new classification-accuracy measurement. Existing allocation audits still
report allocation-free inference. Ordinary C tests use committed references
and the locally generated synthetic file, without Python or downloads.

## Reproduce

Run from the repository root on macOS or Linux:

```sh
cmake -S . -B build-robustness -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_FLAGS=-Werror
cmake --build build-robustness --parallel 4
ctest --test-dir build-robustness --output-on-failure --no-tests=error --verbose

# Focused additions:
ctest --test-dir build-robustness -R '^(loader_failures|buffer_boundaries)$' --verbose

# Release assertions and protected pages remain active:
cmake -S . -B build-robustness-release -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS=-Werror
cmake --build build-robustness-release --parallel 4
ctest --test-dir build-robustness-release --output-on-failure --no-tests=error

# Optional Python checks use the existing documented development environment:
cmake -S . -B build-robustness -DCINDER_ENABLE_NUMPY_TESTS=ON \
  -DCINDER_ENABLE_EXPORTER_TESTS=ON -DCINDER_ENABLE_MNIST_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-robustness --parallel 4
ctest --test-dir build-robustness --output-on-failure --no-tests=error
```

The Linux image names its compiler `/usr/bin/gcc-13`; select it with
`-DCMAKE_C_COMPILER=/usr/bin/gcc-13`. The fresh verification directories/logs
are under ignored `build-robustness-20261030/`. The saved evidence file contains
the sanitizer environment, test summaries, source hashes, and leak results.

## Limits and next gate

Guard pages are enabled only on the supported macOS/Linux platforms and test
selected legal shapes at page edges; they are not an exhaustive memory-safety
proof. Allocation and I/O faults are deterministic test substitutions around
the real runtime, not tests of every possible kernel or filesystem failure.
No fuzzing campaign, native 32-bit run, or additional platform certification is
claimed. Existing overflow tests exercise checked arithmetic and lowered
ceilings without pretending to emulate another ABI.

Valid pointers and true accessible lengths, intact owning objects, permitted
buffer overlap, and serialized use of a shared model remain caller obligations.
The documented tiny-clock rejection policy also remains in force. No production
correction was warranted by the reviewed paths or new tests.

The existing CI workflow automatically includes both additions through CTest.
**This scoped review did not run hosted CI.** The additions were subsequently
committed as `cace0df`; [M6 verification](M6_VERIFICATION.md#hosted-ci) records
their passing macOS/Linux CI and final closure. Later release-candidate changes
still require appropriate checks.
