# v0.1.0 release-candidate verification — October 30, 2026

**Release-candidate verification is complete: local checks and all five hosted
CI jobs pass.** The
candidate is `1e5a5f657d962dd062e073432ca5d864fcfba04a`, whose executable reports
`Cinder 0.1.0`. This report does not authorize or announce a tag or publication.

## Checkout and documented setup

Verification started with a clean working tree. A new HTTPS clone of
`https://github.com/farukumarx64/cinder.git` was created under the ignored
`build-release-candidate-20261030/source/` directory. GitHub still held
`cace0df`, so the committed candidate was fetched from the local repository
and checked out detached. No uncommitted source, previous build directory,
fixture cache, or Python environment was copied into the checkout.

After the authorized push, a second plain HTTPS clone retrieved `1e5a5f6`
directly from GitHub. Its commit, complete tree, and README hashes match the
tested checkout, and both checkouts remain clean. This confirms the documented
clone command reaches the candidate without a local fetch.

The candidate tree is `a41ad662c812a3796c2d8204ceaab5e8dbd29358`. Its README
SHA-256 is `6a6ec7c8d0f98a4fc60fe04dd77f8cf67a2e8f2f6af6c20a37b7b67103591e1b`.
All 14 README shell blocks pass shell syntax checks and its local file links
resolve. Git, the compiler, CMake, and a suitable Python were already installed;
system/package-manager installation commands were not rerun.

The README configuration and test recipes use fresh build directories. Local
verification adds `-DCMAKE_C_FLAGS=-Werror`, limits build parallelism to four,
and enables verbose CTest output with `--no-tests=error`. CTest runs from the
build directory. These changes retain the documented build options and tests.

## Local results

Host: Apple M4 arm64, macOS 27.0.1 (26A434), Apple Clang 21.0.0, CMake 4.4.4.
All builds use C11 and `-Wall -Wextra -Wpedantic -Werror`; completed builds
produce zero compiler warnings.

[Local evidence](verification/2026-10-30-release-candidate-local.txt) records
the checkout, commands, numerical/lifecycle output, and eight native leak
reports. Complete logs remain in the ignored verification directory.

| Build | Result | CTest duration |
| --- | --- | --- |
| Debug, ordinary C tests | **38/38** | 35.51 s |
| Release, ordinary C tests | **38/38** | 29.47 s |
| Debug, ASan + UBSan | **38/38**, no findings | 46.94 s |
| Release, `BUILD_TESTING=OFF` | Configure/build/version pass | No tests configured |
| Debug, all optional Python tests, existing pinned environment | **47/47**, zero compiler warnings | 48.61 s |
| Debug, all optional Python tests, fresh environment | **47/47**, zero compiler warnings | 47.38 s |

Debug uses `-g`; Release uses `-O3 -DNDEBUG`. Every completed build prints
exactly `Cinder 0.1.0` followed by a newline, with empty stderr. Ordinary C
tests use only committed fixtures and require neither Python nor an MNIST
download. The macOS sanitizer suite uses
`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`.

The README virtual-environment recipe completes in a new environment using
the installed Python 3.12.14 interpreter: both pip installations succeed,
providing NumPy 2.3.5 and PyTorch 2.14.1. All 47 tests pass there, and
`pip check` finds no broken requirements. The evidence preserves the resolved
transitive dependency versions. Before the slow fresh download completed, an
independent build also passed 47/47 with the existing pinned environment.
Neither suite downloads MNIST or retrains the model.

The documented `.venv/bin/python examples/mnist/export.py --check` command
also passes in the new environment, reproducing the committed `mnist.cnd`
exactly. The source checkout remains clean at the candidate SHA after every
check. Sandboxed pip inspection emitted a cache-permission warning; the
inspection commands succeeded and needed no cache.

The documented Release CLI examples pass: MNIST image 0 predicts **7**, and
the tiny worked input predicts **0**, matching the README output. The MNIST
benchmark prints its nine-line report for **1,000 measured iterations** after
**10 warm-ups**. This is command verification during other checks, not a new
performance baseline.

## Numerical and lifecycle checks

The fresh Release run retains shape and finite-value checks and the rule
`abs(actual - expected) <= 1e-5 + 1e-5 * abs(expected)`. No tolerance changed.

| Comparison | Values | Maximum absolute error, local macOS Release |
| --- | ---: | ---: |
| Tensor NumPy fixtures | 341 | `1.1920929e-7` |
| Tiny round trip, all four stages | 120 | `5.96046448e-8` |
| MNIST first Dense | 12,800 | `6.19888306e-6` |
| MNIST ReLU | 12,800 | `5.7220459e-6` |
| MNIST logits | 1,000 | `5.7220459e-6` |
| MNIST probabilities | 1,000 | `3.57627869e-7` |

All **400 MNIST stage vectors / 27,600 values** pass. CLI prediction and
top-three index agreement is **100/100**. Agreement is separate from accuracy:
the saved subset has 99/100 correct predictions in each runtime; the earlier
Python full-test result remains 97.48%. This verification does not retrain the
model or measure C accuracy on the full test split.

The full suites exercise malformed files, injected allocation/I/O failures,
protected buffer boundaries, poisoned workspaces, repeated inference, failure
recovery, independent instances, and caller-serialized shared access. Actual
local output includes:

- **1,766 rejected loader calls**, with NULL returned ownership and no newly
  retained blocks or streams; **162 chunk-boundary failures** each recover.
- MNIST audit: **300 cycles / 1,384 inference calls**, all 21 allocation points
  failed twice, unchanged parameters, and zero final live blocks/bytes/streams.
- Loaded tiny-model lifecycle: **502 successful loads / 100 rejected loads**,
  **4,720 cycles / 14,160 calls**.
- CLI setup: **100 repeated calls per model**, all 23 allocation points tested
  with and without error details, plus benchmark failure recovery.

Audited inference cycles make **zero Cinder allocation/free calls or loader
I/O**. This does not claim that every platform-library operation allocates
nothing.

## Native macOS leak checks

Eight fresh `leaks --atExit` checks exit 0 and each report **0 leaks / 0 bytes**:
loader failures, protected buffer boundaries, input-reader failures, CLI setup,
controlled benchmark failures, MNIST lifecycle, the README MNIST `run` command,
and the production MNIST benchmark. They use the unsanitized Release build
after its CTest suite finishes.

The tool emits its usual restricted-process inspection notice but completes
the leak analyses. Guard-page mappings are checked separately by the boundary
tests; a heap leak check does not establish mapping cleanup. Apple
LeakSanitizer is unavailable in this setup.

## Hosted CI

After explicit authorization, the three reviewed commits through `1e5a5f6`
were pushed to `main`. [Run 41029277234](https://github.com/farukumarx64/cinder/actions/runs/41029277234)
passed all five jobs on its first attempt, completing at **08:56:46 UTC**
(09:56:46 Africa/Lagos) on October 30. Every downloaded job log confirms the
full candidate SHA through the checkout's `git log -1 --format=%H` output.

| Hosted job | Compiler | Result | CTest duration |
| --- | --- | --- | --- |
| [macOS Debug](https://github.com/farukumarx64/cinder/actions/runs/41029277234/job/123974820255) | Apple Clang 17.0.0 | **38/38** | 13.90 s |
| [macOS Release](https://github.com/farukumarx64/cinder/actions/runs/41029277234/job/123974820716) | Apple Clang 17.0.0 | **38/38** | 9.32 s |
| [Linux Debug, including Python](https://github.com/farukumarx64/cinder/actions/runs/41029277234/job/123974820460) | GCC 13.3.0 | **47/47** | 38.20 s |
| [Linux Release](https://github.com/farukumarx64/cinder/actions/runs/41029277234/job/123974820375) | GCC 13.3.0 | **38/38** | 4.74 s |
| [Linux ASan + UBSan + LeakSanitizer](https://github.com/farukumarx64/cinder/actions/runs/41029277234/job/123974820476) | Clang 18.1.3 | **38/38**, no findings | 20.46 s |

All five logs contain zero compiler warnings and pass the exact `0.1.0`
version smoke test. The sanitizer log confirms
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. Linux Debug installs the
optional dependencies and passes all nine additional tests; ordinary jobs use
committed C fixtures without Python or MNIST downloads. The unchanged workflow
targets `macos-15` and `ubuntu-24.04`; local builds additionally use `-Werror`.

Every job passes all 27,600 MNIST stage values and 100/100 CLI prediction and
top-three comparisons. Hosted macOS numerical errors match the local results.
The largest hosted Linux stage error is `6.67572021e-6`, in the first Dense;
maximum probability error is `3.57627869e-7`. Tolerances remain unchanged.
Lifecycle, allocation-failure, buffer-boundary, and controlled benchmark checks
also pass on both platforms.

[CI metadata](verification/2026-10-30-release-candidate-ci.json) records the
revision, conclusions, steps, toolchains, version checks, and test totals.
[Original log excerpts](verification/2026-10-30-release-candidate-ci.txt)
preserve numerical/lifecycle output and sanitizer settings. Complete job logs
remain in the ignored verification directory.

## Remaining release step

The final verification checklist is satisfied for `1e5a5f6`; no implementation
fix was needed. This report records that verified revision. A `v0.1.0` tag and
GitHub release still require authorization, so M7 remains open for publication.
No tag or release was created by this verification. Later runtime, test, build,
or workflow changes require appropriate verification of the changed candidate.

## Scope and limitations

This step records evidence; it changes no runtime source, public API, tests,
fixtures, build configuration, or workflow. The [release notes](../RELEASE_NOTES.md)
retain the supported features and constraints, including single-vector CPU
inference, caller-managed synchronization, `.cnd` v1 resource limits, deferred
`info`, and explicit failure for unresolved tiny-model timing intervals.

These checks cover the exercised platforms and deterministic cases. They do
not establish exhaustive fuzzing coverage, 32-bit support, every toolchain,
or portable performance guarantees. Native hosted macOS leak checking remains
unavailable; the required Linux sanitizer job supplies hosted leak detection.
