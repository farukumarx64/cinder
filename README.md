# Cinder

A tiny neural-network inference runtime in C, built to understand how pretrained
models become forward-pass computations over arrays in memory.

The planned v0.1.0 will load sequential float32 models and execute Dense, ReLU,
and Softmax layers on the CPU, with results checked against a Python reference.
Training happens outside the runtime.

## Current status

**Phase 1 (M1) is complete locally:** tensor allocation, cleanup, basic access,
matrix-vector multiplication, bias addition, ReLU, and stable Softmax are
implemented and tested. The math operations match deterministic NumPy float32
references within the documented tolerance.
Phase 0 is complete, including the MIT license and verified development setup.
Phase 2 has started with the Dense-layer contract; its functions are declared
but not implemented yet.
The project contains a CMake build and a CLI that supports `--version`.
The command prints `Cinder 0.1.0-dev` and exits successfully with status `0`.

- C11 compilation with `-Wall`, `-Wextra`, and `-Wpedantic` is configured.
- A local Git repository on `main` tracks source and documentation. Generated
  build files are ignored.
- The macOS build was verified on Apple Silicon with Apple Clang 21.0.0 and
  CMake 4.4.4. GitHub Actions also verified builds and tests on macOS 15 and
  Ubuntu 24.04.
- CTest provides the version-command smoke test, two tensor allocation tests,
  and tests for tensor access, matrix-vector multiplication, bias addition,
  ReLU, Softmax, and NumPy reference fixtures.
- Opt-in AddressSanitizer and UndefinedBehaviorSanitizer builds passed the
  smoke test on macOS, both separately and together.
- The public GitHub repository is [farukumarx64/cinder](https://github.com/farukumarx64/cinder).
  The local `main` branch tracks `origin/main` over SSH.
- GitHub Actions passed all three jobs: macOS, Linux, and Linux with ASan,
  UBSan, and leak detection enabled.
- The documented macOS setup was verified from a fresh GitHub clone of
  `1190c0e` on October 8, 2026. The ordinary build, sanitizer build, smoke
  tests, native leak check, and build without tests all passed.
- The [tensor contract](docs/TENSOR.md) defines the representation, ownership,
  indexing, and error behavior. Creation and cleanup are implemented, including
  size validation, zero initialization, and allocation failure handling.
- Checked `fill`, `get`, and `set` support row-major indexing, reject invalid
  indices and non-finite values, and preserve data or outputs on failure.
- `cinder_tensor_matvec` computes `y = Wx` with exact input-length and minimum
  output-capacity checks. It rejects non-finite sources and running sums,
  requires separate output storage, and allocates no memory.
- `cinder_tensor_add_bias` adds one bias per output in place, completing
  `y = Wx + b` after matvec. It checks equal, nonzero vector lengths, finite
  sources and results, and does not allocate memory.
- `cinder_tensor_relu` replaces negative finite values with zero in place.
  It preserves positive values and existing zeros, validates the full vector
  before writing, and leaves it unchanged on any failure.
- `cinder_tensor_softmax` subtracts the maximum score, exponentiates, and
  normalizes in place without allocating. It uses double intermediates with
  float32 storage, accepts underflow to zero, and rejects empty/non-finite
  inputs before changing the vector.
- On October 16, 2026, all 9 ordinary Debug tests passed locally. Release and
  combined ASan/UBSan builds with fresh NumPy generation passed all 10 tests.
  Reference tests compared 79 vectors / 341 results with the unchanged
  `atol = rtol = 1e-5` rule. The largest absolute error was `1.1920929e-7`.
  See [the reference test guide](docs/NUMPY_TESTS.md) for coverage and limits.
- The Softmax and NumPy reference executables reported zero native leaks on
  October 16, as did access, matrix-vector, bias, and ReLU tests on October 15 and the allocation
  tests on October 9. Hosted CI verification of the NumPy changes is pending.
- The [Dense-layer contract](docs/DENSE.md) specifies dimensions, row-major
  weights, a bias vector, owned parameter copies, public functions, buffer
  overlap rules, and validation versus arithmetic failure behavior.
- The next scoped task is implementing Dense-layer creation and cleanup.
- Model loading, inference, and the `info`, `run`, and `bench` commands are
  planned features; they are not implemented yet.

See [the roadmap](ROADMAP.md#current-progress) for the current checklist and
next scoped task.

## Prerequisites

- CMake 3.16 or newer.
- A C11-capable compiler, such as Apple Clang, Clang, or GCC.
- A build tool supported by CMake, such as Make or Ninja. The verified macOS
  setup uses Make from the Xcode Command Line Tools.
- Git for version control.

Python and NumPy are optional for the C build and its saved reference tests.
Regenerating references requires Python 3.11+ and NumPy; the tested dependency
is pinned in `tests/requirements-numpy.txt`.
Softmax requires `double` to have wider range and precision than `float`,
checked at compile time. CMake links the system math library (`m`) on Unix
for exponentiation; no extra third-party library is needed.

## Build and run

From the project root:

```sh
cmake -S . -B build
cmake --build build
./build/cinder --version
```

The first command configures the project and generates build files in `build/`.
The second compiles and links the executable. The third prints:

```text
Cinder 0.1.0-dev
```

The version number comes from CMake's project version, with `-dev` appended for
the development build. The CLI currently accepts exactly one argument:
`--version`. Missing, unknown, or extra arguments print a usage message to
standard error and return a nonzero exit status.

Keep generated files in a separate build directory. The ignore rules cover
`build/`, `build-*/`, and `cmake-build-*/`, along with common generated CMake files.

## Run the tests

CTest ships with CMake. Tests are enabled by default through `BUILD_TESTING`.
Configure and build the project before running the tests:

```sh
cmake -S . -B build
cmake --build build
(cd build && ctest --output-on-failure)
```

The `cli_version` test runs `cinder --version` and requires exit status `0`,
exactly `Cinder 0.1.0-dev` followed by a newline on standard output, and empty
standard error. The expected version follows the CMake project version.
Failures show diagnostics; the test has a ten-second timeout.

Eight more CTest tests exercise tensor memory management, access, and math:

- `tensor_allocation` checks dimensions, zero initialization, writable storage,
  independent tensors, invalid arguments, both size-overflow cases, repeated
  creation/cleanup, and optional error details against the `cinder_core` library.
- `tensor_allocation_failures` forces the first and second allocations to fail,
  checks partial cleanup and NULL outputs, and confirms validation happens
  before allocation. It also checks that access, matvec, bias addition, ReLU,
  and Softmax do not allocate.
  It compiles the same implementation with allocator
  substitutions confined to the test; production has no failure switches.
- `tensor_access` checks fills across several shapes, reads and writes against
  known row-major storage, NULL pointers, boundary and huge indices, NaN and
  infinities, unchanged data/outputs on error, and optional error details.
  Finite extremes, subnormal values, and signed zero are also exercised.
- `tensor_matvec` checks a hand-calculated non-square product, decimal values,
  identity/zero cases, single-row/column shapes, input lengths, output capacity,
  unchanged sources and spare output elements, non-finite values, and arithmetic
  overflow. It uses `atol = 1e-5` and `rtol = 1e-5` for numeric comparisons.
- `tensor_bias` checks in-place addition, vector lengths, byte-count overflow,
  non-finite sources and sums, failure preservation, partial updates on
  arithmetic overflow, unchanged bias and spare output storage, and error
  details. Integration cases verify `Wx + b` and a zero input whose result is
  exactly the bias. It uses the same numerical tolerance rule as matvec.
- `tensor_relu` checks negative/zero/positive values, finite extremes, signed
  zeros, subnormal values, invalid counts and pointers, non-finite values,
  unchanged data on failure, and untouched spare capacity. It also checks
  repeated application and the full `ReLU(Wx + b)` sequence. Expected values
  are compared exactly because ReLU introduces no rounding arithmetic.
- `tensor_softmax` checks known probabilities, large score offsets, equal and
  extreme values, tiny probabilities and allowed underflow, and a long vector
  with many small contributions to its sum. It also checks empty/non-finite
  input rejection, unchanged arrays on validation failure, optional errors,
  spare capacity, and composed `Softmax(ReLU(Wx + b))` results. Expected values
  use independent constants, with tighter checks for tiny probabilities.
- `tensor_numpy_fixtures` compares each math operation and composed intermediate
  results with saved NumPy float32 references. It checks shapes and finiteness
  before applying `atol = rtol = 1e-5`, and reports the failing fixture, stage,
  index, values, and error limit. Coverage includes 11 matrix/bias fixtures and
  12 standalone activation fixtures, for 79 result vectors / 341 values.

All tensor tests have ten-second timeouts. Their checks stay active in Release
builds, where ordinary C `assert` checks could be disabled by `NDEBUG`.

To configure a build without tests, pass `-DBUILD_TESTING=OFF` to CMake.

### Generate fresh NumPy references

The default build uses the saved fixture header without running Python. To
also compare against newly generated references, use Python 3.11 or newer
(for example, `python3.12`) and a local virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements-numpy.txt
cmake -S . -B build-numpy \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCINDER_ENABLE_NUMPY_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-numpy
ctest --test-dir build-numpy --output-on-failure
```

This adds the tenth test, `tensor_numpy_generated`, using a generated header
inside the build directory. The saved fixtures stay unchanged. CMake fails
clearly if NumPy is missing or testing is disabled. See
[NumPy reference verification](docs/NUMPY_TESTS.md) for the formulas, fixture
generation command, reproducibility policy, and recorded results.

## Sanitizer builds

Sanitizers add checks that run while the program executes. Use a GCC or Clang
toolchain with its sanitizer runtime libraries installed. Both options default
to `OFF` and can be enabled independently:

| CMake option | Purpose |
| --- | --- |
| `CINDER_ENABLE_ASAN` | Detect memory errors such as out-of-bounds access and use-after-free. |
| `CINDER_ENABLE_UBSAN` | Detect undefined behavior such as signed integer overflow and invalid pointer alignment. |

For development, enable both in a separate Debug build:

```sh
cmake -S . -B build-sanitize \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCINDER_ENABLE_ASAN=ON \
  -DCINDER_ENABLE_UBSAN=ON
cmake --build build-sanitize
(cd build-sanitize && ctest --output-on-failure)
```

Shared CMake settings apply warning and sanitizer flags to the core library,
CLI, and all test executables, including sanitizer flags at executable link
time. Sanitizer builds retain frame pointers for clearer stack traces. UBSan
uses `-fno-sanitize-recover=undefined` so a detected error terminates the program
with a failure instead of continuing. CTest reports any failing test.

CMake remembers these options per build directory. Use `build/` for ordinary
builds and `build-sanitize/` for sanitizer builds; explicitly set either option
to `OFF` when disabling it in an existing build directory. Benchmark with
sanitizers disabled because their checks add runtime overhead.

The current tests exercise the version command, tensor allocation/cleanup,
basic access, matrix-vector multiplication, bias addition, ReLU, and Softmax.
NumPy reference comparisons are included; model inference tests remain future steps.
Passing the current tests is not evidence that future code is free of memory errors.

See the upstream [AddressSanitizer documentation](https://clang.llvm.org/docs/AddressSanitizer.html)
and [UndefinedBehaviorSanitizer documentation](https://clang.llvm.org/docs/UndefinedBehaviorSanitizer.html)
for the supported checks and runtime options.

### Leak checks

Leak detection depends on the toolchain. On the verified Apple Silicon setup
with Apple Clang 21.0.0, setting `ASAN_OPTIONS=detect_leaks=1` reports that leak
detection is unsupported. The default macOS sanitizer command above does not
force that setting.

On macOS, run the native `leaks` tool against the ordinary build:

```sh
leaks --atExit -- ./build/cinder --version
leaks --atExit -- ./build/test_tensor
leaks --atExit -- ./build/test_tensor_alloc_failures
leaks --atExit -- ./build/test_tensor_access
leaks --atExit -- ./build/test_tensor_matvec
leaks --atExit -- ./build/test_tensor_bias
leaks --atExit -- ./build/test_tensor_relu
leaks --atExit -- ./build/test_tensor_softmax
leaks --atExit -- ./build/test_tensor_numpy
```

These commands reported zero leaks for the version command and tensor test
executables during their respective verification steps. The tool needs
permission to inspect the process it launches; a restrictive sandbox can
block it. An inspection error is not a successful leak check.

On Linux with a LeakSanitizer-capable toolchain, explicitly enable leak checks
when running the sanitizer build's tests:

```sh
(cd build-sanitize && ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure)
```

The Linux sanitizer CI job runs the full CTest suite with leak detection
enabled. The new NumPy fixture checks will run after these changes are pushed;
their hosted results are pending. See the
[LeakSanitizer documentation](https://clang.llvm.org/docs/LeakSanitizer.html).

## Continuous integration

[The CI workflow](.github/workflows/ci.yml) runs on pushes to `main`, pull
requests, and manual dispatch from the repository's Actions tab.

| Job | Runner | Checks |
| --- | --- | --- |
| Linux build | Ubuntu 24.04 | Debug configuration, compilation, and CTest, including freshly generated NumPy references. |
| macOS build | macOS 15 | Debug configuration, compilation, and CTest. |
| Linux sanitizers | Ubuntu 24.04 with Clang | Debug build with ASan and UBSan, then CTest with leak detection enabled. |

Each job starts from a fresh checkout, prints its toolchain versions, and fails
if configuration, compilation, or testing fails. CTest also fails if no tests
are registered. Jobs have a ten-minute limit, and newer runs replace older runs
for the same branch or pull request.

All jobs test the saved NumPy fixtures. The ordinary Linux job additionally
creates a Python virtual environment, installs the pinned reference dependency,
and enables `CINDER_ENABLE_NUMPY_TESTS`. Hosted results for this addition are
pending a future push.

The workflow uses read-only repository permissions and a pinned checkout action.
Check results and failure logs in [GitHub Actions](https://github.com/farukumarx64/cinder/actions/workflows/ci.yml).
All three jobs passed in the [first hosted run](https://github.com/farukumarx64/cinder/actions/runs/37680825782)
on October 7, 2026.
Leak detection is enabled only in the Linux sanitizer job because the verified
Apple Clang runtime does not support LeakSanitizer on this Mac.

## Repository layout

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Core library and executable targets, shared compiler/sanitizer settings, and CTest registration. |
| `src/cli.c` | Command-line argument handling and version output. |
| `src/tensor.c` | Tensor allocation, cleanup, checked access, matrix-vector multiplication, bias addition, ReLU, Softmax, and error reporting. |
| `include/cinder/tensor.h` | Tensor representation and function declarations. |
| `include/cinder/dense.h` | Dense representation and create/free/forward declarations; implementation pending. |
| `include/cinder/error.h` | Status codes and caller-owned error details. |
| `docs/TENSOR.md` | Tensor layout, ownership, validation, and failure contract. |
| `docs/DENSE.md` | Dense parameter shapes, ownership, public API, overlap, failures, and worked example. |
| `docs/NUMPY_TESTS.md` | Reference calculations, fixtures, tolerances, regeneration, and verification results. |
| `tests/check_version.cmake` | Checks the version command's exit status and output. |
| `tests/test_tensor.c` | Tests the core library's allocation and cleanup API. |
| `tests/test_tensor_access.c` | Tests row-major access, bounds, finite values, and failure behavior. |
| `tests/test_tensor_matvec.c` | Tests matrix-vector products, dimensions, output capacity, and numeric failures. |
| `tests/test_tensor_bias.c` | Tests bias addition, vector lengths, numeric failures, and composed `Wx + b` results. |
| `tests/test_tensor_relu.c` | Tests ReLU values, validation, failure preservation, and composed `ReLU(Wx + b)` results. |
| `tests/test_tensor_softmax.c` | Tests stable probabilities, extreme/tiny values, validation, and composed math results. |
| `tests/test_tensor_numpy.c` | Compares Cinder's math with saved or freshly generated NumPy fixtures. |
| `tests/generate_numpy_fixtures.py` | Generates independent float32 NumPy references with fixed inputs and seed. |
| `tests/fixtures/numpy_fixtures.h` | Saved input arrays and NumPy expected results, tracked in Git. |
| `tests/requirements-numpy.txt` | Pinned optional NumPy dependency for reference generation. |
| `tests/test_tensor_alloc_failures.c` | Forces allocation failures and tracks cleanup with test-only allocator substitutions. |
| `.github/workflows/ci.yml` | macOS/Linux build jobs and the Linux sanitizer job. |
| `ROADMAP.md` | Scope, implementation decisions, milestones, and progress. |
| `LICENSE` | MIT license and copyright notice. |
| `.gitignore` | Generated build files and macOS metadata exclusions. |

## Development workflow

Work on one scoped task at a time. Define its completion check, make the change,
verify the relevant behavior, and update the roadmap with the result and next
task. Record unresolved issues before moving on.

The roadmap is the source of truth for progress. Its future command examples
and benchmark numbers describe targets, not existing capabilities or measured
performance. Phases 0 and 1 are complete locally. Phase 2's Dense-layer contract
is defined; creation and cleanup are next.

The v0.1.0 scope excludes training, GPU execution, convolutional layers,
quantization, SIMD, multithreading, and arbitrary computation graphs.

## License

Cinder is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Faruk Umar.
