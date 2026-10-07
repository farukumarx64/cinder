# Cinder

A tiny neural-network inference runtime in C, built to understand how pretrained
models become forward-pass computations over arrays in memory.

The planned v0.1.0 will load sequential float32 models and execute Dense, ReLU,
and Softmax layers on the CPU, with results checked against a Python reference.
Training happens outside the runtime.

## Current status

**Phase 0: project setup is in progress.** The project currently contains a CMake
build and a CLI that supports `--version`. The command prints `Cinder 0.1.0-dev`
and exits successfully with status `0`.

- C11 compilation with `-Wall`, `-Wextra`, and `-Wpedantic` is configured.
- A local Git repository on `main` tracks source and documentation. Generated
  build files are ignored.
- The macOS build was verified on Apple Silicon with Apple Clang 21.0.0 and
  CMake 4.4.4. GitHub Actions also verified builds and tests on macOS 15 and
  Ubuntu 24.04.
- CTest provides a smoke test for the version command.
- Opt-in AddressSanitizer and UndefinedBehaviorSanitizer builds passed the
  smoke test on macOS, both separately and together.
- The public GitHub repository is [farukumarx64/cinder](https://github.com/farukumarx64/cinder).
  The local `main` branch tracks `origin/main` over SSH.
- GitHub Actions passed all three jobs: macOS, Linux, and Linux with ASan,
  UBSan, and leak detection enabled.
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

Python and machine-learning packages are not required for the current C build.
They will be introduced with the reference tests and exporter.

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

## Run the smoke test

CTest ships with CMake. Tests are enabled by default through `BUILD_TESTING`.
Configure and build the executable before running the test:

```sh
cmake -S . -B build
cmake --build build
(cd build && ctest --output-on-failure)
```

The `cli_version` test runs `cinder --version` and requires exit status `0`,
exactly `Cinder 0.1.0-dev` followed by a newline on standard output, and empty
standard error. The expected version follows the CMake project version.
Failures show diagnostics; the test has a ten-second timeout.

To configure a build without tests, pass `-DBUILD_TESTING=OFF` to CMake.

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

The flags apply to both compilation and linking. Sanitizer builds retain frame
pointers for clearer stack traces. UBSan uses `-fno-sanitize-recover=undefined`
so a detected error terminates the program with a failure instead of continuing.
CTest reports failures through the existing smoke test.

CMake remembers these options per build directory. Use `build/` for ordinary
builds and `build-sanitize/` for sanitizer builds; explicitly set either option
to `OFF` when disabling it in an existing build directory. Benchmark with
sanitizers disabled because their checks add runtime overhead.

The current smoke test exercises only the version command. Additional tests
will need to exercise tensor allocation and inference as those features arrive.
Passing the current test is not evidence that future memory-management code is
free of errors.

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
```

This reported zero leaks for the current version-command execution. The tool
needs permission to inspect the process it launches; a restrictive sandbox can
block it. An inspection error is not a successful leak check.

On Linux with a LeakSanitizer-capable toolchain, explicitly enable leak checks
when running the sanitizer build's tests:

```sh
(cd build-sanitize && ASAN_OPTIONS=detect_leaks=1 ctest --output-on-failure)
```

The Linux sanitizer CI job passed the version-command smoke test with leak
detection enabled. See the
[LeakSanitizer documentation](https://clang.llvm.org/docs/LeakSanitizer.html).

## Continuous integration

[The CI workflow](.github/workflows/ci.yml) runs on pushes to `main`, pull
requests, and manual dispatch from the repository's Actions tab.

| Job | Runner | Checks |
| --- | --- | --- |
| Linux build | Ubuntu 24.04 | Debug configuration, compilation, and CTest. |
| macOS build | macOS 15 | Debug configuration, compilation, and CTest. |
| Linux sanitizers | Ubuntu 24.04 with Clang | Debug build with ASan and UBSan, then CTest with leak detection enabled. |

Each job starts from a fresh checkout, prints its toolchain versions, and fails
if configuration, compilation, or testing fails. CTest also fails if no tests
are registered. Jobs have a ten-minute limit, and newer runs replace older runs
for the same branch or pull request.

The workflow uses read-only repository permissions and a pinned checkout action.
Check results and failure logs in [GitHub Actions](https://github.com/farukumarx64/cinder/actions/workflows/ci.yml).
All three jobs passed in the [first hosted run](https://github.com/farukumarx64/cinder/actions/runs/37680825782)
on October 7, 2026.
Leak detection is enabled only in the Linux sanitizer job because the verified
Apple Clang runtime does not support LeakSanitizer on this Mac.

## Repository layout

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Executable target, version definition, compiler/sanitizer settings, and CTest registration. |
| `src/cli.c` | Command-line argument handling and version output. |
| `tests/check_version.cmake` | Checks the version command's exit status and output. |
| `.github/workflows/ci.yml` | macOS/Linux build jobs and the Linux sanitizer job. |
| `ROADMAP.md` | Scope, implementation decisions, milestones, and progress. |
| `.gitignore` | Generated build files and macOS metadata exclusions. |

## Development workflow

Work on one scoped task at a time. Define its completion check, make the change,
verify the relevant behavior, and update the roadmap with the result and next
task. Record unresolved issues before moving on.

The roadmap is the source of truth for progress. Its future command examples
and benchmark numbers describe targets, not existing capabilities or measured
performance. Phase 0 ends only when its setup checks are complete.

The v0.1.0 scope excludes training, GPU execution, convolutional layers,
quantization, SIMD, multithreading, and arbitrary computation graphs.

## License

The license choice is pending the project owner's selection.
