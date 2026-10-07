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
  CMake 4.4.4. Linux support is planned but has not yet been verified.
- CTest provides a smoke test for the version command.
- Sanitizers, CI, and a GitHub remote are still pending.
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

## Repository layout

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Executable target, version definition, compiler settings, and CTest registration. |
| `src/cli.c` | Command-line argument handling and version output. |
| `tests/check_version.cmake` | Checks the version command's exit status and output. |
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
