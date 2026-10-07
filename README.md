# Cinder

A tiny neural-network inference runtime in C, built to understand how pretrained
models become forward-pass computations over arrays in memory.

The planned v0.1.0 will load sequential float32 models and execute Dense, ReLU,
and Softmax layers on the CPU, with results checked against a Python reference.
Training happens outside the runtime.

## Current status

**Phase 0: project setup is in progress.** The project currently contains a CMake
build and a minimal C entry point. Running the executable produces no output
and exits successfully with status `0`.

- C11 compilation with `-Wall`, `-Wextra`, and `-Wpedantic` is configured.
- A local Git repository on `main` tracks source and documentation. Generated
  build files are ignored.
- The macOS build was verified on Apple Silicon with Apple Clang 21.0.0 and
  CMake 4.4.4. Linux support is planned but has not yet been verified.
- Version handling, automated tests, sanitizers, CI, and a GitHub remote are
  still pending.
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
./build/cinder
```

The first command configures the project and generates build files in `build/`.
The second compiles and links the executable. The third runs the minimal entry
point; silent completion is the expected behavior at this stage.

Keep generated files in a separate build directory. The ignore rules cover
`build/`, `build-*/`, and `cmake-build-*/`, along with common generated CMake files.

## Repository layout

| Path | Purpose |
| --- | --- |
| `CMakeLists.txt` | Executable target, C standard, and compiler warnings. |
| `src/cli.c` | Minimal program entry point; CLI behavior will be added here. |
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
