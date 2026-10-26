# MNIST C inference versus Python

**Verified locally October 26, 2026.** Cinder's loaded MNIST model matches every
saved Python/PyTorch float32 stage for the first 100 official test images, using
the unchanged `atol = rtol = 1e-5` rule. This exercises the public C loader and
model APIs. The `cinder run` command remains an upcoming implementation step.

## Measured results

All **400 vectors / 27,600 values** pass shape, finiteness, and numerical checks:

| Stage | Shape across 100 images | Maximum absolute error | Maximum fraction of allowed error |
| --- | --- | ---: | ---: |
| First Dense | `[100, 128]` | `6.19888306e-6` | `0.178311515` |
| ReLU | `[100, 128]` | `5.7220459e-6` | `0.178311515` |
| Second Dense / logits | `[100, 10]` | `5.7220459e-6` | `0.176519426` |
| Softmax probabilities | `[100, 10]` | `3.57627869e-7` | `0.0287552465` |

For each element, the shared comparator requires:

```text
abs(C_value - Python_reference) <= 1e-5 + 1e-5 * abs(Python_reference)
```

The final column divides each element's absolute error by its own allowed
error, then takes the maximum for that stage. A value above `1` fails. The
largest observed fraction is about **17.83%** of the permitted error. This
measure remains useful near zero, where a plain relative-error ratio is
undefined or misleading. Absolute differences are calculated in `double`
from the two float32 values. No tolerance or golden answer was changed.

The three accuracy/agreement figures answer different questions:

| Measurement | Actual result | Meaning |
| --- | ---: | --- |
| C versus saved Python prediction agreement | **100/100 (100%)** | Both runtimes choose the same class on every saved image. |
| Classification on this fixed test subset | **C 99/100; Python 99/100 (99%)** | Each prediction is compared with the official label. |
| Previously measured full official test accuracy | **Python 9,748/10,000 (97.48%)** | Training/evaluation result from [RESULTS.md](RESULTS.md); this step does not run all 10,000 images in C. |

The preserved misclassification is test index `8`: label `5`, prediction `6`.
Runtime agreement includes that case. The 100-image regression subset is not a
replacement for full-test evaluation or an estimate of generalization accuracy.

## C-only stage validation

[`tests/test_mnist_inference.c`](../../tests/test_mnist_inference.c) is registered
as `mnist_inference_fixtures` in every ordinary testing build. It needs the
committed model and [fixtures](FIXTURES.md), with no Python interpreter, PyTorch,
dataset cache, download, or training step.

The model is opaque through its public API. To observe intermediate stages,
the test loads four models: the complete saved model and three temporary
prefixes ending after Dense1, ReLU, and Dense2. The test checks the fixed MNIST
header, record types, dimensions, payload sizes, and boundaries. It copies
original record/parameter bytes, changing only the header's layer count for
each prefix. Their lengths are `401952`, `401968`, and `407144` bytes; the full
model is `407160` bytes. Each copy passes through `cinder_model_load`.
These constants describe this fixture, not a general-purpose file parser.

Every prefix starts with the **original 784-value image input** and executes
all preceding layers in C. Saved intermediate answers are used only as
comparison targets. The fourth comparison runs the full original `mnist.cnd`,
so final probabilities include the accumulated numerical effects of the whole
network. The optional Python test independently verifies that the three
prefixes match exports of the corresponding checkpoint layer sequences.

Before comparison the C test checks:

- Exact model input/output dimensions and exact reference/input file lengths,
  including EOF. Four-byte words are decoded explicitly in little-endian order.
- Finite input and reference values; the shared
  [`numpy_compare.h`](../../tests/numpy_compare.h) also checks every actual value
  and rejects empty/mismatched vectors before tolerance arithmetic. Despite the
  helper's historical name, these MNIST references come from **PyTorch**.
- Dataset indices `0..99`, labels in `0..9`, and every input value against its
  saved row-major pixel divided by `255.0f`.
- Unchanged caller input, untouched guard values and spare output slots, and
  both exact output capacity and one extra slot.
- Final probabilities in `[0,1]`, sum within `1e-6` of one, and zero-based
  argmax with the lowest index winning ties.

Models are reused across the 100 inputs. All owned models, temporary model
bytes, file handles, and generated prefix files are released on test success
and failure. C assertions remain active in Release builds. Scratch files live
in the build directory; source fixtures are opened only for reading.

## Fresh export and rejection controls

[`tests/check_mnist_inference.py`](../../tests/check_mnist_inference.py) adds
`mnist_inference_exported` when `CINDER_ENABLE_MNIST_TESTS=ON`, independently of
the NumPy and exporter test options. It restores the saved checkpoint, exports
a new model, requires byte equality with `mnist.cnd`, verifies all three prefix
identities, and calls the same C executable against copies of the saved
references. It never recomputes expected answers to match a C result.

It also requires clean failures for **28 deliberately altered cases**:

- For each of the four stage files: wrong first value, truncated array, extra
  byte, final-element NaN, and wrong final value on image `99` (20 cases).
- Truncated, oversized, infinite, and incorrectly preprocessed input (4).
- Incorrect dataset index and out-of-range label (2).
- A valid finite bias change causing numerical disagreement, and finite
  extreme weights causing non-finite accumulation during inference (2).

Every run must remove its scratch prefixes, including failed comparisons.
Mutations affect temporary copies and are restored after each check.
A separate valid label-only change lowers both subset accuracies to `98/100`
while the test still passes all stage comparisons and reports `100/100` runtime
agreement. This explicitly tests the separation between the two measurements.

## Reproduce

From the repository root, the C-only path is:

```sh
cmake -S . -B build-mnist-check -DCMAKE_BUILD_TYPE=Debug
cmake --build build-mnist-check --parallel
ctest --test-dir build-mnist-check -R '^mnist_inference_fixtures$' -V
```

For the offline fresh-export check, use the example's existing Python
environment and [pinned dependencies](requirements.txt):

```sh
cmake -S . -B build-mnist-check -DCINDER_ENABLE_MNIST_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-mnist-check --parallel
ctest --test-dir build-mnist-check -R '^mnist_inference_' -V
```

The MNIST option now registers four optional tests: training, export, reference
fixtures, and fresh-export C agreement. The saved-fixture C test is always
registered when `BUILD_TESTING=ON`. Use verbose CTest output to see numerical
maxima and prediction/accuracy counts on successful runs.

## Verification and remaining scope

Local environment: macOS 27.0.1 arm64, Apple Clang 21.0.0, CMake 4.4.4,
Python 3.12.14, PyTorch 2.14.1, and NumPy 2.3.5. Debug, Release, and sanitizer
builds produced the same maxima shown above.

| Check | Actual result |
| --- | --- |
| Debug, all optional Python suites enabled (`build-mnist`) | **33/33 CTest tests passed** |
| Release, all Python options OFF (`build-mnist-c-only`) | **24/24 passed**, no Python discovery or execution |
| Debug, ASan + UBSan, Python options OFF (`build-mnist-sanitize`) | **24/24 passed**, no sanitizer findings |
| Fresh export and all 28 corruption controls using the sanitized C executable | Passed; label-only separation check also passed |
| Native macOS `leaks --atExit` on the Release MNIST comparator | **0 leaks / 0 leaked bytes** |

Sanitizers used `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1` and
`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. LeakSanitizer is unsupported
on this macOS target; the native leak check is separate. `leaks` needed access
outside the execution sandbox to inspect its child process and emitted its
standard restricted-process memory-display notice before reporting zero leaks.

The existing workflow includes the new C test in macOS, Linux, and Linux
sanitizer jobs, and the optional fresh-export check in the Linux Python job.
The numerical-validation revision passed all three existing jobs in
[run 40318184929](https://github.com/farukumarx64/cinder/actions/runs/40318184929).
The subsequent [lifecycle audit and expanded CI matrix](LIFECYCLE.md) record
their own verification status separately. Other
architectures/toolchains may produce different rounding errors and must pass
the same tolerance rather than reproduce these exact maxima.

The training script, checkpoint, exported model, saved answers, production C
runtime, and CLI implementation are unchanged. M4 remains open: input-file
reading, `cinder run`, final hosted lifecycle verification, and documentation/
closure remain in the roadmap. The larger-model lifecycle audit now passes locally.
