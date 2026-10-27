# MNIST demonstration and M4 verification

**October 27, 2026 (Africa/Lagos): the input reader and `cinder run` are
implemented, and local verification passes. M4 remains open until hosted CI
passes for these additions.** Earlier lifecycle CI results do not substitute
for verification of the new CLI.

This guide connects the saved Python checkpoint, exported model, committed
inputs, C executable, and measured results. Training happened externally;
Cinder loads parameters and performs CPU inference.

## Run the saved demonstration without Python or downloads

From the repository root, with a C11 compiler and CMake 3.16 or newer:

```sh
cmake -S . -B build-m4-demo -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build-m4-demo --parallel
ctest --test-dir build-m4-demo --output-on-failure
./build-m4-demo/cinder --version
./build-m4-demo/cinder run examples/mnist/mnist.cnd \
  tests/fixtures/mnist/inputs/00000.f32
```

The ordinary suite uses committed model/reference files. It does not discover
Python, install dependencies, download MNIST, or retrain a network. Use a fresh
build directory if an existing cache has optional Python tests enabled.

Actual local C stdout for official test index **0**, whose label is **7**:

```text
Model: mnist.cnd
Prediction: 7
Scores:
7	0.9978019
3	0.0018843736
2	0.000236767111
```

There is one literal tab between each index and score. Success exits `0` and
leaves stderr empty. Scores are printed with `%.9g`; last digits can vary with
the compiler/CPU within the numerical tolerance. Output index `i` means digit
`i` for this checkpoint. The final Softmax makes these scores probabilities.

The tiny saved network also works:

```sh
./build-m4-demo/cinder run tests/fixtures/tiny/tiny.cnd \
  tests/fixtures/tiny/inputs/worked.f32
```

```text
Model: tiny.cnd
Prediction: 0
Scores:
0	0.904650509
1	0.0953494608
```

## Input and command behavior

The MNIST input is exactly **784 little-endian IEEE binary32 values / 3,136
bytes**, with no header or trailing data. Flatten a 28 × 28 image in row-major
order, cast its uint8 pixels to float32, and divide by float32 `255.0`.
The CLI receives already-preprocessed values; it does not decode image files.

The general command accepts any supported model, including a model without
Softmax. It reports the greatest raw output value and the top `min(3, M)`
scores, breaking exact ties by the smaller output index. It never adds an
activation or assigns digit meanings to another model.

The [command contract](../../docs/CLI.md) specifies exact formatting and errors:

- `0`: version or prediction output, including its final flush, succeeded.
- `1`: model/input/inference/allocation/output failure. No prediction is printed
  before successful validation and inference. A later write failure can leave
  partial stdout.
- `2`: invalid command, argument count, or empty path. Stderr includes usage.

Model validation happens before input reading. The private reader checks the
host representation, checked byte extent, measured file length, every read,
finite values, clean EOF, and successful close. It decodes bytes explicitly
instead of treating disk bytes as native float objects. The model and CLI
buffers are released before prediction output. Earlier failures survive a
subsequent close failure. Paths are escaped for display, not changed for lookup.

## Measured accuracy and runtime agreement

These measurements answer different questions:

| Measurement | Actual result | Scope |
| --- | --- | --- |
| Python classification accuracy | **9,748/10,000 = 97.48%** | Entire official test split, evaluated after training on the separate 60,000-image training split; measured October 25 |
| C/Python classification accuracy on fixtures | **99/100 each** | Fixed test indices `0..99`; index 8 has label 5 and prediction 6 |
| C/Python prediction agreement | **100/100** | Same predicted index for every saved input, including the shared wrong answer |
| C/Python stage comparison | **400 vectors / 27,600 finite values pass** | Dense1, ReLU, final logits, and Softmax; shape checks plus `atol = rtol = 1e-5` |
| CLI prediction and ordered top-three index agreement | **100/100** | Real executable processes using the committed inputs and saved Python probabilities |

The numerical rule is `abs(C - reference) <= 1e-5 + 1e-5 * abs(reference)`.
The following are observed maxima, not extra tolerance settings:

| Stage | Local/macOS maximum absolute error | Previously verified Linux maximum absolute error |
| --- | ---: | ---: |
| First Dense | `6.19888306e-6` | `6.67572021e-6` |
| ReLU | `5.7220459e-6` | `4.76837158e-6` |
| Final logits | `5.7220459e-6` | `5.7220459e-6` |
| Softmax probabilities | `3.57627869e-7` | `3.57627869e-7` |

The local all-stage comparison was rerun with this CLI addition. Linux values
come from the earlier lifecycle revision, documented with its exact CI run in
[VALIDATION.md](VALIDATION.md). The maximum local error for the 300 displayed
CLI scores, parsed back to floats, is also `3.57627869e-7`.
We have **not measured full 10,000-image C accuracy**. The 100-image regression
subset and runtime agreement must not be reported as that result.

## Reproduce the Python artifacts

The saved model is `Dense(784,128) → ReLU → Dense(128,10) → Softmax(10)` with
101,770 parameters. Training used logits with CrossEntropyLoss, seed `20261025`,
five epochs, batch size 64, Adam at `0.001`, CPU float32, one intra-op thread,
deterministic algorithms, and no test-set training or checkpoint selection.
See [RESULTS.md](RESULTS.md) for the full settings, training history, saved
configuration, and command to re-evaluate the checkpoint without training.

For the optional Python work, use Python 3.11+ and the pinned dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r examples/mnist/requirements.txt
.venv/bin/python -m pip install -r tests/requirements-numpy.txt

# Read the saved checkpoint and compare a temporary export; no data required.
.venv/bin/python examples/mnist/export.py --check

# Use the cached original test archives; no downloads or fixture overwrites.
.venv/bin/python examples/mnist/generate_fixtures.py --check
```

On Linux, install the PyTorch requirements with
`--index-url https://download.pytorch.org/whl/cpu`; install NumPy from its normal
index. For missing test archives, explicitly add `--download` to the fixture
command. Exact fixture regeneration, including environment metadata, is a
same-environment check; cross-platform numerical tests retain the tolerance.

To repeat training, choose a new output directory:

```sh
.venv/bin/python examples/mnist/train.py --download \
  --seed 20261025 --epochs 5 --batch-size 64 --learning-rate 0.001 \
  --output-dir examples/mnist/runs/reproduce
.venv/bin/python examples/mnist/export.py \
  --checkpoint examples/mnist/runs/reproduce/checkpoint.pt \
  --output examples/mnist/runs/reproduce/mnist.cnd
```

An existing training output directory is rejected. Omit `--download` to require
the four cached archives. The reference uses the saved October 25 checkpoint;
this CLI step did not retrain or replace it. Same-environment repeat training
previously matched all parameters and metadata; other environments may differ.

For fresh-export comparisons and all optional offline tests:

```sh
cmake -S . -B build-mnist -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=ON \
  -DCINDER_ENABLE_NUMPY_TESTS=ON -DCINDER_ENABLE_EXPORTER_TESTS=ON \
  -DCINDER_ENABLE_MNIST_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-mnist --parallel
ctest --test-dir build-mnist --output-on-failure
```

## Verification of the CLI addition

Local environment: macOS 27.0.1 arm64, Apple Clang 21.0.0, CMake 4.4.4,
Python 3.12.14, NumPy 2.3.5, and PyTorch 2.14.1. Checked October 27, 2026:

| Check | Result |
| --- | --- |
| Debug, all optional Python suites | **38/38 passed** |
| Release, C-only | **29/29 passed** |
| Fresh `build-m4-demo` directory using the commands above | **29/29 passed**; version and both saved-model commands produced the documented output |
| Debug ASan + UBSan, C-only | **29/29 passed**, no findings |
| Real CLI subprocess cases | **122 passed**, including 100 MNIST inputs |
| Native macOS leaks: MNIST success and missing-input failure | **0 leaks / 0 leaked bytes** for both |
| Native macOS leaks: input-reader and CLI failure audits | **0 leaks / 0 leaked bytes** for both |
| New C sources/tests, C11 warnings as errors | Passed |
| Saved checkpoint export and complete fixture regeneration | Identical model bytes; **108/108 fixture files identical** |
| New CLI revision in hosted macOS/Linux Debug/Release and Linux sanitizer CI | **Pending; M4 remains open** |

Three new CTest cases complement `cli_version`:

- [`cli_input`](../../tests/test_input.c) compiles the real private reader with
  test-only allocation/stdio wrappers. It checks bit-preserved signed zero,
  subnormal and extreme finite values, all short lengths around a six-value
  input, extra bytes, all three non-finite categories, extent overflow, open/
  seek/size/read/EOF/close/allocation failures, first-error precedence, optional
  diagnostics, one close attempt, and zero retained allocations.
- [`cli_audit`](../../tests/test_cli.c) executes the real dispatcher with controlled
  API responses. It verifies arguments before file access, failure propagation,
  bounded/escaped messages, all status tags, zero/overflowing dimensions,
  output allocation, non-finite outputs, top-three ordering/ties/negative scores/
  signed zero, version behavior, every output write and final-flush failure,
  stderr failure, and cleanup before output. Model behavior itself is tested
  through the real executable and existing model suites.
- [`cli_process`](../../tests/test_cli_process.c) launches the built executable
  with actual files: tiny/MNIST fixtures, raw-score models of different sizes,
  literal hyphen paths, unusual names and symlinks, malformed/unsupported/over-limit
  models, malformed inputs, real arithmetic overflow and a succeeding retry.
  It checks exit codes, empty stdout on failure, exact formatting, ranking,
  and numerical agreement independently of Python execution.

The existing [lifecycle audit](LIFECYCLE.md) remains in every full suite, covering
repeated calls, independent models, caller-serialized shared access, allocation
counts, poisoned workspaces, and construction-failure cleanup.

## Limits and milestone gate

- Supported verification hosts are macOS arm64 and Linux x86-64. Big-endian and
  other float representations have not been exercised; unsupported encodings
  are rejected. Files must be seekable and stable while read.
- The command processes one already-preprocessed vector. Image decoding,
  batching, `info`, `bench`, and latency/performance claims remain future work.
- Numerical agreement is tolerance-based across platforms. The checkpoint,
  exported model, saved inputs and expected answers were preserved unchanged.
- Native macOS leak checks pass locally; this Mac does not support LeakSanitizer.
  Local ASan uses `detect_leaks=0:halt_on_error=1`; UBSan uses
  `halt_on_error=1:print_stacktrace=1`. Hosted Linux uses LeakSanitizer. The
  earlier hosted native macOS leak step did not complete and was removed; see
  [the recorded limitation](LIFECYCLE.md#hosted-ci). No ThreadSanitizer result is claimed.
- A same-model library call must be serialized by its caller. The CLI runs
  once per process. Output-write failure can leave partial text; host signal
  termination is outside the normal `0/1/2` exit mapping.

M4 closes only when the new code has passed all five hosted jobs and their
exact revision/results have been recorded here. Its original October 19–22
schedule remains a historical target; completion will use the actual date.
