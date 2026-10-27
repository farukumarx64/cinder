# Deterministic MNIST reference fixtures

[`tests/fixtures/mnist/`](../../tests/fixtures/mnist/) preserves the **first 100
images of the official test split**, with zero-based dataset indices `0..99`.
They remain in original dataset order. There is no random selection, shuffling,
class balancing, or filtering based on whether the model predicts correctly.

The references use the saved [checkpoint](reference/checkpoint.pt) and are tied
to the exported [`mnist.cnd`](mnist.cnd) by SHA-256. Before computing answers, the
generator exports the checkpoint into a temporary file and requires it to match
the saved model byte for byte. A mismatched checkpoint/model pair is rejected.
Neither file is changed, and training is never run.

## What is saved

There are **108 files totaling 548,481 bytes**:

| File(s) | Shape | Encoding | Total bytes |
| --- | --- | --- | ---: |
| `inputs/00000.f32` through `inputs/00099.f32` | 784 values per image | Little-endian float32 | 313,600 |
| `dense1.f32` | 100 × 128 | Little-endian float32, row-major | 51,200 |
| `relu.f32` | 100 × 128 | Little-endian float32, row-major | 51,200 |
| `logits.f32` | 100 × 10 | Little-endian float32, row-major | 4,000 |
| `probabilities.f32` | 100 × 10 | Little-endian float32, row-major | 4,000 |
| `indices.u32` | 100 | Little-endian unsigned 32-bit integers | 400 |
| `labels.u32` | 100 | Little-endian unsigned 32-bit integers | 400 |
| `images.u8` | 100 × 28 × 28 | Original unsigned 8-bit pixels, row-major | 78,400 |
| `manifest.json` | Metadata and 100 case records | UTF-8 JSON | 45,281 |

All binary files contain only the listed values: no header, padding, or footer.
Every individual input is exactly **3,136 bytes**, matching the CLI's raw
input format. `images.u8` keeps just these selected original pixels so offline
tests can independently verify preprocessing without downloading the dataset.
The complete MNIST archives remain outside Git in the existing ignored cache.

Row `r` always refers to the same image across every file. In a stage file of
width `W`, row `r` starts at byte offset `r * W * 4` and contains exactly `W`
float32 values. Indices and labels start at offset `r * 4`; original image pixels
start at offset `r * 784`. Input filenames use the zero-padded dataset index.

The [manifest](../../tests/fixtures/mnist/manifest.json) records:

- Each case's row, dataset index, true label, predicted label, and input filename.
- Every payload file's exact shape, encoding, byte length, and SHA-256. The
  manifest does not hash itself, avoiding a circular dependency.
- Original test-archive URLs, counts, sizes, MD5 checksums, and SHA-256 hashes.
- The checkpoint/model hashes, preprocessing contract, stage widths, execution
  settings, Python/PyTorch/platform/build details, and generator/source hashes.
- Shape/finiteness requirements and the existing `atol = rtol = 1e-5` rule for
  future numerical comparisons.

## How the answers are computed

[`generate_fixtures.py`](generate_fixtures.py) reads only the two official test
archives. The existing reader verifies checksums, IDX headers, exact lengths,
dimensions, counts, and labels before taking the first 100 samples. It does not
read the training archives.

The generator reuses `data.preprocess_images`: flatten each image in row-major
order, cast to float32, and divide by float32 `255.0`. For image pixel `(r,c)`,
the input feature is at `r * 28 + c`. There is no augmentation or further scaling.

The inference model is restored through the export wrapper, including Softmax,
and is in evaluation mode. Computation uses CPU float32, one intra-op thread,
deterministic algorithms, seed `20261025`, and `torch.inference_mode()`.
Selection itself uses no randomness; the seed fixes the surrounding setup.

Each image runs separately as a **one-dimensional 784-value vector**, matching
the C runtime's execution shape. All four stages form one continuous chain:

```text
input → first Dense output → ReLU output → final logits → probabilities
```

Every stage must have the expected shape and finite float32 values. The generator
also checks that probabilities lie in `[0,1]` and sum to one within `1e-6`.
Values are saved by explicitly packing their float32 bit patterns little-endian.

One-vector execution avoids switching to batched matrix multiplication when
defining the oracle. Its rounding can differ from the batch-256 evaluation used
for the full test-accuracy report. These are PyTorch reference values; future
C comparisons must use the saved answers with shape and finite checks and:

```text
abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)
```

That numerical tolerance is separate from **exact fixture regeneration**, which
requires every file byte, including the manifest, to match. Exact regeneration
was verified in the recorded environment; it is not promised across different
platforms, dependency versions, or arithmetic backends. Offline stage-comparison
tests use the numerical tolerance so they do not require macOS bit patterns on Linux.

## Regenerate and check

Reuse the pinned [Python requirements](requirements.txt). From the repository root:

```sh
# Regenerate from the original cached test archives, then compare all files.
.venv/bin/python examples/mnist/generate_fixtures.py --check

# Create a separate directory for review; its parent must exist.
mkdir -p examples/mnist/runs
.venv/bin/python examples/mnist/generate_fixtures.py \
  --output examples/mnist/runs/fixtures-review
```

For a checkout without cached test data, add `--download` explicitly. It downloads
only missing test archives and checks their checksums. `--data-dir`, `--checkpoint`,
and `--model` select other locations. Defaults are anchored to the script/repository,
so direct invocation also works from another working directory.

Without `--check`, generation requires a **new output directory**. It refuses
existing directories instead of overwriting golden answers. An ordinary write
failure removes the newly created partial fixture directory; existing answers
are preserved. Concurrent modification and crash/power-loss durability are not
guaranteed. Keep sources unchanged during generation.

`--check` regenerates the bounded fixture set in memory and compares the entire
saved file inventory. Changed, missing, or extra files are reported as errors.
It never repairs the destination. Successful commands exit zero; reported errors
exit one. Keep commentary files outside the fixture directory so they do not
become unexpected inventory entries.

## Measured results and verification — October 26, 2026

The subset contains **400 stage vectors / 27,600 output values**, plus 78,400
preprocessed input values. The model predicts **99/100** labels correctly.
Index **8** is retained with true label **5** and predicted label **6**. These
fixtures are a deterministic regression sample, not a new estimate of overall
MNIST accuracy. The full official test-split result remains **97.48%**.

Two fresh CLI processes read the original official test archives and generated
separate directories. Both reproduced all 108 saved files byte for byte.
`--check` also passed. The manifest's SHA-256 is:

```text
fb35bb7b0b7775e5d57cf963b2ef0eb392046cd9a67af378758426f35655f635
```

The new [`test_mnist_fixtures.py`](../../tests/test_mnist_fixtures.py) suite has
11 offline tests. They independently check preprocessing for every pixel,
all four stages for all 100 inputs, indices/labels/shape/hash consistency,
test-only archive selection, same-environment process repeatability, mismatched
model rejection, arithmetic overflow rejection, existing-output protection,
partial-write cleanup, and deliberate fixture corruption/missing/extra files.
The tests use the saved selected pixels or synthetic archives; network access
is not needed. This lets CI check the real selected images without a full cache.

```sh
.venv/bin/python -m unittest discover -s tests -p test_mnist_fixtures.py -v
```

At the fixture-generation step, `CINDER_ENABLE_MNIST_TESTS=ON` added `mnist_fixtures` alongside
`mnist_training` and `mnist_export`. The existing Linux CI configuration enables
that option. Ordinary C tests still require neither Python nor MNIST downloads.
Local fixture tests passed 11/11 on macOS arm64 with Python 3.12.14 and PyTorch
2.14.1. The rebuilt combined Debug suite passed **31/31 CTest tests**, including
all three MNIST suites. The MNIST-only configuration registered all three without
requiring the exporter/NumPy test options. With Python options OFF, the C-only
Release configuration registered 23 C tests without Python discovery at that step.
Python compilation and whitespace checks passed. No C runtime changes or new
sanitizer runs were part of the fixture-generation step. The subsequent
[C-versus-Python validation](VALIDATION.md) passes all 27,600 values and reports
100/100 prediction agreement, with 99/100 subset accuracy for each runtime.
That step adds C-only comparison, fresh-export integration, corruption controls,
and local Debug/Release/sanitizer/leak verification. The subsequent
[lifecycle step](LIFECYCLE.md) also passes all five hosted jobs.

The [`cinder run` command](../../docs/CLI.md) now reads these inputs and prints
zero-based output indices and up to three raw scores (probabilities for this
Softmax model). See [the demonstration](DEMO.md) for actual output, CLI tests,
and M4 verification status. The fixture answers remain unchanged.
