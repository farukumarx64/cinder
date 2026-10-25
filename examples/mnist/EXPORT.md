# Trained MNIST model export

The saved reference checkpoint is exported as
[`mnist.cnd`](mnist.cnd): **407,160 bytes**, containing this sequence:

```text
Dense(784,128) -> ReLU -> Dense(128,10) -> Softmax(10)
```

The input is one vector of 784 float32 values, using the existing row-major
flattening and `/255` preprocessing. The output is ten probabilities, with
indices `0..9` corresponding to digit labels. The model contains the same
101,770 trained parameters as the [reference checkpoint](reference/checkpoint.pt).
Softmax adds no parameters; it operates on the final feature axis (`dim=-1`).

The [97.48% reference accuracy](RESULTS.md) was measured in Python during the
previous training step. This step verifies binary export and parameter identity;
it does not yet measure Cinder's accuracy or compare C/Python MNIST outputs.

## Restoring the checkpoint

[`export.py`](export.py) is a small example-specific wrapper around the existing
[`tools/export_model.py`](../../tools/export_model.py). It does not implement a
second binary writer, perform training, or read/download MNIST images.

`load_inference_model(checkpoint_path)`:

1. Reads the checkpoint using `torch.load(..., map_location="cpu", weights_only=True)`.
2. Requires a `model_state_dict` with exactly `hidden.weight`, `hidden.bias`,
   `logits.weight`, and `logits.bias`. Checks ordinary strided CPU float32
   tensors, exact shapes, and finite values before restoration. This prevents
   `load_state_dict` from silently converting a wrong parameter dtype.
3. Creates the original flat `784 -> 128 -> 10` model and restores all parameters
   with `strict=True`. Checkpoint metadata does not change the architecture.
4. Appends `nn.Softmax(dim=-1)` as the `probabilities` module, then calls `eval()`
   on the complete sequence, including the newly appended activation.

The training checkpoint is read-only and retains its logits-only architecture
and metadata. The inference model is a separate object. Changing the random seed
or the default floating-point dtype does not change exported bytes: the model is
explicitly float32 and every initialized parameter is replaced by saved values.

The wrapper calls `export_model(model, output, input_size=784)`. The existing
exporter checks connections and resource limits, copies parameters, and writes
the authoritative [.cnd version 1 format](../../docs/MODEL_FORMAT.md).

## Commands and file behavior

Use the existing pinned [requirements](requirements.txt). From the repository root:

```sh
# Regenerate the saved model from the saved checkpoint.
.venv/bin/python examples/mnist/export.py

# Compare a temporary regeneration without changing the saved model.
.venv/bin/python examples/mnist/export.py --check

# Select another compatible checkpoint and destination explicitly.
.venv/bin/python examples/mnist/export.py \
  --checkpoint examples/mnist/reference/checkpoint.pt \
  --output /tmp/cinder-mnist.cnd
```

Defaults are anchored to the example directory, so direct execution works from
other working directories. `python -m examples.mnist.export --check` also works
from the repository root. Success exits with status zero; checkpoint, validation,
file, and byte-mismatch errors reported by the CLI exit with status one.

The destination parent must exist. Successful export atomically replaces an
existing destination through the existing writer. Invalid checkpoints and
ordinary writer failures preserve an existing output. The selected checkpoint
path, including a symlink resolving to it, cannot be used as the output.

`--check` exports into a temporary directory, compares the entire byte sequences,
then removes the temporary output. A missing or different destination fails;
the comparison never repairs or overwrites it. Export does not change the
checkpoint, `run.json`, training log, or training verification record.

## Byte inspection

The 16-byte file header is:

```text
43 4e 44 52 01 00 00 00 10 03 00 00 04 00 00 00
```

It records `CNDR`, format version `1`, input size `784`, and four layers.
Every layer has a 16-byte header. All integer fields and IEEE binary32 parameter
words are little-endian; Dense weights are stored row-major, followed by biases.

| Layer | Record offset | Input → output | Parameter payload |
| --- | ---: | --- | ---: |
| First Dense | 16 | 784 → 128 | 401,920 bytes |
| ReLU | 401,952 | 128 → 128 | 0 bytes |
| Second Dense | 401,968 | 128 → 10 | 5,160 bytes |
| Softmax | 407,144 | 10 → 10 | 0 bytes |

The complete length is `16 + 4 * 16 + 407080 = 407160` bytes. There is no
padding, footer, training metadata, filename, or timestamp in the model file.
The SHA-256 of the saved export is:

```text
b0376dd2562f0a89d95cc4f9f1172005e293d0ab9c4a0c9c345fff2dd98fb8bd
```

[`reference/export.json`](reference/export.json) records this hash, the source
checkpoint hash, byte layout, source-code hashes, environment, and verification
results separately from the model file. The original training records remain
unchanged.

## Verification — October 26, 2026

Two independent Python processes restored the same checkpoint and exported to
different temporary paths. Both outputs matched each other and the saved
`mnist.cnd` byte for byte. An independent `struct` decoder checked the header,
all four record headers, exact EOF, and all 101,770 weight/bias bit patterns
against the checkpoint. The checkpoint's original SHA-256 remained unchanged.

The new [`test_mnist_export.py`](../../tests/test_mnist_export.py) suite has
11 tests. It covers:

- Flat architecture, evaluation mode throughout, exact restoration, and Softmax
  over features for both a single input and a batch. Synthetic probes produce
  finite probabilities summing to one and retain the logits' predicted class.
- Independent byte inspection and exact comparison against the saved artifact
  and its checkpoint/hash record.
- Different random seeds/default dtypes, fresh-process regeneration, and a
  changed checkpoint parameter that must change the export.
- Missing/extra keys, wrong shapes/dtypes/storage, and non-finite parameters;
  rejected checkpoints leave existing output untouched.
- Missing/corrupt checkpoint files, output-path failures, checkpoint overwrite
  protection, and byte-mismatch detection without modifying the destination.

Run these checks without any dataset or training:

```sh
.venv/bin/python -m unittest discover -s tests -p test_mnist_export.py -v
```

`CINDER_ENABLE_MNIST_TESTS=ON` now registers both `mnist_training` and
`mnist_export`. The existing Linux CI configuration already enables that option.
Ordinary C builds still need no Python, PyTorch, or MNIST downloads.

Local verification used macOS 27.0.1 arm64, Python 3.12.14, and PyTorch 2.14.1.
The dedicated export suite passed 11/11 tests; the rebuilt combined Debug suite
passed 30/30 CTest tests, including both MNIST suites and the existing NumPy,
exporter, and C round-trip checks. The MNIST-only configuration registers both
suites independently of the NumPy/exporter options. With all Python options OFF,
the C-only Release configuration still registers 23 C tests without Python
discovery. The module-form `--check`, Python compilation, and whitespace checks
also passed. Hosted verification is pending an authorized commit/push; no C
runtime changes or new sanitizer runs are part of this step.

The integration run exposed an existing training CLI test that assumed the
default run directory was absent. Real training had created that directory.
The test now supplies a temporary output path so its missing-data check does
not depend on local experiments. Training code and saved training artifacts
remain unchanged.

Next: generate deterministic real-image reference fixtures from this saved
checkpoint, including inputs and Dense1/ReLU/logits/probability outputs.
The later C comparisons and lifecycle audit are still required to close M4.
