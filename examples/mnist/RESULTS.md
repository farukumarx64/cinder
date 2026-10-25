# MNIST reference training results — October 25, 2026

The Python reference model achieved **97.48% accuracy on the official MNIST
test split: 9,748 correct predictions out of 10,000**, with 252 errors and
mean cross-entropy loss **0.0826597822714597**. These are measured results from
the saved final-epoch checkpoint, not estimates from synthetic tests.

The model is `Dense(784,128) -> ReLU -> Dense(128,10)`, with biases and 101,770
float32 parameters. Its outputs are logits. These measurements concern Python
training/evaluation. The subsequent [export step](EXPORT.md) produces a trained
`.cnd` model with Softmax; MNIST inference in Cinder is still a later step.

## Fixed settings and split isolation

Training used the script at revision
`e8ebcfc528731aa6e33e2000747376d9b2884bee`, without changing its implementation.
The settings were fixed before looking at the test result:

| Setting | Actual value |
| --- | --- |
| Seed | `20261025` |
| Training data | Original 60,000 training images and labels |
| Evaluation data | Original 10,000 test images and labels |
| Epochs | 5; final epoch retained |
| Batch size | 64; last batch retained; 938 updates per epoch, 4,690 total |
| Optimizer | Adam, learning rate `0.001`, betas `(0.9,0.999)`, epsilon `1e-8`, no weight decay or AMSGrad; foreach/fused disabled |
| Loss | Mean CrossEntropyLoss on logits, no label smoothing |
| Preprocessing | Row-major flattening, cast uint8 to float32, divide by float32 `255.0`; no augmentation or further normalization |
| Execution | CPU float32, one intra-op thread, deterministic algorithms, zero data-loader workers |
| Evaluation | `eval()` and `inference_mode()`, fixed order, batch size 256 |
| Environment | macOS 27.0.1 arm64, Python 3.12.14, PyTorch 2.14.1 |

The optimizer's data loader contains only the official training split. The
test loader runs after all five epochs, with gradients disabled. Test examples
were never used for optimizer updates, early stopping, hyperparameter changes,
or checkpoint selection. The first completed run remains the reference. A
second run used exactly the same settings solely to check reproducibility.

The four original archives were downloaded through the existing reader, checked
against their expected MD5 checksums, and validated for IDX headers, lengths,
dimensions, sample counts, and label range. Their SHA-256 hashes and sizes are
saved in [run.json](reference/run.json). The downloaded archives remain in the
ignored `data/` cache; the dataset is not added to the repository.

## Measured training history

| Epoch | Training samples | Mean training loss | Training accuracy |
| ---: | ---: | ---: | ---: |
| 1 | 60,000 | 0.3463040715 | 90.78% |
| 2 | 60,000 | 0.1567024002 | 95.42% |
| 3 | 60,000 | 0.1102760812 | 96.73% |
| 4 | 60,000 | 0.0837146755 | 97.50% |
| 5 | 60,000 | 0.0666776966 | 97.99% |

Training metrics describe predictions made during each epoch while parameters
were changing. They are sample-weighted, including the short final batch; they
are not a separate evaluation of the final checkpoint on the training set.
The **97.48% test accuracy** is measured after the fifth epoch, using the fixed
final parameters on all 10,000 held-out test samples.

## Saved reference artifacts

These files are intentionally outside the ignored experiment directory so the
reference parameters and evidence can be versioned with the example:

| File | Contents |
| --- | --- |
| [reference/checkpoint.pt](reference/checkpoint.pt) | Final CPU parameter tensors and metadata; 414,587 bytes |
| [reference/run.json](reference/run.json) | Exact configuration, optimizer settings, per-epoch metrics, test metrics, architecture, preprocessing, dataset hashes, environment/build information, and source hashes |
| [reference/training.log](reference/training.log) | Console output from the first successful full training run |
| [reference/verification.json](reference/verification.json) | Source revision, actual commands, artifact sizes/hashes, fresh-process reload checks, repeat-training comparison, and limitations |

The checkpoint contains four tensors: `hidden.weight [128,784]`,
`hidden.bias [128]`, `logits.weight [10,128]`, and `logits.bias [10]`. All are
finite float32 values. It is an inference checkpoint; optimizer state for
resuming training is not included. Load it using `weights_only=True` and
`map_location="cpu"`, then restore it into `build_model()` with `strict=True`.

The preserved checkpoint's SHA-256 is:

```text
f388dd5d4a058f6b62013685b9f41c4cb6994a8de5d04d2e8b1c00be2b4aeaea
```

This hash identifies the saved artifact. Reproducible training is checked using
parameter values and metadata, not by requiring identical `.pt` container bytes.

## Verification performed

1. **Reload in a fresh Python process:** strictly restored all four parameter
   tensors, checked their shapes, dtype, finiteness, total parameter count,
   metadata, and source hashes. Read only the two official test archives and
   evaluated 40 batches, including the final 16 samples. All logits were finite.
   Correct count, accuracy, and sample-weighted loss exactly matched `run.json`.
   Parameters remained unchanged and no gradients were created.
2. **Repeat full training in another fresh process:** used cached archives and
   the unchanged seed/settings. All 101,770 float32 parameter values matched
   bit for bit, all metadata matched exactly, and `run.json` matched byte for
   byte. The repeated checkpoint was not substituted for the original.
3. **Preserve and check artifacts:** copied the first run's checkpoint, JSON,
   and log without reserializing them; recorded their sizes and SHA-256 hashes.

Exact repeatability was observed in the recorded macOS environment. It is not a
guarantee for other CPUs, operating systems, or dependency versions. There is
no separate validation split or hyperparameter search in this fixed experiment.
Repeated test evaluation here checks saved-model integrity; it does not supply
additional independent evidence of generalization.

## Reproduce training

Install the pinned [requirements](requirements.txt) as described in the
[example guide](README.md). From the repository root, use a new output directory:

```sh
.venv/bin/python examples/mnist/train.py --download \
  --seed 20261025 --epochs 5 --batch-size 64 --learning-rate 0.001 \
  --output-dir examples/mnist/runs/reproduce
```

An existing output directory is rejected. `--download` fetches only missing
archives; omit it to require cached data. The two actual training commands are
recorded in `reference/verification.json`.

After reproducing in the same environment, compare the saved parameters and
metadata with the original:

```sh
.venv/bin/python - <<'PY'
import torch

a = torch.load("examples/mnist/reference/checkpoint.pt",
               map_location="cpu", weights_only=True)
b = torch.load("examples/mnist/runs/reproduce/checkpoint.pt",
               map_location="cpu", weights_only=True)
assert a["model_state_dict"].keys() == b["model_state_dict"].keys()
for name, tensor in a["model_state_dict"].items():
    assert torch.equal(tensor.view(torch.int32),
                       b["model_state_dict"][name].view(torch.int32)), name
assert a["metadata"] == b["metadata"]
print("All parameters and metadata match exactly.")
PY
```

## Re-evaluate the saved checkpoint

This command does not train, download data, or modify the checkpoint. It uses
the cached test archives and the shared preprocessing helper. The exact metric
assertion matches the verification performed in the recorded environment:

```sh
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import torch
from examples.mnist.data import RESOURCES, decode_idx, read_archive, preprocess_images
from examples.mnist.train import build_model, configure_reproducibility

root = Path("examples/mnist")
report = json.loads((root / "reference/run.json").read_text())
checkpoint = torch.load(root / "reference/checkpoint.pt",
                        map_location="cpu", weights_only=True)
assert checkpoint["metadata"] == report
configure_reproducibility(report["config"]["seed"])
model = build_model()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()
images, labels = [decode_idx(read_archive(root / "data", r, False), r)
                  for r in RESOURCES[2:]]
assert len(images) == len(labels) == 10000
loss_sum, correct = 0.0, 0
with torch.inference_mode():
    for start in range(0, len(labels), 256):
        target = labels[start:start + 256]
        logits = model(preprocess_images(images[start:start + 256]))
        assert tuple(logits.shape) == (len(target), 10)
        assert bool(torch.isfinite(logits).all())
        loss = torch.nn.functional.cross_entropy(logits, target)
        assert bool(torch.isfinite(loss))
        loss_sum += float(loss) * len(target)
        correct += int((logits.argmax(dim=1) == target).sum())
actual = {"correct": correct, "samples": len(labels),
          "accuracy": correct / len(labels), "loss": loss_sum / len(labels)}
assert actual == report["test"], actual
print(json.dumps(actual, indent=2))
PY
```

## Remaining work

The subsequent [trained export](EXPORT.md) is now implemented and verified from
this checkpoint. Real-image stage fixtures, the `cinder run` CLI, C/Python
numerical comparisons, and larger-model lifecycle verification remain separate
steps. M4 remains open. Ordinary C tests still need neither Python nor MNIST
downloads. No hosted CI or full C rebuild was run for the original
artifact/documentation-only training step; export checks are recorded separately.
