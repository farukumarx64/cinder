# Reproducible MNIST reference training

Phase 4 starts with this external Python example. The training script and its
offline synthetic tests are implemented; a full MNIST training/evaluation run,
trained `.cnd` export, and real-image C comparisons remain subsequent steps.
No digit-recognition accuracy is claimed by the synthetic tests.

## Model and preprocessing

`train.py` creates a flat CPU float32 `torch.nn.Sequential`:

```text
784 inputs -> Linear(784, 128) -> ReLU -> Linear(128, 10) -> 10 logits
```

Both Linear layers include biases: 101,770 learned parameters in total. The
named layers are `hidden`, `relu`, and `logits`. Flattening is outside the model,
so its modules are already supported by Cinder's existing exporter.

Training uses `CrossEntropyLoss` on unnormalized logits and integer labels
`0..9`. There is no Softmax in the training model. A future export step will
append `Softmax(dim=-1)` to produce probabilities. This follows PyTorch's
[CrossEntropyLoss contract](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html).

`data.preprocess_images` implements the existing
[Cinder preprocessing contract](../../docs/MODEL_FORMAT.md#input-file-and-preprocessing-contract):

1. Accept a nonempty batch of CPU `uint8` images shaped `[N,28,28]`.
2. Flatten each image in row-major order; pixel `(row,column)` becomes feature
   `row * 28 + column`.
3. Convert to float32, then divide by float32 `255.0`, producing `[N,784]` in
   `[0,1]`. Original pixels remain unchanged.

Training and evaluation call this same helper. Future sample/reference
generation must reuse it. There is no mean/std normalization, resizing,
augmentation, or preprocessing inside the C runtime. One future raw C input
file contains 784 little-endian float32 values, exactly 3,136 bytes.

## Dependencies and data

Python 3.11+ and PyTorch are the only direct requirements. `requirements.txt`
includes the existing exporter pin, currently `torch==2.14.1`; the example uses
neither torchvision nor NumPy APIs. From the repository root, reuse the exporter
environment or install the requirements into a virtual environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r examples/mnist/requirements.txt
```

For Linux CPU wheels:

```sh
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  -r examples/mnist/requirements.txt
```

`data.py` reads the four original compressed IDX archives: training images and
labels (60,000 each), and test images and labels (10,000 each). Archive checksums
and the HTTPS mirror come from the official
[torchvision MNIST implementation](https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py).
Existing archives are checked before decoding. The reader validates IDX magic,
counts, image dimensions, exact decompressed lengths, and labels in `0..9`.
Archive reads and decompression are bounded to the expected dataset sizes.

Missing files cause an error unless `--download` is explicitly supplied.
Downloads are checksum-checked before being saved through a temporary file.
An existing corrupt archive is rejected, not silently replaced. The cache
defaults to `examples/mnist/data/`; an existing torchvision cache's `MNIST/raw`
directory works if it retains the four original `.gz` files.

## Training settings

| Setting | Default/behavior |
| --- | --- |
| Seed | `20261025`; configurable with `--seed` |
| Epochs | `5`; configurable with `--epochs` |
| Training batch size | `64`; configurable with `--batch-size` |
| Learning rate | `0.001`; configurable with `--learning-rate` |
| Optimizer | Adam; betas `(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, no AMSGrad; foreach/fused paths disabled |
| Loss | Mean CrossEntropyLoss, no label smoothing |
| Execution | CPU float32, one intra-op thread, deterministic algorithms required |
| Training order | Shuffle every epoch with a dedicated seeded CPU generator |
| Data-loader workers | `0`; no dropped final batch |
| Evaluation | Fixed order, batch size `256`, evaluation mode, gradients disabled |
| Checkpoint choice | Final requested epoch; no selection using test results |

Each training batch clears gradients, computes logits/loss, backpropagates, and
updates parameters. Reported epoch loss is weighted by sample count so a short
last batch is counted correctly. Training accuracy describes predictions made
during that epoch as parameters change. Non-finite losses or parameters abort
the run. The official test split is evaluated once after all epochs; it never
participates in gradient updates or checkpoint selection.

Python and PyTorch RNGs are seeded before model construction. Shuffling uses a
separate generator so initialization cannot consume its random sequence.
Repeatability is tested for the same environment, inputs, seed, and settings.
Identical results across different PyTorch versions, operating systems, CPUs,
or backends are not guaranteed; see PyTorch's
[reproducibility guidance](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Run and reload

The following command performs real MNIST training; it has **not** been run as
part of the script-implementation step:

```sh
.venv/bin/python examples/mnist/train.py --download \
  --seed 20261025 --epochs 5 --batch-size 64 --learning-rate 0.001 \
  --output-dir examples/mnist/runs/reference
```

Omit `--download` to require cached files. Use `--data-dir PATH` to select a
different cache. Defaults are anchored to the script directory, so direct
execution also works outside the repository. `--help` lists the options.

The output directory must not already exist. This preserves previous runs and
forces explicit names for new experiments. A completed run writes:

- `checkpoint.pt`: copied CPU parameter tensors under `model_state_dict`, plus
  the same metadata contained in the JSON report. It saves parameters rather
  than pickling a model class. This is a final inference checkpoint; training
  resume/optimizer-state restoration is not implemented.
- `run.json`: architecture, preprocessing, seed/settings, actual optimizer
  defaults, per-epoch training metrics, final test metrics, dataset filenames
  and MD5/SHA-256 hashes, Python/PyTorch versions, platform and PyTorch build
  details, and hashes of the example source/requirements files. Accuracy is
  stored as a fraction and printed as a percentage.

The JSON report is written last. Ordinary write failures remove the new partial
checkpoint/report and run directory; existing runs are never overwritten.
Crash/power-loss durability and concurrent writes to the same cache/run paths
are not guaranteed. Default data/run directories are Git-ignored, while future
reviewed model/reference fixtures can be tracked explicitly elsewhere.

From the repository root, reload for reference inference with:

```python
import torch
from examples.mnist.train import build_model

checkpoint = torch.load("examples/mnist/runs/reference/checkpoint.pt",
                        map_location="cpu", weights_only=True)
model = build_model()
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
model.eval()  # Returns logits; probabilities are added at the export step.
```

See PyTorch's [state-dictionary guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
Reproducibility checks compare tensors and recorded results; `.pt` container
bytes are not the reproducibility contract.

## Offline tests

```sh
.venv/bin/python -m unittest discover -s tests -p test_mnist_training.py -v
```

The suite uses 23 synthetic training samples and 11 synthetic evaluation
samples. It exercises real gradients/optimizer updates, exact preprocessing,
IDX rejection, mocked downloads, different seeds, test-data isolation,
sample-weighted loss, checkpoint reload, CLI errors, and write-failure cleanup.
Two fresh subprocesses must produce identical parameter tensors and metadata.
These samples are test machinery; their metrics are not MNIST accuracy.

To register the suite with CTest independently of exporter/NumPy tests:

```sh
cmake -S . -B build-mnist -DCMAKE_BUILD_TYPE=Debug \
  -DCINDER_ENABLE_MNIST_TESTS=ON -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-mnist --parallel
ctest --test-dir build-mnist -R '^mnist_training$' --output-on-failure
```

The new option defaults to OFF, requires `BUILD_TESTING=ON` and an importable
PyTorch, and installs nothing. Ordinary C builds still need no Python. The
Linux CI job enables the offline suite alongside the existing Python checks;
it does not download MNIST or run full training.

## Verification — October 25, 2026

Local checks used Apple Silicon macOS, Apple Clang 21.0.0, CMake 4.4.4,
Python 3.12.14, and PyTorch 2.14.1. The combined suite also used NumPy 2.3.5.

| Check | Actual result |
| --- | --- |
| Offline training-script suite | 18/18 unittest methods passed |
| Fresh Debug build, NumPy/exporter/MNIST options enabled | 29/29 CTest tests passed |
| Fresh default C-only Release build | 23/23 passed; no Python discovery or execution |
| MNIST option enabled alone | `mnist_training` passed with exporter/NumPy options OFF |
| MNIST option with `BUILD_TESTING=OFF` | Configuration rejected with the expected diagnostic |
| Python source compilation and whitespace checks | Passed |

The independent-process test reproduced every parameter tensor and all recorded
metadata exactly in this environment. Download behavior was tested with mocked
responses; no live MNIST download or full-dataset training was performed.
Hosted CI for this addition remains pending a future authorized commit/push.
Next: run real MNIST training/evaluation and record the measured results before
exporting a trained model or claiming M4 completion.
