"""Train Cinder's reproducible CPU MNIST reference model; outputs are logits."""

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .data import PREPROCESSING, load_mnist, preprocess_images
else:  # Also support: python examples/mnist/train.py ...
    from data import PREPROCESSING, load_mnist, preprocess_images


EXAMPLE = Path(__file__).resolve().parent
ARCHITECTURE = ["Dense(784,128,bias=True)", "ReLU", "Dense(128,10,bias=True)"]


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20261025
    epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.001

    def validate(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in 0..4294967295")
        for name in ("epochs", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")


def build_model() -> nn.Sequential:
    """A flat float32 CPU sequence compatible with the existing exporter."""
    return nn.Sequential(OrderedDict([
        ("hidden", nn.Linear(784, 128, bias=True, device="cpu", dtype=torch.float32)),
        ("relu", nn.ReLU()),
        ("logits", nn.Linear(128, 10, bias=True, device="cpu", dtype=torch.float32)),
    ]))


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _dataset(images: torch.Tensor, labels: torch.Tensor) -> TensorDataset:
    # Validate without allocating a float32 copy of the complete dataset.
    if (images.device.type != "cpu" or images.dtype != torch.uint8 or
            images.ndim != 3 or tuple(images.shape[1:]) != (28, 28) or len(images) == 0):
        raise ValueError("images must be nonempty CPU uint8 [N,28,28]")
    if (labels.device.type != "cpu" or labels.dtype != torch.int64 or
            tuple(labels.shape) != (len(images),) or
            bool(((labels < 0) | (labels > 9)).any())):
        raise ValueError("labels must be CPU int64 [N], matching images, with values in 0..9")
    return TensorDataset(images, labels)


def _metrics(loss_sum: float, correct: int, samples: int) -> dict:
    return {"loss": loss_sum / samples, "correct": correct,
            "samples": samples, "accuracy": correct / samples}


def train_network(train_images, train_labels, test_images, test_labels,
                  config: TrainingConfig, *, progress: bool = False):
    """Fixed-epoch training followed by one held-out evaluation; no selection."""
    config.validate()
    training = _dataset(train_images, train_labels)
    testing = _dataset(test_images, test_labels)
    configure_reproducibility(config.seed)
    model = build_model()
    # Model initialization and batch shuffling have separate random streams.
    shuffle = torch.Generator(device="cpu").manual_seed(config.seed)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True,
                        generator=shuffle, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                                 amsgrad=False, foreach=False, fused=False)
    criterion = nn.CrossEntropyLoss(reduction="mean")
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum, correct, samples = 0.0, 0, 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(preprocess_images(images))
            loss = criterion(logits, labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"non-finite training loss in epoch {epoch}")
            loss.backward()
            optimizer.step()
            count = len(labels)
            loss_sum += float(loss.detach()) * count
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            samples += count
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise ValueError(f"non-finite model parameters in epoch {epoch}")
        history.append({"epoch": epoch, **_metrics(loss_sum, correct, samples)})
        if progress:
            print(f"epoch {epoch}/{config.epochs}: training loss={history[-1]['loss']:.6f}, "
                  f"accuracy={history[-1]['accuracy']:.2%}", flush=True)

    # The official test split never participates in gradients or checkpoint
    # selection. Evaluate once, after the requested fixed number of epochs.
    model.eval()
    loss_sum, correct, samples = 0.0, 0, 0
    evaluation = DataLoader(testing, batch_size=256, shuffle=False, num_workers=0,
                            generator=torch.Generator(device="cpu").manual_seed(config.seed))
    with torch.inference_mode():
        for images, labels in evaluation:
            logits = model(preprocess_images(images))
            loss = criterion(logits, labels)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise ValueError("non-finite evaluation result")
            loss_sum += float(loss) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum())
            samples += len(labels)
    return model, {"training_history": history, "test": _metrics(loss_sum, correct, samples),
                   "optimizer": {"name": "Adam", **optimizer.defaults,
                                 "betas": list(optimizer.defaults["betas"])}}


def run_metadata(config: TrainingConfig, dataset: dict, results: dict) -> dict:
    return {
        "format_version": 1, "architecture": ARCHITECTURE,
        "parameter_count": 101770, "outputs": "logits; no Softmax during training",
        "preprocessing": PREPROCESSING, "config": asdict(config), "dataset": dataset,
        "execution": {"device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
                      "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                      "shuffle": True, "shuffle_seed": config.seed, "num_workers": 0,
                      "drop_last": False, "evaluation_batch_size": 256,
                      "loss": "CrossEntropyLoss(mean)", "label_smoothing": 0.0,
                      "checkpoint_selection": "final epoch; no test-set selection"},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "machine": platform.machine(),
                        "torch_build": torch.__config__.show()},
        "source_sha256": {name: hashlib.sha256((EXAMPLE / name).read_bytes()).hexdigest()
                          for name in ("train.py", "data.py", "requirements.txt")},
        **results,
    }


def save_run(directory: Path, model: nn.Sequential, metadata: dict) -> None:
    """Save a final-epoch inference checkpoint and its human-readable record."""
    # Validate JSON before creating anything; run.json is written last.
    record = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
    directory.mkdir(parents=True, exist_ok=False)
    checkpoint, report = directory / "checkpoint.pt", directory / "run.json"
    try:
        state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        torch.save({"model_state_dict": state, "metadata": metadata}, checkpoint)
        report.write_text(record, encoding="utf-8")
    except BaseException:
        checkpoint.unlink(missing_ok=True)
        report.unlink(missing_ok=True)
        directory.rmdir()
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXAMPLE / "data",
                        help="directory containing the four original MNIST .gz archives")
    parser.add_argument("--output-dir", type=Path, default=EXAMPLE / "runs" / "reference",
                        help="new directory for checkpoint.pt and run.json; must not exist")
    parser.add_argument("--download", action="store_true", help="download missing MNIST archives")
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    args = parser.parse_args(argv)
    config = TrainingConfig(args.seed, args.epochs, args.batch_size, args.learning_rate)
    try:
        config.validate()
        if args.output_dir.exists():
            raise FileExistsError(f"Output directory already exists: {args.output_dir}")
        tensors, dataset = load_mnist(args.data_dir, download=args.download)
        model, results = train_network(*tensors, config, progress=True)
        save_run(args.output_dir, model, run_metadata(config, dataset, results))
    except (OSError, ValueError, RuntimeError, EOFError) as error:
        print(f"MNIST training failed: {error}", file=sys.stderr)
        return 1
    print(f"test accuracy={results['test']['accuracy']:.2%}; saved {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
