"""Preserve inputs and every inference stage for official MNIST test indices 0..99."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import struct
import sys
import tempfile

import torch

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from examples.mnist import data
from examples.mnist.export import load_inference_model
from examples.mnist.train import configure_reproducibility
from tools.export_model import export_model


EXAMPLE = Path(__file__).resolve().parent
COUNT = 100
SEED = 20261025
STAGES = (("dense1", 128), ("relu", 128), ("logits", 10), ("probabilities", 10))


def select_test_images(directory: Path, *, download: bool = False):
    """Validate the two complete official test archives, then take indices 0..99."""
    tensors, provenance = [], []
    for resource in data.RESOURCES[2:]:
        archive = data.read_archive(directory, resource, download)
        tensors.append(data.decode_idx(archive, resource)[:COUNT].clone())
        provenance.append({"filename": resource.filename, "url": data.MIRROR + resource.filename,
                           "md5": resource.md5, "sha256": hashlib.sha256(archive).hexdigest(),
                           "bytes": len(archive), "samples": resource.count})
    return *tensors, provenance


def _float_bytes(values: torch.Tensor) -> bytes:
    # Store float32 bits in explicit little-endian order, including signed zero.
    words = values.contiguous().view(torch.int32).reshape(-1).tolist()
    return struct.pack(f"<{len(words)}i", *words)


def build_files(images, labels, archives, checkpoint: Path, model_path: Path) -> dict[str, bytes]:
    """Build a bounded fixture set in memory, without writing the destination.

    images/labels must already be the selected 100 samples. The CLI always
    obtains them from checked official test archives. Offline tests can supply
    the saved selected pixels, labels, and provenance without needing a cache.
    """
    if (images.device.type != "cpu" or images.dtype != torch.uint8 or
            tuple(images.shape) != (COUNT, 28, 28)):
        raise ValueError("expected exactly 100 CPU uint8 images shaped [100,28,28]")
    if (labels.device.type != "cpu" or labels.dtype != torch.int64 or
            tuple(labels.shape) != (COUNT,) or bool(((labels < 0) | (labels > 9)).any())):
        raise ValueError("expected 100 CPU int64 labels in 0..9")

    configure_reproducibility(SEED)
    model = load_inference_model(checkpoint)
    model_bytes = model_path.read_bytes()
    # Tie the answers to the saved binary that Cinder will load later.
    with tempfile.TemporaryDirectory(prefix="cinder-mnist-fixture-model-") as temporary:
        regenerated = Path(temporary) / "mnist.cnd"
        export_model(model, regenerated, input_size=784)
        if regenerated.read_bytes() != model_bytes:
            raise ValueError("saved model does not match the checkpoint export")

    inputs = data.preprocess_images(images)
    stages = {name: [] for name, _ in STAGES}
    with torch.inference_mode():
        # Match the C runtime's single-vector execution; avoid batch-dependent
        # matrix multiplication kernels when defining the reference arithmetic.
        for row in range(COUNT):
            current = inputs[row]
            for layer, (name, width) in zip(model, STAGES, strict=True):
                current = layer(current)
                if (current.dtype != torch.float32 or tuple(current.shape) != (width,) or
                        not bool(torch.isfinite(current).all())):
                    raise ValueError(f"sample {row}, {name}: expected a finite float32 vector of length {width}")
                stages[name].append(current.clone())
    stages = {name: torch.stack(rows) for name, rows in stages.items()}
    probabilities = stages["probabilities"]
    if (bool(((probabilities < 0) | (probabilities > 1)).any()) or
            not bool(torch.allclose(probabilities.sum(-1), torch.ones(COUNT, dtype=torch.float32),
                                    atol=1e-6, rtol=0))):
        raise ValueError("invalid Softmax probability range or normalization")
    predictions = probabilities.argmax(-1)

    files, descriptions = {}, {}

    def add(name, content, shape, encoding):
        files[name] = content
        descriptions[name] = {"shape": shape, "encoding": encoding, "bytes": len(content),
                              "sha256": hashlib.sha256(content).hexdigest()}

    add("images.u8", bytes(images.contiguous().reshape(-1).tolist()), [COUNT, 28, 28], "uint8, row-major")
    add("indices.u32", struct.pack(f"<{COUNT}I", *range(COUNT)), [COUNT], "little-endian uint32")
    add("labels.u32", struct.pack(f"<{COUNT}I", *labels.tolist()), [COUNT], "little-endian uint32")
    encoding = "little-endian IEEE 754 binary32, row-major, no header"
    for name, width in STAGES:
        add(f"{name}.f32", _float_bytes(stages[name]), [COUNT, width], encoding)
    cases = []
    for row in range(COUNT):
        name = f"inputs/{row:05d}.f32"
        add(name, _float_bytes(inputs[row]), [784], encoding)
        cases.append({"row": row, "dataset_index": row, "label": int(labels[row]),
                      "prediction": int(predictions[row]), "input_file": name})

    manifest = {
        "fixture_version": 1, "count": COUNT,
        "selection": {"dataset": "MNIST", "split": "official test split",
                      "rule": "indices 0..99 in original order; no shuffle or prediction filtering",
                      "archives": archives},
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_bytes": len(model_bytes), "preprocessing": data.PREPROCESSING,
        "stages": [{"name": name, "file": f"{name}.f32", "width": width,
                    "row_bytes": width * 4} for name, width in STAGES],
        "comparison": {"atol": 1e-5, "rtol": 1e-5, "require_shape_match": True, "require_finite": True},
        "execution": {"device": "cpu", "dtype": "float32", "threads": 1,
                      "seed": SEED, "deterministic_algorithms": True,
                      "evaluation_mode": True, "gradients": False, "vector_at_a_time": True},
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "platform": platform.platform(), "torch_build": torch.__config__.show()},
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in ("examples/mnist/generate_fixtures.py", "examples/mnist/data.py",
                                       "examples/mnist/train.py", "examples/mnist/export.py", "tools/export_model.py")},
        "subset_result": {"correct": int((predictions == labels).sum()), "samples": COUNT},
        "files": descriptions, "cases": cases,
    }
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return files


def write_files(directory: Path, files: dict[str, bytes]) -> None:
    """Create a new fixture directory; never overwrite existing golden answers."""
    directory.mkdir(exist_ok=False)
    try:
        (directory / "inputs").mkdir()
        for name, content in files.items():
            (directory / name).write_bytes(content)
    except BaseException:
        shutil.rmtree(directory)
        raise


def check_files(directory: Path, expected: dict[str, bytes]) -> None:
    """Compare the complete inventory and bytes without changing any saved file."""
    actual = {p.relative_to(directory).as_posix(): p.read_bytes()
              for p in directory.rglob("*") if p.is_file()}
    changed = [name for name in sorted(actual.keys() | expected.keys())
               if actual.get(name) != expected.get(name)]
    if changed:
        raise ValueError("fixture mismatch: " + ", ".join(changed))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=EXAMPLE / "data")
    parser.add_argument("--checkpoint", type=Path, default=EXAMPLE / "reference/checkpoint.pt")
    parser.add_argument("--model", type=Path, default=EXAMPLE / "mnist.cnd")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/mnist")
    parser.add_argument("--download", action="store_true", help="download missing test archives explicitly")
    parser.add_argument("--check", action="store_true", help="compare regeneration without changing saved files")
    args = parser.parse_args(argv)
    try:
        if not args.check and (args.output.exists() or args.output.is_symlink()):
            raise FileExistsError(f"output already exists: {args.output}; use --check or a new directory")
        images, labels, archives = select_test_images(args.data_dir, download=args.download)
        files = build_files(images, labels, archives, args.checkpoint, args.model)
        if args.check:
            check_files(args.output, files)
        else:
            write_files(args.output, files)
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        print(f"MNIST fixture generation failed: {error}", file=sys.stderr)
        return 1
    print(f"{'Verified' if args.check else 'Generated'} {len(files)} files for test indices 0..99: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
