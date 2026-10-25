"""MNIST bytes and the preprocessing shared by training and future fixtures."""

from dataclasses import dataclass
import gzip
import hashlib
import io
from pathlib import Path
import struct
import tempfile
from urllib.request import urlopen

import torch


# Source/checksums: torchvision/datasets/mnist.py in the official pytorch/vision
# repository. Read original IDX bytes directly; torchvision is not a dependency.
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
PREPROCESSING = {
    "input_shape": [28, 28], "input_dtype": "uint8", "output_shape": [784],
    "output_dtype": "float32", "order": "row-major",
    "operation": "cast to float32, then divide by float32 255.0",
    "range": [0.0, 1.0], "augmentation": "none",
}


@dataclass(frozen=True)
class Resource:
    filename: str
    md5: str
    count: int
    images: bool


RESOURCES = (
    Resource("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873", 60000, True),
    Resource("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432", 60000, False),
    Resource("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3", 10000, True),
    Resource("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c", 10000, False),
)


def preprocess_images(images: torch.Tensor) -> torch.Tensor:
    """Convert a nonempty CPU uint8 [N,28,28] batch to float32 [N,784]."""
    if (images.device.type != "cpu" or images.dtype != torch.uint8 or
            images.ndim != 3 or tuple(images.shape[1:]) != (28, 28) or len(images) == 0):
        raise ValueError("images must be a nonempty CPU uint8 tensor with shape [N,28,28]")
    return images.reshape(len(images), 784).to(torch.float32).contiguous().div_(
        torch.tensor(255.0, dtype=torch.float32, device="cpu"))


def decode_idx(archive: bytes, resource: Resource) -> torch.Tensor:
    """Bound decompression and reject wrong headers, sizes, or class labels."""
    header_size, pixels = (16, 784) if resource.images else (8, 1)
    expected = header_size + resource.count * pixels
    with gzip.GzipFile(fileobj=io.BytesIO(archive)) as stream:
        raw = stream.read(expected + 1)
    if len(raw) != expected:
        raise ValueError(f"{resource.filename}: wrong IDX length (expected {expected} bytes)")
    shape = (2051, resource.count, 28, 28) if resource.images else (2049, resource.count)
    if struct.unpack(">" + "I" * len(shape), raw[:header_size]) != shape:
        raise ValueError(f"{resource.filename}: wrong IDX magic, sample count, or dimensions")
    values = torch.frombuffer(bytearray(raw[header_size:]), dtype=torch.uint8)
    if resource.images:
        return values.reshape(resource.count, 28, 28)
    if bool((values > 9).any()):
        raise ValueError(f"{resource.filename}: labels must be in 0..9")
    return values.to(torch.int64)


def _checked_archive(archive: bytes, resource: Resource) -> bytes:
    if not archive or len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError(f"{resource.filename}: empty or oversized archive")
    if hashlib.md5(archive, usedforsecurity=False).hexdigest() != resource.md5:
        raise ValueError(f"{resource.filename}: archive checksum mismatch")
    return archive


def read_archive(directory: Path, resource: Resource, download: bool) -> bytes:
    path = directory / resource.filename
    if path.exists():
        with path.open("rb") as file:
            return _checked_archive(file.read(MAX_ARCHIVE_BYTES + 1), resource)
    if not download:
        raise FileNotFoundError(f"Missing {path}; supply the original archive or use --download")
    with urlopen(MIRROR + resource.filename, timeout=30) as response:
        archive = _checked_archive(response.read(MAX_ARCHIVE_BYTES + 1), resource)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=".mnist-", delete=False) as file:
            temporary = Path(file.name)
            file.write(archive)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return archive


def load_mnist(directory: Path, *, download: bool = False):
    """Return original train/test tensors plus archive provenance; no shuffle."""
    tensors, files = [], []
    for resource in RESOURCES:
        archive = read_archive(directory, resource, download)
        tensors.append(decode_idx(archive, resource))
        files.append({"filename": resource.filename, "url": MIRROR + resource.filename,
                      "md5": resource.md5, "sha256": hashlib.sha256(archive).hexdigest(),
                      "bytes": len(archive), "samples": resource.count})
    return tuple(tensors), {"name": "MNIST", "files": files,
                           "training_split": "official training split",
                           "evaluation_split": "official test split"}
