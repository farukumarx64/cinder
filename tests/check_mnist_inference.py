"""Compare a fresh checkpoint export in C with saved Python MNIST stages, offline."""

import argparse
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.mnist.export import load_inference_model
from tools.export_model import export_model


def check_inference(executable: Path) -> None:
    executable = executable.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="cinder mnist comparison ") as temporary:
        directory = Path(temporary)
        fixtures = directory / "fixtures"
        scratch = directory / "scratch"
        scratch.mkdir()
        shutil.copytree(ROOT / "tests/fixtures/mnist", fixtures)
        # Export parameters afresh, but never regenerate the expected answers.
        model = load_inference_model(ROOT / "examples/mnist/reference/checkpoint.pt")
        model_path = directory / "mnist.cnd"
        export_model(model, model_path, input_size=784)
        original_model = model_path.read_bytes()
        if original_model != (ROOT / "examples/mnist/mnist.cnd").read_bytes():
            raise AssertionError("Fresh checkpoint export differs from saved mnist.cnd")

        # Independently confirm that the C test's fixed record boundaries give
        # precisely the prefixes produced by the production Python exporter.
        for count, end in enumerate((401952, 401968, 407144), start=1):
            prefix = directory / f"prefix-{count}.cnd"
            export_model(model[:count], prefix, input_size=784)
            expected = bytearray(original_model[:end])
            struct.pack_into("<I", expected, 12, count)
            if prefix.read_bytes() != expected:
                raise AssertionError(f"Prefix {count} differs from the C test's record boundary")

        def run():
            result = subprocess.run([str(executable), str(model_path), str(fixtures), str(scratch)],
                                    cwd=directory, capture_output=True, text=True, timeout=30)
            if list(scratch.iterdir()):
                raise AssertionError("C test did not clean up its temporary prefix files")
            return result

        result = run()
        if result.returncode != 0:
            raise AssertionError(f"Fresh MNIST export/load/inference failed ({result.returncode}):\n"
                                 f"{result.stdout}{result.stderr}")
        print(result.stdout, end="")
        rejected = 0

        def reject(path: Path, changed: bytes, diagnostic: str):
            nonlocal rejected
            original = path.read_bytes()
            try:
                path.write_bytes(changed)
                result = run()
                if result.returncode != 1 or diagnostic not in result.stderr:
                    raise AssertionError(f"Expected clean rejection containing {diagnostic!r}, "
                                         f"got exit {result.returncode}:\n{result.stdout}{result.stderr}")
                rejected += 1
            finally:
                path.write_bytes(original)

        for stage, width in (("dense1", 128), ("relu", 128), ("logits", 10), ("probabilities", 10)):
            path = fixtures / f"{stage}.f32"
            original = path.read_bytes()
            changed = bytearray(original)
            struct.pack_into("<f", changed, 0, struct.unpack_from("<f", original)[0] + 1.0)
            reject(path, changed, f"inputs/00000.f32/{stage}[")
            reject(path, original[:-4], "expected exactly")
            reject(path, original + b"\0", "expected exactly")
            # Check both ends of the saved array, including every last stage.
            offset = (100 * width - 1) * 4
            changed = bytearray(original)
            struct.pack_into("<f", changed, offset, float("nan"))
            reject(path, changed, "non-finite fixture value")
            changed = bytearray(original)
            struct.pack_into("<f", changed, offset, struct.unpack_from("<f", original, offset)[0] + 1.0)
            reject(path, changed, f"inputs/00099.f32/{stage}[")

        path = fixtures / "inputs/00000.f32"
        original = path.read_bytes()
        reject(path, original[:-1], "expected exactly 3136 bytes")
        reject(path, original + b"\0", "expected exactly 3136 bytes")
        reject(path, struct.pack("<f", float("inf")) + original[4:], "non-finite fixture value")
        reject(path, struct.pack("<f", 1.0) + original[4:], "input differs from saved pixel / 255")
        for name, value, diagnostic in (("indices.u32", 1, "decode_word(indices"),
                                        ("labels.u32", 10, "label < 10")):
            path = fixtures / name
            reject(path, struct.pack("<I", value) + path.read_bytes()[4:], diagnostic)

        # A valid finite parameter change must be caught by numerical agreement,
        # rather than merely by a malformed-model rejection in the loader.
        changed = bytearray(original_model)
        bias_offset = 32 + 784 * 128 * 4
        struct.pack_into("<f", changed, bias_offset,
                         struct.unpack_from("<f", original_model, bias_offset)[0] + 1.0)
        reject(model_path, changed, "inputs/00000.f32/dense1[")
        # Finite parameters can overflow during inference: require a reported
        # runtime failure, never acceptance of non-finite actual outputs.
        changed = bytearray(original_model)
        changed[32:32 + 784 * 4] = struct.pack("<I", 0x7f7fffff) * 784
        reject(model_path, changed, "dense1: status")

        # Changing a valid label changes accuracy, not the runtime comparison.
        path = fixtures / "labels.u32"
        original = path.read_bytes()
        try:
            path.write_bytes(struct.pack("<I", 8) + original[4:])  # First sample is a correctly predicted 7.
            result = run()
            if (result.returncode != 0 or
                    "Runtime prediction agreement (C vs saved Python): 100/100." not in result.stdout or
                    "Classification on fixed test subset: C 98/100; Python 98/100." not in result.stdout):
                raise AssertionError(f"Accuracy was not separate from runtime agreement:\n"
                                     f"{result.stdout}{result.stderr}")
        finally:
            path.write_bytes(original)
        print(f"Fresh export and three prefix identities passed; {rejected} corrupted cases rejected; "
              "label-only change affected accuracy without affecting runtime agreement.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path,
                        help="built test_mnist_inference executable")
    args = parser.parse_args()
    check_inference(args.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
