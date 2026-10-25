"""Export fresh .cnd models and run the C stage comparator against saved NumPy data."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

from generate_cnd_fixtures import generate, inventory


def check_round_trip(executable: Path) -> None:
    executable = executable.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="cinder round trip ") as temporary:
        directory = Path(temporary)
        # This calls the production Python exporter for the full model and each
        # prefix. Expected outputs are preserved from the saved NumPy fixtures.
        generate(directory)

        def run():
            return subprocess.run([str(executable), str(directory)], cwd=directory,
                                  capture_output=True, text=True, timeout=30)

        result = run()
        if result.returncode != 0:
            raise AssertionError(f"Fresh export/load/inference failed ({result.returncode}):\n"
                                 f"{result.stdout}{result.stderr}")
        print(result.stdout, end="")

        def reject_changed_file(relative: str, changed: bytes, diagnostic: str):
            path = directory / relative
            original = path.read_bytes()
            try:
                path.write_bytes(changed)
                result = run()
                # Require a clean comparison failure, not a crash or a missing
                # executable. These controls prove each stage and input is used.
                if result.returncode != 1 or diagnostic not in result.stderr:
                    raise AssertionError(f"Expected rejection containing {diagnostic!r}, "
                                         f"got exit {result.returncode}:\n{result.stdout}{result.stderr}")
            finally:
                path.write_bytes(original)

        before = inventory(directory)
        for stage, relative in (("dense1", "prefixes/dense1.cnd"),
                                ("relu", "prefixes/relu.cnd"),
                                ("logits", "prefixes/logits.cnd"),
                                ("softmax", "tiny.cnd")):
            changed = bytearray((directory / relative).read_bytes())
            # First Dense bias[0]: 0.5 -> 1.5. The file remains valid, finite,
            # and correctly shaped, but the result must miss the reference.
            struct.pack_into("<f", changed, 64, 1.5)
            reject_changed_file(relative, changed, f"worked/{stage}[")

        reject_changed_file("prefixes/dense1.cnd", (directory / "tiny.cnd").read_bytes(),
                            "dense1: loaded model shape mismatch")
        relative = "inputs/worked.f32"
        original = (directory / relative).read_bytes()
        reject_changed_file(relative, original[:-1], "exactly eight bytes")
        reject_changed_file(relative, original + b"\0", "exactly eight bytes")
        reject_changed_file(relative, struct.pack("<2f", float("inf"), -1.0), "Non-finite input fixture")
        reject_changed_file(relative, struct.pack("<2f", 3.0, -1.0), "differs from saved NumPy input bits")
        if inventory(directory) != before:
            raise AssertionError("Sensitivity checks did not restore the generated fixtures")
        print("Fresh Python export -> C load -> inference passed; nine sensitivity checks rejected altered fixtures.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path,
                        help="built test_model_round_trip executable")
    args = parser.parse_args()
    check_round_trip(args.executable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
