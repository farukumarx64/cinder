"""Export the saved tiny network and preserve its NumPy references, without recomputing them."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile

# Allow direct execution from any working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_model import export_model
from cnd_fixture_support import case_names, fixture_values, tiny_network


def generate(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "inputs").mkdir(exist_ok=True)
    (directory / "prefixes").mkdir(exist_ok=True)
    model = tiny_network()
    export_model(model, directory / "tiny.cnd")
    stage_models = {}
    for count, stage in enumerate(("dense1", "relu", "logits"), start=1):
        relative = f"prefixes/{stage}.cnd"
        export_model(model[:count], directory / relative)
        stage_models[stage] = relative
    stage_models["softmax"] = "tiny.cnd"
    cases = []
    for name in case_names():
        values = fixture_values(f"tiny_{name}_input")
        input_file = f"inputs/{name}.f32"
        (directory / input_file).write_bytes(struct.pack(f"<{len(values)}f", *values))
        cases.append({
            "name": name,
            "input_file": input_file,
            "stages": {stage: [value.hex() for value in fixture_values(f"tiny_{name}_{stage}")]
                       for stage in ("dense1", "relu", "logits", "softmax")},
        })
    manifest = {
        "fixture_version": 1,
        "model": "tiny.cnd",
        "model_sha256": hashlib.sha256((directory / "tiny.cnd").read_bytes()).hexdigest(),
        "stage_models": stage_models,
        "stage_model_sha256": {stage: hashlib.sha256((directory / path).read_bytes()).hexdigest()
                               for stage, path in stage_models.items()},
        "input_encoding": "raw little-endian IEEE 754 binary32",
        "stage_encoding": "exact binary32 values as hexadecimal float strings",
        "reference_source": "tests/fixtures/numpy_fixtures.h",
        "input_size": 2,
        "output_size": 2,
        "atol": 1e-5,
        "rtol": 1e-5,
        "cases": cases,
    }
    (directory / "references.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def inventory(directory: Path) -> dict[str, bytes]:
    return {path.relative_to(directory).as_posix(): path.read_bytes()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/tiny")
    parser.add_argument("--check", action="store_true", help="compare regeneration without changing saved files")
    args = parser.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory(prefix="cinder-fixtures-") as temporary:
            generated = Path(temporary)
            generate(generated)
            actual, expected = inventory(args.output), inventory(generated)
            changed = [name for name in sorted(actual.keys() | expected.keys())
                       if actual.get(name) != expected.get(name)]
            if changed:
                print("Fixture mismatch: " + ", ".join(changed), file=sys.stderr)
                return 1
            print(f"All {len(expected)} fixture files match deterministic regeneration.")
    else:
        generate(args.output)
        print(f"Generated tiny.cnd, three prefixes, ten inputs, and references.json in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
