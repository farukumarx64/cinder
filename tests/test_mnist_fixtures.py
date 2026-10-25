"""Offline checks of the saved 100-image MNIST reference set and its generator."""

from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from examples.mnist import data, export, generate_fixtures as generator


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/mnist"
SAVED = ROOT / "tests/fixtures/mnist"
CHECKPOINT = EXAMPLE / "reference/checkpoint.pt"
MODEL = EXAMPLE / "mnist.cnd"


def inventory(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


def floats(content, width):
    assert len(content) == 100 * width * 4
    return torch.tensor(struct.unpack(f"<{100 * width}f", content), dtype=torch.float32).reshape(100, width)


class MnistFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cinder mnist fixtures ")
        self.directory = Path(self.temporary.name)
        self.files = inventory(SAVED)
        self.manifest = json.loads(self.files["manifest.json"])
        self.images = torch.frombuffer(bytearray(self.files["images.u8"]), dtype=torch.uint8).reshape(100, 28, 28)
        self.labels = torch.tensor(struct.unpack("<100I", self.files["labels.u32"]), dtype=torch.int64)
        self.archives = self.manifest["selection"]["archives"]
        self.threads, self.rng = torch.get_num_threads(), torch.get_rng_state()
        self.python_rng = random.getstate()
        self.deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(1)
        self.network = patch.object(data, "urlopen", side_effect=AssertionError("unexpected network"))
        self.network.start()

    def tearDown(self):
        self.network.stop()
        torch.set_num_threads(self.threads)
        torch.set_rng_state(self.rng)
        random.setstate(self.python_rng)
        torch.use_deterministic_algorithms(self.deterministic)
        self.temporary.cleanup()

    def build(self, **overrides):
        args = {"images": self.images, "labels": self.labels, "archives": self.archives,
                "checkpoint": CHECKPOINT, "model_path": MODEL}
        return generator.build_files(**(args | overrides))

    def test_inventory_provenance_indices_labels_and_shapes(self):
        self.assertEqual(len(self.files), 108)
        self.assertEqual(set(self.files), set(self.manifest["files"]) | {"manifest.json"})
        self.assertEqual(struct.unpack("<100I", self.files["indices.u32"]), tuple(range(100)))
        self.assertEqual(len(self.manifest["cases"]), 100)
        for row, case in enumerate(self.manifest["cases"]):
            self.assertEqual((case["row"], case["dataset_index"], case["label"]), (row, row, int(self.labels[row])))
            self.assertEqual(case["input_file"], f"inputs/{row:05d}.f32")
        for name, description in self.manifest["files"].items():
            content = self.files[name]
            self.assertEqual(len(content), description["bytes"], name)
            self.assertEqual(hashlib.sha256(content).hexdigest(), description["sha256"], name)
            elements = 1
            for dimension in description["shape"]:
                elements *= dimension
            self.assertEqual(len(content), elements * (1 if name.endswith(".u8") else 4), name)
        for name, path in (("checkpoint", CHECKPOINT), ("model", MODEL)):
            self.assertEqual(self.manifest[f"{name}_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for name, digest in self.manifest["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        self.assertEqual(self.manifest["comparison"], {"atol": 1e-5, "rtol": 1e-5,
                                                     "require_shape_match": True, "require_finite": True})
        self.assertTrue(all(record["samples"] == 10000 for record in self.archives))
        self.assertTrue(all(record["filename"].startswith("t10k-") for record in self.archives))

    def test_every_input_matches_raw_pixels_and_row_major_division(self):
        for row in range(100):
            pixels = self.images[row].reshape(-1).tolist()
            # Independent scalar arithmetic and packing, not preprocess_images.
            expected = struct.pack("<784f", *(pixel / 255.0 for pixel in pixels))
            self.assertEqual(self.files[f"inputs/{row:05d}.f32"], expected)

    def test_all_saved_stages_match_checkpoint_with_shape_and_finite_checks(self):
        model = export.load_inference_model(CHECKPOINT)
        arrays = {name: floats(self.files[f"{name}.f32"], width) for name, width in generator.STAGES}
        with torch.inference_mode():
            for row in range(100):
                values = torch.tensor(struct.unpack("<784f", self.files[f"inputs/{row:05d}.f32"]), dtype=torch.float32)
                for layer, (name, width) in zip(model, generator.STAGES, strict=True):
                    values = layer(values)
                    reference = arrays[name][row]
                    self.assertEqual(tuple(values.shape), (width,))
                    self.assertTrue(bool(torch.isfinite(values).all() & torch.isfinite(reference).all()))
                    torch.testing.assert_close(values, reference, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.equal(arrays["relu"], arrays["dense1"].clamp_min(0)))
        probabilities = arrays["probabilities"]
        self.assertTrue(bool(((probabilities >= 0) & (probabilities <= 1)).all()))
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(100), atol=1e-6, rtol=0)
        predicted = probabilities.argmax(-1).tolist()
        self.assertEqual(predicted, [case["prediction"] for case in self.manifest["cases"]])
        self.assertEqual(self.manifest["subset_result"]["correct"], sum(p == int(label) for p, label in zip(predicted, self.labels)))

    def test_offline_regeneration_uses_selected_pixels_without_dataset_cache(self):
        checkpoint_before, model_before = CHECKPOINT.read_bytes(), MODEL.read_bytes()
        with patch.object(generator, "select_test_images", side_effect=AssertionError("must not load full dataset")):
            actual = self.build()
        self.assertEqual(actual.keys(), self.files.keys())
        for name, width in generator.STAGES:
            a, b = floats(actual[f"{name}.f32"], width), floats(self.files[f"{name}.f32"], width)
            self.assertTrue(bool(torch.isfinite(a).all() & torch.isfinite(b).all()))
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)
        for name in self.files:
            if name.startswith("inputs/") or name.endswith((".u8", ".u32")):
                self.assertEqual(actual[name], self.files[name])
        self.assertEqual(json.loads(actual["manifest.json"])["cases"], self.manifest["cases"])
        self.assertEqual(CHECKPOINT.read_bytes(), checkpoint_before)
        self.assertEqual(MODEL.read_bytes(), model_before)

    def test_reader_selects_first_100_from_only_the_test_archives(self):
        images = torch.full((10000, 28, 28), 19, dtype=torch.uint8)
        labels = torch.full((10000,), 9, dtype=torch.uint8)
        images[:100], labels[:100] = self.images, self.labels.to(torch.uint8)
        resources = list(data.RESOURCES[:2])
        for resource, tensor in zip(data.RESOURCES[2:], (images, labels)):
            header = (2051, 10000, 28, 28) if resource.images else (2049, 10000)
            raw = struct.pack(">" + "I" * len(header), *header) + bytes(tensor.reshape(-1).tolist())
            content = gzip.compress(raw, mtime=0)
            (self.directory / resource.filename).write_bytes(content)
            resources.append(replace(resource, md5=hashlib.md5(content, usedforsecurity=False).hexdigest()))
        with patch.object(data, "RESOURCES", tuple(resources)):
            actual_images, actual_labels, provenance = generator.select_test_images(self.directory)
        self.assertTrue(torch.equal(actual_images, self.images))
        self.assertTrue(torch.equal(actual_labels, self.labels))
        self.assertEqual([item["filename"] for item in provenance], [r.filename for r in resources[2:]])
        for record in provenance:
            self.assertEqual(record["sha256"], hashlib.sha256((self.directory / record["filename"]).read_bytes()).hexdigest())

    def test_invalid_selection_rejected_before_checkpoint_loading(self):
        cases = [{"images": self.images[:99]}, {"images": self.images.float()},
                 {"images": self.images.reshape(100, 784)}, {"labels": self.labels[:99]},
                 {"labels": self.labels.float()}, {"labels": torch.full_like(self.labels, 10)},
                 {"labels": torch.full_like(self.labels, -1)}]
        with patch.object(generator, "load_inference_model", side_effect=AssertionError("unexpected load")):
            for changed in cases:
                with self.subTest(changed=list(changed)), self.assertRaises(ValueError):
                    self.build(**changed)

    def test_mismatched_model_and_nonfinite_stage_are_rejected(self):
        model = self.directory / "wrong.cnd"
        content = bytearray(MODEL.read_bytes())
        content[32] ^= 1
        model.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.build(model_path=model)
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        saved["model_state_dict"]["hidden.weight"].fill_(torch.finfo(torch.float32).max)
        checkpoint = self.directory / "overflow.pt"
        torch.save(saved, checkpoint)
        export.export_checkpoint(checkpoint, model)
        with self.assertRaisesRegex(ValueError, "dense1.*finite"):
            self.build(checkpoint=checkpoint, model_path=model)

    def test_comparison_detects_changed_missing_and_extra_files_without_repair(self):
        output = self.directory / "fixtures"
        generator.write_files(output, self.files)
        generator.check_files(output, self.files)
        targets = ["images.u8", "indices.u32", "labels.u32", "inputs/00000.f32", "manifest.json"]
        targets += [f"{name}.f32" for name, _ in generator.STAGES]
        for name in targets:
            path = output / name
            changed = bytearray(self.files[name])
            changed[-1] ^= 1
            path.write_bytes(changed)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "fixture mismatch"):
                generator.check_files(output, self.files)
            self.assertEqual(path.read_bytes(), changed)
            path.write_bytes(self.files[name])
        (output / "labels.u32").unlink()
        with self.assertRaisesRegex(ValueError, "labels.u32"):
            generator.check_files(output, self.files)
        (output / "labels.u32").write_bytes(self.files["labels.u32"])
        (output / "unexpected.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "unexpected.bin"):
            generator.check_files(output, self.files)

    def test_writing_preserves_existing_answers_and_cleans_partial_files(self):
        output = self.directory / "fixtures"
        output.mkdir()
        sentinel = output / "keep"
        sentinel.write_bytes(b"original")
        with self.assertRaises(FileExistsError):
            generator.write_files(output, self.files)
        self.assertEqual(sentinel.read_bytes(), b"original")
        new = self.directory / "new"
        original_write = Path.write_bytes

        def fail_write(path, content):
            if path.name == "00002.f32":
                raise OSError("forced fixture write failure")
            return original_write(path, content)

        with patch.object(Path, "write_bytes", fail_write), self.assertRaisesRegex(OSError, "forced fixture"):
            generator.write_files(new, self.files)
        self.assertFalse(new.exists())
        generator.write_files(new, self.files)
        generator.check_files(new, self.files)

    def test_cli_help_missing_data_and_existing_output_from_other_directory(self):
        command = [sys.executable, str(EXAMPLE / "generate_fixtures.py")]
        for arguments, expected, message in ((["--help"], 0, "--check"),
                (["--data-dir", str(self.directory), "--output", str(self.directory / "new")], 1, "--download"),
                (["--output", str(self.directory)], 1, "output already exists")):
            result = subprocess.run(command + arguments, cwd=self.directory, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertIn(message, result.stdout + result.stderr)
        self.assertFalse((self.directory / "new").exists())

    def test_fresh_process_regeneration_is_byte_identical_in_one_environment(self):
        code = """
import json
from pathlib import Path
import struct
import sys
import torch
sys.path.insert(0, sys.argv[1])
from examples.mnist import generate_fixtures as g
root = Path(sys.argv[1])
saved = root / 'tests/fixtures/mnist'
manifest = json.loads((saved / 'manifest.json').read_text())
images = torch.frombuffer(bytearray((saved / 'images.u8').read_bytes()), dtype=torch.uint8).reshape(100,28,28)
labels = torch.tensor(struct.unpack('<100I', (saved / 'labels.u32').read_bytes()), dtype=torch.int64)
files = g.build_files(images, labels, manifest['selection']['archives'],
                     root / 'examples/mnist/reference/checkpoint.pt', root / 'examples/mnist/mnist.cnd')
g.write_files(Path(sys.argv[2]), files)
"""
        outputs = [self.directory / "first", self.directory / "second"]
        for output in outputs:
            result = subprocess.run([sys.executable, "-c", code, str(ROOT), str(output)], cwd=self.directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(inventory(outputs[0]), inventory(outputs[1]))


if __name__ == "__main__":
    unittest.main()
