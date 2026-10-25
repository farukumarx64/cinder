"""Offline MNIST-script checks: synthetic IDX files and actual tiny training runs."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from examples.mnist import data, train
from tools.export_model import validate_model


ROOT = Path(__file__).resolve().parents[1]


def synthetic_data():
    # Ten visibly different classes, plus three samples to exercise a partial
    # final batch. This checks machinery, not real digit-recognition quality.
    labels = torch.arange(23, dtype=torch.int64) % 10
    images = torch.zeros(23, 28, 28, dtype=torch.uint8)
    for i, label in enumerate(labels):
        images[i, int(label):int(label) + 2, :] = 255
        images[i, 27, i] = 127
    return images, labels, images[:11].clone(), labels[:11].clone()


def synthetic_archives(directory):
    tensors = synthetic_data()
    resources = []
    for original, tensor in zip(data.RESOURCES, tensors):
        header = (2051, len(tensor), 28, 28) if original.images else (2049, len(tensor))
        raw = struct.pack(">" + "I" * len(header), *header)
        raw += bytes(tensor.reshape(-1).tolist())
        archive = gzip.compress(raw, mtime=0)
        resource = replace(original, count=len(tensor),
                           md5=hashlib.md5(archive, usedforsecurity=False).hexdigest())
        (directory / resource.filename).write_bytes(archive)
        resources.append(resource)
    return tuple(resources), tensors


class MnistTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cinder mnist ")
        self.directory = Path(self.temporary.name)
        self.config = train.TrainingConfig(seed=123, epochs=2, batch_size=7)
        self.previous_threads = torch.get_num_threads()
        self.previous_deterministic = torch.are_deterministic_algorithms_enabled()
        self.previous_rng = torch.get_rng_state()
        self.previous_python_rng = random.getstate()
        # No test in this suite is permitted to download a dataset.
        self.network = patch.object(data, "urlopen", side_effect=AssertionError("unexpected network"))
        self.network.start()

    def tearDown(self):
        self.network.stop()
        torch.set_num_threads(self.previous_threads)
        torch.use_deterministic_algorithms(self.previous_deterministic)
        torch.set_rng_state(self.previous_rng)
        random.setstate(self.previous_python_rng)
        self.temporary.cleanup()

    def test_preprocessing_exact_row_major_float32_and_input_preservation(self):
        images = torch.zeros(2, 28, 28, dtype=torch.uint8)
        images[0, 0, 1], images[0, 1, 0], images[0, 27, 27] = 1, 127, 255
        images[1, 3, 4] = 128
        saved = images.clone()
        actual = data.preprocess_images(images)
        self.assertEqual((tuple(actual.shape), actual.dtype), ((2, 784), torch.float32))
        for sample, row, col, value in ((0, 0, 1, 1), (0, 1, 0, 127),
                                         (0, 27, 27, 255), (1, 3, 4, 128)):
            expected = torch.tensor(value, dtype=torch.float32) / torch.tensor(255, dtype=torch.float32)
            self.assertEqual(actual[sample, row * 28 + col], expected)
        self.assertEqual(actual.min(), 0)
        self.assertEqual(actual.max(), 1)
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(images, saved))
        # Logical row-major order must also hold for a noncontiguous input view.
        transposed = images.transpose(1, 2)
        self.assertFalse(transposed.is_contiguous())
        self.assertEqual(data.preprocess_images(transposed)[0, 28], actual[0, 1])

    def test_preprocessing_rejects_wrong_shape_dtype_and_empty_batches(self):
        for images in (torch.zeros(28, 28, dtype=torch.uint8),
                       torch.zeros(1, 784, dtype=torch.uint8),
                       torch.zeros(1, 28, 28), torch.zeros(0, 28, 28, dtype=torch.uint8)):
            with self.subTest(shape=images.shape, dtype=images.dtype), self.assertRaises(ValueError):
                data.preprocess_images(images)

    def test_original_idx_bytes_decode_without_pixel_reordering(self):
        resources, expected = synthetic_archives(self.directory)
        with patch.object(data, "RESOURCES", resources):
            actual, provenance = data.load_mnist(self.directory)
        for a, e in zip(actual, expected):
            self.assertTrue(torch.equal(a, e))
        self.assertEqual(provenance["evaluation_split"], "official test split")
        for record, resource in zip(provenance["files"], resources):
            archive = (self.directory / resource.filename).read_bytes()
            self.assertEqual(record["sha256"], hashlib.sha256(archive).hexdigest())

    def test_idx_rejects_truncation_extra_bytes_headers_and_labels(self):
        resources, _ = synthetic_archives(self.directory)
        for resource in resources[:2]:
            original = gzip.decompress((self.directory / resource.filename).read_bytes())
            mutations = [original[:-1], original + b"\0", b"\0\0\0\0" + original[4:],
                         original[:4] + struct.pack(">I", resource.count + 1) + original[8:]]
            if resource.images:
                mutations.append(original[:8] + struct.pack(">II", 1, 784) + original[16:])
            else:
                mutations.append(original[:8] + b"\x0a" + original[9:])
            for raw in mutations:
                with self.subTest(file=resource.filename), self.assertRaises(ValueError):
                    data.decode_idx(gzip.compress(raw, mtime=0), resource)
        with self.assertRaises((OSError, EOFError)):
            data.decode_idx(b"not gzip", resources[0])

    def test_missing_and_corrupted_archives_are_not_silently_replaced(self):
        resource = data.RESOURCES[0]
        with self.assertRaisesRegex(FileNotFoundError, "--download"):
            data.read_archive(self.directory, resource, False)
        path = self.directory / resource.filename
        path.write_bytes(b"bad archive")
        with self.assertRaisesRegex(ValueError, "checksum"):
            data.read_archive(self.directory, resource, True)
        self.assertEqual(path.read_bytes(), b"bad archive")

    def test_explicit_download_checks_hash_and_cleans_temporary_files(self):
        resources, _ = synthetic_archives(self.directory)
        resource = resources[0]
        path = self.directory / resource.filename
        archive = path.read_bytes()
        path.unlink()
        with patch.object(data, "urlopen", return_value=io.BytesIO(archive)) as request:
            self.assertEqual(data.read_archive(self.directory, resource, True), archive)
        request.assert_called_once_with(data.MIRROR + resource.filename, timeout=30)
        self.assertEqual(path.read_bytes(), archive)
        self.assertEqual(list(self.directory.glob(".mnist-*")), [])
        path.unlink()
        with patch.object(data, "urlopen", return_value=io.BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "checksum"):
                data.read_archive(self.directory, resource, True)
        self.assertFalse(path.exists())
        with patch.object(data, "urlopen", return_value=io.BytesIO(archive)), \
                patch.object(Path, "replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                data.read_archive(self.directory, resource, True)
        self.assertEqual(list(self.directory.glob(".mnist-*")), [])

    def test_architecture_outputs_logits_and_matches_exporter_dimensions(self):
        model = train.build_model()
        self.assertEqual([type(layer) for layer in model], [nn.Linear, nn.ReLU, nn.Linear])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 101770)
        snapshot = validate_model(model)
        self.assertEqual([(layer.input_size, layer.output_size) for layer in snapshot.layers],
                         [(784, 128), (128, 128), (128, 10)])
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.logits.bias.copy_(torch.arange(-5, 5, dtype=torch.float32))
        output = model(torch.zeros(3, 784, dtype=torch.float32))
        self.assertTrue(torch.equal(output[0], model.logits.bias))
        expected = torch.logsumexp(output[0], dim=0) - output[0, 9]
        self.assertAlmostEqual(float(nn.CrossEntropyLoss()(output, torch.full((3,), 9)).detach()),
                               float(expected.detach()), places=6)

    def test_training_updates_parameters_uses_logits_and_all_samples(self):
        train.configure_reproducibility(self.config.seed)
        initial = {k: v.clone() for k, v in train.build_model().state_dict().items()}
        seen = []
        original_loss = nn.CrossEntropyLoss.forward

        def observe(criterion, logits, labels):
            seen.append((logits.detach().clone(), len(labels), torch.is_grad_enabled()))
            return original_loss(criterion, logits, labels)

        with patch.object(nn.CrossEntropyLoss, "forward", observe):
            model, results = train.train_network(*synthetic_data(), self.config)
        self.assertFalse(model.training)
        self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()))
        self.assertEqual([row["samples"] for row in results["training_history"]], [23, 23])
        self.assertEqual(results["test"]["samples"], 11)
        self.assertEqual(sum(count for _, count, grad in seen if grad), 46)
        self.assertEqual(sum(count for _, count, grad in seen if not grad), 11)
        self.assertTrue(any(bool((logits < 0).any()) for logits, _, _ in seen))
        self.assertEqual(results["optimizer"]["lr"], self.config.learning_rate)
        self.assertTrue(all(math.isfinite(row["loss"]) for row in results["training_history"]))

    def test_loss_is_sample_weighted_for_a_partial_final_batch(self):
        observed = []
        original_loss = nn.CrossEntropyLoss.forward

        def observe(criterion, logits, labels):
            loss = original_loss(criterion, logits, labels)
            if torch.is_grad_enabled():
                observed.append((float(loss.detach()), len(labels)))
            return loss

        with patch.object(nn.CrossEntropyLoss, "forward", observe):
            _, result = train.train_network(*synthetic_data(), replace(self.config, epochs=1))
        self.assertEqual([count for _, count in observed], [7, 7, 7, 2])
        expected = sum(loss * count for loss, count in observed) / 23
        self.assertEqual(result["training_history"][0]["loss"], expected)

    def test_test_split_cannot_change_training_parameters(self):
        tensors = synthetic_data()
        first, first_results = train.train_network(*tensors, self.config)
        altered = (*tensors[:2], torch.full_like(tensors[2], 255), (tensors[3] + 1) % 10)
        second, second_results = train.train_network(*altered, self.config)
        for key, value in first.state_dict().items():
            self.assertTrue(torch.equal(value, second.state_dict()[key]))
        self.assertEqual(first_results["training_history"], second_results["training_history"])

    def test_seed_controls_repeatability_and_changes_initialization(self):
        tensors = synthetic_data()
        first, first_results = train.train_network(*tensors, self.config)
        torch.rand(97)  # Ambient RNG use must not affect the next seeded run.
        second, second_results = train.train_network(*tensors, self.config)
        self.assertEqual(first_results, second_results)
        self.assertTrue(all(torch.equal(value, second.state_dict()[key])
                            for key, value in first.state_dict().items()))
        third, _ = train.train_network(*tensors, replace(self.config, seed=124))
        self.assertTrue(any(not torch.equal(value, third.state_dict()[key])
                            for key, value in first.state_dict().items()))

    def test_checkpoint_and_report_reload_without_custom_python_objects(self):
        model, results = train.train_network(*synthetic_data(), self.config)
        metadata = train.run_metadata(self.config, {"name": "synthetic test data"}, results)
        directory = self.directory / "run"
        train.save_run(directory, model, metadata)
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["metadata"], json.loads((directory / "run.json").read_text()))
        restored = train.build_model()
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        restored.eval()
        inputs = data.preprocess_images(synthetic_data()[2])
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(inputs), restored(inputs)))
        self.assertEqual(metadata["execution"]["threads"], 1)
        self.assertTrue(metadata["execution"]["deterministic_algorithms"])
        self.assertEqual(metadata["environment"]["torch"], str(torch.__version__))
        self.assertEqual(metadata["source_sha256"]["train.py"],
                         hashlib.sha256((ROOT / "examples/mnist/train.py").read_bytes()).hexdigest())

    def test_existing_results_are_preserved_and_failed_writes_are_cleaned(self):
        model = train.build_model()
        directory = self.directory / "existing"
        directory.mkdir()
        sentinel = directory / "checkpoint.pt"
        sentinel.write_bytes(b"existing checkpoint")
        with self.assertRaises(FileExistsError):
            train.save_run(directory, model, {})
        self.assertEqual(sentinel.read_bytes(), b"existing checkpoint")
        directory = self.directory / "failed"

        def fail_save(_payload, path):
            path.write_bytes(b"partial")
            raise OSError("disk failure")

        with patch.object(torch, "save", fail_save), self.assertRaisesRegex(OSError, "disk failure"):
            train.save_run(directory, model, {})
        self.assertFalse(directory.exists())
        with patch.object(Path, "write_text", side_effect=OSError("report failure")):
            with self.assertRaisesRegex(OSError, "report failure"):
                train.save_run(directory, model, {})
        self.assertFalse(directory.exists())
        with self.assertRaises(ValueError):
            train.save_run(directory, model, {"loss": float("nan")})
        self.assertFalse(directory.exists())

    def test_invalid_settings_and_dataset_shapes_fail_before_training(self):
        for setting in (replace(self.config, seed=-1), replace(self.config, seed=2**32),
                        replace(self.config, epochs=0), replace(self.config, batch_size=0),
                        replace(self.config, learning_rate=0),
                        replace(self.config, learning_rate=float("nan")),
                        replace(self.config, learning_rate=float("inf"))):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                setting.validate()
        images, labels, test_images, test_labels = synthetic_data()
        for invalid in (labels[:-1], labels.float(), torch.full_like(labels, 10)):
            with self.assertRaises(ValueError):
                train.train_network(images, invalid, test_images, test_labels, self.config)
        with self.assertRaises(ValueError):
            train.train_network(images.float(), labels, test_images, test_labels, self.config)

    def test_nonfinite_training_cannot_produce_a_checkpoint(self):
        model = train.build_model()
        with torch.no_grad():
            model.hidden.weight.fill_(float("nan"))
        with patch.object(train, "build_model", return_value=model):
            with self.assertRaisesRegex(ValueError, "non-finite training loss"):
                train.train_network(*synthetic_data(), self.config)

    def test_cli_complete_path_on_local_synthetic_archives(self):
        resources, _ = synthetic_archives(self.directory)
        output = self.directory / "output"
        arguments = ["--data-dir", str(self.directory), "--output-dir", str(output),
                     "--epochs", "1", "--batch-size", "7", "--seed", "123"]
        with patch.object(data, "RESOURCES", resources), redirect_stdout(io.StringIO()):
            self.assertEqual(train.main(arguments), 0)
        record = json.loads((output / "run.json").read_text())
        self.assertEqual(record["config"]["seed"], 123)
        self.assertEqual(record["test"]["samples"], 11)
        with redirect_stderr(io.StringIO()), patch.object(train, "load_mnist") as loader:
            self.assertEqual(train.main(arguments), 1)
            loader.assert_not_called()

    def test_cli_help_and_errors_work_outside_repository(self):
        script = str(ROOT / "examples/mnist/train.py")
        result = subprocess.run([sys.executable, script, "--help"], cwd=self.directory,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--download", result.stdout)
        for arguments, message in ((["--epochs", "0"], "positive integer"),
                                   (["--data-dir", str(self.directory)], "--download")):
            result = subprocess.run([sys.executable, script, *arguments], cwd=self.directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn(message, result.stderr)

    def test_same_environment_repeats_in_fresh_processes(self):
        code = """
from pathlib import Path
import sys
from tests_support import synthetic_data
from examples.mnist import train
config = train.TrainingConfig(seed=123, epochs=2, batch_size=7)
model, result = train.train_network(*synthetic_data(), config)
train.save_run(Path(sys.argv[1]), model, train.run_metadata(config, {'name': 'synthetic'}, result))
"""
        # Keep subprocess fixtures separate from production CLI/data handling.
        helper = self.directory / "tests_support.py"
        import inspect
        helper.write_text("import torch\n" + inspect.getsource(synthetic_data))
        import os
        environment = {**os.environ, "PYTHONPATH": str(self.directory) + os.pathsep + str(ROOT)}
        for name in ("first", "second"):
            result = subprocess.run([sys.executable, "-c", code, str(self.directory / name)],
                                    cwd=self.directory, env=environment, capture_output=True,
                                    text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        first = torch.load(self.directory / "first/checkpoint.pt", weights_only=True)
        second = torch.load(self.directory / "second/checkpoint.pt", weights_only=True)
        self.assertEqual(first["metadata"], second["metadata"])
        for name, parameter in first["model_state_dict"].items():
            self.assertTrue(torch.equal(parameter, second["model_state_dict"][name]))


if __name__ == "__main__":
    unittest.main()
