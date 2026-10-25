"""Offline checks of trained-checkpoint restoration, export, and regeneration."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

import torch
from torch import nn

from examples.mnist import export, train


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/mnist"
CHECKPOINT = EXAMPLE / "reference/checkpoint.pt"
SAVED = EXAMPLE / "mnist.cnd"


class MnistExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cinder mnist export ")
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "model.cnd"
        self.checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        self.rng = torch.get_rng_state()
        self.threads = torch.get_num_threads()
        self.dtype = torch.get_default_dtype()
        torch.set_num_threads(1)

    def tearDown(self):
        torch.set_rng_state(self.rng)
        torch.set_num_threads(self.threads)
        torch.set_default_dtype(self.dtype)
        self.temporary.cleanup()

    def test_restores_parameters_and_appends_feature_softmax_in_eval_mode(self):
        source = CHECKPOINT.read_bytes()
        with patch.object(train, "train_network", side_effect=AssertionError("must not train")):
            model = export.load_inference_model(CHECKPOINT)
        self.assertEqual([type(layer) for layer in model], [nn.Linear, nn.ReLU, nn.Linear, nn.Softmax])
        self.assertEqual(model[-1].dim, -1)
        self.assertTrue(all(not module.training for module in model.modules()))
        self.assertEqual(list(model.state_dict()), list(self.checkpoint["model_state_dict"]))
        for name, parameter in model.state_dict().items():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertEqual(parameter.device.type, "cpu")
            self.assertTrue(torch.equal(parameter.view(torch.int32),
                                        self.checkpoint["model_state_dict"][name].view(torch.int32)))
        logits_model = train.build_model()
        logits_model.load_state_dict(self.checkpoint["model_state_dict"], strict=True)
        logits_model.eval()
        inputs = torch.stack((torch.zeros(784), torch.linspace(0, 1, 784)))
        with torch.inference_mode():
            logits = logits_model(inputs)
            probabilities = model(inputs)
            self.assertEqual(tuple(probabilities.shape), (2, 10))
            self.assertTrue(torch.equal(probabilities, torch.softmax(logits, dim=-1)))
            self.assertTrue(torch.equal(probabilities.argmax(-1), logits.argmax(-1)))
            self.assertTrue(bool(torch.isfinite(probabilities).all()))
            self.assertTrue(bool(((probabilities >= 0) & (probabilities <= 1)).all()))
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(2), atol=1e-6, rtol=0)
            self.assertEqual(tuple(model(inputs[0]).shape), (10,))
        self.assertEqual(CHECKPOINT.read_bytes(), source)

    def test_independent_binary_layout_and_every_saved_parameter_bit(self):
        export.export_checkpoint(CHECKPOINT, self.output)
        data = self.output.read_bytes()
        self.assertEqual(struct.unpack_from("<4sIII", data), (b"CNDR", 1, 784, 4))
        self.assertEqual(len(data), 407160)
        # Offsets and sizes follow the v1 contract, not exporter helpers.
        records = [(16, 1, 784, 128, 401920, "hidden"),
                   (401952, 2, 128, 128, 0, None),
                   (401968, 1, 128, 10, 5160, "logits"),
                   (407144, 3, 10, 10, 0, None)]
        for offset, kind, ins, outs, size, name in records:
            self.assertEqual(struct.unpack_from("<IIII", data, offset), (kind, ins, outs, size))
            if name is not None:
                expected = []
                for suffix in ("weight", "bias"):
                    tensor = self.checkpoint["model_state_dict"][f"{name}.{suffix}"]
                    expected.extend(tensor.view(torch.int32).reshape(-1).tolist())
                self.assertEqual(struct.unpack_from(f"<{size // 4}i", data, offset + 16), tuple(expected))
        self.assertEqual(records[-1][0] + 16, len(data))
        self.assertEqual(data, SAVED.read_bytes())

    def test_saved_manifest_identifies_the_checkpoint_and_export(self):
        manifest = json.loads((EXAMPLE / "reference/export.json").read_text())
        self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest())
        self.assertEqual(manifest["sha256"], hashlib.sha256(SAVED.read_bytes()).hexdigest())
        self.assertEqual(manifest["file_bytes"], SAVED.stat().st_size)
        self.assertEqual((manifest["input_size"], manifest["output_size"], manifest["parameter_count"]),
                         (784, 10, 101770))

    def test_export_uses_restored_weights_despite_rng_and_default_dtype(self):
        source = CHECKPOINT.read_bytes()
        for seed, dtype in ((7, torch.float32), (999, torch.float64)):
            torch.manual_seed(seed)
            torch.set_default_dtype(dtype)
            export.export_checkpoint(CHECKPOINT, self.output)
            self.assertEqual(self.output.read_bytes(), SAVED.read_bytes())
        self.assertEqual(CHECKPOINT.read_bytes(), source)

    def test_changed_checkpoint_parameters_change_export(self):
        checkpoint = self.directory / "changed.pt"
        self.checkpoint["model_state_dict"]["hidden.weight"][0, 0] += 0.25
        torch.save(self.checkpoint, checkpoint)
        export.export_checkpoint(checkpoint, self.output)
        data = self.output.read_bytes()
        self.assertNotEqual(data, SAVED.read_bytes())
        self.assertEqual(struct.unpack_from("<i", data, 32)[0],
                         int(self.checkpoint["model_state_dict"]["hidden.weight"].view(torch.int32)[0, 0]))

    def test_invalid_checkpoint_parameters_preserve_existing_output(self):
        original = self.checkpoint["model_state_dict"]
        bad_values = [None, torch.zeros(1), original["hidden.weight"].double(),
                      original["hidden.weight"].to(torch.int32), original["hidden.weight"].to_sparse()]
        for number in (float("nan"), float("inf"), -float("inf")):
            tensor = original["hidden.weight"].clone()
            tensor[0, 0] = number
            bad_values.append(tensor)
        checkpoint = self.directory / "invalid.pt"
        self.output.write_bytes(b"keep existing destination")
        for value in bad_values:
            state = {**original, "hidden.weight": value}
            torch.save({"model_state_dict": state}, checkpoint)
            with warnings.catch_warnings():
                # The deliberately invalid sparse checkpoint triggers a loader
                # scan warning before our storage-layout rejection.
                warnings.filterwarnings("ignore", message="Validating sparse tensor invariants.*", category=UserWarning)
                with self.subTest(value=type(value)), self.assertRaisesRegex(ValueError, "hidden.weight"):
                    export.export_checkpoint(checkpoint, self.output)
            self.assertEqual(self.output.read_bytes(), b"keep existing destination")
        self.assertEqual(list(self.directory.glob(".cinder-*.tmp")), [])

    def test_missing_extra_or_invalid_state_is_rejected(self):
        state = self.checkpoint["model_state_dict"]
        missing = {key: value for key, value in state.items() if key != "logits.bias"}
        for content in ([], {}, {"model_state_dict": []}, {"model_state_dict": missing},
                        {"model_state_dict": {**state, "extra": torch.zeros(1)}}):
            checkpoint = self.directory / "invalid.pt"
            torch.save(content, checkpoint)
            with self.subTest(content=type(content)), self.assertRaises(ValueError):
                export.export_checkpoint(checkpoint, self.output)
            self.assertFalse(self.output.exists())

    def test_check_compares_without_modification_and_detects_drift(self):
        self.output.write_bytes(SAVED.read_bytes())
        before = self.output.stat().st_mtime_ns
        export.export_checkpoint(CHECKPOINT, self.output, check=True)
        self.assertEqual(self.output.stat().st_mtime_ns, before)
        corrupted = bytearray(self.output.read_bytes())
        corrupted[32] ^= 1
        self.output.write_bytes(corrupted)
        with self.assertRaisesRegex(ValueError, "export mismatch"):
            export.export_checkpoint(CHECKPOINT, self.output, check=True)
        self.assertEqual(self.output.read_bytes(), corrupted)
        self.output.unlink()
        with self.assertRaises(FileNotFoundError):
            export.export_checkpoint(CHECKPOINT, self.output, check=True)
        self.assertFalse(self.output.exists())

    def test_output_cannot_replace_checkpoint_or_its_symlink(self):
        checkpoint = self.directory / "checkpoint.pt"
        source = CHECKPOINT.read_bytes()
        checkpoint.write_bytes(source)
        alias = self.directory / "alias.cnd"
        alias.symlink_to(checkpoint)
        for output in (checkpoint, alias):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "must differ"):
                export.export_checkpoint(checkpoint, output)
        self.assertEqual(checkpoint.read_bytes(), source)

    def test_cli_reports_file_and_checkpoint_failures(self):
        corrupt = self.directory / "corrupt.pt"
        corrupt.write_bytes(b"not a checkpoint")
        for checkpoint, output in ((self.directory / "missing.pt", self.output), (corrupt, self.output),
                                    (CHECKPOINT, self.directory / "missing/model.cnd")):
            error = io.StringIO()
            with redirect_stderr(error), redirect_stdout(io.StringIO()):
                result = export.main(["--checkpoint", str(checkpoint), "--output", str(output)])
            self.assertEqual(result, 1)
            self.assertIn("MNIST export failed:", error.getvalue())
            self.assertFalse(output.exists())

    def test_fresh_process_exports_from_same_checkpoint_are_identical(self):
        source = CHECKPOINT.read_bytes()
        outputs = [self.directory / "one.cnd", self.directory / "two.cnd"]
        command = [sys.executable, str(EXAMPLE / "export.py"), "--checkpoint", str(CHECKPOINT)]
        for output in outputs:
            result = subprocess.run(command + ["--output", str(output)], cwd=self.directory,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
        self.assertEqual(outputs[0].read_bytes(), SAVED.read_bytes())
        result = subprocess.run(command + ["--output", str(outputs[0]), "--check"], cwd=self.directory,
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Verified identical export", result.stdout)
        self.assertEqual(CHECKPOINT.read_bytes(), source)


if __name__ == "__main__":
    unittest.main()
