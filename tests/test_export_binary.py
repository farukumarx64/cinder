"""Exact .cnd bytes, independent decoding, fixture regeneration, and file failures."""

import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

import tools.export_model as exporter
from cnd_fixture_support import case_names, fixture_values, tiny_network
from generate_cnd_fixtures import generate, inventory


ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "tests/fixtures/tiny"
# Independent byte oracle copied from the v1 specification, not the writer.
TINY_BYTES = bytes.fromhex("""
43 4e 44 52 01 00 00 00 02 00 00 00 04 00 00 00
01 00 00 00 02 00 00 00 04 00 00 00 30 00 00 00
00 00 80 3f 00 00 00 40 00 00 80 bf 00 00 00 00
00 00 00 3f 00 00 00 bf 00 00 40 40 00 00 80 3f
00 00 00 3f 00 00 80 bf 00 00 00 40 00 00 00 00
02 00 00 00 04 00 00 00 04 00 00 00 00 00 00 00
01 00 00 00 04 00 00 00 02 00 00 00 28 00 00 00
00 00 80 3f 00 00 00 00 00 00 80 bf 00 00 00 3f
00 00 00 00 00 00 80 3f 00 00 00 3f 00 00 80 bf
00 00 00 00 00 00 00 3f 03 00 00 00 02 00 00 00
02 00 00 00 00 00 00 00
""")


def decode_known_file(data):
    """Test-only inspection of valid output; not the future untrusted-file loader."""
    magic, version, current, count = struct.unpack_from("<4sIII", data)
    assert magic == b"CNDR" and version == 1
    offset, records = 16, []
    for _ in range(count):
        kind, ins, outs, size = struct.unpack_from("<IIII", data, offset)
        assert ins == current
        payload = data[offset + 16:offset + 16 + size]
        assert len(payload) == size
        if kind == 1:
            assert size == 4 * (ins * outs + outs)
            values = torch.tensor(struct.unpack(f"<{size // 4}f", payload), dtype=torch.float32)
            weights, bias = values[:ins * outs].reshape(outs, ins), values[ins * outs:]
        else:
            assert kind in (2, 3) and size == 0 and ins == outs
            weights = bias = None
        records.append((offset, kind, ins, outs, size, weights, bias))
        current, offset = outs, offset + 16 + size
    assert offset == len(data)
    return records


class FailingFile:
    """Wrap a real temporary file so failures occur after real file creation."""
    def __init__(self, stream, failure):
        self.stream, self.failure = stream, failure

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.stream.__exit__(kind, value, traceback)
        if self.failure == "close" and kind is None:
            raise OSError("forced close failure")

    def write(self, data):
        if self.failure == "short":
            return self.stream.write(data[:-1])
        return self.stream.write(data)

    def flush(self):
        if self.failure == "flush":
            raise OSError("forced flush failure")
        return self.stream.flush()


class BinaryExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cinder-export-test-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / "model.cnd"

    def test_tiny_file_matches_specification_saved_fixture_and_exact_offsets(self):
        self.assertIsNone(exporter.export_model(tiny_network(), self.path))
        data = self.path.read_bytes()
        self.assertEqual(data, TINY_BYTES)
        self.assertEqual(data, (SAVED / "tiny.cnd").read_bytes())
        self.assertEqual(len(data), 168)
        self.assertEqual(struct.unpack_from("<4sIII", data), (b"CNDR", 1, 2, 4))
        records = decode_known_file(data)
        self.assertEqual([record[:5] for record in records],
                         [(16, 1, 2, 4, 48), (80, 2, 4, 4, 0),
                          (96, 1, 4, 2, 40), (152, 3, 2, 2, 0)])
        for index, number in ((0, 1), (2, 2)):
            self.assertEqual(records[index][5].flatten().tolist(), fixture_values(f"tiny_network_weights{number}"))
            self.assertEqual(records[index][6].tolist(), fixture_values(f"tiny_network_bias{number}"))
        specification = (ROOT / "docs/MODEL_FORMAT.md").read_text()
        documented = " ".join(re.findall(r"^[0-9a-f]{4}: ([0-9a-f ]+)$", specification, re.M))
        self.assertEqual(data, bytes.fromhex(documented))

    def test_saved_inputs_and_all_decoded_stages_match_numpy(self):
        exporter.export_model(tiny_network(), self.path)
        records = decode_known_file(self.path.read_bytes())
        manifest = json.loads((SAVED / "references.json").read_text())
        self.assertEqual(manifest["fixture_version"], 1)
        self.assertEqual((manifest["input_size"], manifest["output_size"]), (2, 2))
        self.assertEqual((manifest["atol"], manifest["rtol"]), (1e-5, 1e-5))
        self.assertEqual(manifest["model_sha256"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual([case["name"] for case in manifest["cases"]], case_names())
        self.assertEqual(len(manifest["cases"]), 10)
        for case in manifest["cases"]:
            data = (SAVED / case["input_file"]).read_bytes()
            self.assertEqual(data, struct.pack("<2f", *fixture_values(f"tiny_{case['name']}_input")))
            current = torch.tensor(struct.unpack("<2f", data), dtype=torch.float32)
            for record, stage in zip(records, ("dense1", "relu", "logits", "softmax")):
                with self.subTest(case=case["name"], stage=stage):
                    _, kind, _, _, _, weights, bias = record
                    if kind == 1:
                        current = torch.mv(weights, current) + bias
                    elif kind == 2:
                        current = torch.relu(current)
                    else:
                        current = torch.softmax(current, dim=-1)
                    expected = [float.fromhex(value) for value in case["stages"][stage]]
                    original = fixture_values(f"tiny_{case['name']}_{stage}")
                    self.assertEqual(struct.pack(f"<{len(expected)}f", *expected),
                                     struct.pack(f"<{len(original)}f", *original))
                    reference = torch.tensor(expected, dtype=torch.float32)
                    self.assertTrue(bool(torch.isfinite(current).all() and torch.isfinite(reference).all()))
                    torch.testing.assert_close(current, reference, atol=1e-5, rtol=1e-5)

    def test_complete_fixture_regeneration_is_byte_identical(self):
        first, second = self.directory / "first", self.directory / "second"
        generate(first)
        generate(second)
        self.assertEqual(len(inventory(first)), 15)
        self.assertEqual(inventory(first), inventory(second))
        self.assertEqual(inventory(first), inventory(SAVED))

    def test_exported_prefixes_match_the_independent_full_model_bytes(self):
        generate(self.directory)
        manifest = json.loads((self.directory / "references.json").read_text())
        expected_paths = {"dense1": "prefixes/dense1.cnd", "relu": "prefixes/relu.cnd",
                          "logits": "prefixes/logits.cnd", "softmax": "tiny.cnd"}
        self.assertEqual(manifest["stage_models"], expected_paths)
        for count, (stage, end) in enumerate(zip(expected_paths, (80, 96, 152, 168)), start=1):
            with self.subTest(stage=stage):
                expected = bytearray(TINY_BYTES[:end])
                struct.pack_into("<I", expected, 12, count)
                data = (self.directory / expected_paths[stage]).read_bytes()
                self.assertEqual(data, expected)
                self.assertEqual(manifest["stage_model_sha256"][stage], hashlib.sha256(data).hexdigest())

    def test_generator_check_mode_detects_drift_without_changing_files(self):
        generate(self.directory)
        command = [sys.executable, str(ROOT / "tests/generate_cnd_fixtures.py"),
                   "--check", "--output", str(self.directory)]
        result = subprocess.run(command, cwd=self.directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.directory / "tiny.cnd").write_bytes(b"corrupted")
        before = inventory(self.directory)
        result = subprocess.run(command, cwd=self.directory, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1)
        self.assertIn("tiny.cnd", result.stderr)
        self.assertEqual(inventory(self.directory), before)

    def test_existing_longer_file_is_replaced_and_repeated_exports_are_identical(self):
        self.path.write_bytes(b"old" * 1000)
        model = tiny_network()
        for _ in range(3):
            exporter.export_model(model, str(self.path))
            self.assertEqual(self.path.read_bytes(), TINY_BYTES)
            self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_single_dense_activation_only_and_shared_layers(self):
        layer = nn.Linear(2, 2)
        cases = [(nn.Sequential(nn.Linear(1, 1)), None, [1]),
                 (nn.Sequential(nn.ReLU()), 65536, [2]),
                 (nn.Sequential(nn.ReLU(), nn.Softmax(dim=-1)), 3, [2, 3]),
                 (nn.Sequential(layer, layer), None, [1, 1])]
        for model, size, kinds in cases:
            with self.subTest(kinds=kinds, size=size):
                exporter.export_model(model, self.path, input_size=size)
                records = decode_known_file(self.path.read_bytes())
                self.assertEqual([record[1] for record in records], kinds)
        # A one-activation file is the minimum permitted 32-byte format.
        exporter.export_model(nn.Sequential(nn.ReLU()), self.path, input_size=1)
        self.assertEqual(len(self.path.read_bytes()), 32)

    def test_noncontiguous_layout_is_written_row_major(self):
        layer = nn.Linear(2, 3)
        layer.weight = nn.Parameter(torch.tensor([[1., 2., 3.], [4., 5., 6.]]).t())
        layer.bias = nn.Parameter(torch.tensor([7., 99., 8., 99., 9., 99.])[::2])
        exporter.export_model(nn.Sequential(layer), self.path)
        self.assertEqual(struct.unpack_from("<9f", self.path.read_bytes(), 32),
                         (1., 4., 2., 5., 3., 6., 7., 8., 9.))

    def test_signed_zeros_subnormal_and_finite_extreme_bits_survive(self):
        bits = [0, -2147483648, 1, -2147483647, 2139095039, -8388609]
        layer = nn.Linear(len(bits), 1)
        layer.weight = nn.Parameter(torch.tensor(bits, dtype=torch.int32).view(torch.float32).reshape(1, -1))
        layer.bias = nn.Parameter(torch.tensor([-0.0]))
        exporter.export_model(nn.Sequential(layer), self.path)
        self.assertEqual(self.path.read_bytes()[32:], struct.pack("<7i", *bits, -2147483648))

    def test_chunk_boundaries_preserve_order_and_bound_write_size(self):
        layer = nn.Linear(7, 586)  # 4,102 weights: one full 4,096-word chunk plus a tail.
        with torch.no_grad():
            layer.weight.copy_(torch.arange(4102, dtype=torch.float32).reshape(586, 7))
            layer.bias.copy_(torch.arange(586, dtype=torch.float32) * -1)
        sizes, write = [], exporter._write_exact
        def record(stream, data):
            sizes.append(len(data))
            write(stream, data)
        with patch.object(exporter, "_write_exact", side_effect=record):
            exporter.export_model(nn.Sequential(layer), self.path)
        self.assertEqual(sizes, [16, 16, 16384, 24, 2344])
        payload = self.path.read_bytes()[32:]
        self.assertEqual(payload, struct.pack("<4688f", *range(4102), *(-float(n) for n in range(586))))

    def test_invalid_model_does_not_create_or_replace_any_file(self):
        for exists in (False, True):
            if exists:
                self.path.write_bytes(b"keep")
            before = inventory(self.directory)
            invalid = tiny_network()
            with torch.no_grad():
                invalid[2].bias[0] = float("nan")
            with patch.object(exporter.tempfile, "NamedTemporaryFile", side_effect=AssertionError("file opened")):
                with self.assertRaises(exporter.ModelValidationError):
                    exporter.export_model(invalid, self.path)
            self.assertEqual(inventory(self.directory), before)

    def test_write_failure_at_each_record_or_parameter_preserves_destination(self):
        write = exporter._write_exact
        for exists in (False, True):
            for fail_at in range(1, 10):
                for error in (OSError, MemoryError):
                    self.path.unlink(missing_ok=True)
                    if exists:
                        self.path.write_bytes(b"keep")
                    before, calls = inventory(self.directory), 0
                    def fail(stream, data):
                        nonlocal calls
                        calls += 1
                        write(stream, data)
                        if calls == fail_at:
                            raise error("forced partial export")
                    with self.subTest(exists=exists, fail_at=fail_at, error=error), \
                            patch.object(exporter, "_write_exact", side_effect=fail):
                        with self.assertRaisesRegex(error, "forced partial export"):
                            exporter.export_model(tiny_network(), self.path)
                    self.assertEqual(inventory(self.directory), before)
        exporter.export_model(tiny_network(), self.path)
        self.assertEqual(self.path.read_bytes(), TINY_BYTES)

    def test_short_write_flush_and_close_failures_clean_temporary_file(self):
        factory = tempfile.NamedTemporaryFile
        self.path.write_bytes(b"keep")
        for failure in ("short", "flush", "close"):
            opened = []
            def create(**kwargs):
                stream = factory(**kwargs)
                opened.append(stream)
                self.assertEqual(Path(stream.name).parent, self.directory)
                return FailingFile(stream, failure)
            with self.subTest(failure=failure), patch.object(exporter.tempfile, "NamedTemporaryFile", side_effect=create):
                with self.assertRaises(OSError):
                    exporter.export_model(tiny_network(), self.path)
            self.assertTrue(all(stream.closed for stream in opened))
            self.assertEqual(inventory(self.directory), {"model.cnd": b"keep"})

    def test_creation_fsync_and_replace_failures_preserve_destination(self):
        self.path.write_bytes(b"keep")
        for owner, name in ((exporter.tempfile, "NamedTemporaryFile"),
                            (exporter.os, "fsync"), (exporter.os, "replace")):
            with self.subTest(operation=name), patch.object(owner, name, side_effect=OSError("forced I/O failure")):
                with self.assertRaisesRegex(OSError, "forced I/O failure"):
                    exporter.export_model(tiny_network(), self.path)
            self.assertEqual(inventory(self.directory), {"model.cnd": b"keep"})

    def test_incorrect_written_length_is_not_published(self):
        self.path.write_bytes(b"keep")
        with patch.object(exporter, "_write_snapshot", side_effect=lambda stream, snapshot: stream.write(b"short")):
            with self.assertRaisesRegex(OSError, "incorrect byte count"):
                exporter.export_model(tiny_network(), self.path)
        self.assertEqual(inventory(self.directory), {"model.cnd": b"keep"})

    def test_missing_parent_and_directory_destination_are_errors(self):
        with self.assertRaises(FileNotFoundError):
            exporter.export_model(tiny_network(), self.directory / "missing/model.cnd")
        with self.assertRaises(OSError):
            exporter.export_model(tiny_network(), self.directory)
        self.assertEqual(inventory(self.directory), {})

    def test_symlink_is_replaced_without_modifying_its_target(self):
        original = self.directory / "original.cnd"
        original.write_bytes(b"keep original")
        self.path.symlink_to(original)
        exporter.export_model(tiny_network(), self.path)
        self.assertFalse(self.path.is_symlink())
        self.assertEqual(original.read_bytes(), b"keep original")
        self.assertEqual(self.path.read_bytes(), TINY_BYTES)

    def test_export_uses_owned_snapshot_and_never_runs_forward(self):
        model = tiny_network()
        original_write = exporter._write_snapshot
        def change_source_then_write(stream, snapshot):
            with torch.no_grad():
                model[0].weight.fill_(999)
            original_write(stream, snapshot)
        with patch.object(exporter, "_write_snapshot", side_effect=change_source_then_write), \
                patch.object(nn.Sequential, "forward", side_effect=AssertionError("forward called")):
            exporter.export_model(model, self.path)
        self.assertEqual(self.path.read_bytes(), TINY_BYTES)


if __name__ == "__main__":
    unittest.main()
