"""Real PyTorch validation tests; no exporter files are written."""

from dataclasses import FrozenInstanceError
import re
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn.modules import module as module_hooks
from torch.nn.utils import parametrize, prune

from cnd_fixture_support import FIXTURES, fixture, tiny_network

from tools.export_model import (
    LayerType,
    MAX_LAYERS,
    MAX_PARAMETER_BYTES,
    MAX_VECTOR_SIZE,
    ModelValidationError,
    validate_model,
)


def broadcast_linear(ins, outs):
    """Large logical shapes with tiny source storage; validation must bound copies."""
    layer = nn.Linear(1, 1)
    layer.in_features, layer.out_features = ins, outs
    value = torch.tensor(0.0)
    layer.weight = nn.Parameter(value.expand(outs, ins))
    layer.bias = nn.Parameter(value.expand(outs))
    return layer


class ExporterValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_tiny_network_metadata_and_row_major_parameters(self):
        result = validate_model(tiny_network())
        self.assertEqual((result.input_size, result.output_size), (2, 2))
        self.assertEqual([layer.type_id for layer in result.layers], [1, 2, 1, 3])
        self.assertEqual([(layer.input_size, layer.output_size) for layer in result.layers],
                         [(2, 4), (4, 4), (4, 2), (2, 2)])
        self.assertEqual([layer.payload_bytes for layer in result.layers], [48, 0, 40, 0])
        self.assertEqual((result.parameter_count, result.parameter_bytes, result.file_bytes),
                         (22, 88, 168))
        for index, number in ((0, 1), (2, 2)):
            layer = result.layers[index]
            self.assertTrue(torch.equal(layer.weights.flatten(), fixture(f"tiny_network_weights{number}")))
            self.assertTrue(torch.equal(layer.bias, fixture(f"tiny_network_bias{number}")))
        for index in (1, 3):
            self.assertIsNone(result.layers[index].weights)
            self.assertIsNone(result.layers[index].bias)

    def test_extracted_network_matches_all_saved_numpy_stages(self):
        snapshot = validate_model(tiny_network())
        cases = re.findall(r'\{"[^"]+", NUMPY_VECTOR\(tiny_(\w+)_input\)', FIXTURES)
        self.assertEqual(len(cases), 10)
        for case in cases:
            current = fixture(f"tiny_{case}_input")
            for layer, stage in zip(snapshot.layers, ("dense1", "relu", "logits", "softmax")):
                with self.subTest(case=case, stage=stage):
                    if layer.type_id == LayerType.DENSE:
                        current = torch.mv(layer.weights, current) + layer.bias
                    elif layer.type_id == LayerType.RELU:
                        current = torch.relu(current)
                    else:
                        current = torch.softmax(current, dim=-1)
                    self.assertTrue(bool(torch.isfinite(current).all()))
                    torch.testing.assert_close(current, fixture(f"tiny_{case}_{stage}"),
                                               atol=1e-5, rtol=1e-5)

    def test_explicit_input_size_and_activation_prefix(self):
        model = nn.Sequential(nn.ReLU(), nn.Softmax(dim=-1), nn.Linear(3, 2))
        for size in (None, 3):
            with self.subTest(size=size):
                snapshot = validate_model(model, input_size=size)
                self.assertEqual(snapshot.input_size, 3)
                self.assertEqual([(layer.input_size, layer.output_size) for layer in snapshot.layers],
                                 [(3, 3), (3, 3), (3, 2)])
        with self.assertRaisesRegex(ModelValidationError, "layer 2.*expects 3.*has 4"):
            validate_model(model, input_size=4)

    def test_activation_only_requires_explicit_size(self):
        model = nn.Sequential(nn.ReLU(), nn.Softmax(dim=-1))
        with self.assertRaisesRegex(ModelValidationError, "activation-only.*input_size"):
            validate_model(model)
        result = validate_model(model, input_size=5)
        self.assertEqual((result.input_size, result.output_size, result.parameter_bytes), (5, 5, 0))
        self.assertEqual(result.file_bytes, 48)

    def test_scalar_and_single_dense_need_no_softmax(self):
        result = validate_model(nn.Sequential(nn.Linear(1, 1)))
        self.assertEqual(len(result.layers), 1)
        self.assertEqual((result.input_size, result.output_size, result.parameter_count), (1, 1, 2))

    def test_inplace_relu_and_training_modes_are_preserved(self):
        for inplace in (False, True):
            for training in (False, True):
                with self.subTest(inplace=inplace, training=training):
                    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(inplace=inplace))
                    model.train(training)
                    # Mixed module training flags must not be silently normalized.
                    model[0].train(not training)
                    flags = [module.training for module in model.modules()]
                    result = validate_model(model)
                    self.assertEqual(result.layers[1].type_id, LayerType.RELU)
                    self.assertEqual([module.training for module in model.modules()], flags)
                    self.assertEqual(model[1].inplace, inplace)

    def test_repeated_modules_preserve_every_execution_and_own_copies(self):
        linear, relu = nn.Linear(2, 2), nn.ReLU()
        result = validate_model(nn.Sequential(linear, linear, relu, relu))
        self.assertEqual([layer.type_id for layer in result.layers], [1, 1, 2, 2])
        self.assertEqual(result.parameter_count, 12)
        self.assertNotEqual(result.layers[0].weights.data_ptr(), result.layers[1].weights.data_ptr())

    def test_snapshot_is_independent_detached_and_preserves_gradients(self):
        model = tiny_network()
        model[0].weight.grad = torch.ones_like(model[0].weight)
        model[0].bias.requires_grad_(False)
        parameters = list(model.parameters())
        pointers = [p.data_ptr() for p in parameters]
        values = [p.detach().clone() for p in parameters]
        flags = [p.requires_grad for p in parameters]
        gradient = model[0].weight.grad
        result = validate_model(model)
        another = validate_model(model)
        for original, value, pointer, flag in zip(parameters, values, pointers, flags):
            self.assertTrue(torch.equal(original, value))
            self.assertEqual(original.data_ptr(), pointer)
            self.assertEqual(original.requires_grad, flag)
        self.assertIs(model[0].weight.grad, gradient)
        self.assertTrue(torch.equal(gradient, torch.ones_like(gradient)))
        for index in (0, 2):
            self.assertNotEqual(result.layers[index].weights.data_ptr(), another.layers[index].weights.data_ptr())
            for value in (result.layers[index].weights, result.layers[index].bias):
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad_fn)
                self.assertTrue(value.is_contiguous())
        with torch.no_grad():
            model[0].weight.fill_(77)
        self.assertTrue(torch.equal(result.layers[0].weights.flatten(), fixture("tiny_network_weights1")))
        result.layers[2].weights.fill_(88)
        self.assertTrue(torch.equal(model[2].weight.flatten(), fixture("tiny_network_weights2")))
        with self.assertRaises(FrozenInstanceError):
            result.input_size = 100

    def test_noncontiguous_parameters_become_contiguous_without_transposition(self):
        layer = nn.Linear(2, 3)
        layer.weight = nn.Parameter(torch.tensor([[1., 2., 3.], [4., 5., 6.]]).t())
        layer.bias = nn.Parameter(torch.tensor([7., 99., 8., 99., 9., 99.])[::2])
        self.assertFalse(layer.weight.is_contiguous())
        self.assertFalse(layer.bias.is_contiguous())
        result = validate_model(nn.Sequential(layer)).layers[0]
        self.assertTrue(result.weights.is_contiguous() and result.bias.is_contiguous())
        self.assertEqual(result.weights.flatten().tolist(), [1., 4., 2., 5., 3., 6.])
        self.assertEqual(result.bias.tolist(), [7., 8., 9.])
        self.assertFalse(layer.weight.is_contiguous())

    def test_finite_extremes_signed_zero_and_subnormal_bits_survive(self):
        layer = nn.Linear(4, 1)
        # Negative zero, smallest positive subnormal, largest finite, negative largest.
        bits = torch.tensor([-2147483648, 1, 2139095039, -8388609], dtype=torch.int32)
        layer.weight = nn.Parameter(bits.view(torch.float32).reshape(1, 4))
        layer.bias = nn.Parameter(torch.tensor([-0.0]))
        snapshot = validate_model(nn.Sequential(layer)).layers[0]
        self.assertTrue(torch.equal(snapshot.weights.view(torch.int32).flatten(), bits))
        self.assertTrue(bool(torch.signbit(snapshot.bias[0])))

    def test_invalid_roots_and_empty_sequence(self):
        for value in (None, [], {}, "model.pth", nn.Linear(2, 2), nn.ModuleList([nn.ReLU()])):
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(ModelValidationError, "Sequential"):
                validate_model(value)
        with self.assertRaisesRegex(ModelValidationError, "1..1024 layers"):
            validate_model(nn.Sequential())

    def test_unsupported_modules_nested_containers_and_subclasses(self):
        class ChangedLinear(nn.Linear):
            def forward(self, value):
                raise AssertionError("must not execute custom forward")

        class ChangedSequential(nn.Sequential):
            pass

        for layer in (nn.Dropout().eval(), nn.Identity(), nn.Sigmoid(), nn.ReLU6(),
                      nn.LogSoftmax(dim=-1), nn.Sequential(nn.ReLU()),
                      nn.LazyLinear(3), ChangedLinear(2, 3)):
            with self.subTest(kind=type(layer).__name__), self.assertRaisesRegex(ModelValidationError, "layer 0.*unsupported module"):
                validate_model(nn.Sequential(layer), input_size=2)
        with self.assertRaisesRegex(ModelValidationError, "exact torch.nn.Sequential"):
            validate_model(ChangedSequential(nn.ReLU()), input_size=2)

    def test_softmax_axis_must_be_explicit_minus_one(self):
        for dim in (None, 0, 1, -2, -1.0, True):
            layer = nn.Softmax(dim=-1)
            # PyTorch rejects some types at construction; still check that
            # corrupted public metadata gets a Cinder diagnostic on validation.
            layer.dim = dim
            with self.subTest(dim=dim), self.assertRaisesRegex(ModelValidationError, "Softmax requires dim=-1"):
                validate_model(nn.Sequential(layer), input_size=2)

    def test_relu_inplace_must_be_boolean(self):
        for value in (1, "yes", None):
            with self.subTest(value=value), self.assertRaisesRegex(ModelValidationError, "inplace must be bool"):
                validate_model(nn.Sequential(nn.ReLU(inplace=value)), input_size=2)

    def test_input_size_validation(self):
        for size in (0, -1, True, 2.0, "2", MAX_VECTOR_SIZE + 1):
            with self.subTest(size=size), self.assertRaisesRegex(ModelValidationError, "input_size.*integer"):
                validate_model(nn.Sequential(nn.Linear(2, 2)), input_size=size)

    def test_linear_dimension_metadata_validation(self):
        for field in ("in_features", "out_features"):
            for value in (0, -1, True, 2.0, MAX_VECTOR_SIZE + 1):
                layer = nn.Linear(2, 2)
                setattr(layer, field, value)
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ModelValidationError, field):
                    validate_model(nn.Sequential(layer))
            layer = nn.Linear(2, 2)
            delattr(layer, field)
            with self.subTest(missing=field), self.assertRaisesRegex(ModelValidationError, field):
                validate_model(nn.Sequential(layer))

    def test_connections_rejected_before_any_parameter_copy(self):
        for model, size in ((nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(4, 1)), None),
                            (nn.Sequential(nn.Linear(2, 3)), 3)):
            with self.subTest(size=size), patch.object(torch.Tensor, "clone", side_effect=AssertionError("early copy")):
                with self.assertRaisesRegex(ModelValidationError, "expects .*preceding vector"):
                    validate_model(model, input_size=size)

    def test_missing_bias_or_weight_and_wrong_shapes(self):
        with self.assertRaisesRegex(ModelValidationError, "bias is required"):
            validate_model(nn.Sequential(nn.Linear(2, 3, bias=False)))
        for name, value in (("weight", None), ("weight", torch.zeros(2, 3)),
                            ("weight", torch.zeros(6)), ("bias", torch.zeros(3, 1)),
                            ("bias", torch.zeros(2))):
            layer = nn.Linear(2, 3)
            setattr(layer, name, nn.Parameter(value) if value is not None else None)
            with self.subTest(name=name, shape=None if value is None else value.shape):
                with self.assertRaisesRegex(ModelValidationError, name):
                    validate_model(nn.Sequential(layer))

    def test_wrong_parameter_dtypes(self):
        for dtype in (torch.float16, torch.float64, torch.bfloat16, torch.complex64, torch.int32):
            for name in ("weight", "bias"):
                layer = nn.Linear(2, 3)
                value = getattr(layer, name).detach().to(dtype)
                setattr(layer, name, nn.Parameter(value, requires_grad=False))
                with self.subTest(dtype=dtype, name=name), self.assertRaisesRegex(ModelValidationError, "expected torch.float32"):
                    validate_model(nn.Sequential(layer))

    def test_non_cpu_storage(self):
        with self.assertRaisesRegex(ModelValidationError, "expected CPU storage.*meta"):
            validate_model(nn.Sequential(nn.Linear(2, 3, device="meta")))

    def test_sparse_parameters_and_tensor_subclasses(self):
        layer = nn.Linear(2, 3)
        layer.weight = nn.Parameter(layer.weight.detach().to_sparse())
        with self.assertRaisesRegex(ModelValidationError, "ordinary strided tensor"):
            validate_model(nn.Sequential(layer))

        class CustomParameter(nn.Parameter):
            pass

        layer = nn.Linear(2, 3)
        layer.weight = CustomParameter(layer.weight.detach())
        with self.assertRaisesRegex(ModelValidationError, "plain Tensor or Parameter"):
            validate_model(nn.Sequential(layer))

    def test_nonfinite_parameters_and_successful_retry(self):
        for name in ("weight", "bias"):
            for value in (float("nan"), float("inf"), -float("inf")):
                model = tiny_network()
                parameter = getattr(model[2], name)
                with torch.no_grad():
                    parameter.flatten()[-1] = value
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ModelValidationError, f"layer 2.*{name}.*finite"):
                    validate_model(model)
                # Restore just the caller's invalid value; validation needs no reset.
                with torch.no_grad():
                    parameter.flatten()[-1] = 0
                self.assertEqual(validate_model(model).output_size, 2)

    def test_local_hooks_are_rejected_without_execution(self):
        for on_root in (False, True):
            for before in (False, True):
                model = tiny_network()
                module = model if on_root else model[0]
                register = module.register_forward_pre_hook if before else module.register_forward_hook
                def callback(*args):
                    raise AssertionError("hook must not execute")
                handle = register(callback)
                try:
                    with self.subTest(root=on_root, before=before), self.assertRaisesRegex(ModelValidationError, "forward hooks"):
                        validate_model(model)
                finally:
                    handle.remove()

    def test_global_hooks_are_rejected_without_execution(self):
        for register in (module_hooks.register_module_forward_hook, module_hooks.register_module_forward_pre_hook):
            model = tiny_network()
            handle = register(lambda *args: self.fail("global hook executed"))
            try:
                with self.assertRaisesRegex(ModelValidationError, "global forward hooks"):
                    validate_model(model)
            finally:
                handle.remove()

    def test_overridden_and_compiled_forward_are_rejected(self):
        for target in ("root", "layer"):
            for attribute in ("forward", "_call_impl", "_compiled_call_impl"):
                model = tiny_network()
                module = model if target == "root" else model[0]
                setattr(module, attribute, lambda *args: self.fail("execution occurred"))
                with self.subTest(target=target, attribute=attribute):
                    with self.assertRaisesRegex(ModelValidationError, "execution is not supported"):
                        validate_model(model)

    def test_pruning_and_parametrizations_are_rejected(self):
        layer = nn.Linear(2, 3)
        prune.identity(layer, "weight")
        with self.assertRaisesRegex(ModelValidationError, "forward hooks"):
            validate_model(nn.Sequential(layer))
        layer = nn.Linear(2, 3)
        parametrize.register_parametrization(layer, "weight", nn.Identity())
        with self.assertRaisesRegex(ModelValidationError, "unsupported module"):
            validate_model(nn.Sequential(layer))

    def test_extra_registered_state_and_nested_children_are_rejected(self):
        for target in ("root", "linear", "activation"):
            model = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
            module = model if target == "root" else model[0 if target == "linear" else 1]
            module.register_buffer("extra", torch.zeros(1))
            with self.subTest(target=target), self.assertRaisesRegex(ModelValidationError, "buffers"):
                validate_model(model)
            del module.extra
            module.register_parameter("extra", nn.Parameter(torch.zeros(1)))
            with self.subTest(target=target), self.assertRaisesRegex(ModelValidationError, "parameters"):
                validate_model(model)
        layer = nn.ReLU()
        layer.add_module("child", nn.ReLU())
        with self.assertRaisesRegex(ModelValidationError, "nested modules"):
            validate_model(nn.Sequential(layer), input_size=2)

    def test_layer_count_and_vector_limits_are_inclusive(self):
        relu = nn.ReLU()
        model = nn.Sequential(*([relu] * MAX_LAYERS))
        result = validate_model(model, input_size=MAX_VECTOR_SIZE)
        self.assertEqual(len(result.layers), MAX_LAYERS)
        self.assertEqual(result.output_size, MAX_VECTOR_SIZE)
        model.append(relu)
        with self.assertRaisesRegex(ModelValidationError, "1..1024 layers"):
            validate_model(model, input_size=2)
        result = validate_model(nn.Sequential(broadcast_linear(1, MAX_VECTOR_SIZE)))
        self.assertEqual(result.output_size, MAX_VECTOR_SIZE)

    def test_parameter_limit_accepts_exact_boundary(self):
        # 4096 * (4095 weights + 1 bias) * 4 = exactly 64 MiB.
        result = validate_model(nn.Sequential(broadcast_linear(4095, 4096)))
        self.assertEqual(result.parameter_bytes, MAX_PARAMETER_BYTES)
        self.assertTrue(bool((result.layers[0].weights == 0).all()))

    def test_parameter_limits_reject_per_layer_and_cumulative_before_copy(self):
        models = [nn.Sequential(broadcast_linear(4096, 4096)),
                  nn.Sequential(broadcast_linear(MAX_VECTOR_SIZE, MAX_VECTOR_SIZE)),
                  nn.Sequential(broadcast_linear(4095, 4096), broadcast_linear(4096, 1))]
        for model in models:
            with self.subTest(layers=len(model)), patch.object(torch.Tensor, "clone", side_effect=AssertionError("early copy")):
                with self.assertRaisesRegex(ModelValidationError, "parameters exceed"):
                    validate_model(model)

    def test_late_unsupported_layer_rejected_before_copy(self):
        model = nn.Sequential(nn.Linear(2, 3), nn.Dropout())
        with patch.object(torch.Tensor, "clone", side_effect=AssertionError("early copy")):
            with self.assertRaisesRegex(ModelValidationError, "layer 1.*Dropout"):
                validate_model(model)

    def test_validation_does_not_call_forward_or_perform_file_io(self):
        model = tiny_network()
        with patch.object(nn.Sequential, "forward", side_effect=AssertionError("forward called")), \
                patch.object(nn.Linear, "forward", side_effect=AssertionError("forward called")), \
                patch("builtins.open", side_effect=AssertionError("file opened")), \
                patch.object(torch, "load", side_effect=AssertionError("checkpoint read")), \
                patch.object(torch, "save", side_effect=AssertionError("file written")):
            self.assertEqual(validate_model(model).file_bytes, 168)
            with self.assertRaises(ModelValidationError):
                validate_model(nn.Sequential(nn.Identity()))

    def test_allocation_error_propagates_and_retry_leaves_source_intact(self):
        model = tiny_network()
        expected = model[0].weight.detach().clone()
        with patch.object(torch.Tensor, "clone", side_effect=MemoryError("forced snapshot failure")):
            with self.assertRaisesRegex(MemoryError, "forced snapshot failure"):
                validate_model(model)
        self.assertTrue(torch.equal(model[0].weight, expected))
        self.assertEqual(validate_model(model).file_bytes, 168)


if __name__ == "__main__":
    unittest.main()
