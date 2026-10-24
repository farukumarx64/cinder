"""Shared saved NumPy data for exporter tests and binary fixture generation."""

from collections import OrderedDict
from pathlib import Path
import re

import torch
from torch import nn


FIXTURES = (Path(__file__).parent / "fixtures" / "numpy_fixtures.h").read_text()


def fixture_values(name):
    match = re.search(r"static const float " + re.escape(name) + r"\[\] = \{(.*?)\};", FIXTURES, re.S)
    if match is None:
        raise ValueError(f"Missing saved fixture: {name}")
    return [float.fromhex(value.strip().removesuffix("f"))
            for value in match.group(1).split(",") if value.strip()]


def fixture(name):
    return torch.tensor(fixture_values(name), dtype=torch.float32)


def case_names():
    return re.findall(r'\{"[^"]+", NUMPY_VECTOR\(tiny_(\w+)_input\)', FIXTURES)


def tiny_network():
    model = nn.Sequential(OrderedDict([
        ("hidden", nn.Linear(2, 4)),
        ("relu", nn.ReLU()),
        ("logits", nn.Linear(4, 2)),
        ("probabilities", nn.Softmax(dim=-1)),
    ]))
    with torch.no_grad():
        for layer, number in ((model[0], 1), (model[2], 2)):
            layer.weight.copy_(fixture(f"tiny_network_weights{number}").reshape_as(layer.weight))
            layer.bias.copy_(fixture(f"tiny_network_bias{number}"))
    return model
