# NumPy reference verification

Phase 1 math is compared with independent NumPy float32 calculations. This
checks matrix-vector multiplication, bias addition, ReLU, and stable Softmax,
including intermediate outputs of composed operations. The runtime's source
and public API did not need changes to pass these comparisons.

Phase 2 also reuses the saved matrix/bias fixtures in the `dense_forward` test,
comparing `cinder_dense_forward` directly against all 11 `Wx + b` reference
vectors (41 values). This test always uses the saved header, including when
`CINDER_ENABLE_NUMPY_TESTS` is enabled; that option adds fresh comparisons for
the tensor math. See [Dense verification](DENSE.md#completion-and-next-implementation-steps).

The `layer_interface` test also uses the saved header. It compares common
dispatch against both activation references for all 12 activation fixtures,
and against `Wx + b` for all 11 Dense fixtures: 35 result vectors. It compares
ReLU exactly, uses the existing absolute-plus-relative tolerance for Dense and
Softmax, and checks Softmax's probability range and normalization separately.
These layer comparisons remain saved-reference tests even when fresh tensor
generation is enabled. See [layer verification](LAYER.md#verification-and-next-step).

## Two ways to run the same comparison

The ordinary C build runs `tensor_numpy_fixtures` using the saved
[`tests/fixtures/numpy_fixtures.h`](../tests/fixtures/numpy_fixtures.h).
Python and NumPy are not required to compile or run this test:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
ctest --test-dir build --output-on-failure
```

That header is an intentional test artifact tracked in Git, even though it is
generated. It contains inputs, dimensions, and expected outputs as exact C11
hexadecimal float literals. The compiler therefore reconstructs the float32
values without losing precision through decimal formatting. Weights are
flattened in C row-major order. Array lengths are derived with `sizeof`.

`CINDER_ENABLE_NUMPY_TESTS=ON` adds `tensor_numpy_generated`. Its build runs the
generator, writes another header under the build directory, and compiles the
same C comparison test against those newly calculated references. The saved
header is not overwritten. Both executables link the real `cinder_core` library
and receive the normal warning and sanitizer settings.

Use Python 3.11 or newer to create a local environment. For example, with
Python 3.12 available:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r tests/requirements-numpy.txt
cmake -S . -B build-numpy \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DCINDER_ENABLE_NUMPY_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-numpy
ctest --test-dir build-numpy --output-on-failure
```

The requirements file pins NumPy 2.3.5, the version used to create the saved
fixtures. CMake reports the chosen NumPy version, rejects a missing NumPy
installation, and rejects enabling this option with `BUILD_TESTING=OFF`.
Use `Python3_EXECUTABLE` to select another environment deliberately. CMake does
not install Python packages automatically. `.venv/` and Python bytecode are
ignored by Git.

To combine generation with sanitizers, configure a separate Debug build with
`CINDER_ENABLE_ASAN=ON`, `CINDER_ENABLE_UBSAN=ON`, and the same NumPy options.
The build regenerates its header when the generator or requirements file
changes. After changing only the Python/NumPy installation, use a new build
directory or remove that build's `numpy-fixtures/numpy_fixtures.h` to force
regeneration. NumPy is used during the build; the CTest executable itself has
no Python runtime dependency.

## Reference calculations and determinism

[`tests/generate_numpy_fixtures.py`](../tests/generate_numpy_fixtures.py) does
not call Cinder. It uses NumPy operations directly:

```python
product = np.matmul(weights, inputs, dtype=np.float32)
biased = np.add(product, bias, dtype=np.float32)
relu = np.maximum(biased, np.float32(0))

shifted = np.subtract(scores, np.max(scores), dtype=np.float32)
exponentials = np.exp(shifted, dtype=np.float32)
total = np.sum(exponentials, dtype=np.float32)
probabilities = np.divide(exponentials, total, dtype=np.float32)
```

All input arrays, weights, biases, intermediate reference arrays, and expected
outputs are explicitly float32. The generator verifies their shapes, dtypes,
and finiteness. Overflow, division by zero, and invalid arithmetic fail
generation; underflow to finite zero is allowed by the math contract.

Most inputs are written out explicitly. Three additional non-square fixtures
use `Generator(PCG64(20261016))`: bounded integer draws are scaled by powers of
two, producing exactly representable float32 inputs. The generator records
the NumPy version and seed in the header and writes no timestamps.

Repeating generation in the verified local environment produced byte-identical
headers. NumPy's math libraries and accumulation strategies can differ across
platforms, so identical reference bits across every platform are not required.
The saved and newly generated references are each compared numerically against
Cinder using the same tolerance.

Cinder's Softmax uses double intermediates; the NumPy oracle deliberately uses
float32 arithmetic. This compares observable results with an independent
float32 implementation instead of reproducing Cinder's internal precision
choices. Small rounding differences are expected and measured.

## Fixture coverage

There are 11 matrix/bias fixtures:

| Case | Shape | Purpose |
| --- | --- | --- |
| Hand-calculated example | `2 x 3` | Row-major indexing, signs, and intermediate results. |
| Decimal values | `3 x 2` | Non-exact binary representations and rounding. |
| Identity weights | `4 x 4` | Preserve the input before bias. |
| Zero input | `3 x 4` | The biased result must equal the bias. |
| Zero weights | `2 x 3` | Zero product despite nonzero input. |
| Single row | `1 x 5` | Multiple inputs, one output. |
| Single column | `4 x 1` | One input, multiple outputs. |
| Scalar | `1 x 1` | Smallest valid matrix. |
| Seeded non-square matrices | `3 x 7`, `7 x 11`, `11 x 17` | More combinations and longer accumulation. |

For each fixture the test checks all five result vectors:

1. `Wx`.
2. `Wx + b`, including the logits before Softmax.
3. `Softmax(Wx + b)`.
4. `ReLU(Wx + b)`.
5. `Softmax(ReLU(Wx + b))`.

There are also 12 standalone activation fixtures, each checked against both
ReLU and Softmax: a singleton, zeros, signed zeros, mixed signs, all-negative
scores, all-positive scores, large positive offsets, large negative offsets,
tied maxima, equal maximum finite floats, underflow, and a 33-element spread
from `-20` to `20`. In total, each tensor reference test compares **79 vectors and
341 individual results**.

Output arrays begin as NaN so missing writes fail the finiteness check. A spare
element remains NaN after every operation, checking the logical length boundary.
The copied weight tensor must remain unchanged, and all test allocations are
released after each fixture.

## Comparison rule and failure diagnostics

[`tests/test_tensor_numpy.c`](../tests/test_tensor_numpy.c) checks matching,
nonzero vector lengths and finite values on both sides before comparing:

```text
abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)
```

This is the roadmap's `atol = rtol = 1e-5` rule, with the reference on the
right-hand side. The test calculates the differences and limits in double so
the comparison itself does not introduce unnecessary float32 rounding.
The tolerances were not loosened to obtain a pass.

An explicit shape check matters because
[`numpy.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
permits broadcasting, and an explicit finite check matters because matching
infinities can otherwise compare equal. Neither behavior is accepted here.

Failures name the fixture, operation, element index, actual and reference
values, absolute error, and allowed error. Successful runs print the number of
compared vectors/elements, the largest absolute error, and the largest fraction
of the allowed tolerance used by any comparison. A fraction above one fails.

The comparator also checks its own rejection behavior with wrong numbers,
reordered values, mismatched/empty lengths, NaN, and both infinities on either
side. These checks stay active in Release builds.

The general absolute tolerance can hide errors in extremely small probabilities.
The existing Softmax unit tests therefore remain necessary: they apply tighter
checks to subnormal probabilities, underflow, and normalization. Existing unit
tests also retain invalid-input, allocation-failure, signed-zero, and opposite
finite-extreme coverage. This NumPy suite covers successful finite calculations;
it does not redefine Cinder's error behavior using NumPy's behavior.

## Updating the saved fixtures

Edit the readable generator, then regenerate and review the saved header:

```sh
.venv/bin/python tests/generate_numpy_fixtures.py
cmake --build build
ctest --test-dir build --output-on-failure
```

Use `--output /path/to/numpy_fixtures.h` to generate a separate copy for review.
Do not hand-edit expected numbers to silence a failure. Investigate shape,
layout, arithmetic, and reference differences first; document any justified
tolerance change explicitly.

## Verification recorded on October 16, 2026

Local verification used Apple Clang 21.0.0, CMake 4.4.4, Python 3.12.14, and
NumPy 2.3.5. Python and NumPy came from the installed Codex dependency runtime;
the system Python 3.9.6 was not modified.

- The ordinary Debug build passed all 9 CTest tests using the saved fixtures.
- Release and combined ASan/UBSan builds each passed all 10 tests, including
  saved and newly generated NumPy references.
- Each reference run compared 79 vectors / 341 results. The largest absolute
  error was `1.1920929e-7`; the largest fraction of tolerance was
  `0.00600148273`, approximately 0.60% of the allowed error.
- Repeated local generation produced identical files.
- CMake correctly rejected missing NumPy and incompatible testing options.
- The build with tests disabled succeeded and registered zero tests.
- Native macOS `leaks` reported zero leaks for `test_tensor_numpy`.

The CI workflow runs the saved fixtures on macOS, Linux, and Linux with
ASan/UBSan/leak detection. The ordinary Linux job also installs the pinned NumPy
dependency in a virtual environment and generates fresh references. Hosted
results for these changes remain pending a future push.

This completes the local Phase 1 math verification. Phase 2's standalone
[Dense API](DENSE.md) and [common layer dispatch](LAYER.md) are now also verified
against the saved references.
Sequential models, serialized models, an exporter, and MNIST remain future
verification steps.
