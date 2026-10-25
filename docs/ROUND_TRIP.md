# Complete exporter/loader round trip

The fixed `2 -> 4 -> 2` network is checked through **Python export -> `.cnd` ->
C loading -> C inference** at every stage. The expected outputs remain the saved
NumPy float32 references; exporting never recomputes the expected answers.
No production runtime API or implementation changes were required.

## Exposing each stage through the public model API

The model is opaque and does not expose intermediate buffers. The fixture
generator exports three prefixes of the same fixed network, plus the complete
model. Each prefix executes all preceding layers from the original input.

| Stage | Saved model | Layers executed | Output size | File bytes |
| --- | --- | --- | ---: | ---: |
| First Dense | `prefixes/dense1.cnd` | Dense | 4 | 80 |
| ReLU | `prefixes/relu.cnd` | Dense, ReLU | 4 | 96 |
| Final logits | `prefixes/logits.cnd` | Dense, ReLU, Dense | 2 | 152 |
| Probabilities | `tiny.cnd` | Dense, ReLU, Dense, Softmax | 2 | 168 |

[`tests/generate_cnd_fixtures.py`](../tests/generate_cnd_fixtures.py) calls the
public `export_model` for all four models. The prefix headers declare their
actual layer counts. A binary-export test independently compares each prefix
with the corresponding bytes of the specification's tiny-network example,
changing only the declared layer count. This checks that the prefixes contain
the same parameters and preceding layers as the complete model.

All models are saved under [`tests/fixtures/tiny`](../tests/fixtures/tiny).
The original `tiny.cnd`, ten raw input files, saved C reference arrays, and all
JSON expected outputs are unchanged. The fixture set grows from 12 to 15 files.
`references.json` additionally maps stages to model files and records their
SHA-256 hashes; this is fixture metadata, not a change to `.cnd` version 1.

## The C comparison

[`tests/test_model_round_trip.c`](../tests/test_model_round_trip.c) links the
normal `cinder_core` library. It loads all four files with `cinder_model_load`,
then reads each of the ten `inputs/<case>.f32` files and runs `cinder_model_run`.
It checks:

1. Model input size is two and output sizes are exactly `4, 4, 2, 2`.
2. Each raw input file contains exactly eight bytes. The test explicitly decodes
   two little-endian binary32 values, rejects non-finite encodings, and checks
   that the decoded input bits match the corresponding saved NumPy input.
3. Every result has the same nonempty shape as its saved reference. Both actual
   and expected values must be finite before comparison.
4. The shared [`numpy_compare.h`](../tests/numpy_compare.h) applies the unchanged
   rule to each value:

   ```text
   abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)
   ```

5. Input storage, guard values, and spare output capacity remain unchanged.
   Exact output capacity and one spare slot alternate across inputs. Softmax
   values must also lie in `[0, 1]` and sum to one within `1e-6`.

All four loaded models reuse their working buffers across ten different inputs.
Every inference starts from the original input file; a saved intermediate
reference is never passed into a later layer. The fourth model runs the entire
network through Softmax. Prefix outputs expose the earlier stages through the
same public API without a tracing API or access to private model storage.

The test prints per-stage and combined error statistics and releases every
loaded model on both success and comparison failure. Its checks remain active
in Release builds. Raw input decoding is test support, not a new CLI/input API.

## Two CTest paths

`model_round_trip_fixtures` is always registered when `BUILD_TESTING=ON`. It
uses committed model/input files and the saved NumPy C header. Configuring,
building, and running this path requires neither Python nor PyTorch nor NumPy.

For a C-only build:

```sh
cmake -S . -B build-round-trip -DCMAKE_BUILD_TYPE=Release \
  -DCINDER_ENABLE_NUMPY_TESTS=OFF -DCINDER_ENABLE_EXPORTER_TESTS=OFF
cmake --build build-round-trip --parallel
ctest --test-dir build-round-trip -R '^model_round_trip_fixtures$' --output-on-failure -V
```

`model_round_trip_exported` is added by `CINDER_ENABLE_EXPORTER_TESTS=ON`.
[`tests/check_model_round_trip.py`](../tests/check_model_round_trip.py) creates a
fresh temporary directory, generates model prefixes/the complete model and
input files through the existing fixture generator, then invokes the **same C
executable** on that directory. Expected outputs still come from the saved C
header. It removes temporary exports when finished and never overwrites the
committed fixtures. Paths are passed as subprocess arguments without a shell.

With the optional Python/PyTorch environment from [the exporter guide](EXPORTER.md):

```sh
cmake -S . -B build-round-trip-exporter -DCMAKE_BUILD_TYPE=Debug \
  -DCINDER_ENABLE_EXPORTER_TESTS=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build-round-trip-exporter --parallel
ctest --test-dir build-round-trip-exporter -R '^model_round_trip_' --output-on-failure -V
```

Enabling fresh NumPy generation is separate: it still adds the existing tensor
and in-memory network comparisons. Both round-trip tests deliberately use the
saved references. The Python round-trip driver itself does not need NumPy.

## Checks that detect an incorrectly passing comparison

After the successful live round trip, the Python driver makes nine temporary
changes and requires the C executable to exit with a comparison error, not a
crash. Each file is restored afterward:

- Change the first Dense bias in each of the four model files separately.
  These remain valid, finite, correctly shaped `.cnd` files; the corresponding
  stage must report a numerical mismatch. This proves all four files are used.
- Substitute the full model for the first-Dense prefix: reject its output shape.
- Truncate an input, append an input byte, encode an infinity, or replace the
  worked input with different finite values: reject the input length, finiteness,
  or saved-input identity as appropriate.

The driver checks the expected diagnostic for each altered fixture. Successful
generation and the separate binary-export tests also verify that all 15 fixture
files regenerate identically. Check saved fixtures without modifying them with:

```sh
.venv/bin/python tests/generate_cnd_fixtures.py --check
```

## Verification — October 25, 2026

Both the saved-file test and the fresh Python-export test compare **40 vectors /
120 values** (10 inputs x four stages) per successful run. Local Debug results:

| Stage | Vectors | Values | Maximum absolute error |
| --- | ---: | ---: | ---: |
| First Dense | 10 | 40 | 0 |
| ReLU | 10 | 40 | 0 |
| Final logits | 10 | 20 | 0 |
| Softmax | 10 | 20 | `5.96046448e-8` |

The largest fraction of the permitted tolerance is `0.00303036345`, about
0.303% of the allowance. Exact first-three-stage equality is a measured result
for these fixtures and this toolchain, not a new cross-platform guarantee.

Verification ran on Apple Silicon macOS with Apple Clang 21.0.0 and CMake
4.4.4. Optional checks used Python 3.12.14, PyTorch 2.14.1, and NumPy 2.3.5.

| Configuration or check | Result |
| --- | --- |
| Debug, optional NumPy and exporter tests enabled | 27/27 CTest tests passed |
| Fresh Release build, both Python options disabled | 22/22 CTest tests passed; no Python discovery or execution |
| Debug, ASan + UBSan, both optional Python suites enabled | 27/27 CTest tests passed; no sanitizer diagnostics |
| Python validation and binary-export suites | 33 + 18 unittest methods passed, included in the optional CTest run |
| Fixture regeneration check | All 15 saved files matched byte for byte |
| New C test compiled with warnings as errors | C11, `-Wall -Wextra -Wpedantic -Werror` syntax check passed |
| Native macOS `leaks --atExit` on the C round-trip executable | 0 leaks / 0 leaked bytes |

The sanitizer run also exercised the nine deliberate failures through the
instrumented C executable. The native leak check covered the successful C
round trip; it emitted macOS's restricted-process memory-display warning but
completed with status zero and reported zero leaks. LeakSanitizer is not
supported by this Mac's Apple Clang runtime. Big-endian and other native ABI
execution remain untested.

The existing Linux CI build enables exporter tests and will run the live Python
round trip. The macOS build and Linux sanitizer jobs run the saved C fixture path by
default. The subsequent [loaded-model lifecycle audit](M3_VERIFICATION.md) also
passes locally. Hosted results for both additions remain pending a future push;
M3 stays open until that final verification passes.
