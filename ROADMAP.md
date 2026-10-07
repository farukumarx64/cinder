# Cinder

**A Tiny Neural-Network Inference Runtime in C**

Cinder is a lightweight neural-network inference runtime written in C.

The goal of `v0.1.0` is simple:

> Take a neural network trained elsewhere, load its weights in pure C, perform forward inference on the CPU, and produce results matching a trusted Python implementation.

Cinder is **not** intended to replace PyTorch, TensorFlow, ONNX Runtime, or other full machine-learning frameworks.

The purpose of the project is to understand what happens underneath those abstractions.

This plan was revised on October 7, 2026 to clarify implementation contracts and introduce verification earlier. The project began in an empty folder; the checklist below records actual progress, while the later milestones remain targets.

The daily dates are planning targets. Preserve the milestone order, the October 25 feature freeze, and the October 31 release target; adjust individual workdays to actual progress. Begin with M0 even though its original target dates have passed.

All predictions, accuracy percentages, and benchmark timings shown in this document are illustrative until measured.

## Current Progress

Phase 0 is in progress. Complete and verify each scoped setup task before moving to tensor implementation.

- [x] Install CMake and verify the existing C compiler.
- [x] Configure the C11 executable with `-Wall`, `-Wextra`, and `-Wpedantic`; build and run the minimal entry point.
- [x] Initialize Git on `main` and add ignore rules for generated build files and macOS metadata.
- [x] Add the basic README describing the current setup, prerequisites, build commands, and development workflow.
- [ ] Choose and add the project license. **Current scoped task; awaiting the owner's selection.**
- [ ] Implement `cinder --version`.
- [ ] Add the CTest version-command smoke test.
- [ ] Configure sanitizer builds and supported leak checks.
- [ ] Set up the GitHub repository and remote after choosing its location and visibility.
- [ ] Add and verify macOS and Linux CI.
- [ ] Verify the documented setup from a fresh checkout before closing M0.

---

# 1. v0.1.0 Goal

By the end of October, Cinder should support a workflow like:

```bash
$ cinder info mnist.cnd
```

Output:

```text
Model: MNIST MLP
Input: 784

Layers:
  Dense 784 -> 128
  ReLU
  Dense 128 -> 10
  Softmax

Parameters: 101,770
```

Inference:

```bash
$ cinder run mnist.cnd digit.bin
```

Output:

```text
Prediction:

7    98.42%
1     0.71%
9     0.39%

Inference time: 0.84 ms
```

Benchmarking:

```bash
$ cinder bench mnist.cnd digit.bin --iterations 1000
```

Output:

```text
Iterations:       1000
Average:          0.79 ms
Min:              0.73 ms
Max:              1.12 ms
```

That is enough for a strong `v0.1.0`.

---

# 2. MVP Scope

## Required for v0.1.0

- [ ] Pure C runtime
- [ ] CPU inference
- [ ] `float32` tensors
- [ ] Dense / fully connected layers
- [ ] ReLU activation
- [ ] Softmax activation
- [ ] Sequential neural networks
- [ ] Model loading from disk
- [ ] Custom Cinder model format
- [ ] Python model exporter
- [ ] CLI inference
- [ ] CLI model inspection
- [ ] Benchmark command
- [ ] Unit tests
- [ ] Python/reference comparison tests
- [ ] Memory-error testing from the first allocation code onward
- [ ] Automated macOS and Linux build/test checks
- [ ] Tiny exported-network reference test before MNIST
- [ ] MNIST demonstration
- [ ] macOS compilation
- [ ] Linux compilation
- [ ] GitHub release

---

# 3. Explicitly Out of Scope

Do **not** include the following in `v0.1.0`:

- training
- backpropagation
- autograd
- CUDA
- GPU inference
- convolutional layers
- transformers
- attention
- ONNX
- Python bindings
- HTTP serving
- networking
- INT8 quantization
- SIMD
- NEON
- AVX
- multithreading
- arbitrary computation graphs
- dynamic tensor shapes

These can become future versions.

The goal of `v0.1.0` is not feature completeness.

The goal is a **small, understandable, reliable inference runtime**.

---

## Implementation Decisions for v0.1.0

These decisions refine the existing scope without adding operators or a general framework:

- **Execution:** one sample per inference, fixed dimensions, and a sequential list of Dense, ReLU, and Softmax layers. No batching in v0.1.0.
- **Representation:** contiguous float32 vectors and matrices. Dense weights have shape `[output_size][input_size]` in row-major order. Keep the tensor abstraction small; strides, views, broadcasting, and general tensor machinery are unnecessary for this release.
- **API:** callers provide input length and output capacity. The API exposes model input/output sizes and returns status codes with useful error details. Ownership is explicit.
- **Formats:** document fixed-width fields, little-endian byte order, IEEE 754 binary32 values, weight layout, and input preprocessing before writing the serializer and loader. Never serialize native C structs directly.
- **Exporter:** accept a constructed Python `torch.nn.Sequential` containing only explicitly supported layer configurations. Reject unsupported configurations clearly. Importing arbitrary `.pth` files is outside v0.1.0.
- **Verification:** run unit tests, sanitizers, and macOS/Linux CI early. Compare intermediate outputs and final outputs against deterministic float32 reference fixtures with explicit absolute and relative tolerances.
- **First end-to-end checkpoint:** export the tiny `2 → 4 → 2` network, load it in C, and match its reference outputs before beginning MNIST integration.

The detailed binary layout belongs in `docs/MODEL_FORMAT.md` during Phase 3. These decisions are enough to start M0 and the core math work now.

---

# 4. Repository Structure

```text
cinder/
│
├── CMakeLists.txt
├── README.md
├── LICENSE
├── ROADMAP.md
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── include/
│   └── cinder/
│       ├── tensor.h
│       ├── ops.h
│       ├── model.h
│       ├── runtime.h
│       └── error.h
│
├── src/
│   ├── tensor.c
│   ├── ops.c
│   ├── model.c
│   ├── runtime.c
│   ├── error.c
│   └── cli.c
│
├── tests/
│   ├── test_tensor.c
│   ├── test_ops.c
│   ├── test_model.c
│   ├── test_runtime.c
│   └── fixtures/  # tiny model, inputs, and per-layer reference outputs
│
├── tools/
│   └── export_model.py
│
├── examples/
│   └── mnist/
│       ├── train.py
│       ├── model.cnd
│       ├── sample.bin
│       └── README.md
│
└── docs/
    ├── MODEL_FORMAT.md
    └── ARCHITECTURE.md
```

Use **CMake** as the build system.

---

# 5. Public API

Keep the initial API small while making buffer sizes and failure details explicit.

Proposed interface:

```c
cinder_status_t cinder_model_load(
    const char *path,
    cinder_model_t **out_model,
    cinder_error_t *error
);

size_t cinder_model_input_size(const cinder_model_t *model);
size_t cinder_model_output_size(const cinder_model_t *model);

cinder_status_t cinder_model_run(
    cinder_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);

void cinder_model_free(cinder_model_t *model);
```

API requirements:

- `input_count` and `output_capacity` count float elements, not bytes.
- Require the exact model input count and enough output capacity before computation begins. As in any pointer-based C API, the caller must report the actual available buffer sizes.
- Return `CINDER_OK` on success and a specific status on failure. Define status codes and caller-owned error details in `error.h`; avoid relying on printed messages or global error state.
- Model loading sets `*out_model` to `NULL` on failure and frees any partially loaded state.
- The model owns its layers, weights, biases, and reusable working buffers. The caller owns the input/output arrays; the runtime does not retain those pointers after a run.
- Document supported buffer aliasing. Initially require separate input and output buffers and leave output unspecified if execution fails.
- Inference on a single model instance is sequential. A model that owns mutable working buffers must not be run concurrently.
- `cinder_model_free(NULL)` is safe.

The runtime supports only sequential models in v0.1.0.

Internally:

```text
Input
  ↓
Dense
  ↓
ReLU
  ↓
Dense
  ↓
Softmax
  ↓
Output
```

No computational graph engine.

No scheduler.

No DAG.

No unnecessary abstractions.

---

# 6. Cinder Model Format

Cinder should have a very small custom binary model format.

Use:

```text
.cnd
```

Example:

```text
mnist.cnd
```

A simplified representation could be:

```text
MAGIC
VERSION
INPUT_SIZE
LAYER_COUNT

LAYER
  TYPE
  INPUT_SIZE
  OUTPUT_SIZE
  WEIGHTS
  BIASES

LAYER
  TYPE
  ...

END
```

Use a magic header such as:

```text
CNDR
```

Followed by:

```text
version = 1
```

The magic header identifies Cinder files and the version permits rejection of unsupported formats. Structural validation catches malformed files; these fields alone do not detect every corrupted weight.

Possible conceptual structure:

```text
CNDR
│
├── Version
├── Metadata
├── Input shape
├── Layer count
│
├── Dense
│   ├── Input size
│   ├── Output size
│   ├── Weights
│   └── Bias
│
├── ReLU
│
├── Dense
│   ├── Input size
│   ├── Output size
│   ├── Weights
│   └── Bias
│
└── Softmax
```

The Python exporter converts a supported trained model into this format.

Before writing the exporter or loader, the format specification must define:

- Exact offsets or record layout, numeric layer identifiers, and payload lengths.
- Fixed-width unsigned integer fields and little-endian encoding; do not write `size_t`, native enum values, pointers, or padded structs to disk.
- Little-endian IEEE 754 binary32 parameters, with Dense weights stored as row-major `[output_size][input_size]` followed by `[output_size]` biases.
- Consistent layer dimensions: Dense input must match the previous output; ReLU and Softmax preserve the vector length.
- A documented policy for model names/metadata and the end of the file. The CLI's model name must have a defined source.
- Limits for layer count, dimensions, and total parameter bytes. Check multiplication/addition overflow and available payload bytes before allocation or reading.
- Rejection of zero dimensions, unsupported records, incomplete payloads, and non-finite parameters. Clean up all allocations on failure.

A finite weight changed by a bit flip may still be structurally valid. General corruption detection would require an integrity mechanism such as a checksum, which is not required for v0.1.0.

### Input File Contract

For `cinder run` and `cinder bench`, the input file contains exactly the model's input count of little-endian IEEE 754 binary32 values, with no header. Reject missing or extra bytes and non-finite inputs.

The MNIST example uses 784 values obtained by flattening a 28 × 28 image in row-major order and converting pixel values to float32 in `[0, 1]` by dividing by 255. Use the same preprocessing in training, reference inference, and sample generation. The C runtime receives already-preprocessed values.

Document this input contract in `docs/MODEL_FORMAT.md` and the MNIST README. The exporter/example tooling must generate sample files so users do not have to guess their layout.

Cinder itself only performs inference.

---

# 7. October Development Roadmap

## Phase 0 — October 4–5

### Project Bootstrap

Create:

```text
repo
CMake
src/
include/
tests/
README
ROADMAP
LICENSE
```

Get this working:

```bash
cmake -S . -B build
cmake --build build
```

Then:

```bash
./build/cinder --version
```

Output:

```text
Cinder 0.1.0-dev
```

Enable warnings:

```text
-Wall
-Wextra
-Wpedantic
```

Also establish verification during bootstrap:

- Register a minimal smoke test with CTest and run it through `ctest --test-dir build --output-on-failure`.
- Add macOS and Linux CI jobs that configure, build, and run tests.
- Add a CMake option for AddressSanitizer and UndefinedBehaviorSanitizer in development builds, with a sanitizer CI job on a supported runner.
- Enable leak detection where supported and document which platform/check provides it.

Use sanitizer builds as soon as tensor allocations exist. Cross-platform builds and memory checks are part of development, not first attempted in the release week.

Refresh:

- pointers
- structs
- heap allocation
- `malloc`
- `calloc`
- `free`
- header files
- static memory
- dynamic memory
- row-major arrays

### Milestone

```text
M0 — Cinder builds, prints its version, and passes a smoke test with CI and sanitizer configuration in place
```

---

# 8. Phase 1 — October 6–9

## Tensor and Mathematical Operations

Create a minimal tensor representation.

For `v0.1.0`, two-dimensional tensors are enough.

Example:

```c
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} cinder_tensor_t;
```

Implement:

```text
cinder_tensor_create
cinder_tensor_free
cinder_tensor_fill
cinder_tensor_get
cinder_tensor_set
```

Then implement the operations needed for inference:

```text
matrix × vector
bias addition
ReLU
Softmax
```

The central Dense operation is:

```text
y = Wx + b
```

---

## October 6

Implement tensor allocation and memory management.

Define ownership, reject invalid dimensions, check allocation-size overflow, and handle allocation failures. Test these behaviors under the sanitizer build as they are introduced.

Understand exactly:

```text
struct
   ↓
dimensions
   ↓
data pointer
   ↓
contiguous float array
```

---

## October 7

Implement matrix-vector multiplication.

Write or deeply understand the first implementation yourself.

Conceptually:

```text
weight matrix

[w11 w12 w13]
[w21 w22 w23]

×

input

[x1]
[x2]
[x3]

=

output

[y1]
[y2]
```

Do not treat this as magic generated by Codex.

You should understand exactly what every loop is calculating.

---

## October 8

Implement:

```text
ReLU
Softmax
```

ReLU:

```text
f(x) = max(0, x)
```

For Softmax, use a numerically stable implementation. Reject empty vectors. Define non-finite values as unsupported in v0.1.0 and report an error if computation produces a non-finite result.

Instead of simply:

```c
exp(x)
```

subtract the largest logit before exponentiation.

Conceptually:

```text
softmax(x_i) =
exp(x_i - max(x))
-----------------
Σ exp(x_j - max(x))
```

---

## October 9

Test all operations against Python/NumPy.

Example:

```text
NumPy result
      ↓
compare
      ↓
Cinder result
```

Use a documented comparison rule:

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Start with `atol = 1e-5` and `rtol = 1e-5` for the small deterministic float32 fixtures. Confirm these thresholds against the supported operations and model sizes; record any justified changes rather than silently weakening a failing test.

Verify equal shapes and finite values before numerical comparison. Exercise negative values, zeros, non-square matrices, and large finite logits. Keep reference arrays and parameters explicitly float32.

For networks, compare each layer's output, including logits before Softmax, as well as final probabilities. A matching top prediction alone is insufficient.

NumPy documents this tolerance rule in [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).

### Milestone

```text
M1 — Cinder can perform the mathematics required for neural inference
```

---

# 9. Phase 2 — October 10–14

## Layers and Runtime

Create basic layer types:

```c
typedef enum {
    CINDER_LAYER_DENSE,
    CINDER_LAYER_RELU,
    CINDER_LAYER_SOFTMAX
} cinder_layer_type_t;
```

Conceptually:

```text
model
│
├── layer 0
├── layer 1
├── layer 2
└── layer 3
```

Runtime execution can initially be roughly:

```c
for (size_t i = 0; i < model->layer_count; i++) {
    cinder_execute_layer(...);
}
```

---

## October 10–11

Implement the Dense layer.

Each Dense layer needs:

```text
input size
output size
weights
biases
```

---

## October 12

Implement activation layers:

```text
ReLU
Softmax
```

---

## October 13

Implement sequential model execution.

Initially hard-code a tiny network:

```text
2 inputs
   ↓
Dense 2 → 4
   ↓
ReLU
   ↓
Dense 4 → 2
   ↓
Softmax
```

Feed deterministic inputs and known weights through it and verify every layer's output against Python. Keep these fixtures for the exporter/loader round trip in Phase 3. Per-layer inspection can stay inside tests; it does not need a new public tracing API.

---

## October 14

Focus on memory ownership.

For every pointer in the runtime, you should be able to answer:

> Who allocated this?

and:

> Who is responsible for freeing it?

Avoid unclear ownership.

Once the basic execution path is correct, size two reusable working buffers from the largest required activation vector and allocate them with the model. Keep ownership explicit and avoid allocation inside a normal inference call. This is sufficient for v0.1.0; a general arena or memory planner can wait.

### Milestone

```text
M2 — A complete hard-coded neural network executes entirely in C
```

---

# 10. Phase 3 — October 15–18

## Model Serialization

Replace the hard-coded application path with model loading, while retaining the tiny network as a test fixture.

Models should now come from:

```text
model.cnd
```

The process becomes:

```text
Python model
     ↓
Cinder exporter
     ↓
model.cnd
     ↓
Cinder loader
     ↓
runtime
```

---

## October 15

Write:

```text
docs/MODEL_FORMAT.md
```

Do this **before writing the parser**.

Document the binary format clearly.

For example:

```text
Offset     Type        Meaning

0          char[4]     "CNDR"
4          uint32      version
8          uint32      input size
12         uint32      layer count
...
```

Resolve all format and input-contract requirements from Section 6 before implementing the serializer or parser. The byte-offset table above is illustrative, not the final specification. Define explicit resource limits and a trailing-data policy in the final spec.

The exact format can evolve during development, provided the specification, exporter, loader, and fixtures change together.

---

## October 16

Build the Python serializer in `tools/export_model.py` with a small callable interface.

Example usage from Python after constructing or training a supported model:

```python
from tools.export_model import export_model

export_model(model, "model.cnd")
```

For v0.1.0, accept a flat `torch.nn.Sequential` with `Linear` layers with bias, `ReLU`, and `Softmax(dim=-1)` over the feature vector. Export on the CPU in float32, preserving the specified weight layout. Reject unsupported modules, dimensions, dtypes, or configurations with clear errors rather than approximating their behavior.

Checkpoint loading belongs in the example's Python code, which knows the architecture. A `.pth` file containing a `state_dict` is not a complete description of execution order or parameter-free activations; see [PyTorch's save/load example](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html).

Export the existing tiny `2 → 4 → 2` network first. MNIST integration comes after this path is verified.

---

## October 17

Implement the C model loader and its malformed-file tests together. Validate every field before using it for allocation, indexing, or payload reads; ensure partial-load failures release all allocated memory.

Conceptually:

```text
open file
   ↓
read CNDR magic
   ↓
validate version
   ↓
read model metadata
   ↓
allocate layers
   ↓
read parameters
   ↓
model ready
```

---

## October 18

Complete validation coverage and the first end-to-end checkpoint.

Test:

```text
invalid magic
unsupported version
truncated model
invalid dimensions
unknown layer types
missing data
inconsistent adjacent layer dimensions
size arithmetic overflow and resource limits
non-finite parameters
trailing data according to the format policy
cleanup after partial-load failures
```

Errors should be understandable.

Example:

```text
Cinder error: invalid model header
```

rather than:

```text
Segmentation fault
```

### Milestone

```text
M3 — The tiny network exports from Python, loads in C, and matches reference outputs layer by layer
```

Required end-to-end fixture:

```text
Known 2 → 4 → 2 sequential model + deterministic inputs
     ↓
Python reference outputs for every layer
     ↓
export_model → tiny.cnd → Cinder loader → runtime
     ↓
Matching shapes, finite values, and numerical agreement
```

Check in the tiny model, inputs, expected outputs, and a reproducible way to regenerate them. The C-only tests should run from fixtures without requiring Python or downloading MNIST.

Do not begin MNIST integration until this checkpoint passes. At this point Cinder has become a genuine inference runtime.

---

# 11. Phase 4 — October 19–22

## MNIST Demonstration

Create a simple MNIST classifier externally.

Architecture:

```text
784
 ↓
Dense 784 → 128
 ↓
ReLU
 ↓
Dense 128 → 10
 ↓
Softmax
```

Train using Python.

PyTorch is fine. Use the preprocessing contract from Section 6 and record the training seed and example dependency versions for reproducibility.

If training with `CrossEntropyLoss`, pass logits to the loss; append Softmax to the inference/export model afterward. Use evaluation mode for reference inference.

Training is not part of Cinder.

---

## October 19

Train the reference MNIST model.

---

## October 20

Export the trained model:

```text
PyTorch
   ↓
export_model.py
   ↓
mnist.cnd
```

---

## October 21

Run MNIST inference with Cinder.

Example:

```bash
cinder run mnist.cnd sample.bin
```

Output:

```text
Prediction:

7    98.42%
1     0.71%
9     0.39%
```

---

## October 22

Create reference/golden tests.

Take the same inputs and run:

```text
PyTorch
```

and:

```text
Cinder
```

Compare outputs.

Example:

```text
PyTorch:

7 = 0.984123

Cinder:

7 = 0.984121
```

Test against many images.

Ideally:

```text
100–1000 samples
```

and report maximum absolute error as well as pass/fail results under the documented absolute-plus-relative tolerance policy. Compare logits and final probabilities; include intermediate-layer comparisons in diagnostic tests.

Track classification accuracy on a named dataset split separately from runtime agreement. A runtime can match a poorly trained model exactly, so these measure different things.

### Milestone

```text
M4 — Cinder successfully performs real neural-network inference
```

At this point the MVP technically exists.

---

# 12. Phase 5 — October 23–25

## Benchmarking and Runtime Improvements

Add:

```bash
cinder bench
```

Example:

```bash
cinder bench mnist.cnd sample.bin --iterations 1000
```

Measure:

```text
iterations
average latency
minimum latency
maximum latency
throughput
```

Define the benchmark before interpreting results:

- Use an optimized Release build and record the CPU, operating system, compiler, and build options.
- Load the model and input and allocate working buffers before timing.
- Run warm-up iterations, then measure forward inference with a monotonic clock. Exclude file I/O, preprocessing, and output printing from the reported inference time.
- Validate the iteration count and ensure benchmark outputs are consumed so computation cannot be discarded by optimization.
- Report the iteration count, mean, minimum, maximum, and throughput. Treat noisy individual measurements cautiously.
- Confirm numerical correctness after any performance change.

Do not obsess over beating established runtimes.

Benchmark Cinder against **previous versions of Cinder**.

Example:

```text
Cinder naive runtime        1.41 ms
Cinder optimized buffers    0.94 ms
```

The important part is understanding why performance changed.

---

## Memory Reuse

Initially you may accidentally perform allocations during every layer:

```text
Dense
 ↓
malloc

ReLU
 ↓
malloc

Dense
 ↓
malloc
```

Verify that the reusable working buffers introduced with the runtime eliminate allocations during inference. Use an earlier allocation-heavy implementation for comparison only if one exists; do not introduce one just to create a benchmark.

For sequential networks, you may only need:

```text
Buffer A
Buffer B
```

Then:

```text
input
 ↓
Buffer A
 ↓
Buffer B
 ↓
Buffer A
 ↓
Buffer B
```

This avoids unnecessary allocations during inference.

---

# October 25 — FEATURE FREEZE

On October 25:

> No more features for `v0.1.0`.

If you suddenly think:

```text
What if Cinder supports ONNX?
```

Create:

```text
Issue: feat: ONNX importer
Milestone: v0.2.0+
```

Do not implement it.

The final week exists for making the existing runtime reliable.

---

# 13. Phase 6 — October 26–28

## Testing and Robustness

Extend the tests and sanitizer checks already running since Phase 0; this is the final hardening pass, not their first use.

Run:

```text
AddressSanitizer
UndefinedBehaviorSanitizer
Leak detection on a supported platform
```

Test:

```text
memory leaks
double frees
use-after-free
invalid models
zero dimensions
incorrect input sizes
large dimensions
corrupted files
allocation failures
unsupported versions
```

Cinder should fail gracefully. Release requires no known sanitizer findings or leaks in the exercised paths; a passing sanitizer run does not prove the absence of all memory bugs.

Bad:

```text
Segmentation fault
```

Good:

```text
Cinder error:
model expects 784 input values but received 256
```

---

# 14. Phase 7 — October 29–31

## Documentation and Release

The README should explain:

### What is Cinder?

> Cinder is a tiny neural-network inference runtime written in C, created to explore how neural networks execute beneath high-level machine-learning frameworks.

---

## Architecture

```text
                   ┌──────────────┐
model.cnd ───────→ │ Model Loader │
                   └──────┬───────┘
                          ↓
                   ┌──────────────┐
input ───────────→ │   Runtime    │
                   └──────┬───────┘
                          ↓
                   Dense → ReLU
                          ↓
                   Dense → Softmax
                          ↓
                       Output
```

Document:

```text
Installation
Building
CLI
Architecture
Model format
Examples
Benchmarks
Limitations
Roadmap
```

---

# 15. Release v0.1.0

When everything is ready:

```bash
git tag v0.1.0
git push origin v0.1.0
```

Create the GitHub release:

```text
Cinder v0.1.0
```

The release should include:

```text
Basic FP32 neural-network inference
Dense layers
ReLU
Softmax
Cinder .cnd model format
Python exporter
MNIST example
CLI inference
Model inspection
Benchmarking
```

---

# 16. Definition of Done

Cinder `v0.1.0` is finished when:

- [ ] Pure C inference runtime works
- [ ] Builds on macOS
- [ ] Builds on Linux
- [ ] Tensor representation works
- [ ] Matrix-vector multiplication works
- [ ] Dense layers work
- [ ] ReLU works
- [ ] Softmax works
- [ ] `.cnd` models load from disk
- [ ] Python exporter works
- [ ] MNIST model executes successfully
- [ ] Tiny exported-network fixture matches each reference layer before MNIST integration
- [ ] Results satisfy documented absolute and relative tolerances with matching shapes and finite values
- [ ] Input layout and preprocessing are documented and identical in Python and C inputs
- [ ] Buffer lengths, output capacity, ownership, and error behavior are documented and checked
- [ ] `cinder run` works
- [ ] `cinder info` works
- [ ] `cinder bench` works
- [ ] Unit tests pass
- [ ] macOS and Linux CI build and test successfully
- [ ] Sanitizer and supported leak checks report no findings in the exercised test suite
- [ ] Loader rejects malformed structures, excessive sizes, and incomplete data without leaking memory
- [ ] Benchmarks document their environment and measure inference separately from loading and preprocessing
- [ ] Architecture is documented
- [ ] Model format is documented
- [ ] README is complete
- [ ] GitHub `v0.1.0` release exists

Once these boxes are checked:

**Ship it.**

Do not delay the release because of features that belong in future versions.

---

# 17. Post-MVP Roadmap

A possible progression:

```text
v0.1
Basic MLP inference
        ↓
v0.2
Additional operators
        ↓
v0.3
INT8 quantization
        ↓
v0.4
SIMD / NEON / AVX
        ↓
v0.5
CNN inference
        ↓
v0.6
ONNX subset importer
        ↓
v0.7
Multithreading
        ↓
v1.0
Stable runtime and public API
```

There is no requirement to implement all of these.

Cinder should evolve according to what is interesting and educational.

---

# 18. Strong Post-MVP Directions

## Quantization

Add:

```text
FP32
 ↓
INT8
```

Then compare:

```text
model size
memory consumption
latency
accuracy
```

This turns Cinder into a useful systems experiment.

---

## SIMD

Investigate accelerating the core matrix operations.

Possible progression:

```text
naive scalar C
       ↓
loop optimizations
       ↓
ARM NEON
```

Since development is being done on Apple Silicon, ARM NEON would be a particularly interesting direction.

Benchmark every implementation.

---

## Memory Planning

v0.1.0 already sizes and reuses two working buffers for sequential execution. A post-MVP extension could support richer execution patterns by calculating workspace lifetimes and sharing storage across more intermediates.

Consider a reusable arena/workspace only when new model structures justify it.

Conceptually:

```text
Model load
   ↓
determine largest intermediate tensor
   ↓
allocate workspace
   ↓
reuse during inference
```

This moves Cinder closer to how serious inference runtimes approach memory management.

---

# 19. Using AI/Codex While Building Cinder

AI assistance is encouraged, but the core runtime should remain understandable.

You should personally understand:

```text
tensor representation
memory layout
matrix multiplication
Dense forward pass
ReLU
Softmax
binary serialization
binary deserialization
memory ownership
runtime execution loop
```

You do not need to manually type every character.

But you should be able to explain the code.

Use Codex heavily for:

```text
CMake
CI
tests
sanitizer setup
CLI parsing
error-handling boilerplate
documentation
refactoring
code review
cross-platform fixes
```

Use Codex collaboratively for the core runtime.

Avoid prompts equivalent to:

```text
Build an entire neural-network inference engine for me.
```

Instead, work feature by feature.

---

# 20. What Cinder Should Teach

The real objective of the project is not simply another GitHub repository.

By the end of October, you should understand this pipeline:

```text
trained neural network
        ↓
model serialization
        ↓
.cnd file
        ↓
binary bytes
        ↓
C structs
        ↓
weights loaded into memory
        ↓
input tensor
        ↓
matrix multiplication
        ↓
bias
        ↓
activation
        ↓
next layer
        ↓
logits
        ↓
softmax
        ↓
prediction
```

You should be able to explain what happens at every stage.

---

# 21. October Schedule Summary

```text
Oct 4–5
Project setup
        ↓
Oct 6–9
Tensor + math
        ↓
Oct 10–14
Layers + runtime
        ↓
Oct 15–18
Cinder model format + loader
        ↓
Oct 19–22
MNIST inference
        ↓
Oct 23–25
Benchmarking + optimization
        ↓
OCTOBER 25
FEATURE FREEZE
        ↓
Oct 26–28
Testing + sanitizers
        ↓
Oct 29–31
Documentation + release
        ↓
CINDER v0.1.0
```

Follow these milestones in order even if individual daily targets shift. The first implementation step is M0: CMake, a version command, a smoke test, CI, and sanitizer configuration. Then build tensor allocation and matrix-vector multiplication collaboratively, with the core loops understood before moving on.

The **October 25 deadline is the internal deadline**.

October 31 is the actual release deadline.

That gives Cinder almost a full week of buffer for bugs, life interruptions, documentation, and unexpected problems.

---

# Final Objective

By the end of the project, you should be able to say:

> **I built a neural-network inference runtime from scratch in C. It defines its own binary model format, loads pretrained neural networks, manages their parameters in memory, executes forward inference using custom tensor operations, benchmarks runtime performance, and produces outputs validated against a reference implementation.**

That is Cinder `v0.1.0`.
