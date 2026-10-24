# Sequential-model contract

The public declarations are in [`include/cinder/model.h`](../include/cinder/model.h).
Construction, connection validation, size accessors, inference, and cleanup are implemented
in [`src/model.c`](../src/model.c) and linked into `cinder_core`. Each model owns
its ordered layers and two reusable working buffers. `cinder_model_run` executes
through the existing common layer operations without allocating, and copies the
final result to caller output only after every layer succeeds.

## Representation and public API

A model executes a fixed, ordered, nonempty collection of layers on a single
float32 vector. It calls each layer once, in the order supplied at construction.
It does not train, reorder layers, add activations, batch samples, or support
branches. A model may contain one layer, only activations, or a mixture of Dense,
ReLU, and Softmax. It need not end with Softmax; the last layer determines what
the output means.

```c
#include "cinder/layer.h"

typedef struct cinder_model cinder_model_t;

cinder_status_t cinder_model_create(
    cinder_layer_t **layers,
    size_t layer_count,
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

`cinder_model_t` is an opaque type: callers can hold a pointer but cannot inspect
its layout, allocate it by value, or change its fields. The implementation
stores the ordered layer pointers, layer count, external input/output sizes,
working-buffer capacity, and two working-buffer pointers. Keeping those fields
private prevents callers from invalidating connections or replacing buffers
after construction. No public append/remove/reorder or workspace access API is
included in this contract.

The two size accessors return counts of floats, not bytes. A valid model's input
size is the first layer's input size; its output size is the last layer's output
size. Both are positive and fixed. A NULL model returns zero from either
accessor, so zero is an invalid-model sentinel, not a valid empty model. Other
non-NULL pointers must identify live models; getters cannot diagnose dangling
or invented pointers. Accessors allocate nothing and do not use error storage.

Phase 3's [`.cnd` version 1 format contract](MODEL_FORMAT.md) defines the file
layout, validation rules, and resource limits. `cinder_model_load` is implemented
and specified in [the loader guide](LOADER.md). It constructs layers, closes the
validated file, and uses this same model-construction path before publishing
ownership.

## Ownership transfer and construction

The caller first creates layers using the [common layer constructors](LAYER.md).
It passes them in a writable array in execution order. Each entry must be a live,
intact, independently owned layer. Stack descriptors and layers already owned
by another model cannot be transferred. Distinct wrappers must not share owned
Dense payloads or parameter allocations. Construction rejects duplicate layer
pointers, but cannot prove general pointer validity or exclusive ownership.
Those remain caller preconditions.

The model **takes ownership only on successful construction**. It allocates its
own pointer array and copies the pointers in order. It takes the existing layer
objects and their already-owned parameters; it does not copy their parameter
arrays again. This avoids a second set of Dense parameter copies.

| Outcome | Caller array | Layers and their parameters | Model output slot |
| --- | --- | --- | --- |
| Success | Every supplied slot becomes NULL; array storage stays caller-owned. | Exclusively owned by the new model. | Points to the complete model. |
| Any failure | Every slot is unchanged. | Still caller-owned and unchanged; none are freed. | NULL, if an output slot was supplied. |

The source array itself may be a stack array or a caller allocation. The model
never retains its address. On success, the caller may reuse or release that
array immediately. Other aliases to adopted layers are borrowed, read-only
references: do not independently execute, mutate, free, or transfer those
layers again. All such aliases expire when the model is freed. Clearing the
supplied slots does not find or clear aliases elsewhere in the program.

The `out_model` slot must not already own a live model. The source pointer array,
output slot, optional error object, and layer-owned storage must be disjoint.
The caller must not change or free layers or the array during construction.

Construction follows this order:

1. If `out_model == NULL`, return `CINDER_ERROR_INVALID_ARGUMENT`. Otherwise set
   `*out_model = NULL` before any other validation.
2. Reject `layers == NULL` or `layer_count == 0` with `INVALID_ARGUMENT`.
3. Reject `layer_count > SIZE_MAX / sizeof(cinder_layer_t *)` with `OVERFLOW`
   before traversing the array or calculating its allocation size.
4. Visit indices from zero upward. At each index, call `cinder_layer_validate`
   and return its first failure unchanged. Then compare that pointer with earlier
   entries and reject a duplicate with `INVALID_ARGUMENT`. NULL entries fail
   validation. Metadata checks do not rescan all Dense parameter values; their
   validity and immutability come from the layer constructors and ownership rules.
5. Once every entry passes, check adjacent connections in order. Return
   `DIMENSION_MISMATCH` at the first unequal output/input pair.
6. Derive external dimensions and the largest input or output size among all
   layers. Layer validation has already proved that each vector's byte extent
   fits in `size_t`. Never multiply unchecked sizes.
7. Allocate, in order, the model descriptor, private pointer array, working
   buffer A, and working buffer B. Each buffer is a separate allocation of
   `capacity * sizeof(float)` bytes. Any allocation failure returns `ALLOCATION`
   and releases only these new allocations; all supplied layers remain owned
   by the caller.
8. After every allocation succeeds, copy the pointer list, clear the source
   slots, and publish the model. Nothing that can fail remains during this
   ownership transfer.

No allocation occurs until validation finishes. Duplicate detection may use a
simple pairwise scan: quadratic in the layer count, without a temporary set.
The small sequential runtime does not need a more complex ownership registry.
Do not call a destructor that frees adopted layers on a partially constructed
model before ownership has transferred.

The implementation initially sets `layer_count` to zero and every owned pointer
to NULL. That lets `cinder_model_free` clean up each partial allocation state
without touching any caller-owned layer or reading uninitialized pointer-array
entries. Only after all four allocations succeed does construction copy the
ordered pointers, clear source slots, set the adopted layer count, and publish
`*out_model`. Dense parameter arrays are neither copied again nor modified.

The ownership chain after success is:

```text
caller owns model handle
  model owns private layer-pointer array and both working buffers
  model owns each common layer wrapper
    Dense wrapper owns Dense object
      Dense object owns weight/bias tensors and their arrays
```

The caller continues to own the input and destination output arrays. Model
construction does not receive or allocate those arrays. The run call borrows
them only for the duration of execution.

An empty list is an error, not an identity operation. This gives every valid
model well-defined positive external dimensions. It also rejects accidentally
omitted layers instead of producing an apparently successful empty network.

## Connections and dimensions

Every adjacent pair must satisfy:

```text
layers[i - 1].output_size == layers[i].input_size
```

For the planned tiny network:

| Index | Layer | Input size | Output size |
| --- | --- | --- | --- |
| 0 | Dense | 2 | 4 |
| 1 | ReLU | 4 | 4 |
| 2 | Dense | 4 | 2 |
| 3 | Softmax | 2 | 2 |

The model exposes two inputs and two outputs. All three connections match.
Replacing the ReLU with a valid `3 -> 3` ReLU causes model construction to fail
because the first Dense produces four values. A ReLU described as `4 -> 2`
fails the layer's own metadata validation even before connection checking.

There is no truncation, padding, broadcasting, implicit reshape, or automatic
repair of a mismatch. A malformed network never becomes an executable model.
The topology stays fixed afterward, so inference uses the constructor's
validated connections without allocating or rebuilding them on every call.

## Working buffers and execution

Each model owns two distinct reusable float buffers. Their common capacity is:

```text
capacity = max(all layer input sizes and all layer output sizes)
```

This includes external input/output sizes and every intermediate size. For the
`2 -> 4 -> 2` example, both buffers have capacity four. Two buffers are allocated
even for a one-layer model, keeping construction and execution uniform.
Individual allocations are checked separately; no unchecked `2 * capacity`
calculation is needed. These buffers are additional to the owned parameters.

Both buffer allocation and execution are implemented. The buffers are private;
there is no public workspace accessor or requirement that their initial contents
be zero.

After validating the call, copy the caller's logical input into buffer A. This
also protects caller input when the first layer is an activation. Each Dense
reads the current buffer and writes the other one; each ReLU or Softmax updates
the current buffer in place. All operations use `cinder_layer_forward`, whose
activation contract now permits exact input/output aliasing and skips the copy
in that case. Dense still requires separate buffers; partial overlap is never
allowed. The model's caller input/output ranges must still be disjoint.

The model tracks the current buffer index and logical vector length separately.
It passes that logical length and the full working-buffer capacity to each
layer, then updates the current length to the successful layer's output size.
Never pass spare elements as part of a logical input.

| Operation in the example | Reads | Writes | Result count |
| --- | --- | --- | --- |
| Copy validated input | Caller input | A | 2 |
| Dense `2 -> 4` | A | B | 4 |
| ReLU `4 -> 4` | B | B | 4 |
| Dense `4 -> 2` | B | A | 2 |
| Softmax `2 -> 2` | A | A | 2 |
| Publish successful result | A | Caller output | 2 |

The last result may be in either buffer, determined by the number of Dense
layers rather than the total layer count. A single Dense finishes in B; a single
activation finishes in A. The same loop handles either case and activation-only
models. No activation needs an extra copy or a buffer switch.

Working-buffer contents need not be initialized or cleared between calls.
The initial copy and each successful layer supply the complete logical input for
the following operation. In the example, the second Dense leaves old data beyond
A's first two elements; Softmax receives a count of two and never reads those old elements.
If a layer fails, execution stops immediately instead of consuming its partial
result. The next run starts again at layer zero, so no explicit reset is needed.

`cinder_model_run` takes a non-const model pointer because it modifies these
private buffers. Layer topology and parameters remain unchanged. There are no
allocations or frees during execution, including returned-error paths. The
model retains no caller input, output, or error pointer after the call.

## Inference validation and failure behavior

The caller supplies one vector containing exactly the model input size and an
output buffer with at least the model output size. Output need not be initialized.
Counts are float elements; actual readable/writable extents are the caller's
responsibility.

Checks and execution follow this order:

1. NULL model, input, or output: `INVALID_ARGUMENT`.
2. Wrong input count, including zero: `DIMENSION_MISMATCH`.
3. Insufficient output capacity, including zero: `BUFFER_TOO_SMALL`.
4. Any NaN or infinity in the logical input: `NONFINITE`.
5. Copy input into private storage and execute layers in order through
   `cinder_layer_forward`. Stop at the first
   failure and return the existing status and diagnostic message unchanged.
6. Only after all layers succeed, copy exactly the model output size to caller
   output and return `CINDER_OK` with cleared error details.

**Every returned failure preserves the entire caller output**, including
arithmetic failure after earlier layers succeeded. This is stronger than the
individual layer contract, which permits partial arithmetic results: those
partial results stay in private buffers and are never published. Their contents
are unspecified after failure and inaccessible to callers. This output guarantee
applies to valid calls satisfying storage and lifetime preconditions; invalid
pointers or unsynchronized calls cannot be made safe by a status code.

For example, a two-output model given output `{91, 92, 93, 94}` either succeeds
with `{result0, result1, 93, 94}`, or returns an error leaving all four values
unchanged. The preserved values after failure are previous data, not a result
for the failed input; callers must still check the returned status.

Spare output capacity is untouched on every outcome. Input, layer parameters,
and immutable metadata are also unchanged. A failed run does not invalidate
the model. Subsequent runs may use different caller buffers and input values.
Results depend on the current input and fixed parameters, not prior calls.

The initial and final copies add `O(input_size + output_size)` work while
preserving caller input/output without another allocation. Working storage is
two vectors of the largest required size; inference work is the sum of the layer
operations plus validation/copying.
There is no global error state, printing, process termination, or new status
code. Optional caller-owned errors follow [`error.h`](../include/cinder/error.h):
matching status, an empty message on success, and a bounded null-terminated
message on failure.

## Storage, repeated calls, and concurrency

The full advertised input and output ranges must be disjoint; exact and partial
overlap are forbidden. Disjoint slices of the same caller allocation are allowed.
Both ranges and error storage must also be separate from one another and all
model-owned storage, including parameter arrays accessed through old aliases.
Overlap and true accessible extents are caller preconditions, not runtime
checks. The model API deliberately requires caller buffers separate from all
owned data, even where a lower-level layer could read borrowed parameter data.

| Situation | Contract |
| --- | --- |
| Repeated runs on one model | Allowed after the previous call returns, including after a failure; no reset. |
| Simultaneous or reentrant runs on one model | Forbidden, even with different input/output/error buffers. |
| Sharing one model between threads | Caller holds an external lock for each entire run and coordinates lifetime. |
| Parallel runs on independent models | Allowed with separately owned layers/workspaces, separate writable output/error storage, and unchanged readable inputs. |
| Size getters during a run | Allowed: they read immutable metadata, not working buffers. |
| Destruction while any call or borrowed access uses the model | Forbidden; caller must synchronize lifetime. |

The two working buffers are shared by all calls on an instance. Different
caller output buffers therefore do not make simultaneous calls on that instance
safe. There is no internal mutex, busy flag, `BUSY` status, or attempt to detect
concurrent entry. External serialization or independently constructed models
are the explicit coordination choices. Independent models may share read-only
caller input; their owned layers and parameter allocations are not shared.

The runtime does not create worker threads. Permission to call independent
instances from application threads does not introduce internal multithreading
or change the v0.1.0 execution scope.

## Cleanup and construction example

`cinder_model_free(NULL)` is harmless. Freeing a valid model releases every
adopted layer through `cinder_layer_free`, both working buffers, the private
pointer array, and the model descriptor. It allocates nothing. The caller's
model pointer is not reset; all borrowed layer and parameter pointers expire.
Free a model exactly once, after all calls and borrowed accesses finish.

This construction-only example creates the `2 -> 4 -> 2` sequence and can be
compiled, linked, and run with the current library:

```c
#include "cinder/model.h"

int model_creation_example(void)
{
    const float weights1[] = {1, 2, -1, 0, 0.5f, -0.5f, 3, 1};
    const float bias1[] = {0.5f, -1, 2, 0};
    const float weights2[] = {1, 0, -1, 0.5f, 0, 1, 0.5f, -1};
    const float bias2[] = {0, 0.5f};
    cinder_layer_t *layers[4] = {NULL, NULL, NULL, NULL};
    cinder_model_t *model = NULL;
    cinder_error_t error;

    cinder_status_t status = cinder_layer_create_dense(
        2, 4, weights1, 8, bias1, 4, &layers[0], &error);
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_RELU, 4, &layers[1], &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_create_dense(4, 2, weights2, 8, bias2, 2, &layers[2], &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_SOFTMAX, 2, &layers[3], &error);
    }
    if (status == CINDER_OK) {
        status = cinder_model_create(layers, 4, &model, &error);
    }
    int result = status == CINDER_OK && cinder_model_input_size(model) == 2 &&
                 cinder_model_output_size(model) == 2 ? 0 : 1;
    /* On success these slots are NULL and the model owns all four layers.
     * On failure any created layers still belong to the caller. */
    for (size_t i = 0; i < 4; ++i) {
        cinder_layer_free(layers[i]);
    }
    cinder_model_free(model);
    return result;
}
```

The model internally allocates two four-float buffers. Its external dimensions
remain two inputs and two outputs; buffer capacity is not an external dimension.

## Inference example

This example uses the implemented execution API and can link against `cinder_core`:

```c
#include "cinder/model.h"

int model_example(void)
{
    const float weights[] = {1, 2, -1, 0, 0.5f, -0.5f, 3, 1};
    const float bias[] = {0.5f, -1, 2, 0};
    const float input[] = {2, -1};
    float output[] = {91, 92, 93, 94, 95, 96};
    cinder_layer_t *layers[3] = {NULL, NULL, NULL};
    cinder_model_t *model = NULL;
    cinder_error_t error;

    cinder_status_t status = cinder_layer_create_dense(
        2, 4, weights, 8, bias, 4, &layers[0], &error);
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_RELU, 4, &layers[1], &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_SOFTMAX, 4, &layers[2], &error);
    }
    if (status == CINDER_OK) {
        status = cinder_model_create(layers, 3, &model, &error);
        /* Success clears all slots. Failure leaves the layers caller-owned. */
    }

    int result = 1;
    if (status == CINDER_OK && cinder_model_input_size(model) == 2 &&
        cinder_model_output_size(model) == 4) {
        status = cinder_model_run(model, input, 2, output, 6, &error);
        /* Success writes four probabilities; output[4..5] stay 95 and 96.
         * Any returned failure preserves all six original output values. */
        result = status == CINDER_OK ? 0 : 1;
    }

    /* Safe after partial layer creation, model failure, or successful transfer. */
    for (size_t i = 0; i < 3; ++i) {
        cinder_layer_free(layers[i]);
    }
    cinder_model_free(model);
    return result;
}
```

The shorter example uses Dense `2 -> 4`, ReLU, and Softmax to demonstrate
ownership and calls. The runtime tests also execute the full `2 -> 4 -> 2`
network against hand-calculated results. Independent generated NumPy references
now also verify every stage through the layer and model APIs; see
[complete-network verification](NUMPY_TESTS.md#complete-tiny-network).

## Verification and next scoped steps

The initial contract passed six C11 syntax checks on October 22, 2026, before
implementation. Construction, validation, accessors, and cleanup were then
implemented and verified locally on macOS on the same day:

- `model_allocation` links the production library and checks empty/NULL inputs,
  pointer-array overflow, invalid layer metadata with preserved diagnostics,
  duplicates, and early/middle/late connection mismatches. It verifies validation
  ordering, failure preservation, successful transfer with cleared slots,
  corrected-list retries, one-layer models, activation-only models, and the
  example sequence. Borrowed Dense parameters remain unchanged, the caller's
  source pointer array can be freed, and separate models have independent owners.
- `model_allocation_failures` compiles the real model/layer/Dense/tensor sources
  with test-only allocator substitutions. Every one of the four model allocation
  points is forced to fail, with and without optional error details. Each failure
  releases all partial model storage, leaves the original layers alive and
  executable, and permits either caller cleanup or a successful retry. Another
  live model and its buffers survive failed construction unchanged.
- The same test inspects private storage without adding a public workspace API.
  It verifies exact layer order and identity, a separately owned pointer list,
  two distinct writable buffers of the required byte size, and maxima at the
  input, output, or middle of a sequence. Both buffers have four-float capacity
  for the `2 -> 4 -> 2` example, and both exist even for a single-layer model.
  A largest-representable-vector test intercepts the buffer allocation before
  real malloc, verifying byte-count handling without requesting huge memory.
- Allocation tracking confirms invalid construction attempts allocate nothing,
  size getters allocate/free nothing, and cleanup makes no allocations. Every
  successful allocation is freed exactly once, including adopted Dense data.
  Reusing a cleared source slot cannot replace a model-owned layer.
- Debug passed all 16 CTest tests. Release and combined ASan/UBSan builds with
  generated tensor NumPy references enabled each passed all 17 tests. Both new
  model test executables reported zero leaks with native macOS `leaks`.
- A build with tests disabled succeeded and registered zero tests. The
  construction example compiled with C11 warnings as errors, linked against
  that library, and ran. At that construction-only step, the inference example
  was syntax-checked only.

Sequential inference was then implemented on October 22, 2026:

- `model_run` links the production library and checks each prefix of the example
  network against hand-calculated results, repeated different inputs, single
  layers, activation-only and activation-first models, consecutive Dense layers,
  widening/shrinking vectors, exact/spare output capacity, input/parameter
  preservation, validation order, and disjoint slices of one caller array.
- Real multiplication and bias overflow at the first, middle, or last layer
  preserves every caller output byte and the original kernel diagnostic.
  Immediate retries succeed without resetting the model; separately constructed
  instances remain independent when run alternately or one is destroyed.
- Extended `model_allocation_failures` observes the actual dispatch pointers and
  counts: Dense switches buffers, activations alias exactly, and shrinking
  vectors never consume poisoned spare tails. Test-only injected failures after
  partial writes at each of four layer positions verify immediate stopping and
  unchanged caller output, including failure at either activation.
- Allocation counters remain unchanged during success, preflight rejection,
  injected failure, real arithmetic failure, and recovery. The next allocation
  is armed to fail, proving inference does not depend on allocation success.
  Production code has no allocator or dispatch test hooks.
- `layer_interface` additionally checks in-place ReLU/Softmax, exact/spare
  capacity, non-finite values at every input position, validation preservation,
  and successful retry with cleared diagnostics.
- Debug passed 17/17 CTest tests. Release and combined ASan/UBSan builds with
  generated tensor NumPy references enabled passed 18/18 each. The model-run,
  model-storage, and layer test executables each reported zero native macOS leaks.
  The build with tests disabled succeeded and registered zero tests. Both model
  examples and the layer example compiled with C11 warnings as errors, linked
  against that library, and ran successfully.

These tests also passed hosted macOS/Linux CI at M2 closure, recorded below.

On October 23, the complete network was checked against deterministic float32
NumPy fixtures for ten inputs. Tests compare all four stages through a direct
layer sequence and sequential models ending at each stage, including the full
network. They use the same shape/finite checks and `atol = rtol = 1e-5` comparator
as the tensor reference tests. Debug passed 18/18 tests; Release and combined
ASan/UBSan with fresh references passed 20/20. The network executable reported
zero native leaks. Parameters, inputs, and expected stages are saved for the
future exporter/loader round trip. See [the reference guide](NUMPY_TESTS.md#complete-tiny-network)
for the cases, measured errors, and generation commands.

The October 23 [M2 lifecycle audit](M2_VERIFICATION.md) adds 4,120
success/failure/recovery cycles, concurrent independent instances, and externally
serialized shared-model calls. Poisoned private buffers must produce the same
results as baseline runs; allocation tracking now includes attempted `free(NULL)`
calls, and construction retries execute inference before destruction. Debug
passed 19/19 tests; Release and ASan/UBSan with fresh references passed 21/21.
Lifecycle and model-storage executables reported zero native leaks.

Hosted CI for `f5424a4`, including the NumPy and lifecycle tests, passed 19/19
tests on macOS, 21/21 on Linux with fresh references, and 19/19 with Linux
ASan/UBSan/leak detection in
[run 39984963408](https://github.com/farukumarx64/cinder/actions/runs/39984963408).
M2 closed on October 23, 2026. The report records exact evidence and limitations.
The [`.cnd` model-format contract](MODEL_FORMAT.md) is defined and
[Python validation and binary export](EXPORTER.md) are implemented, and the
[C loader](LOADER.md) is implemented with successful-load and basic malformed-file
tests. Broader parser/failure coverage and the full stage-by-stage file round trip
are next; MNIST remains later work.
