# Common layer interface contract

The types and declarations are in [`include/cinder/layer.h`](../include/cinder/layer.h).
The constructors, metadata validator, destructor, and forward dispatcher are
implemented in [`src/layer.c`](../src/layer.c) and linked into `cinder_core`.
Dense, ReLU, and Softmax share this API and reuse their existing tested kernels.
The [sequential-model contract](MODEL.md) defines how models adopt and connect
these layers. Model creation, validation, buffers, and cleanup are implemented;
sequential inference is next.

## Type tag, dimensions, and payload

```c
typedef enum {
    CINDER_LAYER_INVALID = 0,
    CINDER_LAYER_DENSE,
    CINDER_LAYER_RELU,
    CINDER_LAYER_SOFTMAX
} cinder_layer_type_t;

typedef struct {
    cinder_layer_type_t type;
    size_t input_size;
    size_t output_size;
    cinder_dense_t *dense;
} cinder_layer_t;
```

The type tag identifies which operation to execute and how to interpret the
payload. An all-zero descriptor has the invalid tag and cannot execute.
Unknown tags are also rejected, rather than treated as an identity operation.
These enum values describe runtime objects; they do not define future `.cnd`
file IDs, enum widths, or a struct layout to write directly to disk.

Both dimension fields count float elements, not bytes. They must be positive,
and each vector's byte extent must fit in `size_t`.

| Type | Dimensions | Type-specific data | Existing operation |
| --- | --- | --- | --- |
| Dense | Input and output may differ, or be equal. | Non-NULL owned `cinder_dense_t *`; common sizes must equal its sizes. | `cinder_dense_forward` |
| ReLU | `input_size == output_size` | `dense == NULL`; no parameters. | `cinder_tensor_relu` |
| Softmax | `input_size == output_size` | `dense == NULL`; no parameters. | `cinder_tensor_softmax` |

Dense's tensor shapes remain `[output_size][input_size]` for weights and
`[output_size][1]` for bias. The common wrapper does not store another copy of
the parameter arrays; its Dense payload owns those tensors.

A single typed payload pointer is enough for these three layer kinds because
only Dense has type-specific data. A union with a single useful member would
add no capability. ReLU and Softmax need only their type and dimensions. There
is no function-pointer table, `void *` payload, or separate activation object.

The repeated dimensions on the Dense wrapper let model validation read the
same fields for every layer type. They are fixed at construction and validated
against the Dense payload, so the two representations cannot legitimately
disagree. Callers inspect them directly; extra dimension getter functions are
unnecessary for this public representation.

## Ownership and construction

The caller owns a successfully returned `cinder_layer_t`. A Dense wrapper
exclusively owns its Dense object, which owns its tensors and arrays. An
activation wrapper owns only itself. All fields and nested parameters are
read-only to callers after construction. Do not copy an owning wrapper by
value, replace its payload, free the payload directly, or share ownership of
one Dense object between wrappers. Borrowed pointers expire when the layer is
freed. Concurrent calls require separate output/error storage and unchanged,
live sources and parameters.

The declarations provide two construction paths:

```c
cinder_status_t cinder_layer_create_dense(
    size_t input_size,
    size_t output_size,
    const float *weights,
    size_t weight_count,
    const float *bias,
    size_t bias_count,
    cinder_layer_t **out_layer,
    cinder_error_t *error
);

cinder_status_t cinder_layer_create_activation(
    cinder_layer_type_t type,
    size_t size,
    cinder_layer_t **out_layer,
    cinder_error_t *error
);
```

Both reject a NULL output slot with `INVALID_ARGUMENT`. Otherwise they set
`*out_layer = NULL` before other validation. The slot must not already own a
live object. The optional error object and output slot must be separate; for
Dense, neither may overlap either source array. The read-only source arrays
may overlap one another, as allowed by the [Dense contract](DENSE.md).

`cinder_layer_create_dense` delegates validation and copying to
`cinder_dense_create`, preserving its argument/overflow/count/finiteness order
and errors. It then allocates the common wrapper and derives its dimension
fields from the completed Dense object. If that last allocation fails, it
frees the new Dense and returns `ALLOCATION`. Only a fully constructed wrapper
is published. This means six allocation points in total: the existing five
Dense allocations followed by one wrapper allocation. No existing Dense
object is passed in or has its ownership transferred. The caller's parameter
arrays remain owned by the caller and unchanged on every outcome.

`cinder_layer_create_activation` accepts only `CINDER_LAYER_RELU` or
`CINDER_LAYER_SOFTMAX`. Other tags, including Dense and INVALID, or size zero
return `INVALID_ARGUMENT`. Then `size > SIZE_MAX / sizeof(float)` returns
`OVERFLOW`. After validation it allocates one wrapper, sets both dimensions to
`size`, and sets `dense = NULL`. Allocation failure returns `ALLOCATION`.
Neither activation needs a parameter array or an execution buffer of its own.

Every constructor failure leaves the output NULL and frees all partial
allocations. `cinder_layer_free(NULL)` is harmless. For an intact live owner,
free dispatch releases its Dense payload only for Dense, then releases the
wrapper. It allocates nothing and does not reset the caller's pointer. Free
each layer exactly once; a stack descriptor used for metadata inspection is
not an owning layer that can be passed to free or forward.

The [model contract](MODEL.md#ownership-transfer-and-construction) specifies
transfer only when model construction succeeds: the model adopts the layers
and clears the caller's array slots. Failure leaves the layers caller-owned.
This ownership transfer is implemented. Layer inspection and execution do not
transfer ownership; after adoption, the model releases the layers during its
own cleanup. Execution of adopted layers will be handled by `cinder_model_run`,
which remains unimplemented.

## Validation visible to the model

```c
cinder_status_t cinder_layer_validate(
    const cinder_layer_t *layer,
    cinder_error_t *error
);
```

The validator checks metadata only, without allocation, mutation, or scanning
weights, bias, or input values. A non-NULL descriptor must be readable and
fully initialized. Any non-NULL Dense payload must point to a live, intact
Dense object; metadata validation cannot establish pointer validity. Success
does not manufacture an owner or make a manually assembled descriptor safe
to free or execute.

Checks run in this order:

1. NULL descriptor, INVALID tag, or unknown tag: `INVALID_ARGUMENT`.
2. Either dimension is zero: `INVALID_ARGUMENT`.
3. Either vector length exceeds `SIZE_MAX / sizeof(float)`: `OVERFLOW`.
4. Dense lacks a payload, or an activation has one: `INVALID_ARGUMENT`.
5. Dense's common dimensions disagree with its payload, or activation input
   and output lengths differ: `DIMENSION_MISMATCH`.

Optional error storage must be separate from the descriptor and owned data.
Dense's own tensor layout, finite parameters, and intact metadata remain the
invariants established by its constructor. Forward execution checks finite
source values again; metadata validation is not a substitute for that check.

After validating each layer, model construction compares:

```text
previous_layer.output_size == next_layer.input_size
```

For the planned tiny network:

| Position | Layer | Input size | Output size |
| --- | --- | --- | --- |
| 0 | Dense | 2 | 4 |
| 1 | ReLU | 4 | 4 |
| 2 | Dense | 4 | 2 |
| 3 | Softmax | 2 | 2 |

Every adjacent pair matches. A ReLU described as `4 -> 2` is invalid on its
own because it must preserve length. A valid ReLU `3 -> 3` after Dense `2 -> 4`
fails the connection check instead: the preceding layer produces four values,
but ReLU expects three. Dense `4 -> 4` is valid too; Dense can change length
but is not required to change it.

Model construction also uses these fields to determine its external input/output
sizes and the largest working vector, then allocates two reusable buffers.
These construction checks are implemented; model execution is the next step.

## Common execution and switch dispatch

```c
cinder_status_t cinder_layer_forward(
    const cinder_layer_t *layer,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);
```

Execution accepts one input vector and a separate writable output buffer for
every layer kind. The input length must exactly match `input_size`; output
capacity must be at least `output_size`. Output need not be initialized. The
layer neither allocates/frees storage nor retains caller buffers during a call.

Validation order is: reject NULL layer/input/output with `INVALID_ARGUMENT`;
validate layer metadata; reject a wrong input count with `DIMENSION_MISMATCH`;
reject insufficient output capacity with `BUFFER_TOO_SMALL`; validate sources.
Zero input count is a length mismatch for a valid layer; zero output capacity
is insufficient space. Unknown tags must fail before any output write.

The implementation uses a `switch (layer->type)` with these paths:

| Case | Execution path |
| --- | --- |
| `CINDER_LAYER_DENSE` | Delegate to `cinder_dense_forward`, which validates finite input/weights/bias before computing `Wx + b`. |
| `CINDER_LAYER_RELU` | Check every input value is finite, copy exactly `input_size` floats to output, then call `cinder_tensor_relu(output, output_size, error)`. |
| `CINDER_LAYER_SOFTMAX` | Check every input value is finite, copy exactly `input_size` floats to output, then call `cinder_tensor_softmax(output, output_size, error)`. |
| INVALID / default | Return `INVALID_ARGUMENT` with error details; never fall through to another operation. |

Activations use the output itself as the existing kernel's mutable vector.
This adds a linear copy but no temporary allocation. It preserves the input
and gives the future model the same separate-buffer call shape for every type.
The existing tensor activation functions remain available for direct in-place
use. Activation input validation must happen before the copy: copying a bad
input and then relying only on the kernel's rejection would violate output
preservation on validation failure.

For example, a three-value layer receiving `{-1, 0, 1}` produces `{0, 0, 1}`
for ReLU or approximately `{0.09003057, 0.24472847, 0.66524096}` for Softmax.
Both report `3 -> 3`; neither owns weights or bias. With an output capacity of
five, only indices 0 through 2 participate. Indices 3 and 4 are never read or
written, even if they contain NaN. A NaN in the input, including its last
element, is rejected before any output element changes.

The copy and activation each take linear time, so activation forward execution
is `O(size)` time and `O(1)` extra space. The kernel still performs its own
validation, preserving its standalone API guarantees. The wrapper's earlier
scan is necessary to preserve the separate output on validation failure.

## Overlap, failures, and error details

For all layer types, the full advertised output range must be disjoint from
the input range, the wrapper and all Dense-owned storage, and optional error
storage. Exact input/output aliasing is also forbidden for activations through
this interface. Disjoint slices of a single allocation are allowed. Read-only
input may borrow Dense parameter data. Error storage must be disjoint from all
sources and owned objects. Accessible sizes and overlap are caller
preconditions, not runtime checks; callers must not change or free data while
a call uses it.

Success writes exactly `output_size` floats. Spare capacity, input, and owned
parameters stay unchanged on every outcome. Pointer, metadata, length,
capacity, and finite-source validation failures preserve the entire output.
After computation begins, Dense arithmetic failures retain its documented
partial results; Softmax numeric failure may leave copied inputs, exponentials,
or partial probabilities. Discard the result on arithmetic failure. ReLU has
no numeric failure after valid finite input. A failed call does not invalidate
the layer; a later call can reuse it with a different input or output buffer.

Propagate existing kernel statuses and error messages. All status-returning
functions use [`error.h`](../include/cinder/error.h): optional caller-owned
details, matching status codes, an empty message on success, and a bounded,
null-terminated failure message. No new error code, diagnostic allocation,
printing, or process termination is needed.

## Usage example

This example manually executes Dense, ReLU, and Softmax using caller-owned
buffers. A sequential model will automate connection checks and buffer reuse
in a later step:

```c
#include "cinder/layer.h"

int layer_example(void)
{
    const float weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                             0.5f, -0.5f, 3.0f, 1.0f};
    const float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    const float input[] = {2.0f, -1.0f};
    float hidden[4];
    float activated[4];
    float output[6] = {0, 0, 0, 0, 99, 100};
    cinder_layer_t *dense = NULL;
    cinder_layer_t *relu = NULL;
    cinder_layer_t *softmax = NULL;
    cinder_error_t error;
    cinder_status_t status =
        cinder_layer_create_dense(2, 4, weights, 8, bias, 4, &dense, &error);
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_RELU, 4, &relu, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_SOFTMAX, 4, &softmax, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_validate(dense, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_validate(relu, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_validate(softmax, &error);
    }
    if (status == CINDER_OK && (dense->output_size != relu->input_size ||
                               relu->output_size != softmax->input_size)) {
        /* The model constructor reports these connection mismatches automatically. */
        cinder_layer_free(softmax);
        cinder_layer_free(relu);
        cinder_layer_free(dense);
        return 1;
    }
    if (status == CINDER_OK) {
        status = cinder_layer_forward(dense, input, 2, hidden, 4, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_forward(relu, hidden, 4, activated, 4, &error);
        /* On success: activated = {0.5f, 0.0f, 3.5f, 5.0f}; hidden is unchanged. */
    }
    if (status == CINDER_OK) {
        status = cinder_layer_forward(softmax, activated, 4, output, 6, &error);
        /* First four values are probabilities; output[4..5] stay 99 and 100. */
    }
    cinder_layer_free(softmax);
    cinder_layer_free(relu);
    cinder_layer_free(dense);
    return status == CINDER_OK ? 0 : 1;
}
```

## Verification and next step

The interface was first defined and syntax-checked on October 16, 2026.
The runtime implementation was verified locally on macOS on October 21, 2026:

- `layer_interface` links the production library. It checks supported and
  invalid tags, positive/byte-representable sizes, payload/dimension consistency,
  constructor ownership, correct dispatch, and repeated execution with different
  inputs. Metadata faults in tests are restored before cleanup; applications
  must keep constructed objects intact.
- For both activations, tests reject NULL pointers, wrong input counts, small
  output buffers, and NaN/positive infinity/negative infinity at every position.
  They verify the entire output is unchanged on validation failure, sources and
  dimensions remain unchanged, guards and spare capacity are untouched, and
  disjoint slices of one allocation are accepted. Signed zero and subnormal
  ReLU values retain the tensor behavior.
- Saved NumPy references check 12 activation fixtures through each activation
  type plus 11 Dense fixtures: 35 result vectors. ReLU comparisons are exact;
  Dense and Softmax use the existing `atol = rtol = 1e-5` rule. Softmax outputs
  must also be probabilities with a sum within `1e-6` of one. The standalone
  tensor tests retain tighter tiny-probability and extreme-input checks.
- A Dense arithmetic-overflow case checks that dispatch returns the underlying
  error message and partial-output behavior, leaves spare capacity alone, and
  permits a successful subsequent call. Success clears prior diagnostics;
  the API is also tested with NULL optional error storage.
- `layer_allocation_failures` compiles the real layer/Dense/tensor sources with
  allocator substitutions confined to the test. It forces all six Dense
  wrapper allocation failures and the single allocation failure for each
  activation, tracks every allocation/free, and verifies complete cleanup and
  recovery. Invalid construction attempts allocate nothing. A failed Dense
  construction leaves an already-live activation usable.
- After construction, allocation tracking verifies that validation, successful
  and failing forward calls, repeated calls, and destruction make no new
  allocations. Forward and validation make no frees either. No allocator hooks
  were added to production code.
- Debug passed all 14 CTest tests. Release and combined ASan/UBSan builds with
  generated tensor NumPy references enabled each passed all 15 tests. The two
  new test executables reported zero leaks with native macOS `leaks`.
- The build with tests disabled succeeded and registered zero tests. The
  usage example compiled with C11 warnings as errors, linked against that
  library, and ran successfully.

These are local results; hosted macOS/Linux CI for this change has not been
verified. Existing CI jobs automatically include the new CTest registrations.

The [sequential-model contract](MODEL.md) is now defined, including ownership,
connection checks, external dimensions, output preservation on failure, and
serialized access to shared working buffers. Model construction, connection
validation, size accessors, cleanup, and both reusable buffers are implemented.
Sequential inference is the next scoped step.
