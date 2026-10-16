# Common layer interface contract

The types and declarations are in [`include/cinder/layer.h`](../include/cinder/layer.h).
**This step defines the interface only.** The common constructors, validator,
destructor, and dispatcher are not implemented or linked yet. Existing Dense,
ReLU, and Softmax kernels remain implemented and tested. Implementing their
common wrappers and dispatch is the next scoped step.

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

How a future model acquires, copies, or transfers ownership of layers will be
decided in the sequential-model contract. This interface does not silently
transfer ownership when a layer is inspected or executed.

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

After validating each layer, model connection validation can compare:

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

The model can also use these fields to determine its external input/output
sizes and the largest working vector. These are requirements for future model
validation and buffer planning, not an implemented model API.

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

The implementation will use a `switch (layer->type)` with these paths:

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

## Future usage example

This example shows the declared API. It cannot link until the common layer
functions are implemented:

```c
#include "cinder/layer.h"

int layer_example(void)
{
    const float weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                             0.5f, -0.5f, 3.0f, 1.0f};
    const float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    const float input[] = {2.0f, -1.0f};
    float hidden[4];
    float output[4];
    cinder_layer_t *dense = NULL;
    cinder_layer_t *relu = NULL;
    cinder_error_t error;
    cinder_status_t status =
        cinder_layer_create_dense(2, 4, weights, 8, bias, 4, &dense, &error);
    if (status == CINDER_OK) {
        status = cinder_layer_create_activation(CINDER_LAYER_RELU, 4, &relu, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_validate(dense, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_validate(relu, &error);
    }
    if (status == CINDER_OK && dense->output_size != relu->input_size) {
        /* A future model constructor will report connection mismatches. */
        cinder_layer_free(relu);
        cinder_layer_free(dense);
        return 1;
    }
    if (status == CINDER_OK) {
        status = cinder_layer_forward(dense, input, 2, hidden, 4, &error);
    }
    if (status == CINDER_OK) {
        status = cinder_layer_forward(relu, hidden, 4, output, 4, &error);
        /* On success: output = {0.5f, 0.0f, 3.5f, 5.0f}; hidden is unchanged. */
    }
    cinder_layer_free(relu);
    cinder_layer_free(dense);
    return status == CINDER_OK ? 0 : 1;
}
```

## Verification and next step

This change adds declarations and documentation only. On October 16, 2026,
the standalone header, repeated inclusion with existing headers, and every C
block in this document (including the complete future usage example) passed
C11 syntax checks with `-Wall -Wextra -Wpedantic -Werror`. The compiler used
`-fsyntax-only`, producing no object or executable and performing no linking
or execution. This does not establish runtime behavior for unimplemented
functions. Runtime sources, build targets, and CTest tests are unchanged.

Next, implement common construction/cleanup, metadata validation, and switch
dispatch for Dense, ReLU, and Softmax. Test valid and invalid tags, payload and
dimension consistency, preserved activation lengths, caller-buffer bounds,
validation before copying, kernel error propagation, source preservation,
saved NumPy references, and repeated execution without allocation. Force every
constructor allocation failure, including the wrapper allocation after Dense
succeeds, and check cleanup with sanitizers and supported leak detection.
