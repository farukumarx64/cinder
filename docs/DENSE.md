# Dense-layer contract

This is the first Phase 2 contract. The type and public function declarations
are in [`include/cinder/dense.h`](../include/cinder/dense.h). **Creation, cleanup,
and forward execution are specified but not implemented yet.** There is no
`src/dense.c` or Dense build target at this step. Including the header is valid;
calling these functions will require their implementations in subsequent steps.

The layer packages the existing float32 math `y = Wx + b` for one input vector.
It applies no activation. ReLU, Softmax, common layer dispatch, and sequential
model execution will be integrated separately.

## Representation and shapes

```c
typedef struct {
    size_t input_size;
    size_t output_size;
    cinder_tensor_t *weights;
    cinder_tensor_t *bias;
} cinder_dense_t;
```

| Field | Meaning and invariant |
| --- | --- |
| `input_size` | Positive number of input floats. |
| `output_size` | Positive number of output floats. |
| `weights` | Owned tensor with `rows == output_size` and `cols == input_size`. |
| `bias` | Owned column-vector tensor with `rows == output_size` and `cols == 1`. |

Both tensors use the existing [tensor representation](TENSOR.md). Each weight
row contains all connections to one output. The coefficient connecting input
`j` to output `i` is `weights->data[i * input_size + j]`. Bias element `i` is
`bias->data[i]`. Every stored parameter is finite. Signed zeros, subnormals,
and finite extreme values are allowed.

For a `2 -> 4` layer:

```text
input:     2 floats
weights:   4 rows x 2 columns = 8 floats
bias:      4 rows x 1 column  = 4 floats
output:    4 floats
```

Each output is calculated as:

```text
y[i] = sum(W[i, j] * x[j], j = 0 ... input_size - 1) + b[i]
```

Bias is mandatory, with one value per output. Use explicit zero biases when
the desired operation is just `Wx`. There is no broadcasting, transposition,
batch dimension, or implicit activation.

## Ownership and lifetime

The layer owns independent copies of the constructor's weight and bias arrays.
It owns both tensor structs and their element arrays. The caller owns the
returned layer and releases it with `cinder_dense_free`.

The source arrays remain caller-owned on success and failure. After successful
creation, modifying or releasing them cannot change the layer's parameters.
Creating two layers from the same arrays produces independently owned copies.
Sources must remain valid and unchanged while creation reads/copies them.

The public fields make the representation visible, consistent with the tensor
API's learning-oriented layout. Callers must treat **every layer field, nested
tensor field, and parameter element as read-only**. The C pointer types do not
enforce deep immutability; this is an API requirement. In particular, do not
call tensor fill/set/free on a layer's borrowed parameter tensors. Create a
new layer to use different parameters.

A non-NULL layer passed to this API must be a live object created by the API,
with all invariants intact. Do not construct one manually, copy it by value,
replace its pointers, or free its storage individually. The implementation
does not promise to diagnose forged objects, corrupted metadata, dangling
pointers, or incorrectly advertised buffer lengths.

`cinder_dense_free(NULL)` is a no-op. Free each live layer exactly once. Free
does not reset the caller's pointer, so set it to NULL if it remains in scope.
All aliases and borrowed parameter pointers become invalid after free.
Do not modify or free a layer while a call uses it. Concurrent access to
caller-owned mutable output or error storage requires caller coordination.

## Creation

```c
cinder_status_t cinder_dense_create(
    size_t input_size,
    size_t output_size,
    const float *weights,
    size_t weight_count,
    const float *bias,
    size_t bias_count,
    cinder_dense_t **out_layer,
    cinder_error_t *error
);
```

The weight source is a flat, row-major array. Counts measure floats, not bytes:
`weight_count` must equal `output_size * input_size`, and `bias_count` must
equal `output_size`. Counts must match exactly; spare source capacity is not
part of the parameter arrays. Both source pointers are required.

Validation and construction follow this order:

1. Reject a NULL `out_layer` with `INVALID_ARGUMENT`. Otherwise set
   `*out_layer = NULL` before validating any other argument. The output slot
   must not already own a live layer; overwriting such an owner would leak it.
2. Reject NULL weight/bias pointers or either zero dimension with
   `INVALID_ARGUMENT`.
3. Before multiplying dimensions, check
   `output_size > SIZE_MAX / input_size`. Then ensure the weight and bias
   element counts fit in `SIZE_MAX / sizeof(float)`. Return `OVERFLOW` if an
   element count or allocation byte count cannot be represented. Nonzero
   input size makes the weight-byte bound sufficient to cover the bias bound
   as well; both allocation extents must be safe.
4. Require exact weight and bias counts; otherwise `DIMENSION_MISMATCH`.
   No array element is read until size/count validation succeeds.
5. Check all supplied weights and biases for finiteness. NaN or either
   infinity returns `NONFINITE` before any allocation is attempted.
6. Allocate the layer and its parameter tensors, copy the parameters, and
   publish the completed layer through `*out_layer`. Allocation failure returns
   `ALLOCATION`, frees all partial allocations, and leaves `*out_layer` NULL.

Every failure with a non-NULL output slot leaves it NULL. Both source arrays
remain unchanged on every outcome. A constructor never takes ownership of
the caller's arrays, even temporarily. Creation uses time and owned storage
proportional to `output_size * input_size + output_size`.

## Forward execution

```c
cinder_status_t cinder_dense_forward(
    const cinder_dense_t *layer,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);
```

The caller supplies initialized readable input and writable output storage.
Output does not need initialization. The input must contain exactly
`layer->input_size` floats. Output capacity must be at least
`layer->output_size` floats; larger buffers are allowed. Only that many results
are written, and all spare output elements remain untouched on every outcome.

Validation order is:

1. NULL layer, input, or output: `INVALID_ARGUMENT`.
2. Input count different from `input_size`, including zero:
   `DIMENSION_MISMATCH`.
3. Output capacity smaller than `output_size`, including zero:
   `BUFFER_TOO_SMALL`.
4. Non-finite input, weight, or bias value: `NONFINITE`.

Check all sources, including bias, before writing any result. This prevents
the composed operation from writing `Wx` and only afterward discovering an
invalid bias. Pointer, length, capacity, and source-value validation failures
preserve the entire output buffer, assuming the caller's storage and overlap
preconditions hold.

After validation, use `cinder_tensor_matvec` and then
`cinder_tensor_add_bias`. Pass `output_size` as the bias operation's logical
output length, even when the caller supplied extra capacity. The numerical
contract follows those operations: accumulate each row in increasing column
order using float, then add bias using float. Finite underflow to a subnormal
or zero is accepted; exact rounding across toolchains is not required.
Reference comparisons use `atol = rtol = 1e-5` for the established small fixtures.

There are no allocations or retained input/output pointers during a forward
call. Time is `O(output_size * input_size)` and extra space is `O(1)`. The layer
has no mutable execution state; successful calls and failures leave its
parameters and the input unchanged. A failure does not invalidate the layer,
so a later call can try a different valid input or output buffer.

## Buffer overlap

Overlap rules are caller preconditions, not runtime checks or new status codes.
They apply to memory ranges, so different pointer values alone do not establish
that buffers are separate.

| Situation | Rule |
| --- | --- |
| Constructor's weight and bias source ranges overlap | Allowed: both are read-only and copied into independent owned storage. |
| Constructor's `out_layer` slot and optional error object | Must be separate from each other and both source ranges. |
| Forward input and output | No exact or partial overlap, including for a square layer. |
| Forward output and layer-owned storage | No overlap with the layer struct, tensor structs, or either parameter array. |
| Forward input and parameter data | May overlap when the full input range is valid; both are read-only. |
| Forward error object | Must not overlap input, output, the layer, or its parameter storage. |

For forward overlap rules, output means the full advertised `output_capacity`
range; input means the `input_count` range. Disjoint slices of one larger
allocation are allowed. Input/output overlap is forbidden because an early
output write could replace an input needed by a later weight row. Spare output
capacity is still never read or written. Buffer sizes and valid storage cannot
be discovered reliably from a C pointer and must be supplied correctly.

## Arithmetic failures and error details

Finite parameters and input can still produce a non-finite matvec running sum
or biased result. Such a failure returns `CINDER_ERROR_NONFINITE`.
**Discard the entire result after any arithmetic failure.** There is no
rollback or temporary allocation to make arithmetic failures atomic.

| Failure stage | Possible contents of the logical output range |
| --- | --- |
| Argument/source validation | Entire buffer unchanged. |
| Matrix-vector accumulation | Earlier rows contain completed, unbiased dot products; the failing row and later rows retain their pre-call contents. |
| Bias addition | Matvec has already written every row. Earlier elements contain `Wx + b`; the failing element and later elements still contain unbiased `Wx`. |

Unused capacity remains unchanged in all cases. The input and parameters are
never overwritten. Do not treat any of the partial values as a valid Dense
result, even when individual values look finite.

Both status-returning functions use the existing types from
[`error.h`](../include/cinder/error.h); no new status code is needed.
The optional error object belongs to the caller and may be NULL. On success,
its code is `CINDER_OK` and its message is empty. On failure, its code matches
the return status and its message is bounded and null-terminated. Functions
do not allocate error messages, print diagnostics, or terminate the process.

## Worked example and future usage

```text
W = [ 1.0   2.0 ]     x = [ 2.0 ]     b = [ 0.5 ]
    [-1.0   0.0 ]         [-1.0 ]         [-1.0 ]
    [ 0.5  -0.5 ]                        [ 2.0 ]
    [ 3.0   1.0 ]                        [ 0.0 ]

Wx      = [0.0, -2.0, 1.5, 5.0]
Wx + b  = [0.5, -3.0, 3.5, 5.0]
```

This example describes usage once creation, cleanup, and forward execution
are implemented; it cannot link against the current library yet:

```c
#include "cinder/dense.h"

int dense_example(void)
{
    const float weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                             0.5f, -0.5f, 3.0f, 1.0f};
    const float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    const float input[] = {2.0f, -1.0f};
    float output[6];
    cinder_dense_t *layer = NULL;
    cinder_error_t error;

    cinder_status_t status =
        cinder_dense_create(2, 4, weights, 8, bias, 4, &layer, &error);
    if (status == CINDER_OK) {
        status = cinder_dense_forward(layer, input, 2, output, 6, &error);
    }
    if (status == CINDER_OK) {
        /* output[0..3] = {0.5f, -3.0f, 3.5f, 5.0f}.
         * output[4..5] are untouched and remain uninitialized. */
    }
    cinder_dense_free(layer); /* Also safe when construction left it NULL. */
    return status == CINDER_OK ? 0 : 1;
}
```

The two dimensions describe the layer, the two constructor counts describe
the supplied parameter arrays, and the forward capacity describes writable
output storage. They are separate quantities even when some values coincide.

## Completion and next implementation steps

This contract step adds declarations and documentation only. On October 16,
2026, the standalone header, repeated inclusion with the existing headers, and
the complete example passed C11 syntax checks with `-Wall -Wextra -Wpedantic
-Werror`. The compiler used `-fsyntax-only`, so no object or executable was
produced and the declarations were not linked or executed. The worked example's
arithmetic was also checked. These checks do not establish runtime correctness;
Dense runtime tests will accompany the implementations.

Next, implement creation and cleanup. Their tests must cover dimensions,
overflow, exact parameter counts, finite-value validation, copying and source
independence, NULL outputs on failure, optional errors, every allocation-failure
point, partial cleanup, and independent layer ownership.

Then implement forward execution. Verify the worked example and NumPy
references, input lengths, output capacity, validation before writes, arithmetic
partial results, untouched sources/spare capacity, repeated calls after failure,
and absence of allocations during inference. Overlap remains a caller
precondition, so tests must not expect invalid aliasing to return a status.
