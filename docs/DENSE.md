# Dense-layer contract

The type and public function declarations are in
[`include/cinder/dense.h`](../include/cinder/dense.h). **Creation, cleanup, and
single-vector forward execution are implemented** in
[`src/dense.c`](../src/dense.c), part of `cinder_core`.

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

## Creation and cleanup implementation

The constructor validates pointers, dimensions, element/byte counts, exact
parameter lengths, and all source values before allocating anything. It uses
division guards before multiplication, so an overflowing shape cannot wrap
into a smaller allocation. Source arrays are not read until shape and length
checks pass. As with other C APIs, callers must actually provide the readable
storage advertised by valid counts.

After validation, the constructor allocates the layer struct and initializes
its two owned tensor pointers to NULL. It calls `cinder_tensor_create` for the
`output_size x input_size` weights and the `output_size x 1` bias. This reuses
the tensor allocator's size checks, initialization, and allocation-failure
cleanup. Once both tensors exist, `memcpy` copies the parameters into their
arrays without changing the sources, including preserving signed zeros.
Only then does the constructor assign the completed layer to `*out_layer`.

There are five allocation attempts for a successful construction:

| Attempt | Allocation | Cleanup if this attempt fails |
| --- | --- | --- |
| 1 | Dense layer struct | Nothing has been allocated. |
| 2 | Weight tensor struct | Release the layer. |
| 3 | Weight element array | Tensor creation releases its struct; Dense cleanup releases the layer. |
| 4 | Bias tensor struct | Release both weight allocations and the layer. |
| 5 | Bias element array | Tensor creation releases the bias struct; Dense cleanup releases both weight allocations and the layer. |

`cinder_dense_free` calls `cinder_tensor_free` on bias and weights, then frees
the layer struct. Both tensor owners start as NULL, and tensor cleanup accepts
NULL, so this same destructor handles constructor failures as well as complete
layers. Tensor creation publishes its own result only on success; a failing
tensor never leaves a dangling owner in the layer. Dense propagates that
tensor error, preserving its diagnostic message. Cleanup itself allocates no
memory and does not change error details.

A caller can use the implemented lifetime functions now:

```c
#include "cinder/dense.h"

int dense_lifetime_example(void)
{
    float weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                       0.5f, -0.5f, 3.0f, 1.0f};
    float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    cinder_dense_t *layer = NULL;
    cinder_error_t error;
    cinder_status_t status =
        cinder_dense_create(2, 4, weights, 8, bias, 4, &layer, &error);
    if (status != CINDER_OK) {
        return 1; /* layer is NULL; there is no partial object to release. */
    }
    weights[0] = 99.0f; /* The owned weight remains 1.0f. */
    int result = layer->weights->data[0] == 1.0f ? 0 : 1;
    cinder_dense_free(layer);
    layer = NULL;
    return result;
}
```

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

The implementation checks pointers, input length, and output capacity, then
scans the owned bias for non-finite values. It calls `cinder_tensor_matvec`,
which already validates the input and weights before writing any output.
Together these checks validate all three sources before computation, without
adding a second scan of the entire weight matrix in Dense.

If matvec returns an error, Dense returns that status immediately with the
kernel's diagnostic and partial output unchanged. It does not attempt bias
addition after a failed multiplication. On success it calls
`cinder_tensor_add_bias` and returns that function's status and error details.
It passes `output_size` as the bias operation's logical output length, even
when the caller supplied extra capacity. The numerical
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

## Worked example and usage

```text
W = [ 1.0   2.0 ]     x = [ 2.0 ]     b = [ 0.5 ]
    [-1.0   0.0 ]         [-1.0 ]         [-1.0 ]
    [ 0.5  -0.5 ]                        [ 2.0 ]
    [ 3.0   1.0 ]                        [ 0.0 ]

Wx      = [0.0, -2.0, 1.5, 5.0]
Wx + b  = [0.5, -3.0, 3.5, 5.0]
```

This example uses the implemented API and can link against `cinder_core`:

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

The contract, creation/cleanup, and forward pass were implemented and verified
in separate scoped steps on October 16, 2026.

- [`test_dense.c`](../tests/test_dense.c) links the production library and
  checks shape/layout, exact counts, overflow boundaries, non-finite values at
  every parameter position, source preservation, independent copies and
  lifetimes, permitted source overlap, signed zeros/extremes/subnormals,
  optional error details, NULL cleanup, and repeated construction/destruction.
- [`test_dense_alloc_failures.c`](../tests/test_dense_alloc_failures.c)
  compiles the real Dense and tensor implementations with test-only allocator
  substitutions. It forces each of the five allocations to fail, both with
  and without error details, tracks all allocated/freed blocks, and rejects
  untracked or duplicate frees. Every failure leaves `*out_layer == NULL`
  with zero outstanding allocations from that constructor. It also verifies
  recovery after every failure, preservation of an already-live layer,
  zero allocation attempts for invalid inputs, and allocation-free cleanup.
  After creation it arms the next allocation to fail and checks unchanged
  allocation/free counters after every forward call: valid input, invalid
  arguments, non-finite sources, matvec overflow, bias overflow, and repeated
  recovery calls with different inputs. These checks run with and without
  optional error details.
- [`test_dense_forward.c`](../tests/test_dense_forward.c) links the production
  library and checks the worked `2 -> 4` example, 100 repeated executions with
  alternating inputs, exact/spare capacity, null pointers, invalid lengths,
  validation order, and unchanged outputs on validation failure. It also
  checks parameter and input preservation, permitted disjoint slices and
  read-only parameter/input overlap, and finite underflow. Deliberate test-only
  parameter-value corruption probes the non-finite guards; applications still
  must treat owned parameters as read-only.
  Arithmetic-overflow cases cover every output position in a `2 -> 3` layer,
  both multiplication and accumulation overflow, bias overflow, exact partial
  output behavior, error propagation, and subsequent successful calls.
- The forward test also compares the Dense API directly with the 11 saved
  NumPy matrix/bias fixtures (41 output values), covering non-square, identity,
  zero, scalar, single-row/column, decimal, and seeded cases. It uses the same
  `atol = rtol = 1e-5` rule, with shape and finiteness checks first. These
  references are independent of Cinder. Python is not needed for this test;
  the optional generated-reference target continues to test the tensor math.
- With the forward pass included, the full ordinary Debug suite passed 12/12
  tests. Release and combined ASan/UBSan builds, with generated tensor NumPy
  references enabled, passed 13/13. The forward and extended allocation-failure
  executables reported zero native macOS leaks. A Release build with
  `BUILD_TESTING=OFF` passed and registered zero tests. The documented lifetime
  and forward examples compiled, linked, and ran successfully.

These are local macOS results. Existing CI jobs will run the new CTest tests
on macOS/Linux and under Linux sanitizers after publication; hosted results
for this change have not been verified.

The [common layer interface](LAYER.md) now implements a tagged wrapper that
owns this Dense object, exposes its dimensions, and delegates forward execution.
Common constructors, validation, cleanup, and activation dispatch are implemented;
the sequential-model contract is next. Dense applies no activation and does not yet
participate in a model execution API. Buffer overlap remains a caller
precondition, not a runtime-diagnosed error.
