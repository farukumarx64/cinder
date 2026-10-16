# Tensor contract

This is the contract for the first Phase 1 implementation. Types and function
declarations are in `include/cinder/tensor.h` and `include/cinder/error.h`.
Creation, cleanup, fill/get/set, matrix-vector multiplication, bias addition,
ReLU, and Softmax are implemented in `src/tensor.c`, built as the `cinder_core`
static library shared by the CLI and ordinary tensor tests.
The CLI still exposes only `--version`; the tensor API is exercised by C tests.

## Representation and layout

```c
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} cinder_tensor_t;
```

| Field | Meaning |
| --- | --- |
| `rows` | Number of rows; greater than zero. |
| `cols` | Number of columns; greater than zero. |
| `data` | Owned pointer to `rows * cols` consecutive `float` elements. |

`size_t` is the unsigned type used for sizes and indices. Dimensions count
elements, not bytes. The supported platform must represent C `float` as
IEEE 754 binary32 (32 bits, four 8-bit bytes). The allocation implementation
checks byte size, float size, radix, precision, exponent range, and subnormal
support at compile time.

Softmax also requires `double` to have more precision and a wider exponent
range than `float`, enforced at compile time. It uses these wider intermediates
for subtraction, exponentiation, summation, and division; stored tensors and
vectors remain float32. The supported macOS/Linux toolchains satisfy this.

Storage is row-major: all elements of the first row appear before those of the
second row. With zero-based indices:

```text
offset = row * cols + col

matrix (2 rows, 3 columns)       data (6 elements)
[ 1  2  3 ]                    [ 1, 2, 3, 4, 5, 6 ]
[ 4  5  6 ]

element (1, 2) -> offset 1 * 3 + 2 = 5 -> value 6
```

Valid indices satisfy `row < rows` and `col < cols`. A vector can be stored as
an `N x 1` tensor. Dense weights use `[output_size][input_size]`, so each row
contains the weights for one output. There are no strides, views, borrowed
storage, resizing, or implicit broadcasting in this contract.

## Creation and lifetime

`cinder_tensor_create(rows, cols, &tensor, error)` allocates the struct and its
element array, then initializes every element to `0.0f`. On success, it returns
`CINDER_OK` and gives the caller sole ownership through `tensor`.

The output pointer slot must not already own a live tensor. For a non-NULL
`out_tensor`, creation sets `*out_tensor` to `NULL` before validating dimensions.
Every failure leaves that slot `NULL` and releases any partial allocations.
Passing a live owner's slot would overwrite the only pointer to that object;
the caller must free the previous tensor first or use a different slot.

Creation validates sizes before any allocation or unsafe multiplication:

1. Reject a NULL output pointer or a zero dimension.
2. Reject `rows > SIZE_MAX / cols` before calculating the element count.
3. After obtaining the count, reject `count > SIZE_MAX / sizeof(float)` before
   calculating the allocation byte count.
4. Attempt allocation; report failure and clean up if either allocation fails.

`cinder_tensor_free(tensor)` releases the array and the struct.
`cinder_tensor_free(NULL)` does nothing. Free each successful allocation exactly
once. Free does not reset the caller's pointer; set it to `NULL` afterward when
it remains in scope. Passing the same freed pointer again is invalid.

The public fields make the layout visible for learning and math kernels, but
callers must not alter `rows`, `cols`, or the `data` pointer. Do not construct
owning tensors manually, copy the struct by value, or call `free` on its data.
Borrowing a pointer for a function call does not transfer ownership. Every
borrowed tensor or data pointer becomes invalid when the owner frees it.

Direct element access is allowed within bounds. Callers are responsible for
preserving finite values and valid indices when bypassing the checked access
functions. Concurrent access requires caller coordination whenever any access
modifies or frees the tensor.

## Access and mutation

All three access functions are implemented and allocate no memory.

- `cinder_tensor_fill` replaces all elements with one finite value.
- `cinder_tensor_get` reads a single element into a separate caller-owned
  `float`. It reports an error if that element is non-finite.
- `cinder_tensor_set` replaces a single element with a finite value.

All three functions reject NULL required pointers. Get and set check bounds
before computing an offset. Fill and set reject NaN and positive/negative
infinity. On failure, fill and set leave the tensor unchanged; get leaves its
output value unchanged. Get never changes the tensor.

Validation order is required pointers, row/column bounds (for get/set), then
finite values. A set call with both an invalid index and NaN therefore reports
`CINDER_ERROR_OUT_OF_BOUNDS`. A get call with a NULL output pointer reports
`CINDER_ERROR_INVALID_ARGUMENT` before inspecting indices or elements.

Bounds are checked separately: `row >= rows || col >= cols` rejects the call
before calculating `row * cols + col`. For a `2 x 3` tensor, `(0, 3)` is invalid
even though its flattened offset would land inside the array at the next row.
Successful creation already checked the total element count for overflow;
valid indices into a live tensor with intact metadata therefore produce a
representable offset. Fill relies on the same validated dimensions for its loop.

Get checks only the requested element, keeping reads constant-time. It reads
into a local variable, checks finiteness, and only then assigns `*out_value`.
Set is also constant-time. Fill visits all `rows * cols` elements and checks
its supplied value before starting the loop. Finite negative values, signed
zero, subnormal values, and the largest finite floats are accepted.

Output and error objects must be valid writable storage that does not overlap
the tensor, its element array, or each other. Apart from documented NULL
checks, a non-NULL tensor argument must refer to a live object created by this
API with its metadata intact. C cannot reliably diagnose dangling pointers or
prove the capacity of an arbitrary pointer; those are caller obligations.

## Matrix-vector multiplication

```c
cinder_status_t cinder_tensor_matvec(
    const cinder_tensor_t *weights,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);
```

This computes `y = Wx`. For a weight matrix with `rows = M` and `cols = N`,
the input must have exactly `N` elements, and the output buffer must have room
for at least `M` elements. Both sizes count floats, not bytes. Vectors are
passed as contiguous arrays with explicit lengths; no vector tensor allocation
is required. An existing vector tensor's data can be used with the appropriate
element count, provided it has separate storage from the output.

`weights` must be a live tensor with intact metadata. The caller owns all
buffers and must report their available sizes accurately. Output and optional
error storage must not overlap the weight tensor, its array, the input, or each
other. In-place multiplication is unsupported; buffer overlap is a caller
precondition, not something this function detects. The read-only input and
weight data may overlap one another.

Validation happens before any output writes, in this order:

1. Reject NULL weights, input, or output pointers with `INVALID_ARGUMENT`.
2. Require `input_count == weights->cols`, otherwise `DIMENSION_MISMATCH`.
3. Require `output_capacity >= weights->rows`, otherwise `BUFFER_TOO_SMALL`.
4. Require every input value and weight to be finite, otherwise `NONFINITE`.

The loop computes one dot product per row:

```text
y[row] = W[row, 0] * x[0] + W[row, 1] * x[1] + ... + W[row, N-1] * x[N-1]
```

The outer loop chooses the output row and resets a `float sum` to `0.0f`.
The inner loop visits columns in increasing order. The weight at `(row, col)`
is `weights->data[row * weights->cols + col]`. Each column contributes that
weight multiplied by `input[col]`. A completed sum is assigned to `output[row]`,
replacing any previous value there. No bias or activation is applied.

For the hand-calculated non-square fixture:

```text
W = [ 1  -2   3 ]      x = [  2  ]
    [ 4   0  -1 ]          [ -1  ]
                           [ 0.5 ]

y[0] = 1*2 + (-2)*(-1) + 3*0.5 = 2 + 2 + 1.5 = 5.5
y[1] = 4*2 + 0*(-1) + (-1)*0.5 = 8 + 0 - 0.5 = 7.5
```

Inputs and weights remain unchanged. On success, exactly `M` output elements
are written; any spare capacity is untouched. Pointer, size, and source-value
validation failures leave the output unchanged. During computation, each
updated sum is checked for finiteness. Arithmetic overflow can occur even with
finite inputs and returns `CINDER_ERROR_NONFINITE`. Earlier rows may already
have been written, so callers must discard the result after any failure.
The failing row is not written; unused output capacity remains untouched.

The accumulator is a `float`, matching the stored data. Underflow to a finite
subnormal value or zero is allowed. Comparisons use the roadmap's tolerance
rule, `abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)`, rather than
requiring identical rounding across compilers or numerical libraries. A sum
that becomes non-finite is rejected immediately, even if later terms would
cancel an earlier large term in exact arithmetic.

The operation allocates no memory and takes `O(M * N)` time with `O(1)` extra
space. The validation pass adds a scan of the source arrays but preserves the
output when any source value is invalid. Creation already proved that the
weight element count and row offsets fit in `size_t`; matvec relies on the
same intact-metadata requirement as get/set.

## Bias addition

```c
cinder_status_t cinder_tensor_add_bias(
    float *output,
    size_t output_count,
    const float *bias,
    size_t bias_count,
    cinder_error_t *error
);
```

This updates the multiplication result in place: `output[i] += bias[i]` for
each element. Both counts are logical vector lengths measured in floats and
must be equal and nonzero. There is one bias per output row, not per input
column. No scalar broadcasting is performed. For a matrix with `M` rows,
call matvec first and then pass `M` as `output_count` and `bias_count`.

The caller owns the buffers and supplies at least `output_count` initialized
readable/writable floats and `bias_count` initialized readable floats. The
output range, bias range, and optional error object must not overlap; this is
a caller precondition and is not checked at runtime. Output is updated in its
existing storage, and bias is never modified. Memory beyond `output_count` is
neither read nor written. A pointer cannot prove its buffer's actual capacity,
so lengths must accurately describe accessible elements.

Validation runs in this order:

1. Reject NULL output or bias pointers with `INVALID_ARGUMENT`.
2. Reject either zero count with `INVALID_ARGUMENT`.
3. Reject unequal counts with `DIMENSION_MISMATCH`.
4. Reject `output_count > SIZE_MAX / sizeof(float)` with `OVERFLOW`. This
   ensures the vector's byte extent can fit in `size_t` without multiplying
   an unchecked count. The equal-length bias has the same extent.
5. Require every output and bias value to be finite, otherwise `NONFINITE`.

Only after the full validation pass does a second loop calculate each
`float sum = output[i] + bias[i]`. It checks that sum for finiteness before
storing it back to `output[i]`. Pointer, size, and source-value failures leave
the entire output unchanged. Arithmetic overflow reports `NONFINITE` and can
leave earlier elements updated; the failing element and later elements are
not written. As with matvec, discard the result on any failure.

Using the matrix and input from the preceding example:

```text
Wx = [5.5, 7.5]
b  = [0.5, -2.0]

y[0] = 5.5 + 0.5  = 6.0
y[1] = 7.5 + (-2) = 5.5
```

The two calls compose as follows, with `weights` already containing that
`2 x 3` example:

```c
const float input[] = {2.0f, -1.0f, 0.5f};
const float bias[] = {0.5f, -2.0f};
float output[4];
cinder_error_t error;

cinder_status_t status =
    cinder_tensor_matvec(weights, input, 3, output, 4, &error);
if (status == CINDER_OK) {
    status = cinder_tensor_add_bias(output, weights->rows, bias, 2, &error);
}
if (status == CINDER_OK) {
    /* output[0] is 6.0f; output[1] is 5.5f. Only those two are results. */
}
```

Matvec receives the output buffer capacity (`4`). Bias addition receives the
number of results (`weights->rows`, which is `2`), so spare, uninitialized
storage is not treated as part of the vector. Apply bias only after matvec
succeeds. Bias addition uses `O(M)` time, `O(1)` extra space, and no allocations.
Together these functions implement the arithmetic `y = Wx + b`; the layer
abstraction and model execution remain future steps.

## ReLU

```c
cinder_status_t cinder_tensor_relu(
    float *values,
    size_t count,
    cinder_error_t *error
);
```

ReLU (rectified linear unit) applies `max(0, value)` to each finite element.
This implementation replaces values below zero with positive `0.0f` and leaves
positive values and existing zeros unchanged. In particular, `-0.0f` compares
equal to zero in C, so its sign is preserved; a negative nonzero value becomes
positive zero. Negative and positive subnormal values follow the same rule.

```text
before: [-3.5, 0.0, 2.75]
after:  [ 0.0, 0.0, 2.75]
```

The function updates the vector in place and allocates no memory. `count` is
the number of initialized readable/writable floats to process, not spare
capacity. After matvec and bias addition, use `weights->rows`. The caller must
provide that much valid storage; the function cannot discover the allocation
size from a pointer. The optional error object must not overlap the vector.
Elements beyond `count` are neither read nor written.

Validation runs before any mutation:

1. Reject a NULL pointer or zero count with `INVALID_ARGUMENT`.
2. Reject `count > SIZE_MAX / sizeof(float)` with `OVERFLOW`.
3. Reject NaN or either infinity anywhere in the vector with `NONFINITE`.

Negative infinity is rejected under the same finite-value policy as the other
math operations. A full validation pass precedes the replacement loop, so a
non-finite value at the end cannot leave earlier negative elements changed.
Every failure preserves the entire vector. After validation, comparisons and
zero assignments cannot overflow, so there is no arithmetic failure path.

The replacement loop visits each element once:

```c
for (size_t i = 0; i < count; ++i) {
    if (values[i] < 0.0f) {
        values[i] = 0.0f;
    }
}
```

The validation and replacement passes together take `O(count)` time with
`O(1)` extra space. Applying ReLU a second time leaves the result unchanged.

ReLU composes after successful matvec and bias calls. Using the same matrix
and input as the matvec example, with bias `[0.5, -9.5]`:

```text
Wx              = [5.5,  7.5]
Wx + b          = [6.0, -2.0]
ReLU(Wx + b)    = [6.0,  0.0]
```

```c
/* status is the result of the preceding bias-addition call. */
if (status == CINDER_OK) {
    status = cinder_tensor_relu(output, weights->rows, &error);
}
```

Consume the result only when the final status is `CINDER_OK`. The layer
abstraction and model execution remain separate future steps.

## Softmax

```c
cinder_status_t cinder_tensor_softmax(
    float *values,
    size_t count,
    cinder_error_t *error
);
```

Softmax turns a vector of finite scores into probabilities. It preserves the
vector length and ordering: larger scores receive larger probabilities, except
where rounding makes them equal. Outputs are finite, lie in `[0, 1]`, and sum
approximately to one. Floating-point rounding means an exact sum of one is not
required, and sufficiently small probabilities may become zero.

The stable formula subtracts the largest input before exponentiation:

```text
m = max(values)
probability[i] = exp(values[i] - m) / sum(exp(values[j] - m))
```

In exact arithmetic, subtracting the same value multiplies every exponential
by the same factor, which cancels between numerator and denominator. The
probabilities are unchanged, but every exponent is now nonpositive, so each
exponential is at most one. At least one maximum contributes `exp(0) = 1`,
preventing the normalization sum from being zero even if other terms underflow.

For example, directly exponentiating the scores below would overflow float32:

```text
scores:           [1001,       1002,       1003]
subtract 1003:    [  -2,         -1,          0]
exponentials:     [0.1353353,  0.3678794,  1.0000000]
sum:              1.5032147
probabilities:   [0.09003057, 0.24472847, 0.66524096]
```

The function makes three passes:

1. Validate every input and find the maximum, starting from the first element.
2. Compute each shifted exponential, store it in the same array as a float,
   and add that stored term to a `double` total.
3. Divide each stored exponential by the total using `double` arithmetic,
   then store the resulting float32 probability in place.

The subtraction converts both operands before doing arithmetic:

```c
double shifted = (double)values[i] - (double)maximum;
```

This accepts opposite finite extremes: `-FLT_MAX - FLT_MAX` would overflow if
performed in float, but fits in the wider double range. `exp` evaluates the
shifted value as a double. The sum also uses double so small terms are less
likely to disappear when added to a larger running total. It sums the rounded
float terms actually stored in the array, which are also the numerators used
in the final division. These choices improve range and precision without
changing the public float32 storage contract or promising exact arithmetic.

Finite underflow is accepted. `[0, -90]` retains a tiny subnormal probability;
`[0, -104]` produces `[1, 0]` after rounding to float32. Equal scores produce a
uniform distribution, a single score produces `[1]`, and
`[-FLT_MAX, FLT_MAX]` produces `[0, 1]`. NaN and either infinity are unsupported
inputs, including vectors containing only infinities; no special probability
distribution is invented for them.

Validation runs before modifying the vector:

1. A NULL pointer or zero count returns `INVALID_ARGUMENT`.
2. `count > SIZE_MAX / sizeof(float)` returns `OVERFLOW` before reading data.
3. Any NaN or infinity returns `NONFINITE` after checking the relevant element.

Pointer, size, and input-value failures leave the entire array unchanged.
Defensive checks also reject a non-finite exponential, a nonpositive or
non-finite normalization sum, or a non-finite probability as `NONFINITE`.
A numeric failure after computation begins may leave scratch exponentials or
partially normalized values in the array. Discard the result on any failure.
For valid finite inputs under the supported floating-point behavior, the
maximum shift and wider intermediates keep these calculations finite with a
positive denominator.

`count` is the logical vector length, not spare capacity. The caller supplies
at least `count` initialized, readable/writable floats and keeps optional error
storage separate. Elements beyond `count` are neither read nor written. As
with ReLU, after matvec pass `weights->rows`, not the output buffer capacity.
The array is overwritten, so preserve a separate copy if original scores are
needed. Softmax takes `O(count)` time, `O(1)` extra space, and no allocations.

The math operations can be composed by checking each returned status:

```text
Wx                      = [5.5,        7.5]
Wx + [0.5, -9.5]        = [6.0,       -2.0]
ReLU(Wx + b)            = [6.0,        0.0]
Softmax(ReLU(Wx + b))   = [0.99752738, 0.00247262]
```

This is a composition test, not a requirement to place ReLU before Softmax;
a model's layer order determines which activation is used. Apply Softmax only
after the preceding operation succeeds and consume it only on `CINDER_OK`.

`exp` introduces a system math-library dependency. CMake links `m` on Unix and
propagates it from the static core library to consumers. The allocation-failure
test compiles the implementation directly, so it links `m` separately.

## Errors

| Status | Meaning |
| --- | --- |
| `CINDER_OK` | The operation completed successfully. |
| `CINDER_ERROR_INVALID_ARGUMENT` | A required pointer is NULL, a creation dimension is zero, or a bias/ReLU/Softmax vector is empty. |
| `CINDER_ERROR_OVERFLOW` | A tensor element count, allocation byte count, or bias/ReLU/Softmax vector byte count cannot fit in `size_t`. |
| `CINDER_ERROR_ALLOCATION` | An allocation failed after dimensions and sizes were validated. |
| `CINDER_ERROR_OUT_OF_BOUNDS` | A get/set index is outside the tensor's dimensions. |
| `CINDER_ERROR_NONFINITE` | A supplied/read value, math source value, or computed result is NaN or infinity, or a Softmax normalization sum is nonpositive. |
| `CINDER_ERROR_DIMENSION_MISMATCH` | The matvec input length differs from the weight columns, or the output/bias lengths differ. |
| `CINDER_ERROR_BUFFER_TOO_SMALL` | The matvec output capacity is less than the number of weight rows. |

Validation order is required pointers, dimensions or bounds, then numeric
values or allocation sizes as applicable. An operation stops at the first
failed check and returns its status. Only creation allocates memory; all current
access and math operations use existing storage. `OVERFLOW` describes integer
size calculations; floating-point overflow during math is reported as `NONFINITE`.

Each fallible function takes an optional `cinder_error_t *error`. If provided,
the caller owns this object and need not initialize it before the call. The
function sets `error->code` to the returned status. On failure it supplies a
useful, null-terminated message in the 128-byte inline buffer; on success it
sets the message to the empty string. Long messages may be truncated while
remaining terminated. Message wording is not a stable programmatic interface.

Passing `NULL` for error details is supported; the return status still reports
the result. Functions do not print diagnostics, terminate the process, use
global error state, or allocate memory for error messages. The caller decides
whether to report, propagate, or handle an error.

## Allocation verification

`tensor_allocation` links the ordinary `cinder_core` library and exercises
zero initialization, several shapes, writable contiguous storage, independent
allocations, invalid arguments, both overflow checks, optional error details,
and repeated cleanup. `cinder_tensor_free(NULL)` is also exercised.

`tensor_allocation_failures` compiles the actual `src/tensor.c` implementation
inside a test translation unit, replacing only its calls to `malloc`, `calloc`,
and `free`. The wrappers can fail a selected allocation and track live blocks.
They verify cleanup when either allocation fails, absence of allocation during
validation failures, and successful creation after an earlier failure. The
largest byte-safe element count is checked with the first allocation forced to
fail, so the test does not attempt an enormous allocation. This executable does
not link a second copy of the implementation from `cinder_core`.

On October 9, 2026, all three CTest tests passed locally on macOS in Debug,
Release, and combined ASan/UBSan builds. The core library and test translation
units receive the same warning and sanitizer settings. Native macOS leak
checks reported zero leaks for both tensor test executables. A build with
`BUILD_TESTING=OFF` also passed. These are the local results from that step.

## Access verification

`tensor_access` checks fill over `1 x 1`, rectangular, row-vector, and
column-vector shapes. Get reads known flat-array fixtures, while set is
checked against raw storage and unchanged neighboring elements; correctness
does not depend solely on a set/get round trip. Tests cover boundary and huge
indices, NULL pointers, NaN, both infinities, validation order, unchanged
outputs/data on failure, signed zero, finite extremes, and optional error
details. Successful calls must clear previous error details.

The allocation-failure test also arms the next allocation to fail and checks
that successful and rejected access calls leave allocation counts unchanged.

On October 15, 2026, all four CTest tests passed locally on macOS in Debug,
Release, and combined ASan/UBSan builds. The new access executable reported
zero leaks under native macOS `leaks`. A build with `BUILD_TESTING=OFF` also
succeeded. These are the local results from the access step.

## Matrix-vector verification

`tensor_matvec` checks the hand-calculated `2 x 3` fixture above, decimal values,
identity and zero-input cases, a single row, a single column, and `1 x 1`
matrices with extreme finite values. The comparison checks finiteness first and
uses the tolerance rule above; zero and smallest-subnormal results also receive
exact checks so that the absolute tolerance cannot hide those mistakes.

Tests exercise exact and excess output capacity, guard elements, unchanged
sources, NULL pointers, short/long input lengths, insufficient output space,
validation order, and optional error details. NaN and both infinities are
tested at every position in the non-square fixture's input and weights. Finite
values that overflow a product or running sum verify the arithmetic error path,
including a completed row before the failing row. The allocator test confirms
that matvec does not allocate on success or validation failure.

On October 15, 2026, all five CTest tests passed locally on macOS in Debug,
Release, and combined ASan/UBSan builds. Native macOS `leaks` reported zero leaks
for the matvec executable; a build with `BUILD_TESTING=OFF` also succeeded.
These are the local results from the matvec step. Subsequent NumPy reference
verification is recorded below.

## Bias verification

`tensor_bias` checks positive, negative, zero, fractional, and extreme finite
values across vector lengths 1, 2, and 5. It uses the same absolute/relative
tolerance rule as matvec, with exact checks for zero and subnormal results.
Tests verify unchanged bias storage and guards around the output, zero-length
and mismatched vectors, NULL pointers, unrepresentable byte counts, validation
order, optional error details, and clearing previous errors on success.

NaN and both infinities are tested at every position in both vectors. Positive
and negative arithmetic overflow are exercised at the first, middle, and last
elements to check the documented partial-update behavior. The allocator test
also confirms no allocation on success, validation failure, or arithmetic
failure.

An integration test computes `Wx` and then adds `b` for the non-square example
above. A second case uses a zero input and verifies that `Wx + b` equals `b`.
Spare output capacity containing NaN remains untouched, demonstrating why the
bias count must use the result length rather than the buffer capacity.

On October 15, 2026, all six CTest tests passed locally on macOS in Debug,
Release, and combined ASan/UBSan builds. Native macOS `leaks` reported zero leaks
for the bias executable, and a build with `BUILD_TESTING=OFF` succeeded.
These are the local results from the bias-addition step.

## ReLU verification

`tensor_relu` tests mixed, all-negative, all-positive, and single-element
vectors, including both signed zeros, the smallest positive/negative subnormal
values, and the largest finite floats. ReLU only compares values and stores
zero, so results are checked exactly, including the sign of zero. A second
application must leave the entire result unchanged.

Tests cover NULL pointers, zero and unrepresentable counts, NaN and both
infinities at every vector position, unchanged data on failure, validation
order, optional error details, and clearing earlier errors on success. Guard
elements and NaN in spare capacity verify the logical-length boundary. The
allocator test confirms no allocation on success or failure.

An integration test checks all three intermediate results in the non-square
`ReLU(Wx + b)` example above and leaves spare output capacity untouched.

On October 15, 2026, all seven CTest tests passed locally on macOS in Debug,
Release, and combined ASan/UBSan builds. Native macOS `leaks` reported zero leaks
for the ReLU executable, and a build with `BUILD_TESTING=OFF` succeeded.
These are the local results from the ReLU step.

## Softmax verification

`tensor_softmax` checks known probabilities, large positive and negative score
offsets, permutations, equal scores, single-element vectors, signed zeros,
opposite finite extremes, and tied extreme maxima. Reference constants were
evaluated independently with Python's Decimal at 60-digit precision; the test
executable has no Python dependency. Ordinary comparisons use the established
`atol = rtol = 1e-5` rule. Every valid fixture also checks finite outputs in
`[0, 1]` with a double-accumulated probability sum within `1e-6` of one.

Tiny-probability cases use a tighter absolute bound of `FLT_TRUE_MIN`, plus
explicit checks that representable subnormal probabilities stay positive and
unrepresentable probabilities round to zero. A 10,001-element vector contains
one score of zero and 10,000 scores of `-18`; it verifies that the many small
exponentials contribute to normalization instead of disappearing in a float
sum. Those small results receive relative-error checks without the ordinary
absolute tolerance.

Validation tests cover NULL pointers, zero/overflow counts, NaN and both
infinities at every position, all-non-finite vectors, unchanged array bytes on
rejection, validation order, optional error details, and cleared errors on
success. Guard elements and NaN in spare capacity check the length boundary.
The integration case verifies the intermediate results of
`Softmax(ReLU(Wx + b))`. The allocation-failure executable confirms that
successful and rejected Softmax calls make no allocations.

On October 16, 2026, all eight CTest tests passed locally on macOS in Debug,
Release, and combined ASan/UBSan builds. Native macOS `leaks` reported zero
leaks for the Softmax executable. A build with `BUILD_TESTING=OFF` compiled,
printed the version, and registered zero tests. Hosted CI verification of
Softmax awaits a future push.

## NumPy reference verification

All four math operations are now compared with deterministic float32 NumPy
references. The saved fixture header is tested in ordinary C builds; an opt-in
build generates fresh references with NumPy and compiles the same comparison
test against them. Every intermediate vector is checked, including logits
before Softmax, with equal lengths and finite values required before applying
`abs(actual - reference) <= 1e-5 + 1e-5 * abs(reference)`.

On October 16, 2026, the Debug build passed all 9 tests. Release and combined
ASan/UBSan builds with fresh NumPy generation each passed all 10 tests. Each
reference run compared 79 vectors / 341 results across 11 matrix/bias and 12
activation fixtures. The largest absolute error was `1.1920929e-7`; no tolerance
change or math implementation change was needed. The saved references were
generated with Python 3.12.14 and NumPy 2.3.5. Hosted CI results remain pending.

See [NumPy reference verification](NUMPY_TESTS.md) for fixture coverage,
regeneration, diagnostics, local results, and the limits of these comparisons.
This completes local Phase 1 verification. The [Dense-layer contract](DENSE.md)
now defines how Phase 2 will package these tensors and math kernels. Dense
creation and cleanup are the next implementation step.
