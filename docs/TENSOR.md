# Tensor contract

This is the contract for the first Phase 1 implementation. Types and function
declarations are in `include/cinder/tensor.h` and `include/cinder/error.h`.
The functions are not implemented or linked into the CLI yet.

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
must check this platform requirement at compile time.

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

- `cinder_tensor_fill` replaces all elements with one finite value.
- `cinder_tensor_get` reads a single element into a separate caller-owned
  `float`. It reports an error if that element is non-finite.
- `cinder_tensor_set` replaces a single element with a finite value.

All three functions reject NULL required pointers. Get and set check bounds
before computing an offset. Fill and set reject NaN and positive/negative
infinity. On failure, fill and set leave the tensor unchanged; get leaves its
output value unchanged. Get never changes the tensor.

Output and error objects must be valid writable storage that does not overlap
the tensor, its element array, or each other. Apart from documented NULL
checks, a non-NULL tensor argument must refer to a live object created by this
API with its metadata intact. C cannot reliably diagnose dangling pointers or
prove the capacity of an arbitrary pointer; those are caller obligations.

## Errors

| Status | Meaning |
| --- | --- |
| `CINDER_OK` | The operation completed successfully. |
| `CINDER_ERROR_INVALID_ARGUMENT` | A required pointer is NULL, or a creation dimension is zero. |
| `CINDER_ERROR_OVERFLOW` | The element count or allocation byte count cannot fit in `size_t`. |
| `CINDER_ERROR_ALLOCATION` | An allocation failed after dimensions and sizes were validated. |
| `CINDER_ERROR_OUT_OF_BOUNDS` | A get/set index is outside the tensor's dimensions. |
| `CINDER_ERROR_NONFINITE` | A supplied fill/set value, or the element read by get, is NaN or infinity. |

Validation order is required pointers, dimensions or bounds, then numeric
values or allocation sizes as applicable. An operation stops at the first
failed check and returns its status. Only creation allocates memory; fill,
get, and set do not.

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

## Next implementation check

Implement create/free first, with CTest coverage for zero initialization,
invalid dimensions, both size-overflow cases, and allocation failure cleanup.
Run those tests with ASan and UBSan enabled on the new code and test target.
Fill/get/set and their bounds/finite-value tests follow as a separate step.
