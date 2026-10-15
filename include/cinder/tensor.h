#ifndef CINDER_TENSOR_H
#define CINDER_TENSOR_H

#include <stddef.h>

#include "cinder/error.h"

/*
 * Creation, cleanup, checked fill/get/set, matrix-vector multiplication,
 * bias addition, and ReLU are implemented.
 * See docs/TENSOR.md for ownership, failure behavior, and examples.
 *
 * A live tensor owns rows * cols contiguous float32 elements. Both dimensions
 * are positive. The element at (row, col) is data[row * cols + col], using
 * zero-based indices. Cinder requires an IEEE 754 binary32 C float platform.
 *
 * The caller owns the tensor returned by create. Treat rows, cols, and the
 * data pointer as read-only metadata; do not copy this struct by value or
 * replace/free its data pointer. Element values may be changed, but must be
 * finite. Borrowed pointers remain valid only until the owner frees it.
 * Access functions require a live tensor with intact metadata. Output/error
 * storage must not overlap the tensor, its array, or each other. Validation
 * order is required pointers, indices (when present), then finite values.
 */
typedef struct {
    size_t rows;
    size_t cols;
    float *data;
} cinder_tensor_t;

/*
 * Create a tensor with every element initialized to 0.0f.
 * out_tensor must point to a writable pointer slot that owns no live tensor.
 * For non-NULL out_tensor, *out_tensor is set to NULL before other validation
 * and remains NULL on failure. Partial allocations are released on failure.
 *
 * INVALID_ARGUMENT: NULL out_tensor or either dimension is zero.
 * OVERFLOW: element count or allocation byte count cannot fit in size_t.
 * ALLOCATION: memory allocation failed after validation.
 */
cinder_status_t cinder_tensor_create(
    size_t rows,
    size_t cols,
    cinder_tensor_t **out_tensor,
    cinder_error_t *error
);

/*
 * Release both the element array and the tensor. NULL is a no-op.
 * A non-NULL argument must be a live tensor returned by create, freed once
 * by its owner. All aliases become invalid; the caller's pointer is not reset.
 */
void cinder_tensor_free(cinder_tensor_t *tensor);

/*
 * Set every element to value. NULL tensor is INVALID_ARGUMENT; a non-finite
 * value is NONFINITE. On failure, the tensor remains unchanged.
 */
cinder_status_t cinder_tensor_fill(
    cinder_tensor_t *tensor,
    float value,
    cinder_error_t *error
);

/*
 * Read one element into separate caller-owned storage.
 * NULL tensor/out_value is INVALID_ARGUMENT. An invalid index is
 * OUT_OF_BOUNDS; a non-finite element is NONFINITE.
 * On failure, *out_value remains unchanged. The tensor is never modified.
 */
cinder_status_t cinder_tensor_get(
    const cinder_tensor_t *tensor,
    size_t row,
    size_t col,
    float *out_value,
    cinder_error_t *error
);

/*
 * Write one element. NULL tensor is INVALID_ARGUMENT; an invalid index is
 * OUT_OF_BOUNDS; a non-finite value is NONFINITE.
 * On failure, the tensor remains unchanged.
 */
cinder_status_t cinder_tensor_set(
    cinder_tensor_t *tensor,
    size_t row,
    size_t col,
    float value,
    cinder_error_t *error
);

/*
 * Compute output = weights * input without allocating memory.
 * weights must be a live tensor with intact metadata. Sizes count floats:
 * input_count must equal weights->cols; output_capacity >= weights->rows.
 * The caller must provide the actual readable/writable buffer sizes. Output
 * and error storage must not overlap weights, its array, input, or each other.
 *
 * INVALID_ARGUMENT: a required pointer is NULL.
 * DIMENSION_MISMATCH: input_count differs from the number of columns.
 * BUFFER_TOO_SMALL: output_capacity is less than the number of rows.
 * NONFINITE: an input/weight or an intermediate sum is NaN or infinity.
 * Validation follows that order, with all inputs checked before computation.
 *
 * On success, writes exactly weights->rows results; any extra output elements
 * stay unchanged. Validation failures leave output unchanged. Arithmetic
 * failure can leave earlier rows written: discard the result on any failure.
 * Weights and input are never modified; unused output capacity is untouched
 * even on failure. Accumulation uses float in increasing column order.
 */
cinder_status_t cinder_tensor_matvec(
    const cinder_tensor_t *weights,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);

/*
 * Add bias[i] to output[i] in place, without allocating memory.
 * Counts describe vector lengths in floats, not spare buffer capacity. After
 * matvec, pass weights->rows as output_count. Both counts must be nonzero and
 * equal. The caller supplies at least that many initialized readable/writable
 * output floats and readable bias floats. The two ranges and optional error
 * storage must not overlap; overlap is a caller precondition, not checked.
 *
 * Validation order and statuses:
 * INVALID_ARGUMENT: NULL output/bias or either count is zero.
 * DIMENSION_MISMATCH: vector lengths differ.
 * OVERFLOW: the vector byte count cannot fit in size_t.
 * NONFINITE: an output/bias value or a computed sum is NaN or infinity.
 * All source values are checked before any output is written.
 *
 * Success updates exactly output_count elements and leaves bias unchanged.
 * Validation failures preserve output. Arithmetic failure can leave earlier
 * elements updated; the failing element and later ones are not written.
 * Discard the result on any failure. Storage beyond output_count is untouched.
 */
cinder_status_t cinder_tensor_add_bias(
    float *output,
    size_t output_count,
    const float *bias,
    size_t bias_count,
    cinder_error_t *error
);

/*
 * Apply ReLU in place: replace values below zero with +0.0f. Positive values
 * and existing zeros (including -0.0f) stay unchanged. No memory is allocated.
 * count is the vector length in floats, not spare capacity; after matvec use
 * weights->rows. The caller supplies at least count initialized, readable and
 * writable floats. Optional error storage must not overlap the vector.
 *
 * Validation order and statuses:
 * INVALID_ARGUMENT: NULL values or zero count.
 * OVERFLOW: the vector byte count cannot fit in size_t.
 * NONFINITE: any value is NaN or infinity, including negative infinity.
 * The whole vector is validated before mutation, so every failure leaves it
 * unchanged. Storage beyond count is neither read nor written.
 */
cinder_status_t cinder_tensor_relu(
    float *values,
    size_t count,
    cinder_error_t *error
);

#endif /* CINDER_TENSOR_H */
