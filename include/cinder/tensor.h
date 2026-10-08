#ifndef CINDER_TENSOR_H
#define CINDER_TENSOR_H

#include <stddef.h>

#include "cinder/error.h"

/*
 * Contract declarations only; implementations will be added separately.
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

#endif /* CINDER_TENSOR_H */
