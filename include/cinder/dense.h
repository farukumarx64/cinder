#ifndef CINDER_DENSE_H
#define CINDER_DENSE_H

#include <stddef.h>

#include "cinder/tensor.h"

/*
 * Creation and cleanup are implemented; forward execution is declared only.
 * See docs/DENSE.md for validation order, ownership, overlap, and examples.
 *
 * A Dense layer computes y = Wx + b for one float32 input vector.
 * Both sizes are positive. weights has shape [output_size][input_size] in
 * row-major order; bias has shape [output_size][1]. The layer owns both tensors
 * and their arrays, copied from the constructor's caller-owned source arrays.
 *
 * All fields, nested tensor metadata, and parameter elements are read-only to
 * callers. Do not construct a layer manually, copy it by value, mutate its
 * parameters, or free/replace its tensors. A live layer comes from create and
 * is released once with cinder_dense_free. Borrowed pointers expire on free.
 */
typedef struct {
    size_t input_size;
    size_t output_size;
    cinder_tensor_t *weights;
    cinder_tensor_t *bias;
} cinder_dense_t;

/*
 * Create a layer owning independent copies of finite float32 parameters.
 * Counts are in floats: weight_count == output_size * input_size and
 * bias_count == output_size. Bias is required; supply explicit zeros if needed.
 * The original arrays remain caller-owned and unchanged on success or failure.
 * On success, they can be changed/released without affecting the layer.
 *
 * out_layer is a writable pointer slot that must not already own a live layer.
 * If non-NULL, *out_layer is set to NULL before other argument validation and
 * stays NULL on every failure. Any partial allocations are released.
 * The source arrays may overlap each other because both are read-only. The
 * out_layer slot and optional error object must not overlap each other or the
 * source arrays. Actual accessible storage is the caller's responsibility.
 *
 * Validation order:
 * 1. NULL out_layer: INVALID_ARGUMENT; otherwise clear *out_layer.
 * 2. NULL weights/bias or zero input/output size: INVALID_ARGUMENT.
 * 3. Unrepresentable parameter element/byte counts: OVERFLOW.
 * 4. Incorrect weight_count or bias_count: DIMENSION_MISMATCH.
 * 5. Any NaN or infinity in either source array: NONFINITE.
 * 6. Allocate and copy; allocation failure: ALLOCATION with full cleanup.
 * No allocation is attempted until all parameters have been validated.
 */
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

/*
 * Release the layer, both owned tensors, and their arrays. NULL is a no-op.
 * A non-NULL layer must be live and freed exactly once. The caller's pointer
 * is not reset; all aliases and borrowed parameter pointers become invalid.
 */
void cinder_dense_free(cinder_dense_t *layer);

/*
 * Compute output = weights * input, then add bias, with no activation or
 * allocation. Use the existing matvec float accumulation order and add bias
 * afterward. output need not be initialized. All counts are in floats.
 *
 * layer must be live with intact metadata and owned tensors. The caller supplies
 * input_count readable floats and output_capacity writable floats. The full
 * advertised output range must not overlap input, layer/parameter storage, or
 * optional error storage. Exact and partial input/output overlap are forbidden.
 * Read-only input may overlap parameter data. Error storage must not overlap
 * any source or the layer/parameter storage. Overlap is a caller precondition,
 * not diagnosed at runtime; distinct slices of one allocation are allowed if
 * their required ranges do not overlap. Never modify/free the layer during use.
 *
 * Validation order:
 * 1. NULL layer/input/output: INVALID_ARGUMENT.
 * 2. input_count != input_size (including zero): DIMENSION_MISMATCH.
 * 3. output_capacity < output_size (including zero): BUFFER_TOO_SMALL.
 * 4. Any non-finite input, weight, or bias: NONFINITE.
 * All source values, including bias, are checked before any output is written.
 * These validation failures leave the entire output unchanged.
 *
 * Success writes exactly output_size elements; unused capacity is untouched.
 * A non-finite matvec running sum or biased result returns NONFINITE. Earlier
 * results may already be written: discard the entire result on arithmetic
 * failure. In a bias failure, matvec has already written every output element.
 * Layer parameters and input stay unchanged on every outcome; unused output
 * capacity is also unchanged on every outcome. Finite underflow is allowed.
 * A failure does not invalidate the layer; it may be used again.
 *
 * Both status-returning functions follow error.h: optional error details,
 * returned status mirrored in error->code, empty message on success, and a
 * bounded null-terminated message on failure, with no diagnostic allocation.
 */
cinder_status_t cinder_dense_forward(
    const cinder_dense_t *layer,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);

#endif /* CINDER_DENSE_H */
