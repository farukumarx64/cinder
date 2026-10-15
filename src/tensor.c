#include "cinder/tensor.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

_Static_assert(CHAR_BIT == 8 && sizeof(float) == 4,
               "Cinder requires 32-bit floats and 8-bit bytes.");
_Static_assert(FLT_RADIX == 2 && FLT_MANT_DIG == 24 &&
               FLT_MIN_EXP == -125 && FLT_MAX_EXP == 128 &&
               FLT_HAS_SUBNORM == 1,
               "Cinder requires the precision and range of IEEE 754 binary32.");

static cinder_status_t tensor_result(
    cinder_status_t status,
    cinder_error_t *error,
    const char *message
)
{
    if (error != NULL) {
        size_t i = 0;
        error->code = status;
        while (i < sizeof(error->message) - 1 && message[i] != '\0') {
            error->message[i] = message[i];
            ++i;
        }
        error->message[i] = '\0';
    }
    return status;
}

cinder_status_t cinder_tensor_create(
    size_t rows,
    size_t cols,
    cinder_tensor_t **out_tensor,
    cinder_error_t *error
)
{
    if (out_tensor == NULL) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Tensor output pointer must not be NULL.");
    }
    *out_tensor = NULL;

    if (rows == 0 || cols == 0) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Tensor rows and columns must be greater than zero.");
    }
    if (rows > SIZE_MAX / cols) {
        return tensor_result(CINDER_ERROR_OVERFLOW, error,
                             "Tensor element count exceeds SIZE_MAX.");
    }
    size_t count = rows * cols;
    if (count > SIZE_MAX / sizeof(float)) {
        return tensor_result(CINDER_ERROR_OVERFLOW, error,
                             "Tensor allocation byte count exceeds SIZE_MAX.");
    }

    cinder_tensor_t *tensor = malloc(sizeof(*tensor));
    if (tensor == NULL) {
        return tensor_result(CINDER_ERROR_ALLOCATION, error,
                             "Could not allocate the tensor struct.");
    }
    tensor->data = calloc(count, sizeof(*tensor->data));
    if (tensor->data == NULL) {
        free(tensor);
        return tensor_result(CINDER_ERROR_ALLOCATION, error,
                             "Could not allocate the tensor elements.");
    }

    tensor->rows = rows;
    tensor->cols = cols;
    /* Publish ownership only after both allocations succeed. */
    *out_tensor = tensor;
    return tensor_result(CINDER_OK, error, "");
}

void cinder_tensor_free(cinder_tensor_t *tensor)
{
    if (tensor == NULL) {
        return;
    }
    free(tensor->data);
    free(tensor);
}

cinder_status_t cinder_tensor_fill(
    cinder_tensor_t *tensor,
    float value,
    cinder_error_t *error
)
{
    if (tensor == NULL) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Tensor must not be NULL.");
    }
    if (!isfinite(value)) {
        return tensor_result(CINDER_ERROR_NONFINITE, error,
                             "Tensor fill value must be finite.");
    }

    /* Creation validated this product; live tensor metadata must stay intact. */
    size_t count = tensor->rows * tensor->cols;
    for (size_t i = 0; i < count; ++i) {
        tensor->data[i] = value;
    }
    return tensor_result(CINDER_OK, error, "");
}

cinder_status_t cinder_tensor_get(
    const cinder_tensor_t *tensor,
    size_t row,
    size_t col,
    float *out_value,
    cinder_error_t *error
)
{
    if (tensor == NULL || out_value == NULL) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Tensor and output value pointers must not be NULL.");
    }
    if (row >= tensor->rows || col >= tensor->cols) {
        return tensor_result(CINDER_ERROR_OUT_OF_BOUNDS, error,
                             "Tensor row or column index is out of bounds.");
    }

    float value = tensor->data[row * tensor->cols + col];
    if (!isfinite(value)) {
        return tensor_result(CINDER_ERROR_NONFINITE, error,
                             "Tensor element must be finite.");
    }
    /* Preserve the caller's output until every check has passed. */
    *out_value = value;
    return tensor_result(CINDER_OK, error, "");
}

cinder_status_t cinder_tensor_set(
    cinder_tensor_t *tensor,
    size_t row,
    size_t col,
    float value,
    cinder_error_t *error
)
{
    if (tensor == NULL) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Tensor must not be NULL.");
    }
    if (row >= tensor->rows || col >= tensor->cols) {
        return tensor_result(CINDER_ERROR_OUT_OF_BOUNDS, error,
                             "Tensor row or column index is out of bounds.");
    }
    if (!isfinite(value)) {
        return tensor_result(CINDER_ERROR_NONFINITE, error,
                             "Tensor element value must be finite.");
    }

    tensor->data[row * tensor->cols + col] = value;
    return tensor_result(CINDER_OK, error, "");
}

cinder_status_t cinder_tensor_matvec(
    const cinder_tensor_t *weights,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
)
{
    if (weights == NULL || input == NULL || output == NULL) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Weights, input, and output pointers must not be NULL.");
    }
    if (input_count != weights->cols) {
        return tensor_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                             "Input length must equal the number of weight columns.");
    }
    if (output_capacity < weights->rows) {
        return tensor_result(CINDER_ERROR_BUFFER_TOO_SMALL, error,
                             "Output capacity must cover every weight row.");
    }

    /* Validate all source values before writing any output. */
    for (size_t col = 0; col < input_count; ++col) {
        if (!isfinite(input[col])) {
            return tensor_result(CINDER_ERROR_NONFINITE, error,
                                 "Matrix-vector input values must be finite.");
        }
    }
    size_t count = weights->rows * weights->cols;
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(weights->data[i])) {
            return tensor_result(CINDER_ERROR_NONFINITE, error,
                                 "Matrix-vector weights must be finite.");
        }
    }

    for (size_t row = 0; row < weights->rows; ++row) {
        float sum = 0.0f;
        size_t row_offset = row * weights->cols;
        for (size_t col = 0; col < weights->cols; ++col) {
            sum += weights->data[row_offset + col] * input[col];
            if (!isfinite(sum)) {
                return tensor_result(CINDER_ERROR_NONFINITE, error,
                                     "Matrix-vector accumulation became non-finite.");
            }
        }
        /* Commit one completed row; later arithmetic failures do not undo it. */
        output[row] = sum;
    }
    return tensor_result(CINDER_OK, error, "");
}

cinder_status_t cinder_tensor_add_bias(
    float *output,
    size_t output_count,
    const float *bias,
    size_t bias_count,
    cinder_error_t *error
)
{
    if (output == NULL || bias == NULL) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Output and bias pointers must not be NULL.");
    }
    if (output_count == 0 || bias_count == 0) {
        return tensor_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                             "Output and bias vectors must not be empty.");
    }
    if (output_count != bias_count) {
        return tensor_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                             "Bias length must equal the output vector length.");
    }
    if (output_count > SIZE_MAX / sizeof(*output)) {
        return tensor_result(CINDER_ERROR_OVERFLOW, error,
                             "Bias addition vector byte count exceeds SIZE_MAX.");
    }

    /* Check both vectors completely before modifying the multiplication result. */
    for (size_t i = 0; i < output_count; ++i) {
        if (!isfinite(output[i]) || !isfinite(bias[i])) {
            return tensor_result(CINDER_ERROR_NONFINITE, error,
                                 "Output and bias values must be finite.");
        }
    }
    for (size_t i = 0; i < output_count; ++i) {
        float sum = output[i] + bias[i];
        if (!isfinite(sum)) {
            return tensor_result(CINDER_ERROR_NONFINITE, error,
                                 "Bias addition produced a non-finite sum.");
        }
        /* Store only a valid sum; earlier updates remain if a later one fails. */
        output[i] = sum;
    }
    return tensor_result(CINDER_OK, error, "");
}
