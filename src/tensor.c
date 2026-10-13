#include "cinder/tensor.h"

#include <float.h>
#include <limits.h>
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
