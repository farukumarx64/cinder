#include "cinder/dense.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static cinder_status_t dense_result(
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

cinder_status_t cinder_dense_create(
    size_t input_size,
    size_t output_size,
    const float *weights,
    size_t weight_count,
    const float *bias,
    size_t bias_count,
    cinder_dense_t **out_layer,
    cinder_error_t *error
)
{
    if (out_layer == NULL) {
        return dense_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Dense output pointer must not be NULL.");
    }
    *out_layer = NULL;

    if (weights == NULL || bias == NULL || input_size == 0 || output_size == 0) {
        return dense_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Dense parameters must not be NULL and sizes must be positive.");
    }
    if (output_size > SIZE_MAX / input_size) {
        return dense_result(CINDER_ERROR_OVERFLOW, error,
                            "Dense weight element count exceeds SIZE_MAX.");
    }
    size_t required_weights = output_size * input_size;
    if (required_weights > SIZE_MAX / sizeof(float) ||
        output_size > SIZE_MAX / sizeof(float)) {
        return dense_result(CINDER_ERROR_OVERFLOW, error,
                            "Dense parameter allocation byte count exceeds SIZE_MAX.");
    }
    if (weight_count != required_weights || bias_count != output_size) {
        return dense_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                            "Dense parameter counts must exactly match the layer dimensions.");
    }

    /* Validate both sources before attempting any allocation or copy. */
    for (size_t i = 0; i < weight_count; ++i) {
        if (!isfinite(weights[i])) {
            return dense_result(CINDER_ERROR_NONFINITE, error,
                                "Dense weights must be finite.");
        }
    }
    for (size_t i = 0; i < bias_count; ++i) {
        if (!isfinite(bias[i])) {
            return dense_result(CINDER_ERROR_NONFINITE, error,
                                "Dense biases must be finite.");
        }
    }

    cinder_dense_t *layer = malloc(sizeof(*layer));
    if (layer == NULL) {
        return dense_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the Dense layer struct.");
    }
    layer->input_size = input_size;
    layer->output_size = output_size;
    /* NULL owners let the same destructor clean up every partial state. */
    layer->weights = NULL;
    layer->bias = NULL;

    cinder_status_t status = cinder_tensor_create(
        output_size, input_size, &layer->weights, error);
    if (status != CINDER_OK) {
        cinder_dense_free(layer);
        return status; /* Preserve the tensor allocator's error details. */
    }
    status = cinder_tensor_create(output_size, 1, &layer->bias, error);
    if (status != CINDER_OK) {
        cinder_dense_free(layer);
        return status;
    }

    memcpy(layer->weights->data, weights, weight_count * sizeof(*weights));
    memcpy(layer->bias->data, bias, bias_count * sizeof(*bias));
    /* Publish ownership only when both tensors and their copies are complete. */
    *out_layer = layer;
    return dense_result(CINDER_OK, error, "");
}

void cinder_dense_free(cinder_dense_t *layer)
{
    if (layer == NULL) {
        return;
    }
    cinder_tensor_free(layer->bias);
    cinder_tensor_free(layer->weights);
    free(layer);
}

cinder_status_t cinder_dense_forward(
    const cinder_dense_t *layer,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
)
{
    if (layer == NULL || input == NULL || output == NULL) {
        return dense_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Dense layer, input, and output must not be NULL.");
    }
    if (input_count != layer->input_size) {
        return dense_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                            "Dense input length must equal the layer input size.");
    }
    if (output_capacity < layer->output_size) {
        return dense_result(CINDER_ERROR_BUFFER_TOO_SMALL, error,
                            "Dense output capacity must cover the layer output size.");
    }

    /* Matvec checks input and weights before writing. Check bias here so all
     * sources are validated before even the first multiplication result. */
    for (size_t i = 0; i < layer->output_size; ++i) {
        if (!isfinite(layer->bias->data[i])) {
            return dense_result(CINDER_ERROR_NONFINITE, error,
                                "Dense biases must be finite.");
        }
    }
    cinder_status_t status = cinder_tensor_matvec(
        layer->weights, input, input_count, output, output_capacity, error);
    if (status != CINDER_OK) {
        return status; /* Preserve the kernel's error and any partial output. */
    }
    /* Only the logical output participates in bias addition, not spare space. */
    return cinder_tensor_add_bias(output, layer->output_size,
                                  layer->bias->data, layer->output_size, error);
}
