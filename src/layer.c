#include "cinder/layer.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static cinder_status_t layer_result(
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

cinder_status_t cinder_layer_create_dense(
    size_t input_size,
    size_t output_size,
    const float *weights,
    size_t weight_count,
    const float *bias,
    size_t bias_count,
    cinder_layer_t **out_layer,
    cinder_error_t *error
)
{
    if (out_layer == NULL) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Layer output pointer must not be NULL.");
    }
    *out_layer = NULL;

    cinder_dense_t *dense = NULL;
    cinder_status_t status = cinder_dense_create(
        input_size, output_size, weights, weight_count, bias, bias_count,
        &dense, error);
    if (status != CINDER_OK) {
        return status;
    }

    cinder_layer_t *layer = malloc(sizeof(*layer));
    if (layer == NULL) {
        cinder_dense_free(dense);
        return layer_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the Dense layer wrapper.");
    }
    layer->type = CINDER_LAYER_DENSE;
    layer->input_size = dense->input_size;
    layer->output_size = dense->output_size;
    layer->dense = dense;
    *out_layer = layer;
    return layer_result(CINDER_OK, error, "");
}

cinder_status_t cinder_layer_create_activation(
    cinder_layer_type_t type,
    size_t size,
    cinder_layer_t **out_layer,
    cinder_error_t *error
)
{
    if (out_layer == NULL) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Layer output pointer must not be NULL.");
    }
    *out_layer = NULL;
    if ((type != CINDER_LAYER_RELU && type != CINDER_LAYER_SOFTMAX) || size == 0) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Activation type must be ReLU or Softmax and size must be positive.");
    }
    if (size > SIZE_MAX / sizeof(float)) {
        return layer_result(CINDER_ERROR_OVERFLOW, error,
                            "Activation vector byte count exceeds SIZE_MAX.");
    }

    cinder_layer_t *layer = malloc(sizeof(*layer));
    if (layer == NULL) {
        return layer_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the activation layer wrapper.");
    }
    layer->type = type;
    layer->input_size = size;
    layer->output_size = size;
    layer->dense = NULL;
    *out_layer = layer;
    return layer_result(CINDER_OK, error, "");
}

cinder_status_t cinder_layer_validate(
    const cinder_layer_t *layer,
    cinder_error_t *error
)
{
    if (layer == NULL ||
        (layer->type != CINDER_LAYER_DENSE && layer->type != CINDER_LAYER_RELU &&
         layer->type != CINDER_LAYER_SOFTMAX)) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Layer must not be NULL and must have a supported type.");
    }
    if (layer->input_size == 0 || layer->output_size == 0) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Layer input and output sizes must be positive.");
    }
    if (layer->input_size > SIZE_MAX / sizeof(float) ||
        layer->output_size > SIZE_MAX / sizeof(float)) {
        return layer_result(CINDER_ERROR_OVERFLOW, error,
                            "Layer vector byte count exceeds SIZE_MAX.");
    }
    if ((layer->type == CINDER_LAYER_DENSE && layer->dense == NULL) ||
        (layer->type != CINDER_LAYER_DENSE && layer->dense != NULL)) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Dense layers require a payload; activation layers must not have one.");
    }
    if (layer->type == CINDER_LAYER_DENSE) {
        if (layer->input_size != layer->dense->input_size ||
            layer->output_size != layer->dense->output_size) {
            return layer_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                                "Layer dimensions must match the Dense payload.");
        }
    } else if (layer->input_size != layer->output_size) {
        return layer_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                            "Activation layers must preserve the vector size.");
    }
    return layer_result(CINDER_OK, error, "");
}

void cinder_layer_free(cinder_layer_t *layer)
{
    if (layer == NULL) {
        return;
    }
    if (layer->type == CINDER_LAYER_DENSE) {
        cinder_dense_free(layer->dense);
    }
    free(layer);
}

cinder_status_t cinder_layer_forward(
    const cinder_layer_t *layer,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
)
{
    if (layer == NULL || input == NULL || output == NULL) {
        return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Layer, input, and output must not be NULL.");
    }
    cinder_status_t status = cinder_layer_validate(layer, error);
    if (status != CINDER_OK) {
        return status;
    }
    if (input_count != layer->input_size) {
        return layer_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                            "Input length must equal the layer input size.");
    }
    if (output_capacity < layer->output_size) {
        return layer_result(CINDER_ERROR_BUFFER_TOO_SMALL, error,
                            "Output capacity must cover the layer output size.");
    }

    switch (layer->type) {
        case CINDER_LAYER_DENSE:
            return cinder_dense_forward(layer->dense, input, input_count,
                                        output, output_capacity, error);
        case CINDER_LAYER_RELU:
        case CINDER_LAYER_SOFTMAX:
            /* Validate every source value before the first output write. The
             * existing kernels operate in place, on the copied logical vector. */
            for (size_t i = 0; i < input_count; ++i) {
                if (!isfinite(input[i])) {
                    return layer_result(CINDER_ERROR_NONFINITE, error,
                                        "Activation inputs must be finite.");
                }
            }
            memcpy(output, input, input_count * sizeof(*input));
            if (layer->type == CINDER_LAYER_RELU) {
                return cinder_tensor_relu(output, layer->output_size, error);
            }
            return cinder_tensor_softmax(output, layer->output_size, error);
        default:
            return layer_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                                "Layer type is not supported.");
    }
}
