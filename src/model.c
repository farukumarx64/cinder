#include "cinder/model.h"

#include <stdint.h>
#include <stdlib.h>

/* Private representation: callers inspect dimensions through the accessors. */
struct cinder_model {
    cinder_layer_t **layers;
    size_t layer_count;
    size_t input_size;
    size_t output_size;
    size_t buffer_capacity;
    float *buffers[2];
};

static cinder_status_t model_result(
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

cinder_status_t cinder_model_create(
    cinder_layer_t **layers,
    size_t layer_count,
    cinder_model_t **out_model,
    cinder_error_t *error
)
{
    if (out_model == NULL) {
        return model_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Model output pointer must not be NULL.");
    }
    *out_model = NULL;
    if (layers == NULL || layer_count == 0) {
        return model_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                            "Model layers must not be NULL and the sequence must not be empty.");
    }
    if (layer_count > SIZE_MAX / sizeof(*layers)) {
        return model_result(CINDER_ERROR_OVERFLOW, error,
                            "Model layer pointer array byte count exceeds SIZE_MAX.");
    }

    /* Validate all entries before checking connections or allocating anything. */
    for (size_t i = 0; i < layer_count; ++i) {
        cinder_status_t status = cinder_layer_validate(layers[i], error);
        if (status != CINDER_OK) {
            return status;
        }
        for (size_t previous = 0; previous < i; ++previous) {
            if (layers[i] == layers[previous]) {
                return model_result(CINDER_ERROR_INVALID_ARGUMENT, error,
                                    "A model cannot own the same layer more than once.");
            }
        }
    }
    for (size_t i = 1; i < layer_count; ++i) {
        if (layers[i - 1]->output_size != layers[i]->input_size) {
            return model_result(CINDER_ERROR_DIMENSION_MISMATCH, error,
                                "Adjacent layer output and input sizes must match.");
        }
    }

    size_t capacity = 0;
    for (size_t i = 0; i < layer_count; ++i) {
        if (layers[i]->input_size > capacity) {
            capacity = layers[i]->input_size;
        }
        if (layers[i]->output_size > capacity) {
            capacity = layers[i]->output_size;
        }
    }
    /* Layer validation proved every vector's float byte extent fits size_t. */
    size_t buffer_bytes = capacity * sizeof(float);
    cinder_model_t *model = malloc(sizeof(*model));
    if (model == NULL) {
        return model_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the model struct.");
    }
    model->layers = NULL;
    /* Zero adopted layers makes cleanup safe while the caller still owns all. */
    model->layer_count = 0;
    model->input_size = layers[0]->input_size;
    model->output_size = layers[layer_count - 1]->output_size;
    model->buffer_capacity = capacity;
    model->buffers[0] = NULL;
    model->buffers[1] = NULL;

    model->layers = malloc(layer_count * sizeof(*model->layers));
    if (model->layers == NULL) {
        cinder_model_free(model);
        return model_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the model layer pointer array.");
    }
    model->buffers[0] = malloc(buffer_bytes);
    if (model->buffers[0] == NULL) {
        cinder_model_free(model);
        return model_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the first model working buffer.");
    }
    model->buffers[1] = malloc(buffer_bytes);
    if (model->buffers[1] == NULL) {
        cinder_model_free(model);
        return model_result(CINDER_ERROR_ALLOCATION, error,
                            "Could not allocate the second model working buffer.");
    }

    /* No fallible work remains: adopt in order, clear slots, then publish. */
    for (size_t i = 0; i < layer_count; ++i) {
        model->layers[i] = layers[i];
        layers[i] = NULL;
    }
    model->layer_count = layer_count;
    *out_model = model;
    return model_result(CINDER_OK, error, "");
}

size_t cinder_model_input_size(const cinder_model_t *model)
{
    return model == NULL ? 0 : model->input_size;
}

size_t cinder_model_output_size(const cinder_model_t *model)
{
    return model == NULL ? 0 : model->output_size;
}

void cinder_model_free(cinder_model_t *model)
{
    if (model == NULL) {
        return;
    }
    for (size_t i = 0; i < model->layer_count; ++i) {
        cinder_layer_free(model->layers[i]);
    }
    free(model->buffers[1]);
    free(model->buffers[0]);
    free(model->layers);
    free(model);
}
