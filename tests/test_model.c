#include "cinder/model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Checks remain active in Release builds. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static const float weights[] = {1, 2, -1, 0, 0.5f, -0.5f, 3, 1};
static const float bias[] = {0.5f, -1, 2, 0};
/* This live model stays owned by main; copies are non-owning output sentinels. */
static cinder_model_t *output_sentinel;

static void check_result(cinder_status_t actual, cinder_status_t expected,
                         const cinder_error_t *error)
{
    CHECK(actual == expected);
    if (error != NULL) {
        CHECK(error->code == expected);
        CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
        CHECK((error->message[0] == '\0') == (expected == CINDER_OK));
    }
}

static cinder_layer_t *activation(cinder_layer_type_t type, size_t size)
{
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_activation(type, size, &layer, NULL) == CINDER_OK);
    return layer;
}

static cinder_layer_t *dense(size_t inputs, size_t outputs)
{
    CHECK(inputs * outputs <= COUNT(weights) && outputs <= COUNT(bias));
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_dense(inputs, outputs, weights, inputs * outputs,
                                    bias, outputs, &layer, NULL) == CINDER_OK);
    return layer;
}

static void check_parameters(const cinder_layer_t *layer, size_t inputs, size_t outputs)
{
    CHECK(layer->type == CINDER_LAYER_DENSE && layer->dense != NULL);
    CHECK(layer->input_size == inputs && layer->output_size == outputs);
    CHECK(layer->dense->input_size == inputs && layer->dense->output_size == outputs);
    CHECK(layer->dense->weights->rows == outputs && layer->dense->weights->cols == inputs);
    CHECK(layer->dense->bias->rows == outputs && layer->dense->bias->cols == 1);
    CHECK(memcmp(layer->dense->weights->data, weights, inputs * outputs * sizeof(float)) == 0);
    CHECK(memcmp(layer->dense->bias->data, bias, outputs * sizeof(float)) == 0);
}

static void check_rejected(cinder_layer_t **layers, size_t available_slots,
                           size_t count, cinder_status_t expected, cinder_error_t *error)
{
    cinder_layer_t *saved[4];
    cinder_layer_t metadata[4];
    CHECK(available_slots <= COUNT(saved));
    if (layers != NULL) {
        memcpy(saved, layers, available_slots * sizeof(*layers));
        for (size_t i = 0; i < available_slots; ++i) {
            if (layers[i] != NULL) {
                metadata[i] = *layers[i];
            }
        }
    }
    cinder_model_t *model = output_sentinel;
    check_result(cinder_model_create(layers, count, &model, error), expected, error);
    CHECK(model == NULL);
    if (layers != NULL) {
        CHECK(memcmp(saved, layers, available_slots * sizeof(*layers)) == 0);
        for (size_t i = 0; i < available_slots; ++i) {
            if (layers[i] != NULL) {
                CHECK(layers[i]->type == metadata[i].type);
                CHECK(layers[i]->input_size == metadata[i].input_size);
                CHECK(layers[i]->output_size == metadata[i].output_size);
                CHECK(layers[i]->dense == metadata[i].dense);
            }
        }
    }
    CHECK(cinder_model_input_size(output_sentinel) == 1);
    CHECK(cinder_model_output_size(output_sentinel) == 1);
    cinder_model_free(model);
}

static void check_arguments(cinder_error_t *error)
{
    cinder_layer_t *layers[] = {activation(CINDER_LAYER_RELU, 3),
                                activation(CINDER_LAYER_SOFTMAX, 3)};
    cinder_layer_t *first = layers[0];
    cinder_layer_t *last = layers[1];
    check_result(cinder_model_create(layers, 2, NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    CHECK(layers[0] == first && layers[1] == last);
    check_rejected(NULL, 0, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(NULL, 0, 2, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(NULL, 0, SIZE_MAX, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(layers, 2, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    /* Oversized counts must fail before any array traversal. */
    check_rejected(layers, 2, SIZE_MAX / sizeof(*layers) + 1, CINDER_ERROR_OVERFLOW, error);
    check_rejected(layers, 2, SIZE_MAX, CINDER_ERROR_OVERFLOW, error);
    layers[0] = NULL;
    check_rejected(layers, 2, 2, CINDER_ERROR_INVALID_ARGUMENT, error);
    layers[0] = first;
    layers[1] = NULL;
    check_rejected(layers, 2, 2, CINDER_ERROR_INVALID_ARGUMENT, error);
    layers[1] = last;
    cinder_model_t *model = NULL;
    check_result(cinder_model_create(layers, 2, &model, error), CINDER_OK, error);
    CHECK(layers[0] == NULL && layers[1] == NULL);
    CHECK(cinder_model_input_size(model) == 3 && cinder_model_output_size(model) == 3);
    cinder_model_free(model);
}

static void check_metadata_and_order(cinder_error_t *error)
{
    cinder_layer_t *layer = activation(CINDER_LAYER_RELU, 3);
    cinder_layer_t *dense_layer = dense(2, 3);
    cinder_layer_t *layers[] = {layer};
    const size_t too_large = SIZE_MAX / sizeof(float) + 1;
    const cinder_layer_t faults[] = {
        {CINDER_LAYER_INVALID, 3, 3, NULL},
        {(cinder_layer_type_t)99, 3, 3, NULL},
        {CINDER_LAYER_RELU, 0, 3, NULL},
        {CINDER_LAYER_SOFTMAX, 3, 0, NULL},
        {CINDER_LAYER_RELU, too_large, 3, NULL},
        {CINDER_LAYER_SOFTMAX, 3, too_large, NULL},
        {CINDER_LAYER_DENSE, 2, 3, NULL},
        {CINDER_LAYER_RELU, 3, 3, dense_layer->dense},
        {CINDER_LAYER_SOFTMAX, 3, 3, dense_layer->dense},
        {CINDER_LAYER_RELU, 3, 2, NULL},
        {CINDER_LAYER_SOFTMAX, 2, 3, NULL},
        {CINDER_LAYER_DENSE, 1, 3, dense_layer->dense},
        {CINDER_LAYER_DENSE, 2, 4, dense_layer->dense}
    };
    for (size_t i = 0; i < COUNT(faults); ++i) {
        /* Test-only metadata corruption; restore before freeing the owner. */
        *layer = faults[i];
        cinder_error_t layer_error;
        cinder_status_t expected = cinder_layer_validate(layer, &layer_error);
        CHECK(expected != CINDER_OK);
        check_rejected(layers, 1, 1, expected, error);
        if (error != NULL) {
            CHECK(strcmp(error->message, layer_error.message) == 0);
        }
        *layer = (cinder_layer_t){CINDER_LAYER_RELU, 3, 3, NULL};
    }

    cinder_layer_t *other = activation(CINDER_LAYER_SOFTMAX, 4);
    cinder_layer_t *sequence[] = {layer, other, NULL};
    /* Later invalid metadata wins over an earlier connection mismatch. */
    check_rejected(sequence, 3, 3, CINDER_ERROR_INVALID_ARGUMENT, error);
    layer->input_size = layer->output_size = too_large;
    check_rejected(sequence, 3, 3, CINDER_ERROR_OVERFLOW, error);
    layer->input_size = layer->output_size = 3;
    /* An earlier duplicate wins over a later metadata overflow. */
    other->input_size = other->output_size = too_large;
    sequence[1] = layer;
    sequence[2] = other;
    check_rejected(sequence, 3, 3, CINDER_ERROR_INVALID_ARGUMENT, error);
    other->input_size = other->output_size = 4;
    check_parameters(dense_layer, 2, 3);
    cinder_layer_free(other);
    cinder_layer_free(dense_layer);
    cinder_layer_free(layer);
}

static void check_connections_and_duplicates(cinder_error_t *error)
{
    /* Put a mismatch at each of the three boundaries in a four-layer model. */
    for (size_t boundary = 1; boundary < 4; ++boundary) {
        cinder_layer_t *layers[4];
        for (size_t i = 0; i < 4; ++i) {
            layers[i] = activation(CINDER_LAYER_RELU, i < boundary ? 3 : 4);
        }
        check_rejected(layers, 4, 4, CINDER_ERROR_DIMENSION_MISMATCH, error);
        for (size_t i = 0; i < 4; ++i) {
            cinder_layer_free(layers[i]);
        }
    }
    cinder_layer_t *owners[4];
    for (size_t i = 0; i < 4; ++i) {
        owners[i] = activation(CINDER_LAYER_RELU, 3);
    }
    for (size_t duplicate = 1; duplicate < 4; ++duplicate) {
        cinder_layer_t *layers[4];
        memcpy(layers, owners, sizeof(layers));
        layers[duplicate] = owners[0];
        check_rejected(layers, 4, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    }
    for (size_t i = 0; i < 4; ++i) {
        cinder_layer_free(owners[i]);
    }
    /* A mismatched Dense/activation pair can be corrected and retried. */
    cinder_layer_t *layers[] = {dense(2, 4), activation(CINDER_LAYER_RELU, 3)};
    check_rejected(layers, 2, 2, CINDER_ERROR_DIMENSION_MISMATCH, error);
    check_parameters(layers[0], 2, 4);
    cinder_layer_free(layers[1]);
    layers[1] = activation(CINDER_LAYER_RELU, 4);
    cinder_model_t *model = NULL;
    check_result(cinder_model_create(layers, 2, &model, error), CINDER_OK, error);
    CHECK(layers[0] == NULL && layers[1] == NULL);
    CHECK(cinder_model_input_size(model) == 2 && cinder_model_output_size(model) == 4);
    cinder_model_free(model);
}

static void check_single_layers(cinder_error_t *error)
{
    const size_t shapes[][2] = {{1, 1}, {1, 4}, {4, 1}, {2, 4}, {4, 2}};
    for (size_t i = 0; i < COUNT(shapes); ++i) {
        cinder_layer_t *layers[] = {dense(shapes[i][0], shapes[i][1])};
        cinder_layer_t *borrowed = layers[0];
        cinder_model_t *model = NULL;
        check_result(cinder_model_create(layers, 1, &model, error), CINDER_OK, error);
        CHECK(layers[0] == NULL);
        CHECK(cinder_model_input_size(model) == shapes[i][0]);
        CHECK(cinder_model_output_size(model) == shapes[i][1]);
        check_parameters(borrowed, shapes[i][0], shapes[i][1]);
        cinder_model_free(model);
    }
    const cinder_layer_type_t types[] = {CINDER_LAYER_RELU, CINDER_LAYER_SOFTMAX};
    const size_t sizes[] = {1, 3, 33};
    for (size_t type = 0; type < COUNT(types); ++type) {
        for (size_t size = 0; size < COUNT(sizes); ++size) {
            cinder_layer_t *layers[] = {activation(types[type], sizes[size])};
            cinder_model_t *model = NULL;
            check_result(cinder_model_create(layers, 1, &model, error), CINDER_OK, error);
            CHECK(layers[0] == NULL);
            CHECK(cinder_model_input_size(model) == sizes[size]);
            CHECK(cinder_model_output_size(model) == sizes[size]);
            cinder_model_free(model);
        }
    }
}

static cinder_model_t *create_example(cinder_layer_t **borrowed_dense, cinder_error_t *error)
{
    cinder_layer_t **layers = malloc(4 * sizeof(*layers));
    CHECK(layers != NULL);
    layers[0] = dense(2, 4);
    layers[1] = activation(CINDER_LAYER_RELU, 4);
    layers[2] = dense(4, 2);
    layers[3] = activation(CINDER_LAYER_SOFTMAX, 2);
    *borrowed_dense = layers[0];
    cinder_model_t *model = NULL;
    check_result(cinder_model_create(layers, 4, &model, error), CINDER_OK, error);
    for (size_t i = 0; i < 4; ++i) {
        CHECK(layers[i] == NULL);
    }
    /* The model must keep its own list, not this caller allocation. */
    free(layers);
    CHECK(cinder_model_input_size(model) == 2 && cinder_model_output_size(model) == 2);
    return model;
}

static void check_independent_models(cinder_error_t *error)
{
    cinder_layer_t *first_dense = NULL;
    cinder_layer_t *second_dense = NULL;
    cinder_model_t *first = create_example(&first_dense, error);
    cinder_model_t *second = create_example(&second_dense, error);
    CHECK(first != second && first_dense != second_dense);
    CHECK(first_dense->dense->weights->data != second_dense->dense->weights->data);
    CHECK(first_dense->dense->bias->data != second_dense->dense->bias->data);
    check_parameters(first_dense, 2, 4);
    check_parameters(second_dense, 2, 4);
    cinder_model_free(first);
    CHECK(cinder_model_input_size(second) == 2 && cinder_model_output_size(second) == 2);
    check_parameters(second_dense, 2, 4);
    cinder_model_free(second);
}

int main(void)
{
    cinder_layer_t *sentinel_layers[] = {activation(CINDER_LAYER_RELU, 1)};
    CHECK(cinder_model_create(sentinel_layers, 1, &output_sentinel, NULL) == CINDER_OK);
    CHECK(cinder_model_input_size(NULL) == 0 && cinder_model_output_size(NULL) == 0);
    cinder_model_free(NULL);
    for (int with_error = 0; with_error <= 1; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        check_arguments(error);
        check_metadata_and_order(error);
        check_connections_and_duplicates(error);
        check_single_layers(error);
        check_independent_models(error);
    }
    cinder_model_free(output_sentinel);
    puts("Model creation tests passed: validation, ownership transfer, sizes, and cleanup.");
    return EXIT_SUCCESS;
}
