#include "cinder/layer.h"

#include <numpy_fixtures.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Keep checks active in Release builds. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

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

static cinder_layer_t *create_activation(cinder_layer_type_t type, size_t size)
{
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_activation(type, size, &layer, NULL) == CINDER_OK);
    CHECK(layer != NULL && layer->type == type && layer->dense == NULL);
    CHECK(layer->input_size == size && layer->output_size == size);
    return layer;
}

static void check_activation_creation(cinder_error_t *error)
{
    const cinder_layer_type_t bad_types[] = {
        CINDER_LAYER_INVALID, CINDER_LAYER_DENSE, (cinder_layer_type_t)99
    };
    cinder_layer_t sentinel = {0};
    cinder_layer_t *layer = &sentinel;
    for (size_t i = 0; i < COUNT(bad_types); ++i) {
        check_result(cinder_layer_create_activation(bad_types[i], 3, &layer, error),
                     CINDER_ERROR_INVALID_ARGUMENT, error);
        CHECK(layer == NULL);
        layer = &sentinel;
        /* Unsupported type takes precedence over byte overflow. */
        check_result(cinder_layer_create_activation(bad_types[i], SIZE_MAX, &layer, error),
                     CINDER_ERROR_INVALID_ARGUMENT, error);
        CHECK(layer == NULL);
    }
    const cinder_layer_type_t types[] = {CINDER_LAYER_RELU, CINDER_LAYER_SOFTMAX};
    for (size_t i = 0; i < COUNT(types); ++i) {
        check_result(cinder_layer_create_activation(types[i], 3, NULL, error),
                     CINDER_ERROR_INVALID_ARGUMENT, error);
        layer = &sentinel;
        check_result(cinder_layer_create_activation(types[i], 0, &layer, error),
                     CINDER_ERROR_INVALID_ARGUMENT, error);
        CHECK(layer == NULL);
        layer = &sentinel;
        check_result(cinder_layer_create_activation(types[i], SIZE_MAX / sizeof(float) + 1,
                     &layer, error), CINDER_ERROR_OVERFLOW, error);
        CHECK(layer == NULL);
        /* A wrapper stores dimensions only, so the largest representable size
         * can be constructed/validated without allocating a huge vector. */
        check_result(cinder_layer_create_activation(types[i], SIZE_MAX / sizeof(float),
                     &layer, error), CINDER_OK, error);
        check_result(cinder_layer_validate(layer, error), CINDER_OK, error);
        CHECK(layer->input_size == SIZE_MAX / sizeof(float));
        CHECK(layer->output_size == layer->input_size && layer->dense == NULL);
        cinder_layer_free(layer);
    }
    cinder_layer_free(NULL);
}

static void check_rejected_call(const cinder_layer_t *layer, const float *input,
                                size_t count, size_t capacity,
                                cinder_status_t expected, cinder_error_t *error)
{
    float output[] = {-101.0f, NAN, 12.0f, -13.0f, NAN, -102.0f};
    float saved[COUNT(output)];
    memcpy(saved, output, sizeof(output));
    check_result(cinder_layer_forward(layer, input, count, output + 1, capacity, error),
                 expected, error);
    CHECK(memcmp(output, saved, sizeof(output)) == 0);
}

static void check_metadata(cinder_error_t *error)
{
    const float weights[] = {1, 2, 3, 4, 5, 6};
    const float bias[] = {0, 1, 2};
    cinder_layer_t *dense = NULL;
    CHECK(cinder_layer_create_dense(2, 3, weights, 6, bias, 3, &dense, NULL) == CINDER_OK);
    CHECK(dense->type == CINDER_LAYER_DENSE && dense->dense != NULL);
    check_result(cinder_layer_validate(dense, error), CINDER_OK, error);
    check_result(cinder_layer_validate(NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    const size_t too_large = SIZE_MAX / sizeof(float) + 1;
    const struct {
        cinder_layer_t descriptor;
        cinder_status_t expected;
    } cases[] = {
        {{CINDER_LAYER_INVALID, 3, 3, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{(cinder_layer_type_t)99, 3, 3, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_RELU, 0, 3, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_SOFTMAX, 3, 0, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_RELU, too_large, 3, NULL}, CINDER_ERROR_OVERFLOW},
        {{CINDER_LAYER_SOFTMAX, 3, too_large, NULL}, CINDER_ERROR_OVERFLOW},
        {{CINDER_LAYER_DENSE, 2, 3, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_RELU, 3, 3, dense->dense}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_SOFTMAX, 3, 3, dense->dense}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_DENSE, 1, 3, dense->dense}, CINDER_ERROR_DIMENSION_MISMATCH},
        {{CINDER_LAYER_DENSE, 2, 4, dense->dense}, CINDER_ERROR_DIMENSION_MISMATCH},
        {{CINDER_LAYER_RELU, 3, 2, NULL}, CINDER_ERROR_DIMENSION_MISMATCH},
        {{CINDER_LAYER_SOFTMAX, 2, 3, NULL}, CINDER_ERROR_DIMENSION_MISMATCH},
        /* Check ordering when more than one invariant is broken. */
        {{CINDER_LAYER_INVALID, too_large, 0, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_RELU, 0, too_large, NULL}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_DENSE, too_large, 3, NULL}, CINDER_ERROR_OVERFLOW},
        {{CINDER_LAYER_RELU, 3, 2, dense->dense}, CINDER_ERROR_INVALID_ARGUMENT},
        {{CINDER_LAYER_DENSE, 2, 3, dense->dense}, CINDER_OK},
        {{CINDER_LAYER_RELU, 3, 3, NULL}, CINDER_OK},
        {{CINDER_LAYER_SOFTMAX, 3, 3, NULL}, CINDER_OK}
    };
    cinder_layer_t *activation = create_activation(CINDER_LAYER_RELU, 3);
    const float input[] = {-1, 0, 1};
    for (size_t i = 0; i < COUNT(cases); ++i) {
        check_result(cinder_layer_validate(&cases[i].descriptor, error), cases[i].expected, error);
        if (cases[i].expected != CINDER_OK) {
            /* Deliberate test-only metadata faults probe forward's guard. Restore
             * the constructor-created owner immediately; never free a descriptor. */
            activation->type = cases[i].descriptor.type;
            activation->input_size = cases[i].descriptor.input_size;
            activation->output_size = cases[i].descriptor.output_size;
            activation->dense = cases[i].descriptor.dense;
            check_rejected_call(activation, input, 0, 0, cases[i].expected, error);
            activation->type = CINDER_LAYER_RELU;
            activation->input_size = activation->output_size = 3;
            activation->dense = NULL;
        }
    }
    cinder_layer_free(activation);
    cinder_layer_free(dense);
}

static void check_activation_validation(cinder_layer_type_t type, cinder_error_t *error)
{
    cinder_layer_t *layer = create_activation(type, 3);
    const float input[] = {-1, 0, 1};
    check_rejected_call(NULL, input, 3, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected_call(layer, NULL, 3, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_layer_forward(layer, input, 3, NULL, 4, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    const size_t wrong_counts[] = {0, 1, 2, 4, SIZE_MAX};
    for (size_t i = 0; i < COUNT(wrong_counts); ++i) {
        check_rejected_call(layer, input, wrong_counts[i], 4,
                            CINDER_ERROR_DIMENSION_MISMATCH, error);
    }
    for (size_t capacity = 0; capacity < 3; ++capacity) {
        check_rejected_call(layer, input, 3, capacity, CINDER_ERROR_BUFFER_TOO_SMALL, error);
    }
    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t value = 0; value < COUNT(nonfinite); ++value) {
        for (size_t index = 0; index < 3; ++index) {
            float bad_input[] = {-1, 0, 1};
            bad_input[index] = nonfinite[value];
            float saved[3];
            memcpy(saved, bad_input, sizeof(saved));
            check_rejected_call(layer, bad_input, 3, 4, CINDER_ERROR_NONFINITE, error);
            check_rejected_call(layer, bad_input, 0, 0, CINDER_ERROR_DIMENSION_MISMATCH, error);
            check_rejected_call(layer, bad_input, 3, 0, CINDER_ERROR_BUFFER_TOO_SMALL, error);
            CHECK(memcmp(bad_input, saved, sizeof(saved)) == 0);
        }
    }
    /* Recovery clears stale diagnostics; disjoint slices of one array are legal. */
    float storage[] = {0, 0, 0, NAN, NAN, NAN, NAN};
    check_result(cinder_layer_forward(layer, storage, 3, storage + 3, 4, error), CINDER_OK, error);
    for (size_t i = 0; i < 3; ++i) {
        CHECK(storage[i] == 0.0f);
        CHECK(storage[i + 3] == (type == CINDER_LAYER_RELU ? 0.0f : 1.0f / 3.0f));
    }
    CHECK(isnan(storage[6]));
    CHECK(layer->type == type && layer->input_size == 3 && layer->output_size == 3);
    CHECK(layer->dense == NULL);
    cinder_layer_free(layer);
}

static void check_repeated_dispatch(cinder_error_t *error)
{
    cinder_layer_t *relu = create_activation(CINDER_LAYER_RELU, 3);
    cinder_layer_t *softmax = create_activation(CINDER_LAYER_SOFTMAX, 3);
    const float inputs[][3] = {{-1, 0, 1}, {0, 0, 0}};
    const float relu_expected[][3] = {{0, 0, 1}, {0, 0, 0}};
    const float softmax_expected[][3] = {
        {0.09003057f, 0.24472847f, 0.66524096f}, {1.0f / 3, 1.0f / 3, 1.0f / 3}
    };
    float storage[] = {-101, NAN, NAN, NAN, NAN, -102};
    float saved_spare;
    memcpy(&saved_spare, storage + 4, sizeof(saved_spare));
    for (size_t call = 0; call < 100; ++call) {
        size_t example = call % COUNT(inputs);
        float input[3];
        memcpy(input, inputs[example], sizeof(input));
        size_t capacity = 3 + call % 2;
        check_result(cinder_layer_forward(relu, input, 3, storage + 1, capacity, error),
                     CINDER_OK, error);
        CHECK(memcmp(storage + 1, relu_expected[example], sizeof(input)) == 0);
        check_result(cinder_layer_forward(softmax, input, 3, storage + 1, capacity, error),
                     CINDER_OK, error);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(fabs((double)storage[i + 1] - softmax_expected[example][i]) < 1e-6);
        }
        CHECK(storage[0] == -101 && storage[5] == -102);
        CHECK(memcmp(storage + 4, &saved_spare, sizeof(saved_spare)) == 0);
        CHECK(memcmp(input, inputs[example], sizeof(input)) == 0);
        CHECK(relu->type == CINDER_LAYER_RELU && softmax->type == CINDER_LAYER_SOFTMAX);
        CHECK(relu->input_size == 3 && relu->output_size == 3 && relu->dense == NULL);
        CHECK(softmax->input_size == 3 && softmax->output_size == 3 && softmax->dense == NULL);
    }
    const float signed_input[] = {-0.0f, -FLT_TRUE_MIN, FLT_TRUE_MIN};
    check_result(cinder_layer_forward(relu, signed_input, 3, storage + 1, 4, error), CINDER_OK, error);
    CHECK(storage[1] == 0 && signbit(storage[1]));
    CHECK(storage[2] == 0 && !signbit(storage[2]));
    CHECK(storage[3] == FLT_TRUE_MIN);
    cinder_layer_free(relu);
    cinder_layer_free(softmax);
}

static void check_in_place(cinder_layer_type_t type, cinder_error_t *error)
{
    cinder_layer_t *layer = create_activation(type, 3);
    const float input[] = {-1, 0, 1};
    const float relu[] = {0, 0, 1};
    const float softmax[] = {0.09003057f, 0.24472847f, 0.66524096f};
    const float *expected = type == CINDER_LAYER_RELU ? relu : softmax;
    for (size_t capacity = 3; capacity <= 4; ++capacity) {
        float storage[] = {-101, -1, 0, 1, NAN, -102};
        check_result(cinder_layer_forward(layer, storage + 1, 3, storage + 1,
                                          capacity, error), CINDER_OK, error);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(fabs((double)storage[i + 1] - expected[i]) < 1e-6);
        }
        CHECK(storage[0] == -101 && isnan(storage[4]) && storage[5] == -102);
    }
    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t value = 0; value < COUNT(nonfinite); ++value) {
        for (size_t index = 0; index < 3; ++index) {
            float values[] = {-1, 0, 1, NAN}, saved[4];
            values[index] = nonfinite[value];
            memcpy(saved, values, sizeof(saved));
            check_result(cinder_layer_forward(layer, values, 3, values, 4, error),
                         CINDER_ERROR_NONFINITE, error);
            CHECK(memcmp(values, saved, sizeof(values)) == 0);
        }
    }
    float values[3];
    memcpy(values, input, sizeof(values));
    check_result(cinder_layer_forward(layer, values, 2, values, 3, error),
                 CINDER_ERROR_DIMENSION_MISMATCH, error);
    CHECK(memcmp(values, input, sizeof(values)) == 0);
    check_result(cinder_layer_forward(layer, values, 3, values, 2, error),
                 CINDER_ERROR_BUFFER_TOO_SMALL, error);
    CHECK(memcmp(values, input, sizeof(values)) == 0);
    check_result(cinder_layer_forward(layer, values, 3, values, 3, error), CINDER_OK, error);
    cinder_layer_free(layer);
}

static void compare_fixture(const char *name, const cinder_layer_t *layer,
                             numpy_vector_t input, numpy_vector_t expected)
{
    CHECK(input.count == layer->input_size && expected.count == layer->output_size);
    float *input_copy = malloc(input.count * sizeof(float));
    float *storage = malloc((expected.count + 3) * sizeof(float));
    CHECK(input_copy != NULL && storage != NULL);
    memcpy(input_copy, input.data, input.count * sizeof(float));
    storage[0] = -101;
    for (size_t i = 1; i < expected.count + 2; ++i) {
        storage[i] = NAN;
    }
    storage[expected.count + 2] = -102;
    float saved_spare;
    memcpy(&saved_spare, storage + expected.count + 1, sizeof(saved_spare));
    cinder_error_t error;
    check_result(cinder_layer_forward(layer, input_copy, input.count, storage + 1,
                 expected.count + 1, &error), CINDER_OK, &error);
    double total = 0;
    for (size_t i = 0; i < expected.count; ++i) {
        float actual = storage[i + 1];
        CHECK(isfinite(actual) && isfinite(expected.data[i]));
        double difference = fabs((double)actual - expected.data[i]);
        double tolerance = layer->type == CINDER_LAYER_RELU
            ? 0.0 : 1e-5 + 1e-5 * fabs((double)expected.data[i]);
        if (difference > tolerance) {
            fprintf(stderr, "%s type %d [%zu]: actual %.9g, NumPy %.9g\n",
                    name, (int)layer->type, i, (double)actual, (double)expected.data[i]);
            exit(EXIT_FAILURE);
        }
        if (layer->type == CINDER_LAYER_SOFTMAX) {
            CHECK(actual >= 0 && actual <= 1);
            total += actual;
        }
    }
    if (layer->type == CINDER_LAYER_SOFTMAX) {
        CHECK(fabs(total - 1.0) < 1e-6);
    }
    CHECK(storage[0] == -101 && storage[expected.count + 2] == -102);
    CHECK(memcmp(storage + expected.count + 1, &saved_spare, sizeof(saved_spare)) == 0);
    CHECK(memcmp(input_copy, input.data, input.count * sizeof(float)) == 0);
    CHECK(layer->input_size == input.count && layer->output_size == expected.count);
    check_result(cinder_layer_validate(layer, &error), CINDER_OK, &error);
    free(input_copy);
    free(storage);
}

static void check_numpy_references(void)
{
    for (size_t i = 0; i < COUNT(numpy_activation_fixtures); ++i) {
        const numpy_activation_fixture_t *fixture = &numpy_activation_fixtures[i];
        cinder_layer_t *relu = create_activation(CINDER_LAYER_RELU, fixture->input.count);
        cinder_layer_t *softmax = create_activation(CINDER_LAYER_SOFTMAX, fixture->input.count);
        compare_fixture(fixture->name, relu, fixture->input, fixture->relu);
        compare_fixture(fixture->name, softmax, fixture->input, fixture->softmax);
        cinder_layer_free(relu);
        cinder_layer_free(softmax);
    }
    for (size_t i = 0; i < COUNT(numpy_dense_fixtures); ++i) {
        const numpy_dense_fixture_t *fixture = &numpy_dense_fixtures[i];
        cinder_layer_t *layer = NULL;
        CHECK(cinder_layer_create_dense(fixture->cols, fixture->rows,
              fixture->weights.data, fixture->weights.count, fixture->bias.data,
              fixture->bias.count, &layer, NULL) == CINDER_OK);
        compare_fixture(fixture->name, layer, fixture->input, fixture->biased);
        CHECK(layer->type == CINDER_LAYER_DENSE);
        CHECK(memcmp(layer->dense->weights->data, fixture->weights.data,
                     fixture->weights.count * sizeof(float)) == 0);
        CHECK(memcmp(layer->dense->bias->data, fixture->bias.data,
                     fixture->bias.count * sizeof(float)) == 0);
        cinder_layer_free(layer);
    }
}

static void check_dense_ownership_and_errors(cinder_error_t *error)
{
    float weights[] = {1, 0, FLT_MAX, 0, 0, 1};
    float bias[] = {0.5f, 0, -1};
    cinder_layer_t *layer = NULL;
    check_result(cinder_layer_create_dense(2, 3, weights, 6, bias, 3, &layer, error),
                 CINDER_OK, error);
    CHECK(layer->dense->weights->data != weights && layer->dense->bias->data != bias);
    weights[0] = 99;
    bias[0] = 99;
    CHECK(layer->dense->weights->data[0] == 1 && layer->dense->bias->data[0] == 0.5f);
    const float input[] = {2, 2};
    float output[] = {61, 62, 63, NAN};
    check_result(cinder_layer_forward(layer, input, 2, output, 4, error),
                 CINDER_ERROR_NONFINITE, error);
    CHECK(output[0] == 2 && output[1] == 62 && output[2] == 63 && isnan(output[3]));
    if (error != NULL) {
        float scratch[3];
        cinder_error_t dense_error;
        CHECK(cinder_dense_forward(layer->dense, input, 2, scratch, 3, &dense_error)
              == CINDER_ERROR_NONFINITE);
        CHECK(strcmp(error->message, dense_error.message) == 0);
    }
    const float zero[] = {0, 0};
    check_result(cinder_layer_forward(layer, zero, 2, output, 4, error), CINDER_OK, error);
    CHECK(output[0] == 0.5f && output[1] == 0 && output[2] == -1 && isnan(output[3]));
    CHECK(layer->dense->weights->data[2] == FLT_MAX);
    cinder_layer_free(layer);
}

int main(void)
{
    for (int with_error = 0; with_error <= 1; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        check_activation_creation(error);
        check_metadata(error);
        check_activation_validation(CINDER_LAYER_RELU, error);
        check_activation_validation(CINDER_LAYER_SOFTMAX, error);
        check_repeated_dispatch(error);
        check_in_place(CINDER_LAYER_RELU, error);
        check_in_place(CINDER_LAYER_SOFTMAX, error);
        check_dense_ownership_and_errors(error);
    }
    check_numpy_references();
    puts("Layer tests passed: dispatch, validation, ownership, and 35 saved NumPy vectors.");
    return EXIT_SUCCESS;
}
