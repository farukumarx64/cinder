#include "cinder/model.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Keep assertions active in Release builds. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static cinder_layer_t *dense(size_t inputs, size_t outputs,
                            const float *weights, const float *bias)
{
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_dense(inputs, outputs, weights, inputs * outputs,
                                    bias, outputs, &layer, NULL) == CINDER_OK);
    return layer;
}

static cinder_layer_t *activation(cinder_layer_type_t type, size_t size)
{
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_activation(type, size, &layer, NULL) == CINDER_OK);
    return layer;
}

static cinder_model_t *model_from(cinder_layer_t **layers, size_t count)
{
    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, count, &model, NULL) == CINDER_OK);
    return model;
}

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

static void check_run(cinder_model_t *model, const float *input,
                      const float *expected, cinder_error_t *error)
{
    size_t inputs = cinder_model_input_size(model);
    size_t outputs = cinder_model_output_size(model);
    CHECK(inputs <= 4 && outputs <= 4);
    float input_copy[4];
    memcpy(input_copy, input, inputs * sizeof(float));
    /* Test both uninitialized logical output and spare capacity containing NaN. */
    float exact_output[4];
    check_result(cinder_model_run(model, input_copy, inputs, exact_output, outputs, error),
                 CINDER_OK, error);
    float storage[] = {-101, NAN, NAN, NAN, NAN, NAN, -102};
    check_result(cinder_model_run(model, input_copy, inputs, storage + 1, 5, error),
                 CINDER_OK, error);
    for (size_t i = 0; i < outputs; ++i) {
        CHECK(isfinite(exact_output[i]) && isfinite(storage[i + 1]));
        CHECK(fabs((double)exact_output[i] - expected[i]) <= 1e-6);
        CHECK(exact_output[i] == storage[i + 1]);
    }
    for (size_t i = outputs + 1; i <= 5; ++i) {
        CHECK(isnan(storage[i]));
    }
    CHECK(storage[0] == -101 && storage[6] == -102);
    CHECK(memcmp(input_copy, input, inputs * sizeof(float)) == 0);
}

static const float weights1[] = {1, 2, -1, 0, 0.5f, -0.5f, 3, 1};
static const float bias1[] = {0.5f, -1, 2, 0};
static const float weights2[] = {1, 0, -1, 0.5f, 0, 1, 0.5f, -1};
static const float bias2[] = {0, 0.5f};

static void check_example_prefixes(cinder_error_t *error)
{
    const float inputs[][2] = {{2, -1}, {0, 0}, {2, 2}};
    /* Hand-calculated intermediate vectors and two-value Softmax references.
     * Full generated NumPy network fixtures remain a separate roadmap step. */
    const float expected[][4][4] = {
        {{0.5f, -3, 3.5f, 5}, {0.5f, 0, 3.5f, 5}, {-0.5f, -2.75f},
         {0.904650535f, 0.095349465f}},
        {{0.5f, -1, 2, 0}, {0.5f, 0, 2, 0}, {-1.5f, 1.5f},
         {0.047425873f, 0.952574127f}},
        {{6.5f, -3, 2, 8}, {6.5f, 0, 2, 8}, {8.5f, -6.5f},
         {0.999999694f, 0.000000306f}}
    };
    for (size_t count = 1; count <= 4; ++count) {
        cinder_layer_t *layers[] = {
            dense(2, 4, weights1, bias1), activation(CINDER_LAYER_RELU, 4),
            dense(4, 2, weights2, bias2), activation(CINDER_LAYER_SOFTMAX, 2)
        };
        const cinder_dense_t *first = layers[0]->dense;
        const cinder_dense_t *second = count >= 3 ? layers[2]->dense : NULL;
        cinder_model_t *model = model_from(layers, count);
        for (size_t i = count; i < 4; ++i) {
            cinder_layer_free(layers[i]);
        }
        for (size_t call = 0; call < 60; ++call) {
            size_t example = call % COUNT(inputs);
            check_run(model, inputs[example], expected[example][count - 1], error);
        }
        CHECK(memcmp(first->weights->data, weights1, sizeof(weights1)) == 0);
        CHECK(memcmp(first->bias->data, bias1, sizeof(bias1)) == 0);
        if (second != NULL) {
            CHECK(memcmp(second->weights->data, weights2, sizeof(weights2)) == 0);
            CHECK(memcmp(second->bias->data, bias2, sizeof(bias2)) == 0);
        }
        cinder_model_free(model);
    }
}

static void check_other_sequences(cinder_error_t *error)
{
    const float input[] = {-2, 0, 3};
    const float relu[] = {0, 0, 3};
    const float softmax[] = {0.045278501f, 0.045278501f, 0.909442999f};
    cinder_layer_t *activations[] = {
        activation(CINDER_LAYER_RELU, 3), activation(CINDER_LAYER_SOFTMAX, 3),
        activation(CINDER_LAYER_RELU, 3)
    };
    cinder_model_t *model = model_from(activations, 3);
    check_run(model, input, softmax, error);
    cinder_model_free(model);
    cinder_layer_t *single[] = {activation(CINDER_LAYER_RELU, 3)};
    model = model_from(single, 1);
    check_run(model, input, relu, error);
    cinder_model_free(model);
    single[0] = activation(CINDER_LAYER_SOFTMAX, 3);
    model = model_from(single, 1);
    const float extreme[] = {FLT_MAX, -FLT_MAX, FLT_MAX};
    const float extreme_expected[] = {0.5f, 0, 0.5f};
    check_run(model, extreme, extreme_expected, error);
    cinder_model_free(model);

    const float w1[] = {1, 0, 0, 1, 1, 1}, b1[] = {0, 0, 1};
    const float w2[] = {1, 2, 3}, b2[] = {-1};
    const float w3[] = {2, -1}, b3[] = {1, 0.5f};
    cinder_layer_t *consecutive[] = {
        dense(2, 3, w1, b1), dense(3, 1, w2, b2), dense(1, 2, w3, b3)
    };
    model = model_from(consecutive, 3);
    const float dense_input[] = {2, -1}, dense_expected[] = {11, -4.5f};
    check_run(model, dense_input, dense_expected, error);
    cinder_model_free(model);

    const float w[] = {1, 2}, b[] = {0.5f};
    cinder_layer_t *activation_first[] = {
        activation(CINDER_LAYER_RELU, 2), dense(2, 1, w, b),
        activation(CINDER_LAYER_SOFTMAX, 1)
    };
    model = model_from(activation_first, 3);
    const float first_input[] = {-2, 3}, one[] = {1};
    check_run(model, first_input, one, error);
    cinder_model_free(model);
}

static void check_rejected(cinder_model_t *model, const float *input, size_t count,
                           size_t capacity, cinder_status_t expected, cinder_error_t *error)
{
    float output[] = {-101, NAN, -0.0f, 13, 14, -102};
    float saved[COUNT(output)];
    memcpy(saved, output, sizeof(output));
    check_result(cinder_model_run(model, input, count, output + 1, capacity, error),
                 expected, error);
    CHECK(memcmp(saved, output, sizeof(output)) == 0);
}

static void check_validation(cinder_error_t *error)
{
    cinder_layer_t *layers[] = {dense(2, 4, weights1, bias1)};
    cinder_model_t *model = model_from(layers, 1);
    const float input[] = {2, -1};
    check_rejected(NULL, input, 0, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(model, NULL, 0, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_model_run(model, input, 0, NULL, 0, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    const size_t bad_counts[] = {0, 1, 3, SIZE_MAX};
    for (size_t i = 0; i < COUNT(bad_counts); ++i) {
        check_rejected(model, input, bad_counts[i], 0, CINDER_ERROR_DIMENSION_MISMATCH, error);
    }
    for (size_t capacity = 0; capacity < 4; ++capacity) {
        check_rejected(model, input, 2, capacity, CINDER_ERROR_BUFFER_TOO_SMALL, error);
    }
    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t value = 0; value < COUNT(nonfinite); ++value) {
        for (size_t index = 0; index < 2; ++index) {
            float bad[] = {2, -1}, saved[2];
            bad[index] = nonfinite[value];
            memcpy(saved, bad, sizeof(saved));
            check_rejected(model, bad, 0, 0, CINDER_ERROR_DIMENSION_MISMATCH, error);
            check_rejected(model, bad, 2, 0, CINDER_ERROR_BUFFER_TOO_SMALL, error);
            check_rejected(model, bad, 2, 4, CINDER_ERROR_NONFINITE, error);
            CHECK(memcmp(saved, bad, sizeof(bad)) == 0);
        }
    }
    /* Disjoint slices of one allocation are legal; successful retry clears error. */
    float storage[] = {2, -1, 91, 92, 93, 94, NAN};
    check_result(cinder_model_run(model, storage, 2, storage + 2, 5, error), CINDER_OK, error);
    CHECK(storage[0] == 2 && storage[1] == -1 && isnan(storage[6]));
    CHECK(storage[2] == 0.5f && storage[3] == -3 && storage[4] == 3.5f && storage[5] == 5);
    cinder_model_free(model);
}

static void check_numeric_failures(cinder_error_t *error)
{
    const float identity[] = {1, 0, 0, 1}, zero[] = {0, 0};
    /* Overflow at the first, middle, or final layer; move it between output rows.
     * The middle case follows a successful layer and must skip the last layer. */
    for (size_t position = 0; position < 3; ++position) {
        for (size_t row = 0; row < 2; ++row) {
            for (int bias_failure = 0; bias_failure <= 1; ++bias_failure) {
                float w[] = {1, 0, 0, 1}, b[] = {0, 0};
                w[2 * row] = FLT_MAX;
                if (bias_failure) {
                    b[row] = FLT_MAX;
                }
                cinder_layer_t *layers[] = {
                    dense(2, 2, identity, zero), dense(2, 2, identity, zero),
                    dense(2, 2, identity, zero)
                };
                cinder_layer_free(layers[position]);
                layers[position] = dense(2, 2, w, b);
                const float bad[] = {bias_failure ? 1.0f : 2.0f, 0};
                cinder_error_t expected_error;
                float scratch[2];
                /* Obtain the exact lower-level diagnostic before transferring ownership. */
                CHECK(cinder_layer_forward(layers[position], bad, 2, scratch, 2,
                                           &expected_error) == CINDER_ERROR_NONFINITE);
                const cinder_dense_t *parameters = layers[position]->dense;
                cinder_model_t *model = model_from(layers, 3);
                for (size_t call = 0; call < 5; ++call) {
                    check_rejected(model, bad, 2, 4, CINDER_ERROR_NONFINITE, error);
                    if (error != NULL) {
                        CHECK(strcmp(error->message, expected_error.message) == 0);
                    }
                    check_run(model, zero, b, error);
                    CHECK(memcmp(parameters->weights->data, w, sizeof(w)) == 0);
                    CHECK(memcmp(parameters->bias->data, b, sizeof(b)) == 0);
                }
                cinder_model_free(model);
            }
        }
    }
}

static void check_independent_models(cinder_error_t *error)
{
    cinder_layer_t *first_layers[] = {dense(2, 4, weights1, bias1)};
    cinder_layer_t *second_layers[] = {dense(2, 4, weights1, bias1)};
    cinder_model_t *first = model_from(first_layers, 1);
    cinder_model_t *second = model_from(second_layers, 1);
    const float input1[] = {2, -1}, expected1[] = {0.5f, -3, 3.5f, 5};
    const float input2[] = {0, 0};
    for (size_t i = 0; i < 10; ++i) {
        check_run(first, input1, expected1, error);
        check_run(second, input2, bias1, error);
    }
    cinder_model_free(first);
    check_run(second, input1, expected1, error);
    cinder_model_free(second);
}

int main(void)
{
    for (int with_error = 0; with_error <= 1; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        check_example_prefixes(error);
        check_other_sequences(error);
        check_validation(error);
        check_numeric_failures(error);
        check_independent_models(error);
    }
    puts("Model run tests passed: ordered inference, validation, output preservation, and recovery.");
    return EXIT_SUCCESS;
}
