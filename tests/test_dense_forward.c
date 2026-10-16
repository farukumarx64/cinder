#include "cinder/dense.h"

/* Reuse the independent, saved float32 Wx + b references from Phase 1. */
#include <numpy_fixtures.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Keep checks active in Release builds too. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static const float example_weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                                        0.5f, -0.5f, 3.0f, 1.0f};
static const float example_bias[] = {0.5f, -1.0f, 2.0f, 0.0f};

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

static void check_parameters(const cinder_dense_t *layer, const float *weights,
                              const float *bias, size_t inputs, size_t outputs)
{
    CHECK(layer->input_size == inputs && layer->output_size == outputs);
    CHECK(layer->weights->rows == outputs && layer->weights->cols == inputs);
    CHECK(layer->bias->rows == outputs && layer->bias->cols == 1);
    CHECK(memcmp(layer->weights->data, weights, inputs * outputs * sizeof(float)) == 0);
    CHECK(memcmp(layer->bias->data, bias, outputs * sizeof(float)) == 0);
}

static cinder_dense_t *create_example(void)
{
    cinder_dense_t *layer = NULL;
    CHECK(cinder_dense_create(2, 4, example_weights, 8, example_bias, 4,
                              &layer, NULL) == CINDER_OK);
    return layer;
}

static void check_repeated_execution(cinder_error_t *error)
{
    cinder_dense_t *layer = create_example();
    cinder_tensor_t *weights_owner = layer->weights;
    cinder_tensor_t *bias_owner = layer->bias;
    float *weights_data = weights_owner->data;
    float *bias_data = bias_owner->data;
    const float inputs[][2] = {{2.0f, -1.0f}, {-1.0f, 2.0f}, {0.0f, 0.0f}};
    const float expected[][4] = {{0.5f, -3.0f, 3.5f, 5.0f},
                                 {3.5f, 0.0f, 0.5f, -1.0f},
                                 {0.5f, -1.0f, 2.0f, 0.0f}};
    float storage[] = {-101.0f, NAN, NAN, NAN, NAN, NAN, -102.0f};
    for (size_t call = 0; call < 100; ++call) {
        size_t example = call % COUNT(inputs);
        float input[2];
        memcpy(input, inputs[example], sizeof(input));
        /* Alternate exact capacity and spare capacity; reuse prior output. */
        size_t capacity = 4 + call % 2;
        check_result(cinder_dense_forward(layer, input, 2, storage + 1, capacity, error),
                     CINDER_OK, error);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(storage[i + 1] == expected[example][i]);
        }
        CHECK(storage[0] == -101.0f && storage[6] == -102.0f);
        CHECK(isnan(storage[5]));
        CHECK(memcmp(input, inputs[example], sizeof(input)) == 0);
        check_parameters(layer, example_weights, example_bias, 2, 4);
        CHECK(layer->weights == weights_owner && layer->bias == bias_owner);
        CHECK(weights_owner->data == weights_data && bias_owner->data == bias_data);
    }
    cinder_dense_free(layer);
}

static void check_numpy_references(void)
{
    for (size_t fixture_index = 0; fixture_index < COUNT(numpy_dense_fixtures);
         ++fixture_index) {
        const numpy_dense_fixture_t *fixture = &numpy_dense_fixtures[fixture_index];
        CHECK(fixture->rows > 0 && fixture->cols > 0);
        CHECK(fixture->weights.count == fixture->rows * fixture->cols);
        CHECK(fixture->input.count == fixture->cols);
        CHECK(fixture->bias.count == fixture->rows && fixture->biased.count == fixture->rows);
        cinder_dense_t *layer = NULL;
        cinder_error_t error;
        check_result(cinder_dense_create(fixture->cols, fixture->rows,
                     fixture->weights.data, fixture->weights.count,
                     fixture->bias.data, fixture->bias.count, &layer, &error),
                     CINDER_OK, &error);
        float *storage = malloc((fixture->rows + 2) * sizeof(*storage));
        CHECK(storage != NULL);
        storage[0] = -101.0f;
        for (size_t i = 1; i < fixture->rows + 2; ++i) {
            storage[i] = NAN;
        }
        check_result(cinder_dense_forward(layer, fixture->input.data, fixture->input.count,
                     storage + 1, fixture->rows + 1, &error), CINDER_OK, &error);
        for (size_t i = 0; i < fixture->rows; ++i) {
            float actual = storage[i + 1];
            float expected = fixture->biased.data[i];
            CHECK(isfinite(actual) && isfinite(expected));
            double difference = fabs((double)actual - (double)expected);
            double tolerance = 1e-5 + 1e-5 * fabs((double)expected);
            if (difference > tolerance) {
                fprintf(stderr, "%s[%zu]: actual %.9g, NumPy %.9g, error %.9g > %.9g\n",
                        fixture->name, i, (double)actual, (double)expected,
                        difference, tolerance);
                exit(EXIT_FAILURE);
            }
        }
        CHECK(storage[0] == -101.0f && isnan(storage[fixture->rows + 1]));
        check_parameters(layer, fixture->weights.data, fixture->bias.data,
                         fixture->cols, fixture->rows);
        free(storage);
        cinder_dense_free(layer);
    }
}

static void check_rejected_call(const cinder_dense_t *layer, const float *input,
                                size_t input_count, size_t capacity,
                                cinder_status_t expected, cinder_error_t *error)
{
    float output[] = {-101.0f, NAN, 12.0f, -13.0f, 14.0f, -102.0f};
    float saved[COUNT(output)];
    memcpy(saved, output, sizeof(output));
    check_result(cinder_dense_forward(layer, input, input_count, output, capacity, error),
                 expected, error);
    CHECK(memcmp(output, saved, sizeof(output)) == 0);
}

static void check_validation(cinder_error_t *error)
{
    cinder_dense_t *layer = create_example();
    const float input[] = {2.0f, -1.0f};
    check_rejected_call(NULL, input, 2, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected_call(layer, NULL, 2, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_dense_forward(layer, input, 2, NULL, 4, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    const size_t wrong_counts[] = {0, 1, 3, SIZE_MAX};
    for (size_t i = 0; i < COUNT(wrong_counts); ++i) {
        check_rejected_call(layer, input, wrong_counts[i], 4,
                            CINDER_ERROR_DIMENSION_MISMATCH, error);
    }
    for (size_t capacity = 0; capacity < 4; ++capacity) {
        check_rejected_call(layer, input, 2, capacity, CINDER_ERROR_BUFFER_TOO_SMALL, error);
    }
    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t value = 0; value < COUNT(nonfinite); ++value) {
        for (size_t index = 0; index < 2; ++index) {
            float bad_input[] = {2.0f, -1.0f};
            bad_input[index] = nonfinite[value];
            float saved[2];
            memcpy(saved, bad_input, sizeof(saved));
            check_rejected_call(layer, bad_input, 2, 4, CINDER_ERROR_NONFINITE, error);
            check_rejected_call(layer, bad_input, 1, 0, CINDER_ERROR_DIMENSION_MISMATCH, error);
            check_rejected_call(layer, bad_input, 2, 0, CINDER_ERROR_BUFFER_TOO_SMALL, error);
            CHECK(memcmp(bad_input, saved, sizeof(saved)) == 0);
        }
        /* Deliberate test-only value corruption probes the finite-source guards.
         * Application callers must treat owned parameters as read-only. */
        for (size_t index = 0; index < 8; ++index) {
            layer->weights->data[index] = nonfinite[value];
            float saved[8];
            memcpy(saved, layer->weights->data, sizeof(saved));
            check_rejected_call(layer, input, 2, 4, CINDER_ERROR_NONFINITE, error);
            check_parameters(layer, saved, example_bias, 2, 4);
            layer->weights->data[index] = example_weights[index];
        }
        for (size_t index = 0; index < 4; ++index) {
            layer->bias->data[index] = nonfinite[value];
            float saved[4];
            memcpy(saved, layer->bias->data, sizeof(saved));
            check_rejected_call(layer, input, 2, 4, CINDER_ERROR_NONFINITE, error);
            check_parameters(layer, example_weights, saved, 2, 4);
            layer->bias->data[index] = example_bias[index];
        }
    }
    /* A successful call after validation failure clears the error details. */
    float output[4];
    check_result(cinder_dense_forward(layer, input, 2, output, 4, error), CINDER_OK, error);
    CHECK(output[0] == 0.5f && output[1] == -3.0f && output[2] == 3.5f && output[3] == 5.0f);
    check_parameters(layer, example_weights, example_bias, 2, 4);
    cinder_dense_free(layer);
}

static void check_arithmetic_failures(cinder_error_t *error)
{
    /* Move overflow through every output row, covering early and late failures. */
    for (size_t row = 0; row < 3; ++row) {
        for (int bias_failure = 0; bias_failure <= 1; ++bias_failure) {
            float weights[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
            float bias[] = {0.5f, 1.0f, 2.0f};
            weights[2 * row] = FLT_MAX;
            weights[2 * row + 1] = bias_failure ? 0.0f : FLT_MAX;
            if (bias_failure) {
                bias[row] = FLT_MAX;
            }
            cinder_dense_t *layer = NULL;
            CHECK(cinder_dense_create(2, 3, weights, 6, bias, 3, &layer, NULL) == CINDER_OK);
            const float inputs[][2] = {{1.0f, 1.0f}, {2.0f, 0.0f}};
            /* For matvec, exercise sum overflow and multiplication overflow. */
            size_t cases = bias_failure ? 1 : COUNT(inputs);
            for (size_t example = 0; example < cases; ++example) {
                float input[2];
                memcpy(input, inputs[example], sizeof(input));
                float output[] = {61.0f, 62.0f, 63.0f, NAN};
                check_result(cinder_dense_forward(layer, input, 2, output, 4, error),
                             CINDER_ERROR_NONFINITE, error);
                for (size_t i = 0; i < 3; ++i) {
                    float expected;
                    if (bias_failure) {
                        expected = i == row ? FLT_MAX : 1.0f;
                        if (i < row) {
                            expected += bias[i];
                        }
                    } else {
                        expected = i < row ? input[0] : 61.0f + (float)i;
                    }
                    CHECK(output[i] == expected);
                }
                CHECK(isnan(output[3]));
                CHECK(memcmp(input, inputs[example], sizeof(input)) == 0);
                check_parameters(layer, weights, bias, 2, 3);

                /* Check propagation against the failing kernel's diagnostic. */
                if (error != NULL) {
                    float scratch[3];
                    cinder_error_t kernel_error;
                    cinder_status_t status = cinder_tensor_matvec(
                        layer->weights, input, 2, scratch, 3, &kernel_error);
                    if (bias_failure) {
                        CHECK(status == CINDER_OK);
                        status = cinder_tensor_add_bias(scratch, 3, bias, 3, &kernel_error);
                    }
                    CHECK(status == CINDER_ERROR_NONFINITE);
                    CHECK(strcmp(error->message, kernel_error.message) == 0);
                }
                const float zero_input[] = {0.0f, 0.0f};
                check_result(cinder_dense_forward(layer, zero_input, 2, output, 4, error),
                             CINDER_OK, error);
                CHECK(memcmp(output, bias, sizeof(bias)) == 0);
                CHECK(isnan(output[3]));
                check_parameters(layer, weights, bias, 2, 3);
            }
            cinder_dense_free(layer);
        }
    }
}

static void check_permitted_storage(cinder_error_t *error)
{
    cinder_dense_t *layer = create_example();
    /* Disjoint input/output slices of a single array are allowed. */
    float storage[] = {2.0f, -1.0f, NAN, NAN, NAN, NAN, NAN};
    check_result(cinder_dense_forward(layer, storage, 2, storage + 2, 5, error),
                 CINDER_OK, error);
    CHECK(storage[0] == 2.0f && storage[1] == -1.0f && isnan(storage[6]));
    CHECK(storage[2] == 0.5f && storage[3] == -3.0f && storage[4] == 3.5f && storage[5] == 5.0f);
    /* Read-only input may borrow owned parameter data. Output remains separate. */
    float output[4];
    check_result(cinder_dense_forward(layer, layer->weights->data, 2, output, 4, error),
                 CINDER_OK, error);
    CHECK(output[0] == 5.5f && output[1] == -2.0f && output[2] == 1.5f && output[3] == 5.0f);
    check_result(cinder_dense_forward(layer, layer->bias->data, 2, output, 4, error),
                 CINDER_OK, error);
    CHECK(output[0] == -1.0f && output[1] == -1.5f && output[2] == 2.75f && output[3] == 0.5f);
    check_parameters(layer, example_weights, example_bias, 2, 4);
    cinder_dense_free(layer);

    /* Finite underflow is accepted, including a subnormal result and zero. */
    const float tiny_weights[] = {FLT_TRUE_MIN, FLT_TRUE_MIN};
    const float zero_bias[] = {0.0f};
    const float tiny_input[] = {1.0f, 0.0f};
    CHECK(cinder_dense_create(2, 1, tiny_weights, 2, zero_bias, 1, &layer, NULL) == CINDER_OK);
    check_result(cinder_dense_forward(layer, tiny_input, 2, output, 1, error), CINDER_OK, error);
    CHECK(output[0] == FLT_TRUE_MIN);
    const float underflow_input[] = {0.25f, 0.0f};
    check_result(cinder_dense_forward(layer, underflow_input, 2, output, 1, error), CINDER_OK, error);
    CHECK(output[0] == 0.0f);
    cinder_dense_free(layer);
}

int main(void)
{
    for (int with_error = 0; with_error <= 1; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        check_repeated_execution(error);
        check_validation(error);
        check_arithmetic_failures(error);
        check_permitted_storage(error);
    }
    check_numpy_references();
    puts("Dense forward tests passed, including 11 saved NumPy fixtures.");
    return EXIT_SUCCESS;
}
