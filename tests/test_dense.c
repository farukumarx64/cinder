#include "cinder/dense.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Keep checks active in Release builds, where assert() would be disabled. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void check_error(cinder_status_t status, const cinder_error_t *error)
{
    CHECK(error->code == status);
    CHECK(error->message[0] != '\0');
    CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
}

static void check_rejected(
    size_t input_size, size_t output_size,
    const float *weights, size_t weight_count,
    const float *bias, size_t bias_count,
    cinder_status_t expected
)
{
    /* A non-owning sentinel checks that creation clears the output slot. */
    cinder_dense_t sentinel = {0};
    cinder_dense_t *layer = &sentinel;
    cinder_error_t error;
    memset(&error, 0x7f, sizeof(error));
    CHECK(cinder_dense_create(input_size, output_size, weights, weight_count,
                              bias, bias_count, &layer, &error) == expected);
    CHECK(layer == NULL);
    check_error(expected, &error);

    layer = &sentinel;
    CHECK(cinder_dense_create(input_size, output_size, weights, weight_count,
                              bias, bias_count, &layer, NULL) == expected);
    CHECK(layer == NULL);
}

static void check_parameters(
    const cinder_dense_t *layer,
    size_t input_size, size_t output_size,
    const float *weights, const float *bias
)
{
    CHECK(layer != NULL);
    CHECK(layer->input_size == input_size);
    CHECK(layer->output_size == output_size);
    CHECK(layer->weights != NULL && layer->bias != NULL);
    CHECK(layer->weights != layer->bias);
    CHECK(layer->weights->rows == output_size);
    CHECK(layer->weights->cols == input_size);
    CHECK(layer->bias->rows == output_size && layer->bias->cols == 1);
    CHECK(layer->weights->data != NULL && layer->bias->data != NULL);
    CHECK(layer->weights->data != layer->bias->data);
    CHECK(memcmp(layer->weights->data, weights,
                 input_size * output_size * sizeof(*weights)) == 0);
    CHECK(memcmp(layer->bias->data, bias, output_size * sizeof(*bias)) == 0);
}

static void check_valid_shape(size_t input_size, size_t output_size)
{
    const float weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                             0.5f, -0.5f, 3.0f, 1.0f};
    const float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    CHECK(input_size * output_size <= sizeof(weights) / sizeof(weights[0]));
    CHECK(output_size <= sizeof(bias) / sizeof(bias[0]));
    cinder_dense_t *layer = NULL;
    cinder_error_t error;
    /* Success must clear details left behind by a failed call. */
    CHECK(cinder_dense_create(0, output_size, weights, 0, bias, output_size,
                              &layer, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_dense_create(input_size, output_size, weights,
                              input_size * output_size, bias, output_size,
                              &layer, &error) == CINDER_OK);
    CHECK(error.code == CINDER_OK && error.message[0] == '\0');
    check_parameters(layer, input_size, output_size, weights, bias);
    CHECK(layer->weights->data != weights && layer->bias->data != bias);
    /* Confirm that the copied matrix uses the tensor API's row-major layout. */
    for (size_t row = 0; row < output_size; ++row) {
        for (size_t col = 0; col < input_size; ++col) {
            float value = 0.0f;
            CHECK(cinder_tensor_get(layer->weights, row, col, &value, NULL) == CINDER_OK);
            CHECK(value == weights[row * input_size + col]);
        }
    }
    cinder_dense_free(layer);
}

static void check_owned_copies(void)
{
    const float expected_weights[] = {1.0f, 2.0f, -1.0f, 0.0f,
                                      0.5f, -0.5f, 3.0f, 1.0f};
    const float expected_bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    float *weights = malloc(sizeof(expected_weights));
    float *bias = malloc(sizeof(expected_bias));
    CHECK(weights != NULL && bias != NULL);
    memcpy(weights, expected_weights, sizeof(expected_weights));
    memcpy(bias, expected_bias, sizeof(expected_bias));
    cinder_dense_t *first = NULL;
    cinder_dense_t *second = NULL;
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, &first, NULL) == CINDER_OK);
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, &second, NULL) == CINDER_OK);
    CHECK(memcmp(weights, expected_weights, sizeof(expected_weights)) == 0);
    CHECK(memcmp(bias, expected_bias, sizeof(expected_bias)) == 0);
    CHECK(first != second);
    CHECK(first->weights != second->weights && first->bias != second->bias);
    CHECK(first->weights->data != second->weights->data);
    CHECK(first->bias->data != second->bias->data);
    CHECK(first->weights->data != weights && second->weights->data != weights);
    CHECK(first->bias->data != bias && second->bias->data != bias);

    /* Changes and deallocation of the original arrays cannot affect a layer. */
    for (size_t i = 0; i < 8; ++i) {
        weights[i] = 99.0f;
    }
    for (size_t i = 0; i < 4; ++i) {
        bias[i] = -99.0f;
    }
    check_parameters(first, 2, 4, expected_weights, expected_bias);
    check_parameters(second, 2, 4, expected_weights, expected_bias);
    free(bias);
    free(weights);
    check_parameters(first, 2, 4, expected_weights, expected_bias);
    check_parameters(second, 2, 4, expected_weights, expected_bias);
    cinder_dense_free(first);
    /* Freeing one layer must not invalidate another created from the same data. */
    check_parameters(second, 2, 4, expected_weights, expected_bias);
    cinder_dense_free(second);
}

static void check_special_values_and_overlap(void)
{
    const float values[] = {0.0f, -0.0f, FLT_MAX, -FLT_MAX,
                            FLT_MIN, -FLT_MIN, FLT_TRUE_MIN, -FLT_TRUE_MIN};
    float shared[8];
    memcpy(shared, values, sizeof(values));
    cinder_dense_t *layer = NULL;
    /* Constructor sources may overlap, including sharing the very same base. */
    for (size_t offset = 0; offset <= 4; ++offset) {
        CHECK(cinder_dense_create(2, 4, shared, 8, shared + offset, 4,
                                  &layer, NULL) == CINDER_OK);
        check_parameters(layer, 2, 4, values, values + offset);
        CHECK(memcmp(shared, values, sizeof(values)) == 0);
        CHECK(!signbit(layer->weights->data[0]));
        CHECK(signbit(layer->weights->data[1]));
        cinder_dense_free(layer);
    }
}

int main(void)
{
    float weights[8] = {1.0f, 2.0f, -1.0f, 0.0f, 0.5f, -0.5f, 3.0f, 1.0f};
    float bias[4] = {0.5f, -1.0f, 2.0f, 0.0f};
    float original_weights[8];
    float original_bias[4];
    memcpy(original_weights, weights, sizeof(weights));
    memcpy(original_bias, bias, sizeof(bias));
    cinder_error_t error;
    memset(&error, 0x7f, sizeof(error));
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, NULL, &error)
          == CINDER_ERROR_INVALID_ARGUMENT);
    check_error(CINDER_ERROR_INVALID_ARGUMENT, &error);
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, NULL, NULL)
          == CINDER_ERROR_INVALID_ARGUMENT);

    check_rejected(2, 4, NULL, 8, bias, 4, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected(2, 4, weights, 8, NULL, 4, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected(0, 0, weights, 0, bias, 0, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected(0, 4, weights, 0, bias, 4, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected(2, 0, weights, 0, bias, 0, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected(SIZE_MAX, 0, weights, 0, bias, 0, CINDER_ERROR_INVALID_ARGUMENT);

    /* These requests must be rejected before reading the small source arrays. */
    check_rejected(SIZE_MAX, 2, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW);
    check_rejected(2, SIZE_MAX, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW);
    check_rejected(SIZE_MAX / 2 + 1, 2, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW);
    check_rejected(2, SIZE_MAX / 2 + 1, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW);
    check_rejected(SIZE_MAX / sizeof(float) + 1, 1, weights, 0, bias, 0,
                   CINDER_ERROR_OVERFLOW);
    check_rejected(1, SIZE_MAX / sizeof(float) + 1, weights, 0, bias, 0,
                   CINDER_ERROR_OVERFLOW);
    /* The byte-safe boundary passes overflow checks, then fails exact counts. */
    check_rejected(SIZE_MAX / sizeof(float), 1, weights, 0, bias, 0,
                   CINDER_ERROR_DIMENSION_MISMATCH);
    check_rejected(1, SIZE_MAX / sizeof(float), weights, 0, bias, 0,
                   CINDER_ERROR_DIMENSION_MISMATCH);

    const size_t wrong_weights[] = {0, 7, 9, SIZE_MAX};
    const size_t wrong_bias[] = {0, 3, 5, SIZE_MAX};
    for (size_t i = 0; i < 4; ++i) {
        check_rejected(2, 4, weights, wrong_weights[i], bias, 4,
                       CINDER_ERROR_DIMENSION_MISMATCH);
        check_rejected(2, 4, weights, 8, bias, wrong_bias[i],
                       CINDER_ERROR_DIMENSION_MISMATCH);
    }
    CHECK(memcmp(weights, original_weights, sizeof(weights)) == 0);
    CHECK(memcmp(bias, original_bias, sizeof(bias)) == 0);

    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t value = 0; value < 3; ++value) {
        for (size_t i = 0; i < 8; ++i) {
            float saved[8];
            weights[i] = nonfinite[value];
            memcpy(saved, weights, sizeof(saved));
            check_rejected(2, 4, weights, 8, bias, 4, CINDER_ERROR_NONFINITE);
            /* Length checks take precedence over scanning non-finite sources. */
            check_rejected(2, 4, weights, 7, bias, 4, CINDER_ERROR_DIMENSION_MISMATCH);
            CHECK(memcmp(weights, saved, sizeof(weights)) == 0);
            CHECK(memcmp(bias, original_bias, sizeof(bias)) == 0);
            weights[i] = original_weights[i];
        }
        for (size_t i = 0; i < 4; ++i) {
            float saved[4];
            bias[i] = nonfinite[value];
            memcpy(saved, bias, sizeof(saved));
            check_rejected(2, 4, weights, 8, bias, 4, CINDER_ERROR_NONFINITE);
            CHECK(memcmp(bias, saved, sizeof(bias)) == 0);
            CHECK(memcmp(weights, original_weights, sizeof(weights)) == 0);
            bias[i] = original_bias[i];
        }
    }

    check_valid_shape(1, 1);
    check_valid_shape(2, 4);
    check_valid_shape(3, 2);
    check_valid_shape(2, 3);
    check_valid_shape(4, 1);
    check_valid_shape(1, 4);
    check_owned_copies();
    check_special_values_and_overlap();
    for (size_t i = 0; i < 100; ++i) {
        check_valid_shape(2, 4);
    }
    cinder_dense_free(NULL);
    puts("Dense allocation tests passed.");
    return EXIT_SUCCESS;
}
