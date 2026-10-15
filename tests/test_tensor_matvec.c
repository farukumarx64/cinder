#include "cinder/tensor.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Keep checks active in Release builds too. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static void check_result(
    cinder_status_t actual,
    cinder_status_t expected,
    const cinder_error_t *error
)
{
    CHECK(actual == expected);
    if (error != NULL) {
        CHECK(error->code == expected);
        CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
        CHECK((error->message[0] == '\0') == (expected == CINDER_OK));
    }
}

static void check_close(float actual, float expected)
{
    CHECK(isfinite(actual) && isfinite(expected));
    double difference = (double)actual - (double)expected;
    if (difference < 0.0) {
        difference = -difference;
    }
    double magnitude = expected < 0.0f ? -(double)expected : (double)expected;
    CHECK(difference <= 1e-5 + 1e-5 * magnitude);
}

static void check_product(
    size_t rows,
    size_t cols,
    const float *matrix,
    const float *vector,
    const float *expected,
    cinder_error_t *error
)
{
    /* Fixed test buffers cover every fixture below, including guard elements. */
    CHECK(rows <= 4 && cols <= 4);
    cinder_tensor_t *weights = NULL;
    CHECK(cinder_tensor_create(rows, cols, &weights, NULL) == CINDER_OK);
    float *original_data = weights->data;
    memcpy(weights->data, matrix, rows * cols * sizeof(float));
    float input[4];
    memcpy(input, vector, cols * sizeof(float));

    for (size_t extra = 0; extra <= 1; ++extra) {
        float storage[6] = {-123.0f, -123.0f, -123.0f, -123.0f, -123.0f, -123.0f};
        /* An error followed by success must clear optional error details. */
        check_result(cinder_tensor_matvec(weights, input, cols, storage + 1,
                                         rows - 1, error),
                     CINDER_ERROR_BUFFER_TOO_SMALL, error);
        check_result(cinder_tensor_matvec(weights, input, cols, storage + 1,
                                         rows + extra, error),
                     CINDER_OK, error);
        CHECK(storage[0] == -123.0f);
        for (size_t row = 0; row < rows; ++row) {
            check_close(storage[row + 1], expected[row]);
            /* Do not let the tolerance hide a zero or subnormal mistake. */
            if (expected[row] == 0.0f || expected[row] == FLT_TRUE_MIN) {
                CHECK(storage[row + 1] == expected[row]);
            }
        }
        for (size_t i = rows + 1; i < 6; ++i) {
            CHECK(storage[i] == -123.0f);
        }
        CHECK(memcmp(weights->data, matrix, rows * cols * sizeof(float)) == 0);
        CHECK(memcmp(input, vector, cols * sizeof(float)) == 0);
        CHECK(weights->rows == rows && weights->cols == cols);
        CHECK(weights->data == original_data);
    }
    cinder_tensor_free(weights);
}

static void check_fixtures(cinder_error_t *error)
{
    /* By hand: [1*2 + (-2)*(-1) + 3*0.5, 4*2 + 0*(-1) + (-1)*0.5]. */
    const float matrix[] = {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f};
    const float input[] = {2.0f, -1.0f, 0.5f};
    const float expected[] = {5.5f, 7.5f};
    check_product(2, 3, matrix, input, expected, error);

    const float decimal_matrix[] = {0.1f, -0.2f, 0.3f, 0.4f, 0.5f, -0.6f};
    const float decimal_input[] = {0.7f, -0.8f, 0.9f};
    const float decimal_expected[] = {0.5f, -0.66f};
    check_product(2, 3, decimal_matrix, decimal_input, decimal_expected, error);

    const float row[] = {1.0f, -2.0f, 0.0f, 4.0f};
    const float row_input[] = {3.0f, 2.0f, 5.0f, -1.0f};
    const float row_expected[] = {-5.0f};
    check_product(1, 4, row, row_input, row_expected, error);
    const float scalar[] = {-3.0f};
    const float column_expected[] = {-3.0f, 6.0f, 0.0f, -12.0f};
    check_product(4, 1, row, scalar, column_expected, error);

    const float identity[] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float identity_input[] = {-2.25f, 3.5f};
    check_product(2, 2, identity, identity_input, identity_input, error);
    const float zeros[] = {0.0f, 0.0f, 0.0f};
    const float zero_expected[] = {0.0f, 0.0f};
    check_product(2, 3, matrix, zeros, zero_expected, error);

    const float one[] = {1.0f};
    const float extremes[] = {FLT_MAX, -FLT_MAX, FLT_TRUE_MIN, -0.0f};
    for (size_t i = 0; i < sizeof(extremes) / sizeof(extremes[0]); ++i) {
        check_product(1, 1, extremes + i, one, extremes + i, error);
    }
    const float tiny[] = {FLT_TRUE_MIN};
    const float half[] = {0.5f};
    const float zero[] = {0.0f};
    check_product(1, 1, tiny, half, zero, error);
}

static void check_rejected(
    const cinder_tensor_t *weights,
    const float *input,
    size_t input_count,
    size_t output_capacity,
    cinder_status_t expected,
    cinder_error_t *error
)
{
    float output[] = {-101.0f, -102.0f, -103.0f, -104.0f};
    const float before[] = {-101.0f, -102.0f, -103.0f, -104.0f};
    check_result(cinder_tensor_matvec(weights, input, input_count, output + 1,
                                     output_capacity, error), expected, error);
    CHECK(memcmp(output, before, sizeof(output)) == 0);
}

static void check_validation(cinder_error_t *error)
{
    const float matrix[] = {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f};
    float input[] = {2.0f, -1.0f, 0.5f};
    const float original_input[] = {2.0f, -1.0f, 0.5f};
    cinder_tensor_t *weights = NULL;
    CHECK(cinder_tensor_create(2, 3, &weights, NULL) == CINDER_OK);
    memcpy(weights->data, matrix, sizeof(matrix));

    check_rejected(NULL, input, 0, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(weights, NULL, 0, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_matvec(weights, input, 0, NULL, 0, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_matvec(NULL, NULL, 0, NULL, 0, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);

    const size_t wrong_counts[] = {0, 1, 2, 4, SIZE_MAX};
    for (size_t i = 0; i < sizeof(wrong_counts) / sizeof(wrong_counts[0]); ++i) {
        check_rejected(weights, input, wrong_counts[i], 2,
                       CINDER_ERROR_DIMENSION_MISMATCH, error);
    }
    check_rejected(weights, input, 3, 0, CINDER_ERROR_BUFFER_TOO_SMALL, error);
    check_rejected(weights, input, 3, 1, CINDER_ERROR_BUFFER_TOO_SMALL, error);

    /* Sizes are validated before inspecting any source values. */
    input[0] = NAN;
    check_rejected(weights, input, 2, 0, CINDER_ERROR_DIMENSION_MISMATCH, error);
    check_rejected(weights, input, 3, 1, CINDER_ERROR_BUFFER_TOO_SMALL, error);
    memcpy(input, original_input, sizeof(input));

    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t v = 0; v < sizeof(nonfinite) / sizeof(nonfinite[0]); ++v) {
        for (size_t col = 0; col < 3; ++col) {
            input[col] = nonfinite[v];
            float before[3];
            memcpy(before, input, sizeof(before));
            check_rejected(weights, input, 3, 2, CINDER_ERROR_NONFINITE, error);
            CHECK(memcmp(input, before, sizeof(input)) == 0);
            CHECK(memcmp(weights->data, matrix, sizeof(matrix)) == 0);
            memcpy(input, original_input, sizeof(input));
        }
        for (size_t i = 0; i < 6; ++i) {
            /* Bypass set solely to exercise defensive numeric validation. */
            weights->data[i] = nonfinite[v];
            float before[6];
            memcpy(before, weights->data, sizeof(before));
            check_rejected(weights, input, 3, 2, CINDER_ERROR_NONFINITE, error);
            CHECK(memcmp(weights->data, before, sizeof(before)) == 0);
            CHECK(memcmp(input, original_input, sizeof(input)) == 0);
            memcpy(weights->data, matrix, sizeof(matrix));
        }
    }
    CHECK(memcmp(weights->data, matrix, sizeof(matrix)) == 0);
    CHECK(memcmp(input, original_input, sizeof(input)) == 0);
    cinder_tensor_free(weights);
}

static void check_arithmetic_failure(cinder_error_t *error)
{
    /* Row zero succeeds; row one overflows from finite values. */
    const float cases[][6] = {
        {1.0f, 0.0f, 0.0f, FLT_MAX, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, -FLT_MAX, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f, FLT_MAX, FLT_MAX, -FLT_MAX}
    };
    const float inputs[][3] = {{2.0f, 1.0f, 1.0f}, {2.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};
    cinder_tensor_t *weights = NULL;
    CHECK(cinder_tensor_create(2, 3, &weights, NULL) == CINDER_OK);
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        memcpy(weights->data, cases[i], sizeof(cases[i]));
        float input[3];
        memcpy(input, inputs[i], sizeof(input));
        float storage[] = {-101.0f, -102.0f, -103.0f, -104.0f};
        check_result(cinder_tensor_matvec(weights, input, 3, storage + 1, 3, error),
                     CINDER_ERROR_NONFINITE, error);
        CHECK(storage[0] == -101.0f && storage[3] == -104.0f);
        CHECK(storage[1] == input[0]);
        CHECK(storage[2] == -103.0f);
        CHECK(memcmp(weights->data, cases[i], sizeof(cases[i])) == 0);
        CHECK(memcmp(input, inputs[i], sizeof(input)) == 0);
    }
    cinder_tensor_free(weights);
}

int main(void)
{
    for (size_t with_error = 0; with_error < 2; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        check_fixtures(error);
        check_validation(error);
        check_arithmetic_failure(error);
    }
    puts("Tensor matrix-vector tests passed.");
    return EXIT_SUCCESS;
}
