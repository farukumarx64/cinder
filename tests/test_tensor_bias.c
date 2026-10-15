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
    if (expected == 0.0f || magnitude < FLT_MIN) {
        CHECK(actual == expected);
    }
}

static void check_addition(
    size_t count,
    const float *initial,
    const float *bias_values,
    const float *expected,
    cinder_error_t *error
)
{
    CHECK(count > 0 && count <= 5);
    float storage[7] = {-123.0f, -123.0f, -123.0f, -123.0f, -123.0f, -123.0f, -123.0f};
    memcpy(storage + 1, initial, count * sizeof(float));
    float bias[5];
    memcpy(bias, bias_values, count * sizeof(float));

    /* Success must clear details left by a preceding failure. */
    check_result(cinder_tensor_add_bias(storage + 1, count, bias, count + 1, error),
                 CINDER_ERROR_DIMENSION_MISMATCH, error);
    check_result(cinder_tensor_add_bias(storage + 1, count, bias, count, error),
                 CINDER_OK, error);
    for (size_t i = 0; i < count; ++i) {
        check_close(storage[i + 1], expected[i]);
    }
    CHECK(storage[0] == -123.0f);
    for (size_t i = count + 1; i < 7; ++i) {
        CHECK(storage[i] == -123.0f);
    }
    CHECK(memcmp(bias, bias_values, count * sizeof(float)) == 0);
}

static void check_fixtures(cinder_error_t *error)
{
    const float initial[] = {5.5f, 7.5f, -3.0f, 0.0f, 2.0f};
    const float bias[] = {0.5f, -2.0f, 4.0f, -1.0f, 0.0f};
    const float expected[] = {6.0f, 5.5f, 1.0f, -1.0f, 2.0f};
    const size_t counts[] = {1, 2, 5};
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
        check_addition(counts[i], initial, bias, expected, error);
    }

    const float zeros[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    check_addition(5, initial, zeros, initial, error);
    check_addition(5, zeros, bias, bias, error);
    const float decimals[] = {0.1f, -0.4f};
    const float decimal_bias[] = {0.2f, 0.1f};
    const float decimal_expected[] = {0.3f, -0.3f};
    check_addition(2, decimals, decimal_bias, decimal_expected, error);

    const float extremes[] = {FLT_MAX, -FLT_MAX, FLT_MAX, FLT_TRUE_MIN, -FLT_TRUE_MIN};
    const float extreme_bias[] = {0.0f, 0.0f, -FLT_MAX, FLT_TRUE_MIN, -FLT_TRUE_MIN};
    const float extreme_expected[] = {FLT_MAX, -FLT_MAX, 0.0f, 2 * FLT_TRUE_MIN, -2 * FLT_TRUE_MIN};
    check_addition(5, extremes, extreme_bias, extreme_expected, error);
}

static void check_rejected(
    const float *initial,
    size_t output_count,
    const float *bias,
    size_t bias_count,
    cinder_status_t expected,
    cinder_error_t *error
)
{
    float output[] = {-101.0f, 0.0f, 0.0f, 0.0f, -102.0f};
    memcpy(output + 1, initial, 3 * sizeof(float));
    float before[5];
    memcpy(before, output, sizeof(before));
    check_result(cinder_tensor_add_bias(output + 1, output_count, bias, bias_count, error),
                 expected, error);
    CHECK(memcmp(output, before, sizeof(output)) == 0);
}

static void check_validation(cinder_error_t *error)
{
    float initial[] = {1.0f, 2.0f, 3.0f};
    float bias[] = {0.5f, -1.0f, 2.0f};
    const float original_bias[] = {0.5f, -1.0f, 2.0f};
    check_result(cinder_tensor_add_bias(NULL, 3, bias, 3, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_add_bias(NULL, 0, NULL, 0, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(initial, 3, NULL, 3, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(initial, 0, bias, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(initial, 0, bias, 3, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(initial, 3, bias, 0, CINDER_ERROR_INVALID_ARGUMENT, error);

    const size_t wrong_counts[] = {1, 2, 4, SIZE_MAX};
    for (size_t i = 0; i < sizeof(wrong_counts) / sizeof(wrong_counts[0]); ++i) {
        check_rejected(initial, 3, bias, wrong_counts[i], CINDER_ERROR_DIMENSION_MISMATCH, error);
        check_rejected(initial, wrong_counts[i], bias, 3, CINDER_ERROR_DIMENSION_MISMATCH, error);
    }
    const size_t too_large[] = {SIZE_MAX / sizeof(float) + 1, SIZE_MAX};
    for (size_t i = 0; i < sizeof(too_large) / sizeof(too_large[0]); ++i) {
        check_rejected(initial, too_large[i], bias, too_large[i], CINDER_ERROR_OVERFLOW, error);
    }

    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t v = 0; v < sizeof(nonfinite) / sizeof(nonfinite[0]); ++v) {
        for (size_t i = 0; i < 3; ++i) {
            float saved = initial[i];
            initial[i] = nonfinite[v];
            check_rejected(initial, 3, bias, 3, CINDER_ERROR_NONFINITE, error);
            initial[i] = saved;

            saved = bias[i];
            bias[i] = nonfinite[v];
            float before[3];
            memcpy(before, bias, sizeof(before));
            check_rejected(initial, 3, bias, 3, CINDER_ERROR_NONFINITE, error);
            CHECK(memcmp(bias, before, sizeof(before)) == 0);
            bias[i] = saved;
        }
    }
    CHECK(memcmp(bias, original_bias, sizeof(bias)) == 0);

    /* Size errors take precedence over numeric validation. */
    bias[2] = NAN;
    check_rejected(initial, 3, bias, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(initial, 3, bias, 2, CINDER_ERROR_DIMENSION_MISMATCH, error);
    check_rejected(initial, SIZE_MAX, bias, SIZE_MAX, CINDER_ERROR_OVERFLOW, error);
}

static void check_arithmetic_failure(cinder_error_t *error)
{
    const float extremes[] = {FLT_MAX, -FLT_MAX};
    const float expected_prefix[] = {1.5f, 1.0f, 5.0f};
    for (size_t sign = 0; sign < 2; ++sign) {
        for (size_t failed = 0; failed < 3; ++failed) {
            float storage[] = {-101.0f, 1.0f, 2.0f, 3.0f, -102.0f};
            float bias[] = {0.5f, -1.0f, 2.0f};
            storage[failed + 1] = extremes[sign];
            bias[failed] = extremes[sign];
            float before[5];
            float original_bias[3];
            memcpy(before, storage, sizeof(before));
            memcpy(original_bias, bias, sizeof(original_bias));
            check_result(cinder_tensor_add_bias(storage + 1, 3, bias, 3, error),
                         CINDER_ERROR_NONFINITE, error);
            for (size_t i = 0; i < failed; ++i) {
                CHECK(storage[i + 1] == expected_prefix[i]);
            }
            CHECK(memcmp(storage + failed + 1, before + failed + 1,
                         (3 - failed) * sizeof(float)) == 0);
            CHECK(storage[0] == -101.0f && storage[4] == -102.0f);
            CHECK(memcmp(bias, original_bias, sizeof(bias)) == 0);
        }
    }
}

static void check_matvec_then_bias(cinder_error_t *error)
{
    const float matrix[] = {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f};
    const float inputs[][3] = {{2.0f, -1.0f, 0.5f}, {0.0f, 0.0f, 0.0f}};
    const float products[][2] = {{5.5f, 7.5f}, {0.0f, 0.0f}};
    const float expected[][2] = {{6.0f, 5.5f}, {0.5f, -2.0f}};
    const float original_bias[] = {0.5f, -2.0f};
    cinder_tensor_t *weights = NULL;
    CHECK(cinder_tensor_create(2, 3, &weights, NULL) == CINDER_OK);
    memcpy(weights->data, matrix, sizeof(matrix));
    for (size_t example = 0; example < 2; ++example) {
        float input[3];
        float bias[2];
        memcpy(input, inputs[example], sizeof(input));
        memcpy(bias, original_bias, sizeof(bias));
        /* The third output slot is spare capacity, not part of the result. */
        float storage[] = {-101.0f, NAN, NAN, NAN, -102.0f};
        check_result(cinder_tensor_matvec(weights, input, 3, storage + 1, 3, error),
                     CINDER_OK, error);
        for (size_t i = 0; i < weights->rows; ++i) {
            check_close(storage[i + 1], products[example][i]);
        }
        check_result(cinder_tensor_add_bias(storage + 1, weights->rows, bias, 2, error),
                     CINDER_OK, error);
        for (size_t i = 0; i < weights->rows; ++i) {
            check_close(storage[i + 1], expected[example][i]);
        }
        CHECK(storage[0] == -101.0f && storage[4] == -102.0f);
        CHECK(isnan(storage[3]));
        CHECK(memcmp(weights->data, matrix, sizeof(matrix)) == 0);
        CHECK(memcmp(input, inputs[example], sizeof(input)) == 0);
        CHECK(memcmp(bias, original_bias, sizeof(bias)) == 0);
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
        check_matvec_then_bias(error);
    }
    puts("Tensor bias addition tests passed.");
    return EXIT_SUCCESS;
}
