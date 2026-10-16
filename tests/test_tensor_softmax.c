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

static void check_close(float actual, double expected)
{
    CHECK(isfinite(actual) && isfinite(expected));
    CHECK(fabs((double)actual - expected) <= 1e-5 + 1e-5 * fabs(expected));
}

static void check_distribution(const float *values, size_t count)
{
    double total = 0.0;
    for (size_t i = 0; i < count; ++i) {
        CHECK(isfinite(values[i]));
        CHECK(values[i] >= 0.0f && values[i] <= 1.0f);
        total += (double)values[i];
    }
    CHECK(fabs(total - 1.0) <= 1e-6);
}

static void check_values(
    const float *initial,
    const double *expected,
    size_t count,
    cinder_error_t *error
)
{
    CHECK(count > 0 && count <= 4);
    float storage[6];
    for (size_t i = 0; i < 6; ++i) {
        storage[i] = -123.0f;
    }
    memcpy(storage + 1, initial, count * sizeof(float));
    /* Non-finite spare capacity must not be examined or changed. */
    storage[count + 1] = NAN;
    check_result(cinder_tensor_softmax(storage + 1, 0, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_softmax(storage + 1, count, error), CINDER_OK, error);
    check_distribution(storage + 1, count);
    for (size_t i = 0; i < count; ++i) {
        check_close(storage[i + 1], expected[i]);
        if (expected[i] == 0.0 || expected[i] == 1.0) {
            CHECK(storage[i + 1] == (float)expected[i]);
        }
    }
    CHECK(storage[0] == -123.0f);
    CHECK(isnan(storage[count + 1]));
    for (size_t i = count + 2; i < 6; ++i) {
        CHECK(storage[i] == -123.0f);
    }
}

static void check_fixtures(cinder_error_t *error)
{
    /* Constants evaluated independently with Python Decimal at 60-digit precision. */
    const double expected[] = {0.0900305731703804580, 0.244728471054797652,
                               0.665240955774821890};
    const float initial[] = {1.0f, 2.0f, 3.0f};
    const float shifted_up[] = {1001.0f, 1002.0f, 1003.0f};
    const float shifted_down[] = {-999.0f, -998.0f, -997.0f};
    check_values(initial, expected, 3, error);
    check_values(shifted_up, expected, 3, error);
    check_values(shifted_down, expected, 3, error);
    /* Rotate the maximum through every position. */
    for (size_t offset = 1; offset < 3; ++offset) {
        float rotated[3];
        double rotated_expected[3];
        for (size_t i = 0; i < 3; ++i) {
            rotated[i] = initial[(i + offset) % 3];
            rotated_expected[i] = expected[(i + offset) % 3];
        }
        check_values(rotated, rotated_expected, 3, error);
    }
    const float mixed[] = {-2.0f, 0.0f, 1.0f};
    const double mixed_expected[] = {0.0351190269593397242, 0.259496460342419117,
                                     0.705384512698241158};
    check_values(mixed, mixed_expected, 3, error);

    const float equal_cases[] = {-FLT_MAX, -12.5f, -0.0f, 0.0f, 12.5f, FLT_MAX};
    const double uniform[] = {0.25, 0.25, 0.25, 0.25};
    const double one[] = {1.0};
    for (size_t i = 0; i < sizeof(equal_cases) / sizeof(equal_cases[0]); ++i) {
        float equal[] = {equal_cases[i], equal_cases[i], equal_cases[i], equal_cases[i]};
        check_values(equal, uniform, 4, error);
        check_values(equal, one, 1, error);
    }
    const float zeros[] = {-0.0f, 0.0f, -0.0f, 0.0f};
    check_values(zeros, uniform, 4, error);

    const float extremes[] = {-FLT_MAX, FLT_MAX};
    const double extreme_expected[] = {0.0, 1.0};
    check_values(extremes, extreme_expected, 2, error);
    const float reverse[] = {FLT_MAX, -FLT_MAX};
    const double reverse_expected[] = {1.0, 0.0};
    check_values(reverse, reverse_expected, 2, error);
    const float tied[] = {FLT_MAX, -FLT_MAX, FLT_MAX};
    const double tied_expected[] = {0.5, 0.0, 0.5};
    check_values(tied, tied_expected, 3, error);
}

static void check_tiny_probabilities(cinder_error_t *error)
{
    const float scores[] = {-90.0f, -100.0f, -104.0f};
    const double expected[] = {8.19401262399051543e-40, 3.72007597602083596e-44,
                               6.81355682154529851e-46};
    for (size_t i = 0; i < 3; ++i) {
        float values[] = {0.0f, scores[i]};
        check_result(cinder_tensor_softmax(values, 2, error), CINDER_OK, error);
        check_distribution(values, 2);
        CHECK(values[0] == 1.0f);
        /* The general 1e-5 tolerance would hide lost subnormal probabilities. */
        CHECK(fabs((double)values[1] - expected[i]) <= (double)FLT_TRUE_MIN);
        if (i < 2) {
            CHECK(values[1] > 0.0f && values[1] < FLT_MIN);
        } else {
            CHECK(values[1] == 0.0f); /* Too small for float32: allowed underflow. */
        }
    }
}

static void check_long_vector(cinder_error_t *error)
{
    const size_t count = 10001;
    float *values = malloc(count * sizeof(*values));
    CHECK(values != NULL);
    values[0] = 0.0f;
    for (size_t i = 1; i < count; ++i) {
        values[i] = -18.0f;
    }
    check_result(cinder_tensor_softmax(values, count, error), CINDER_OK, error);
    check_distribution(values, count);
    check_close(values[0], 0.999847723394249086);
    /* These terms would individually disappear when added to 1 in float. */
    const double expected_small = 1.52276605750914484e-8;
    for (size_t i = 1; i < count; ++i) {
        CHECK(fabs((double)values[i] - expected_small) <= 1e-5 * expected_small);
    }
    free(values);
}

static void check_rejected(
    const float *initial,
    size_t count,
    cinder_status_t expected,
    cinder_error_t *error
)
{
    float storage[] = {-101.0f, 0.0f, 0.0f, 0.0f, -102.0f};
    memcpy(storage + 1, initial, 3 * sizeof(float));
    float before[5];
    memcpy(before, storage, sizeof(before));
    check_result(cinder_tensor_softmax(storage + 1, count, error), expected, error);
    CHECK(memcmp(storage, before, sizeof(storage)) == 0);
}

static void check_validation(cinder_error_t *error)
{
    check_result(cinder_tensor_softmax(NULL, 0, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_softmax(NULL, 3, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_softmax(NULL, SIZE_MAX, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    const float initial[] = {1.0f, 2.0f, 3.0f};
    check_rejected(initial, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_rejected(initial, SIZE_MAX / sizeof(float) + 1, CINDER_ERROR_OVERFLOW, error);
    check_rejected(initial, SIZE_MAX, CINDER_ERROR_OVERFLOW, error);

    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    for (size_t v = 0; v < sizeof(nonfinite) / sizeof(nonfinite[0]); ++v) {
        for (size_t i = 0; i < 3; ++i) {
            float values[3];
            memcpy(values, initial, sizeof(values));
            values[i] = nonfinite[v];
            check_rejected(values, 3, CINDER_ERROR_NONFINITE, error);
            /* Pointer/size validation precedes numeric inspection. */
            check_rejected(values, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
            check_rejected(values, SIZE_MAX, CINDER_ERROR_OVERFLOW, error);
        }
        const float all_nonfinite[] = {nonfinite[v], nonfinite[v], nonfinite[v]};
        check_rejected(all_nonfinite, 3, CINDER_ERROR_NONFINITE, error);
    }
}

static void check_composed_math(cinder_error_t *error)
{
    const float matrix[] = {1.0f, -2.0f, 3.0f, 4.0f, 0.0f, -1.0f};
    const float input[] = {2.0f, -1.0f, 0.5f};
    const float bias[] = {0.5f, -9.5f};
    cinder_tensor_t *weights = NULL;
    CHECK(cinder_tensor_create(2, 3, &weights, NULL) == CINDER_OK);
    memcpy(weights->data, matrix, sizeof(matrix));
    float storage[] = {-101.0f, NAN, NAN, NAN, -102.0f};
    float *output = storage + 1;
    check_result(cinder_tensor_matvec(weights, input, 3, output, 3, error), CINDER_OK, error);
    CHECK(output[0] == 5.5f && output[1] == 7.5f);
    check_result(cinder_tensor_add_bias(output, weights->rows, bias, 2, error), CINDER_OK, error);
    CHECK(output[0] == 6.0f && output[1] == -2.0f);
    check_result(cinder_tensor_relu(output, weights->rows, error), CINDER_OK, error);
    CHECK(output[0] == 6.0f && output[1] == 0.0f);
    check_result(cinder_tensor_softmax(output, weights->rows, error), CINDER_OK, error);
    check_close(output[0], 0.997527376843365226);
    check_close(output[1], 0.00247262315663477433);
    check_distribution(output, weights->rows);
    CHECK(storage[0] == -101.0f && storage[4] == -102.0f);
    CHECK(isnan(output[2]));
    CHECK(memcmp(weights->data, matrix, sizeof(matrix)) == 0);
    cinder_tensor_free(weights);
}

int main(void)
{
    for (size_t with_error = 0; with_error < 2; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        check_fixtures(error);
        check_tiny_probabilities(error);
        check_long_vector(error);
        check_validation(error);
        check_composed_math(error);
    }
    puts("Tensor Softmax tests passed.");
    return EXIT_SUCCESS;
}
