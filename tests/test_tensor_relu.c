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

static void check_values(
    const float *initial,
    const float *expected,
    size_t count,
    cinder_error_t *error
)
{
    CHECK(count > 0 && count <= 8);
    float storage[10];
    for (size_t i = 0; i < 10; ++i) {
        storage[i] = -123.0f;
    }
    memcpy(storage + 1, initial, count * sizeof(float));
    /* Non-finite spare capacity must not be examined or changed. */
    storage[count + 1] = NAN;
    check_result(cinder_tensor_relu(storage + 1, 0, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_relu(storage + 1, count, error), CINDER_OK, error);
    for (size_t i = 0; i < count; ++i) {
        CHECK(storage[i + 1] == expected[i]);
        if (expected[i] == 0.0f) {
            CHECK(!!signbit(storage[i + 1]) == !!signbit(expected[i]));
        }
    }
    CHECK(storage[0] == -123.0f);
    CHECK(isnan(storage[count + 1]));
    for (size_t i = count + 2; i < 10; ++i) {
        CHECK(storage[i] == -123.0f);
    }

    /* Applying ReLU twice must have the same effect as applying it once. */
    float before[10];
    memcpy(before, storage, sizeof(before));
    check_result(cinder_tensor_relu(storage + 1, count, error), CINDER_OK, error);
    CHECK(memcmp(storage, before, sizeof(storage)) == 0);
}

static void check_fixtures(cinder_error_t *error)
{
    const float initial[] = {-FLT_MAX, -3.5f, -FLT_TRUE_MIN, -0.0f,
                            0.0f, FLT_TRUE_MIN, 2.75f, FLT_MAX};
    const float expected[] = {0.0f, 0.0f, 0.0f, -0.0f,
                             0.0f, FLT_TRUE_MIN, 2.75f, FLT_MAX};
    check_values(initial, expected, 8, error);
    for (size_t i = 0; i < 8; ++i) {
        check_values(initial + i, expected + i, 1, error);
    }
    check_values(initial, expected, 3, error);         /* All negative. */
    check_values(initial + 5, expected + 5, 3, error); /* All positive. */
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
    check_result(cinder_tensor_relu(storage + 1, count, error), expected, error);
    CHECK(memcmp(storage, before, sizeof(storage)) == 0);
}

static void check_validation(cinder_error_t *error)
{
    check_result(cinder_tensor_relu(NULL, 0, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_relu(NULL, 3, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_relu(NULL, SIZE_MAX, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    const float initial[] = {-3.0f, -2.0f, -1.0f};
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
    }
}

static void check_matvec_bias_relu(cinder_error_t *error)
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
    CHECK(!signbit(output[1]));
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
        check_validation(error);
        check_matvec_bias_relu(error);
    }
    puts("Tensor ReLU tests passed.");
    return EXIT_SUCCESS;
}
