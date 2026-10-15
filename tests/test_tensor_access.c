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

static void check_value(float actual, float expected)
{
    CHECK(actual == expected);
    if (expected == 0.0f) {
        CHECK(!!signbit(actual) == !!signbit(expected));
    }
}

static void check_fill(size_t rows, size_t cols, cinder_error_t *error)
{
    const float values[] = {3.5f, -2.25f, 0.0f, -0.0f, FLT_MAX, -FLT_MAX, FLT_TRUE_MIN};
    cinder_tensor_t *tensor = NULL;
    CHECK(cinder_tensor_create(rows, cols, &tensor, NULL) == CINDER_OK);
    float *original_data = tensor->data;

    for (size_t v = 0; v < sizeof(values) / sizeof(values[0]); ++v) {
        /* A success must clear the error left by a rejected fill. */
        check_result(cinder_tensor_fill(tensor, NAN, error), CINDER_ERROR_NONFINITE, error);
        check_result(cinder_tensor_fill(tensor, values[v], error), CINDER_OK, error);
        for (size_t i = 0; i < rows * cols; ++i) {
            check_value(tensor->data[i], values[v]);
        }
    }
    CHECK(tensor->rows == rows && tensor->cols == cols);
    CHECK(tensor->data == original_data);
    cinder_tensor_free(tensor);
}

static void check_indexing(cinder_error_t *error)
{
    const size_t coordinates[][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
    const float initial[] = {-3.0f, 0.0f, 2.5f, 7.0f, -4.0f, 11.0f};
    const float replacements[] = {9.0f, -0.0f, FLT_MAX, FLT_TRUE_MIN, -6.0f, 0.0f};
    cinder_tensor_t *tensor = NULL;
    CHECK(cinder_tensor_create(2, 3, &tensor, NULL) == CINDER_OK);
    float *original_data = tensor->data;
    memcpy(tensor->data, initial, sizeof(initial));
    float expected[6];
    memcpy(expected, initial, sizeof(expected));

    for (size_t i = 0; i < 6; ++i) {
        float value = -123.0f;
        /* Check reads against a known flat array, independently of set. */
        check_result(cinder_tensor_get(tensor, 2, 0, &value, error), CINDER_ERROR_OUT_OF_BOUNDS, error);
        check_result(cinder_tensor_get(tensor, coordinates[i][0], coordinates[i][1], &value, error), CINDER_OK, error);
        check_value(value, initial[i]);
    }
    CHECK(memcmp(tensor->data, initial, sizeof(initial)) == 0);

    for (size_t i = 0; i < 6; ++i) {
        check_result(cinder_tensor_set(tensor, 0, 3, 1.0f, error), CINDER_ERROR_OUT_OF_BOUNDS, error);
        check_result(cinder_tensor_set(tensor, coordinates[i][0], coordinates[i][1], replacements[i], error), CINDER_OK, error);
        expected[i] = replacements[i];
        /* Every other element must retain its previous value. */
        for (size_t j = 0; j < 6; ++j) {
            check_value(tensor->data[j], expected[j]);
        }
    }
    for (size_t i = 0; i < 6; ++i) {
        float value = -123.0f;
        check_result(cinder_tensor_get(tensor, coordinates[i][0], coordinates[i][1], &value, error), CINDER_OK, error);
        check_value(value, replacements[i]);
    }
    CHECK(tensor->rows == 2 && tensor->cols == 3);
    CHECK(tensor->data == original_data);
    cinder_tensor_free(tensor);
}

static void check_failures(cinder_error_t *error)
{
    const float initial[] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f};
    const size_t invalid[][2] = {
        {2, 0}, {0, 3}, {1, 3}, {2, 3}, {0, 4},
        {SIZE_MAX, 0}, {0, SIZE_MAX}, {SIZE_MAX, SIZE_MAX},
        {SIZE_MAX / 3 + 1, 0}
    };
    const float nonfinite[] = {NAN, INFINITY, -INFINITY};
    cinder_tensor_t *tensor = NULL;
    CHECK(cinder_tensor_create(2, 3, &tensor, NULL) == CINDER_OK);
    float *original_data = tensor->data;
    memcpy(tensor->data, initial, sizeof(initial));
    float output = -123.0f;

    check_result(cinder_tensor_fill(NULL, NAN, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_set(NULL, SIZE_MAX, SIZE_MAX, NAN, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_get(NULL, 0, 0, &output, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    CHECK(output == -123.0f);
    check_result(cinder_tensor_get(tensor, SIZE_MAX, SIZE_MAX, NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_tensor_get(NULL, 0, 0, NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    CHECK(memcmp(tensor->data, initial, sizeof(initial)) == 0);

    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        check_result(cinder_tensor_get(tensor, invalid[i][0], invalid[i][1], &output, error), CINDER_ERROR_OUT_OF_BOUNDS, error);
        CHECK(output == -123.0f);
        check_result(cinder_tensor_set(tensor, invalid[i][0], invalid[i][1], 8.0f, error), CINDER_ERROR_OUT_OF_BOUNDS, error);
        CHECK(memcmp(tensor->data, initial, sizeof(initial)) == 0);
        /* Bounds are checked before rejecting a non-finite value. */
        check_result(cinder_tensor_set(tensor, invalid[i][0], invalid[i][1], NAN, error), CINDER_ERROR_OUT_OF_BOUNDS, error);
        CHECK(memcmp(tensor->data, initial, sizeof(initial)) == 0);
    }

    for (size_t i = 0; i < sizeof(nonfinite) / sizeof(nonfinite[0]); ++i) {
        check_result(cinder_tensor_fill(tensor, nonfinite[i], error), CINDER_ERROR_NONFINITE, error);
        CHECK(memcmp(tensor->data, initial, sizeof(initial)) == 0);
        check_result(cinder_tensor_set(tensor, 1, 1, nonfinite[i], error), CINDER_ERROR_NONFINITE, error);
        CHECK(memcmp(tensor->data, initial, sizeof(initial)) == 0);

        /* Deliberately bypass set to exercise get's non-finite check. */
        tensor->data[4] = nonfinite[i];
        float before[6];
        memcpy(before, tensor->data, sizeof(before));
        output = -123.0f;
        check_result(cinder_tensor_get(tensor, 1, 1, &output, error), CINDER_ERROR_NONFINITE, error);
        CHECK(output == -123.0f);
        CHECK(memcmp(tensor->data, before, sizeof(before)) == 0);
        /* Get validates only the selected element. */
        check_result(cinder_tensor_get(tensor, 0, 0, &output, error), CINDER_OK, error);
        CHECK(output == initial[0]);
        memcpy(tensor->data, initial, sizeof(initial));
    }
    CHECK(tensor->rows == 2 && tensor->cols == 3);
    CHECK(tensor->data == original_data);
    cinder_tensor_free(tensor);
}

int main(void)
{
    const size_t shapes[][2] = {{1, 1}, {2, 3}, {3, 2}, {1, 7}, {7, 1}};
    /* Exercise every operation both with and without optional error details. */
    for (size_t with_error = 0; with_error < 2; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
            check_fill(shapes[i][0], shapes[i][1], error);
        }
        check_indexing(error);
        check_failures(error);
    }
    puts("Tensor access tests passed.");
    return EXIT_SUCCESS;
}
