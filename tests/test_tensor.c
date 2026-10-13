#include "cinder/tensor.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Unlike assert(), these checks also run in Release builds with NDEBUG. */
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

static void check_rejected_shape(size_t rows, size_t cols, cinder_status_t expected)
{
    /* A non-owning sentinel verifies that create resets the output on error. */
    cinder_tensor_t sentinel = {0};
    cinder_tensor_t *tensor = &sentinel;
    cinder_error_t error;
    memset(&error, 0x7f, sizeof(error));
    CHECK(cinder_tensor_create(rows, cols, &tensor, &error) == expected);
    CHECK(tensor == NULL);
    check_error(expected, &error);

    tensor = &sentinel;
    CHECK(cinder_tensor_create(rows, cols, &tensor, NULL) == expected);
    CHECK(tensor == NULL);
}

static void check_valid_shape(size_t rows, size_t cols)
{
    cinder_tensor_t *tensor = NULL;
    cinder_error_t error;
    /* A successful call must clear an error left by an earlier call. */
    CHECK(cinder_tensor_create(0, cols, &tensor, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_tensor_create(rows, cols, &tensor, &error) == CINDER_OK);
    CHECK(error.code == CINDER_OK);
    CHECK(error.message[0] == '\0');
    CHECK(tensor != NULL);
    CHECK(tensor->rows == rows);
    CHECK(tensor->cols == cols);
    CHECK(tensor->data != NULL);

    size_t count = rows * cols;
    for (size_t i = 0; i < count; ++i) {
        CHECK(tensor->data[i] == 0.0f);
        tensor->data[i] = (float)(i + 1);
    }
    /* Touch the whole array, including its final element, under ASan. */
    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            size_t index = row * cols + col;
            CHECK(tensor->data[index] == (float)(index + 1));
        }
    }
    cinder_tensor_free(tensor);
}

int main(void)
{
    cinder_error_t error;
    memset(&error, 0x7f, sizeof(error));
    CHECK(cinder_tensor_create(2, 3, NULL, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    check_error(CINDER_ERROR_INVALID_ARGUMENT, &error);
    CHECK(cinder_tensor_create(2, 3, NULL, NULL) == CINDER_ERROR_INVALID_ARGUMENT);

    check_rejected_shape(0, 0, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected_shape(0, 3, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected_shape(2, 0, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected_shape(SIZE_MAX, 0, CINDER_ERROR_INVALID_ARGUMENT);
    check_rejected_shape(SIZE_MAX, 2, CINDER_ERROR_OVERFLOW);
    check_rejected_shape(2, SIZE_MAX, CINDER_ERROR_OVERFLOW);
    /* These products would wrap to zero without the element-count guard. */
    check_rejected_shape(SIZE_MAX / 2 + 1, 2, CINDER_ERROR_OVERFLOW);
    check_rejected_shape(2, SIZE_MAX / 2 + 1, CINDER_ERROR_OVERFLOW);
    check_rejected_shape(SIZE_MAX / sizeof(float) + 1, 1, CINDER_ERROR_OVERFLOW);

    check_valid_shape(1, 1);
    check_valid_shape(2, 3);
    check_valid_shape(3, 2);
    check_valid_shape(7, 1);
    check_valid_shape(1, 7);

    cinder_tensor_t *first = NULL;
    cinder_tensor_t *second = NULL;
    CHECK(cinder_tensor_create(2, 3, &first, NULL) == CINDER_OK);
    CHECK(cinder_tensor_create(2, 3, &second, NULL) == CINDER_OK);
    CHECK(first != second);
    CHECK(first->data != second->data);
    first->data[5] = 42.0f;
    CHECK(second->data[5] == 0.0f);
    cinder_tensor_free(first);
    second->data[5] = 7.0f;
    CHECK(second->data[5] == 7.0f);
    cinder_tensor_free(second);

    for (size_t i = 0; i < 100; ++i) {
        check_valid_shape(2, 3);
    }
    cinder_tensor_free(NULL);
    puts("Tensor allocation tests passed.");
    return EXIT_SUCCESS;
}
