#include "cinder/tensor.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static size_t allocation_calls;
static size_t fail_on_call;
static void *live_blocks[2];
static size_t live_count;

static void remember_block(void *block)
{
    CHECK(block != NULL);
    CHECK(live_count < sizeof(live_blocks) / sizeof(live_blocks[0]));
    live_blocks[live_count++] = block;
}

static void *test_malloc(size_t size)
{
    ++allocation_calls;
    if (allocation_calls == fail_on_call) {
        return NULL;
    }
    void *block = malloc(size);
    remember_block(block);
    return block;
}

static void *test_calloc(size_t count, size_t size)
{
    ++allocation_calls;
    if (allocation_calls == fail_on_call) {
        return NULL;
    }
    void *block = calloc(count, size);
    remember_block(block);
    return block;
}

static void test_free(void *block)
{
    if (block == NULL) {
        return;
    }
    for (size_t i = 0; i < live_count; ++i) {
        if (live_blocks[i] == block) {
            --live_count;
            live_blocks[i] = live_blocks[live_count];
            live_blocks[live_count] = NULL;
            free(block);
            return;
        }
    }
    CHECK(0 && "free received an untracked or already freed allocation");
}

/*
 * Compile the actual implementation with replacements limited to this test.
 * System headers and the wrappers above are parsed before these macros, so
 * wrappers still call the real allocator. Production has no fault switches.
 */
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#include "../src/tensor.c"
#undef malloc
#undef calloc
#undef free

static void reset_allocator(size_t failure)
{
    CHECK(live_count == 0);
    allocation_calls = 0;
    fail_on_call = failure;
}

static void check_allocation_failure(size_t failure, cinder_error_t *error)
{
    reset_allocator(failure);
    cinder_tensor_t sentinel = {0};
    cinder_tensor_t *tensor = &sentinel;
    CHECK(cinder_tensor_create(2, 3, &tensor, error) == CINDER_ERROR_ALLOCATION);
    CHECK(tensor == NULL);
    CHECK(allocation_calls == failure);
    CHECK(live_count == 0);
    if (error != NULL) {
        CHECK(error->code == CINDER_ERROR_ALLOCATION);
        CHECK(error->message[0] != '\0');
        CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
    }
}

int main(void)
{
    cinder_error_t error;
    for (size_t failure = 1; failure <= 2; ++failure) {
        memset(&error, 0x7f, sizeof(error));
        check_allocation_failure(failure, &error);
        check_allocation_failure(failure, NULL);
    }

    /* Reject invalid requests without even attempting an allocation. */
    reset_allocator(1);
    cinder_tensor_t *tensor = NULL;
    CHECK(cinder_tensor_create(2, 3, NULL, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_tensor_create(0, 3, &tensor, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_tensor_create(3, 0, &tensor, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_tensor_create(SIZE_MAX, 2, &tensor, &error) == CINDER_ERROR_OVERFLOW);
    CHECK(cinder_tensor_create(SIZE_MAX / 2 + 1, 2, &tensor, &error) == CINDER_ERROR_OVERFLOW);
    CHECK(cinder_tensor_create(SIZE_MAX / sizeof(float) + 1, 1, &tensor, &error) == CINDER_ERROR_OVERFLOW);
    CHECK(allocation_calls == 0);
    CHECK(live_count == 0);

    /* Largest byte-safe count passes validation; fail before reserving it. */
    reset_allocator(1);
    CHECK(cinder_tensor_create(SIZE_MAX / sizeof(float), 1, &tensor, &error) == CINDER_ERROR_ALLOCATION);
    CHECK(tensor == NULL);
    CHECK(allocation_calls == 1);
    CHECK(live_count == 0);

    /* Creation still works after failures, and normal cleanup frees both blocks. */
    reset_allocator(0);
    CHECK(cinder_tensor_create(2, 3, &tensor, &error) == CINDER_OK);
    CHECK(error.code == CINDER_OK);
    CHECK(error.message[0] == '\0');
    CHECK(allocation_calls == 2);
    CHECK(live_count == 2);

    /* Access must work even when the next allocation would fail. */
    fail_on_call = allocation_calls + 1;
    float value = -1.0f;
    CHECK(cinder_tensor_fill(tensor, 2.0f, &error) == CINDER_OK);
    CHECK(cinder_tensor_set(tensor, 1, 2, 3.0f, &error) == CINDER_OK);
    CHECK(cinder_tensor_get(tensor, 1, 2, &value, &error) == CINDER_OK);
    CHECK(value == 3.0f);
    CHECK(cinder_tensor_fill(tensor, NAN, &error) == CINDER_ERROR_NONFINITE);
    CHECK(cinder_tensor_set(tensor, 0, 3, 1.0f, &error) == CINDER_ERROR_OUT_OF_BOUNDS);
    CHECK(cinder_tensor_get(tensor, SIZE_MAX, 0, &value, &error) == CINDER_ERROR_OUT_OF_BOUNDS);
    CHECK(value == 3.0f);
    CHECK(allocation_calls == 2);
    CHECK(live_count == 2);

    /* Matrix-vector multiplication also uses only caller-provided storage. */
    const float input[] = {1.0f, 2.0f, 3.0f};
    float output[] = {-1.0f, -1.0f};
    CHECK(cinder_tensor_matvec(tensor, input, 3, output, 2, &error) == CINDER_OK);
    CHECK(output[0] == 12.0f && output[1] == 15.0f);
    CHECK(cinder_tensor_matvec(tensor, input, 2, output, 2, &error) == CINDER_ERROR_DIMENSION_MISMATCH);
    CHECK(cinder_tensor_matvec(tensor, input, 3, output, 1, &error) == CINDER_ERROR_BUFFER_TOO_SMALL);
    tensor->data[5] = NAN;
    CHECK(cinder_tensor_matvec(tensor, input, 3, output, 2, &error) == CINDER_ERROR_NONFINITE);
    CHECK(output[0] == 12.0f && output[1] == 15.0f);
    CHECK(allocation_calls == 2);
    CHECK(live_count == 2);

    /* Bias uses the matvec result in place, including on its error paths. */
    const float bias[] = {0.5f, -2.0f};
    CHECK(cinder_tensor_add_bias(output, 2, bias, 2, &error) == CINDER_OK);
    CHECK(output[0] == 12.5f && output[1] == 13.0f);
    CHECK(cinder_tensor_add_bias(output, 2, bias, 1, &error) == CINDER_ERROR_DIMENSION_MISMATCH);
    const float bad_bias[] = {0.0f, NAN};
    CHECK(cinder_tensor_add_bias(output, 2, bad_bias, 2, &error) == CINDER_ERROR_NONFINITE);
    CHECK(output[0] == 12.5f && output[1] == 13.0f);
    output[1] = FLT_MAX;
    const float overflowing_bias[] = {0.5f, FLT_MAX};
    CHECK(cinder_tensor_add_bias(output, 2, overflowing_bias, 2, &error) == CINDER_ERROR_NONFINITE);
    CHECK(output[0] == 13.0f && output[1] == FLT_MAX);
    CHECK(allocation_calls == 2);
    CHECK(live_count == 2);

    /* ReLU also works with the next allocation armed to fail. */
    output[0] = -3.0f;
    CHECK(cinder_tensor_relu(output, 2, &error) == CINDER_OK);
    CHECK(output[0] == 0.0f && output[1] == FLT_MAX);
    CHECK(cinder_tensor_relu(output, 0, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_tensor_relu(output, SIZE_MAX, &error) == CINDER_ERROR_OVERFLOW);
    output[0] = -3.0f;
    output[1] = NAN;
    CHECK(cinder_tensor_relu(output, 2, &error) == CINDER_ERROR_NONFINITE);
    CHECK(output[0] == -3.0f && isnan(output[1]));
    CHECK(allocation_calls == 2);
    CHECK(live_count == 2);

    /* Softmax reuses the vector for exponentials; no scratch allocation. */
    output[0] = FLT_MAX;
    output[1] = FLT_MAX;
    CHECK(cinder_tensor_softmax(output, 2, &error) == CINDER_OK);
    CHECK(output[0] == 0.5f && output[1] == 0.5f);
    CHECK(cinder_tensor_softmax(output, 0, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_tensor_softmax(output, SIZE_MAX, &error) == CINDER_ERROR_OVERFLOW);
    output[0] = -3.0f;
    output[1] = NAN;
    CHECK(cinder_tensor_softmax(output, 2, &error) == CINDER_ERROR_NONFINITE);
    CHECK(output[0] == -3.0f && isnan(output[1]));
    CHECK(allocation_calls == 2);
    CHECK(live_count == 2);

    cinder_tensor_free(tensor);
    CHECK(live_count == 0);
    cinder_tensor_free(NULL);
    CHECK(live_count == 0);

    puts("Tensor allocation failure tests passed.");
    return EXIT_SUCCESS;
}
