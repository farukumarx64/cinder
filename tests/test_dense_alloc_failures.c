#include "cinder/dense.h"

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

enum { DENSE_ALLOCATION_COUNT = 5 };
static size_t allocation_calls;
static size_t fail_on_call;
static size_t free_calls;
static void *live_blocks[2 * DENSE_ALLOCATION_COUNT];
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
            ++free_calls;
            free(block);
            return;
        }
    }
    CHECK(0 && "free received an untracked or already freed allocation");
}

/*
 * Compile both real implementations with test-only allocator substitutions.
 * Parse their system headers first; the wrappers above use the real allocator.
 * This executable does not also link cinder_core. Production has no fault hooks.
 */
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#include "../src/tensor.c"
#include "../src/dense.c"
#undef malloc
#undef calloc
#undef free

static void reset_allocator(size_t failure)
{
    CHECK(live_count == 0);
    allocation_calls = 0;
    free_calls = 0;
    fail_on_call = failure;
}

static void check_allocation_failure(size_t failure, cinder_error_t *error)
{
    float weights[] = {1.0f, 2.0f, -1.0f, 0.0f, 0.5f, -0.5f, 3.0f, 1.0f};
    float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    float saved_weights[8];
    float saved_bias[4];
    memcpy(saved_weights, weights, sizeof(weights));
    memcpy(saved_bias, bias, sizeof(bias));
    reset_allocator(failure);
    cinder_dense_t sentinel = {0};
    cinder_dense_t *layer = &sentinel;
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, &layer, error)
          == CINDER_ERROR_ALLOCATION);
    CHECK(layer == NULL);
    CHECK(allocation_calls == failure);
    CHECK(free_calls == failure - 1);
    CHECK(live_count == 0);
    CHECK(memcmp(weights, saved_weights, sizeof(weights)) == 0);
    CHECK(memcmp(bias, saved_bias, sizeof(bias)) == 0);
    if (error != NULL) {
        CHECK(error->code == CINDER_ERROR_ALLOCATION);
        CHECK(error->message[0] != '\0');
        CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
    }
    cinder_dense_free(layer);
    CHECK(free_calls == failure - 1);

    /* Recover after every injected failure, using the same error/output slots. */
    reset_allocator(0);
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, &layer, error) == CINDER_OK);
    CHECK(layer != NULL);
    CHECK(allocation_calls == DENSE_ALLOCATION_COUNT);
    CHECK(live_count == DENSE_ALLOCATION_COUNT);
    CHECK(free_calls == 0);
    CHECK(memcmp(layer->weights->data, saved_weights, sizeof(weights)) == 0);
    CHECK(memcmp(layer->bias->data, saved_bias, sizeof(bias)) == 0);
    if (error != NULL) {
        CHECK(error->code == CINDER_OK && error->message[0] == '\0');
    }
    /* Neither normal cleanup nor NULL cleanup may attempt an allocation. */
    fail_on_call = allocation_calls + 1;
    cinder_dense_free(NULL);
    CHECK(live_count == DENSE_ALLOCATION_COUNT && free_calls == 0);
    cinder_dense_free(layer);
    CHECK(live_count == 0);
    CHECK(free_calls == DENSE_ALLOCATION_COUNT);
    CHECK(allocation_calls == DENSE_ALLOCATION_COUNT);
}

static void check_no_allocation(
    size_t input_size, size_t output_size,
    const float *weights, size_t weight_count,
    const float *bias, size_t bias_count,
    cinder_status_t expected
)
{
    reset_allocator(1);
    cinder_dense_t sentinel = {0};
    cinder_dense_t *layer = &sentinel;
    cinder_error_t error;
    CHECK(cinder_dense_create(input_size, output_size, weights, weight_count,
                              bias, bias_count, &layer, &error) == expected);
    CHECK(layer == NULL);
    CHECK(error.code == expected);
    CHECK(allocation_calls == 0 && live_count == 0 && free_calls == 0);
}

static void check_existing_layer_survives(size_t failure)
{
    const float weights[] = {1.0f, 2.0f};
    const float bias[] = {0.5f, -1.0f};
    cinder_dense_t *existing = NULL;
    cinder_dense_t *failed = NULL;
    reset_allocator(0);
    CHECK(cinder_dense_create(1, 2, weights, 2, bias, 2, &existing, NULL) == CINDER_OK);
    CHECK(live_count == DENSE_ALLOCATION_COUNT);
    /* Count only the second constructor's calls while the first layer lives. */
    allocation_calls = 0;
    fail_on_call = failure;
    CHECK(cinder_dense_create(1, 2, weights, 2, bias, 2, &failed, NULL)
          == CINDER_ERROR_ALLOCATION);
    CHECK(failed == NULL);
    CHECK(allocation_calls == failure && free_calls == failure - 1);
    CHECK(live_count == DENSE_ALLOCATION_COUNT);
    CHECK(memcmp(existing->weights->data, weights, sizeof(weights)) == 0);
    CHECK(memcmp(existing->bias->data, bias, sizeof(bias)) == 0);
    cinder_dense_free(existing);
    CHECK(live_count == 0);
    CHECK(free_calls == failure - 1 + DENSE_ALLOCATION_COUNT);
    CHECK(allocation_calls == failure);
}

int main(void)
{
    cinder_error_t error;
    /* Layer, weights struct, weights data, bias struct, bias data. */
    for (size_t failure = 1; failure <= DENSE_ALLOCATION_COUNT; ++failure) {
        memset(&error, 0x7f, sizeof(error));
        check_allocation_failure(failure, &error);
        check_allocation_failure(failure, NULL);
        check_existing_layer_survives(failure);
    }

    const float weights[] = {1.0f, 2.0f, -1.0f, 0.0f, 0.5f, -0.5f, 3.0f, 1.0f};
    const float bias[] = {0.5f, -1.0f, 2.0f, 0.0f};
    reset_allocator(1);
    CHECK(cinder_dense_create(2, 4, weights, 8, bias, 4, NULL, &error)
          == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(allocation_calls == 0 && live_count == 0 && free_calls == 0);
    check_no_allocation(2, 4, NULL, 8, bias, 4, CINDER_ERROR_INVALID_ARGUMENT);
    check_no_allocation(2, 4, weights, 8, NULL, 4, CINDER_ERROR_INVALID_ARGUMENT);
    check_no_allocation(0, 4, weights, 0, bias, 4, CINDER_ERROR_INVALID_ARGUMENT);
    check_no_allocation(2, 0, weights, 0, bias, 0, CINDER_ERROR_INVALID_ARGUMENT);
    check_no_allocation(SIZE_MAX, 2, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW);
    check_no_allocation(2, SIZE_MAX / 2 + 1, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW);
    check_no_allocation(SIZE_MAX / sizeof(float) + 1, 1, weights, 0, bias, 0,
                        CINDER_ERROR_OVERFLOW);
    check_no_allocation(1, SIZE_MAX / sizeof(float) + 1, weights, 0, bias, 0,
                        CINDER_ERROR_OVERFLOW);
    check_no_allocation(2, 4, weights, 7, bias, 4, CINDER_ERROR_DIMENSION_MISMATCH);
    check_no_allocation(2, 4, weights, 9, bias, 4, CINDER_ERROR_DIMENSION_MISMATCH);
    check_no_allocation(2, 4, weights, 8, bias, 3, CINDER_ERROR_DIMENSION_MISMATCH);
    check_no_allocation(2, 4, weights, 8, bias, 5, CINDER_ERROR_DIMENSION_MISMATCH);
    float bad_weights[8];
    float bad_bias[4];
    memcpy(bad_weights, weights, sizeof(weights));
    memcpy(bad_bias, bias, sizeof(bias));
    bad_weights[7] = NAN;
    bad_bias[3] = INFINITY;
    check_no_allocation(2, 4, bad_weights, 8, bias, 4, CINDER_ERROR_NONFINITE);
    check_no_allocation(2, 4, weights, 8, bad_bias, 4, CINDER_ERROR_NONFINITE);

    puts("Dense allocation failure tests passed (all five allocation points).");
    return EXIT_SUCCESS;
}
