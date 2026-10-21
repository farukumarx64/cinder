#include "cinder/layer.h"

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

enum { DENSE_WRAPPER_ALLOCATIONS = 6 };
static size_t allocation_calls;
static size_t fail_on_call;
static size_t free_calls;
static void *live_blocks[2 * DENSE_WRAPPER_ALLOCATIONS];
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

/* Compile the real ownership chain with test-only allocator substitutions.
 * System headers above are parsed before macros; production has no fault hooks. */
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#include "../src/tensor.c"
#include "../src/dense.c"
#include "../src/layer.c"
#undef malloc
#undef calloc
#undef free

static void reset_allocator(size_t failure)
{
    CHECK(live_count == 0);
    allocation_calls = free_calls = 0;
    fail_on_call = failure;
}

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

static void check_creation_failure(cinder_layer_type_t type, size_t failure,
                                   cinder_error_t *error)
{
    const float weights[] = {1, 2, 3, 4, 5, 6};
    const float bias[] = {0, 1, 2};
    cinder_layer_t sentinel = {0};
    cinder_layer_t *layer = &sentinel;
    size_t required = type == CINDER_LAYER_DENSE ? DENSE_WRAPPER_ALLOCATIONS : 1;
    reset_allocator(failure);
    cinder_status_t status = type == CINDER_LAYER_DENSE
        ? cinder_layer_create_dense(2, 3, weights, 6, bias, 3, &layer, error)
        : cinder_layer_create_activation(type, 3, &layer, error);
    check_result(status, CINDER_ERROR_ALLOCATION, error);
    CHECK(layer == NULL && live_count == 0);
    CHECK(allocation_calls == failure && free_calls == failure - 1);
    cinder_layer_free(layer);
    CHECK(free_calls == failure - 1);

    reset_allocator(0);
    status = type == CINDER_LAYER_DENSE
        ? cinder_layer_create_dense(2, 3, weights, 6, bias, 3, &layer, error)
        : cinder_layer_create_activation(type, 3, &layer, error);
    check_result(status, CINDER_OK, error);
    CHECK(layer != NULL && layer->type == type);
    CHECK(allocation_calls == required && live_count == required && free_calls == 0);
    if (type == CINDER_LAYER_DENSE) {
        CHECK(memcmp(layer->dense->weights->data, weights, sizeof(weights)) == 0);
        CHECK(memcmp(layer->dense->bias->data, bias, sizeof(bias)) == 0);
    } else {
        CHECK(layer->dense == NULL && layer->input_size == 3 && layer->output_size == 3);
    }
    fail_on_call = allocation_calls + 1;
    check_result(cinder_layer_validate(layer, error), CINDER_OK, error);
    cinder_layer_free(NULL);
    CHECK(allocation_calls == required && live_count == required && free_calls == 0);
    cinder_layer_free(layer);
    CHECK(allocation_calls == required && live_count == 0 && free_calls == required);
}

static void check_invalid_creation(cinder_error_t *error)
{
    const float weights[] = {1, 2, 3, 4, 5, 6};
    const float bias[] = {0, 1, 2};
    const float bad_bias[] = {0, 1, INFINITY};
    const float bad_weights[] = {1, 2, 3, 4, 5, NAN};
    reset_allocator(1);
    check_result(cinder_layer_create_dense(2, 3, weights, 6, bias, 3, NULL, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_layer_create_activation(CINDER_LAYER_RELU, 3, NULL, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    const struct {
        size_t inputs, outputs;
        const float *weights;
        size_t weight_count;
        const float *bias;
        size_t bias_count;
        cinder_status_t expected;
    } cases[] = {
        {0, 3, weights, 6, bias, 3, CINDER_ERROR_INVALID_ARGUMENT},
        {2, 0, weights, 6, bias, 3, CINDER_ERROR_INVALID_ARGUMENT},
        {2, 3, NULL, 6, bias, 3, CINDER_ERROR_INVALID_ARGUMENT},
        {2, 3, weights, 6, NULL, 3, CINDER_ERROR_INVALID_ARGUMENT},
        {SIZE_MAX, 2, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW},
        {SIZE_MAX / sizeof(float) + 1, 1, weights, 0, bias, 0, CINDER_ERROR_OVERFLOW},
        {2, 3, weights, 5, bias, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {2, 3, weights, 7, bias, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {2, 3, weights, 6, bias, 2, CINDER_ERROR_DIMENSION_MISMATCH},
        {2, 3, weights, 6, bias, 4, CINDER_ERROR_DIMENSION_MISMATCH},
        {2, 3, bad_weights, 6, bias, 3, CINDER_ERROR_NONFINITE},
        {2, 3, weights, 6, bad_bias, 3, CINDER_ERROR_NONFINITE}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        cinder_layer_t sentinel = {0};
        cinder_layer_t *layer = &sentinel;
        check_result(cinder_layer_create_dense(cases[i].inputs, cases[i].outputs,
                     cases[i].weights, cases[i].weight_count, cases[i].bias,
                     cases[i].bias_count, &layer, error), cases[i].expected, error);
        CHECK(layer == NULL);
        if (error != NULL) {
            cinder_dense_t *dense = NULL;
            cinder_error_t dense_error;
            CHECK(cinder_dense_create(cases[i].inputs, cases[i].outputs,
                  cases[i].weights, cases[i].weight_count, cases[i].bias,
                  cases[i].bias_count, &dense, &dense_error) == cases[i].expected);
            CHECK(dense == NULL && strcmp(error->message, dense_error.message) == 0);
        }
    }
    const cinder_layer_type_t types[] = {
        CINDER_LAYER_INVALID, CINDER_LAYER_DENSE, (cinder_layer_type_t)99,
        CINDER_LAYER_RELU, CINDER_LAYER_SOFTMAX
    };
    for (size_t i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
        cinder_layer_t sentinel = {0};
        cinder_layer_t *layer = &sentinel;
        check_result(cinder_layer_create_activation(types[i], 0, &layer, error),
                     CINDER_ERROR_INVALID_ARGUMENT, error);
        CHECK(layer == NULL);
        layer = &sentinel;
        check_result(cinder_layer_create_activation(types[i], SIZE_MAX, &layer, error),
                     i < 3 ? CINDER_ERROR_INVALID_ARGUMENT : CINDER_ERROR_OVERFLOW, error);
        CHECK(layer == NULL);
    }
    CHECK(allocation_calls == 0 && free_calls == 0 && live_count == 0);
}

static void check_no_working_allocations(cinder_layer_type_t type, cinder_error_t *error)
{
    const float weights[] = {1, 0, 0, 0, FLT_MAX, 0, 0, 0, 1};
    const float bias[] = {0.5f, 0, -1};
    cinder_layer_t *layer = NULL;
    size_t required = type == CINDER_LAYER_DENSE ? DENSE_WRAPPER_ALLOCATIONS : 1;
    reset_allocator(0);
    cinder_status_t status = type == CINDER_LAYER_DENSE
        ? cinder_layer_create_dense(3, 3, weights, 9, bias, 3, &layer, error)
        : cinder_layer_create_activation(type, 3, &layer, error);
    check_result(status, CINDER_OK, error);
    CHECK(allocation_calls == required && live_count == required);
    /* Fail the very next allocation, and count even attempted allocations/frees. */
    fail_on_call = allocation_calls + 1;
    const float input[] = {-1, 0, 1};
    float output[] = {NAN, NAN, NAN, NAN};
    check_result(cinder_layer_validate(layer, error), CINDER_OK, error);
    check_result(cinder_layer_validate(NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_layer_forward(NULL, input, 3, output, 4, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_layer_forward(layer, NULL, 3, output, 4, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_layer_forward(layer, input, 3, NULL, 4, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_layer_forward(layer, input, 2, output, 4, error),
                 CINDER_ERROR_DIMENSION_MISMATCH, error);
    check_result(cinder_layer_forward(layer, input, 3, output, 2, error),
                 CINDER_ERROR_BUFFER_TOO_SMALL, error);
    const float bad_input[] = {-1, 0, NAN};
    check_result(cinder_layer_forward(layer, bad_input, 3, output, 4, error),
                 CINDER_ERROR_NONFINITE, error);
    if (type == CINDER_LAYER_DENSE) {
        const float overflow[] = {0, 2, 0};
        check_result(cinder_layer_forward(layer, overflow, 3, output, 4, error),
                     CINDER_ERROR_NONFINITE, error);
    }
    CHECK(allocation_calls == required && free_calls == 0 && live_count == required);
    const float zero[] = {0, 0, 0};
    for (size_t i = 0; i < 100; ++i) {
        check_result(cinder_layer_forward(layer, i % 2 ? zero : input, 3, output, 4, error),
                     CINDER_OK, error);
        CHECK(allocation_calls == required && free_calls == 0 && live_count == required);
        CHECK(isnan(output[3]));
        CHECK(layer->type == type && layer->input_size == 3 && layer->output_size == 3);
        if (type == CINDER_LAYER_DENSE) {
            CHECK(memcmp(layer->dense->weights->data, weights, sizeof(weights)) == 0);
            CHECK(memcmp(layer->dense->bias->data, bias, sizeof(bias)) == 0);
        } else {
            CHECK(layer->dense == NULL);
        }
    }
    cinder_layer_free(layer);
    CHECK(allocation_calls == required && free_calls == required && live_count == 0);
}

static void check_existing_owner_survives(void)
{
    const float weights[] = {1, 2};
    const float bias[] = {0, 1};
    for (size_t failure = 1; failure <= DENSE_WRAPPER_ALLOCATIONS; ++failure) {
        reset_allocator(0);
        cinder_layer_t *existing = NULL;
        cinder_layer_t *failed = NULL;
        CHECK(cinder_layer_create_activation(CINDER_LAYER_RELU, 2, &existing, NULL) == CINDER_OK);
        allocation_calls = 0;
        fail_on_call = failure;
        CHECK(cinder_layer_create_dense(1, 2, weights, 2, bias, 2, &failed, NULL)
              == CINDER_ERROR_ALLOCATION);
        CHECK(failed == NULL && live_count == 1 && free_calls == failure - 1);
        const float input[] = {-1, 2};
        float output[2];
        CHECK(cinder_layer_forward(existing, input, 2, output, 2, NULL) == CINDER_OK);
        CHECK(output[0] == 0 && output[1] == 2 && allocation_calls == failure);
        cinder_layer_free(existing);
        CHECK(live_count == 0 && free_calls == failure);
    }
}

int main(void)
{
    for (int with_error = 0; with_error <= 1; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        /* Dense's five allocations, followed by the common wrapper. */
        for (size_t failure = 1; failure <= DENSE_WRAPPER_ALLOCATIONS; ++failure) {
            check_creation_failure(CINDER_LAYER_DENSE, failure, error);
        }
        check_creation_failure(CINDER_LAYER_RELU, 1, error);
        check_creation_failure(CINDER_LAYER_SOFTMAX, 1, error);
        check_invalid_creation(error);
        check_no_working_allocations(CINDER_LAYER_DENSE, error);
        check_no_working_allocations(CINDER_LAYER_RELU, error);
        check_no_working_allocations(CINDER_LAYER_SOFTMAX, error);
    }
    check_existing_owner_survives();
    puts("Layer allocation tests passed: six Dense failures, one per activation, no working allocations.");
    return EXIT_SUCCESS;
}
