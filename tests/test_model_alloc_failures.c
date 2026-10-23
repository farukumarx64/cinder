#include "cinder/model.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

enum { MODEL_ALLOCATIONS = 4 };
static size_t allocation_calls;
static size_t fail_on_call;
static size_t free_calls;
static size_t attempted_sizes[128];
static struct { void *pointer; size_t size; } live_blocks[64];
static size_t live_count;

static int fail_allocation(size_t size)
{
    CHECK(allocation_calls < COUNT(attempted_sizes));
    attempted_sizes[allocation_calls++] = size;
    return allocation_calls == fail_on_call;
}

static void remember_block(void *block, size_t size)
{
    CHECK(block != NULL && live_count < COUNT(live_blocks));
    live_blocks[live_count].pointer = block;
    live_blocks[live_count].size = size;
    ++live_count;
}

static size_t block_size(const void *block)
{
    for (size_t i = 0; i < live_count; ++i) {
        if (live_blocks[i].pointer == block) {
            return live_blocks[i].size;
        }
    }
    CHECK(0 && "expected a live allocation");
    return 0;
}

static void *test_malloc(size_t size)
{
    if (fail_allocation(size)) {
        return NULL;
    }
    void *block = malloc(size);
    remember_block(block, size);
    return block;
}

static void *test_calloc(size_t count, size_t size)
{
    CHECK(count == 0 || size <= SIZE_MAX / count);
    if (fail_allocation(count * size)) {
        return NULL;
    }
    void *block = calloc(count, size);
    remember_block(block, count * size);
    return block;
}

static void test_free(void *block)
{
    if (block == NULL) {
        return;
    }
    for (size_t i = 0; i < live_count; ++i) {
        if (live_blocks[i].pointer == block) {
            --live_count;
            live_blocks[i] = live_blocks[live_count];
            live_blocks[live_count].pointer = NULL;
            live_blocks[live_count].size = 0;
            ++free_calls;
            free(block);
            return;
        }
    }
    CHECK(0 && "free received an untracked or already freed allocation");
}

/* Compile the complete real ownership chain with test-only substitutions.
 * Including model.c also permits workspace inspection without a public API or
 * production testing hooks. This executable must not also link cinder_core. */
#define malloc test_malloc
#define calloc test_calloc
#define free test_free
#include "../src/tensor.c"
#include "../src/dense.c"
#include "../src/layer.c"
/* Observe dispatch only inside model.c; forward real calls to the tested layer
 * implementation. Fault injection checks short-circuiting even at activations,
 * whose valid finite inputs normally cannot produce an arithmetic failure. */
static size_t forward_calls;
static size_t fail_on_forward;
static struct {
    const cinder_layer_t *layer;
    const float *input;
    float *output;
    size_t count;
    size_t capacity;
} forward_trace[8];

static cinder_status_t observed_forward(
    const cinder_layer_t *layer, const float *input, size_t count,
    float *output, size_t capacity, cinder_error_t *error)
{
    CHECK(forward_calls < COUNT(forward_trace));
    forward_trace[forward_calls].layer = layer;
    forward_trace[forward_calls].input = input;
    forward_trace[forward_calls].output = output;
    forward_trace[forward_calls].count = count;
    forward_trace[forward_calls].capacity = capacity;
    ++forward_calls;
    if (forward_calls == fail_on_forward) {
        /* Simulate a layer failure after a partial write. */
        output[0] = NAN;
        if (error != NULL) {
            error->code = CINDER_ERROR_NONFINITE;
            strcpy(error->message, "Injected layer failure after partial output.");
        }
        return CINDER_ERROR_NONFINITE;
    }
    return cinder_layer_forward(layer, input, count, output, capacity, error);
}

#define cinder_layer_forward observed_forward
#include "../src/model.c"
#undef cinder_layer_forward
#undef malloc
#undef calloc
#undef free

static const float weights[] = {1, 2, -1, 0, 0.5f, -0.5f, 3, 1};
static const float bias[] = {0.5f, -1, 2, 0};

static void reset_allocator(void)
{
    CHECK(live_count == 0);
    allocation_calls = free_calls = fail_on_call = 0;
}

static cinder_layer_t *activation(cinder_layer_type_t type, size_t size)
{
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_activation(type, size, &layer, NULL) == CINDER_OK);
    return layer;
}

static cinder_layer_t *dense(size_t inputs, size_t outputs)
{
    CHECK(inputs * outputs <= COUNT(weights) && outputs <= COUNT(bias));
    cinder_layer_t *layer = NULL;
    CHECK(cinder_layer_create_dense(inputs, outputs, weights, inputs * outputs,
                                    bias, outputs, &layer, NULL) == CINDER_OK);
    return layer;
}

static void example_layers(cinder_layer_t **layers)
{
    layers[0] = dense(2, 4);
    layers[1] = activation(CINDER_LAYER_RELU, 4);
    layers[2] = dense(4, 2);
    layers[3] = activation(CINDER_LAYER_SOFTMAX, 2);
}

static void check_parameters(const cinder_layer_t *layer)
{
    CHECK(block_size(layer) == sizeof(*layer));
    if (layer->type == CINDER_LAYER_DENSE) {
        size_t count = layer->input_size * layer->output_size;
        CHECK(block_size(layer->dense) == sizeof(*layer->dense));
        CHECK(block_size(layer->dense->weights) == sizeof(cinder_tensor_t));
        CHECK(block_size(layer->dense->bias) == sizeof(cinder_tensor_t));
        CHECK(block_size(layer->dense->weights->data) == count * sizeof(float));
        CHECK(block_size(layer->dense->bias->data) == layer->output_size * sizeof(float));
        CHECK(memcmp(layer->dense->weights->data, weights, count * sizeof(float)) == 0);
        CHECK(memcmp(layer->dense->bias->data, bias, layer->output_size * sizeof(float)) == 0);
    } else {
        CHECK(layer->dense == NULL);
    }
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

static void check_private_storage(cinder_model_t *model, cinder_layer_t **expected,
                                   size_t count, size_t capacity)
{
    CHECK(block_size(model) == sizeof(*model));
    CHECK(block_size(model->layers) == count * sizeof(*model->layers));
    CHECK(model->layer_count == count && model->buffer_capacity == capacity);
    CHECK(cinder_model_input_size(model) == expected[0]->input_size);
    CHECK(cinder_model_output_size(model) == expected[count - 1]->output_size);
    CHECK(model->buffers[0] != model->buffers[1]);
    CHECK(block_size(model->buffers[0]) == capacity * sizeof(float));
    CHECK(block_size(model->buffers[1]) == capacity * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        CHECK(model->layers[i] == expected[i]);
        check_parameters(expected[i]);
    }
    /* Exercise the full writable extent and independence of the two buffers.
     * Initial contents are intentionally unspecified; never assume zeroes. */
    for (size_t i = 0; i < capacity; ++i) {
        model->buffers[0][i] = (float)i + 1;
        model->buffers[1][i] = -(float)i - 1;
    }
    for (size_t i = 0; i < capacity; ++i) {
        CHECK(model->buffers[0][i] == (float)i + 1);
        CHECK(model->buffers[1][i] == -(float)i - 1);
    }
    for (size_t i = 0; i < count; ++i) {
        check_parameters(expected[i]);
    }
}

static void check_failure(size_t failure, int retry, cinder_error_t *error)
{
    reset_allocator();
    cinder_layer_t *layers[4];
    example_layers(layers);
    cinder_layer_t *saved[4];
    memcpy(saved, layers, sizeof(saved));
    size_t layer_allocations = allocation_calls;
    CHECK(live_count == layer_allocations && free_calls == 0);
    fail_on_call = allocation_calls + failure;
    cinder_model_t sentinel = {0};
    cinder_model_t *model = &sentinel;
    check_result(cinder_model_create(layers, 4, &model, error), CINDER_ERROR_ALLOCATION, error);
    CHECK(model == NULL && memcmp(layers, saved, sizeof(saved)) == 0);
    CHECK(allocation_calls == layer_allocations + failure);
    CHECK(free_calls == failure - 1 && live_count == layer_allocations);
    for (size_t i = 0; i < 4; ++i) {
        check_parameters(layers[i]);
        CHECK(cinder_layer_validate(layers[i], NULL) == CINDER_OK);
    }
    /* Failure must leave layers executable by their original owner. */
    const float input[] = {2, -1};
    float output[4];
    CHECK(cinder_layer_forward(layers[0], input, 2, output, 4, NULL) == CINDER_OK);
    CHECK(output[0] == 0.5f && output[1] == -3 && output[2] == 3.5f && output[3] == 5);
    CHECK(allocation_calls == layer_allocations + failure && free_calls == failure - 1);
    cinder_model_free(model);
    CHECK(free_calls == failure - 1);

    if (retry) {
        fail_on_call = 0;
        check_result(cinder_model_create(layers, 4, &model, error), CINDER_OK, error);
        CHECK(live_count == layer_allocations + MODEL_ALLOCATIONS);
        for (size_t i = 0; i < 4; ++i) {
            CHECK(layers[i] == NULL);
        }
        check_private_storage(model, saved, 4, 4);
        fail_on_call = allocation_calls + 1;
        size_t calls_before_cleanup = allocation_calls;
        for (size_t i = 0; i < 100; ++i) {
            CHECK(cinder_model_input_size(model) == 2 && cinder_model_output_size(model) == 2);
        }
        CHECK(free_calls == failure - 1);
        cinder_model_free(model);
        CHECK(allocation_calls == calls_before_cleanup);
    } else {
        /* The caller can instead abandon construction and free its own layers. */
        fail_on_call = allocation_calls + 1;
        for (size_t i = 0; i < 4; ++i) {
            cinder_layer_free(layers[i]);
        }
        CHECK(allocation_calls == layer_allocations + failure);
    }
    CHECK(live_count == 0 && free_calls == allocation_calls - 1);
}

static void check_no_allocation(cinder_layer_t **layers, size_t count,
                                cinder_status_t expected, cinder_error_t *error)
{
    size_t before = allocation_calls;
    size_t before_live = live_count;
    size_t before_free = free_calls;
    fail_on_call = before + 1;
    cinder_model_t sentinel = {0};
    cinder_model_t *model = &sentinel;
    check_result(cinder_model_create(layers, count, &model, error), expected, error);
    CHECK(model == NULL && allocation_calls == before);
    CHECK(live_count == before_live && free_calls == before_free);
}

static void check_validation_without_allocation(cinder_error_t *error)
{
    reset_allocator();
    cinder_layer_t *layers[4];
    example_layers(layers);
    cinder_layer_t *saved[4];
    memcpy(saved, layers, sizeof(saved));
    fail_on_call = allocation_calls + 1;
    check_result(cinder_model_create(layers, 4, NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    check_no_allocation(NULL, 1, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_no_allocation(layers, 0, CINDER_ERROR_INVALID_ARGUMENT, error);
    check_no_allocation(layers, SIZE_MAX / sizeof(*layers) + 1, CINDER_ERROR_OVERFLOW, error);
    layers[3] = NULL;
    check_no_allocation(layers, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    layers[3] = saved[3];
    layers[1] = layers[0];
    check_no_allocation(layers, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    layers[1] = saved[1];
    /* Test-only metadata faults are restored before ownership changes/free. */
    layers[1]->type = CINDER_LAYER_INVALID;
    check_no_allocation(layers, 4, CINDER_ERROR_INVALID_ARGUMENT, error);
    layers[1]->type = CINDER_LAYER_RELU;
    layers[1]->input_size = layers[1]->output_size = SIZE_MAX / sizeof(float) + 1;
    check_no_allocation(layers, 4, CINDER_ERROR_OVERFLOW, error);
    layers[1]->input_size = layers[1]->output_size = 3;
    check_no_allocation(layers, 4, CINDER_ERROR_DIMENSION_MISMATCH, error);
    layers[1]->input_size = layers[1]->output_size = 4;
    CHECK(memcmp(layers, saved, sizeof(saved)) == 0);
    CHECK(free_calls == 0);
    for (size_t i = 0; i < 4; ++i) {
        check_parameters(layers[i]);
        cinder_layer_free(layers[i]);
    }
    CHECK(live_count == 0 && free_calls == allocation_calls);
    size_t before = allocation_calls;
    CHECK(cinder_model_input_size(NULL) == 0 && cinder_model_output_size(NULL) == 0);
    cinder_model_free(NULL);
    CHECK(allocation_calls == before && free_calls == before);
}

static void check_capacity_case(cinder_layer_t **layers, size_t count, size_t capacity)
{
    cinder_layer_t *saved[4];
    CHECK(count <= COUNT(saved));
    memcpy(saved, layers, count * sizeof(*layers));
    size_t before = allocation_calls;
    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, count, &model, NULL) == CINDER_OK);
    CHECK(allocation_calls == before + MODEL_ALLOCATIONS);
    CHECK(model->layers != layers);
    for (size_t i = 0; i < count; ++i) {
        CHECK(layers[i] == NULL);
    }
    check_private_storage(model, saved, count, capacity);
    /* Reusing a source slot must not replace any adopted pointer. */
    layers[0] = activation(CINDER_LAYER_RELU, 1);
    fail_on_call = allocation_calls + 1;
    size_t before_free = allocation_calls;
    cinder_model_free(model);
    CHECK(allocation_calls == before_free && live_count == 1);
    CHECK(cinder_layer_validate(layers[0], NULL) == CINDER_OK);
    cinder_layer_free(layers[0]);
    CHECK(live_count == 0 && free_calls == allocation_calls);
}

static void check_capacities(void)
{
    reset_allocator();
    cinder_layer_t *example[4];
    example_layers(example);
    check_capacity_case(example, 4, 4);
    const size_t shapes[][2] = {{1, 1}, {4, 1}, {1, 4}};
    const size_t capacities[] = {1, 4, 4};
    for (size_t i = 0; i < COUNT(shapes); ++i) {
        reset_allocator();
        cinder_layer_t *layers[] = {dense(shapes[i][0], shapes[i][1])};
        check_capacity_case(layers, 1, capacities[i]);
    }
    reset_allocator();
    cinder_layer_t *middle_maximum[] = {dense(3, 2), dense(2, 4), dense(4, 1)};
    check_capacity_case(middle_maximum, 3, 4);
    reset_allocator();
    cinder_layer_t *activations[] = {activation(CINDER_LAYER_RELU, 33),
                                    activation(CINDER_LAYER_SOFTMAX, 33)};
    check_capacity_case(activations, 2, 33);
}

static void check_largest_representable_buffer(void)
{
    reset_allocator();
    size_t capacity = SIZE_MAX / sizeof(float);
    cinder_layer_t *layers[] = {activation(CINDER_LAYER_RELU, capacity)};
    cinder_layer_t *saved = layers[0];
    size_t before = allocation_calls;
    /* Refuse the first working-buffer allocation BEFORE calling real malloc.
     * The largest byte-representable vector must reach allocation, not overflow. */
    fail_on_call = before + 3;
    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, 1, &model, NULL) == CINDER_ERROR_ALLOCATION);
    CHECK(model == NULL && layers[0] == saved);
    CHECK(allocation_calls == before + 3 && free_calls == 2 && live_count == 1);
    CHECK(attempted_sizes[before + 2] == capacity * sizeof(float));
    cinder_layer_free(layers[0]);
    CHECK(live_count == 0 && free_calls == allocation_calls - 1);
}

static void check_existing_model_survives(void)
{
    for (size_t failure = 1; failure <= MODEL_ALLOCATIONS; ++failure) {
        reset_allocator();
        cinder_layer_t *single[] = {activation(CINDER_LAYER_RELU, 3)};
        cinder_model_t *existing = NULL;
        CHECK(cinder_model_create(single, 1, &existing, NULL) == CINDER_OK);
        for (size_t i = 0; i < 3; ++i) {
            existing->buffers[0][i] = 17;
            existing->buffers[1][i] = 23;
        }
        cinder_layer_t *layers[4];
        example_layers(layers);
        size_t before = allocation_calls;
        size_t before_live = live_count;
        fail_on_call = before + failure;
        cinder_model_t *failed = NULL;
        CHECK(cinder_model_create(layers, 4, &failed, NULL) == CINDER_ERROR_ALLOCATION);
        CHECK(failed == NULL && live_count == before_live);
        CHECK(allocation_calls == before + failure && free_calls == failure - 1);
        CHECK(cinder_model_input_size(existing) == 3 && cinder_model_output_size(existing) == 3);
        for (size_t i = 0; i < 3; ++i) {
            CHECK(existing->buffers[0][i] == 17 && existing->buffers[1][i] == 23);
        }
        for (size_t i = 0; i < 4; ++i) {
            check_parameters(layers[i]);
            cinder_layer_free(layers[i]);
        }
        CHECK(live_count == 1 + MODEL_ALLOCATIONS);
        cinder_model_free(existing);
        CHECK(live_count == 0 && free_calls == allocation_calls - 1);
    }
}

static void check_run_storage(cinder_error_t *error)
{
    reset_allocator();
    cinder_layer_t *layers[4];
    example_layers(layers);
    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, 4, &model, NULL) == CINDER_OK);
    const float inputs[][2] = {{2, -1}, {0, 0}, {-1, 2}};
    size_t allocations_before = allocation_calls, frees_before = free_calls;
    size_t live_before = live_count;
    /* An attempted allocation would fail; it must not even be attempted. */
    fail_on_call = allocation_calls + 1;
    for (size_t failure = 0; failure <= 4; ++failure) {
        for (size_t example = 0; example < COUNT(inputs); ++example) {
            /* Poison both buffers, including spare tails, before every run. */
            for (size_t b = 0; b < 2; ++b) {
                for (size_t i = 0; i < model->buffer_capacity; ++i) {
                    model->buffers[b][i] = NAN;
                }
            }
            float output[] = {91, 92, NAN}, saved[3];
            memcpy(saved, output, sizeof(output));
            forward_calls = 0;
            fail_on_forward = failure;
            check_result(cinder_model_run(model, inputs[example], 2, output, 3, error),
                         failure == 0 ? CINDER_OK : CINDER_ERROR_NONFINITE, error);
            size_t calls = failure == 0 ? 4 : failure;
            CHECK(forward_calls == calls);
            /* Initial input copy -> A; Dense A->B; ReLU B->B;
             * Dense B->A; Softmax A->A. Counts shrink to two, capacity stays four. */
            const size_t reads[] = {0, 1, 1, 0}, writes[] = {1, 1, 0, 0};
            const size_t counts[] = {2, 4, 4, 2};
            for (size_t i = 0; i < calls; ++i) {
                CHECK(forward_trace[i].layer == model->layers[i]);
                CHECK(forward_trace[i].input == model->buffers[reads[i]]);
                CHECK(forward_trace[i].output == model->buffers[writes[i]]);
                CHECK(forward_trace[i].count == counts[i]);
                CHECK(forward_trace[i].capacity == 4);
            }
            if (failure != 0) {
                CHECK(memcmp(output, saved, sizeof(output)) == 0);
                if (error != NULL) {
                    CHECK(strcmp(error->message,
                                 "Injected layer failure after partial output.") == 0);
                }
                /* Retry immediately with the partially written buffers intact. */
                fail_on_forward = forward_calls = 0;
                check_result(cinder_model_run(model, inputs[example], 2, output, 3, error),
                             CINDER_OK, error);
                CHECK(forward_calls == 4);
            }
            CHECK(isfinite(output[0]) && isfinite(output[1]) && isnan(output[2]));
            CHECK(fabs((double)output[0] + output[1] - 1.0) < 1e-6);
            CHECK(isnan(model->buffers[0][2]) && isnan(model->buffers[0][3]));
            CHECK(allocation_calls == allocations_before && free_calls == frees_before);
            CHECK(live_count == live_before);
            for (size_t i = 0; i < 4; ++i) {
                check_parameters(model->layers[i]);
            }
        }
    }
    /* Every preflight rejection skips dispatch and allocates/frees nothing. */
    forward_calls = fail_on_forward = 0;
    float output[] = {91, 92, NAN}, saved[3];
    memcpy(saved, output, sizeof(output));
    check_result(cinder_model_run(NULL, inputs[0], 2, output, 3, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_model_run(model, NULL, 2, output, 3, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_model_run(model, inputs[0], 2, NULL, 3, error),
                 CINDER_ERROR_INVALID_ARGUMENT, error);
    check_result(cinder_model_run(model, inputs[0], 1, output, 3, error),
                 CINDER_ERROR_DIMENSION_MISMATCH, error);
    check_result(cinder_model_run(model, inputs[0], 2, output, 1, error),
                 CINDER_ERROR_BUFFER_TOO_SMALL, error);
    const float bad[] = {1, NAN};
    check_result(cinder_model_run(model, bad, 2, output, 3, error),
                 CINDER_ERROR_NONFINITE, error);
    CHECK(forward_calls == 0 && memcmp(output, saved, sizeof(output)) == 0);
    CHECK(allocation_calls == allocations_before && free_calls == frees_before);
    cinder_model_free(model);
    CHECK(live_count == 0 && free_calls == allocation_calls);
}

static void check_run_arithmetic_without_allocation(cinder_error_t *error)
{
    for (int bias_failure = 0; bias_failure <= 1; ++bias_failure) {
        reset_allocator();
        const float w[] = {1, 0, FLT_MAX, 0};
        const float b[] = {0, bias_failure ? FLT_MAX : 0};
        cinder_layer_t *layers[] = {activation(CINDER_LAYER_RELU, 2), NULL,
                                    activation(CINDER_LAYER_RELU, 2)};
        CHECK(cinder_layer_create_dense(2, 2, w, 4, b, 2, &layers[1], NULL) == CINDER_OK);
        cinder_model_t *model = NULL;
        CHECK(cinder_model_create(layers, 3, &model, NULL) == CINDER_OK);
        const float input[] = {bias_failure ? 1.0f : 2.0f, 0};
        float output[] = {91, 92, NAN}, saved[3];
        memcpy(saved, output, sizeof(output));
        size_t allocations_before = allocation_calls, frees_before = free_calls;
        fail_on_call = allocation_calls + 1;
        forward_calls = fail_on_forward = 0;
        check_result(cinder_model_run(model, input, 2, output, 3, error),
                     CINDER_ERROR_NONFINITE, error);
        CHECK(forward_calls == 2 && memcmp(output, saved, sizeof(output)) == 0);
        const float zero[] = {0, 0};
        forward_calls = 0;
        check_result(cinder_model_run(model, zero, 2, output, 3, error), CINDER_OK, error);
        CHECK(forward_calls == 3 && output[0] == b[0] && output[1] == b[1]);
        CHECK(isnan(output[2]));
        CHECK(allocation_calls == allocations_before && free_calls == frees_before);
        cinder_model_free(model);
        CHECK(live_count == 0 && free_calls == allocation_calls);
    }
}

int main(void)
{
    for (int with_error = 0; with_error <= 1; ++with_error) {
        cinder_error_t details;
        memset(&details, 0x7f, sizeof(details));
        cinder_error_t *error = with_error ? &details : NULL;
        for (size_t failure = 1; failure <= MODEL_ALLOCATIONS; ++failure) {
            check_failure(failure, 0, error);
            check_failure(failure, 1, error);
        }
        check_validation_without_allocation(error);
        check_run_storage(error);
        check_run_arithmetic_without_allocation(error);
    }
    check_capacities();
    check_largest_representable_buffer();
    check_existing_model_survives();
    puts("Model storage tests passed: allocation rollback, ownership, buffer reuse, and allocation-free inference.");
    return EXIT_SUCCESS;
}
