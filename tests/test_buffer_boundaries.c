#include "cinder/model.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/* Assertions must remain active with -DNDEBUG. A stray access into a guard
 * page or a write to read-only input terminates the test, including in Release. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

typedef struct {
    void *mapping;
    size_t mapping_bytes;
    float *values;
} guarded_vector_t;

/* Run at each end of the accessible page: one orientation catches underreads/
 * underwrites, the other overreads/overwrites. All advertised extents are valid;
 * none of these tests passes an invalid pointer or fabricated tensor object. */
static guarded_vector_t guarded(const float *values, size_t count, bool at_end, bool read_only)
{
    long page_size = sysconf(_SC_PAGESIZE);
    CHECK(page_size > 0 && (uintmax_t)page_size <= SIZE_MAX / 3);
    size_t page = (size_t)page_size;
    CHECK(count > 0 && count <= page / sizeof(float));
    void *mapping = mmap(NULL, page * 3, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    CHECK(mapping != MAP_FAILED);
    unsigned char *accessible = (unsigned char *)mapping + page;
    CHECK(mprotect(accessible, page, PROT_READ | PROT_WRITE) == 0);
    float *data = (float *)(accessible + (at_end ? page - count * sizeof(float) : 0));
    memcpy(data, values, count * sizeof(float));
    if (read_only) {
        CHECK(mprotect(accessible, page, PROT_READ) == 0);
    }
    return (guarded_vector_t){mapping, page * 3, data};
}

static void release(guarded_vector_t *vector)
{
    CHECK(munmap(vector->mapping, vector->mapping_bytes) == 0);
    *vector = (guarded_vector_t){0};
}

static void equal(const float *actual, const float *expected, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        CHECK(isfinite(actual[i]) && fabsf(actual[i] - expected[i]) <= 1e-6f);
    }
}

static void boundaries(bool at_end)
{
    const float weights[] = {1, 2, -1, 0, .5f, -.5f};
    const float biases[] = {.5f, -1, 2};
    const float input[] = {2, -1};
    const float invalid[] = {2, NAN};
    const float overflow[] = {FLT_MAX, FLT_MAX};
    const float activation[] = {-1, 0, 1};
    const float sentinel[] = {91, 92, 93};
    const float product[] = {0, -2, 1.5f};
    const float affine[] = {.5f, -3, 3.5f};
    const float activated[] = {.5f, 0, 3.5f};
    const float relu_expected[] = {0, 0, 1};
    const float softmax_expected[] = {.0900305732f, .244728476f, .665240943f};
    guarded_vector_t w = guarded(weights, 6, at_end, true);
    guarded_vector_t b = guarded(biases, 3, at_end, true);
    guarded_vector_t x = guarded(input, 2, at_end, true);
    guarded_vector_t bad = guarded(invalid, 2, at_end, true);
    guarded_vector_t huge = guarded(overflow, 2, at_end, true);
    guarded_vector_t a = guarded(activation, 3, at_end, true);
    guarded_vector_t y = guarded(sentinel, 3, at_end, false);
    guarded_vector_t short_y = guarded(sentinel, 2, at_end, false);

    cinder_tensor_t *tensor = NULL;
    CHECK(cinder_tensor_create(3, 2, &tensor, NULL) == CINDER_OK);
    memcpy(tensor->data, w.values, sizeof(weights));
    CHECK(cinder_tensor_matvec(tensor, x.values, 2, y.values, 3, NULL) == CINDER_OK);
    equal(y.values, product, 3);
    CHECK(cinder_tensor_add_bias(y.values, 3, b.values, 3, NULL) == CINDER_OK);
    equal(y.values, affine, 3);
    CHECK(cinder_tensor_relu(y.values, 3, NULL) == CINDER_OK);
    equal(y.values, activated, 3);
    memcpy(y.values, a.values, sizeof(activation));
    CHECK(cinder_tensor_softmax(y.values, 3, NULL) == CINDER_OK);
    equal(y.values, softmax_expected, 3);
    CHECK(cinder_tensor_matvec(tensor, x.values, 2, short_y.values, 2, NULL) == CINDER_ERROR_BUFFER_TOO_SMALL);
    equal(short_y.values, sentinel, 2);
    cinder_tensor_free(tensor);

    cinder_dense_t *dense = NULL;
    cinder_layer_t *layers[2] = {NULL, NULL};
    CHECK(cinder_dense_create(2, 3, w.values, 6, b.values, 3, &dense, NULL) == CINDER_OK);
    CHECK(cinder_layer_create_dense(2, 3, w.values, 6, b.values, 3, &layers[0], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_activation(CINDER_LAYER_RELU, 3, &layers[1], NULL) == CINDER_OK);
    /* Constructors must have copied parameters. Revoke access to the caller's
     * originals but reserve their addresses so later allocation cannot reuse
     * them and accidentally hide a retained pointer. */
    CHECK(mprotect(w.mapping, w.mapping_bytes, PROT_NONE) == 0);
    CHECK(mprotect(b.mapping, b.mapping_bytes, PROT_NONE) == 0);
    CHECK(cinder_dense_forward(dense, x.values, 2, y.values, 3, NULL) == CINDER_OK);
    equal(y.values, affine, 3);
    CHECK(cinder_layer_forward(layers[0], x.values, 2, y.values, 3, NULL) == CINDER_OK);
    equal(y.values, affine, 3);
    CHECK(cinder_dense_forward(dense, x.values, 2, short_y.values, 2, NULL) == CINDER_ERROR_BUFFER_TOO_SMALL);
    equal(short_y.values, sentinel, 2);

    memcpy(y.values, sentinel, sizeof(sentinel));
    CHECK(cinder_dense_forward(dense, x.values, 1, y.values, 3, NULL) == CINDER_ERROR_DIMENSION_MISMATCH);
    equal(y.values, sentinel, 3);
    CHECK(cinder_dense_forward(dense, bad.values, 2, y.values, 3, NULL) == CINDER_ERROR_NONFINITE);
    equal(y.values, sentinel, 3);
    CHECK(cinder_dense_forward(dense, x.values, 2, y.values, 3, NULL) == CINDER_OK);
    equal(y.values, affine, 3);
    cinder_dense_free(dense);

    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, 2, &model, NULL) == CINDER_OK);
    CHECK(layers[0] == NULL && layers[1] == NULL);
    CHECK(cinder_model_run(model, x.values, 2, short_y.values, 2, NULL) == CINDER_ERROR_BUFFER_TOO_SMALL);
    equal(short_y.values, sentinel, 2);
    memcpy(y.values, sentinel, sizeof(sentinel));
    CHECK(cinder_model_run(model, x.values, 1, y.values, 3, NULL) == CINDER_ERROR_DIMENSION_MISMATCH);
    equal(y.values, sentinel, 3);
    CHECK(cinder_model_run(model, bad.values, 2, y.values, 3, NULL) == CINDER_ERROR_NONFINITE);
    equal(y.values, sentinel, 3);
    /* Finite inputs can still overflow inside Dense. The model must preserve
     * caller output even after execution has started, then accept a valid call. */
    CHECK(cinder_model_run(model, huge.values, 2, y.values, 3, NULL) == CINDER_ERROR_NONFINITE);
    equal(y.values, sentinel, 3);
    CHECK(cinder_model_run(model, x.values, 2, y.values, 3, NULL) == CINDER_OK);
    equal(y.values, activated, 3);
    cinder_model_free(model);
    release(&w);
    release(&b);

    const cinder_layer_type_t types[] = {CINDER_LAYER_RELU, CINDER_LAYER_SOFTMAX};
    for (size_t i = 0; i < 2; ++i) {
        cinder_layer_t *layer = NULL;
        const float *expected = i == 0 ? relu_expected : softmax_expected;
        CHECK(cinder_layer_create_activation(types[i], 3, &layer, NULL) == CINDER_OK);
        CHECK(cinder_layer_forward(layer, a.values, 3, short_y.values, 2, NULL) == CINDER_ERROR_BUFFER_TOO_SMALL);
        equal(short_y.values, sentinel, 2);
        CHECK(cinder_layer_forward(layer, a.values, 3, y.values, 3, NULL) == CINDER_OK);
        equal(y.values, expected, 3);
        memcpy(y.values, a.values, sizeof(activation));
        CHECK(cinder_layer_forward(layer, y.values, 3, y.values, 3, NULL) == CINDER_OK);
        equal(y.values, expected, 3);
        cinder_layer_free(layer);
    }
    release(&x);
    release(&bad);
    release(&huge);
    release(&a);
    release(&y);
    release(&short_y);
}

int main(void)
{
    boundaries(false);
    boundaries(true);
    puts("Protected buffers: both page edges passed; read-only inputs/parameters, exact capacities, "
         "constructor copies, kernels/layers/models, rejection and recovery; all mappings released.");
    return EXIT_SUCCESS;
}
