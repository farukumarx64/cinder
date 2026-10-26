#ifndef CINDER_TEST_MNIST_LIFECYCLE_AUDIT_H
#define CINDER_TEST_MNIST_LIFECYCLE_AUDIT_H

/* Test-only compilation of the real runtime. Never link this translation unit
 * with cinder_core: the substitutions and private inspection stay here. */
typedef struct { void *pointer; size_t bytes; } tracked_block_t;
static tracked_block_t live_blocks[128];
static size_t live_count, live_bytes, allocation_calls, frees, free_attempts;
static size_t fail_allocation, forward_calls, fail_forward;
static size_t opens, closes, reads;
static FILE *open_stream;

static void *remember_allocation(void *pointer, size_t bytes)
{
    CHECK(pointer != NULL && live_count < COUNT(live_blocks));
    live_blocks[live_count++] = (tracked_block_t){pointer, bytes};
    live_bytes += bytes;
    return pointer;
}

static void *tracked_malloc(size_t bytes)
{
    if (++allocation_calls == fail_allocation) {
        return NULL;
    }
    return remember_allocation(malloc(bytes), bytes);
}

static void *tracked_calloc(size_t count, size_t bytes)
{
    CHECK(count == 0 || bytes <= SIZE_MAX / count);
    if (++allocation_calls == fail_allocation) {
        return NULL;
    }
    return remember_allocation(calloc(count, bytes), count * bytes);
}

static void tracked_free(void *pointer)
{
    ++free_attempts;
    if (pointer == NULL) {
        return;
    }
    for (size_t i = 0; i < live_count; ++i) {
        if (live_blocks[i].pointer == pointer) {
            size_t bytes = live_blocks[i].bytes;
            live_bytes -= bytes;
            live_blocks[i] = live_blocks[--live_count];
            ++frees;
            memset(pointer, 0xa5, bytes);
            free(pointer);
            return;
        }
    }
    CHECK(0 && "untracked or double free");
}

static FILE *tracked_fopen(const char *path, const char *mode)
{
    CHECK(open_stream == NULL && strcmp(mode, "rb") == 0);
    ++opens;
    open_stream = fopen(path, mode);
    return open_stream;
}

static int tracked_fclose(FILE *file)
{
    CHECK(file != NULL && file == open_stream);
    open_stream = NULL;
    ++closes;
    return fclose(file);
}

static size_t tracked_fread(void *data, size_t size, size_t count, FILE *file)
{
    CHECK(file != NULL && file == open_stream);
    ++reads;
    return fread(data, size, count, file);
}

#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#include "../src/tensor.c"
#include "../src/dense.c"
#include "../src/layer.c"

static cinder_status_t tracked_forward(const cinder_layer_t *layer,
    const float *input, size_t count, float *output, size_t capacity, cinder_error_t *error)
{
    ++forward_calls;
    CHECK(forward_calls <= 4 && count == layer->input_size && capacity == 784);
    CHECK((input == output) == (layer->type != CINDER_LAYER_DENSE));
    if (forward_calls == fail_forward) {
        /* A late failure can leave a partly overwritten workspace. The model
         * must stop dispatch and preserve caller output despite that state. */
        output[0] = NAN;
        if (error != NULL) {
            error->code = CINDER_ERROR_NONFINITE;
            strcpy(error->message, "Injected MNIST layer failure after partial write.");
        }
        return CINDER_ERROR_NONFINITE;
    }
    return cinder_layer_forward(layer, input, count, output, capacity, error);
}

#define cinder_layer_forward tracked_forward
#include "../src/model.c"
#undef cinder_layer_forward
#define fopen tracked_fopen
#define fclose tracked_fclose
#define fread tracked_fread
#include "../src/loader.c"
#undef fread
#undef fclose
#undef fopen
#undef free
#undef calloc
#undef malloc

#endif
