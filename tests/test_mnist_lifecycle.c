#include "cinder/model.h"
#include "numpy_compare.h"

#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifndef CINDER_MNIST_AUDIT
#include <pthread.h>
#endif

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
enum { SAMPLES = 100, INPUTS = 784, OUTPUTS = 10, MODEL_BYTES = 407160,
       PARAMETER_COUNT = 101770, PUBLIC_FAILURES = 6 };

#ifdef CINDER_MNIST_AUDIT
#include "mnist_lifecycle_audit.h"
#endif

static float inputs[SAMPLES][INPUTS], probabilities[SAMPLES][OUTPUTS];
static float baseline[SAMPLES][OUTPUTS], overflowing[INPUTS];
static bool baseline_ready[SAMPLES];
static unsigned char model_bytes[MODEL_BYTES];
#ifdef CINDER_MNIST_AUDIT
static float parameters[PARAMETER_COUNT];
#endif

static uint32_t word(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
           (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static float decode_float(const unsigned char *bytes)
{
    uint32_t bits = word(bytes);
    float value;
    memcpy(&value, &bits, sizeof(bits));
    CHECK(isfinite(value));
    return value;
}

static void read_exact(const char *path, unsigned char *bytes, size_t count)
{
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fread(bytes, 1, count, file) == count);
    CHECK(fgetc(file) == EOF && feof(file) && !ferror(file));
    CHECK(fclose(file) == 0);
}

static void read_vector_file(const char *path, float *values, size_t count)
{
    unsigned char bytes[SAMPLES * OUTPUTS * 4];
    CHECK(count <= sizeof(bytes) / 4);
    read_exact(path, bytes, count * 4);
    for (size_t i = 0; i < count; ++i) {
        values[i] = decode_float(bytes + 4 * i);
    }
}

static void load_fixtures(const char *model_path, const char *directory)
{
    /* Called after the real loader has established host binary32 support. */
    read_exact(model_path, model_bytes, sizeof(model_bytes));
    CHECK(memcmp(model_bytes, "CNDR", 4) == 0 && word(model_bytes + 4) == 1);
    CHECK(word(model_bytes + 8) == INPUTS && word(model_bytes + 12) == 4);
    static const uint32_t records[][5] = {
        {16, 1, 784, 128, 401920}, {401952, 2, 128, 128, 0},
        {401968, 1, 128, 10, 5160}, {407144, 3, 10, 10, 0}
    };
    for (size_t record = 0; record < COUNT(records); ++record) {
        for (size_t field = 1; field < 5; ++field) {
            CHECK(word(model_bytes + records[record][0] + (field - 1) * 4) == records[record][field]);
        }
    }
    char path[4096];
    for (size_t row = 0; row < SAMPLES; ++row) {
        int length = snprintf(path, sizeof(path), "%s/inputs/%05zu.f32", directory, row);
        CHECK(length >= 0 && (size_t)length < sizeof(path));
        read_vector_file(path, inputs[row], INPUTS);
    }
    int length = snprintf(path, sizeof(path), "%s/probabilities.f32", directory);
    CHECK(length >= 0 && (size_t)length < sizeof(path));
    /* Read into a flat array, then copy rows; no pointer arithmetic across a
     * C subarray's declared extent, even though the 2-D storage is contiguous. */
    float flat[SAMPLES * OUTPUTS];
    read_vector_file(path, flat, COUNT(flat));
    memcpy(probabilities, flat, sizeof(probabilities));
    /* Choose signs to make every term in Dense1's first dot product positive.
     * Finite inputs then cause a real arithmetic overflow, not input rejection. */
    for (size_t col = 0; col < INPUTS; ++col) {
        overflowing[col] = copysignf(FLT_MAX, decode_float(model_bytes + 32 + col * 4));
    }
#ifdef CINDER_MNIST_AUDIT
    for (size_t i = 0; i < 100480; ++i) {
        parameters[i] = decode_float(model_bytes + 32 + i * 4);
    }
    for (size_t i = 0; i < 1290; ++i) {
        parameters[100480 + i] = decode_float(model_bytes + 401984 + i * 4);
    }
#endif
}

static void check_status(cinder_status_t actual, cinder_status_t expected,
                         const cinder_error_t *error)
{
    CHECK(actual == expected);
    if (error != NULL) {
        CHECK(error->code == expected && memchr(error->message, '\0', sizeof(error->message)) != NULL);
        CHECK((error->message[0] == '\0') == (expected == CINDER_OK));
    }
}

static cinder_model_t *load_model(const char *path)
{
    cinder_model_t *model = NULL;
    cinder_error_t error;
    check_status(cinder_model_load(path, &model, &error), CINDER_OK, &error);
    CHECK(model != NULL && cinder_model_input_size(model) == INPUTS);
    CHECK(cinder_model_output_size(model) == OUTPUTS);
#ifdef CINDER_MNIST_AUDIT
    CHECK(open_stream == NULL);
#endif
    return model;
}

#ifdef CINDER_MNIST_AUDIT
static void check_owned_parameters(const cinder_model_t *model)
{
    CHECK(model->layer_count == 4 && model->buffer_capacity == INPUTS);
    CHECK(model->buffers[0] != model->buffers[1]);
    const cinder_dense_t *a = model->layers[0]->dense, *b = model->layers[2]->dense;
    CHECK(memcmp(a->weights->data, parameters, 100352 * sizeof(float)) == 0);
    CHECK(memcmp(a->bias->data, parameters + 100352, 128 * sizeof(float)) == 0);
    CHECK(memcmp(b->weights->data, parameters + 100480, 1280 * sizeof(float)) == 0);
    CHECK(memcmp(b->bias->data, parameters + 101760, 10 * sizeof(float)) == 0);
    for (size_t buffer = 0; buffer < 2; ++buffer) {
        bool found = false;
        for (size_t i = 0; i < live_count; ++i) {
            if (live_blocks[i].pointer == model->buffers[buffer]) {
                CHECK(live_blocks[i].bytes == INPUTS * sizeof(float));
                found = true;
            }
        }
        CHECK(found);
    }
}

static void poison_workspaces(cinder_model_t *model)
{
    for (size_t buffer = 0; buffer < 2; ++buffer) {
        for (size_t i = 0; i < model->buffer_capacity; ++i) {
            model->buffers[buffer][i] = NAN;
        }
    }
}
#endif

static void successful_run(cinder_model_t *model, size_t row, float storage[OUTPUTS + 3],
                           cinder_error_t *error)
{
    row %= SAMPLES;
    float input[INPUTS];
    memcpy(input, inputs[row], sizeof(input));
#ifdef CINDER_MNIST_AUDIT
    forward_calls = 0;
#endif
    check_status(cinder_model_run(model, input, INPUTS, storage + 1, OUTPUTS + row % 2, error),
                 CINDER_OK, error);
    numpy_vector_t reference = {probabilities[row], OUTPUTS};
    char name[64];
    int length = snprintf(name, sizeof(name), "test-index-%zu", row);
    CHECK(length >= 0 && (size_t)length < sizeof(name));
    CHECK(compare_vector(NULL, name, "MNIST lifecycle", storage + 1, OUTPUTS, reference, true));
    CHECK(memcmp(input, inputs[row], sizeof(input)) == 0);
    CHECK(storage[0] == -101 && storage[OUTPUTS + 2] == -102 && isnan(storage[OUTPUTS + 1]));
    if (!baseline_ready[row]) {
        memcpy(baseline[row], storage + 1, sizeof(baseline[row]));
        baseline_ready[row] = true;
    } else {
        CHECK(memcmp(baseline[row], storage + 1, sizeof(baseline[row])) == 0);
    }
#ifdef CINDER_MNIST_AUDIT
    CHECK(forward_calls == 4);
#endif
}

static void cycle(cinder_model_t *model, size_t index)
{
    size_t row = index % SAMPLES, kind = index % PUBLIC_FAILURES;
    float output[OUTPUTS + 3] = {0}, saved[OUTPUTS + 3];
    output[0] = -101; output[OUTPUTS + 1] = NAN; output[OUTPUTS + 2] = -102;
    cinder_error_t error;
    memset(&error, 0xa5, sizeof(error));
#ifdef CINDER_MNIST_AUDIT
    kind = index % (PUBLIC_FAILURES + 4);
    size_t allocations_before = allocation_calls, frees_before = free_attempts;
    size_t opens_before = opens, closes_before = closes, reads_before = reads;
    CHECK(fail_allocation == 0 && fail_forward == 0);
    fail_allocation = allocation_calls + 1;
    check_owned_parameters(model);
    poison_workspaces(model);
#endif
    successful_run(model, row, output, &error);
    successful_run(model, row, output, NULL); /* Same input, bit-identical result. */
    memcpy(saved, output, sizeof(saved));

    float bad[INPUTS + 1];
    memcpy(bad, inputs[row], sizeof(inputs[row]));
    bad[INPUTS] = 0;
    if (kind == 3) bad[INPUTS - 1] = NAN;
    if (kind == 4) bad[INPUTS - 1] = INFINITY;
    if (kind == 5) memcpy(bad, overflowing, sizeof(overflowing));
    cinder_status_t expected = kind < 2 ? CINDER_ERROR_DIMENSION_MISMATCH :
                               kind == 2 ? CINDER_ERROR_BUFFER_TOO_SMALL : CINDER_ERROR_NONFINITE;
#ifdef CINDER_MNIST_AUDIT
    poison_workspaces(model);
    forward_calls = 0;
    fail_forward = kind >= PUBLIC_FAILURES ? kind - PUBLIC_FAILURES + 1 : 0;
#endif
    bool details = (index / PUBLIC_FAILURES) % 2 == 0;
    cinder_error_t *diagnostic = details ? &error : NULL;
    size_t count = kind == 0 ? INPUTS - 1 : kind == 1 ? INPUTS + 1 : INPUTS;
    size_t capacity = kind == 2 ? OUTPUTS - 1 : OUTPUTS + 1;
    check_status(cinder_model_run(model, bad, count, output + 1, capacity, diagnostic), expected, diagnostic);
    CHECK(memcmp(output, saved, sizeof(saved)) == 0);
#ifdef CINDER_MNIST_AUDIT
    CHECK(forward_calls == (fail_forward != 0 ? fail_forward : kind == 5 ? 1U : 0U));
    fail_forward = 0;
#endif
    /* Recover immediately on a different input; do not clear the failed state. */
    successful_run(model, row + 37, output, &error);
#ifdef CINDER_MNIST_AUDIT
    check_owned_parameters(model);
    CHECK(allocation_calls == allocations_before && free_attempts == frees_before);
    CHECK(opens == opens_before && closes == closes_before && reads == reads_before);
    /* The first Dense writes 128 values; its unused workspace tail stays NaN. */
    for (size_t i = 128; i < INPUTS; ++i) CHECK(isnan(model->buffers[1][i]));
    fail_allocation = 0;
#endif
}

static void write_model(const char *scratch, const unsigned char *bytes, size_t count)
{
    FILE *file = fopen(scratch, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(bytes, 1, count, file) == count);
    CHECK(fclose(file) == 0);
}

static void reload_cycles(const char *scratch)
{
    unsigned char changed[MODEL_BYTES];
    for (size_t i = 0; i < 20; ++i) {
        write_model(scratch, model_bytes, sizeof(model_bytes));
        cinder_model_t *first = load_model(scratch), *second = load_model(scratch);
        CHECK(first != second);
        cycle(first, i);
        memcpy(changed, model_bytes, sizeof(changed));
        cinder_status_t expected = CINDER_ERROR_FORMAT;
        size_t bytes = sizeof(changed);
        switch (i % 4) {
            case 0: changed[407144] = 4; break; /* Last record: invalid type. */
            case 1: --bytes; break; /* Truncated final record. */
            case 2:
                changed[407140] = 0; changed[407141] = 0;
                changed[407142] = 0x80; changed[407143] = 0x7f;
                expected = CINDER_ERROR_NONFINITE;
                break;
            default: changed[4] = 2; expected = CINDER_ERROR_UNSUPPORTED; break;
        }
        write_model(scratch, changed, bytes);
        cinder_model_t *failed = first; /* Borrowed sentinel, not ownership transfer. */
        cinder_error_t error;
#ifdef CINDER_MNIST_AUDIT
        size_t retained = live_count, retained_bytes = live_bytes;
#endif
        check_status(cinder_model_load(scratch, &failed, &error), expected, &error);
        CHECK(failed == NULL);
#ifdef CINDER_MNIST_AUDIT
        CHECK(live_count == retained && live_bytes == retained_bytes && open_stream == NULL);
#endif
        cycle(first, i + 7);
        cinder_model_free(first);
        CHECK(remove(scratch) == 0);
        cycle(second, i + 11); /* Source file and the first owner are gone. */
        cinder_model_free(second);
        write_model(scratch, model_bytes, sizeof(model_bytes));
        cinder_model_t *retry = load_model(scratch);
        CHECK(remove(scratch) == 0);
        cycle(retry, i + 19);
        cinder_model_free(retry);
    }
}

#ifndef CINDER_MNIST_AUDIT
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t ready;
    size_t arrivals;
} start_gate_t;
typedef struct {
    cinder_model_t *model;
    pthread_mutex_t *run_lock;
    start_gate_t *gate;
    size_t offset;
} worker_t;

static void *worker(void *argument)
{
    worker_t *work = argument;
    CHECK(pthread_mutex_lock(&work->gate->mutex) == 0);
    if (++work->gate->arrivals == 2) CHECK(pthread_cond_broadcast(&work->gate->ready) == 0);
    while (work->gate->arrivals != 2) {
        CHECK(pthread_cond_wait(&work->gate->ready, &work->gate->mutex) == 0);
    }
    CHECK(pthread_mutex_unlock(&work->gate->mutex) == 0);
    for (size_t i = 0; i < 50; ++i) {
        if (work->run_lock != NULL) CHECK(pthread_mutex_lock(work->run_lock) == 0);
        cycle(work->model, i + work->offset);
        if (work->run_lock != NULL) CHECK(pthread_mutex_unlock(work->run_lock) == 0);
    }
    return NULL;
}

static void thread_pair(cinder_model_t *first, cinder_model_t *second)
{
    start_gate_t gate = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
    pthread_mutex_t run_lock = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t *lock = first == second ? &run_lock : NULL;
    worker_t work[] = {{first, lock, &gate, 0}, {second, lock, &gate, 53}};
    pthread_t threads[2];
    for (size_t i = 0; i < 2; ++i) CHECK(pthread_create(&threads[i], NULL, worker, &work[i]) == 0);
    for (size_t i = 0; i < 2; ++i) CHECK(pthread_join(threads[i], NULL) == 0);
    CHECK(pthread_cond_destroy(&gate.ready) == 0 && pthread_mutex_destroy(&gate.mutex) == 0);
    CHECK(pthread_mutex_destroy(&run_lock) == 0);
}
#else
static void allocation_failures(const char *path, cinder_model_t *survivor)
{
    CHECK(live_count == 18 && open_stream == NULL);
    tracked_block_t retained[18];
    memcpy(retained, live_blocks, sizeof(retained));
    size_t retained_bytes = live_bytes;
    for (size_t point = 1; point <= 21; ++point) {
        for (size_t details = 0; details < 2; ++details) {
            size_t before = allocation_calls, opened = opens, closed = closes;
            fail_allocation = before + point;
            cinder_model_t *failed = survivor;
            cinder_error_t error;
            check_status(cinder_model_load(path, &failed, details ? &error : NULL),
                         CINDER_ERROR_ALLOCATION, details ? &error : NULL);
            CHECK(failed == NULL && allocation_calls == before + point);
            CHECK(live_count == COUNT(retained) && live_bytes == retained_bytes && open_stream == NULL);
            CHECK(opens == opened + 1 && closes == closed + 1);
            for (size_t i = 0; i < COUNT(retained); ++i) {
                bool found = false;
                for (size_t j = 0; j < live_count; ++j) {
                    if (retained[i].pointer == live_blocks[j].pointer) {
                        CHECK(retained[i].bytes == live_blocks[j].bytes);
                        found = true;
                    }
                }
                CHECK(found);
            }
            fail_allocation = 0;
            check_owned_parameters(survivor);
            float output[OUTPUTS + 3] = {0};
            output[0] = -101; output[OUTPUTS + 1] = NAN; output[OUTPUTS + 2] = -102;
            successful_run(survivor, point, output, &error);
            cinder_model_t *retry = load_model(path);
            successful_run(retry, point + 37, output, &error);
            cinder_model_free(retry);
            CHECK(live_count == COUNT(retained) && live_bytes == retained_bytes);
        }
    }
}
#endif

int main(int argc, char **argv)
{
    CHECK(argc == 4); /* Committed model, fixtures, per-test scratch filename. */
    cinder_model_t *first = load_model(argv[1]);
    load_fixtures(argv[1], argv[2]);
    cinder_model_t *second = load_model(argv[1]);
    CHECK(first != second);
#ifdef CINDER_MNIST_AUDIT
    CHECK(allocation_calls == 42 && live_count == 36);
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 2; ++j) CHECK(first->buffers[i] != second->buffers[j]);
    }
    for (size_t i = 0; i < 4; ++i) {
        CHECK(first->layers[i] != second->layers[i]);
        if (first->layers[i]->dense != NULL) {
            CHECK(first->layers[i]->dense != second->layers[i]->dense);
            CHECK(first->layers[i]->dense->weights->data != second->layers[i]->dense->weights->data);
            CHECK(first->layers[i]->dense->bias->data != second->layers[i]->dense->bias->data);
        }
    }
#endif
    for (size_t row = 0; row < SAMPLES; ++row) {
        float output[OUTPUTS + 3] = {0};
        output[0] = -101; output[OUTPUTS + 1] = NAN; output[OUTPUTS + 2] = -102;
        successful_run(first, row, output, NULL);
    }
    /* Freeze all baselines before threads: workers only read these arrays. */
    for (size_t row = 0; row < SAMPLES; ++row) CHECK(baseline_ready[row]);
    for (size_t i = 0; i < SAMPLES; ++i) {
        cycle(first, i * 37);
        cycle(second, i * 37 + 11);
    }
#ifndef CINDER_MNIST_AUDIT
    thread_pair(first, second);
#endif
    cinder_model_free(first);
    for (size_t i = 0; i < 20; ++i) cycle(second, i + 13);
    reload_cycles(argv[3]);
#ifndef CINDER_MNIST_AUDIT
    thread_pair(second, second);
#else
    allocation_failures(argv[1], second);
#endif
    cinder_model_free(second);
    cinder_model_free(NULL);
#ifdef CINDER_MNIST_AUDIT
    CHECK(live_count == 0 && live_bytes == 0 && open_stream == NULL && opens == closes);
    CHECK(allocation_calls == frees + 42);
    puts("MNIST audit: 104 successful loads, 62 rejected loads, 300 cycles / 1384 inference calls; "
         "poisoned 784-float workspaces, partial-write failures at all four layers, unchanged parameters, "
         "zero allocations/frees or loader I/O during cycles, all 21 allocation points failed twice, "
         "and zero final live blocks/bytes/streams.");
#else
    puts("MNIST lifecycle: 62 successful loads, 20 rejected loads, 500 cycles / 2100 inference calls; "
         "all 100 inputs match saved references and bit-identical repeat baselines, independent models, "
         "deleted source files, concurrent independent instances, and caller-serialized shared access.");
#endif
    return EXIT_SUCCESS;
}
