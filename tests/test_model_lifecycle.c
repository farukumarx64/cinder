#include "cinder/model.h"
#include "numpy_compare.h"

#include <float.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static cinder_model_t *create_network(const char *path)
{
    if (path != NULL) {
        cinder_model_t *model = NULL;
        cinder_error_t error;
        CHECK(cinder_model_load(path, &model, &error) == CINDER_OK);
        CHECK(model != NULL && error.code == CINDER_OK && error.message[0] == '\0');
        return model;
    }
    const numpy_network_fixture_t *n = &numpy_tiny_network;
    CHECK(n->input_size == 2 && n->hidden_size == 4 && n->output_size == 2);
    cinder_layer_t *layers[4] = {NULL, NULL, NULL, NULL};
    CHECK(cinder_layer_create_dense(n->input_size, n->hidden_size,
          n->weights1.data, n->weights1.count, n->bias1.data, n->bias1.count,
          &layers[0], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_activation(CINDER_LAYER_RELU, n->hidden_size,
          &layers[1], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_dense(n->hidden_size, n->output_size,
          n->weights2.data, n->weights2.count, n->bias2.data, n->bias2.count,
          &layers[2], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_activation(CINDER_LAYER_SOFTMAX, n->output_size,
          &layers[3], NULL) == CINDER_OK);
    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, COUNT(layers), &model, NULL) == CINDER_OK);
    return model;
}

static void successful_run(cinder_model_t *model, size_t index, float *output,
                           cinder_error_t *error)
{
    const numpy_network_case_t *fixture = &numpy_network_cases[index % COUNT(numpy_network_cases)];
    CHECK(fixture->input.count == 2 && fixture->softmax.count == 2);
    float input[2];
    memcpy(input, fixture->input.data, sizeof(input));
    CHECK(cinder_model_run(model, input, COUNT(input), output, 3, error) == CINDER_OK);
    CHECK(error->code == CINDER_OK && error->message[0] == '\0');
    CHECK(compare_vector(NULL, fixture->name, "lifecycle", output, 2, fixture->softmax, true));
    CHECK(isnan(output[2]));
    CHECK(memcmp(input, fixture->input.data, sizeof(input)) == 0);
    CHECK(cinder_model_input_size(model) == 2 && cinder_model_output_size(model) == 2);
}

static void exercise_cycle(cinder_model_t *model, size_t index)
{
    cinder_error_t error;
    float output[] = {91, 92, NAN}, saved[3];
    successful_run(model, index, output, &error);
    memcpy(saved, output, sizeof(saved));
    const float input[] = {2, -1};
    const float nonfinite[] = {NAN, 0};
    const float overflowing[] = {FLT_MAX, FLT_MAX};
    cinder_status_t status, expected;
    switch ((index / COUNT(numpy_network_cases)) % 4) {
        case 0:
            status = cinder_model_run(model, input, 1, output, 3, &error);
            expected = CINDER_ERROR_DIMENSION_MISMATCH;
            break;
        case 1:
            status = cinder_model_run(model, input, 2, output, 1, &error);
            expected = CINDER_ERROR_BUFFER_TOO_SMALL;
            break;
        case 2:
            status = cinder_model_run(model, nonfinite, 2, output, 3, &error);
            expected = CINDER_ERROR_NONFINITE;
            break;
        default:
            status = cinder_model_run(model, overflowing, 2, output, 3, &error);
            expected = CINDER_ERROR_NONFINITE;
            break;
    }
    CHECK(status == expected && error.code == expected && error.message[0] != '\0');
    CHECK(memchr(error.message, '\0', sizeof(error.message)) != NULL);
    CHECK(memcmp(output, saved, sizeof(output)) == 0);
    /* A different input succeeds immediately, with the failed run's private
     * contents intact and the previous output/error storage reused. */
    successful_run(model, index + 3, output, &error);
}

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
    const char *path;
} worker_t;

static void *run_worker(void *argument)
{
    worker_t *worker = argument;
    start_gate_t *gate = worker->gate;
    CHECK(pthread_mutex_lock(&gate->mutex) == 0);
    ++gate->arrivals;
    if (gate->arrivals == 2) {
        CHECK(pthread_cond_broadcast(&gate->ready) == 0);
    }
    while (gate->arrivals != 2) {
        CHECK(pthread_cond_wait(&gate->ready, &gate->mutex) == 0);
    }
    CHECK(pthread_mutex_unlock(&gate->mutex) == 0);
    for (size_t i = 0; i < (worker->path != NULL ? 100U : 1000U); ++i) {
        if (worker->path != NULL) {
            /* Concurrent loads use independent FILE streams and owners. */
            cinder_model_t *model = create_network(worker->path);
            exercise_cycle(model, i + worker->offset);
            cinder_model_free(model);
            continue;
        }
        if (worker->run_lock != NULL) {
            CHECK(pthread_mutex_lock(worker->run_lock) == 0);
        }
        exercise_cycle(worker->model, i + worker->offset);
        if (worker->run_lock != NULL) {
            CHECK(pthread_mutex_unlock(worker->run_lock) == 0);
        }
    }
    return NULL;
}

static void check_thread_pair(cinder_model_t *first, cinder_model_t *second,
                              const char *path)
{
    start_gate_t gate = {PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER, 0};
    pthread_mutex_t run_lock = PTHREAD_MUTEX_INITIALIZER;
    /* Independent models need no shared run lock. Sharing one model requires
     * caller serialization; this test never intentionally races that instance. */
    pthread_mutex_t *lock = path == NULL && first == second ? &run_lock : NULL;
    worker_t workers[] = {{first, lock, &gate, 0, path}, {second, lock, &gate, 7, path}};
    pthread_t threads[2];
    for (size_t i = 0; i < COUNT(threads); ++i) {
        CHECK(pthread_create(&threads[i], NULL, run_worker, &workers[i]) == 0);
    }
    for (size_t i = 0; i < COUNT(threads); ++i) {
        CHECK(pthread_join(threads[i], NULL) == 0);
    }
    CHECK(pthread_mutex_destroy(&run_lock) == 0);
    CHECK(pthread_cond_destroy(&gate.ready) == 0);
    CHECK(pthread_mutex_destroy(&gate.mutex) == 0);
}

static void write_model(const char *path, const unsigned char *bytes, size_t size)
{
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(bytes, 1, size, file) == size);
    CHECK(fclose(file) == 0);
}

static void check_load_cycles(const char *path, const char *scratch)
{
    unsigned char bytes[168];
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fread(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
    CHECK(fgetc(file) == EOF && !ferror(file));
    CHECK(fclose(file) == 0);
    for (size_t i = 0; i < 100; ++i) {
        write_model(scratch, bytes, sizeof(bytes));
        cinder_model_t *first = create_network(scratch);
        cinder_model_t *second = create_network(scratch);
        CHECK(first != second);
        exercise_cycle(first, i);

        unsigned char changed[168];
        memcpy(changed, bytes, sizeof(changed));
        cinder_status_t expected;
        switch (i % 4) {
            case 0: /* Reject after earlier layers have been constructed. */
                changed[152] = 4;
                expected = CINDER_ERROR_FORMAT;
                break;
            case 1:
                changed[4] = 2;
                expected = CINDER_ERROR_UNSUPPORTED;
                break;
            case 2: /* Last Dense bias becomes positive infinity. */
                changed[148] = 0; changed[149] = 0;
                changed[150] = 0x80; changed[151] = 0x7f;
                expected = CINDER_ERROR_NONFINITE;
                break;
            default:
                expected = CINDER_ERROR_IO;
                break;
        }
        if (expected == CINDER_ERROR_IO) {
            CHECK(remove(scratch) == 0);
        } else {
            write_model(scratch, changed, sizeof(changed));
        }
        /* Borrowed sentinel: clearing the output must not free its owner. */
        cinder_model_t *recovered = first;
        cinder_error_t error;
        memset(&error, 0xa5, sizeof(error));
        bool details = (i / 4) % 2 == 0;
        CHECK(cinder_model_load(scratch, &recovered, details ? &error : NULL) == expected);
        CHECK(recovered == NULL);
        if (details) {
            CHECK(error.code == expected && error.message[0] != '\0');
            CHECK(memchr(error.message, '\0', sizeof(error.message)) != NULL);
        }
        exercise_cycle(first, i + 3);
        cinder_model_free(first);
        if (expected != CINDER_ERROR_IO) {
            CHECK(remove(scratch) == 0);
        }
        /* Neither the source file nor the other owner remains alive. */
        exercise_cycle(second, i + 7);
        cinder_model_free(second);

        write_model(scratch, bytes, sizeof(bytes));
        CHECK(cinder_model_load(scratch, &recovered, &error) == CINDER_OK);
        CHECK(error.code == CINDER_OK && error.message[0] == '\0');
        CHECK(remove(scratch) == 0);
        exercise_cycle(recovered, i + 9);
        cinder_model_free(recovered);
        cinder_model_free(NULL);
    }
    check_thread_pair(NULL, NULL, path);
}

int main(int argc, char **argv)
{
    CHECK(argc == 1 || argc == 3);
    const char *path = argc == 3 ? argv[1] : NULL;
    if (path != NULL) {
        check_load_cycles(path, argv[2]);
    }
    cinder_model_t *first = create_network(path);
    cinder_model_t *second = create_network(path);
    /* Interleaved serial calls, then concurrent independent instances. */
    for (size_t i = 0; i < 40; ++i) {
        exercise_cycle(first, i);
        exercise_cycle(second, i + 7);
    }
    check_thread_pair(first, second, NULL);
    cinder_model_free(first);
    /* Destroying one owner must not invalidate the other model's workspace. */
    for (size_t i = 0; i < 40; ++i) {
        exercise_cycle(second, i);
    }
    check_thread_pair(second, second, NULL);
    cinder_model_free(second);
    if (path != NULL) {
        puts("Loaded lifecycle passed: 502 successful loads, 100 rejected loads, "
             "4720 success/failure/recovery cycles (14160 inference calls), "
             "concurrent independent loads, deleted source files, and caller-serialized threads.");
    } else {
        puts("Model lifecycle passed: 4120 success/failure/recovery cycles, independent owners, and caller-serialized threads.");
    }
    return EXIT_SUCCESS;
}
