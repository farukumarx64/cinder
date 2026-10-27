/* Compile the real CLI setup, reader and runtime with test-only allocation/I/O
 * counters. Actual CLI process tests also exercise the normally linked sources. */
#include "../src/cli_session.h"
#include "../src/benchmark.h"
#include "../src/timer.h"
#include "numpy_compare.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
typedef struct { void *pointer; size_t bytes; } block_t;
static block_t blocks[128];
static size_t live, bytes_live, allocations, frees, fail_allocation;
static size_t opens, closes, reads, fail_open, fail_close, loads, inputs, forwards;
static size_t output_allocations;
static size_t benchmark_runs, timer_inits, timer_starts, timer_stops;
static bool input_ready;
static FILE *stream;

static void *remember(void *pointer, size_t bytes)
{
    CHECK(pointer != NULL && live < COUNT(blocks));
    blocks[live++] = (block_t){pointer, bytes};
    bytes_live += bytes;
    return pointer;
}
static void *tracked_malloc(size_t bytes)
{
    if (input_ready) {
        /* Only the caller-output allocation follows successful input close. */
        CHECK(stream == NULL && closes == 2);
        ++output_allocations;
    }
    if (++allocations == fail_allocation) return NULL;
    return remember(malloc(bytes), bytes);
}
static void *tracked_calloc(size_t count, size_t bytes)
{
    CHECK(!input_ready && (count == 0 || bytes <= SIZE_MAX / count));
    if (++allocations == fail_allocation) return NULL;
    return remember(calloc(count, bytes), count * bytes);
}
static void tracked_free(void *pointer)
{
    if (pointer == NULL) return;
    for (size_t i = 0; i < live; ++i) {
        if (blocks[i].pointer == pointer) {
            bytes_live -= blocks[i].bytes;
            memset(pointer, 0xa5, blocks[i].bytes);
            blocks[i] = blocks[--live];
            ++frees;
            free(pointer);
            return;
        }
    }
    CHECK(0 && "untracked or double free");
}
static FILE *tracked_open(const char *path, const char *mode)
{
    CHECK(stream == NULL && strcmp(mode, "rb") == 0);
    if (++opens == fail_open) return NULL;
    stream = fopen(path, mode);
    CHECK(stream != NULL);
    return stream;
}
static int tracked_close(FILE *file)
{
    CHECK(file != NULL && file == stream);
    CHECK(fclose(file) == 0);
    stream = NULL;
    return ++closes == fail_close ? EOF : 0;
}
static size_t tracked_read(void *data, size_t size, size_t count, FILE *file)
{
    CHECK(file != NULL && file == stream);
    ++reads;
    return fread(data, size, count, file);
}
#define malloc tracked_malloc
#define calloc tracked_calloc
#define free tracked_free
#include "../src/tensor.c"
#include "../src/dense.c"
#include "../src/layer.c"
static cinder_status_t tracked_forward(const cinder_layer_t *layer, const float *input,
    size_t count, float *output, size_t capacity, cinder_error_t *error)
{
    ++forwards;
    return cinder_layer_forward(layer, input, count, output, capacity, error);
}
#define cinder_layer_forward tracked_forward
#include "../src/model.c"
#undef cinder_layer_forward
#define fopen tracked_open
#define fclose tracked_close
#define fread tracked_read
#include "../src/loader.c"
#include "../src/input.c"
#undef fopen
#undef fclose
#undef fread
static cinder_status_t tracked_load(const char *path, cinder_model_t **model, cinder_error_t *error)
{
    ++loads;
    return cinder_model_load(path, model, error);
}
static cinder_status_t tracked_input(const char *path, size_t count, float **values, cinder_error_t *error)
{
    ++inputs;
    cinder_status_t status = cinder_input_read(path, count, values, error);
    CHECK(stream == NULL);
    input_ready = status == CINDER_OK;
    return status;
}
#define cinder_model_load tracked_load
#define cinder_input_read tracked_input
#include "../src/cli_session.c"
#undef cinder_model_load
#undef cinder_input_read
static cinder_status_t benchmark_run(cinder_model_t *model, const float *input,
    size_t count, float *output, size_t capacity, cinder_error_t *error)
{
    ++benchmark_runs;
    return cinder_model_run(model, input, count, output, capacity, error);
}
static cinder_status_t benchmark_timer_init(cinder_timer_t *timer, cinder_error_t *error)
{
    ++timer_inits;
    return cinder_timer_init(timer, error);
}
static cinder_status_t benchmark_timer_start(cinder_timer_t *timer, cinder_error_t *error)
{
    ++timer_starts;
    return cinder_timer_start(timer, error);
}
static cinder_status_t benchmark_timer_stop(cinder_timer_t *timer, uint64_t *elapsed, cinder_error_t *error)
{
    ++timer_stops;
    return cinder_timer_stop(timer, elapsed, error);
}
#define cinder_model_run benchmark_run
#define cinder_timer_init benchmark_timer_init
#define cinder_timer_start benchmark_timer_start
#define cinder_timer_stop benchmark_timer_stop
#define fopen tracked_open
#define fclose tracked_close
#define fread tracked_read
#include "../src/benchmark.c"
#undef cinder_model_run
#undef cinder_timer_init
#undef cinder_timer_start
#undef cinder_timer_stop
#undef fopen
#undef fclose
#undef fread
#undef malloc
#undef calloc
#undef free

static void reset(void)
{
    CHECK(live == 0 && bytes_live == 0 && stream == NULL);
    allocations = frees = fail_allocation = opens = closes = reads = 0;
    fail_open = fail_close = loads = inputs = forwards = output_allocations = 0;
    input_ready = false;
    benchmark_runs = timer_inits = timer_starts = timer_stops = 0;
}
static void empty(const cinder_cli_session_t *session)
{
    CHECK(session->model == NULL && session->input == NULL && session->output == NULL);
    CHECK(session->input_size == 0 && session->output_size == 0);
    CHECK(live == 0 && bytes_live == 0 && stream == NULL);
}
static size_t block_size(const void *pointer)
{
    for (size_t i = 0; i < live; ++i) if (blocks[i].pointer == pointer) return blocks[i].bytes;
    CHECK(0 && "missing allocation");
    return 0;
}
static void make_path(char *path, size_t capacity, const char *root, const char *name)
{
    int length = snprintf(path, capacity, "%s/%s", root, name);
    CHECK(length > 0 && (size_t)length < capacity);
}
static void copy_file(const char *from, const char *to)
{
    FILE *source = fopen(from, "rb"), *target = fopen(to, "wb");
    CHECK(source != NULL && target != NULL);
    unsigned char bytes[4096];
    size_t count;
    while ((count = fread(bytes, 1, sizeof(bytes), source)) != 0) CHECK(fwrite(bytes, 1, count, target) == count);
    CHECK(!ferror(source) && feof(source));
    CHECK(fclose(source) == 0 && fclose(target) == 0);
}
static void repeat(cinder_cli_session_t *session, numpy_vector_t reference, size_t repetitions)
{
    size_t allocations_before = allocations, frees_before = frees, reads_before = reads;
    size_t opens_before = opens, closes_before = closes, loads_before = loads, inputs_before = inputs;
    float saved[784];
    CHECK(session->input_size <= COUNT(saved));
    memcpy(saved, session->input, session->input_size * sizeof(float));
    fail_allocation = allocations + 1; /* Any unexpected inference allocation fails. */
    for (size_t i = 0; i < repetitions; ++i) {
        for (size_t j = 0; j < session->output_size; ++j) session->output[j] = NAN;
        CHECK(cinder_model_run(session->model, session->input, session->input_size,
                              session->output, session->output_size, NULL) == CINDER_OK);
        CHECK(compare_vector(NULL, "cli_setup", "probabilities", session->output,
                             session->output_size, reference, true));
        CHECK(memcmp(saved, session->input, session->input_size * sizeof(float)) == 0);
    }
    CHECK(allocations == allocations_before && frees == frees_before && reads == reads_before);
    CHECK(opens == opens_before && closes == closes_before && loads == loads_before && inputs == inputs_before);
}
static void benchmark_reuse(cinder_cli_session_t *session, numpy_vector_t reference)
{
    size_t allocations_before = allocations, frees_before = frees, reads_before = reads;
    size_t opens_before = opens, closes_before = closes, forwards_before = forwards;
    cinder_model_t *model = session->model;
    float *input = session->input, *output = session->output, saved[784];
    CHECK(session->input_size <= COUNT(saved));
    memcpy(saved, input, session->input_size * sizeof(float));
    fail_allocation = allocations + 1;
    cinder_benchmark_result_t result;
    cinder_benchmark_failure_t problem;
    CHECK(cinder_benchmark_execute(session, 1, &result, &problem) == CINDER_OK);
    CHECK(result.iterations == 1 && result.total_ns == result.minimum_ns && result.total_ns == result.maximum_ns);
    CHECK(result.resolution_ns > 0 && result.total_ns > 0);
    CHECK(compare_vector(NULL, "benchmark", "probabilities", output, session->output_size, reference, true));
    CHECK(benchmark_runs == 11 && timer_inits == 1 && timer_starts == 1 && timer_stops == 1);
    /* Fail on warm-up input validation, then reuse the same session successfully. */
    input[0] = NAN;
    cinder_benchmark_result_t previous = result;
    CHECK(cinder_benchmark_execute(session, 5, &result, &problem) == CINDER_ERROR_NONFINITE);
    CHECK(result.iterations == previous.iterations && result.total_ns == previous.total_ns);
    CHECK(problem.phase == CINDER_BENCHMARK_WARMUP && problem.iteration == 1);
    CHECK(benchmark_runs == 12 && timer_inits == 2 && timer_starts == 1 && timer_stops == 1);
    input[0] = saved[0];
    CHECK(cinder_benchmark_execute(session, 5, &result, NULL) == CINDER_OK);
    CHECK(result.iterations == 5 && result.minimum_ns > 0 && result.maximum_ns >= result.minimum_ns);
    CHECK(result.total_ns >= result.maximum_ns && result.resolution_ns > 0);
    CHECK(compare_vector(NULL, "benchmark", "probabilities", output, session->output_size, reference, true));
    CHECK(benchmark_runs == 27 && timer_inits == 3 && timer_starts == 6 && timer_stops == 6);
    CHECK(forwards == forwards_before + 26 * 4); /* Both fixtures have four layers. */
    CHECK(session->model == model && session->input == input && session->output == output);
    CHECK(memcmp(saved, input, session->input_size * sizeof(float)) == 0);
    CHECK(allocations == allocations_before && frees == frees_before && reads == reads_before);
    CHECK(opens == opens_before && closes == closes_before && loads == 1 && inputs == 1);
}
static size_t success(const char *model, const char *input, size_t input_size,
    numpy_vector_t reference, cinder_cli_session_t *session, bool remove_sources, size_t repetitions)
{
    reset();
    cinder_cli_failure_t failure = {"stale", "stale", {CINDER_ERROR_IO, "old error"}};
    CHECK(cinder_cli_prepare(model, input, session, &failure) == CINDER_OK);
    CHECK(failure.detail.code == CINDER_OK && failure.detail.message[0] == '\0');
    CHECK(strcmp(failure.stage, "inference") == 0 && failure.path == NULL);
    CHECK(loads == 1 && inputs == 1 && opens == 2 && closes == 2 && forwards == 0);
    CHECK(output_allocations == 1 && session->input != session->output);
    CHECK(session->input_size == input_size && session->output_size == reference.count);
    CHECK(block_size(session->input) == input_size * sizeof(float));
    CHECK(block_size(session->output) == reference.count * sizeof(float));
    size_t count = allocations;
    cinder_model_t *saved_model = session->model;
    float *saved_input = session->input, *saved_output = session->output;
    /* Accidental double preparation must not lose existing ownership. */
    CHECK(cinder_cli_prepare(model, input, session, &failure) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(strcmp(failure.detail.message, "CLI session must be empty before preparation.") == 0);
    CHECK(session->model == saved_model && session->input == saved_input && session->output == saved_output);
    CHECK(session->input_size == input_size && session->output_size == reference.count);
    CHECK(allocations == count && loads == 1 && inputs == 1);
    if (remove_sources) CHECK(remove(model) == 0 && remove(input) == 0);
    repeat(session, reference, repetitions);
    if (remove_sources) benchmark_reuse(session, reference);
    cinder_cli_release(session);
    empty(session);
    size_t released = frees;
    cinder_cli_release(session);
    cinder_cli_release(NULL);
    CHECK(frees == released);
    return count;
}
static void failure(const char *model, const char *input, cinder_cli_session_t *session,
    cinder_status_t expected, const char *stage, bool details)
{
    cinder_cli_failure_t problem = {0};
    CHECK(cinder_cli_prepare(model, input, session, details ? &problem : NULL) == expected);
    empty(session);
    CHECK(forwards == 0);
    if (details) {
        CHECK(problem.detail.code == expected && problem.detail.message[0] != '\0');
        CHECK(strcmp(problem.stage, stage) == 0);
        CHECK(problem.path == (strcmp(stage, "model") == 0 ? model : strcmp(stage, "input") == 0 ? input : NULL));
    }
    size_t released = frees;
    cinder_cli_release(session);
    CHECK(frees == released);
}
static void exercise(const char *original_model, const char *original_input, const char *scratch,
    size_t input_size, numpy_vector_t reference)
{
    char model[4096], input[4096];
    make_path(model, sizeof(model), scratch, "model.cnd");
    make_path(input, sizeof(input), scratch, "input.bin");
    copy_file(original_model, model); copy_file(original_input, input);
    cinder_cli_session_t session = {0};
    size_t points = success(model, input, input_size, reference, &session, true, 100);
    CHECK(points == 23); /* Existing four-layer loader: 21; input and output: 2. */
    copy_file(original_model, model); copy_file(original_input, input);
    for (size_t point = 1; point <= points; ++point) {
        for (int details = 0; details < 2; ++details) {
            reset(); fail_allocation = point;
            const char *stage = point <= points - 2 ? "model" : point == points - 1 ? "input" : "inference";
            failure(model, input, &session, CINDER_ERROR_ALLOCATION, stage, details != 0);
            CHECK(allocations == point);
            success(model, input, input_size, reference, &session, false, 1);
        }
    }
    for (size_t operation = 0; operation < 4; ++operation) {
        for (int details = 0; details < 2; ++details) {
            reset();
            if (operation < 2) fail_open = operation + 1; else fail_close = operation - 1;
            const char *stage = operation % 2 == 0 ? "model" : "input";
            failure(model, input, &session, CINDER_ERROR_IO, stage, details != 0);
            CHECK(opens == (operation % 2 + 1));
            CHECK(closes == (operation < 2 ? opens - 1 : opens));
            CHECK(output_allocations == 0);
            success(model, input, input_size, reference, &session, false, 1);
        }
    }
    FILE *file = fopen(input, "ab"); CHECK(file != NULL && fputc(0, file) != EOF && fclose(file) == 0);
    reset(); fail_close = 2;
    failure(model, input, &session, CINDER_ERROR_FORMAT, "input", true);
    CHECK(output_allocations == 0); /* Length error wins over failing close. */
    copy_file(original_input, input);
    success(model, input, input_size, reference, &session, false, 1);
    file = fopen(input, "r+b"); CHECK(file != NULL && fseek(file, -4, SEEK_END) == 0);
    const unsigned char infinity[] = {0, 0, 0x80, 0x7f};
    CHECK(fwrite(infinity, 1, 4, file) == 4 && fclose(file) == 0);
    reset(); fail_close = 2;
    failure(model, input, &session, CINDER_ERROR_NONFINITE, "input", true);
    CHECK(output_allocations == 0);
    copy_file(original_input, input);
    success(model, input, input_size, reference, &session, false, 1);
    reset();
    CHECK(cinder_cli_prepare(model, input, &session, NULL) == CINDER_OK);
    repeat(&session, reference, 1);
    cinder_cli_release(&session); empty(&session);
    CHECK(remove(model) == 0 && remove(input) == 0);
    printf("CLI setup %zu -> %zu: one load/read, exact input/output allocations, 100 repeated calls plus benchmark warm-up/measurement/recovery after source deletion, 23 allocation points failed with/without details, I/O/format/non-finite failures and successful retries; zero retained blocks/bytes/streams.\n", input_size, reference.count);
}

int main(int argc, char **argv)
{
    CHECK(argc == 3); /* source directory and private scratch directory */
    reset();
    cinder_cli_failure_t problem;
    CHECK(cinder_cli_prepare("unused", "unused", NULL, &problem) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(problem.detail.code == CINDER_ERROR_INVALID_ARGUMENT && strcmp(problem.stage, "inference") == 0);
    CHECK(opens == 0 && allocations == 0 && loads == 0);
    cinder_cli_session_t invalid = {.input_size = 1};
    CHECK(cinder_cli_prepare("unused", "unused", &invalid, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(invalid.input_size == 1 && opens == 0 && allocations == 0 && loads == 0);
    cinder_cli_release(&invalid); empty(&invalid);

    char model[4096], input[4096], reference_path[4096];
    make_path(model, sizeof(model), argv[1], "tests/fixtures/tiny/tiny.cnd");
    make_path(input, sizeof(input), argv[1], "tests/fixtures/tiny/inputs/worked.f32");
    exercise(model, input, argv[2], 2, numpy_network_cases[0].softmax);

    make_path(model, sizeof(model), argv[1], "examples/mnist/mnist.cnd");
    make_path(input, sizeof(input), argv[1], "tests/fixtures/mnist/inputs/00000.f32");
    make_path(reference_path, sizeof(reference_path), argv[1], "tests/fixtures/mnist/probabilities.f32");
    FILE *file = fopen(reference_path, "rb"); CHECK(file != NULL);
    float reference[10];
    for (size_t i = 0; i < COUNT(reference); ++i) {
        unsigned char bytes[4]; CHECK(fread(bytes, 1, 4, file) == 4);
        uint32_t bits = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
                        (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
        memcpy(reference + i, &bits, sizeof(bits));
    }
    CHECK(fclose(file) == 0);
    exercise(model, input, argv[2], 784, (numpy_vector_t){reference, COUNT(reference)});
    return 0;
}
