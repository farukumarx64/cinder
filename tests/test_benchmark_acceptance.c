/* Real command -> setup -> model -> timer -> statistics -> report.
 * Only OS clock values and explicitly requested faults are substituted.
 * Test capture files/allocations are outside the production ownership ledger. */
#include "../src/benchmark.h"
#include "../src/timer.h"
#include "numpy_compare.h"
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static FILE *test_stderr(void) { return stderr; }

#define COUNT(a) (sizeof(a) / sizeof((a)[0]))
#define CHECK(x) do { if (!(x)) { \
    fprintf(test_stderr(), "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

typedef struct { void *pointer; size_t bytes; } block_t;
static block_t blocks[128];
static size_t live, live_bytes, allocations, frees, fail_allocation;
static size_t opens, closes, reads, fail_open, fail_close, loads, input_reads;
static size_t calls, forwards, elements, element_index, fail_call, fail_layer;
static size_t corrupt_call, corrupt_element, queries, clock_reads, fault_read;
static size_t writes, flushes, fail_write, invocations;
static bool executing, fail_flush, fail_stderr, partial_write, large_uptime;
static bool inspecting;
static FILE *owned_stream, *captured_out, *captured_err;
static float corrupt_value, saved_input[784];
static cinder_model_t *saved_model;
static const float *saved_source;
static float *saved_destination;
static numpy_vector_t reference;
static struct timespec current, resolution, last_start, last_end;
static int query_result;
static uint64_t durations[3];
static char model_path[4096], input_path[4096], out_text[4096], err_text[4096];
static enum { CLOCK_GOOD, CLOCK_READ_FAIL, CLOCK_NEGATIVE_SECONDS, CLOCK_NEGATIVE_NANOS,
    CLOCK_INVALID_NANOS, CLOCK_ZERO, CLOCK_BACKWARD, CLOCK_BETWEEN, CLOCK_OVERFLOW } clock_fault;

static void advance(uint64_t ns)
{
    current.tv_sec += (time_t)(ns / UINT64_C(1000000000));
    current.tv_nsec += (long)(ns % UINT64_C(1000000000));
    if (current.tv_nsec >= 1000000000L) { ++current.tv_sec; current.tv_nsec -= 1000000000L; }
}
static void *remember(void *pointer, size_t bytes)
{
    CHECK(pointer != NULL && live < COUNT(blocks));
    blocks[live++] = (block_t){pointer, bytes}; live_bytes += bytes;
    return pointer;
}
static void *tracked_malloc(size_t bytes)
{
    CHECK(!executing && queries == 0);
    advance(UINT64_C(1000000000)); /* Setup is deliberately much slower than inference. */
    if (++allocations == fail_allocation) return NULL;
    return remember(malloc(bytes), bytes);
}
static void *tracked_calloc(size_t count, size_t bytes)
{
    CHECK(!executing && queries == 0 && (count == 0 || bytes <= SIZE_MAX / count));
    advance(UINT64_C(1000000000));
    if (++allocations == fail_allocation) return NULL;
    return remember(calloc(count, bytes), count * bytes);
}
static void tracked_free(void *pointer)
{
    CHECK(!executing); /* Include even free(NULL) attempts in this guard. */
    if (pointer == NULL) return;
    for (size_t i = 0; i < live; ++i) {
        if (blocks[i].pointer == pointer) {
            live_bytes -= blocks[i].bytes;
            memset(pointer, 0xa5, blocks[i].bytes);
            blocks[i] = blocks[--live]; ++frees; free(pointer); return;
        }
    }
    CHECK(0 && "untracked or double free");
}
static FILE *tracked_open(const char *path, const char *mode)
{
    CHECK(!executing && queries == 0 && owned_stream == NULL && strcmp(mode, "rb") == 0);
    if (++opens == fail_open) return NULL;
    owned_stream = fopen(path, mode); CHECK(owned_stream != NULL); return owned_stream;
}
static int tracked_close(FILE *file)
{
    CHECK(!executing && file != NULL && file == owned_stream);
    CHECK(fclose(file) == 0); owned_stream = NULL;
    return ++closes == fail_close ? EOF : 0;
}
static size_t tracked_read(void *data, size_t size, size_t count, FILE *file)
{
    CHECK(!executing && file != NULL && file == owned_stream); ++reads;
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
    if (calls == fail_call && (forwards - 1) % 4 + 1 == fail_layer) {
        output[0] = NAN; /* Simulate a partially written workspace before failure. */
        error->code = CINDER_ERROR_NONFINITE;
        strcpy(error->message, "Injected layer failure.");
        return error->code;
    }
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
static cinder_status_t tracked_load(const char *path, cinder_model_t **model, cinder_error_t *error)
{
    ++loads; return cinder_model_load(path, model, error);
}
static cinder_status_t tracked_input(const char *path, size_t count, float **values, cinder_error_t *error)
{
    ++input_reads; return cinder_input_read(path, count, values, error);
}
#define cinder_model_load tracked_load
#define cinder_input_read tracked_input
#include "../src/cli_session.c"
#undef cinder_model_load
#undef cinder_input_read

#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
static int fake_getres(clockid_t clock, struct timespec *value)
{
    CHECK(clock == CLOCK_MONOTONIC_RAW && executing && calls == 0);
    CHECK(allocations == 23 && loads == 1 && input_reads == 1 && opens == 2 && closes == 2);
    CHECK(owned_stream == NULL && live > 0);
    ++queries; advance(UINT64_C(100000000000)); *value = resolution; return query_result;
}
static int fake_gettime(clockid_t clock, struct timespec *value)
{
    CHECK(clock == CLOCK_MONOTONIC_RAW && executing && queries == 1 && calls >= 10);
    ++clock_reads;
    CHECK(calls == 10 + clock_reads / 2);
    if (clock_reads % 2 == 0) CHECK(element_index == 0); /* No output scan inside timing. */
    else CHECK(elements == calls * reference.count);
    if (large_uptime && clock_reads == 1) current = (struct timespec){(time_t)INT64_MAX - 100, 999999500};
    *value = current;
    if (clock_reads == fault_read) {
        switch (clock_fault) {
            case CLOCK_READ_FAIL: return -1;
            case CLOCK_NEGATIVE_SECONDS: value->tv_sec = -1; break;
            case CLOCK_NEGATIVE_NANOS: value->tv_nsec = -1; break;
            case CLOCK_INVALID_NANOS: value->tv_nsec = 1000000000L; break;
            case CLOCK_ZERO: *value = last_start; break;
            case CLOCK_BACKWARD: case CLOCK_BETWEEN:
                *value = clock_fault == CLOCK_BACKWARD ? last_start : last_end;
                if (value->tv_nsec == 0) { --value->tv_sec; value->tv_nsec = 999999999; }
                else --value->tv_nsec;
                break;
            case CLOCK_OVERFLOW:
                *value = last_start; value->tv_sec += (time_t)UINT64_C(18446744074); break;
            default: break;
        }
    }
    if (clock_reads % 2) last_start = *value; else last_end = *value;
    return 0;
}
#define clock_getres fake_getres
#define clock_gettime fake_gettime
#endif
#include "../src/timer.c"
#undef clock_getres
#undef clock_gettime

static cinder_status_t tracked_run(cinder_model_t *model, const float *input, size_t count,
    float *output, size_t capacity, cinder_error_t *error)
{
    CHECK(executing && queries == 1 && owned_stream == NULL && closes == 2);
    CHECK(count <= COUNT(saved_input) && capacity == reference.count);
    if (calls == 0) {
        saved_model = model; saved_source = input; saved_destination = output;
        memcpy(saved_input, input, count * sizeof(float));
    }
    CHECK(model == saved_model && input == saved_source && output == saved_destination);
    CHECK(memcmp(saved_input, input, count * sizeof(float)) == 0);
    ++calls; element_index = 0;
    CHECK(clock_reads == (calls <= 10 ? 0 : (calls - 10) * 2 - 1));
    for (size_t i = 0; i < capacity; ++i) output[i] = -123.0f;
    cinder_status_t status = cinder_model_run(model, input, count, output, capacity, error);
    CHECK(memcmp(saved_input, input, count * sizeof(float)) == 0);
    if (status == CINDER_OK) {
        CHECK(compare_vector(NULL, "benchmark_acceptance", "probabilities", output, capacity, reference, true));
        if (calls == corrupt_call) output[corrupt_element] = corrupt_value;
    } else {
        for (size_t i = 0; i < capacity; ++i) CHECK(output[i] == -123.0f);
    }
    advance(calls <= 10 ? UINT64_C(1000000000) : durations[(calls - 11) % 3]);
    return status;
}
static int finite_value(double value) { return isfinite(value); }
static int observe_finite(float value)
{
    CHECK(executing && clock_reads % 2 == 0 && element_index < reference.count);
    float expected = saved_destination[element_index++]; ++elements;
    CHECK(value == expected || (isnan(value) && isnan(expected)));
    advance(UINT64_C(1000000000)); /* Observation time must not enter statistics. */
    return finite_value(value);
}
#define cinder_model_run tracked_run
#undef isfinite
#define isfinite observe_finite
#include "../src/benchmark.c"
#undef isfinite
#define isfinite(value) finite_value((double)(value))
#undef cinder_model_run
#include "../src/benchmark_stats.c"
static cinder_status_t tracked_execute(const cinder_cli_session_t *session, uint32_t iterations,
    cinder_benchmark_result_t *result, cinder_benchmark_failure_t *failure)
{
    cinder_benchmark_result_t saved;
    memcpy(&saved, result, sizeof(saved));
    executing = true;
    cinder_status_t status = cinder_benchmark_execute(session, iterations, result, failure);
    executing = false;
    /* Even a late clock failure must not expose partially accumulated results. */
    if (status != CINDER_OK) CHECK(memcmp(result, &saved, sizeof(saved)) == 0);
    return status;
}
#undef fopen
#undef fclose
#undef fread
#undef malloc
#undef calloc
#undef free

static int output_failed(FILE *file)
{
    CHECK((live == 0 && live_bytes == 0) ||
          (inspecting && file == captured_out && live == 1 && live_bytes == 4 * sizeof(cinder_layer_info_t)));
    CHECK(owned_stream == NULL && !executing);
    if (file == captured_err) return fail_stderr;
    CHECK(file == captured_out); return ++writes == fail_write;
}
static int output_printf(FILE *file, const char *format, ...)
{
    int fail = output_failed(file);
    va_list args; va_start(args, format);
    int result;
    if (fail) {
        if (partial_write && file == captured_out) {
            char bytes[512]; CHECK(vsnprintf(bytes, sizeof(bytes), format, args) > 0);
            CHECK(fputc(bytes[0], file) != EOF);
        }
        result = -1;
    } else result = vfprintf(file, format, args);
    va_end(args); return result;
}
static int output_puts(const char *text, FILE *file)
{
    if (!output_failed(file)) return fputs(text, file);
    if (partial_write && file == captured_out) CHECK(fputc(text[0], file) != EOF);
    return EOF;
}
static int output_putc(int value, FILE *file)
{
    return output_failed(file) ? EOF : fputc(value, file);
}
static int output_flush(FILE *file)
{
    CHECK(live == 0 && owned_stream == NULL && file == captured_out); ++flushes;
    return fail_flush ? EOF : fflush(file);
}
#undef stdout
#undef stderr
#define stdout captured_out
#define stderr captured_err
#define fprintf output_printf
#define fputs output_puts
#define fputc output_putc
#define fflush output_flush
#define cinder_benchmark_execute tracked_execute
#define cinder_model_load tracked_load
#define malloc tracked_malloc
#define free tracked_free
#define main cli_main
#include "../src/cli.c"
#undef main
#undef cinder_model_load
#undef malloc
#undef free
#undef stdout
#undef stderr
#undef fprintf
#undef fputs
#undef fputc
#undef fflush
#undef cinder_benchmark_execute

static void reset(void)
{
    CHECK(live == 0 && live_bytes == 0 && owned_stream == NULL);
    allocations = frees = fail_allocation = opens = closes = reads = fail_open = fail_close = 0;
    loads = input_reads = calls = forwards = elements = element_index = fail_call = fail_layer = 0;
    corrupt_call = corrupt_element = queries = clock_reads = fault_read = writes = flushes = fail_write = 0;
    executing = fail_flush = fail_stderr = partial_write = large_uptime = false;
    inspecting = false;
    saved_model = NULL; saved_source = NULL; saved_destination = NULL;
    clock_fault = CLOCK_GOOD; query_result = 0; corrupt_value = NAN;
    current = (struct timespec){100, 999999500}; resolution = (struct timespec){0, 42};
    last_start = last_end = (struct timespec){0, 0};
    durations[0] = 1000; durations[1] = 2000; durations[2] = 3000;
    captured_out = tmpfile(); captured_err = tmpfile(); CHECK(captured_out != NULL && captured_err != NULL);
}
static void capture(FILE *file, char *text, size_t capacity)
{
    CHECK(fflush(file) == 0 && fseek(file, 0, SEEK_SET) == 0);
    size_t bytes = fread(text, 1, capacity - 1, file);
    CHECK(feof(file) && !ferror(file)); text[bytes] = '\0'; CHECK(fclose(file) == 0);
}
static void invoke_args(int argc, char **argv, int expected)
{
    int actual = cli_main(argc, argv);
    CHECK(live == 0 && live_bytes == 0 && owned_stream == NULL && !executing);
    CHECK(frees == allocations - (fail_allocation != 0 && fail_allocation <= allocations ? 1 : 0));
    capture(captured_out, out_text, sizeof(out_text)); capture(captured_err, err_text, sizeof(err_text));
    if (actual != expected) fprintf(test_stderr(), "Unexpected command result %d: %s%s", actual, out_text, err_text);
    CHECK(actual == expected); ++invocations;
}
static void invoke(const char *count, int expected)
{
    char *args[] = {"cinder", "bench", model_path, input_path, "--iterations", (char *)count, NULL};
    invoke_args(6, args, expected);
}
static void diagnostic_is(const char *stage, const char *tag, const char *message)
{
    char expected[512]; snprintf(expected, sizeof(expected), "cinder: %s [%s]: %s\n", stage, tag, message);
    CHECK(out_text[0] == '\0' && writes == 0 && flushes == 0);
    if (strcmp(err_text, expected) != 0) fprintf(test_stderr(), "Expected %sActual %s", expected, err_text);
    CHECK(strcmp(err_text, expected) == 0);
}
static void success(void)
{
    reset(); invoke("3", 0);
    const char *name = strrchr(model_path, '/'); CHECK(name != NULL);
    char expected[512]; snprintf(expected, sizeof(expected), "Model: %s\nIterations: 3\n"
        "Warm-up iterations: 10\nClock: monotonic\nClock resolution (ns): 42\n"
        "Mean latency (us): 2\nMinimum latency (us): 1\nMaximum latency (us): 3\n"
        "Throughput (inferences/s): 500000\n", name + 1);
    CHECK(strcmp(out_text, expected) == 0 && err_text[0] == '\0');
    CHECK(allocations == 23 && loads == 1 && input_reads == 1 && opens == 2 && closes == 2);
    CHECK(queries == 1 && clock_reads == 6 && calls == 13 && forwards == 52);
    CHECK(elements == 13 * reference.count && flushes == 1);
}
static void arguments(void)
{
    char *args[] = {"cinder", "bench", model_path, input_path, "--iterations", "3", "extra", NULL};
    const char *bad_counts[] = {"", "0", "1000001", "18446744073709551616", "999999999999x",
        "-1", "+1", "1.0", "0x1", "1e3", " 1", "1\n", "\xD9\xA1"};
    for (size_t i = 0; i < COUNT(bad_counts); ++i) {
        reset(); invoke(bad_counts[i], 2);
        CHECK(out_text[0] == '\0' && strstr(err_text, "cinder: usage:") == err_text);
        CHECK(loads == 0 && input_reads == 0 && allocations == 0 && opens == 0 && queries == 0 && calls == 0);
    }
    for (int argc = 2; argc <= 7; ++argc) {
        if (argc == 6) continue;
        reset(); invoke_args(argc, args, 2);
        CHECK(loads == 0 && allocations == 0 && opens == 0 && queries == 0);
    }
    for (int slot = 2; slot <= 4; ++slot) {
        char *old = args[slot]; args[slot] = "";
        reset(); invoke_args(6, args, 2); CHECK(opens == 0 && allocations == 0 && queries == 0);
        args[slot] = old;
    }
    /* Prove max-count parser acceptance without running a million real models. */
    reset(); fail_open = 1; invoke("1000000", 1);
    CHECK(opens == 1 && allocations == 0 && queries == 0 && strstr(err_text, "[IO]") != NULL);
}
static void setup_failures(void)
{
    for (size_t point = 1; point <= 23; ++point) {
        reset(); fail_allocation = point; invoke("3", 1);
        CHECK(allocations == point && queries == 0 && calls == 0 && clock_reads == 0);
        CHECK(out_text[0] == '\0' && writes == 0 && strstr(err_text, "[ALLOCATION]") != NULL);
        success();
    }
    for (size_t point = 1; point <= 2; ++point) {
        reset(); fail_open = point; invoke("3", 1);
        CHECK(opens == point && closes == point - 1 && queries == 0 && calls == 0);
        CHECK(out_text[0] == '\0' && strstr(err_text, "[IO]") != NULL); success();
        reset(); fail_close = point; invoke("3", 1);
        CHECK(opens == point && closes == point && queries == 0 && calls == 0);
        CHECK(out_text[0] == '\0' && strstr(err_text, "[IO]") != NULL); success();
    }
}
static void timing_failures(void)
{
    reset(); query_result = -1; invoke("3", 1);
    diagnostic_is("benchmark", "IO", "Could not query monotonic clock resolution.");
    CHECK(queries == 1 && calls == 0 && clock_reads == 0); success();
    const struct timespec bad_resolutions[] = {{0,0}, {-1,0}, {0,-1}, {0,1000000000L}};
    for (size_t i = 0; i < COUNT(bad_resolutions); ++i) {
        reset(); resolution = bad_resolutions[i]; invoke("3", 1);
        diagnostic_is("benchmark", "IO", "Monotonic clock resolution is invalid.");
        CHECK(queries == 1 && calls == 0 && clock_reads == 0); success();
    }
    for (size_t read = 1; read <= 6; ++read) {
        for (int mode = CLOCK_READ_FAIL; mode <= CLOCK_INVALID_NANOS; ++mode) {
            reset(); fault_read = read; clock_fault = mode; invoke("3", 1);
            char stage[64]; snprintf(stage, sizeof(stage), "benchmark iteration %zu", (read + 1) / 2);
            diagnostic_is(stage, "IO", mode == CLOCK_READ_FAIL ? (read % 2 ?
                "Could not read start timestamp." : "Could not read end timestamp.") :
                "Monotonic clock returned an invalid timestamp.");
            CHECK(clock_reads == read && calls == 10 + read / 2);
            CHECK(elements == (10 + (read - 1) / 2) * reference.count); success();
        }
    }
    for (size_t call = 1; call <= 3; ++call) {
        for (int mode = CLOCK_ZERO; mode <= CLOCK_BETWEEN; ++mode) {
            if (mode == CLOCK_BETWEEN && call == 1) continue;
            reset(); clock_fault = mode; fault_read = 2 * call - (mode == CLOCK_BETWEEN);
            invoke("3", 1);
            char stage[64]; snprintf(stage, sizeof(stage), "benchmark iteration %zu", call);
            diagnostic_is(stage, "IO", mode == CLOCK_ZERO ?
                "Measured interval is zero; clock resolution is insufficient." : "Monotonic clock moved backwards.");
            CHECK(clock_reads == fault_read && calls == 10 + fault_read / 2); success();
        }
    }
    if (sizeof(time_t) >= sizeof(int64_t)) {
        reset(); resolution = (struct timespec){(time_t)UINT64_C(18446744074),0}; invoke("3", 1);
        diagnostic_is("benchmark", "OVERFLOW", "Clock resolution nanoseconds are not representable.");
        CHECK(calls == 0); success();
        reset(); clock_fault = CLOCK_OVERFLOW; fault_read = 2; invoke("3", 1);
        diagnostic_is("benchmark iteration 1", "OVERFLOW", "Elapsed nanoseconds are not representable.");
        CHECK(calls == 11 && clock_reads == 2); success();
        reset(); durations[0] = UINT64_MAX; durations[1] = 1; invoke("3", 1);
        diagnostic_is("benchmark iteration 2", "OVERFLOW", "Total elapsed nanoseconds are not representable.");
        CHECK(calls == 12 && clock_reads == 4 && elements == 12 * reference.count); success();
        reset(); large_uptime = true; invoke("3", 0);
        CHECK(strstr(out_text, "Mean latency (us): 2\nMinimum latency (us): 1\nMaximum latency (us): 3\n") != NULL);
    }
}
static void quantized_and_late_zero_intervals(void)
{
    /* 125/3 ns clock ticks can produce 41 and 42 ns integer differences.
     * Resolution is metadata: retain both, without rounding 41 up to 42. */
    reset(); durations[0] = 41; durations[1] = 42; durations[2] = 42; invoke("3", 0);
    CHECK(strstr(out_text, "Clock resolution (ns): 42\nMean latency (us): 0.0416666667\n"
        "Minimum latency (us): 0.041\nMaximum latency (us): 0.042\n"
        "Throughput (inferences/s): 24000000\n") != NULL);
    CHECK(calls == 13 && clock_reads == 6 && elements == 13 * reference.count);

    /* Recreate both saved failure positions and a failure on the final call.
     * Fixed timestamps, no dependency on runner speed or a native zero event. */
    const size_t positions[] = {9853, 37270, 100000};
    for (size_t i = 0; i < COUNT(positions); ++i) {
        const size_t call = positions[i];
        reset(); durations[0] = 41; durations[1] = 42; durations[2] = 42;
        clock_fault = CLOCK_ZERO; fault_read = 2 * call;
        invoke("100000", 1);
        char stage[64]; snprintf(stage, sizeof(stage), "benchmark iteration %zu", call);
        diagnostic_is(stage, "IO", "Measured interval is zero; clock resolution is insufficient.");
        CHECK(queries == 1 && clock_reads == 2 * call && calls == 10 + call);
        CHECK(forwards == calls * 4 && elements == (10 + call - 1) * reference.count);
        CHECK(last_start.tv_sec == last_end.tv_sec && last_start.tv_nsec == last_end.tv_nsec);
        success(); /* A fresh command still succeeds after rejection and cleanup. */
    }
    puts("Benchmark timing: 41/42 ns samples retained; zero at 9853, 37270, and 100000 "
         "stops immediately with no partial result/report, complete cleanup, and successful recovery.");
}

static void inference_failures(void)
{
    for (size_t call = 1; call <= 13; ++call) {
        char stage[64]; snprintf(stage, sizeof(stage), "benchmark %s %zu",
            call <= 10 ? "warm-up" : "iteration", call <= 10 ? call : call - 10);
        for (size_t layer = 1; layer <= 4; ++layer) {
            reset(); fail_call = call; fail_layer = layer; invoke("3", 1);
            diagnostic_is(stage, "NONFINITE", "Injected layer failure.");
            CHECK(calls == call && forwards == (call - 1) * 4 + layer);
            CHECK(clock_reads == (call <= 10 ? 0 : 2 * (call - 10)));
            CHECK(elements == (call - 1) * reference.count); success();
        }
        const float invalid[] = {NAN, INFINITY, -INFINITY};
        for (size_t element = 0; element < reference.count; ++element) {
            for (size_t value = 0; value < COUNT(invalid); ++value) {
                reset(); corrupt_call = call; corrupt_element = element; corrupt_value = invalid[value]; invoke("3", 1);
                char message[128]; snprintf(message, sizeof(message), "Output value at index %zu is not finite.", element);
                diagnostic_is(stage, "NONFINITE", message);
                CHECK(calls == call && elements == (call - 1) * reference.count + element + 1);
                CHECK(clock_reads == (call <= 10 ? 0 : 2 * (call - 10)));
            }
        }
        success();
    }
    reset(); fail_call = 12; fail_layer = 2; clock_fault = CLOCK_READ_FAIL; fault_read = 4; invoke("3", 1);
    diagnostic_is("benchmark iteration 2", "NONFINITE", "Injected layer failure.");
    CHECK(clock_reads == 4 && calls == 12); success();
    reset(); corrupt_call = 12; clock_fault = CLOCK_ZERO; fault_read = 4; invoke("3", 1);
    diagnostic_is("benchmark iteration 2", "IO", "Measured interval is zero; clock resolution is insufficient.");
    CHECK(elements == 11 * reference.count); success();
    if (sizeof(time_t) >= sizeof(int64_t)) {
        reset(); durations[0] = UINT64_MAX; durations[1] = 1; corrupt_call = 12; invoke("3", 1);
        diagnostic_is("benchmark iteration 2", "NONFINITE", "Output value at index 0 is not finite.");
        CHECK(clock_reads == 4 && elements == 11 * reference.count + 1); success();
    }
}
static void output_failures(void)
{
    success(); size_t points = writes;
    for (size_t point = 1; point <= points; ++point) {
        reset(); fail_write = point; partial_write = true; invoke("3", 1);
        CHECK(writes == point && flushes == 0 && calls == 13 && clock_reads == 6);
        CHECK(strcmp(err_text, "cinder: output [IO]: Could not write benchmark output.\n") == 0);
        CHECK(strstr(out_text, "Throughput (inferences/s):") == NULL); success();
    }
    reset(); fail_flush = true; invoke("3", 1);
    CHECK(flushes == 1 && calls == 13 && strstr(out_text, "Throughput (inferences/s): 500000\n") != NULL);
    CHECK(strcmp(err_text, "cinder: output [IO]: Could not write benchmark output.\n") == 0); success();
    reset(); fail_flush = fail_stderr = true; invoke("3", 1); CHECK(err_text[0] == '\0'); success();
}
static void fixture(const char *root, bool mnist)
{
    CHECK(snprintf(model_path, sizeof(model_path), "%s/%s", root,
        mnist ? "examples/mnist/mnist.cnd" : "tests/fixtures/tiny/tiny.cnd") < (int)sizeof(model_path));
    CHECK(snprintf(input_path, sizeof(input_path), "%s/%s", root,
        mnist ? "tests/fixtures/mnist/inputs/00000.f32" : "tests/fixtures/tiny/inputs/worked.f32") < (int)sizeof(input_path));
    if (!mnist) reference = numpy_network_cases[0].softmax;
    else {
        char path[4096]; static float values[10];
        CHECK(snprintf(path, sizeof(path), "%s/tests/fixtures/mnist/probabilities.f32", root) < (int)sizeof(path));
        FILE *file = fopen(path, "rb"); CHECK(file != NULL);
        for (size_t i = 0; i < COUNT(values); ++i) {
            unsigned char bytes[4]; CHECK(fread(bytes, 1, 4, file) == 4);
            uint32_t bits = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
                (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
            memcpy(values + i, &bits, sizeof(bits));
        }
        CHECK(fclose(file) == 0); reference = (numpy_vector_t){values, COUNT(values)};
    }
}
static void inspection(void)
{
    char *args[] = {"cinder", "info", model_path, NULL};
    reset(); inspecting = true; invoke_args(3, args, 0);
    CHECK(loads == 1 && opens == 1 && closes == 1);
    CHECK(input_reads == 0 && calls == 0 && forwards == 0 && queries == 0 && clock_reads == 0);
    CHECK(strstr(out_text, "Layers: 4\n") != NULL && err_text[0] == '\0');
    size_t allocation_count = allocations, write_count = writes;
    for (size_t failure = 1; failure <= allocation_count; ++failure) {
        reset(); inspecting = true; fail_allocation = failure; invoke_args(3, args, 1);
        CHECK(out_text[0] == '\0' && strstr(err_text, "[ALLOCATION]") != NULL);
        CHECK(closes == 1 && input_reads == 0 && calls == 0 && queries == 0);
        reset(); inspecting = true; invoke_args(3, args, 0);
    }
    for (size_t failure = 1; failure <= write_count; ++failure) {
        reset(); inspecting = true; fail_write = failure; partial_write = true;
        invoke_args(3, args, 1);
        CHECK(strcmp(err_text, "cinder: output [IO]: Could not write inspection output.\n") == 0);
    }
    reset(); inspecting = true; fail_flush = true; invoke_args(3, args, 1);
    CHECK(strcmp(err_text, "cinder: output [IO]: Could not write inspection output.\n") == 0);
    reset(); inspecting = true; fail_close = 1; invoke_args(3, args, 1);
    CHECK(out_text[0] == '\0' && strstr(err_text, "[IO]") != NULL && closes == 1);
    reset(); inspecting = true; invoke_args(3, args, 0);
    printf("Inspection acceptance: %zu allocation points and %zu output writes faulted; no inference/input/timing, no retained blocks/bytes/files.\n",
           allocation_count, write_count);
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    for (int mnist = 0; mnist < 2; ++mnist) {
        fixture(argv[1], mnist != 0);
        inspection();
        if (!CINDER_HAVE_MONOTONIC_RAW_CLOCK) {
            reset(); invoke("3", 1);
            diagnostic_is("benchmark", "UNSUPPORTED", "No monotonic clock is available.");
            CHECK(allocations == 23 && calls == 0 && queries == 0 && clock_reads == 0 && closes == 2);
            continue;
        }
        success(); arguments(); setup_failures(); timing_failures();
        if (!mnist) quantized_and_late_zero_intervals();
        inference_failures(); output_failures();
        /* Neither extreme is the last sample; the minimum is not the first.
         * Catch accidental first/last assignment instead of true min/max. */
        reset(); durations[0] = 3000; durations[1] = 1000; durations[2] = 2000; invoke("3", 0);
        CHECK(strstr(out_text, "Mean latency (us): 2\nMinimum latency (us): 1\nMaximum latency (us): 3\n") != NULL);
        CHECK(strstr(out_text, "Throughput (inferences/s): 500000\n") != NULL);
        reset(); invoke("0001", 0);
        CHECK(calls == 11 && clock_reads == 2 && elements == 11 * reference.count);
        CHECK(strstr(out_text, "Iterations: 1\n") != NULL && strstr(out_text, "Mean latency (us): 1\n") != NULL);
        reset(); durations[0] = 1; durations[1] = 1; durations[2] = 2; invoke("3", 0);
        CHECK(strstr(out_text, "Mean latency (us): 0.00133333333\n") != NULL);
        CHECK(strstr(out_text, "Throughput (inferences/s): 750000000\n") != NULL);
        printf("Benchmark acceptance %s: real model/CLI, deterministic clock/report, faults and recovery; zero retained blocks/bytes/files.\n", mnist ? "MNIST" : "tiny");
    }
    printf("Benchmark acceptance: %zu command invocations passed; clock backend %s; no speed threshold.\n",
        invocations, CINDER_HAVE_MONOTONIC_RAW_CLOCK ? "controlled" : "unavailable");
    return 0;
}
