/* Real loop with controlled model/timer calls and output-read observation. */
#include "../src/benchmark.h"
#include "../src/timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static max_align_t model_token;
static float input_values[] = {2.0f, -1.0f}, output_values[3];
static cinder_cli_session_t session;
static uint32_t runs, starts, stops, inits, observed, next_element;
static uint32_t fail_run, fail_start, fail_stop, bad_output_call, bad_output_index;
static cinder_status_t init_status;
static float bad_value;
static bool active, varying_intervals, overflow_intervals;
static uint64_t now, started_at;

static cinder_status_t response(cinder_error_t *error, cinder_status_t status, const char *message)
{
    if (error != NULL) {
        error->code = status;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}
static size_t input_size(const cinder_model_t *model)
{
    CHECK(model == session.model); return 2;
}
static size_t output_size(const cinder_model_t *model)
{
    CHECK(model == session.model); return 3;
}
static cinder_status_t model_run(cinder_model_t *model, const float *input, size_t count,
    float *output, size_t capacity, cinder_error_t *error)
{
    CHECK(model == session.model && input == input_values && output == output_values);
    CHECK(count == 2 && capacity == 3 && input[0] == 2.0f && input[1] == -1.0f);
    CHECK(inits == 1);
    ++runs;
    CHECK(active == (runs > 10));
    CHECK(starts == (runs > 10 ? runs - 10 : 0));
    CHECK(stops == (runs > 10 ? runs - 11 : 0));
    next_element = 0;
    now += runs <= 10 ? UINT64_C(1000000000) :
        varying_intervals ? UINT64_C(1000) * (runs - 10) : UINT64_C(1000);
    if (runs == fail_run) return response(error, CINDER_ERROR_DIMENSION_MISMATCH,
                                         "injected inference failure");
    for (size_t i = 0; i < 3; ++i) output[i] = (float)runs + (float)i * 0.25f;
    if (runs == bad_output_call) output[bad_output_index] = bad_value;
    return response(error, CINDER_OK, "");
}
static cinder_status_t timer_init(cinder_timer_t *timer, cinder_error_t *error)
{
    CHECK(runs == 0 && starts == 0 && stops == 0);
    ++inits;
    *timer = (cinder_timer_t){.resolution_ns = 42};
    now += UINT64_C(1000000000); /* Setup time must not contribute. */
    return response(error, init_status, init_status == CINDER_OK ? "" : "injected clock setup failure");
}
static cinder_status_t timer_start(cinder_timer_t *timer, cinder_error_t *error)
{
    CHECK(timer->resolution_ns == 42 && !active && runs == 10 + stops);
    CHECK(observed == runs * 3); /* All warm-up/previous outputs consumed first. */
    ++starts;
    if (starts == fail_start) return response(error, CINDER_ERROR_IO, "injected start failure");
    active = true; started_at = now;
    return response(error, CINDER_OK, "");
}
static cinder_status_t timer_stop(cinder_timer_t *timer, uint64_t *elapsed, cinder_error_t *error)
{
    CHECK(timer->resolution_ns == 42 && active && runs == 10 + starts);
    CHECK(next_element == 0); /* No observation between the two clock reads. */
    ++stops; active = false;
    if (stops == fail_stop) return response(error, CINDER_ERROR_IO, "injected end failure");
    *elapsed = overflow_intervals ? (stops == 1 ? UINT64_MAX : 1) : now - started_at;
    return response(error, CINDER_OK, "");
}
static int observed_finite(float value)
{
    CHECK(!active && next_element < 3);
    float expected = output_values[next_element];
    CHECK(value == expected || (isnan(value) && isnan(expected)));
    ++next_element; ++observed;
    now += UINT64_C(1000000000); /* Observation cost must not enter elapsed time. */
    return isfinite(value);
}

#define cinder_model_input_size input_size
#define cinder_model_output_size output_size
#define cinder_model_run model_run
#define cinder_timer_init timer_init
#define cinder_timer_start timer_start
#define cinder_timer_stop timer_stop
#undef isfinite
#define isfinite observed_finite
/* Intentionally unresolved: direct allocation, deallocation, file access, or
 * report printing added to the real loop would fail this test's link. */
#define malloc benchmark_must_not_malloc
#define calloc benchmark_must_not_calloc
#define realloc benchmark_must_not_realloc
#define free benchmark_must_not_free
#define fopen benchmark_must_not_fopen
#define fclose benchmark_must_not_fclose
#define fread benchmark_must_not_fread
#define printf benchmark_must_not_printf
#define fprintf benchmark_must_not_fprintf
#define fputs benchmark_must_not_fputs
#define fwrite benchmark_must_not_fwrite
#include "../src/benchmark.c"
#undef cinder_model_input_size
#undef cinder_model_output_size
#undef cinder_model_run
#undef cinder_timer_init
#undef cinder_timer_start
#undef cinder_timer_stop
#undef isfinite
#undef malloc
#undef calloc
#undef realloc
#undef free
#undef fopen
#undef fclose
#undef fread
#undef printf
#undef fprintf
#undef fputs
#undef fwrite

static void reset(void)
{
    session = (cinder_cli_session_t){(cinder_model_t *)&model_token, input_values, output_values, 2, 3};
    runs = starts = stops = inits = observed = next_element = 0;
    fail_run = fail_start = fail_stop = bad_output_call = bad_output_index = 0;
    init_status = CINDER_OK;
    bad_value = NAN;
    active = varying_intervals = overflow_intervals = false;
    now = started_at = 0;
    for (size_t i = 0; i < 3; ++i) output_values[i] = NAN;
}
static void failed(cinder_status_t expected, cinder_benchmark_phase_t phase,
    uint32_t iteration, const char *message, bool details)
{
    cinder_benchmark_result_t result, original;
    memset(&result, 0xa5, sizeof(result)); memcpy(&original, &result, sizeof(result));
    cinder_benchmark_failure_t failure = {0};
    CHECK(cinder_benchmark_execute(&session, 3, &result, details ? &failure : NULL) == expected);
    CHECK(memcmp(&result, &original, sizeof(result)) == 0 && !active);
    if (details) {
        CHECK(failure.phase == phase && failure.iteration == iteration);
        CHECK(failure.detail.code == expected && strcmp(failure.detail.message, message) == 0);
    }
}
static void success(uint32_t count, bool varying, bool details)
{
    reset(); varying_intervals = varying;
    cinder_benchmark_result_t result;
    cinder_benchmark_failure_t failure = {CINDER_BENCHMARK_MEASURED, 99, {CINDER_ERROR_IO, "old"}};
    CHECK(cinder_benchmark_execute(&session, count, &result, details ? &failure : NULL) == CINDER_OK);
    CHECK(result.iterations == count && result.resolution_ns == 42);
    CHECK(result.minimum_ns == 1000 && result.maximum_ns == (varying ? count * UINT64_C(1000) : 1000));
    CHECK(result.total_ns == (varying ? (uint64_t)count * (count + 1) / 2 * 1000 : (uint64_t)count * 1000));
    CHECK(runs == count + 10 && starts == count && stops == count && inits == 1);
    CHECK(observed == 3 * (count + 10) && !active);
    if (details) {
        CHECK(failure.phase == CINDER_BENCHMARK_SETUP && failure.iteration == 0);
        CHECK(failure.detail.code == CINDER_OK && failure.detail.message[0] == '\0');
    }
}
static void failures(void)
{
    for (int details = 0; details < 2; ++details) {
        for (uint32_t call = 1; call <= 13; ++call) {
            reset(); fail_run = call;
            failed(CINDER_ERROR_DIMENSION_MISMATCH,
                   call <= 10 ? CINDER_BENCHMARK_WARMUP : CINDER_BENCHMARK_MEASURED,
                   call <= 10 ? call : call - 10, "injected inference failure", details != 0);
            CHECK(runs == call && starts == (call <= 10 ? 0 : call - 10) && stops == starts);
            CHECK(observed == (call - 1) * 3);
            /* A subsequent independent invocation succeeds after every failure. */
            success(1, false, details != 0);
        }
        for (uint32_t call = 1; call <= 3; ++call) {
            reset(); fail_start = call;
            failed(CINDER_ERROR_IO, CINDER_BENCHMARK_MEASURED, call, "injected start failure", details != 0);
            CHECK(runs == 9 + call && starts == call && stops == call - 1 && observed == runs * 3);
            reset(); fail_stop = call;
            failed(CINDER_ERROR_IO, CINDER_BENCHMARK_MEASURED, call, "injected end failure", details != 0);
            CHECK(runs == 10 + call && starts == call && stops == call && observed == (runs - 1) * 3);
            reset(); fail_run = call + 10; fail_stop = call;
            failed(CINDER_ERROR_DIMENSION_MISMATCH, CINDER_BENCHMARK_MEASURED, call,
                   "injected inference failure", details != 0);
            CHECK(stops == call && observed == (runs - 1) * 3);
            reset(); bad_output_call = call + 10; fail_stop = call;
            failed(CINDER_ERROR_IO, CINDER_BENCHMARK_MEASURED, call, "injected end failure", details != 0);
            CHECK(observed == (runs - 1) * 3); /* Timer error precedes non-finite output. */
        }
        const cinder_status_t clock_errors[] = {CINDER_ERROR_IO, CINDER_ERROR_UNSUPPORTED, CINDER_ERROR_OVERFLOW};
        for (size_t i = 0; i < 3; ++i) {
            reset(); init_status = clock_errors[i];
            failed(init_status, CINDER_BENCHMARK_SETUP, 0, "injected clock setup failure", details != 0);
            CHECK(inits == 1 && runs == 0 && starts == 0 && stops == 0 && observed == 0);
        }
        reset(); overflow_intervals = true;
        failed(CINDER_ERROR_OVERFLOW, CINDER_BENCHMARK_MEASURED, 2,
               "Total elapsed nanoseconds are not representable.", details != 0);
        CHECK(runs == 12 && starts == 2 && stops == 2 && observed == 36);
    }
    /* A single representable maximum interval is accepted without wrap. */
    reset(); overflow_intervals = true;
    cinder_benchmark_result_t result;
    CHECK(cinder_benchmark_execute(&session, 1, &result, NULL) == CINDER_OK);
    CHECK(result.iterations == 1 && result.minimum_ns == UINT64_MAX &&
          result.maximum_ns == UINT64_MAX && result.total_ns == UINT64_MAX);
}
static void nonfinite_outputs(void)
{
    const float values[] = {NAN, INFINITY, -INFINITY};
    for (uint32_t call = 1; call <= 13; ++call) {
        for (uint32_t element = 0; element < 3; ++element) {
            for (size_t value = 0; value < 3; ++value) {
                reset(); bad_output_call = call; bad_output_index = element; bad_value = values[value];
                char message[128];
                snprintf(message, sizeof(message), "Output value at index %u is not finite.", element);
                failed(CINDER_ERROR_NONFINITE,
                       call <= 10 ? CINDER_BENCHMARK_WARMUP : CINDER_BENCHMARK_MEASURED,
                       call <= 10 ? call : call - 10, message, true);
                CHECK(runs == call && observed == (call - 1) * 3 + element + 1);
                CHECK(starts == (call <= 10 ? 0 : call - 10) && stops == starts);
            }
        }
    }
}
static void invalid_arguments(void)
{
    for (size_t test = 0; test < 12; ++test) {
        reset();
        cinder_benchmark_result_t result = {0};
        cinder_benchmark_failure_t failure;
        const cinder_cli_session_t *argument = &session;
        cinder_benchmark_result_t *destination = &result;
        uint32_t count = 3;
        switch (test) {
            case 0: argument = NULL; break;
            case 1: destination = NULL; break;
            case 2: session.model = NULL; break;
            case 3: session.input = NULL; break;
            case 4: session.output = NULL; break;
            case 5: session.input_size = 0; break;
            case 6: session.output_size = 0; break;
            case 7: count = 0; break;
            case 8: count = CINDER_BENCHMARK_MAX_ITERATIONS + 1; break;
            case 9: count = UINT32_MAX; break;
            case 10: session.input_size = 1; break;
            case 11: session.output_size = 2; break;
        }
        cinder_status_t expected = test < 10 ? CINDER_ERROR_INVALID_ARGUMENT : CINDER_ERROR_DIMENSION_MISMATCH;
        CHECK(cinder_benchmark_execute(argument, count, destination, &failure) == expected);
        CHECK(failure.detail.code == expected && failure.phase == CINDER_BENCHMARK_SETUP && failure.iteration == 0);
        CHECK(inits == 0 && runs == 0 && starts == 0 && stops == 0 && observed == 0);
    }
}

int main(void)
{
    invalid_arguments();
    success(1, false, true); success(3, true, true);
    success(CINDER_BENCHMARK_MAX_ITERATIONS, false, false);
    failures(); nonfinite_outputs();
    puts("Benchmark loop: 10 untimed warm-ups, exact measured counts through 1,000,000, all outputs observed outside timing, checked totals, failure precedence and recovery passed.");
    return 0;
}
