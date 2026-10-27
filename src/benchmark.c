#include "benchmark.h"
#include "timer.h"

#include <math.h>
#include <stdio.h>

static cinder_status_t benchmark_error(cinder_benchmark_failure_t *failure,
    cinder_status_t status, const char *message)
{
    failure->detail.code = status;
    snprintf(failure->detail.message, sizeof(failure->detail.message), "%s", message);
    return status;
}

static cinder_status_t observe_output(const cinder_cli_session_t *session,
    volatile float *sink, cinder_benchmark_failure_t *failure)
{
    for (size_t i = 0; i < session->output_size; ++i) {
        float value = session->output[i];
        if (!isfinite(value)) {
            failure->detail.code = CINDER_ERROR_NONFINITE;
            snprintf(failure->detail.message, sizeof(failure->detail.message),
                     "Output value at index %zu is not finite.", i);
            return CINDER_ERROR_NONFINITE;
        }
        /* Every finite element is observable. Keep the vectors non-volatile
         * and keep these stores outside the start/end interval. */
        *sink = value;
    }
    return CINDER_OK;
}

cinder_status_t cinder_benchmark_execute(const cinder_cli_session_t *session,
    uint32_t iterations, cinder_benchmark_result_t *result,
    cinder_benchmark_failure_t *failure)
{
    cinder_benchmark_failure_t ignored;
    if (failure == NULL) failure = &ignored;
    *failure = (cinder_benchmark_failure_t){0};
    if (session == NULL || result == NULL || session->model == NULL ||
        session->input == NULL || session->output == NULL ||
        session->input_size == 0 || session->output_size == 0) {
        return benchmark_error(failure, CINDER_ERROR_INVALID_ARGUMENT,
                               "A prepared session and result destination are required.");
    }
    if (iterations == 0 || iterations > CINDER_BENCHMARK_MAX_ITERATIONS) {
        return benchmark_error(failure, CINDER_ERROR_INVALID_ARGUMENT,
                               "Iterations must be between 1 and 1000000.");
    }
    if (session->input_size != cinder_model_input_size(session->model) ||
        session->output_size != cinder_model_output_size(session->model)) {
        return benchmark_error(failure, CINDER_ERROR_DIMENSION_MISMATCH,
                               "Session dimensions do not match the loaded model.");
    }

    cinder_timer_t timer;
    cinder_status_t status = cinder_timer_init(&timer, &failure->detail);
    if (status != CINDER_OK) return status;
    cinder_benchmark_result_t pending = {.resolution_ns = timer.resolution_ns};
    volatile float sink = 0.0f;

    failure->phase = CINDER_BENCHMARK_WARMUP;
    for (uint32_t i = 1; i <= CINDER_BENCHMARK_WARMUPS; ++i) {
        failure->iteration = i;
        status = cinder_model_run(session->model, session->input, session->input_size,
                                  session->output, session->output_size, &failure->detail);
        if (status != CINDER_OK) {
            failure->detail.code = status;
            return status;
        }
        status = observe_output(session, &sink, failure);
        if (status != CINDER_OK) return status;
    }

    failure->phase = CINDER_BENCHMARK_MEASURED;
    for (uint32_t i = 1; i <= iterations; ++i) {
        failure->iteration = i;
        cinder_error_t model_error = {0}, timer_error = {0};
        uint64_t elapsed;
        status = cinder_timer_start(&timer, &failure->detail);
        if (status != CINDER_OK) return status;
        status = cinder_model_run(session->model, session->input, session->input_size,
                                  session->output, session->output_size, &model_error);
        cinder_status_t clock_status = cinder_timer_stop(&timer, &elapsed, &timer_error);
        if (status != CINDER_OK) {
            failure->detail = model_error;
            failure->detail.code = status;
            return status;
        }
        if (clock_status != CINDER_OK) {
            failure->detail = timer_error;
            return clock_status;
        }
        status = observe_output(session, &sink, failure);
        if (status != CINDER_OK) return status;
        if (elapsed > UINT64_MAX - pending.total_ns) {
            return benchmark_error(failure, CINDER_ERROR_OVERFLOW,
                                   "Total elapsed nanoseconds are not representable.");
        }
        pending.total_ns += elapsed;
        if (pending.iterations == 0 || elapsed < pending.minimum_ns) pending.minimum_ns = elapsed;
        if (elapsed > pending.maximum_ns) pending.maximum_ns = elapsed;
        ++pending.iterations;
    }
    *result = pending;
    *failure = (cinder_benchmark_failure_t){0};
    return CINDER_OK;
}
