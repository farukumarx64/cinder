#include "benchmark.h"

#include <math.h>
#include <stdio.h>

static cinder_status_t statistics_error(cinder_error_t *error,
    cinder_status_t status, const char *message)
{
    if (error != NULL) {
        error->code = status;
        snprintf(error->message, sizeof(error->message), "%s", message);
    }
    return status;
}

cinder_status_t cinder_benchmark_statistics(const cinder_benchmark_result_t *result,
    cinder_benchmark_statistics_t *statistics, cinder_error_t *error)
{
    if (result == NULL || statistics == NULL) {
        return statistics_error(error, CINDER_ERROR_INVALID_ARGUMENT,
            "Benchmark measurements and statistics destination are required.");
    }
    /* Guard the integer denominators before converting or dividing. A zero
     * interval has already failed in the timer; never repair one here. */
    if (result->iterations == 0 || result->total_ns == 0 ||
        result->minimum_ns == 0 || result->maximum_ns == 0) {
        return statistics_error(error, CINDER_ERROR_NONFINITE,
            "Benchmark statistics must be finite and positive.");
    }
    /* Check aggregate bounds using quotient/remainder, avoiding min/max * N
     * multiplication which could overflow even for valid measurements. */
    uint64_t quotient = result->total_ns / result->iterations;
    uint64_t remainder = result->total_ns % result->iterations;
    if (result->iterations > CINDER_BENCHMARK_MAX_ITERATIONS ||
        result->resolution_ns == 0 || result->minimum_ns > result->maximum_ns ||
        result->maximum_ns > result->total_ns || quotient < result->minimum_ns ||
        quotient > result->maximum_ns || (quotient == result->maximum_ns && remainder != 0)) {
        return statistics_error(error, CINDER_ERROR_INVALID_ARGUMENT,
            "Benchmark measurements are inconsistent.");
    }
    double count = (double)result->iterations, total = (double)result->total_ns;
    cinder_benchmark_statistics_t pending = {
        .mean_us = (total / count) / 1000.0,
        .minimum_us = (double)result->minimum_ns / 1000.0,
        .maximum_us = (double)result->maximum_ns / 1000.0,
        .inferences_per_second = (count / total) * 1000000000.0
    };
    if (!isfinite(pending.mean_us) || pending.mean_us <= 0.0 ||
        !isfinite(pending.minimum_us) || pending.minimum_us <= 0.0 ||
        !isfinite(pending.maximum_us) || pending.maximum_us <= 0.0 ||
        !isfinite(pending.inferences_per_second) || pending.inferences_per_second <= 0.0) {
        return statistics_error(error, CINDER_ERROR_NONFINITE,
            "Benchmark statistics must be finite and positive.");
    }
    *statistics = pending;
    if (error != NULL) *error = (cinder_error_t){0};
    return CINDER_OK;
}
