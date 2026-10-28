#include "../src/benchmark.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static void near(double actual, double expected)
{
    CHECK(isfinite(actual) && actual > 0.0);
    CHECK(fabs(actual - expected) <= fabs(expected) * 1e-14);
}

static void success(cinder_benchmark_result_t raw, double mean, double minimum,
                    double maximum, double throughput)
{
    cinder_benchmark_result_t original;
    memcpy(&original, &raw, sizeof(raw));
    for (int with_error = 0; with_error < 2; ++with_error) {
        cinder_error_t error; memset(&error, 'X', sizeof(error));
        cinder_benchmark_statistics_t stats;
        CHECK(cinder_benchmark_statistics(&raw, &stats, with_error ? &error : NULL) == CINDER_OK);
        near(stats.mean_us, mean); near(stats.minimum_us, minimum);
        near(stats.maximum_us, maximum); near(stats.inferences_per_second, throughput);
        near(stats.mean_us * stats.inferences_per_second, 1000000.0);
        CHECK(memcmp(&raw, &original, sizeof(raw)) == 0);
        if (with_error) CHECK(error.code == CINDER_OK && error.message[0] == '\0');
    }
}

static void rejected(const cinder_benchmark_result_t *raw, cinder_status_t expected)
{
    for (int with_error = 0; with_error < 2; ++with_error) {
        cinder_benchmark_statistics_t stats, original;
        memset(&stats, 0x5a, sizeof(stats)); memcpy(&original, &stats, sizeof(stats));
        cinder_error_t error = {0};
        CHECK(cinder_benchmark_statistics(raw, &stats, with_error ? &error : NULL) == expected);
        CHECK(memcmp(&stats, &original, sizeof(stats)) == 0);
        if (with_error) {
            CHECK(error.code == expected && error.message[0] != '\0');
            if (expected == CINDER_ERROR_NONFINITE) {
                CHECK(strcmp(error.message, "Benchmark statistics must be finite and positive.") == 0);
            }
        }
    }
}

int main(void)
{
    cinder_benchmark_result_t raw = {3, 1, 6000, 1000, 3000};
    success(raw, 2.0, 1.0, 3.0, 500000.0);
    /* Unequal durations 1 and 9 us: average of reciprocal rates is WRONG
     * (555555.55...), whereas two calls over 10 us is exactly 200000/s. */
    success((cinder_benchmark_result_t){2, 42, 10000, 1000, 9000}, 5, 1, 9, 200000);
    success((cinder_benchmark_result_t){1, 42, 1, 1, 1}, .001, .001, .001, 1e9);
    /* 1,1,2 ns: a fractional-ns mean is retained without integer division. */
    success((cinder_benchmark_result_t){3, 42, 4, 1, 2}, 1.0/750.0, .001, .002, 750000000);
    success((cinder_benchmark_result_t){1000000, 42, 1000000, 1, 1}, .001, .001, .001, 1e9);
    success((cinder_benchmark_result_t){1, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX},
            1.8446744073709552e16, 1.8446744073709552e16, 1.8446744073709552e16,
            5.421010862427522e-11);
    success((cinder_benchmark_result_t){2, 1, UINT64_MAX, 1, UINT64_MAX-1},
            9.223372036854776e15, .001, 1.8446744073709552e16, 1.0842021724855044e-10);
    rejected(NULL, CINDER_ERROR_INVALID_ARGUMENT);
    cinder_error_t error;
    CHECK(cinder_benchmark_statistics(&raw, NULL, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(error.code == CINDER_ERROR_INVALID_ARGUMENT);
    const cinder_benchmark_result_t zero[] = {
        {0,1,6000,1000,3000}, {3,1,0,1000,3000}, {3,1,6000,0,3000}, {3,1,6000,1000,0}
    };
    for (size_t i = 0; i < sizeof(zero)/sizeof(zero[0]); ++i) rejected(&zero[i], CINDER_ERROR_NONFINITE);
    const cinder_benchmark_result_t inconsistent[] = {
        {1000001,1,1000001,1,1}, {3,0,6000,1000,3000}, {3,1,6000,3000,1000},
        {3,1,6000,1000,6001}, {3,1,2999,1000,2000}, {3,1,9001,1000,3000},
        {3,1,12000,1000,3000}
    };
    for (size_t i = 0; i < sizeof(inconsistent)/sizeof(inconsistent[0]); ++i) {
        rejected(&inconsistent[i], CINDER_ERROR_INVALID_ARGUMENT);
    }
    puts("Benchmark statistics: known arithmetic, fractional means, rate definition, uint64 bounds, zero guards and unchanged failure output passed.");
    return 0;
}
