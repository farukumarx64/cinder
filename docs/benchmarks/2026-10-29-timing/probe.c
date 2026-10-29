/* Manual diagnostic, not a benchmark result or a speed-gated CTest.
 * Link the real benchmark/core libraries; observe unchanged native timestamps
 * before the real timer handles them. Recording adds overhead, so these
 * observations must not be presented as ordinary CLI latency measurements. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <mach/mach_time.h>
#endif
#include "benchmark.h"

static struct timespec captured[2];
static size_t reads;
static int capture_clock(clockid_t id, struct timespec *value)
{
    int status = clock_gettime(id, value);
    if (status == 0) captured[reads % 2] = *value;
    ++reads;
    return status;
}
#define clock_gettime capture_clock
#include "../../../src/timer.c"
#undef clock_gettime

static void json_string(const char *value)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p < 32) printf("\\u%04x", (unsigned)*p);
        else putchar(*p);
    }
    putchar('"');
}

static int clock_pairs(void)
{
    struct timespec res;
    if (clock_getres(CLOCK_MONOTONIC_RAW, &res) != 0) return 1;
    uint64_t histogram[513] = {0}, above = 0, backwards = 0;
    struct timespec first_zero = {0};
    /* Fixed collection size. Keep every outcome; no retry until positive. */
    for (size_t i = 0; i < 1000000; ++i) {
        struct timespec start, end;
        if (clock_gettime(CLOCK_MONOTONIC_RAW, &start) != 0 ||
            clock_gettime(CLOCK_MONOTONIC_RAW, &end) != 0) return 1;
        int64_t seconds = (int64_t)end.tv_sec - (int64_t)start.tv_sec;
        int64_t nanos = (int64_t)end.tv_nsec - (int64_t)start.tv_nsec;
        if (seconds < 0 || (seconds == 0 && nanos < 0)) { ++backwards; continue; }
        if (nanos < 0) { --seconds; nanos += 1000000000; }
        if (seconds != 0 || nanos > 512) ++above;
        else {
            if (nanos == 0 && histogram[0] == 0) first_zero = start;
            ++histogram[nanos];
        }
    }
    printf("{\"kind\":\"adjacent_clock_reads\",\"pairs\":1000000,"
           "\"resolution\":[%jd,%ld],\"backwards\":%" PRIu64
           ",\"above_512_ns\":%" PRIu64 ",\"first_equal_timestamp\":[%jd,%ld]",
           (intmax_t)res.tv_sec, res.tv_nsec, backwards, above,
           (intmax_t)first_zero.tv_sec, first_zero.tv_nsec);
#ifdef __APPLE__
    mach_timebase_info_data_t info;
    if (mach_timebase_info(&info) != KERN_SUCCESS) return 1;
    printf(",\"mach_timebase_numer\":%u,\"mach_timebase_denom\":%u", info.numer, info.denom);
#endif
    printf(",\"histogram_ns\":{");
    int separator = 0;
    for (size_t i = 0; i < 513; ++i) if (histogram[i]) {
        printf("%s\"%zu\":%" PRIu64, separator ? "," : "", i, histogram[i]);
        separator = 1;
    }
    puts("}}");
    return backwards ? 1 : 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && strcmp(argv[1], "clock") == 0) return clock_pairs();
    if (argc != 4) {
        fputs("Usage: probe clock | probe <model.cnd> <input.bin> <iterations>\n", stderr);
        return 2;
    }
    uint32_t count = 0;
    for (const char *p = argv[3]; *p; ++p) {
        if (*p < '0' || *p > '9' || count > (CINDER_BENCHMARK_MAX_ITERATIONS - (*p - '0')) / 10) return 2;
        count = count * 10 + (uint32_t)(*p - '0');
    }
    if (count == 0) return 2;
    cinder_cli_session_t session = {0};
    cinder_cli_failure_t setup;
    if (cinder_cli_prepare(argv[1], argv[2], &session, &setup) != CINDER_OK) {
        fprintf(stderr, "%s: %s\n", setup.stage, setup.detail.message);
        return 1;
    }
    cinder_benchmark_result_t result = {0};
    cinder_benchmark_failure_t failure;
    cinder_status_t status = cinder_benchmark_execute(&session, count, &result, &failure);
    struct timespec res = {0};
    int res_status = clock_getres(CLOCK_MONOTONIC_RAW, &res);
    printf("{\"kind\":\"instrumented_benchmark\",\"requested\":%" PRIu32
           ",\"status\":%d,\"phase\":%d,\"iteration\":%" PRIu32
           ",\"clock_reads\":%zu,\"resolution_query_status\":%d,\"resolution\":[%jd,%ld],"
           "\"last_start\":[%jd,%ld],\"last_end\":[%jd,%ld],\"published_iterations\":%" PRIu32
           ",\"published_total_ns\":%" PRIu64 ",\"message\":",
           count, status, failure.phase, failure.iteration, reads, res_status,
           (intmax_t)res.tv_sec, res.tv_nsec, (intmax_t)captured[0].tv_sec,
           captured[0].tv_nsec, (intmax_t)captured[1].tv_sec, captured[1].tv_nsec,
           result.iterations, result.total_ns);
    json_string(failure.detail.message);
    printf(",\"output\":[");
    /* All measured calls in this investigation use finite fixture inputs.
     * Only read output after reaching measured inference, including a zero-time
     * rejection after a successful inference. */
    if (status == CINDER_OK || (failure.phase == CINDER_BENCHMARK_MEASURED && reads % 2 == 0)) {
        for (size_t i = 0; i < session.output_size; ++i)
            printf("%s%.9g", i ? "," : "", (double)session.output[i]);
    }
    puts("]}");
    cinder_cli_release(&session);
    return status == CINDER_OK ? 0 : 1;
}
