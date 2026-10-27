#ifndef CINDER_BENCHMARK_H
#define CINDER_BENCHMARK_H

#include "cli_session.h"
#include <stdint.h>

#define CINDER_BENCHMARK_WARMUPS UINT32_C(10)
#define CINDER_BENCHMARK_MAX_ITERATIONS UINT32_C(1000000)

/* CLI-private raw measurements. Only measured calls contribute; derived
 * latency/throughput and report formatting are separate from this helper. */
typedef struct {
    uint32_t iterations;
    uint64_t resolution_ns;
    uint64_t total_ns;
    uint64_t minimum_ns;
    uint64_t maximum_ns;
} cinder_benchmark_result_t;

typedef enum {
    CINDER_BENCHMARK_SETUP = 0,
    CINDER_BENCHMARK_WARMUP,
    CINDER_BENCHMARK_MEASURED
} cinder_benchmark_phase_t;

typedef struct {
    cinder_benchmark_phase_t phase;
    uint32_t iteration; /* One-based within phase; zero for setup/success. */
    cinder_error_t detail;
} cinder_benchmark_failure_t;

/* Borrow an intact prepared session. Validate counts/pointers/dimensions, query
 * the timer once, run exactly 10 untimed warm-ups, then 1..1,000,000 measured
 * calls. Reuse the same model and buffers; no loading, allocation/free, reset,
 * preprocessing, or printing. The session remains caller-owned on every path;
 * the caller must eventually release it. Input/parameters are preserved, while
 * model workspaces and output are reused according to the model contract.
 * Output may already contain a completed call's values when a later call fails;
 * only the result object below has the all-or-nothing publication guarantee.
 *
 * Every successful call's finite outputs are copied to a local volatile sink
 * outside timing. Measured calls always attempt the end read, including when
 * inference fails; the inference error takes precedence over an end-clock error.
 * Stop on the first failure: no retry or partial result. Publish *result only
 * after all calls and checked integer accumulation succeed; otherwise leave it
 * unchanged. Optional failure details are cleared on success and retain phase,
 * one-based call number, and the original error on failure.
 *
 * The session object, its allocations, result, and optional failure must be disjoint.
 * Do not execute/release/mutate a session concurrently. Independent sessions
 * have independent timers/sinks. All additional storage is constant in N. */
cinder_status_t cinder_benchmark_execute(const cinder_cli_session_t *session,
    uint32_t iterations, cinder_benchmark_result_t *result,
    cinder_benchmark_failure_t *failure);

#endif
