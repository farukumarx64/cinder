#ifndef CINDER_TIMER_H
#define CINDER_TIMER_H

#include "cinder/error.h"
#include <stdbool.h>
#include <stdint.h>

/* CLI-private monotonic timer. No heap storage, file handles, or cleanup.
 * Initialize before use; only resolution_ns is caller-readable metadata.
 * All other fields are implementation state: do not modify them or interleave
 * measurements on one timer. Independent timers have independent state. */
typedef struct {
    int64_t seconds;
    int32_t nanoseconds;
} cinder_timestamp_t;

typedef struct {
    uint64_t resolution_ns;
    cinder_timestamp_t start;
    cinder_timestamp_t previous_end;
    bool running;
    bool has_previous_end;
} cinder_timer_t;

/* Reset the timer and query CLOCK_MONOTONIC_RAW resolution once. Failure leaves
 * an uninitialized (zeroed) timer. No fallback to calendar or CPU time. */
cinder_status_t cinder_timer_init(cinder_timer_t *timer, cinder_error_t *error);

/* Read a normalized start, checking it against the last successful end.
 * Failure leaves timer unchanged; starting an active timer is a usage error. */
cinder_status_t cinder_timer_start(cinder_timer_t *timer, cinder_error_t *error);

/* Read the end and consume the active measurement, even if the clock fails.
 * Publish elapsed_ns and previous_end only on success. Subtract timestamps
 * before checked uint64 conversion; zero duration is an insufficient-resolution
 * error. Positive durations are retained, even below reported resolution.
 * No rounding, retry, batching, or overhead subtraction is performed.
 * A valid call always attempts the end read: call this even if inference failed,
 * using a separate error object so the inference failure can take precedence.
 * NULL arguments / no active measurement are rejected without a clock read and
 * leave state unchanged. A later start may recover after an operational error;
 * the benchmark caller must nevertheless stop on its first failure. */
cinder_status_t cinder_timer_stop(cinder_timer_t *timer, uint64_t *elapsed_ns,
                                  cinder_error_t *error);

/* Error details are optional and cleared on success. Destinations must not
 * overlap timer storage or each other. No operation prints diagnostics. */

#endif
