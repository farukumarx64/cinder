#include "timer.h"

#include <stdio.h>

/* CMake probes the raw clock ID and both POSIX calls with
 * _POSIX_C_SOURCE=200809L (plus _DARWIN_C_SOURCE on macOS), linking librt when
 * needed. The unavailable branch also builds without POSIX clock types. */
#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
#include <time.h>
#endif

#define NANOSECONDS_PER_SECOND UINT64_C(1000000000)

static cinder_status_t timer_result(cinder_error_t *error,
    cinder_status_t status, const char *message)
{
    if (error != NULL) {
        error->code = status;
        error->message[0] = '\0';
        if (message[0] != '\0') {
            snprintf(error->message, sizeof(error->message), "%s", message);
        }
    }
    return status;
}

#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
static bool normalized(const struct timespec *raw, cinder_timestamp_t *stamp)
{
    if (raw->tv_sec < 0 || (uintmax_t)raw->tv_sec > INT64_MAX ||
        raw->tv_nsec < 0 || raw->tv_nsec >= 1000000000L) return false;
    stamp->seconds = (int64_t)raw->tv_sec;
    stamp->nanoseconds = (int32_t)raw->tv_nsec;
    return true;
}

static bool to_nanoseconds(uint64_t seconds, uint32_t nanoseconds, uint64_t *value)
{
    if (seconds > (UINT64_MAX - nanoseconds) / NANOSECONDS_PER_SECOND) return false;
    *value = seconds * NANOSECONDS_PER_SECOND + nanoseconds;
    return true;
}

static bool before(cinder_timestamp_t left, cinder_timestamp_t right)
{
    return left.seconds < right.seconds ||
        (left.seconds == right.seconds && left.nanoseconds < right.nanoseconds);
}

static cinder_status_t read_timestamp(cinder_timestamp_t *stamp,
    const char *failure, cinder_error_t *error)
{
    struct timespec raw = {0};
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &raw) != 0) {
        return timer_result(error, CINDER_ERROR_IO, failure);
    }
    if (!normalized(&raw, stamp)) {
        return timer_result(error, CINDER_ERROR_IO,
                            "Monotonic clock returned an invalid timestamp.");
    }
    return CINDER_OK;
}
#endif

cinder_status_t cinder_timer_init(cinder_timer_t *timer, cinder_error_t *error)
{
    if (timer == NULL) {
        return timer_result(error, CINDER_ERROR_INVALID_ARGUMENT,
                            "Timer pointer must not be NULL.");
    }
    *timer = (cinder_timer_t){0};
#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
    struct timespec raw = {0};
    if (clock_getres(CLOCK_MONOTONIC_RAW, &raw) != 0) {
        return timer_result(error, CINDER_ERROR_IO,
                            "Could not query monotonic clock resolution.");
    }
    cinder_timestamp_t resolution;
    if (!normalized(&raw, &resolution) ||
        (resolution.seconds == 0 && resolution.nanoseconds == 0)) {
        return timer_result(error, CINDER_ERROR_IO,
                            "Monotonic clock resolution is invalid.");
    }
    if (!to_nanoseconds((uint64_t)resolution.seconds,
                        (uint32_t)resolution.nanoseconds, &timer->resolution_ns)) {
        return timer_result(error, CINDER_ERROR_OVERFLOW,
                            "Clock resolution nanoseconds are not representable.");
    }
    return timer_result(error, CINDER_OK, "");
#else
    return timer_result(error, CINDER_ERROR_UNSUPPORTED,
                        "No monotonic clock is available.");
#endif
}

cinder_status_t cinder_timer_start(cinder_timer_t *timer, cinder_error_t *error)
{
    if (timer == NULL || timer->resolution_ns == 0 || timer->running) {
        return timer_result(error, CINDER_ERROR_INVALID_ARGUMENT,
                            "An initialized, inactive timer is required.");
    }
#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
    cinder_timestamp_t start;
    cinder_status_t status = read_timestamp(&start, "Could not read start timestamp.", error);
    if (status != CINDER_OK) return status;
    if (timer->has_previous_end && before(start, timer->previous_end)) {
        return timer_result(error, CINDER_ERROR_IO, "Monotonic clock moved backwards.");
    }
    timer->start = start;
    timer->running = true;
    return timer_result(error, CINDER_OK, "");
#else
    return timer_result(error, CINDER_ERROR_UNSUPPORTED,
                        "No monotonic clock is available.");
#endif
}

cinder_status_t cinder_timer_stop(cinder_timer_t *timer, uint64_t *elapsed_ns,
                                  cinder_error_t *error)
{
    if (timer == NULL || timer->resolution_ns == 0 || !timer->running || elapsed_ns == NULL) {
        return timer_result(error, CINDER_ERROR_INVALID_ARGUMENT,
                            "An active timer and elapsed output are required.");
    }
    timer->running = false;
#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
    cinder_timestamp_t end;
    cinder_status_t status = read_timestamp(&end, "Could not read end timestamp.", error);
    if (status != CINDER_OK) return status;
    if (before(end, timer->start)) {
        return timer_result(error, CINDER_ERROR_IO, "Monotonic clock moved backwards.");
    }
    uint64_t seconds = (uint64_t)(end.seconds - timer->start.seconds);
    int32_t nanoseconds = end.nanoseconds - timer->start.nanoseconds;
    if (nanoseconds < 0) {
        --seconds; /* Ordering above guarantees a whole second to borrow. */
        nanoseconds += 1000000000;
    }
    uint64_t elapsed;
    if (!to_nanoseconds(seconds, (uint32_t)nanoseconds, &elapsed)) {
        return timer_result(error, CINDER_ERROR_OVERFLOW,
                            "Elapsed nanoseconds are not representable.");
    }
    if (elapsed == 0) {
        return timer_result(error, CINDER_ERROR_IO,
                            "Measured interval is zero; clock resolution is insufficient.");
    }
    timer->previous_end = end;
    timer->has_previous_end = true;
    *elapsed_ns = elapsed;
    return timer_result(error, CINDER_OK, "");
#else
    return timer_result(error, CINDER_ERROR_UNSUPPORTED,
                        "No monotonic clock is available.");
#endif
}
