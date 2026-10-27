/* Substitute only the platform calls; test the real timer/state/arithmetic. */
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/timer.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static struct timespec resolution, current;
static int resolution_result, read_result;
static size_t resolution_calls, read_calls;
static int fake_getres(clockid_t clock, struct timespec *value)
{
    CHECK(clock == CLOCK_MONOTONIC_RAW && value != NULL);
    ++resolution_calls;
    *value = resolution;
    return resolution_result;
}
static int fake_gettime(clockid_t clock, struct timespec *value)
{
    CHECK(clock == CLOCK_MONOTONIC_RAW && value != NULL);
    ++read_calls;
    *value = current;
    return read_result;
}
#define clock_getres fake_getres
#define clock_gettime fake_gettime
#include "../src/timer.c"
#undef clock_getres
#undef clock_gettime

static void error_is(const cinder_error_t *error, cinder_status_t code, const char *message)
{
    CHECK(error->code == code && strcmp(error->message, message) == 0);
}
static void reset(void)
{
    resolution = (struct timespec){0, 10};
    current = (struct timespec){0, 0};
    resolution_result = read_result = 0;
    resolution_calls = read_calls = 0;
}
static void empty(const cinder_timer_t *timer)
{
    CHECK(timer->resolution_ns == 0 && !timer->running && !timer->has_previous_end);
    CHECK(timer->start.seconds == 0 && timer->start.nanoseconds == 0);
    CHECK(timer->previous_end.seconds == 0 && timer->previous_end.nanoseconds == 0);
}
static void initialization(void)
{
    reset();
    cinder_timer_t timer = {0};
    cinder_error_t error = {CINDER_ERROR_IO, "stale"};
    CHECK(cinder_timer_init(NULL, &error) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(resolution_calls == 0 && read_calls == 0);
    CHECK(cinder_timer_init(&timer, &error) == CINDER_OK);
    CHECK(timer.resolution_ns == 10 && resolution_calls == 1 && read_calls == 0);
    error_is(&error, CINDER_OK, "");

    const struct timespec invalid[] = {{0, 0}, {-1, 0}, {0, -1}, {0, 1000000000L}};
    for (size_t i = 0; i < sizeof(invalid) / sizeof(invalid[0]); ++i) {
        resolution = invalid[i];
        CHECK(cinder_timer_init(&timer, &error) == CINDER_ERROR_IO);
        error_is(&error, CINDER_ERROR_IO, "Monotonic clock resolution is invalid.");
        empty(&timer);
        CHECK(cinder_timer_init(&timer, NULL) == CINDER_ERROR_IO);
    }
    resolution_result = -1;
    CHECK(cinder_timer_init(&timer, &error) == CINDER_ERROR_IO);
    error_is(&error, CINDER_ERROR_IO, "Could not query monotonic clock resolution.");
    empty(&timer);
    resolution_result = 0;
    resolution = (struct timespec){2, 3};
    CHECK(cinder_timer_init(&timer, NULL) == CINDER_OK);
    CHECK(timer.resolution_ns == UINT64_C(2000000003));
    if (sizeof(time_t) >= sizeof(int64_t)) {
        resolution = (struct timespec){(time_t)UINT64_C(18446744073), 709551615};
        CHECK(cinder_timer_init(&timer, &error) == CINDER_OK);
        CHECK(timer.resolution_ns == UINT64_MAX);
        ++resolution.tv_nsec; /* Maximum + one: addition would overflow. */
        CHECK(cinder_timer_init(&timer, &error) == CINDER_ERROR_OVERFLOW);
        error_is(&error, CINDER_ERROR_OVERFLOW,
                 "Clock resolution nanoseconds are not representable.");
        empty(&timer);
        resolution = (struct timespec){(time_t)UINT64_C(18446744074), 0};
        CHECK(cinder_timer_init(&timer, NULL) == CINDER_ERROR_OVERFLOW);
        empty(&timer);
    }
    resolution = (struct timespec){0, 1};
    CHECK(cinder_timer_init(&timer, &error) == CINDER_OK); /* Recovery. */
    error_is(&error, CINDER_OK, "");
    CHECK(timer.resolution_ns == 1 && read_calls == 0);
}

static void interval(struct timespec start, struct timespec end,
    cinder_status_t expected, uint64_t expected_ns, const char *message)
{
    for (int details = 0; details < 2; ++details) {
        reset();
        cinder_timer_t timer;
        cinder_error_t error = {CINDER_ERROR_IO, "stale"};
        cinder_error_t *destination = details ? &error : NULL;
        CHECK(cinder_timer_init(&timer, NULL) == CINDER_OK);
        current = start;
        CHECK(cinder_timer_start(&timer, destination) == CINDER_OK);
        CHECK(timer.running && read_calls == 1);
        if (details) error_is(&error, CINDER_OK, "");
        current = end;
        uint64_t elapsed = 123;
        CHECK(cinder_timer_stop(&timer, &elapsed, destination) == expected);
        CHECK(!timer.running && resolution_calls == 1 && read_calls == 2);
        CHECK(elapsed == (expected == CINDER_OK ? expected_ns : 123));
        CHECK(timer.has_previous_end == (expected == CINDER_OK));
        if (details) error_is(&error, expected, message);
        if (expected == CINDER_OK) {
            CHECK(timer.previous_end.seconds == end.tv_sec);
            CHECK(timer.previous_end.nanoseconds == end.tv_nsec);
        } else { /* An operational failure consumes the measurement; a new one can succeed. */
            current = (struct timespec){1, 1};
            CHECK(cinder_timer_start(&timer, destination) == CINDER_OK);
            current.tv_nsec = 2;
            CHECK(cinder_timer_stop(&timer, &elapsed, destination) == CINDER_OK);
            CHECK(elapsed == 1);
        }
    }
}

static void arithmetic(void)
{
    interval((struct timespec){0, 0}, (struct timespec){0, 1}, CINDER_OK, 1, "");
    /* Keep positive observations below, equal to, and above 10 ns resolution. */
    interval((struct timespec){7, 100}, (struct timespec){7, 110}, CINDER_OK, 10, "");
    interval((struct timespec){7, 100}, (struct timespec){7, 111}, CINDER_OK, 11, "");
    interval((struct timespec){3, 999999900}, (struct timespec){4, 100}, CINDER_OK, 200, "");
    interval((struct timespec){3, 999999999}, (struct timespec){4, 0}, CINDER_OK, 1, "");
    interval((struct timespec){3, 100}, (struct timespec){5, 300}, CINDER_OK, UINT64_C(2000000200), "");
    interval((struct timespec){1, 0}, (struct timespec){1, 0}, CINDER_ERROR_IO, 0,
             "Measured interval is zero; clock resolution is insufficient.");
    interval((struct timespec){2, 0}, (struct timespec){1, 999999999}, CINDER_ERROR_IO, 0,
             "Monotonic clock moved backwards.");
    interval((struct timespec){2, 5}, (struct timespec){2, 4}, CINDER_ERROR_IO, 0,
             "Monotonic clock moved backwards.");
    if (sizeof(time_t) >= sizeof(int64_t)) {
        /* Uptime is too large to convert to uint64 ns or double without loss;
         * subtracting first still retains a one-nanosecond interval. */
        interval((struct timespec){(time_t)INT64_MAX, 7},
                 (struct timespec){(time_t)INT64_MAX, 8}, CINDER_OK, 1, "");
        interval((struct timespec){0, 0},
                 (struct timespec){(time_t)UINT64_C(18446744073), 709551615}, CINDER_OK, UINT64_MAX, "");
        /* Borrow reduces an initially too-large seconds difference to fit. */
        interval((struct timespec){0, 999999999},
                 (struct timespec){(time_t)UINT64_C(18446744074), 709551614}, CINDER_OK, UINT64_MAX, "");
        interval((struct timespec){0, 0},
                 (struct timespec){(time_t)UINT64_C(18446744073), 709551616}, CINDER_ERROR_OVERFLOW, 0,
                 "Elapsed nanoseconds are not representable.");
        interval((struct timespec){0, 0},
                 (struct timespec){(time_t)UINT64_C(18446744074), 0}, CINDER_ERROR_OVERFLOW, 0,
                 "Elapsed nanoseconds are not representable.");
    }
}

static void clock_failures(void)
{
    const struct timespec invalid[] = {{-1, 0}, {0, -1}, {0, 1000000000L}};
    for (int details = 0; details < 2; ++details) {
        for (size_t failure = 0; failure < 4; ++failure) {
            for (int at_end = 0; at_end < 2; ++at_end) {
                reset();
                cinder_timer_t timer;
                cinder_error_t error;
                cinder_error_t *destination = details ? &error : NULL;
                CHECK(cinder_timer_init(&timer, destination) == CINDER_OK);
                if (at_end) CHECK(cinder_timer_start(&timer, destination) == CINDER_OK);
                if (failure == 0) read_result = -1;
                else current = invalid[failure - 1];
                uint64_t elapsed = 456;
                cinder_status_t status = at_end ? cinder_timer_stop(&timer, &elapsed, destination) :
                                                   cinder_timer_start(&timer, destination);
                CHECK(status == CINDER_ERROR_IO && !timer.running && !timer.has_previous_end);
                CHECK(elapsed == 456 && read_calls == (size_t)(at_end + 1));
                if (details) error_is(&error, CINDER_ERROR_IO, failure ?
                    "Monotonic clock returned an invalid timestamp." : at_end ?
                    "Could not read end timestamp." : "Could not read start timestamp.");
                read_result = 0; current = (struct timespec){10, 0};
                CHECK(cinder_timer_start(&timer, destination) == CINDER_OK);
                current.tv_nsec = 1;
                CHECK(cinder_timer_stop(&timer, &elapsed, destination) == CINDER_OK);
                CHECK(elapsed == 1);
                if (details) error_is(&error, CINDER_OK, "");
            }
        }
    }
}

static void state_and_ordering(void)
{
    reset();
    cinder_timer_t timer = {0}, independent;
    cinder_error_t error;
    uint64_t elapsed = 789;
    CHECK(cinder_timer_start(NULL, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_timer_start(&timer, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_timer_stop(NULL, &elapsed, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_timer_stop(&timer, &elapsed, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(read_calls == 0 && elapsed == 789);
    CHECK(cinder_timer_init(&timer, NULL) == CINDER_OK);
    CHECK(cinder_timer_init(&independent, NULL) == CINDER_OK);
    current = (struct timespec){10, 5};
    CHECK(cinder_timer_start(&timer, NULL) == CINDER_OK);
    CHECK(cinder_timer_start(&timer, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_timer_stop(&timer, NULL, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(timer.running && read_calls == 1);
    current.tv_nsec = 10;
    CHECK(cinder_timer_stop(&timer, &elapsed, NULL) == CINDER_OK && elapsed == 5);
    CHECK(cinder_timer_stop(&timer, &elapsed, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(read_calls == 2);
    current = (struct timespec){9, 999999999};
    CHECK(cinder_timer_start(&timer, &error) == CINDER_ERROR_IO);
    error_is(&error, CINDER_ERROR_IO, "Monotonic clock moved backwards.");
    current = (struct timespec){10, 9};
    CHECK(cinder_timer_start(&timer, &error) == CINDER_ERROR_IO);
    CHECK(!timer.running && timer.previous_end.seconds == 10 && timer.previous_end.nanoseconds == 10);
    current.tv_nsec = 10; /* Equality to previous end is permitted. */
    CHECK(cinder_timer_start(&timer, &error) == CINDER_OK);
    read_result = -1;
    CHECK(cinder_timer_stop(&timer, &elapsed, &error) == CINDER_ERROR_IO);
    CHECK(!timer.running && timer.previous_end.nanoseconds == 10 && elapsed == 5);
    read_result = 0;
    CHECK(cinder_timer_start(&timer, &error) == CINDER_OK);
    current.tv_nsec = 30;
    CHECK(cinder_timer_stop(&timer, &elapsed, &error) == CINDER_OK && elapsed == 20);
    /* No shared last-end state between timers. */
    current = (struct timespec){0, 0};
    CHECK(cinder_timer_start(&independent, NULL) == CINDER_OK);
    current.tv_nsec = 1;
    CHECK(cinder_timer_stop(&independent, &elapsed, NULL) == CINDER_OK && elapsed == 1);
    CHECK(timer.previous_end.nanoseconds == 30);
    CHECK(cinder_timer_init(&timer, &error) == CINDER_OK); /* Explicit fresh sequence. */
    CHECK(!timer.has_previous_end && !timer.running && timer.resolution_ns == 10);
    CHECK(cinder_timer_start(&timer, NULL) == CINDER_OK);
    current.tv_nsec = 2;
    CHECK(cinder_timer_stop(&timer, &elapsed, NULL) == CINDER_OK && elapsed == 1);
}

int main(void)
{
    initialization(); arithmetic(); clock_failures(); state_and_ordering();
    puts("Timer: checked resolution, normalization, subtraction/borrow/overflow, zero intervals, ordering, state, failures and recovery passed.");
    return 0;
}
