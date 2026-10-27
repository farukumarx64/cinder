#include "../src/timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

int main(void)
{
    cinder_timer_t timer = {.resolution_ns = 99, .running = true, .has_previous_end = true};
    cinder_error_t error = {0};
    CHECK(cinder_timer_init(&timer, &error) == CINDER_ERROR_UNSUPPORTED);
    CHECK(error.code == CINDER_ERROR_UNSUPPORTED);
    CHECK(strcmp(error.message, "No monotonic clock is available.") == 0);
    CHECK(timer.resolution_ns == 0 && !timer.running && !timer.has_previous_end);
    CHECK(timer.start.seconds == 0 && timer.start.nanoseconds == 0);
    CHECK(timer.previous_end.seconds == 0 && timer.previous_end.nanoseconds == 0);
    CHECK(cinder_timer_init(&timer, NULL) == CINDER_ERROR_UNSUPPORTED);
    CHECK(cinder_timer_init(NULL, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    uint64_t elapsed = 7;
    CHECK(cinder_timer_start(&timer, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(cinder_timer_stop(&timer, &elapsed, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    CHECK(elapsed == 7);
    puts("Timer: unavailable backend rejects initialization without a fallback.");
    return 0;
}
