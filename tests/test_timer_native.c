#include "../src/timer.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
#include <errno.h>
#include <time.h>
#endif

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

int main(void)
{
    cinder_timer_t timer;
    cinder_error_t error;
#if CINDER_HAVE_MONOTONIC_RAW_CLOCK
    CHECK(cinder_timer_init(&timer, &error) == CINDER_OK);
    CHECK(timer.resolution_ns > 0);
    size_t positive = 0, unresolved = 0;
    for (size_t i = 0; i < 32; ++i) {
        CHECK(cinder_timer_start(&timer, &error) == CINDER_OK);
        if (i == 0) {
            struct timespec remaining = {0, 2000000};
            while (nanosleep(&remaining, &remaining) != 0) CHECK(errno == EINTR);
        }
        uint64_t elapsed = 0;
        cinder_status_t status = cinder_timer_stop(&timer, &elapsed, &error);
        if (status == CINDER_OK) {
            CHECK(elapsed > 0 && error.code == CINDER_OK && error.message[0] == '\0');
            ++positive;
        } else {
            /* Adjacent real clock reads can coincide. Test the policy without
             * requiring any host speed or retrying/discarding benchmark calls. */
            CHECK(status == CINDER_ERROR_IO && elapsed == 0);
            CHECK(strcmp(error.message,
                "Measured interval is zero; clock resolution is insufficient.") == 0);
            ++unresolved;
        }
    }
    printf("Timer native CLOCK_MONOTONIC_RAW: resolution=%" PRIu64
           " ns; %zu positive and %zu correctly rejected zero intervals (not an inference benchmark).\n",
           timer.resolution_ns, positive, unresolved);
#else
#if defined(__APPLE__) || defined(__linux__)
    CHECK(0 && "Supported macOS/Linux hosts must provide the raw monotonic backend.");
#endif
    CHECK(cinder_timer_init(&timer, &error) == CINDER_ERROR_UNSUPPORTED);
    puts("Timer native: monotonic backend unavailable; explicit rejection passed.");
#endif
    return 0;
}
