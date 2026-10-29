/* Manual process-test control: keep the real clock-resolution query and all
 * runtime/CLI code, but force equal timestamps in every measured interval. */
#include <time.h>
static int zero_clock(clockid_t id, struct timespec *value)
{
    if (id != CLOCK_MONOTONIC_RAW) return -1;
    *value = (struct timespec){10, 7};
    return 0;
}
#define clock_gettime zero_clock
#include "../../../src/timer.c"
