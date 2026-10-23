#ifndef CINDER_TEST_NUMPY_COMPARE_H
#define CINDER_TEST_NUMPY_COMPARE_H

#include <numpy_fixtures.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

/* Shared only by reference tests; production code has no NumPy dependency.
 * Pass NULL stats for comparator self-checks that must not affect the totals. */
typedef struct {
    size_t comparisons;
    size_t elements;
    double largest_absolute_error;
    double largest_tolerance_fraction;
} numpy_comparison_stats_t;

static bool compare_vector(
    numpy_comparison_stats_t *stats,
    const char *fixture,
    const char *stage,
    const float *actual,
    size_t count,
    numpy_vector_t reference,
    bool report
)
{
    if (count != reference.count || count == 0) {
        if (report) {
            fprintf(stderr, "%s/%s: shape mismatch or empty vector: %zu vs %zu\n",
                    fixture, stage, count, reference.count);
        }
        return false;
    }
    /* Reject non-finite values before any tolerance arithmetic. */
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(actual[i]) || !isfinite(reference.data[i])) {
            if (report) {
                fprintf(stderr, "%s/%s[%zu]: non-finite actual or reference\n",
                        fixture, stage, i);
            }
            return false;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        double difference = fabs((double)actual[i] - (double)reference.data[i]);
        double allowed = 1e-5 + 1e-5 * fabs((double)reference.data[i]);
        if (difference > allowed) {
            if (report) {
                fprintf(stderr,
                        "%s/%s[%zu]: actual %.9g, reference %.9g, error %.9g > %.9g\n",
                        fixture, stage, i, (double)actual[i], (double)reference.data[i],
                        difference, allowed);
            }
            return false;
        }
        if (stats != NULL) {
            if (difference > stats->largest_absolute_error) {
                stats->largest_absolute_error = difference;
            }
            if (difference / allowed > stats->largest_tolerance_fraction) {
                stats->largest_tolerance_fraction = difference / allowed;
            }
        }
    }
    if (stats != NULL) {
        ++stats->comparisons;
        stats->elements += count;
    }
    return true;
}

#endif /* CINDER_TEST_NUMPY_COMPARE_H */
