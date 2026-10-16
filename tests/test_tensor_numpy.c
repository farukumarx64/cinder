#include "cinder/tensor.h"

/* CMake chooses saved fixtures or freshly generated build-directory fixtures. */
#include <numpy_fixtures.h>

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static size_t comparisons;
static size_t elements;
static double largest_absolute_error;
static double largest_tolerance_fraction;

static bool compare_vector(
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
        if (report) {
            if (difference > largest_absolute_error) {
                largest_absolute_error = difference;
            }
            if (difference / allowed > largest_tolerance_fraction) {
                largest_tolerance_fraction = difference / allowed;
            }
        }
    }
    if (report) {
        ++comparisons;
        elements += count;
    }
    return true;
}

static void check_comparator(void)
{
    const float expected[] = {0.0f, 2.0f};
    const numpy_vector_t reference = {expected, COUNT(expected)};
    const float close[] = {0.000001f, 2.000001f};
    const float wrong[] = {0.01f, 2.0f};
    const float reordered[] = {2.0f, 0.0f};
    CHECK(compare_vector("self", "equal", expected, 2, reference, false));
    CHECK(compare_vector("self", "close", close, 2, reference, false));
    CHECK(!compare_vector("self", "wrong", wrong, 2, reference, false));
    CHECK(!compare_vector("self", "order", reordered, 2, reference, false));
    CHECK(!compare_vector("self", "shape", expected, 1, reference, false));
    CHECK(!compare_vector("self", "empty", expected, 0, (numpy_vector_t){expected, 0}, false));
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < COUNT(bad); ++i) {
        float nonfinite[] = {0.0f, bad[i]};
        numpy_vector_t invalid_reference = {nonfinite, 2};
        CHECK(!compare_vector("self", "actual", nonfinite, 2, reference, false));
        CHECK(!compare_vector("self", "reference", expected, 2, invalid_reference, false));
        CHECK(!compare_vector("self", "both", nonfinite, 2, invalid_reference, false));
    }
}

static void check_status(cinder_status_t status, const cinder_error_t *error,
                         const char *fixture, const char *stage)
{
    if (status != CINDER_OK) {
        fprintf(stderr, "%s/%s: Cinder status %d: %s\n", fixture, stage, status, error->message);
        exit(EXIT_FAILURE);
    }
    CHECK(error->code == CINDER_OK && error->message[0] == '\0');
}

static void check_output(const char *name, const char *stage, const float *output,
                         size_t count, numpy_vector_t reference)
{
    CHECK(compare_vector(name, stage, output, count, reference, true));
    CHECK(isnan(output[count])); /* Spare capacity must remain untouched. */
}

static void check_dense(const numpy_dense_fixture_t *fixture)
{
    CHECK(fixture->rows > 0 && fixture->cols > 0);
    CHECK(fixture->weights.count == fixture->rows * fixture->cols);
    CHECK(fixture->input.count == fixture->cols);
    CHECK(fixture->bias.count == fixture->rows);
    cinder_error_t error;
    cinder_tensor_t *weights = NULL;
    check_status(cinder_tensor_create(fixture->rows, fixture->cols, &weights, &error),
                 &error, fixture->name, "create");
    memcpy(weights->data, fixture->weights.data, fixture->weights.count * sizeof(float));
    size_t count = weights->rows;
    float *output = malloc((count + 1) * sizeof(*output));
    float *probabilities = malloc((count + 1) * sizeof(*probabilities));
    CHECK(output != NULL && probabilities != NULL);
    for (size_t i = 0; i <= count; ++i) {
        output[i] = NAN;
        probabilities[i] = NAN;
    }

    check_status(cinder_tensor_matvec(weights, fixture->input.data, fixture->input.count,
                                     output, count + 1, &error), &error, fixture->name, "matvec");
    check_output(fixture->name, "matvec", output, count, fixture->matvec);
    check_status(cinder_tensor_add_bias(output, count, fixture->bias.data, fixture->bias.count,
                                       &error), &error, fixture->name, "bias");
    check_output(fixture->name, "bias", output, count, fixture->biased);

    /* Check probabilities directly from logits as well as after ReLU. */
    memcpy(probabilities, output, count * sizeof(*output));
    check_status(cinder_tensor_softmax(probabilities, count, &error),
                 &error, fixture->name, "softmax(logits)");
    check_output(fixture->name, "softmax(logits)", probabilities, count, fixture->softmax);
    check_status(cinder_tensor_relu(output, count, &error), &error, fixture->name, "relu");
    check_output(fixture->name, "relu", output, count, fixture->relu);
    check_status(cinder_tensor_softmax(output, count, &error),
                 &error, fixture->name, "softmax(relu)");
    check_output(fixture->name, "softmax(relu)", output, count, fixture->relu_softmax);

    CHECK(memcmp(weights->data, fixture->weights.data, fixture->weights.count * sizeof(float)) == 0);
    free(probabilities);
    free(output);
    cinder_tensor_free(weights);
}

static void check_activation(const numpy_activation_fixture_t *fixture)
{
    size_t count = fixture->input.count;
    CHECK(count > 0);
    float *output = malloc((count + 1) * sizeof(*output));
    CHECK(output != NULL);
    output[count] = NAN;
    cinder_error_t error;
    memcpy(output, fixture->input.data, count * sizeof(*output));
    check_status(cinder_tensor_relu(output, count, &error), &error, fixture->name, "relu");
    check_output(fixture->name, "relu", output, count, fixture->relu);
    memcpy(output, fixture->input.data, count * sizeof(*output));
    check_status(cinder_tensor_softmax(output, count, &error), &error, fixture->name, "softmax");
    check_output(fixture->name, "softmax", output, count, fixture->softmax);
    free(output);
}

int main(void)
{
    check_comparator();
    CHECK(COUNT(numpy_dense_fixtures) > 0 && COUNT(numpy_activation_fixtures) > 0);
    for (size_t i = 0; i < COUNT(numpy_dense_fixtures); ++i) {
        check_dense(&numpy_dense_fixtures[i]);
    }
    for (size_t i = 0; i < COUNT(numpy_activation_fixtures); ++i) {
        check_activation(&numpy_activation_fixtures[i]);
    }
    printf("NumPy references passed: %zu dense fixtures, %zu activation fixtures, "
           "%zu vectors, %zu elements.\nMax absolute error: %.9g; "
           "max fraction of tolerance: %.9g (must be <= 1).\n",
           COUNT(numpy_dense_fixtures), COUNT(numpy_activation_fixtures), comparisons,
           elements, largest_absolute_error, largest_tolerance_fraction);
    return EXIT_SUCCESS;
}
