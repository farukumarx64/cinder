#include "cinder/model.h"
#include "numpy_compare.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Keep checks active in Release and release loaded models on test failures.
 * The Python integration also deliberately supplies incorrect fixtures. */
#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        goto cleanup; \
    } \
} while (0)

static const struct {
    const char *name;
    const char *file;
    size_t output_size;
} stages[] = {
    {"dense1", "prefixes/dense1.cnd", 4},
    {"relu", "prefixes/relu.cnd", 4},
    {"logits", "prefixes/logits.cnd", 2},
    {"softmax", "tiny.cnd", 2}
};

static bool result_ok(cinder_status_t status, const cinder_error_t *error,
                       const char *stage, const char *operation)
{
    if (status != CINDER_OK) {
        fprintf(stderr, "%s/%s: status %d: %s\n", stage, operation, (int)status, error->message);
        return false;
    }
    return error->code == CINDER_OK && error->message[0] == '\0';
}

static bool make_path(char *path, size_t capacity, const char *root, const char *relative)
{
    int length = snprintf(path, capacity, "%s/%s", root, relative);
    if (length < 0 || (size_t)length >= capacity) {
        fprintf(stderr, "Fixture path exceeds test path capacity.\n");
        return false;
    }
    return true;
}

static bool read_input(const char *path, float input[2])
{
    /* Test-only reading of the separate raw input fixture, not a new runtime
     * input-file API. All models load first, establishing host binary32 support. */
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Could not open input fixture: %s\n", path);
        return false;
    }
    unsigned char bytes[8];
    size_t obtained = fread(bytes, 1, sizeof(bytes), file);
    int tail = fgetc(file);
    bool complete = obtained == sizeof(bytes) && tail == EOF && feof(file) && !ferror(file);
    if (fclose(file) != 0) {
        complete = false;
    }
    if (!complete) {
        fprintf(stderr, "Input fixture must contain exactly eight bytes: %s\n", path);
        return false;
    }
    for (size_t i = 0; i < 2; ++i) {
        const unsigned char *word = bytes + 4 * i;
        uint32_t bits = (uint32_t)word[0] | ((uint32_t)word[1] << 8) |
                        ((uint32_t)word[2] << 16) | ((uint32_t)word[3] << 24);
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
            fprintf(stderr, "Non-finite input fixture: %s\n", path);
            return false;
        }
        memcpy(input + i, &bits, sizeof(bits));
    }
    return true;
}

int main(int argc, char **argv)
{
    cinder_model_t *models[COUNT(stages)] = {NULL, NULL, NULL, NULL};
    numpy_comparison_stats_t stats = {0};
    numpy_comparison_stats_t stage_stats[COUNT(stages)] = {{0}};
    int result = EXIT_FAILURE;
    char path[4096];
    REQUIRE(argc == 2);
    REQUIRE(numpy_tiny_network.input_size == 2 && numpy_tiny_network.hidden_size == 4 &&
            numpy_tiny_network.output_size == 2);
    REQUIRE(COUNT(numpy_network_cases) == 10);

    for (size_t stage = 0; stage < COUNT(stages); ++stage) {
        REQUIRE(make_path(path, sizeof(path), argv[1], stages[stage].file));
        cinder_error_t error;
        cinder_status_t status = cinder_model_load(path, &models[stage], &error);
        REQUIRE(result_ok(status, &error, stages[stage].name, "load"));
        if (cinder_model_input_size(models[stage]) != 2 ||
            cinder_model_output_size(models[stage]) != stages[stage].output_size) {
            fprintf(stderr, "%s: loaded model shape mismatch.\n", stages[stage].name);
            goto cleanup;
        }
    }

    for (size_t i = 0; i < COUNT(numpy_network_cases); ++i) {
        const numpy_network_case_t *fixture = &numpy_network_cases[i];
        const numpy_vector_t expected[] = {fixture->dense1, fixture->relu,
                                           fixture->logits, fixture->softmax};
        char relative[128];
        int length = snprintf(relative, sizeof(relative), "inputs/%s.f32", fixture->name);
        REQUIRE(length >= 0 && (size_t)length < sizeof(relative));
        REQUIRE(make_path(path, sizeof(path), argv[1], relative));
        float input[2];
        REQUIRE(read_input(path, input));
        REQUIRE(fixture->input.count == COUNT(input));
        REQUIRE(isfinite(input[0]) && isfinite(input[1]));
        if (memcmp(input, fixture->input.data, sizeof(input)) != 0) {
            fprintf(stderr, "%s: input fixture differs from saved NumPy input bits.\n", fixture->name);
            goto cleanup;
        }
        for (size_t stage = 0; stage < COUNT(stages); ++stage) {
            REQUIRE(expected[stage].count == stages[stage].output_size);
            float storage[] = {-101, NAN, NAN, NAN, NAN, NAN, -102};
            size_t size = cinder_model_output_size(models[stage]);
            size_t capacity = size + i % 2; /* Exact capacity and one spare slot. */
            cinder_error_t error;
            cinder_status_t status = cinder_model_run(models[stage], input, COUNT(input),
                                                     storage + 1, capacity, &error);
            REQUIRE(result_ok(status, &error, stages[stage].name, fixture->name));
            REQUIRE(compare_vector(&stage_stats[stage], fixture->name, stages[stage].name,
                                   storage + 1, size, expected[stage], true));
            for (size_t tail = size + 1; tail <= 5; ++tail) {
                REQUIRE(isnan(storage[tail]));
            }
            REQUIRE(storage[0] == -101 && storage[6] == -102);
            REQUIRE(memcmp(input, fixture->input.data, sizeof(input)) == 0);
            if (stage == 3) {
                REQUIRE(storage[1] >= 0 && storage[1] <= 1);
                REQUIRE(storage[2] >= 0 && storage[2] <= 1);
                REQUIRE(fabs((double)storage[1] + storage[2] - 1.0) < 1e-6);
            }
            /* Each prefix executes every preceding layer on the original input.
             * No saved intermediate reference is ever used as inference input. */
        }
    }
    for (size_t stage = 0; stage < COUNT(stages); ++stage) {
        const numpy_comparison_stats_t *current = &stage_stats[stage];
        REQUIRE(current->comparisons == COUNT(numpy_network_cases));
        REQUIRE(current->elements == COUNT(numpy_network_cases) * stages[stage].output_size);
        stats.comparisons += current->comparisons;
        stats.elements += current->elements;
        stats.largest_absolute_error = fmax(stats.largest_absolute_error, current->largest_absolute_error);
        stats.largest_tolerance_fraction = fmax(stats.largest_tolerance_fraction, current->largest_tolerance_fraction);
        printf("%s: %zu vectors / %zu values, max absolute error %.9g.\n", stages[stage].name,
               current->comparisons, current->elements, current->largest_absolute_error);
    }
    REQUIRE(stats.comparisons == 40 && stats.elements == 120);
    printf("Round trip: %zu vectors / %zu values, max absolute error %.9g, "
           "max fraction of tolerance %.9g (atol = rtol = 1e-5).\n",
           stats.comparisons, stats.elements, stats.largest_absolute_error, stats.largest_tolerance_fraction);
    result = EXIT_SUCCESS;

cleanup:
    for (size_t stage = 0; stage < COUNT(models); ++stage) {
        cinder_model_free(models[stage]);
    }
    return result;
}
