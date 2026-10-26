#include "cinder/model.h"
#include "numpy_compare.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLES 100
#define INPUT_SIZE 784
#define MAX_WIDTH 128
#define MODEL_BYTES 407160
#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define REQUIRE(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        goto cleanup; \
    } \
} while (0)

/* Fixed fixture layout, not a second general-purpose .cnd parser. Prefixes
 * retain the original record/parameter bytes and change only layer_count.
 * The real loader validates every prefix before any inference is attempted. */
static const struct {
    const char *name;
    size_t width;
    size_t end;
    uint32_t type;
    uint32_t input;
    uint32_t payload;
} stages[] = {
    {"dense1", 128, 401952, 1, 784, 401920},
    {"relu", 128, 401968, 2, 128, 0},
    {"logits", 10, 407144, 1, 128, 5160},
    {"probabilities", 10, MODEL_BYTES, 3, 10, 0}
};

static uint32_t decode_word(const unsigned char *word)
{
    return (uint32_t)word[0] | ((uint32_t)word[1] << 8) |
           ((uint32_t)word[2] << 16) | ((uint32_t)word[3] << 24);
}

static bool make_path(char *path, size_t capacity, const char *root, const char *relative)
{
    int length = snprintf(path, capacity, "%s/%s", root, relative);
    return length >= 0 && (size_t)length < capacity;
}

static bool read_bytes(const char *path, unsigned char *bytes, size_t count)
{
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Could not open fixture: %s\n", path);
        return false;
    }
    bool ok = fread(bytes, 1, count, file) == count;
    int tail = fgetc(file);
    ok = ok && tail == EOF && feof(file) && !ferror(file);
    if (fclose(file) != 0) {
        ok = false;
    }
    if (!ok) {
        fprintf(stderr, "%s: expected exactly %zu bytes.\n", path, count);
    }
    return ok;
}

static bool read_floats(const char *path, float *values, size_t count)
{
    /* Bounded test fixture reader. Loading the full model first establishes
     * host binary32 support; memcpy avoids alignment/aliasing assumptions. */
    unsigned char bytes[SAMPLES * MAX_WIDTH * 4];
    if (count > sizeof(bytes) / 4 || !read_bytes(path, bytes, count * 4)) {
        return false;
    }
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = decode_word(bytes + 4 * i);
        memcpy(values + i, &bits, sizeof(bits));
        if (!isfinite(values[i])) {
            fprintf(stderr, "%s[%zu]: non-finite fixture value.\n", path, i);
            return false;
        }
    }
    return true;
}

static bool result_ok(cinder_status_t status, const cinder_error_t *error, const char *context)
{
    if (status != CINDER_OK) {
        fprintf(stderr, "%s: status %d: %s\n", context, (int)status, error->message);
        return false;
    }
    return error->code == CINDER_OK && error->message[0] == '\0';
}

static size_t prediction(const float *values, size_t count)
{
    size_t best = 0;
    for (size_t i = 1; i < count; ++i) {
        if (values[i] > values[best]) {
            best = i; /* Lowest index wins ties, as in Python argmax. */
        }
    }
    return best;
}

int main(int argc, char **argv)
{
    /* Arguments: full saved model, saved reference directory, scratch directory.
     * Scratch files belong to this test and are removed on success or failure. */
    cinder_model_t *models[COUNT(stages)] = {NULL};
    char prefix_paths[3][4096] = {{0}};
    bool created[3] = {false};
    unsigned char *model_bytes = NULL;
    static float references[COUNT(stages)][SAMPLES * MAX_WIDTH];
    unsigned char indices[SAMPLES * 4], labels[SAMPLES * 4];
    unsigned char pixels[SAMPLES * INPUT_SIZE];
    numpy_comparison_stats_t stats[COUNT(stages)] = {{0}};
    size_t agreements = 0, c_correct = 0, python_correct = 0;
    int result = EXIT_FAILURE;
    char path[4096], relative[128];
    cinder_error_t error;
    REQUIRE(argc == 4);
    REQUIRE(result_ok(cinder_model_load(argv[1], &models[3], &error), &error, "full model/load"));
    model_bytes = malloc(MODEL_BYTES);
    REQUIRE(model_bytes != NULL && read_bytes(argv[1], model_bytes, MODEL_BYTES));
    REQUIRE(memcmp(model_bytes, "CNDR", 4) == 0 && decode_word(model_bytes + 4) == 1);
    REQUIRE(decode_word(model_bytes + 8) == INPUT_SIZE && decode_word(model_bytes + 12) == 4);
    for (size_t stage = 0, start = 16; stage < COUNT(stages); ++stage) {
        REQUIRE(decode_word(model_bytes + start) == stages[stage].type);
        REQUIRE(decode_word(model_bytes + start + 4) == stages[stage].input);
        REQUIRE(decode_word(model_bytes + start + 8) == stages[stage].width);
        REQUIRE(decode_word(model_bytes + start + 12) == stages[stage].payload);
        REQUIRE(start + 16 + stages[stage].payload == stages[stage].end);
        start = stages[stage].end;
        if (stage < 3) {
            int length = snprintf(relative, sizeof(relative), "%s.cnd", stages[stage].name);
            REQUIRE(length >= 0 && (size_t)length < sizeof(relative));
            REQUIRE(make_path(prefix_paths[stage], sizeof(prefix_paths[stage]), argv[3], relative));
            FILE *file = fopen(prefix_paths[stage], "wb");
            REQUIRE(file != NULL);
            created[stage] = true;
            model_bytes[12] = (unsigned char)(stage + 1);
            bool written = fwrite(model_bytes, 1, stages[stage].end, file) == stages[stage].end;
            if (fclose(file) != 0) {
                written = false;
            }
            REQUIRE(written);
            REQUIRE(result_ok(cinder_model_load(prefix_paths[stage], &models[stage], &error),
                              &error, stages[stage].name));
        }
        if (cinder_model_input_size(models[stage]) != INPUT_SIZE ||
            cinder_model_output_size(models[stage]) != stages[stage].width) {
            fprintf(stderr, "%s: loaded model shape mismatch.\n", stages[stage].name);
            goto cleanup;
        }
        int length = snprintf(relative, sizeof(relative), "%s.f32", stages[stage].name);
        REQUIRE(length >= 0 && (size_t)length < sizeof(relative));
        REQUIRE(make_path(path, sizeof(path), argv[2], relative));
        REQUIRE(read_floats(path, references[stage], SAMPLES * stages[stage].width));
    }
    REQUIRE(make_path(path, sizeof(path), argv[2], "indices.u32"));
    REQUIRE(read_bytes(path, indices, sizeof(indices)));
    REQUIRE(make_path(path, sizeof(path), argv[2], "labels.u32"));
    REQUIRE(read_bytes(path, labels, sizeof(labels)));
    REQUIRE(make_path(path, sizeof(path), argv[2], "images.u8"));
    REQUIRE(read_bytes(path, pixels, sizeof(pixels)));

    for (size_t row = 0; row < SAMPLES; ++row) {
        REQUIRE(decode_word(indices + row * 4) == row);
        uint32_t label = decode_word(labels + row * 4);
        REQUIRE(label < 10);
        int length = snprintf(relative, sizeof(relative), "inputs/%05zu.f32", row);
        REQUIRE(length >= 0 && (size_t)length < sizeof(relative));
        REQUIRE(make_path(path, sizeof(path), argv[2], relative));
        float input[INPUT_SIZE], original[INPUT_SIZE];
        REQUIRE(read_floats(path, input, INPUT_SIZE));
        for (size_t i = 0; i < INPUT_SIZE; ++i) {
            float expected = (float)pixels[row * INPUT_SIZE + i] / 255.0f;
            if (input[i] != expected) {
                fprintf(stderr, "%s[%zu]: input differs from saved pixel / 255.\n", relative, i);
                goto cleanup;
            }
        }
        memcpy(original, input, sizeof(input));
        for (size_t stage = 0; stage < COUNT(stages); ++stage) {
            float storage[MAX_WIDTH + 3];
            for (size_t i = 0; i < COUNT(storage); ++i) {
                storage[i] = NAN;
            }
            storage[0] = -101;
            storage[MAX_WIDTH + 2] = -102;
            size_t width = cinder_model_output_size(models[stage]);
            /* Alternate exact capacity and one spare slot. References are
             * comparison targets only, never inputs to any C stage. */
            REQUIRE(result_ok(cinder_model_run(models[stage], input, INPUT_SIZE,
                              storage + 1, width + row % 2, &error), &error, stages[stage].name));
            numpy_vector_t expected = {references[stage] + row * width, width};
            REQUIRE(compare_vector(&stats[stage], relative, stages[stage].name,
                                   storage + 1, width, expected, true));
            REQUIRE(memcmp(input, original, sizeof(input)) == 0);
            REQUIRE(storage[0] == -101 && storage[MAX_WIDTH + 2] == -102);
            for (size_t i = width + 1; i <= MAX_WIDTH + 1; ++i) {
                REQUIRE(isnan(storage[i]));
            }
            if (stage == 3) {
                double total = 0;
                for (size_t i = 0; i < width; ++i) {
                    REQUIRE(storage[i + 1] >= 0 && storage[i + 1] <= 1);
                    total += storage[i + 1];
                }
                REQUIRE(fabs(total - 1.0) <= 1e-6);
                size_t c_prediction = prediction(storage + 1, width);
                size_t python_prediction = prediction(expected.data, width);
                if (c_prediction != python_prediction) {
                    fprintf(stderr, "Test index %zu: C predicts %zu; Python predicts %zu.\n",
                            row, c_prediction, python_prediction);
                }
                agreements += c_prediction == python_prediction;
                c_correct += c_prediction == label;
                python_correct += python_prediction == label;
            }
        }
    }
    for (size_t stage = 0; stage < COUNT(stages); ++stage) {
        REQUIRE(stats[stage].comparisons == SAMPLES);
        REQUIRE(stats[stage].elements == SAMPLES * stages[stage].width);
        printf("%s: %zu vectors / %zu values, max absolute error %.9g, "
               "max fraction of tolerance %.9g.\n", stages[stage].name,
               stats[stage].comparisons, stats[stage].elements,
               stats[stage].largest_absolute_error, stats[stage].largest_tolerance_fraction);
    }
    printf("MNIST: 400 vectors / 27600 values passed (atol = rtol = 1e-5).\n");
    printf("Runtime prediction agreement (C vs saved Python): %zu/%d.\n", agreements, SAMPLES);
    printf("Classification on fixed test subset: C %zu/%d; Python %zu/%d.\n",
           c_correct, SAMPLES, python_correct, SAMPLES);
    REQUIRE(agreements == SAMPLES);
    result = EXIT_SUCCESS;

cleanup:
    free(model_bytes);
    for (size_t stage = 0; stage < COUNT(models); ++stage) {
        cinder_model_free(models[stage]);
    }
    for (size_t stage = 0; stage < COUNT(created); ++stage) {
        if (created[stage] && remove(prefix_paths[stage]) != 0) {
            fprintf(stderr, "Could not remove scratch prefix: %s\n", prefix_paths[stage]);
            result = EXIT_FAILURE;
        }
    }
    return result;
}
