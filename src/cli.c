#include "cinder/model.h"
#include "input.h"

#include <locale.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *status_name(cinder_status_t status)
{
    switch (status) {
#define STATUS(name) case CINDER_ERROR_##name: return #name
        STATUS(INVALID_ARGUMENT); STATUS(OVERFLOW); STATUS(ALLOCATION);
        STATUS(OUT_OF_BOUNDS); STATUS(NONFINITE); STATUS(DIMENSION_MISMATCH);
        STATUS(BUFFER_TOO_SMALL); STATUS(IO); STATUS(FORMAT);
        STATUS(UNSUPPORTED); STATUS(LIMIT);
#undef STATUS
        default: return "UNKNOWN";
    }
}

/* Bound API messages even if a faulty implementation omits the terminator.
 * Paths additionally escape quotes/backslashes; all non-ASCII bytes use hex. */
static bool print_escaped(FILE *stream, const char *text, size_t limit, bool path)
{
    for (size_t i = 0; i < limit && text[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)text[i];
        if (c < 0x20 || c > 0x7e) {
            if (fprintf(stream, "\\x%02X", (unsigned)c) < 0) return false;
        } else {
            if (path && (c == '\\' || c == '"') && fputc('\\', stream) == EOF) return false;
            if (fputc(c, stream) == EOF) return false;
        }
    }
    return true;
}

static int diagnostic(const char *stage, const char *path,
                      cinder_status_t status, const cinder_error_t *error)
{
    fprintf(stderr, "cinder: %s", stage);
    if (path != NULL) {
        fputs(" \"", stderr);
        print_escaped(stderr, path, SIZE_MAX, true);
        fputc('"', stderr);
    }
    fprintf(stderr, " [%s]: ", status_name(status));
    if (error->message[0] == '\0') {
        fputs("Operation failed.", stderr);
    } else {
        print_escaped(stderr, error->message, sizeof(error->message), false);
    }
    fputc('\n', stderr);
    return 1;
}

static int output_failure(void)
{
    fputs("cinder: output [IO]: Could not write prediction output.\n", stderr);
    return 1;
}

static int usage(const char *reason)
{
    fprintf(stderr, "cinder: usage: %s\nUsage:\n"
                    "  cinder --version\n"
                    "  cinder run <model.cnd> <input.bin>\n", reason);
    return 2;
}

typedef struct { size_t index; float value; } score_t;

static int run(const char *model_path, const char *input_path)
{
    cinder_model_t *model = NULL;
    float *input = NULL, *output = NULL;
    cinder_error_t error = {0};
    const char *stage = "model", *path = model_path;
    cinder_status_t status = cinder_model_load(model_path, &model, &error);
    score_t best[3];
    size_t used = 0;
    if (status != CINDER_OK) goto cleanup;

    stage = "inference";
    path = NULL;
    size_t input_size = cinder_model_input_size(model);
    size_t output_size = cinder_model_output_size(model);
#define RUN_FAIL(code, ...) do { \
    status = code; \
    snprintf(error.message, sizeof(error.message), __VA_ARGS__); \
    goto cleanup; \
} while (0)
    if (input_size == 0 || output_size == 0) {
        RUN_FAIL(CINDER_ERROR_INVALID_ARGUMENT, "Loaded model dimensions must be positive.");
    }
    stage = "input";
    path = input_path;
    status = cinder_input_read(input_path, input_size, &input, &error);
    if (status != CINDER_OK) goto cleanup;

    stage = "inference";
    path = NULL;
    if (output_size > SIZE_MAX / sizeof(float)) {
        RUN_FAIL(CINDER_ERROR_OVERFLOW, "Output byte count is not representable.");
    }
    output = malloc(output_size * sizeof(float));
    if (output == NULL) {
        RUN_FAIL(CINDER_ERROR_ALLOCATION, "Could not allocate output buffer.");
    }
    status = cinder_model_run(model, input, input_size, output, output_size, &error);
    if (status != CINDER_OK) goto cleanup;
    for (size_t i = 0; i < output_size; ++i) {
        if (!isfinite(output[i])) {
            RUN_FAIL(CINDER_ERROR_NONFINITE, "Output value at index %zu is not finite.", i);
        }
        /* Ascending index traversal and strict comparison preserve tie order.
         * Only three entries are retained; the caller's vector is not sorted. */
        size_t position = 0;
        while (position < used && output[i] <= best[position].value) ++position;
        if (position < 3) {
            if (used < 3) ++used;
            for (size_t j = used - 1; j > position; --j) best[j] = best[j - 1];
            best[position] = (score_t){i, output[i]};
        }
    }
cleanup:
    free(output);
    free(input);
    cinder_model_free(model);
    if (status != CINDER_OK) return diagnostic(stage, path, status, &error);
#undef RUN_FAIL

    const char *basename = strrchr(model_path, '/');
    basename = basename != NULL ? basename + 1 : model_path;
    if (fputs("Model: ", stdout) == EOF ||
        !print_escaped(stdout, basename, SIZE_MAX, true) ||
        fprintf(stdout, "\nPrediction: %zu\nScores:\n", best[0].index) < 0) {
        return output_failure();
    }
    for (size_t i = 0; i < used; ++i) {
        double value = best[i].value == 0.0f ? 0.0 : (double)best[i].value;
        if (fprintf(stdout, "%zu\t%.9g\n", best[i].index, value) < 0) return output_failure();
    }
    return fflush(stdout) == 0 ? 0 : output_failure();
}

int main(int argc, char *argv[])
{
    /* C startup already uses this locale; make the formatting requirement explicit. */
    setlocale(LC_NUMERIC, "C");
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        if (fprintf(stdout, "Cinder %s\n", CINDER_VERSION) < 0) return output_failure();
        return fflush(stdout) == 0 ? 0 : output_failure();
    }
    if (argc < 2 || strcmp(argv[1], "run") != 0) return usage("expected --version or run.");
    if (argc != 4) return usage("run requires exactly two file paths.");
    if (argv[2][0] == '\0') return usage("model path must not be empty.");
    if (argv[3][0] == '\0') return usage("input path must not be empty.");
    return run(argv[2], argv[3]);
}
