#include "cli_session.h"

#include <locale.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
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
    cinder_cli_session_t session = {0};
    cinder_cli_failure_t failure;
    cinder_status_t status = cinder_cli_prepare(model_path, input_path, &session, &failure);
    score_t best[3];
    size_t used = 0;
    if (status != CINDER_OK) goto cleanup;

    status = cinder_model_run(session.model, session.input, session.input_size,
                              session.output, session.output_size, &failure.detail);
    if (status != CINDER_OK) goto cleanup;
    for (size_t i = 0; i < session.output_size; ++i) {
        if (!isfinite(session.output[i])) {
            status = CINDER_ERROR_NONFINITE;
            failure.detail.code = status;
            snprintf(failure.detail.message, sizeof(failure.detail.message),
                     "Output value at index %zu is not finite.", i);
            goto cleanup;
        }
        /* Ascending index traversal and strict comparison preserve tie order.
         * Only three entries are retained; the caller's vector is not sorted. */
        size_t position = 0;
        while (position < used && session.output[i] <= best[position].value) ++position;
        if (position < 3) {
            if (used < 3) ++used;
            for (size_t j = used - 1; j > position; --j) best[j] = best[j - 1];
            best[position] = (score_t){i, session.output[i]};
        }
    }
cleanup:
    cinder_cli_release(&session);
    if (status != CINDER_OK) return diagnostic(failure.stage, failure.path, status, &failure.detail);

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
