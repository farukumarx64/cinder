#include "benchmark.h"

#include <inttypes.h>
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

static int output_failure(const char *kind)
{
    fprintf(stderr, "cinder: output [IO]: Could not write %s output.\n", kind);
    return 1;
}

static int usage(const char *reason)
{
    fprintf(stderr, "cinder: usage: %s\nUsage:\n"
                    "  cinder --version\n"
                    "  cinder run <model.cnd> <input.bin>\n"
                    "  cinder bench <model.cnd> <input.bin> --iterations N\n"
                    "  cinder info <model.cnd>\n", reason);
    return 2;
}

typedef struct { size_t index; float value; } score_t;

static const char *layer_name(cinder_layer_type_t type)
{
    switch (type) {
        case CINDER_LAYER_DENSE: return "Dense";
        case CINDER_LAYER_RELU: return "ReLU";
        case CINDER_LAYER_SOFTMAX: return "Softmax";
        default: return NULL;
    }
}

static int info(const char *path)
{
    cinder_model_t *model = NULL;
    cinder_error_t error = {0};
    cinder_status_t status = cinder_model_load(path, &model, &error);
    if (status != CINDER_OK) return diagnostic("model", path, status, &error);

    cinder_model_info_t summary = {0};
    cinder_layer_info_t *layers = NULL;
    status = cinder_model_get_info(model, &summary, &error);
    if (status != CINDER_OK) goto cleanup;
    if (summary.layer_count > SIZE_MAX / sizeof(*layers)) {
        status = CINDER_ERROR_OVERFLOW;
        error = (cinder_error_t){CINDER_ERROR_OVERFLOW, "Model metadata byte count is not representable."};
        goto cleanup;
    }
    layers = malloc(summary.layer_count * sizeof(*layers));
    if (layers == NULL) {
        status = CINDER_ERROR_ALLOCATION;
        error = (cinder_error_t){CINDER_ERROR_ALLOCATION, "Could not allocate model metadata."};
        goto cleanup;
    }
    /* Complete inspection before printing, so load/metadata failures produce
     * no partial report. The snapshot lets us release model storage first. */
    for (size_t i = 0; i < summary.layer_count; ++i) {
        status = cinder_model_get_layer_info(model, i, &layers[i], &error);
        if (status != CINDER_OK) goto cleanup;
        if (layer_name(layers[i].type) == NULL) {
            status = CINDER_ERROR_UNSUPPORTED;
            error = (cinder_error_t){CINDER_ERROR_UNSUPPORTED, "Model contains an unsupported layer type."};
            goto cleanup;
        }
    }
cleanup:
    cinder_model_free(model);
    if (status != CINDER_OK) {
        free(layers);
        return diagnostic("inspection", NULL, status, &error);
    }
    const char *basename = strrchr(path, '/');
    basename = basename != NULL ? basename + 1 : path;
    bool ok = fputs("Model: ", stdout) != EOF &&
        print_escaped(stdout, basename, SIZE_MAX, true) &&
        fprintf(stdout, "\nInput: %zu\nOutput: %zu\nLayers: %zu\n",
                summary.input_size, summary.output_size, summary.layer_count) >= 0;
    for (size_t i = 0; ok && i < summary.layer_count; ++i) {
        ok = fprintf(stdout, "  %zu: %s %zu -> %zu (parameters: %zu)\n", i,
                     layer_name(layers[i].type), layers[i].input_size,
                     layers[i].output_size, layers[i].parameter_count) >= 0;
    }
    free(layers);
    if (!ok || fprintf(stdout, "Parameters: %zu\n", summary.parameter_count) < 0) {
        return output_failure("inspection");
    }
    return fflush(stdout) == 0 ? 0 : output_failure("inspection");
}

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
        return output_failure("prediction");
    }
    for (size_t i = 0; i < used; ++i) {
        double value = best[i].value == 0.0f ? 0.0 : (double)best[i].value;
        if (fprintf(stdout, "%zu\t%.9g\n", best[i].index, value) < 0) return output_failure("prediction");
    }
    return fflush(stdout) == 0 ? 0 : output_failure("prediction");
}

static int bench(const char *model_path, const char *input_path, uint32_t iterations)
{
    cinder_cli_session_t session = {0};
    cinder_cli_failure_t setup_failure;
    cinder_status_t status = cinder_cli_prepare(model_path, input_path, &session, &setup_failure);
    if (status != CINDER_OK) {
        cinder_cli_release(&session);
        return diagnostic(setup_failure.stage, setup_failure.path, status, &setup_failure.detail);
    }
    /* Zero is an invalid measurement, not a fallback report. execute publishes
     * only on success; statistics also rejects an unpopulated result. */
    cinder_benchmark_result_t result = {0};
    cinder_benchmark_statistics_t statistics;
    cinder_benchmark_failure_t failure;
    status = cinder_benchmark_execute(&session, iterations, &result, &failure);
    if (status == CINDER_OK) {
        status = cinder_benchmark_statistics(&result, &statistics, &failure.detail);
    }
    cinder_cli_release(&session);
    if (status != CINDER_OK) {
        char stage[64];
        if (failure.phase == CINDER_BENCHMARK_WARMUP) {
            snprintf(stage, sizeof(stage), "benchmark warm-up %" PRIu32, failure.iteration);
        } else if (failure.phase == CINDER_BENCHMARK_MEASURED) {
            snprintf(stage, sizeof(stage), "benchmark iteration %" PRIu32, failure.iteration);
        } else {
            snprintf(stage, sizeof(stage), "benchmark");
        }
        return diagnostic(stage, NULL, status, &failure.detail);
    }
    const char *basename = strrchr(model_path, '/');
    basename = basename != NULL ? basename + 1 : model_path;
    if (fputs("Model: ", stdout) == EOF ||
        !print_escaped(stdout, basename, SIZE_MAX, true) ||
        fprintf(stdout, "\nIterations: %" PRIu32 "\nWarm-up iterations: %" PRIu32
            "\nClock: monotonic\nClock resolution (ns): %" PRIu64
            "\nMean latency (us): %.9g\nMinimum latency (us): %.9g"
            "\nMaximum latency (us): %.9g\nThroughput (inferences/s): %.9g\n",
            result.iterations, CINDER_BENCHMARK_WARMUPS, result.resolution_ns,
            statistics.mean_us, statistics.minimum_us, statistics.maximum_us,
            statistics.inferences_per_second) < 0) {
        return output_failure("benchmark");
    }
    return fflush(stdout) == 0 ? 0 : output_failure("benchmark");
}

static int bench_arguments(int argc, char *argv[])
{
    if (argc != 6) return usage("bench requires two file paths followed by --iterations N.");
    if (argv[2][0] == '\0') return usage("model path must not be empty.");
    if (argv[3][0] == '\0') return usage("input path must not be empty.");
    if (strcmp(argv[4], "--iterations") != 0) {
        return usage("bench requires --iterations after the two file paths.");
    }
    /* Check all bytes first so malformed text wins over an excessive value. */
    if (argv[5][0] == '\0') return usage("iterations must contain only ASCII decimal digits.");
    for (const char *p = argv[5]; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return usage("iterations must contain only ASCII decimal digits.");
    }
    uint32_t count = 0;
    for (const char *p = argv[5]; *p != '\0'; ++p) {
        uint32_t digit = (uint32_t)(*p - '0');
        if (count > (CINDER_BENCHMARK_MAX_ITERATIONS - digit) / 10) {
            return usage("iterations must be between 1 and 1000000.");
        }
        count = count * 10 + digit;
    }
    if (count == 0) return usage("iterations must be between 1 and 1000000.");
    return bench(argv[2], argv[3], count);
}

int main(int argc, char *argv[])
{
    /* C startup already uses this locale; make the formatting requirement explicit. */
    setlocale(LC_NUMERIC, "C");
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        if (fprintf(stdout, "Cinder %s\n", CINDER_VERSION) < 0) return output_failure("prediction");
        return fflush(stdout) == 0 ? 0 : output_failure("prediction");
    }
    if (argc >= 2 && strcmp(argv[1], "bench") == 0) return bench_arguments(argc, argv);
    if (argc >= 2 && strcmp(argv[1], "info") == 0) {
        if (argc != 3) return usage("info requires exactly one file path.");
        if (argv[2][0] == '\0') return usage("model path must not be empty.");
        return info(argv[2]);
    }
    if (argc < 2 || strcmp(argv[1], "run") != 0) return usage("expected --version, run, bench, or info.");
    if (argc != 4) return usage("run requires exactly two file paths.");
    if (argv[2][0] == '\0') return usage("model path must not be empty.");
    if (argv[3][0] == '\0') return usage("input path must not be empty.");
    return run(argv[2], argv[3]);
}
