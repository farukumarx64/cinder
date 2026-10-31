/* Exercise the real dispatcher with controlled API responses. The separate
 * process test covers real files, loader, input decoder and model execution. */
#include "cinder/model.h"
#include "../src/input.h"
#include "../src/benchmark.h"
#include <locale.h>
#include <math.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int condition, int line)
{
    if (!condition) { fprintf(stderr, "CLI audit failed at line %d\n", line); exit(1); }
}
#define CHECK(x) check(!!(x), __LINE__)
static FILE *captured_out, *captured_err;
static size_t live, allocations, fail_allocation, loads, inputs, runs, model_live;
static size_t input_size = 1, output_size = 4, write_calls, fail_write;
static int fail_flush, fail_stderr;
static cinder_status_t load_status, input_status, run_status;
static char api_message[CINDER_ERROR_MESSAGE_CAPACITY];
static float scores[8] = {0.5f, 0.5f, 0.1f, 0.5f};
static int model_token;
static void *blocks[2];
static size_t benchmarks;
static uint32_t requested_iterations;
static cinder_status_t benchmark_status;
static cinder_benchmark_phase_t benchmark_phase;
static uint32_t benchmark_iteration;
static int invalid_statistics;
static cinder_benchmark_result_t benchmark_result;
static cinder_status_t info_status, layer_status;
static size_t info_calls, layer_calls, info_count, fail_layer_index;
static int inspecting, invalid_layer;

static cinder_status_t test_get_info(const cinder_model_t *model,
    cinder_model_info_t *info, cinder_error_t *error);
static cinder_status_t test_get_layer_info(const cinder_model_t *model, size_t index,
    cinder_layer_info_t *info, cinder_error_t *error);

static void *test_malloc(size_t bytes)
{
    if (++allocations == fail_allocation) return NULL;
    CHECK(live < 2);
    void *pointer = malloc(bytes);
    CHECK(pointer != NULL);
    blocks[live++] = pointer;
    return pointer;
}
static void test_free(void *pointer)
{
    if (pointer == NULL) return;
    for (size_t i = 0; i < live; ++i) {
        if (blocks[i] == pointer) {
            blocks[i] = blocks[--live]; free(pointer); return;
        }
    }
    CHECK(0);
}
static cinder_status_t answer(cinder_error_t *error, cinder_status_t status)
{
    CHECK(error != NULL);
    error->code = status;
    memcpy(error->message, api_message, sizeof(error->message));
    return status;
}
static cinder_status_t test_load(const char *path, cinder_model_t **model, cinder_error_t *error)
{
    CHECK(path != NULL && model != NULL && *model == NULL && model_live == 0);
    ++loads;
    if (load_status == CINDER_OK) {
        *model = (cinder_model_t *)&model_token;
        model_live = 1;
    }
    return answer(error, load_status);
}
static cinder_status_t test_get_info(const cinder_model_t *model,
    cinder_model_info_t *info, cinder_error_t *error)
{
    CHECK(model == (cinder_model_t *)&model_token && model_live && inputs == 0 && runs == 0);
    ++info_calls;
    if (info_status == CINDER_OK) *info = (cinder_model_info_t){2, 2, info_count, 22};
    return answer(error, info_status);
}
static cinder_status_t test_get_layer_info(const cinder_model_t *model, size_t index,
    cinder_layer_info_t *info, cinder_error_t *error)
{
    CHECK(model == (cinder_model_t *)&model_token && model_live && index < 4);
    ++layer_calls;
    const cinder_layer_info_t layers[] = {
        {CINDER_LAYER_DENSE, 2, 4, 12}, {CINDER_LAYER_RELU, 4, 4, 0},
        {CINDER_LAYER_DENSE, 4, 2, 10}, {CINDER_LAYER_SOFTMAX, 2, 2, 0}
    };
    if (index == fail_layer_index && layer_status != CINDER_OK) return answer(error, layer_status);
    *info = layers[index];
    if (invalid_layer && index == fail_layer_index) info->type = CINDER_LAYER_INVALID;
    return answer(error, CINDER_OK);
}
static size_t test_input_size(const cinder_model_t *model)
{
    CHECK(model == (cinder_model_t *)&model_token && model_live); return input_size;
}
static size_t test_output_size(const cinder_model_t *model)
{
    CHECK(model == (cinder_model_t *)&model_token && model_live); return output_size;
}
static void test_model_free(cinder_model_t *model)
{
    if (model == NULL) return;
    CHECK(model == (cinder_model_t *)&model_token && model_live == 1);
    model_live = 0;
}
static cinder_status_t test_input(const char *path, size_t count, float **values, cinder_error_t *error)
{
    CHECK(path != NULL && count == input_size && model_live && runs == 0);
    ++inputs;
    if (input_status != CINDER_OK) return answer(error, input_status);
    *values = test_malloc(count * sizeof(float));
    CHECK(*values != NULL);
    (*values)[0] = 1.0f;
    return answer(error, CINDER_OK);
}
static cinder_status_t test_run(cinder_model_t *model, const float *input,
    size_t count, float *output, size_t capacity, cinder_error_t *error)
{
    CHECK(model == (cinder_model_t *)&model_token && model_live && inputs == 1);
    CHECK(count == input_size && capacity == output_size && input != output && input[0] == 1.0f);
    CHECK(live == 2 && output_size <= 8);
    ++runs;
    memcpy(output, scores, output_size * sizeof(float));
    return answer(error, run_status);
}
static cinder_status_t test_benchmark(const cinder_cli_session_t *session, uint32_t iterations,
    cinder_benchmark_result_t *result, cinder_benchmark_failure_t *failure)
{
    CHECK(model_live == 1 && live == 2 && loads == 1 && inputs == 1 && runs == 0);
    CHECK(session->model == (cinder_model_t *)&model_token && session->input != session->output);
    CHECK(session->input_size == input_size && session->output_size == output_size);
    ++benchmarks; requested_iterations = iterations;
    *failure = (cinder_benchmark_failure_t){.phase = benchmark_phase, .iteration = benchmark_iteration};
    answer(&failure->detail, benchmark_status);
    if (benchmark_status == CINDER_OK) {
        *failure = (cinder_benchmark_failure_t){0};
        *result = benchmark_result;
        result->iterations = iterations;
        if (invalid_statistics) result->total_ns = 0;
        else if (iterations != 3) {
            result->minimum_ns = result->maximum_ns = 1000;
            result->total_ns = (uint64_t)iterations * 1000;
        }
    }
    return benchmark_status;
}
static int fail_output(FILE *file)
{
    /* Only the value-copy metadata snapshot may remain during info output. */
    CHECK((live == 0 || (inspecting && live == 1 && file == captured_out)) && model_live == 0);
    if (file == captured_err) return fail_stderr;
    CHECK(file == captured_out);
    return ++write_calls == fail_write;
}
static int test_fprintf(FILE *file, const char *format, ...)
{
    if (fail_output(file)) return -1;
    va_list args;
    va_start(args, format);
    int result = vfprintf(file, format, args);
    va_end(args);
    return result;
}
static int test_fputs(const char *text, FILE *file)
{
    return fail_output(file) ? EOF : fputs(text, file);
}
static int test_fputc(int c, FILE *file)
{
    return fail_output(file) ? EOF : fputc(c, file);
}
static int test_flush(FILE *file)
{
    CHECK(file == captured_out && live == 0 && model_live == 0);
    return fail_flush ? EOF : fflush(file);
}
#undef stdout
#undef stderr
#define stdout captured_out
#define stderr captured_err
#define fprintf test_fprintf
#define fputs test_fputs
#define fputc test_fputc
#define fflush test_flush
#define malloc test_malloc
#define free test_free
#define cinder_model_load test_load
#define cinder_model_input_size test_input_size
#define cinder_model_output_size test_output_size
#define cinder_model_free test_model_free
#define cinder_model_run test_run
#define cinder_model_get_info test_get_info
#define cinder_model_get_layer_info test_get_layer_info
#define cinder_input_read test_input
#define cinder_benchmark_execute test_benchmark
#include "../src/cli_session.c"
#define main cli_main
#include "../src/cli.c"
#undef main
#undef stdout
#undef stderr
#undef fprintf
#undef fputs
#undef fputc
#undef fflush
#undef malloc
#undef free
#undef cinder_model_load
#undef cinder_model_input_size
#undef cinder_model_output_size
#undef cinder_model_free
#undef cinder_model_run
#undef cinder_model_get_info
#undef cinder_model_get_layer_info
#undef cinder_input_read
#undef cinder_benchmark_execute

static void reset(void)
{
    CHECK(live == 0 && model_live == 0);
    captured_out = tmpfile(); captured_err = tmpfile();
    CHECK(captured_out != NULL && captured_err != NULL);
    allocations = fail_allocation = loads = inputs = runs = write_calls = fail_write = 0;
    fail_flush = fail_stderr = 0;
    load_status = input_status = run_status = CINDER_OK;
    memset(api_message, 0, sizeof(api_message));
    input_size = 1; output_size = 4;
    scores[0] = scores[1] = scores[3] = 0.5f; scores[2] = 0.1f;
    benchmarks = requested_iterations = benchmark_iteration = invalid_statistics = 0;
    benchmark_status = CINDER_OK; benchmark_phase = CINDER_BENCHMARK_SETUP;
    benchmark_result = (cinder_benchmark_result_t){3, 1, 6000, 1000, 3000};
    info_status = layer_status = CINDER_OK;
    info_calls = layer_calls = invalid_layer = inspecting = 0;
    info_count = 4; fail_layer_index = 2;
}
static char out_text[4096], err_text[4096];
static void capture(FILE *file, char *text, size_t capacity)
{
    CHECK(fflush(file) == 0 && fseek(file, 0, SEEK_SET) == 0);
    size_t bytes = fread(text, 1, capacity - 1, file);
    CHECK(!ferror(file) && feof(file));
    text[bytes] = '\0';
    CHECK(fclose(file) == 0);
}
static void invoke(int argc, char **argv, int status)
{
    CHECK(cli_main(argc, argv) == status);
    CHECK(live == 0 && model_live == 0);
    capture(captured_out, out_text, sizeof(out_text));
    capture(captured_err, err_text, sizeof(err_text));
}
static void expect_failure(const char *detail)
{
    CHECK(out_text[0] == '\0' && strcmp(err_text, detail) == 0);
}

static void benchmark_cases(void)
{
    char *args[] = {"cinder", "bench", "dir/model.cnd", "input.f32", "--iterations", "3", "extra", NULL};
    reset(); invoke(6, args, 0);
    CHECK(strcmp(out_text, "Model: model.cnd\nIterations: 3\nWarm-up iterations: 10\n"
        "Clock: monotonic\nClock resolution (ns): 1\nMean latency (us): 2\n"
        "Minimum latency (us): 1\nMaximum latency (us): 3\nThroughput (inferences/s): 500000\n") == 0);
    CHECK(err_text[0] == '\0' && loads == 1 && inputs == 1 && benchmarks == 1 && runs == 0);
    CHECK(requested_iterations == 3 && allocations == 2);
    size_t successful_writes = write_calls;
    for (size_t i = 1; i <= successful_writes; ++i) {
        reset(); fail_write = i; invoke(6, args, 1);
        CHECK(strcmp(err_text, "cinder: output [IO]: Could not write benchmark output.\n") == 0);
    }
    reset(); fail_flush = 1; invoke(6, args, 1);
    CHECK(strcmp(err_text, "cinder: output [IO]: Could not write benchmark output.\n") == 0);
    reset(); fail_flush = fail_stderr = 1; invoke(6, args, 1); CHECK(err_text[0] == '\0');

    reset(); benchmark_result = (cinder_benchmark_result_t){3, 42, 4, 1, 2}; invoke(6, args, 0);
    CHECK(strstr(out_text, "Mean latency (us): 0.00133333333\nMinimum latency (us): 0.001\n"
        "Maximum latency (us): 0.002\nThroughput (inferences/s): 750000000\n") != NULL);
    reset(); benchmark_result = (cinder_benchmark_result_t){3, UINT64_MAX, UINT64_MAX, 1, UINT64_MAX-2};
    invoke(6, args, 0);
    CHECK(strstr(out_text, "Clock resolution (ns): 18446744073709551615\n") != NULL);
    CHECK(strstr(out_text, "Mean latency (us): 6.14891469e+15\n") != NULL);
    CHECK(strstr(out_text, "Throughput (inferences/s): 1.62630326e-10\n") != NULL);
    args[2] = "dir/a \"\\\n\t\xC3\xA9.cnd";
    reset(); invoke(6, args, 0);
    CHECK(strstr(out_text, "Model: a \\\"\\\\\\x0A\\x09\\xC3\\xA9.cnd\n") == out_text);
    args[2] = "dir/model.cnd";

    const char *valid[] = {"1", "0003", "1000000", "00000000000000000000000000000000000000000000000000001"};
    const uint32_t counts[] = {1, 3, 1000000, 1};
    for (size_t i = 0; i < sizeof(valid)/sizeof(valid[0]); ++i) {
        args[5] = (char *)valid[i]; reset(); invoke(6, args, 0);
        CHECK(requested_iterations == counts[i] && benchmarks == 1 && err_text[0] == '\0');
        char expected[64]; snprintf(expected, sizeof(expected), "Iterations: %" PRIu32 "\n", counts[i]);
        CHECK(strstr(out_text, expected) != NULL);
    }
    args[5] = "3";
    args[2] = "--iterations"; args[3] = "-";
    reset(); invoke(6, args, 0); CHECK(strstr(out_text, "Model: --iterations\n") == out_text);
    args[2] = "dir/model.cnd"; args[3] = "input.f32";

    const struct { int argc; const char *model, *input, *option, *count, *reason; } invalid[] = {
        {2,"","","","","bench requires two file paths followed by --iterations N."},
        {5,"x","y","--iterations","3","bench requires two file paths followed by --iterations N."},
        {7,"x","y","--iterations","3","bench requires two file paths followed by --iterations N."},
        {6,"","","wrong","x","model path must not be empty."},
        {6,"x","","wrong","x","input path must not be empty."},
        {6,"x","y","--iterations=3","","bench requires --iterations after the two file paths."},
        {6,"x","y","--help","3","bench requires --iterations after the two file paths."},
        {6,"--iterations","3","x","y","bench requires --iterations after the two file paths."},
        {6,"x","y","--","3","bench requires --iterations after the two file paths."},
        {6,"x","y","--warmup","3","bench requires --iterations after the two file paths."}
    };
    for (size_t i = 0; i < sizeof(invalid)/sizeof(invalid[0]); ++i) {
        args[2] = (char *)invalid[i].model; args[3] = (char *)invalid[i].input;
        args[4] = (char *)invalid[i].option; args[5] = (char *)invalid[i].count;
        reset(); invoke(invalid[i].argc, args, 2);
        char expected[512]; snprintf(expected, sizeof(expected), "cinder: usage: %s\nUsage:\n"
            "  cinder --version\n  cinder run <model.cnd> <input.bin>\n"
            "  cinder bench <model.cnd> <input.bin> --iterations N\n  cinder info <model.cnd>\n", invalid[i].reason);
        expect_failure(expected); CHECK(loads == 0 && inputs == 0 && benchmarks == 0 && allocations == 0);
    }
    args[2] = "dir/model.cnd"; args[3] = "input.f32"; args[4] = "--iterations";
    const char *numbers[] = {"", "+1", "-1", " 1", "1 ", "1\n", "0x10", "1e3", "1.0",
        "\xD9\xA1", "999999999999999999999999x", "0", "000", "1000001", "4294967296",
        "99999999999999999999999999999999999999999999999999999999999"};
    for (size_t i = 0; i < sizeof(numbers)/sizeof(numbers[0]); ++i) {
        args[5] = (char *)numbers[i]; reset(); invoke(6, args, 2);
        const char *reason = i < 11 ? "iterations must contain only ASCII decimal digits." :
                                     "iterations must be between 1 and 1000000.";
        char expected[512]; snprintf(expected, sizeof(expected), "cinder: usage: %s\nUsage:\n"
            "  cinder --version\n  cinder run <model.cnd> <input.bin>\n"
            "  cinder bench <model.cnd> <input.bin> --iterations N\n  cinder info <model.cnd>\n", reason);
        expect_failure(expected); CHECK(loads == 0 && benchmarks == 0 && allocations == 0);
    }
    args[5] = "3";
    reset(); load_status = input_status = CINDER_ERROR_IO; invoke(6, args, 1);
    expect_failure("cinder: model \"dir/model.cnd\" [IO]: Operation failed.\n");
    CHECK(inputs == 0 && benchmarks == 0);
    reset(); input_status = CINDER_ERROR_FORMAT; invoke(6, args, 1);
    expect_failure("cinder: input \"input.f32\" [FORMAT]: Operation failed.\n"); CHECK(benchmarks == 0);
    reset(); fail_allocation = 2; invoke(6, args, 1);
    expect_failure("cinder: inference [ALLOCATION]: Could not allocate output buffer.\n"); CHECK(benchmarks == 0);
    const cinder_benchmark_phase_t phases[] = {CINDER_BENCHMARK_SETUP, CINDER_BENCHMARK_WARMUP, CINDER_BENCHMARK_MEASURED};
    const char *stages[] = {"benchmark", "benchmark warm-up 10", "benchmark iteration 3"};
    for (size_t i = 0; i < 3; ++i) {
        reset(); benchmark_status = CINDER_ERROR_IO; benchmark_phase = phases[i];
        benchmark_iteration = i == 1 ? 10 : i == 2 ? 3 : 0;
        strcpy(api_message, "bad\n\xFF"); invoke(6, args, 1);
        char expected[256]; snprintf(expected, sizeof(expected), "cinder: %s [IO]: bad\\x0A\\xFF\n", stages[i]);
        expect_failure(expected); CHECK(benchmarks == 1);
    }
    reset(); benchmark_status = CINDER_ERROR_UNSUPPORTED; invoke(6, args, 1);
    expect_failure("cinder: benchmark [UNSUPPORTED]: Operation failed.\n");
    reset(); benchmark_status = CINDER_ERROR_IO; memset(api_message, 'X', sizeof(api_message));
    invoke(6, args, 1);
    CHECK(out_text[0] == '\0' && strlen(err_text) == strlen("cinder: benchmark [IO]: \n") + sizeof(api_message));
    reset(); invalid_statistics = 1; invoke(6, args, 1);
    expect_failure("cinder: benchmark [NONFINITE]: Benchmark statistics must be finite and positive.\n");
    reset(); invoke(6, args, 0); CHECK(err_text[0] == '\0'); /* Recovery. */
}
static void info_cases(void)
{
    char *args[] = {"cinder", "info", "dir/model.cnd", "extra", NULL};
    reset(); inspecting = 1; invoke(3, args, 0);
    CHECK(strcmp(out_text, "Model: model.cnd\nInput: 2\nOutput: 2\nLayers: 4\n"
        "  0: Dense 2 -> 4 (parameters: 12)\n  1: ReLU 4 -> 4 (parameters: 0)\n"
        "  2: Dense 4 -> 2 (parameters: 10)\n  3: Softmax 2 -> 2 (parameters: 0)\nParameters: 22\n") == 0);
    CHECK(err_text[0] == '\0' && loads == 1 && allocations == 1 && info_calls == 1 && layer_calls == 4);
    CHECK(inputs == 0 && runs == 0 && benchmarks == 0);
    size_t writes = write_calls;
    for (size_t i = 1; i <= writes; ++i) {
        reset(); inspecting = 1; fail_write = i; invoke(3, args, 1);
        CHECK(strcmp(err_text, "cinder: output [IO]: Could not write inspection output.\n") == 0);
    }
    reset(); inspecting = 1; fail_flush = 1; invoke(3, args, 1);
    CHECK(strcmp(err_text, "cinder: output [IO]: Could not write inspection output.\n") == 0);
    reset(); inspecting = 1; fail_write = fail_stderr = 1; invoke(3, args, 1);
    reset(); load_status = CINDER_ERROR_FORMAT; invoke(3, args, 1);
    expect_failure("cinder: model \"dir/model.cnd\" [FORMAT]: Operation failed.\n");
    CHECK(allocations == 0 && info_calls == 0);
    reset(); info_status = CINDER_ERROR_OVERFLOW; invoke(3, args, 1);
    expect_failure("cinder: inspection [OVERFLOW]: Operation failed.\n");
    CHECK(allocations == 0 && layer_calls == 0);
    reset(); info_count = SIZE_MAX; invoke(3, args, 1);
    expect_failure("cinder: inspection [OVERFLOW]: Model metadata byte count is not representable.\n");
    CHECK(allocations == 0);
    reset(); fail_allocation = 1; invoke(3, args, 1);
    expect_failure("cinder: inspection [ALLOCATION]: Could not allocate model metadata.\n");
    CHECK(layer_calls == 0);
    for (size_t i = 0; i < 4; ++i) {
        reset(); fail_layer_index = i; layer_status = CINDER_ERROR_UNSUPPORTED; invoke(3, args, 1);
        expect_failure("cinder: inspection [UNSUPPORTED]: Operation failed.\n");
        CHECK(layer_calls == i + 1);
    }
    reset(); invalid_layer = 1; invoke(3, args, 1);
    expect_failure("cinder: inspection [UNSUPPORTED]: Model contains an unsupported layer type.\n");
    for (int argc = 2; argc <= 4; ++argc) {
        reset(); args[2] = ""; invoke(argc, args, 2);
        const char *reason = argc == 3 ? "model path must not be empty." : "info requires exactly one file path.";
        char expected[512];
        snprintf(expected, sizeof(expected), "cinder: usage: %s\nUsage:\n  cinder --version\n"
            "  cinder run <model.cnd> <input.bin>\n  cinder bench <model.cnd> <input.bin> --iterations N\n"
            "  cinder info <model.cnd>\n", reason);
        expect_failure(expected);
        CHECK(loads == 0 && allocations == 0);
    }
    /* Recovery uses fresh resources, and option-looking paths remain literal. */
    args[2] = "--help"; reset(); inspecting = 1; invoke(3, args, 0);
    CHECK(strstr(out_text, "Model: --help\n") == out_text);
}

int main(void)
{
    benchmark_cases();
    info_cases();
    char *args[] = {"cinder", "run", "dir/model.cnd", "input.f32", "extra", NULL};
    reset(); invoke(4, args, 0);
    CHECK(strcmp(out_text, "Model: model.cnd\nPrediction: 0\nScores:\n0\t0.5\n1\t0.5\n3\t0.5\n") == 0);
    CHECK(err_text[0] == '\0' && loads == 1 && inputs == 1 && runs == 1 && allocations == 2);
    size_t successful_writes = write_calls;
    for (size_t i = 1; i <= successful_writes; ++i) {
        reset(); fail_write = i; invoke(4, args, 1);
        CHECK(strcmp(err_text, "cinder: output [IO]: Could not write prediction output.\n") == 0);
    }
    reset(); fail_flush = 1; invoke(4, args, 1);
    CHECK(strcmp(err_text, "cinder: output [IO]: Could not write prediction output.\n") == 0);
    reset(); fail_flush = fail_stderr = 1; invoke(4, args, 1);
    CHECK(err_text[0] == '\0');

    reset(); output_size = 1; scores[0] = -0.0f; invoke(4, args, 0);
    CHECK(strcmp(out_text, "Model: model.cnd\nPrediction: 0\nScores:\n0\t0\n") == 0);
    reset(); output_size = 2; scores[0] = -2.75f; scores[1] = -0.5f; invoke(4, args, 0);
    CHECK(strcmp(out_text, "Model: model.cnd\nPrediction: 1\nScores:\n1\t-0.5\n0\t-2.75\n") == 0);
    reset(); output_size = 3; scores[0] = -0.0f; scores[1] = 0.0f; scores[2] = -1.0f;
    invoke(4, args, 0);
    CHECK(strcmp(out_text, "Model: model.cnd\nPrediction: 0\nScores:\n0\t0\n1\t0\n2\t-1\n") == 0);
    reset(); output_size = 2; scores[0] = 1.0f; scores[1] = nextafterf(1.0f, 2.0f);
    invoke(4, args, 0);
    CHECK(strstr(out_text, "Prediction: 1\nScores:\n1\t1.00000012\n0\t1\n") != NULL);

    args[2] = "dir/a \"\\\n\t\xC3\xA9.cnd";
    reset(); invoke(4, args, 0);
    CHECK(strstr(out_text, "Model: a \\\"\\\\\\x0A\\x09\\xC3\\xA9.cnd\n") == out_text);
    reset(); load_status = CINDER_ERROR_IO; strcpy(api_message, "bad\n\xFF"); invoke(4, args, 1);
    expect_failure("cinder: model \"dir/a \\\"\\\\\\x0A\\x09\\xC3\\xA9.cnd\" [IO]: bad\\x0A\\xFF\n");
    CHECK(inputs == 0 && runs == 0);
    args[2] = "dir/model.cnd";

    const char *tags[] = {"INVALID_ARGUMENT", "OVERFLOW", "ALLOCATION", "OUT_OF_BOUNDS", "NONFINITE",
        "DIMENSION_MISMATCH", "BUFFER_TOO_SMALL", "IO", "FORMAT", "UNSUPPORTED", "LIMIT", "UNKNOWN"};
    for (size_t i = 0; i < sizeof(tags) / sizeof(tags[0]); ++i) {
        reset(); load_status = (cinder_status_t)(i + 1); invoke(4, args, 1);
        char expected[256];
        snprintf(expected, sizeof(expected), "cinder: model \"dir/model.cnd\" [%s]: Operation failed.\n", tags[i]);
        expect_failure(expected);
    }
    reset(); load_status = CINDER_ERROR_FORMAT; memset(api_message, 'X', sizeof(api_message));
    invoke(4, args, 1); CHECK(strlen(err_text) == strlen("cinder: model \"dir/model.cnd\" [FORMAT]: \n") + sizeof(api_message));
    reset(); input_status = CINDER_ERROR_NONFINITE; strcpy(api_message, "Input failure."); invoke(4, args, 1);
    expect_failure("cinder: input \"input.f32\" [NONFINITE]: Input failure.\n"); CHECK(runs == 0);
    reset(); run_status = CINDER_ERROR_NONFINITE; strcpy(api_message, "Arithmetic overflow."); invoke(4, args, 1);
    expect_failure("cinder: inference [NONFINITE]: Arithmetic overflow.\n"); CHECK(runs == 1);
    reset(); fail_allocation = 2; invoke(4, args, 1);
    expect_failure("cinder: inference [ALLOCATION]: Could not allocate output buffer.\n"); CHECK(runs == 0);
    reset(); output_size = SIZE_MAX / sizeof(float) + 1; invoke(4, args, 1);
    expect_failure("cinder: inference [OVERFLOW]: Output byte count is not representable.\n"); CHECK(allocations == 1);
    for (int dimension = 0; dimension < 2; ++dimension) {
        reset(); if (dimension) output_size = 0; else input_size = 0; invoke(4, args, 1);
        expect_failure("cinder: inference [INVALID_ARGUMENT]: Loaded model dimensions must be positive.\n");
        CHECK(inputs == 0 && allocations == 0);
    }
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < 3; ++i) {
        reset(); scores[3] = bad[i]; invoke(4, args, 1);
        expect_failure("cinder: inference [NONFINITE]: Output value at index 3 is not finite.\n");
    }

    const struct { int argc; const char *command, *model, *input, *reason; } usages[] = {
        {1, "", "x", "y", "expected --version, run, bench, or info."},
        {2, "--help", "x", "y", "expected --version, run, bench, or info."},
        {4, "Run", "x", "y", "expected --version, run, bench, or info."},
        {3, "--version", "x", "y", "expected --version, run, bench, or info."},
        {2, "run", "x", "y", "run requires exactly two file paths."},
        {3, "run", "x", "y", "run requires exactly two file paths."},
        {5, "run", "x", "y", "run requires exactly two file paths."},
        {4, "run", "", "", "model path must not be empty."},
        {4, "run", "x", "", "input path must not be empty."}
    };
    for (size_t i = 0; i < sizeof(usages) / sizeof(usages[0]); ++i) {
        args[1] = (char *)usages[i].command; args[2] = (char *)usages[i].model; args[3] = (char *)usages[i].input;
        reset(); invoke(usages[i].argc, args, 2);
        char expected[512];
        snprintf(expected, sizeof(expected), "cinder: usage: %s\nUsage:\n  cinder --version\n  cinder run <model.cnd> <input.bin>\n  cinder bench <model.cnd> <input.bin> --iterations N\n  cinder info <model.cnd>\n", usages[i].reason);
        expect_failure(expected); CHECK(loads == 0 && inputs == 0 && allocations == 0);
    }
    args[1] = "--version";
    reset(); invoke(2, args, 0);
    CHECK(strcmp(out_text, "Cinder " CINDER_VERSION "\n") == 0 && err_text[0] == '\0');
    reset(); fail_write = 1; invoke(2, args, 1);
    reset(); fail_flush = 1; invoke(2, args, 1);
    puts("CLI audit: formatting, ranking, diagnostics, error precedence, output faults and cleanup passed.");
    return 0;
}
