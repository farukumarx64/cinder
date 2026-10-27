/* Exercise the real dispatcher with controlled API responses. The separate
 * process test covers real files, loader, input decoder and model execution. */
#include "cinder/model.h"
#include "../src/input.h"
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
static int fail_output(FILE *file)
{
    CHECK(live == 0 && model_live == 0); /* Cleanup must precede all output. */
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
#define cinder_input_read test_input
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
#undef cinder_input_read

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
int main(void)
{
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
        {1, "", "x", "y", "expected --version or run."},
        {2, "--help", "x", "y", "expected --version or run."},
        {4, "Run", "x", "y", "expected --version or run."},
        {3, "--version", "x", "y", "expected --version or run."},
        {2, "run", "x", "y", "run requires exactly two file paths."},
        {3, "run", "x", "y", "run requires exactly two file paths."},
        {5, "run", "x", "y", "run requires exactly two file paths."},
        {4, "run", "", "", "model path must not be empty."},
        {4, "run", "x", "", "input path must not be empty."}
    };
    for (size_t i = 0; i < sizeof(usages) / sizeof(usages[0]); ++i) {
        args[1] = (char *)usages[i].command; args[2] = (char *)usages[i].model; args[3] = (char *)usages[i].input;
        reset(); invoke(usages[i].argc, args, 2);
        char expected[256];
        snprintf(expected, sizeof(expected), "cinder: usage: %s\nUsage:\n  cinder --version\n  cinder run <model.cnd> <input.bin>\n", usages[i].reason);
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
