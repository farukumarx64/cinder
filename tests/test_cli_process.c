#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
static char output[4096], error[4096];
static size_t commands, benchmark_commands, native_benchmarks, unresolved_benchmarks;
enum { NATIVE_BENCHMARK_RESULT = -1 };

static void read_text(const char *path, char *text, size_t capacity)
{
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    size_t count = fread(text, 1, capacity - 1, file);
    CHECK(!ferror(file) && feof(file));
    text[count] = '\0';
    CHECK(fclose(file) == 0 && remove(path) == 0);
}
static int execute_command(const char *executable, const char *model, const char *input,
                            const char *iterations, int expected)
{
    pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        int out = open("stdout.txt", O_WRONLY | O_CREAT | O_TRUNC, 0600);
        int err = open("stderr.txt", O_WRONLY | O_CREAT | O_TRUNC, 0600);
        if (out < 0 || err < 0 || dup2(out, STDOUT_FILENO) < 0 || dup2(err, STDERR_FILENO) < 0) _exit(126);
        close(out); close(err);
        if (iterations != NULL) {
            execl(executable, executable, "bench", model, input, "--iterations", iterations, (char *)NULL);
        } else if (input == NULL) {
            execl(executable, executable, "info", model, (char *)NULL);
        } else {
            execl(executable, executable, "run", model, input, (char *)NULL);
        }
        _exit(127);
    }
    int status;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    CHECK(waited == child && WIFEXITED(status));
    read_text("stdout.txt", output, sizeof(output));
    read_text("stderr.txt", error, sizeof(error));
    int actual = WEXITSTATUS(status);
    if (expected != NATIVE_BENCHMARK_RESULT && actual != expected)
        fprintf(stderr, "Unexpected child output: %s%s", output, error);
    CHECK(expected == NATIVE_BENCHMARK_RESULT ? iterations != NULL && (actual == 0 || actual == 1) : actual == expected);
    if (actual == 0) CHECK(error[0] == '\0'); else CHECK(output[0] == '\0');
    ++commands;
    if (iterations != NULL) ++benchmark_commands;
    return actual;
}
static void execute(const char *executable, const char *model, const char *input, int expected)
{
    execute_command(executable, model, input, NULL, expected);
}
static void compare_benchmark(const char *name, unsigned expected_count)
{
    char header[512];
    int length = snprintf(header, sizeof(header), "Model: %s\nIterations: %u\n"
        "Warm-up iterations: 10\nClock: monotonic\nClock resolution (ns): ", name, expected_count);
    CHECK(length > 0 && (size_t)length < sizeof(header) && strncmp(output, header, (size_t)length) == 0);
    const char *cursor = output + length;
    uint64_t resolution;
    double mean, minimum, maximum, throughput;
    int consumed = 0;
    CHECK(sscanf(cursor, "%" SCNu64 "\nMean latency (us): %lf\nMinimum latency (us): %lf\n"
        "Maximum latency (us): %lf\nThroughput (inferences/s): %lf%n",
        &resolution, &mean, &minimum, &maximum, &throughput, &consumed) == 5);
    CHECK(consumed > 0 && cursor[consumed] == '\n' && cursor[consumed+1] == '\0');
    CHECK(resolution > 0 && isfinite(mean) && isfinite(minimum) && isfinite(maximum) && isfinite(throughput));
    CHECK(minimum > 0 && mean >= minimum && maximum >= mean && throughput > 0);
    /* Display rounding is allowed; no latency/speed threshold is imposed. */
    CHECK(fabs(mean * throughput - 1000000.0) <= .02);
    if (expected_count == 1) CHECK(mean == minimum && mean == maximum);
    char formatted[512];
    snprintf(formatted, sizeof(formatted), "%" PRIu64 "\nMean latency (us): %.9g\n"
        "Minimum latency (us): %.9g\nMaximum latency (us): %.9g\nThroughput (inferences/s): %.9g\n",
        resolution, mean, minimum, maximum, throughput);
    CHECK(strcmp(cursor, formatted) == 0);
}
static int zero_interval_report(const char *text, unsigned count)
{
    /* Native smoke cases request at most 17 calls. Compare the exact allowed
     * messages; never accept a generic IO failure or an out-of-range index. */
    CHECK(count >= 1 && count <= 17);
    for (unsigned i = 1; i <= count; ++i) {
        char expected[256];
        snprintf(expected, sizeof(expected), "cinder: benchmark iteration %u [IO]: "
            "Measured interval is zero; clock resolution is insufficient.\n", i);
        if (strcmp(text, expected) == 0) return 1;
    }
    return 0;
}
static void benchmark_command(const char *executable, const char *model, const char *input,
                              const char *iterations, const char *name, unsigned count)
{
    ++native_benchmarks;
    int status = execute_command(executable, model, input, iterations, NATIVE_BENCHMARK_RESULT);
    if (status == 0) compare_benchmark(name, count);
    else {
        if (!zero_interval_report(error, count)) fprintf(stderr, "Unexpected benchmark failure: %s", error);
        CHECK(zero_interval_report(error, count));
        ++unresolved_benchmarks; /* Retain the outcome; do not retry until success. */
    }
}
static void check_zero_interval_reports(void)
{
    /* Exercise the native-failure oracle even when this host resolves every
     * smoke interval. Malformed/other failures must not be mistaken for it. */
    const char *suffix = " [IO]: Measured interval is zero; clock resolution is insufficient.\n";
    const char *indices[] = {"0", "1", "3", "4", "01", "+1"};
    for (size_t i = 0; i < sizeof(indices) / sizeof(indices[0]); ++i) {
        char text[256]; snprintf(text, sizeof(text), "cinder: benchmark iteration %s%s", indices[i], suffix);
        CHECK(zero_interval_report(text, 3) == (i == 1 || i == 2));
    }
    CHECK(!zero_interval_report("cinder: benchmark iteration 1 [IO]: Monotonic clock moved backwards.\n", 3));
    CHECK(!zero_interval_report("cinder: benchmark [UNSUPPORTED]: No monotonic clock is available.\n", 3));
    CHECK(!zero_interval_report("cinder: benchmark iteration 1 [IO]: Measured interval is zero; clock resolution is insufficient.\nextra", 3));
}
static void word(FILE *file, uint32_t value)
{
    for (size_t i = 0; i < 4; ++i) CHECK(fputc((int)((value >> (8 * i)) & 255), file) != EOF);
}
static void value(FILE *file, float number)
{
    uint32_t bits;
    memcpy(&bits, &number, 4); word(file, bits);
}
static void write_input(const char *path, float number)
{
    FILE *file = fopen(path, "wb"); CHECK(file != NULL);
    value(file, number); CHECK(fclose(file) == 0);
}
static void write_model(const char *path, const float *bias, size_t count)
{
    FILE *file = fopen(path, "wb"); CHECK(file != NULL);
    CHECK(fwrite("CNDR", 1, 4, file) == 4);
    word(file, 1); word(file, 1); word(file, 1);
    word(file, 1); word(file, 1); word(file, (uint32_t)count); word(file, (uint32_t)(count * 8));
    for (size_t i = 0; i < count; ++i) value(file, 0.0f);
    for (size_t i = 0; i < count; ++i) value(file, bias[i]);
    CHECK(fclose(file) == 0);
}
static void mutate_word(const char *path, long offset, uint32_t bits)
{
    FILE *file = fopen(path, "r+b"); CHECK(file != NULL && fseek(file, offset, SEEK_SET) == 0);
    word(file, bits); CHECK(fclose(file) == 0);
}
static float read_float(FILE *file)
{
    unsigned char bytes[4]; CHECK(fread(bytes, 1, 4, file) == 4);
    uint32_t bits = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
                    (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
    float result; memcpy(&result, &bits, 4); CHECK(isfinite(result)); return result;
}
static double compare_output(const char *name, const float *reference, size_t count)
{
    size_t best[3] = {0};
    size_t k = count < 3 ? count : 3;
    for (size_t rank = 0; rank < k; ++rank) {
        int found = 0;
        for (size_t i = 0; i < count; ++i) {
            int used = 0;
            for (size_t j = 0; j < rank; ++j) used |= best[j] == i;
            if (!used && (!found || reference[i] > reference[best[rank]])) {
                best[rank] = i; found = 1;
            }
        }
    }
    char header[256];
    int length = snprintf(header, sizeof(header), "Model: %s\nPrediction: %zu\nScores:\n", name, best[0]);
    CHECK(length > 0 && (size_t)length < sizeof(header) && strncmp(output, header, (size_t)length) == 0);
    const char *cursor = output + length;
    double maximum = 0;
    for (size_t rank = 0; rank < k; ++rank) {
        size_t index;
        float score;
        int consumed = 0;
        CHECK(sscanf(cursor, "%zu\t%f%n", &index, &score, &consumed) == 2);
        CHECK(consumed > 0 && cursor[consumed] == '\n' && index == best[rank] && isfinite(score));
        char line[128];
        int bytes = snprintf(line, sizeof(line), "%zu\t%.9g\n", index, score == 0.0f ? 0.0 : (double)score);
        CHECK(bytes == consumed + 1 && strncmp(cursor, line, (size_t)bytes) == 0);
        double difference = fabs((double)score - reference[index]);
        CHECK(difference <= 1e-5 + 1e-5 * fabs((double)reference[index]));
        if (difference > maximum) maximum = difference;
        cursor += bytes;
    }
    CHECK(*cursor == '\0');
    return maximum;
}

int main(int argc, char **argv)
{
    CHECK(argc == 4); /* executable, source directory, private scratch directory */
    check_zero_interval_reports();
    CHECK(chdir(argv[3]) == 0);
    const char *executable = argv[1], *source = argv[2];
    /* An inherited environment must not make score formatting locale-dependent. */
    CHECK(setenv("LC_ALL", "de_DE.UTF-8", 1) == 0);
    write_input("-", 0.0f);
    float bias[] = {0.5f, 0.5f, 0.1f, 0.5f};
    write_model("-model with spaces", bias, 4);
    execute(executable, "-model with spaces", "-", 0);
    compare_output("-model with spaces", bias, 4);
    execute(executable, "-model with spaces", NULL, 0);
    CHECK(strcmp(output, "Model: -model with spaces\nInput: 1\nOutput: 4\nLayers: 1\n"
        "  0: Dense 1 -> 4 (parameters: 8)\nParameters: 8\n") == 0);
    const char *odd = "a \"\\\n\t\xC3\xA9";
    write_model(odd, bias, 4);
    execute(executable, odd, "-", 0);
    compare_output("a \\\"\\\\\\x0A\\x09\\xC3\\xA9", bias, 4);
    benchmark_command(executable, odd, "-", "0003", "a \\\"\\\\\\x0A\\x09\\xC3\\xA9", 3);
    execute(executable, odd, NULL, 0);
    CHECK(strstr(output, "Model: a \\\"\\\\\\x0A\\x09\\xC3\\xA9\n") == output);
    CHECK(symlink(odd, "alias.cnd") == 0);
    execute(executable, "alias.cnd", "-", 0);
    compare_output("alias.cnd", bias, 4);
    CHECK(unlink("alias.cnd") == 0 && remove(odd) == 0);
    execute(executable, odd, NULL, 1);
    CHECK(strstr(error, "[IO]") != NULL);
    execute(executable, odd, "missing", 1);
    CHECK(strstr(error, "cinder: model \"a \\\"\\\\\\x0A\\x09\\xC3\\xA9\" [IO]: ") == error);
    execute_command(executable, odd, "missing", "3", 1);
    CHECK(strstr(error, "cinder: model \"a \\\"\\\\\\x0A\\x09\\xC3\\xA9\" [IO]: ") == error);
    execute_command(executable, "missing", "missing", "999999999999x", 2);
    CHECK(strstr(error, "cinder: usage: iterations must contain only ASCII decimal digits.\n") == error);
    for (size_t count = 1; count <= 3; ++count) {
        bias[0] = -2.75f; bias[1] = -0.5f; bias[2] = -1.0f;
        write_model("scores", bias, count);
        execute(executable, "scores", "-", 0); compare_output("scores", bias, count);
    }
    execute(executable, "scores", "missing", 1);
    CHECK(strcmp(error, "cinder: input \"missing\" [IO]: Could not open input file for reading.\n") == 0);
    execute_command(executable, "scores", "missing", "3", 1);
    CHECK(strcmp(error, "cinder: input \"missing\" [IO]: Could not open input file for reading.\n") == 0);
    for (size_t bytes = 0; bytes <= 5; ++bytes) {
        if (bytes == 4) continue;
        FILE *file = fopen("bad-input", "wb"); CHECK(file != NULL);
        for (size_t i = 0; i < bytes; ++i) CHECK(fputc(0, file) != EOF);
        CHECK(fclose(file) == 0);
        execute(executable, "scores", "bad-input", 1);
        char expected[256];
        snprintf(expected, sizeof(expected), "cinder: input \"bad-input\" [FORMAT]: Expected 4 bytes (1 float32 values); found %zu bytes.\n", bytes);
        CHECK(strcmp(error, expected) == 0);
    }
    const float bad[] = {NAN, INFINITY, -INFINITY};
    for (size_t i = 0; i < 3; ++i) {
        write_input("bad-input", bad[i]); execute(executable, "scores", "bad-input", 1);
        CHECK(strcmp(error, "cinder: input \"bad-input\" [NONFINITE]: Input value at index 0 is not finite.\n") == 0);
    }
    mutate_word("scores", 4, 2); execute(executable, "scores", "missing", 1);
    CHECK(strstr(error, "[UNSUPPORTED]") != NULL);
    execute(executable, "scores", NULL, 1); CHECK(strstr(error, "[UNSUPPORTED]") != NULL);
    write_model("scores", bias, 3); mutate_word("scores", 8, 65537);
    execute(executable, "scores", "missing", 1); CHECK(strstr(error, "[LIMIT]") != NULL);
    write_model("scores", bias, 3); mutate_word("scores", 0, 0);
    execute(executable, "scores", "missing", 1); CHECK(strstr(error, "[FORMAT]") != NULL);
    execute(executable, "scores", NULL, 1); CHECK(strstr(error, "[FORMAT]") != NULL);
    write_model("scores", bias, 1); mutate_word("scores", 32, UINT32_C(0x7fc00000));
    execute(executable, "scores", NULL, 1); CHECK(strstr(error, "[NONFINITE]") != NULL);
    /* A valid model and finite input causing real arithmetic overflow. */
    write_model("scores", bias, 1); mutate_word("scores", 32, UINT32_C(0x7f7fffff));
    write_input("bad-input", 2.0f); execute(executable, "scores", "bad-input", 1);
    CHECK(strstr(error, "cinder: inference [NONFINITE]: ") == error);
    execute_command(executable, "scores", "bad-input", "3", 1);
    CHECK(strstr(error, "cinder: benchmark warm-up 1 [NONFINITE]: ") == error);
    write_input("bad-input", 0.0f); execute(executable, "scores", "bad-input", 0);
    compare_output("scores", bias, 1);
    benchmark_command(executable, "scores", "bad-input", "1", "scores", 1);

    char model[4096], input[4096], references[4096];
    CHECK(snprintf(model, sizeof(model), "%s/tests/fixtures/tiny/tiny.cnd", source) < (int)sizeof(model));
    CHECK(snprintf(input, sizeof(input), "%s/tests/fixtures/tiny/inputs/worked.f32", source) < (int)sizeof(input));
    execute(executable, model, input, 0);
    execute(executable, model, NULL, 0);
    CHECK(strcmp(output, "Model: tiny.cnd\nInput: 2\nOutput: 2\nLayers: 4\n"
        "  0: Dense 2 -> 4 (parameters: 12)\n  1: ReLU 4 -> 4 (parameters: 0)\n"
        "  2: Dense 4 -> 2 (parameters: 10)\n  3: Softmax 2 -> 2 (parameters: 0)\nParameters: 22\n") == 0);
    execute(executable, model, input, 0);
    const float tiny[] = {0.904650509f, 0.0953494608f}; compare_output("tiny.cnd", tiny, 2);
    benchmark_command(executable, model, input, "1", "tiny.cnd", 1);
    benchmark_command(executable, model, input, "17", "tiny.cnd", 17);
    CHECK(snprintf(model, sizeof(model), "%s/examples/mnist/mnist.cnd", source) < (int)sizeof(model));
    CHECK(snprintf(references, sizeof(references), "%s/tests/fixtures/mnist/probabilities.f32", source) < (int)sizeof(references));
    execute(executable, model, NULL, 0);
    CHECK(strcmp(output, "Model: mnist.cnd\nInput: 784\nOutput: 10\nLayers: 4\n"
        "  0: Dense 784 -> 128 (parameters: 100480)\n  1: ReLU 128 -> 128 (parameters: 0)\n"
        "  2: Dense 128 -> 10 (parameters: 1290)\n  3: Softmax 10 -> 10 (parameters: 0)\nParameters: 101770\n") == 0);
    FILE *file = fopen(references, "rb"); CHECK(file != NULL);
    double maximum = 0;
    for (size_t sample = 0; sample < 100; ++sample) {
        float reference[10];
        for (size_t i = 0; i < 10; ++i) reference[i] = read_float(file);
        CHECK(snprintf(input, sizeof(input), "%s/tests/fixtures/mnist/inputs/%05zu.f32", source, sample) < (int)sizeof(input));
        execute(executable, model, input, 0);
        double difference = compare_output("mnist.cnd", reference, 10);
        if (difference > maximum) maximum = difference;
        if (sample == 0 || sample == 8 || sample == 99) {
            benchmark_command(executable, model, input, sample == 0 ? "1" : "17", "mnist.cnd", sample == 0 ? 1 : 17);
        }
    }
    CHECK(fgetc(file) == EOF && feof(file) && !ferror(file) && fclose(file) == 0);
    CHECK(remove("-") == 0 && remove("-model with spaces") == 0 && remove("scores") == 0 && remove("bad-input") == 0);
    printf("CLI processes: %zu passed (%zu benchmark cases); MNIST prediction/top-three index agreement 100/100; maximum displayed-score absolute error %.9g.\n", commands, benchmark_commands, maximum);
    printf("Native benchmark smoke: %zu valid reports; %zu exact zero-interval rejections; no retries.\n",
           native_benchmarks - unresolved_benchmarks, unresolved_benchmarks);
    return 0;
}
