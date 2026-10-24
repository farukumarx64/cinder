#include "cinder/model.h"
#include "numpy_compare.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Keep checks active in Release builds. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static numpy_comparison_stats_t stats;
static size_t rejected_files;

static void check_status(cinder_status_t status, cinder_status_t expected,
                         const cinder_error_t *error)
{
    if (status != expected) {
        fprintf(stderr, "Expected status %d, got %d: %s\n", (int)expected,
                (int)status, error == NULL ? "(no error storage)" : error->message);
        exit(EXIT_FAILURE);
    }
    if (error != NULL) {
        CHECK(error->code == status);
        CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
        CHECK((error->message[0] == '\0') == (status == CINDER_OK));
    }
}

static void write_file(const char *path, const unsigned char *bytes, size_t count)
{
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(bytes, 1, count, file) == count);
    CHECK(fclose(file) == 0);
}

static void check_file(const char *path, const unsigned char *bytes, size_t count)
{
    FILE *file = fopen(path, "rb");
    CHECK(file != NULL);
    for (size_t i = 0; i < count; ++i) {
        CHECK(fgetc(file) == bytes[i]);
    }
    CHECK(fgetc(file) == EOF && feof(file) && !ferror(file));
    CHECK(fclose(file) == 0);
}

static void put_u32(unsigned char *bytes, size_t offset, uint32_t value)
{
    for (size_t i = 0; i < 4; ++i) {
        bytes[offset + i] = (unsigned char)(value >> (8 * i));
    }
}

static void make_header(unsigned char *bytes, uint32_t input, uint32_t count)
{
    memcpy(bytes, "CNDR", 4);
    put_u32(bytes, 4, 1);
    put_u32(bytes, 8, input);
    put_u32(bytes, 12, count);
}

static void make_record(unsigned char *bytes, size_t offset, uint32_t type,
                        uint32_t input, uint32_t output, uint32_t payload)
{
    put_u32(bytes, offset, type);
    put_u32(bytes, offset + 4, input);
    put_u32(bytes, offset + 8, output);
    put_u32(bytes, offset + 12, payload);
}

static cinder_model_t *load_ok(const char *path, bool details)
{
    cinder_model_t *model = NULL;
    cinder_error_t error;
    memset(&error, 0xa5, sizeof(error));
    cinder_status_t status = cinder_model_load(path, &model, details ? &error : NULL);
    check_status(status, CINDER_OK, details ? &error : NULL);
    CHECK(model != NULL);
    return model;
}

static void reject(const char *path, cinder_status_t expected, cinder_model_t *sentinel)
{
    for (int details = 0; details < 2; ++details) {
        /* A borrowed sentinel verifies clearing without abandoning an owner. */
        cinder_model_t *model = sentinel;
        cinder_error_t error;
        memset(&error, 0xa5, sizeof(error));
        cinder_status_t status = cinder_model_load(path, &model, details ? &error : NULL);
        check_status(status, expected, details ? &error : NULL);
        CHECK(model == NULL);
        cinder_model_free(model);
    }
    ++rejected_files;
}

static void check_tiny(cinder_model_t *model)
{
    CHECK(cinder_model_input_size(model) == 2);
    CHECK(cinder_model_output_size(model) == 2);
    for (size_t i = 0; i < COUNT(numpy_network_cases); ++i) {
        const numpy_network_case_t *fixture = &numpy_network_cases[i];
        CHECK(fixture->input.count == 2);
        float input[2];
        memcpy(input, fixture->input.data, sizeof(input));
        float output[3] = {NAN, NAN, 123.0f};
        cinder_error_t error;
        cinder_status_t status = cinder_model_run(model, input, COUNT(input),
                                                output, COUNT(output), &error);
        check_status(status, CINDER_OK, &error);
        CHECK(compare_vector(&stats, fixture->name, "loaded/softmax", output, 2,
                             fixture->softmax, true));
        CHECK(output[2] == 123.0f);
        CHECK(memcmp(input, fixture->input.data, sizeof(input)) == 0);
    }
}

static void check_small_models(const char *path)
{
    unsigned char bytes[68] = {0};
    const float input[] = {-2.0f, 0.0f, 3.0f};
    for (uint32_t type = 2; type <= 3; ++type) {
        make_header(bytes, 3, 1);
        make_record(bytes, 16, type, 3, 3, 0);
        write_file(path, bytes, 32);
        cinder_model_t *model = load_ok(path, type == 2);
        CHECK(cinder_model_input_size(model) == 3);
        CHECK(cinder_model_output_size(model) == 3);
        float output[4] = {NAN, NAN, NAN, 123.0f};
        const float zeros[] = {0.0f, 0.0f, 0.0f};
        const float *values = type == 2 ? input : zeros;
        CHECK(cinder_model_run(model, values, 3, output, 4, NULL) == CINDER_OK);
        for (size_t i = 0; i < 3; ++i) {
            float expected = type == 2 ? (input[i] < 0 ? 0.0f : input[i]) : 1.0f / 3.0f;
            CHECK(fabsf(output[i] - expected) <= 1e-6f);
        }
        CHECK(output[3] == 123.0f);
        cinder_model_free(model);
    }

    /* Independent, explicitly encoded row-major 3x2 Dense fixture. */
    const uint32_t parameters[] = {
        0x3f800000, 0x40000000, 0xbf800000, 0x00000000, 0x3f000000, 0xbf000000,
        0x3f000000, 0xbf800000, 0x40000000
    };
    make_header(bytes, 2, 1);
    make_record(bytes, 16, 1, 2, 3, 36);
    for (size_t i = 0; i < COUNT(parameters); ++i) {
        put_u32(bytes, 32 + 4 * i, parameters[i]);
    }
    write_file(path, bytes, sizeof(bytes));
    cinder_model_t *model = load_ok(path, true);
    CHECK(cinder_model_input_size(model) == 2);
    CHECK(cinder_model_output_size(model) == 3);
    const float dense_input[] = {2.0f, -1.0f};
    float output[4] = {NAN, NAN, NAN, 123.0f};
    CHECK(cinder_model_run(model, dense_input, 2, output, 4, NULL) == CINDER_OK);
    CHECK(output[0] == 0.5f && output[1] == -3.0f && output[2] == 3.5f);
    CHECK(output[3] == 123.0f);
    cinder_model_free(model);

    /* Finite edge encodings are legal parameters, even if some later inputs
     * could overflow inference. Loading must not execute this Dense. */
    const uint32_t finite[] = {0, 0x80000000, 1, 0x80000001, 0x007fffff,
                               0x00800000, 0x7f7fffff, 0xff7fffff};
    make_header(bytes, 1, 1);
    make_record(bytes, 16, 1, 1, 1, 8);
    for (size_t i = 0; i < COUNT(finite); ++i) {
        put_u32(bytes, 32, finite[i]);
        put_u32(bytes, 36, finite[i]);
        write_file(path, bytes, 40);
        model = load_ok(path, true);
        cinder_model_free(model);
    }
}

static void check_malformed(const char *path, const unsigned char *tiny,
                            cinder_model_t *sentinel)
{
    unsigned char bytes[169];
    /* Every shorter prefix must fail, including after earlier layers exist. */
    for (size_t length = 0; length < 168; ++length) {
        write_file(path, tiny, length);
        reject(path, CINDER_ERROR_FORMAT, sentinel);
    }
    const struct {
        const char *name;
        size_t offset;
        uint32_t value;
        cinder_status_t status;
    } changes[] = {
        {"magic", 0, 0, CINDER_ERROR_FORMAT},
        {"version zero", 4, 0, CINDER_ERROR_UNSUPPORTED},
        {"future version", 4, 2, CINDER_ERROR_UNSUPPORTED},
        {"wrong byte order", 4, 0x01000000, CINDER_ERROR_UNSUPPORTED},
        {"zero header input", 8, 0, CINDER_ERROR_FORMAT},
        {"header vector limit", 8, 65537, CINDER_ERROR_LIMIT},
        {"header connection", 8, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {"zero layers", 12, 0, CINDER_ERROR_FORMAT},
        {"layer count limit", 12, 1025, CINDER_ERROR_LIMIT},
        {"insufficient record space", 12, 10, CINDER_ERROR_FORMAT},
        {"zero type", 16, 0, CINDER_ERROR_FORMAT},
        {"unknown type", 96, 4, CINDER_ERROR_FORMAT},
        {"unknown last type", 152, UINT32_MAX, CINDER_ERROR_FORMAT},
        {"zero input", 20, 0, CINDER_ERROR_FORMAT},
        {"zero output", 24, 0, CINDER_ERROR_FORMAT},
        {"input limit", 20, 65537, CINDER_ERROR_LIMIT},
        {"output limit", 24, 65537, CINDER_ERROR_LIMIT},
        {"Dense payload short", 28, 44, CINDER_ERROR_FORMAT},
        {"Dense payload long", 28, 52, CINDER_ERROR_FORMAT},
        {"Dense payload huge", 108, UINT32_MAX, CINDER_ERROR_FORMAT},
        {"ReLU connection", 84, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {"ReLU shape change", 88, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {"ReLU payload", 92, 4, CINDER_ERROR_FORMAT},
        {"Dense connection", 100, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {"Softmax shape change", 160, 3, CINDER_ERROR_DIMENSION_MISMATCH},
        {"Softmax payload", 164, 4, CINDER_ERROR_FORMAT},
        {"infinite weight", 32, 0x7f800000, CINDER_ERROR_NONFINITE},
        {"negative infinite bias", 64, 0xff800000, CINDER_ERROR_NONFINITE},
        {"NaN later weight", 112, 0x7fc00001, CINDER_ERROR_NONFINITE},
        {"signaling NaN last bias", 148, 0x7f800001, CINDER_ERROR_NONFINITE}
    };
    for (size_t i = 0; i < COUNT(changes); ++i) {
        memcpy(bytes, tiny, 168);
        put_u32(bytes, changes[i].offset, changes[i].value);
        write_file(path, bytes, 168);
        printf("Reject: %s\n", changes[i].name);
        reject(path, changes[i].status, sentinel);
    }
    memcpy(bytes, tiny, 168);
    for (int trailing = 0; trailing < 2; ++trailing) {
        bytes[168] = trailing == 0 ? 0 : 0xff;
        write_file(path, bytes, sizeof(bytes));
        reject(path, CINDER_ERROR_FORMAT, sentinel);
    }

    /* A product that would wrap 32-bit arithmetic cannot become a tiny payload. */
    make_header(bytes, 65536, 1);
    make_record(bytes, 16, 1, 65536, 65536, 262144);
    write_file(path, bytes, 32);
    reject(path, CINDER_ERROR_FORMAT, sentinel);
    /* Correctly derived but over-budget payload is rejected before staging. */
    make_header(bytes, 65536, 1);
    make_record(bytes, 16, 1, 65536, 256, 67109888);
    write_file(path, bytes, 32);
    reject(path, CINDER_ERROR_LIMIT, sentinel);
}

static void check_chunked_payload(const char *path)
{
    /* Cross two 4096-byte decoding boundaries, including within the biases. */
    enum { width = 1025 };
    unsigned char bytes[32 + 8 * width];
    make_header(bytes, 1, 1);
    make_record(bytes, 16, 1, 1, width, 8 * width);
    for (size_t i = 0; i < width; ++i) {
        put_u32(bytes, 32 + 4 * i, UINT32_C(0x3f800000));
        put_u32(bytes, 32 + 4 * (width + i),
                i % 2 == 0 ? UINT32_C(0x3f000000) : UINT32_C(0xbf000000));
    }
    write_file(path, bytes, sizeof(bytes));
    cinder_model_t *model = load_ok(path, true);
    CHECK(cinder_model_input_size(model) == 1);
    CHECK(cinder_model_output_size(model) == width);
    const float input = 2.0f;
    float output[width + 1];
    output[width] = 123.0f;
    CHECK(cinder_model_run(model, &input, 1, output, COUNT(output), NULL) == CINDER_OK);
    for (size_t i = 0; i < width; ++i) {
        CHECK(output[i] == (i % 2 == 0 ? 2.5f : 1.5f));
    }
    CHECK(output[width] == 123.0f);
    cinder_model_free(model);
    /* Reject non-finite data in a later read batch, with staging already filled. */
    put_u32(bytes, sizeof(bytes) - 4, UINT32_C(0x7fc00000));
    write_file(path, bytes, sizeof(bytes));
    reject(path, CINDER_ERROR_NONFINITE, NULL);
}

int main(int argc, char **argv)
{
    CHECK(argc == 3);
    const char *fixture_path = argv[1];
    const char *scratch_path = argv[2];
    unsigned char tiny[168];
    FILE *file = fopen(fixture_path, "rb");
    CHECK(file != NULL);
    CHECK(fread(tiny, 1, sizeof(tiny), file) == sizeof(tiny));
    CHECK(fgetc(file) == EOF && !ferror(file));
    CHECK(fclose(file) == 0);

    cinder_model_t *original = load_ok(fixture_path, true);
    check_tiny(original);
    cinder_error_t error;
    check_status(cinder_model_load(fixture_path, NULL, &error),
                 CINDER_ERROR_INVALID_ARGUMENT, &error);
    CHECK(cinder_model_load(NULL, NULL, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    reject(NULL, CINDER_ERROR_INVALID_ARGUMENT, original);
    reject("", CINDER_ERROR_INVALID_ARGUMENT, original);

    write_file(scratch_path, tiny, sizeof(tiny));
    cinder_model_t *first = load_ok(scratch_path, false);
    cinder_model_t *second = load_ok(scratch_path, true);
    CHECK(first != second && first != original && second != original);
    check_file(scratch_path, tiny, sizeof(tiny));
    CHECK(remove(scratch_path) == 0);
    reject(scratch_path, CINDER_ERROR_IO, original);
    check_tiny(first);
    cinder_model_free(first);
    check_tiny(second);
    cinder_model_free(second);

    check_small_models(scratch_path);
    check_chunked_payload(scratch_path);
    check_malformed(scratch_path, tiny, original);
    check_tiny(original); /* Failed loads never invalidate an independent owner. */
    write_file(scratch_path, tiny, sizeof(tiny));
    first = load_ok(scratch_path, true); /* Recovery after the malformed files. */
    CHECK(remove(scratch_path) == 0);
    check_tiny(first);
    cinder_model_free(first);
    cinder_model_free(original);
    check_file(fixture_path, tiny, sizeof(tiny));
    printf("Loader: %zu rejected cases (each with/without error details); "
           "%zu NumPy vectors, %zu values, max absolute error %.9g.\n",
           rejected_files, stats.comparisons, stats.elements, stats.largest_absolute_error);
    return EXIT_SUCCESS;
}
