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

static void check_rejection_matrix(const char *path, const unsigned char *tiny,
                                    cinder_model_t *sentinel)
{
    unsigned char bytes[336];
    const size_t records[] = {16, 80, 96, 152};
    const uint32_t bad_versions[] = {0, 2, 256, 65536, UINT32_MAX};
    const uint32_t bad_types[] = {0, 4, 256, UINT32_MAX};
    const uint32_t bad_dimensions[] = {0, 65537, UINT32_MAX};
    for (size_t i = 0; i < COUNT(bad_versions); ++i) {
        memcpy(bytes, tiny, 168);
        put_u32(bytes, 4, bad_versions[i]);
        write_file(path, bytes, 168);
        reject(path, CINDER_ERROR_UNSUPPORTED, sentinel);
    }
    for (size_t r = 0; r < COUNT(records); ++r) {
        for (size_t i = 0; i < COUNT(bad_types); ++i) {
            memcpy(bytes, tiny, 168);
            put_u32(bytes, records[r], bad_types[i]);
            write_file(path, bytes, 168);
            reject(path, CINDER_ERROR_FORMAT, sentinel);
        }
        for (size_t field = 4; field <= 8; field += 4) {
            for (size_t i = 0; i < COUNT(bad_dimensions); ++i) {
                memcpy(bytes, tiny, 168);
                put_u32(bytes, records[r] + field, bad_dimensions[i]);
                write_file(path, bytes, 168);
                reject(path, i == 0 ? CINDER_ERROR_FORMAT : CINDER_ERROR_LIMIT, sentinel);
            }
        }
        const uint32_t payload = r == 0 ? 48 : r == 2 ? 40 : 0;
        const uint32_t wrong_lengths[] = {payload + 1, payload + 4, UINT32_MAX};
        for (size_t i = 0; i < COUNT(wrong_lengths); ++i) {
            memcpy(bytes, tiny, 168);
            put_u32(bytes, records[r] + 12, wrong_lengths[i]);
            write_file(path, bytes, 168);
            reject(path, CINDER_ERROR_FORMAT, sentinel);
        }
    }
    /* Both infinities, quiet/signaling NaNs, both signs and varied payloads,
     * at every one of the 22 weight/bias positions, including the final word. */
    const uint32_t nonfinite[] = {0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
                                  0x7f800001, 0xff800001, 0x7fffffff, 0xffffffff};
    for (size_t parameter = 0; parameter < 22; ++parameter) {
        size_t offset = parameter < 12 ? 32 + 4 * parameter : 112 + 4 * (parameter - 12);
        for (size_t i = 0; i < COUNT(nonfinite); ++i) {
            memcpy(bytes, tiny, 168);
            put_u32(bytes, offset, nonfinite[i]);
            write_file(path, bytes, 168);
            reject(path, CINDER_ERROR_NONFINITE, sentinel);
        }
    }
    /* Strict EOF rejects every possible one-byte suffix, even padding. */
    memcpy(bytes, tiny, 168);
    for (unsigned int suffix = 0; suffix <= 255; ++suffix) {
        bytes[168] = (unsigned char)suffix;
        write_file(path, bytes, 169);
        reject(path, CINDER_ERROR_FORMAT, sentinel);
    }
    memcpy(bytes + 168, tiny, 168);
    write_file(path, bytes, sizeof(bytes));
    reject(path, CINDER_ERROR_FORMAT, sentinel); /* Concatenated valid models. */

    /* Test all nine type-to-type connections, with valid payloads on each side. */
    for (uint32_t first = 1; first <= 3; ++first) {
        for (uint32_t second = 1; second <= 3; ++second) {
            memset(bytes, 0, sizeof(bytes));
            make_header(bytes, 2, 2);
            uint32_t payload1 = first == 1 ? 24 : 0;
            uint32_t payload2 = second == 1 ? 48 : 0;
            size_t next = 32 + payload1;
            make_record(bytes, 16, first, 2, 2, payload1);
            make_record(bytes, next, second, 3, 3, payload2);
            write_file(path, bytes, next + 16 + payload2);
            reject(path, CINDER_ERROR_DIMENSION_MISMATCH, sentinel);
        }
    }
}

static void write_zeros(FILE *file, uint64_t count)
{
    const unsigned char zeros[65536] = {0};
    while (count != 0) {
        size_t chunk = count > sizeof(zeros) ? sizeof(zeros) : (size_t)count;
        CHECK(fwrite(zeros, 1, chunk, file) == chunk);
        count -= chunk;
    }
}

static void write_dense_record(FILE *file, uint32_t input, uint32_t output)
{
    unsigned char record[16];
    uint64_t payload = ((uint64_t)input + 1) * output * 4;
    CHECK(payload <= UINT32_MAX);
    make_record(record, 0, 1, input, output, (uint32_t)payload);
    CHECK(fwrite(record, 1, sizeof(record), file) == sizeof(record));
    write_zeros(file, payload);
}

static void check_limits(const char *path, cinder_model_t *sentinel)
{
    unsigned char header[16], record[16];
    /* Inclusive vector and layer limits, including an activation-only network. */
    make_header(header, 65536, 1024);
    make_record(record, 0, 2, 65536, 65536, 0);
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(header, 1, sizeof(header), file) == sizeof(header));
    for (size_t i = 0; i < 1024; ++i) {
        CHECK(fwrite(record, 1, sizeof(record), file) == sizeof(record));
    }
    CHECK(fclose(file) == 0);
    cinder_model_t *model = load_ok(path, true);
    CHECK(cinder_model_input_size(model) == 65536);
    CHECK(cinder_model_output_size(model) == 65536);
    cinder_model_free(model);
    file = fopen(path, "r+b");
    CHECK(file != NULL);
    make_header(header, 65536, 1025);
    CHECK(fwrite(header, 1, sizeof(header), file) == sizeof(header));
    CHECK(fseek(file, 0, SEEK_END) == 0);
    CHECK(fwrite(record, 1, sizeof(record), file) == sizeof(record));
    CHECK(fclose(file) == 0);
    reject(path, CINDER_ERROR_LIMIT, sentinel);

    /* Cumulative Dense parameter budget at -4, exactly, and +4 bytes.
     * Every layer payload fits individually; the last case must fail on the
     * cumulative check after the first large Dense has been constructed. */
    const struct { uint32_t input, output; uint64_t bytes; } budgets[] = {
        {65279, 255, UINT64_C(67108860)},
        {65278, 256, UINT64_C(67108864)},
        {65534, 1, UINT64_C(67108868)}
    };
    for (size_t i = 0; i < COUNT(budgets); ++i) {
        uint64_t total = ((uint64_t)budgets[i].input + 1) * 256 * 4 +
                         ((uint64_t)256 + 1) * budgets[i].output * 4;
        CHECK(total == budgets[i].bytes);
        make_header(header, budgets[i].input, 2);
        file = fopen(path, "wb");
        CHECK(file != NULL);
        CHECK(fwrite(header, 1, sizeof(header), file) == sizeof(header));
        write_dense_record(file, budgets[i].input, 256);
        write_dense_record(file, 256, budgets[i].output);
        CHECK(fclose(file) == 0);
        if (i == 2) {
            reject(path, CINDER_ERROR_LIMIT, sentinel);
        } else {
            model = load_ok(path, i == 0);
            CHECK(cinder_model_input_size(model) == budgets[i].input);
            CHECK(cinder_model_output_size(model) == budgets[i].output);
            cinder_model_free(model);
        }
    }

    /* Largest legal file: 64 MiB of parameters plus all 1024 record headers.
     * Generate real bytes instead of mocking the limits or allocating a huge
     * test-side buffer. This also exercises cleanup of 1024 constructed layers. */
    make_header(header, 65535, 1024);
    make_record(record, 0, 2, 256, 256, 0);
    file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(fwrite(header, 1, sizeof(header), file) == sizeof(header));
    write_dense_record(file, 65535, 256);
    for (size_t i = 0; i < 1022; ++i) {
        CHECK(fwrite(record, 1, sizeof(record), file) == sizeof(record));
    }
    CHECK(fwrite(record, 1, sizeof(record) - 1, file) == sizeof(record) - 1);
    CHECK(ftell(file) == 67125263L);
    CHECK(fclose(file) == 0);
    reject(path, CINDER_ERROR_FORMAT, sentinel); /* One missing final header byte. */
    file = fopen(path, "ab");
    CHECK(file != NULL);
    CHECK(fputc(0, file) == 0);
    CHECK(ftell(file) == 67125264L);
    CHECK(fclose(file) == 0);
    model = load_ok(path, true);
    cinder_model_free(model);
    file = fopen(path, "ab");
    CHECK(file != NULL);
    CHECK(fputc(0, file) == 0);
    CHECK(fclose(file) == 0);
    reject(path, CINDER_ERROR_LIMIT, sentinel);
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
    check_rejection_matrix(scratch_path, tiny, original);
    check_limits(scratch_path, original);
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
