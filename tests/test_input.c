#include "../src/input.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

enum fault { NONE, OPEN, FAIL_SEEK_END, TELL, SEEK_START, ALLOCATE,
             READ_ERROR, READ_EOF, READ_NEITHER, PROBE_ERROR, PROBE_EXTRA,
             PROBE_NEITHER, CLOSE };
static enum fault fault;
static int also_fail_close, opened, closed, allocated, reads, probing;
static FILE *stream;
static void *live;

static void *test_malloc(size_t bytes)
{
    ++allocated;
    CHECK(live == NULL);
    if (fault == ALLOCATE) return NULL;
    live = malloc(bytes);
    CHECK(live != NULL);
    return live;
}
static void test_free(void *pointer)
{
    if (pointer == NULL) return;
    CHECK(pointer == live);
    free(pointer);
    live = NULL;
}
static FILE *test_open(const char *path, const char *mode)
{
    CHECK(stream == NULL && strcmp(mode, "rb") == 0);
    ++opened;
    if (fault == OPEN) return NULL;
    stream = fopen(path, mode);
    CHECK(stream != NULL);
    return stream;
}
static int test_seek(FILE *file, long offset, int origin)
{
    CHECK(file == stream);
    if ((fault == FAIL_SEEK_END && origin == SEEK_END) ||
        (fault == SEEK_START && origin == SEEK_SET)) return -1;
    return fseek(file, offset, origin);
}
static long test_tell(FILE *file)
{
    CHECK(file == stream);
    return fault == TELL ? -1L : ftell(file);
}
static size_t test_read(void *bytes, size_t size, size_t count, FILE *file)
{
    CHECK(file == stream && size == 1 && count == 4);
    ++reads;
    if (reads == 2 && (fault == READ_ERROR || fault == READ_EOF || fault == READ_NEITHER)) {
        return fread(bytes, size, count - 1, file);
    }
    return fread(bytes, size, count, file);
}
static int test_probe(FILE *file)
{
    CHECK(file == stream);
    probing = 1;
    if (fault == PROBE_EXTRA) return 'x';
    if (fault == PROBE_ERROR || fault == PROBE_NEITHER) return EOF;
    return fgetc(file);
}
static int test_error(FILE *file)
{
    CHECK(file == stream);
    return (reads == 2 && fault == READ_ERROR) ||
           (probing && fault == PROBE_ERROR) || ferror(file);
}
static int test_eof(FILE *file)
{
    CHECK(file == stream);
    /* Both flags on READ_ERROR establish that IO wins over truncation. */
    return (reads == 2 && (fault == READ_ERROR || fault == READ_EOF)) || feof(file);
}
static int test_close(FILE *file)
{
    CHECK(file == stream);
    ++closed;
    CHECK(fclose(file) == 0);
    stream = NULL;
    return fault == CLOSE || also_fail_close ? EOF : 0;
}
#define malloc test_malloc
#define free test_free
#define fopen test_open
#define fseek test_seek
#define ftell test_tell
#define fread test_read
#define fgetc test_probe
#define ferror test_error
#define feof test_eof
#define fclose test_close
#include "../src/input.c"
#undef malloc
#undef free
#undef fopen
#undef fseek
#undef ftell
#undef fread
#undef fgetc
#undef ferror
#undef feof
#undef fclose

static void reset(enum fault next)
{
    CHECK(stream == NULL && live == NULL);
    fault = next;
    also_fail_close = opened = closed = allocated = reads = probing = 0;
}
static void write_values(const char *path, const uint32_t *bits, size_t count, size_t bytes)
{
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    for (size_t i = 0; i < bytes; ++i) {
        uint32_t value = i / 4 < count ? bits[i / 4] : 0;
        CHECK(fputc((int)((value >> (8 * (i % 4))) & 255), file) != EOF);
    }
    CHECK(fclose(file) == 0);
}
static void failure(const char *path, size_t count, cinder_status_t expected, const char *detail)
{
    float sentinel;
    float *values = &sentinel;
    cinder_error_t error = {0};
    CHECK(cinder_input_read(path, count, &values, &error) == expected);
    CHECK(values == NULL && error.code == expected && strcmp(error.message, detail) == 0);
    CHECK(stream == NULL && live == NULL && closed == (opened && fault != OPEN));
}

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    const char *path = argv[1];
    uint32_t bits[] = {0, UINT32_C(0x80000000), 1, UINT32_C(0x3f800000),
                       UINT32_C(0xc0200000), UINT32_C(0x7f7fffff)};
    write_values(path, bits, 6, 24);
    for (int details = 0; details < 2; ++details) {
        reset(NONE);
        cinder_error_t error = {CINDER_ERROR_IO, "old error"};
        float *values = NULL;
        CHECK(cinder_input_read(path, 6, &values, details ? &error : NULL) == CINDER_OK);
        CHECK(opened == 1 && closed == 1 && allocated == 1 && reads == 6);
        for (size_t i = 0; i < 6; ++i) {
            uint32_t actual;
            memcpy(&actual, values + i, 4);
            CHECK(actual == bits[i]);
        }
        if (details) CHECK(error.code == CINDER_OK && error.message[0] == '\0');
        test_free(values);
    }
    const struct { enum fault fault; cinder_status_t status; const char *message; } cases[] = {
        {OPEN, CINDER_ERROR_IO, "Could not open input file for reading."},
        {FAIL_SEEK_END, CINDER_ERROR_IO, "Could not seek to end of input file."},
        {TELL, CINDER_ERROR_IO, "Could not determine input file size."},
        {SEEK_START, CINDER_ERROR_IO, "Could not seek to start of input file."},
        {ALLOCATE, CINDER_ERROR_ALLOCATION, "Could not allocate input buffer."},
        {READ_ERROR, CINDER_ERROR_IO, "Input file read failed."},
        {READ_EOF, CINDER_ERROR_FORMAT, "Unexpected end of input file at byte 7."},
        {READ_NEITHER, CINDER_ERROR_IO, "Incomplete input file read."},
        {PROBE_ERROR, CINDER_ERROR_IO, "Final input EOF check failed."},
        {PROBE_EXTRA, CINDER_ERROR_FORMAT, "Trailing data after 6 float32 values."},
        {PROBE_NEITHER, CINDER_ERROR_IO, "Final input EOF check failed."},
        {CLOSE, CINDER_ERROR_IO, "Could not close input file."}
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        for (int close_failure = 0; close_failure < 2; ++close_failure) {
            reset(cases[i].fault);
            also_fail_close = close_failure;
            failure(path, 6, cases[i].status, cases[i].message);
        }
        reset(cases[i].fault);
        float *values = NULL;
        CHECK(cinder_input_read(path, 6, &values, NULL) == cases[i].status);
        CHECK(values == NULL && stream == NULL && live == NULL);
    }
    for (size_t bytes = 0; bytes <= 25; ++bytes) {
        if (bytes == 24) continue;
        write_values(path, bits, 6, bytes);
        reset(NONE);
        char expected[128];
        snprintf(expected, sizeof(expected), "Expected 24 bytes (6 float32 values); found %zu bytes.", bytes);
        failure(path, 6, CINDER_ERROR_FORMAT, expected);
        CHECK(allocated == 0 && reads == 0);
    }
    const uint32_t nonfinite[] = {UINT32_C(0x7fc00000), UINT32_C(0x7f800000), UINT32_C(0xff800000)};
    for (size_t i = 0; i < 3; ++i) {
        bits[4] = nonfinite[i]; bits[5] = nonfinite[(i + 1) % 3];
        write_values(path, bits, 6, 24);
        reset(NONE); also_fail_close = 1;
        failure(path, 6, CINDER_ERROR_NONFINITE, "Input value at index 4 is not finite.");
        CHECK(reads == 5 && !probing);
    }
    reset(NONE);
    failure(path, SIZE_MAX / 4 + 1, CINDER_ERROR_OVERFLOW, "Input byte count is not representable.");
    CHECK(opened == 0 && allocated == 0);
    CHECK(cinder_input_read(path, 1, NULL, NULL) == CINDER_ERROR_INVALID_ARGUMENT);
    failure(NULL, 1, CINDER_ERROR_INVALID_ARGUMENT, "Input path and positive count are required.");
    failure("", 1, CINDER_ERROR_INVALID_ARGUMENT, "Input path and positive count are required.");
    failure(path, 0, CINDER_ERROR_INVALID_ARGUMENT, "Input path and positive count are required.");
    CHECK(remove(path) == 0);
    puts("Input reader: binary32 decoding, lengths, non-finite values, I/O faults and cleanup passed.");
    return 0;
}
