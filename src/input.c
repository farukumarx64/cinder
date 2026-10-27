#include "input.h"

#include <float.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(CHAR_BIT == 8 && sizeof(float) == sizeof(uint32_t),
               "Cinder input requires 8-bit bytes and 32-bit floats.");

static cinder_status_t input_result(cinder_error_t *error,
    cinder_status_t status, const char *format, ...)
{
    if (error != NULL) {
        error->code = status;
        va_list args;
        va_start(args, format);
        vsnprintf(error->message, sizeof(error->message), format, args);
        va_end(args);
    }
    return status;
}

static bool input_host_supported(void)
{
    const float values[] = {0.0f, -0.0f, 1.0f, -2.5f, 0x1.2468acp+0f,
                            FLT_MIN, FLT_TRUE_MIN, FLT_MAX};
    const uint32_t bits[] = {0, UINT32_C(0x80000000), UINT32_C(0x3f800000),
        UINT32_C(0xc0200000), UINT32_C(0x3f923456), UINT32_C(0x00800000),
        1, UINT32_C(0x7f7fffff)};
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        uint32_t actual;
        memcpy(&actual, &values[i], sizeof(actual));
        if (actual != bits[i]) {
            return false;
        }
    }
    return true;
}

cinder_status_t cinder_input_read(const char *path, size_t count,
                                  float **values, cinder_error_t *error)
{
    if (values == NULL) {
        return input_result(error, CINDER_ERROR_INVALID_ARGUMENT,
                            "Input output pointer must not be NULL.");
    }
    *values = NULL;
    if (path == NULL || path[0] == '\0' || count == 0) {
        return input_result(error, CINDER_ERROR_INVALID_ARGUMENT,
                            "Input path and positive count are required.");
    }
    if (!input_host_supported()) {
        return input_result(error, CINDER_ERROR_UNSUPPORTED,
                            "Host float encoding is not IEEE binary32.");
    }
    if (count > SIZE_MAX / 4 || count > SIZE_MAX / sizeof(float)) {
        return input_result(error, CINDER_ERROR_OVERFLOW,
                            "Input byte count is not representable.");
    }
    size_t expected = count * 4;
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        return input_result(error, CINDER_ERROR_IO,
                            "Could not open input file for reading.");
    }
    cinder_status_t status = CINDER_OK;
    float *buffer = NULL;
    /* All failures below converge on one close attempt and one buffer cleanup. */
#define INPUT_FAIL(code, ...) do { \
    status = input_result(error, code, __VA_ARGS__); \
    goto cleanup; \
} while (0)
    if (fseek(file, 0, SEEK_END) != 0) {
        INPUT_FAIL(CINDER_ERROR_IO, "Could not seek to end of input file.");
    }
    long length = ftell(file);
    if (length < 0) {
        INPUT_FAIL(CINDER_ERROR_IO, "Could not determine input file size.");
    }
    if ((uintmax_t)length != (uintmax_t)expected) {
        INPUT_FAIL(CINDER_ERROR_FORMAT,
                   "Expected %zu bytes (%zu float32 values); found %ju bytes.",
                   expected, count, (uintmax_t)length);
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        INPUT_FAIL(CINDER_ERROR_IO, "Could not seek to start of input file.");
    }
    buffer = malloc(count * sizeof(float));
    if (buffer == NULL) {
        INPUT_FAIL(CINDER_ERROR_ALLOCATION, "Could not allocate input buffer.");
    }
    for (size_t i = 0; i < count; ++i) {
        unsigned char bytes[4];
        size_t obtained = fread(bytes, 1, sizeof(bytes), file);
        if (ferror(file)) {
            INPUT_FAIL(CINDER_ERROR_IO, "Input file read failed.");
        }
        if (obtained != sizeof(bytes)) {
            if (feof(file)) {
                INPUT_FAIL(CINDER_ERROR_FORMAT,
                           "Unexpected end of input file at byte %zu.", i * 4 + obtained);
            }
            INPUT_FAIL(CINDER_ERROR_IO, "Incomplete input file read.");
        }
        uint32_t bits = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 |
                        (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
            INPUT_FAIL(CINDER_ERROR_NONFINITE,
                       "Input value at index %zu is not finite.", i);
        }
        memcpy(&buffer[i], &bits, sizeof(bits));
    }
    int extra = fgetc(file);
    if (ferror(file)) {
        INPUT_FAIL(CINDER_ERROR_IO, "Final input EOF check failed.");
    }
    if (extra != EOF) {
        INPUT_FAIL(CINDER_ERROR_FORMAT, "Trailing data after %zu float32 values.", count);
    }
    if (!feof(file)) {
        INPUT_FAIL(CINDER_ERROR_IO, "Final input EOF check failed.");
    }
cleanup:
    if (fclose(file) != 0 && status == CINDER_OK) {
        status = input_result(error, CINDER_ERROR_IO, "Could not close input file.");
    }
    if (status != CINDER_OK) {
        free(buffer);
        return status;
    }
    *values = buffer;
    return input_result(error, CINDER_OK, "%s", "");
#undef INPUT_FAIL
}
