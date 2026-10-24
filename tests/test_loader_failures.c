#include "cinder/model.h"

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AUDIT_CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: audit failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)
#define AUDIT_COUNT(array) (sizeof(array) / sizeof((array)[0]))

typedef struct { void *pointer; size_t size; } audit_block_t;
static audit_block_t audit_blocks[2048];
static size_t audit_live, audit_bytes, audit_allocations, audit_frees;
static size_t audit_fail_allocation;
static const size_t audit_native_size_max = SIZE_MAX;
static size_t audit_size_max = SIZE_MAX;

static void *audit_remember(void *pointer, size_t size)
{
    AUDIT_CHECK(pointer != NULL && audit_live < AUDIT_COUNT(audit_blocks));
    audit_blocks[audit_live++] = (audit_block_t){pointer, size};
    audit_bytes += size;
    return pointer;
}

static void *audit_malloc(size_t size)
{
    ++audit_allocations;
    if (audit_allocations == audit_fail_allocation) {
        return NULL;
    }
    return audit_remember(malloc(size), size);
}

static void *audit_calloc(size_t count, size_t size)
{
    AUDIT_CHECK(count == 0 || size <= audit_native_size_max / count);
    ++audit_allocations;
    if (audit_allocations == audit_fail_allocation) {
        return NULL;
    }
    return audit_remember(calloc(count, size), count * size);
}

static void audit_free(void *pointer)
{
    if (pointer == NULL) {
        return;
    }
    for (size_t i = 0; i < audit_live; ++i) {
        if (audit_blocks[i].pointer == pointer) {
            size_t size = audit_blocks[i].size;
            audit_bytes -= size;
            audit_blocks[i] = audit_blocks[--audit_live];
            ++audit_frees;
            memset(pointer, 0xa5, size); /* Expose reliance on discarded staging. */
            free(pointer);
            return;
        }
    }
    AUDIT_CHECK(0 && "untracked or double free");
}

enum audit_read_fault { AUDIT_READ_OK, AUDIT_READ_ERROR, AUDIT_READ_EOF, AUDIT_READ_NEITHER };
enum audit_probe_fault { AUDIT_PROBE_OK, AUDIT_PROBE_ERROR, AUDIT_PROBE_EXTRA, AUDIT_PROBE_NEITHER };
static FILE *audit_stream;
static size_t audit_opens, audit_closes, audit_seeks, audit_reads, audit_probes;
static size_t audit_model_creates;
static bool audit_fail_open, audit_fail_tell, audit_fail_close;
static size_t audit_fail_seek, audit_fail_read;
static enum audit_read_fault audit_read_fault;
static enum audit_probe_fault audit_probe_fault;
static bool audit_read_injected, audit_probe_injected;

static FILE *audit_fopen(const char *path, const char *mode)
{
    AUDIT_CHECK(audit_stream == NULL && strcmp(mode, "rb") == 0);
    ++audit_opens;
    if (audit_fail_open) {
        return NULL;
    }
    audit_stream = fopen(path, mode);
    return audit_stream;
}

static int audit_fseek(FILE *stream, long offset, int origin)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream);
    ++audit_seeks;
    if (audit_seeks == audit_fail_seek) {
        return -1;
    }
    return fseek(stream, offset, origin);
}

static long audit_ftell(FILE *stream)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream);
    return audit_fail_tell ? -1L : ftell(stream);
}

static size_t audit_fread(void *bytes, size_t size, size_t count, FILE *stream)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream && size == 1 && count > 0);
    ++audit_reads;
    if (audit_reads == audit_fail_read) {
        audit_read_injected = true;
        return fread(bytes, size, count - 1, stream);
    }
    return fread(bytes, size, count, stream);
}

static int audit_fgetc(FILE *stream)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream);
    ++audit_probes;
    if (audit_probe_fault != AUDIT_PROBE_OK) {
        audit_probe_injected = true;
        return audit_probe_fault == AUDIT_PROBE_EXTRA ? 0 : EOF;
    }
    return fgetc(stream);
}

static int audit_ferror(FILE *stream)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream);
    return (audit_read_injected && audit_read_fault == AUDIT_READ_ERROR) ||
           (audit_probe_injected && audit_probe_fault == AUDIT_PROBE_ERROR) || ferror(stream);
}

static int audit_feof(FILE *stream)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream);
    /* ERROR deliberately sets both flags, checking that IO takes precedence. */
    return (audit_read_injected && (audit_read_fault == AUDIT_READ_EOF ||
                                    audit_read_fault == AUDIT_READ_ERROR)) || feof(stream);
}

static int audit_fclose(FILE *stream)
{
    AUDIT_CHECK(stream != NULL && stream == audit_stream);
    ++audit_closes;
    audit_stream = NULL;
    AUDIT_CHECK(fclose(stream) == 0);
    /* Real close consumes the stream even when the injected result is EOF. */
    return audit_fail_close ? EOF : 0;
}

/* Substitute only in the real runtime sources, never in fixture-writing code
 * or libc internals. There are no production hooks or public API changes. */
#define malloc audit_malloc
#define calloc audit_calloc
#define free audit_free
#include "../src/tensor.c"
#include "../src/dense.c"
#include "../src/layer.c"
#include "../src/model.c"

static cinder_status_t audit_model_create(cinder_layer_t **layers, size_t count,
                                          cinder_model_t **model, cinder_error_t *error)
{
    AUDIT_CHECK(audit_stream == NULL); /* Close must precede the ownership transfer. */
    ++audit_model_creates;
    return cinder_model_create(layers, count, model, error);
}

#define fopen audit_fopen
#define fseek audit_fseek
#define ftell audit_ftell
#define fread audit_fread
#define fgetc audit_fgetc
#define feof audit_feof
#define ferror audit_ferror
#define fclose audit_fclose
#define cinder_model_create audit_model_create
#define cinder_model_load audit_real_load
/* Format limits fit native size_t on supported 32/64-bit hosts. Lower this
 * ceiling only in the loader's guards to exercise otherwise unreachable native
 * overflow branches. This is NOT a simulation of a different ABI or allocator. */
#undef SIZE_MAX
#define SIZE_MAX audit_size_max
#include "../src/loader.c"
#undef SIZE_MAX
#define SIZE_MAX audit_native_size_max
#undef cinder_model_load
#undef cinder_model_create
#undef fopen
#undef fseek
#undef ftell
#undef fread
#undef fgetc
#undef feof
#undef ferror
#undef fclose
#undef malloc
#undef calloc
#undef free

static size_t audit_rejections;

cinder_status_t cinder_model_load(const char *path, cinder_model_t **model, cinder_error_t *error)
{
    /* Check each failing call, not merely the totals at process exit. Existing
     * model allocations must survive unchanged while all new blocks disappear. */
    audit_block_t before[2048];
    size_t before_count = audit_live, before_bytes = audit_bytes;
    memcpy(before, audit_blocks, before_count * sizeof(*before));
    size_t opens_before = audit_opens, closes_before = audit_closes;
    cinder_status_t status = audit_real_load(path, model, error);
    AUDIT_CHECK(audit_stream == NULL);
    AUDIT_CHECK(audit_opens - opens_before <= 1);
    AUDIT_CHECK(audit_closes - closes_before <= audit_opens - opens_before);
    if (status != CINDER_OK) {
        ++audit_rejections;
        AUDIT_CHECK(model == NULL || *model == NULL);
        AUDIT_CHECK(audit_live == before_count && audit_bytes == before_bytes);
        for (size_t i = 0; i < before_count; ++i) {
            bool found = false;
            for (size_t j = 0; j < audit_live; ++j) {
                if (before[i].pointer == audit_blocks[j].pointer) {
                    AUDIT_CHECK(before[i].size == audit_blocks[j].size);
                    found = true;
                }
            }
            AUDIT_CHECK(found);
        }
    }
    return status;
}

/* Run exactly the same public success/rejection/limit cases through the tracked
 * runtime, with a separate scratch directory from the normally linked test. */
#define main loader_public_tests_main
#include "test_model_load.c"
#undef main

static void audit_reset(void)
{
    AUDIT_CHECK(audit_live == 0 && audit_bytes == 0 && audit_stream == NULL);
    audit_allocations = audit_frees = audit_fail_allocation = 0;
    audit_opens = audit_closes = audit_seeks = audit_reads = audit_probes = 0;
    audit_model_creates = 0;
    audit_fail_open = audit_fail_tell = audit_fail_close = false;
    audit_fail_seek = audit_fail_read = 0;
    audit_read_fault = AUDIT_READ_OK;
    audit_probe_fault = AUDIT_PROBE_OK;
    audit_read_injected = audit_probe_injected = false;
    audit_size_max = audit_native_size_max;
}

static void audit_retry(const char *path)
{
    audit_reset();
    cinder_model_t *model = load_ok(path, true);
    AUDIT_CHECK(audit_opens == 1 && audit_closes == 1 && audit_model_creates == 1);
    size_t allocations = audit_allocations, frees = audit_frees;
    const float input[] = {2.0f, -1.0f};
    float output[2];
    AUDIT_CHECK(cinder_model_run(model, input, 2, output, 2, NULL) == CINDER_OK);
    AUDIT_CHECK(fabsf(output[0] - 0.904650509f) < 1e-6f);
    AUDIT_CHECK(fabsf(output[1] - 0.0953494608f) < 1e-6f);
    AUDIT_CHECK(audit_allocations == allocations && audit_frees == frees);
    cinder_model_free(model);
    AUDIT_CHECK(audit_live == 0 && audit_bytes == 0 && audit_frees == audit_allocations);
}

static cinder_error_t audit_expect_failure(const char *path, cinder_status_t expected, bool details)
{
    cinder_model_t *model = NULL;
    cinder_error_t error;
    memset(&error, 0xa5, sizeof(error));
    cinder_status_t status = cinder_model_load(path, &model, details ? &error : NULL);
    check_status(status, expected, details ? &error : NULL);
    AUDIT_CHECK(model == NULL && audit_live == 0 && audit_bytes == 0);
    AUDIT_CHECK(audit_stream == NULL);
    AUDIT_CHECK(audit_allocations == audit_frees + (audit_fail_allocation != 0 ? 1 : 0));
    return error;
}

static void audit_allocation_failures(const char *path)
{
    audit_retry(path);
    size_t count = audit_allocations;
    AUDIT_CHECK(count == 21); /* list + 2 staging + 2*6 Dense + 2 activations + 4 model */
    for (size_t fail = 1; fail <= count; ++fail) {
        for (int details = 0; details < 2; ++details) {
            audit_reset();
            audit_fail_allocation = fail;
            cinder_error_t error = audit_expect_failure(path, CINDER_ERROR_ALLOCATION, details != 0);
            AUDIT_CHECK(audit_allocations == fail && audit_opens == 1 && audit_closes == 1);
            AUDIT_CHECK(audit_model_creates == (fail > 17 ? 1U : 0U));
            if (details) {
                AUDIT_CHECK(strstr(error.message, "allocate") != NULL);
            }
            audit_retry(path);
        }
    }
    printf("Audit: all %zu allocation points failed individually, with/without details and successful retry.\n", count);
}

static void audit_io_failures(const char *path)
{
    audit_retry(path);
    size_t reads = audit_reads;
    AUDIT_CHECK(reads == 7);
    for (int details = 0; details < 2; ++details) {
        for (size_t operation = 0; operation < 5; ++operation) {
            audit_reset();
            audit_fail_open = operation == 0;
            audit_fail_seek = operation == 1 ? 1 : operation == 3 ? 2 : 0;
            audit_fail_tell = operation == 2;
            audit_fail_close = operation == 4;
            (void)audit_expect_failure(path, CINDER_ERROR_IO, details != 0);
            AUDIT_CHECK(audit_opens == 1 && audit_closes == (operation == 0 ? 0U : 1U));
            AUDIT_CHECK(audit_model_creates == 0);
            AUDIT_CHECK(audit_allocations == (operation == 4 ? 17U : 0U));
            audit_retry(path);
        }
        for (size_t read = 1; read <= reads; ++read) {
            for (enum audit_read_fault mode = AUDIT_READ_ERROR; mode <= AUDIT_READ_NEITHER; ++mode) {
                audit_reset();
                audit_fail_read = read;
                audit_read_fault = mode;
                (void)audit_expect_failure(path,
                    mode == AUDIT_READ_EOF ? CINDER_ERROR_FORMAT : CINDER_ERROR_IO, details != 0);
                AUDIT_CHECK(audit_reads == read && audit_read_injected);
                AUDIT_CHECK(audit_closes == 1 && audit_model_creates == 0);
                audit_retry(path);
            }
        }
        for (enum audit_probe_fault mode = AUDIT_PROBE_ERROR; mode <= AUDIT_PROBE_NEITHER; ++mode) {
            audit_reset();
            audit_probe_fault = mode;
            (void)audit_expect_failure(path,
                mode == AUDIT_PROBE_EXTRA ? CINDER_ERROR_FORMAT : CINDER_ERROR_IO, details != 0);
            AUDIT_CHECK(audit_probes == 1 && audit_probe_injected);
            AUDIT_CHECK(audit_closes == 1 && audit_model_creates == 0);
            audit_retry(path);
        }
    }
    printf("Audit: 29 file-failure scenarios, each with/without details and successful retry.\n");
}

static void audit_first_error(const char *path, const char *scratch)
{
    unsigned char bytes[168];
    FILE *file = fopen(path, "rb");
    AUDIT_CHECK(file != NULL && fread(bytes, 1, sizeof(bytes), file) == sizeof(bytes));
    AUDIT_CHECK(fclose(file) == 0);
    put_u32(bytes, 152, 4); /* Fail after both Dense layers and ReLU exist. */
    write_file(scratch, bytes, sizeof(bytes));
    audit_reset();
    cinder_error_t original = audit_expect_failure(scratch, CINDER_ERROR_FORMAT, true);
    audit_reset();
    audit_fail_close = true;
    cinder_error_t combined = audit_expect_failure(scratch, CINDER_ERROR_FORMAT, true);
    AUDIT_CHECK(strcmp(original.message, combined.message) == 0 && audit_closes == 1);
    audit_retry(path);
    for (size_t fail = 1; fail <= 17; ++fail) {
        audit_reset();
        audit_fail_allocation = fail;
        original = audit_expect_failure(path, CINDER_ERROR_ALLOCATION, true);
        audit_reset();
        audit_fail_allocation = fail;
        audit_fail_close = true;
        combined = audit_expect_failure(path, CINDER_ERROR_ALLOCATION, true);
        AUDIT_CHECK(strcmp(original.message, combined.message) == 0 && audit_closes == 1);
        audit_retry(path);
    }
    AUDIT_CHECK(remove(scratch) == 0);
}

static void audit_native_overflow(const char *path)
{
    const size_t ceilings[] = {2 * sizeof(float) - 1, 4 * sizeof(cinder_layer_t *) - 1,
                               12 * sizeof(float) - 1};
    for (size_t i = 0; i < AUDIT_COUNT(ceilings); ++i) {
        for (int details = 0; details < 2; ++details) {
            audit_reset();
            audit_size_max = ceilings[i];
            (void)audit_expect_failure(path, CINDER_ERROR_OVERFLOW, details != 0);
            AUDIT_CHECK(audit_allocations == (i == 2 ? 1U : 0U));
            AUDIT_CHECK(audit_closes == 1 && audit_model_creates == 0);
            audit_retry(path);
        }
    }
}

static void audit_parameter_bits(const char *scratch)
{
    /* Private inspection is confined to this translation unit. The real public
     * model stays opaque; allocator poisoning makes retained staging visible. */
    const uint32_t bits[] = {0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
                             0x00800000, 0x80800000, 0x3f923456, 0xbf923456,
                             0x7f7fffff, 0xff7fffff};
    unsigned char bytes[32 + 8 * AUDIT_COUNT(bits)];
    make_header(bytes, 1, 1);
    make_record(bytes, 16, 1, 1, AUDIT_COUNT(bits), 8 * AUDIT_COUNT(bits));
    for (size_t i = 0; i < AUDIT_COUNT(bits); ++i) {
        put_u32(bytes, 32 + i * 4, bits[i]);
        put_u32(bytes, 32 + (AUDIT_COUNT(bits) + i) * 4, bits[AUDIT_COUNT(bits) - i - 1]);
    }
    write_file(scratch, bytes, sizeof(bytes));
    audit_reset();
    cinder_model_t *model = load_ok(scratch, true);
    AUDIT_CHECK(remove(scratch) == 0);
    AUDIT_CHECK(model->layer_count == 1);
    const cinder_dense_t *dense = model->layers[0]->dense;
    for (size_t i = 0; i < AUDIT_COUNT(bits); ++i) {
        uint32_t weight, bias;
        memcpy(&weight, &dense->weights->data[i], sizeof(weight));
        memcpy(&bias, &dense->bias->data[i], sizeof(bias));
        AUDIT_CHECK(weight == bits[i] && bias == bits[AUDIT_COUNT(bits) - i - 1]);
    }
    cinder_model_free(model);
    AUDIT_CHECK(audit_allocations == audit_frees);
}

int main(int argc, char **argv)
{
    AUDIT_CHECK(argc == 3);
    AUDIT_CHECK(loader_public_tests_main(argc, argv) == EXIT_SUCCESS);
    AUDIT_CHECK(audit_live == 0 && audit_bytes == 0 && audit_stream == NULL);
    AUDIT_CHECK(audit_allocations == audit_frees);
    audit_allocation_failures(argv[1]);
    audit_io_failures(argv[1]);
    audit_first_error(argv[1], argv[2]);
    audit_native_overflow(argv[1]);
    audit_parameter_bits(argv[2]);
    audit_reset();
    printf("Audit: %zu rejected calls checked for NULL output, unchanged prior owners, zero new live blocks, and no retained stream.\n",
           audit_rejections);
    return EXIT_SUCCESS;
}
