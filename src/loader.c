#include "cinder/model.h"

#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CND_MAX_LAYERS UINT32_C(1024)
#define CND_MAX_VECTOR UINT32_C(65536)
#define CND_MAX_PARAMETERS UINT64_C(67108864)
#define CND_MAX_FILE UINT64_C(67125264)

_Static_assert(CHAR_BIT == 8 && sizeof(float) == sizeof(uint32_t),
               "Cinder loading requires 8-bit bytes and 32-bit floats.");

typedef struct {
    FILE *file;
    uint64_t size;
    uint64_t offset;
    size_t layer;
    bool in_layer;
    cinder_error_t *error;
} cnd_reader_t;

static cinder_status_t loader_result(
    const cnd_reader_t *reader, cinder_status_t status, const char *message
)
{
    cinder_error_t *error = reader->error;
    if (error != NULL) {
        error->code = status;
        if (status == CINDER_OK) {
            error->message[0] = '\0';
        } else if (reader->in_layer) {
            snprintf(error->message, sizeof(error->message),
                     "Layer %zu at byte %" PRIu64 ": %s",
                     reader->layer, reader->offset, message);
        } else {
            snprintf(error->message, sizeof(error->message),
                     "At byte %" PRIu64 ": %s", reader->offset, message);
        }
    }
    return status;
}

static uint32_t cnd_u32le(const unsigned char *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static bool cnd_host_supported(void)
{
    /* Numerical traits alone do not establish object layout. These probes
     * also reject mixed integer/float byte order and missing signed zero or
     * subnormal representations before any encoded bits enter a float. */
    const float values[] = {
        0.0f, -0.0f, 1.0f, -2.5f, 0x1.2468acp+0f,
        FLT_MIN, FLT_TRUE_MIN, FLT_MAX
    };
    const uint32_t bits[] = {
        UINT32_C(0), UINT32_C(0x80000000), UINT32_C(0x3f800000),
        UINT32_C(0xc0200000), UINT32_C(0x3f923456), UINT32_C(0x00800000),
        UINT32_C(1), UINT32_C(0x7f7fffff)
    };
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        uint32_t actual;
        memcpy(&actual, &values[i], sizeof(actual));
        if (actual != bits[i]) {
            return false;
        }
    }
    return true;
}

static cinder_status_t cnd_available(cnd_reader_t *reader, uint64_t count)
{
    /* offset never exceeds size; subtraction avoids overflowing an end offset. */
    if (count > reader->size - reader->offset) {
        return loader_result(reader, CINDER_ERROR_FORMAT,
                             "Declared header or payload is truncated.");
    }
    return CINDER_OK;
}

static cinder_status_t cnd_read(cnd_reader_t *reader, unsigned char *bytes, size_t count)
{
    cinder_status_t status = cnd_available(reader, count);
    if (status != CINDER_OK) {
        return status;
    }
    size_t obtained = fread(bytes, 1, count, reader->file);
    reader->offset += obtained;
    if (obtained != count) {
        if (ferror(reader->file)) {
            return loader_result(reader, CINDER_ERROR_IO, "File read failed.");
        }
        if (feof(reader->file)) {
            return loader_result(reader, CINDER_ERROR_FORMAT, "Unexpected end of file.");
        }
        return loader_result(reader, CINDER_ERROR_IO, "Incomplete file read.");
    }
    return CINDER_OK;
}

static cinder_status_t cnd_dimension(cnd_reader_t *reader, uint32_t dimension)
{
    if (dimension == 0) {
        return loader_result(reader, CINDER_ERROR_FORMAT, "Vector dimension is zero.");
    }
    if (dimension > CND_MAX_VECTOR) {
        return loader_result(reader, CINDER_ERROR_LIMIT, "Vector dimension exceeds 65536.");
    }
    if ((uint64_t)dimension > SIZE_MAX / sizeof(float)) {
        return loader_result(reader, CINDER_ERROR_OVERFLOW, "Vector byte extent exceeds SIZE_MAX.");
    }
    return CINDER_OK;
}

static cinder_status_t cnd_parameters(cnd_reader_t *reader, float *values, size_t count)
{
    /* Bounded byte scratch; only one Dense's aligned staging array is retained. */
    unsigned char bytes[4096];
    size_t done = 0;
    while (done < count) {
        size_t batch = count - done;
        if (batch > sizeof(bytes) / 4) {
            batch = sizeof(bytes) / 4;
        }
        uint64_t start = reader->offset;
        cinder_status_t status = cnd_read(reader, bytes, batch * 4);
        if (status != CINDER_OK) {
            return status;
        }
        for (size_t i = 0; i < batch; ++i) {
            uint32_t bits = cnd_u32le(&bytes[i * 4]);
            if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
                reader->offset = start + i * 4;
                return loader_result(reader, CINDER_ERROR_NONFINITE,
                                     "Parameter encodes NaN or infinity.");
            }
            memcpy(&values[done + i], &bits, sizeof(bits));
        }
        done += batch;
    }
    return CINDER_OK;
}

cinder_status_t cinder_model_load(
    const char *path, cinder_model_t **out_model, cinder_error_t *error
)
{
    cnd_reader_t reader = {NULL, 0, 0, 0, false, error};
    if (out_model == NULL) {
        return loader_result(&reader, CINDER_ERROR_INVALID_ARGUMENT,
                             "Model output pointer must not be NULL.");
    }
    *out_model = NULL;
    if (path == NULL || path[0] == '\0') {
        return loader_result(&reader, CINDER_ERROR_INVALID_ARGUMENT, "Path must not be NULL or empty.");
    }
    if (!cnd_host_supported()) {
        return loader_result(&reader, CINDER_ERROR_UNSUPPORTED, "Host float encoding is not IEEE binary32.");
    }

    cinder_status_t status = CINDER_OK;
    cinder_layer_t **layers = NULL;
    size_t layer_count = 0;
    float *parameters = NULL;
    cinder_model_t *candidate = NULL;
    cinder_error_t cause;
    unsigned char header[16];
    reader.file = fopen(path, "rb");
    if (reader.file == NULL) {
        return loader_result(&reader, CINDER_ERROR_IO, "Could not open file for reading.");
    }
    if (fseek(reader.file, 0, SEEK_END) != 0) {
        status = loader_result(&reader, CINDER_ERROR_IO, "Could not seek to end of file.");
        goto cleanup;
    }
    long length = ftell(reader.file);
    if (length < 0) {
        status = loader_result(&reader, CINDER_ERROR_IO, "Could not determine file size.");
        goto cleanup;
    }
    if ((uintmax_t)length > CND_MAX_FILE) {
        status = loader_result(&reader, CINDER_ERROR_LIMIT, "File exceeds the .cnd size limit.");
        goto cleanup;
    }
    reader.size = (uint64_t)length;
    if (fseek(reader.file, 0, SEEK_SET) != 0) {
        status = loader_result(&reader, CINDER_ERROR_IO, "Could not seek to start of file.");
        goto cleanup;
    }
    status = cnd_read(&reader, header, sizeof(header));
    if (status != CINDER_OK) {
        goto cleanup;
    }
    if (memcmp(header, "CNDR", 4) != 0) {
        status = loader_result(&reader, CINDER_ERROR_FORMAT, "Header magic must be CNDR.");
        goto cleanup;
    }
    if (cnd_u32le(header + 4) != 1) {
        status = loader_result(&reader, CINDER_ERROR_UNSUPPORTED, "Unsupported .cnd version (expected 1).");
        goto cleanup;
    }
    uint32_t expected_input = cnd_u32le(header + 8);
    uint32_t declared_layers = cnd_u32le(header + 12);
    status = cnd_dimension(&reader, expected_input);
    if (status != CINDER_OK) {
        goto cleanup;
    }
    if (declared_layers == 0 || declared_layers > CND_MAX_LAYERS) {
        status = loader_result(&reader,
            declared_layers == 0 ? CINDER_ERROR_FORMAT : CINDER_ERROR_LIMIT,
            "Header layer count must be between 1 and 1024.");
        goto cleanup;
    }
    status = cnd_available(&reader, (uint64_t)declared_layers * 16);
    if (status != CINDER_OK) {
        goto cleanup;
    }
    if ((uint64_t)declared_layers > SIZE_MAX / sizeof(*layers)) {
        status = loader_result(&reader, CINDER_ERROR_OVERFLOW, "Layer list byte extent exceeds SIZE_MAX.");
        goto cleanup;
    }
    layer_count = (size_t)declared_layers;
    layers = malloc(layer_count * sizeof(*layers));
    if (layers == NULL) {
        status = loader_result(&reader, CINDER_ERROR_ALLOCATION, "Could not allocate temporary layer list.");
        goto cleanup;
    }
    for (size_t i = 0; i < layer_count; ++i) {
        layers[i] = NULL;
    }

    uint64_t total_parameters = 0;
    for (size_t i = 0; i < layer_count; ++i) {
        reader.in_layer = true;
        reader.layer = i;
        status = cnd_read(&reader, header, sizeof(header));
        if (status != CINDER_OK) {
            goto cleanup;
        }
        uint32_t type = cnd_u32le(header);
        uint32_t input = cnd_u32le(header + 4);
        uint32_t output = cnd_u32le(header + 8);
        uint32_t payload = cnd_u32le(header + 12);
        if (type < 1 || type > 3) {
            status = loader_result(&reader, CINDER_ERROR_FORMAT, "Unknown layer type ID.");
            goto cleanup;
        }
        status = cnd_dimension(&reader, input);
        if (status != CINDER_OK) {
            goto cleanup;
        }
        status = cnd_dimension(&reader, output);
        if (status != CINDER_OK) {
            goto cleanup;
        }
        if (input != expected_input || (type != 1 && input != output)) {
            status = loader_result(&reader, CINDER_ERROR_DIMENSION_MISMATCH,
                                   "Layer connection or activation dimensions disagree.");
            goto cleanup;
        }
        /* Both dimensions are <= 65536. Product, sum and byte extent fit
         * uint64_t (at most 17,180,131,328 bytes), regardless of size_t width. */
        uint64_t weights = type == 1 ? (uint64_t)input * output : 0;
        uint64_t count = type == 1 ? weights + output : 0;
        uint64_t bytes = count * 4;
        if (bytes != payload) {
            status = loader_result(&reader, CINDER_ERROR_FORMAT, "Payload length does not match layer dimensions.");
            goto cleanup;
        }
        if (bytes > CND_MAX_PARAMETERS - total_parameters) {
            status = loader_result(&reader, CINDER_ERROR_LIMIT, "Cumulative parameter bytes exceed 64 MiB.");
            goto cleanup;
        }
        if (count > SIZE_MAX / sizeof(float)) {
            status = loader_result(&reader, CINDER_ERROR_OVERFLOW, "Parameter staging byte extent exceeds SIZE_MAX.");
            goto cleanup;
        }
        status = cnd_available(&reader, bytes);
        if (status != CINDER_OK) {
            goto cleanup;
        }
        total_parameters += bytes;
        if (type == 1) {
            parameters = malloc((size_t)count * sizeof(*parameters));
            if (parameters == NULL) {
                status = loader_result(&reader, CINDER_ERROR_ALLOCATION, "Could not allocate parameter staging.");
                goto cleanup;
            }
            status = cnd_parameters(&reader, parameters, (size_t)count);
            if (status != CINDER_OK) {
                goto cleanup;
            }
            status = cinder_layer_create_dense((size_t)input, (size_t)output,
                parameters, (size_t)weights, parameters + (size_t)weights,
                (size_t)output, &layers[i], &cause);
            free(parameters);
            parameters = NULL;
        } else {
            cinder_layer_type_t runtime_type = type == 2 ? CINDER_LAYER_RELU : CINDER_LAYER_SOFTMAX;
            status = cinder_layer_create_activation(runtime_type, (size_t)input, &layers[i], &cause);
        }
        if (status != CINDER_OK) {
            status = loader_result(&reader, status, cause.message);
            goto cleanup;
        }
        expected_input = output;
    }
    reader.in_layer = false;
    if (reader.offset != reader.size) {
        status = loader_result(&reader, CINDER_ERROR_FORMAT, "Trailing data after final layer.");
        goto cleanup;
    }
    int next_byte = fgetc(reader.file);
    if (ferror(reader.file)) {
        status = loader_result(&reader, CINDER_ERROR_IO, "Final EOF check failed.");
        goto cleanup;
    }
    if (next_byte != EOF) {
        status = loader_result(&reader, CINDER_ERROR_FORMAT, "Trailing data after final layer.");
        goto cleanup;
    }
    if (!feof(reader.file)) {
        status = loader_result(&reader, CINDER_ERROR_IO, "Final EOF was not confirmed.");
        goto cleanup;
    }
    int close_status = fclose(reader.file);
    reader.file = NULL; /* A close attempt consumes the stream even on error. */
    if (close_status != 0) {
        status = loader_result(&reader, CINDER_ERROR_IO, "Could not close model file.");
        goto cleanup;
    }
    status = cinder_model_create(layers, layer_count, &candidate, &cause);
    if (status != CINDER_OK) {
        status = loader_result(&reader, status, cause.message);
        goto cleanup;
    }
    free(layers); /* Successful model creation cleared and adopted every slot. */
    *out_model = candidate;
    return loader_result(&reader, CINDER_OK, "");

cleanup:
    free(parameters);
    if (layers != NULL) {
        for (size_t i = 0; i < layer_count; ++i) {
            cinder_layer_free(layers[i]);
        }
        free(layers);
    }
    if (reader.file != NULL) {
        (void)fclose(reader.file); /* Preserve the original failure and diagnostic. */
    }
    return status;
}
