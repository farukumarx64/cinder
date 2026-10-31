#include "cinder/model.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)

static void result(cinder_status_t actual, cinder_status_t expected, cinder_error_t *error)
{
    CHECK(actual == expected);
    if (error != NULL) {
        CHECK(error->code == expected);
        CHECK(memchr(error->message, '\0', sizeof(error->message)) != NULL);
        CHECK((error->message[0] == '\0') == (expected == CINDER_OK));
    }
}

static void inspect(cinder_error_t *error)
{
    /* Non-square layers, all-zero parameters: count slots, not nonzero values. */
    const float weights[8] = {0}, bias[4] = {0};
    cinder_layer_t *layers[4] = {0};
    CHECK(cinder_layer_create_dense(2, 4, weights, 8, bias, 4, &layers[0], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_activation(CINDER_LAYER_RELU, 4, &layers[1], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_dense(4, 2, weights, 8, bias, 2, &layers[2], NULL) == CINDER_OK);
    CHECK(cinder_layer_create_activation(CINDER_LAYER_SOFTMAX, 2, &layers[3], NULL) == CINDER_OK);
    cinder_model_t *model = NULL;
    CHECK(cinder_model_create(layers, 4, &model, NULL) == CINDER_OK);
    cinder_model_info_t summary, saved;
    memset(&summary, 0x7f, sizeof(summary));
    memcpy(&saved, &summary, sizeof(saved));
    result(cinder_model_get_info(NULL, &summary, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    CHECK(memcmp(&summary, &saved, sizeof(summary)) == 0);
    result(cinder_model_get_info(model, NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    result(cinder_model_get_info(model, &summary, error), CINDER_OK, error);
    CHECK(summary.input_size == 2 && summary.output_size == 2);
    CHECK(summary.layer_count == 4 && summary.parameter_count == 22);
    const cinder_layer_info_t expected[] = {
        {CINDER_LAYER_DENSE, 2, 4, 12}, {CINDER_LAYER_RELU, 4, 4, 0},
        {CINDER_LAYER_DENSE, 4, 2, 10}, {CINDER_LAYER_SOFTMAX, 2, 2, 0}
    };
    cinder_layer_info_t snapshot[4];
    for (size_t i = 0; i < 4; ++i) {
        result(cinder_model_get_layer_info(model, i, &snapshot[i], error), CINDER_OK, error);
        CHECK(snapshot[i].type == expected[i].type);
        CHECK(snapshot[i].input_size == expected[i].input_size);
        CHECK(snapshot[i].output_size == expected[i].output_size);
        CHECK(snapshot[i].parameter_count == expected[i].parameter_count);
    }
    cinder_layer_info_t sentinel;
    memcpy(&sentinel, &snapshot[3], sizeof(sentinel));
    result(cinder_model_get_layer_info(NULL, 0, &snapshot[3], error), CINDER_ERROR_INVALID_ARGUMENT, error);
    result(cinder_model_get_layer_info(model, 0, NULL, error), CINDER_ERROR_INVALID_ARGUMENT, error);
    result(cinder_model_get_layer_info(model, 4, &snapshot[3], error), CINDER_ERROR_OUT_OF_BOUNDS, error);
    result(cinder_model_get_layer_info(model, SIZE_MAX, &snapshot[3], error), CINDER_ERROR_OUT_OF_BOUNDS, error);
    CHECK(memcmp(&snapshot[3], &sentinel, sizeof(sentinel)) == 0);
    const float input[2] = {1, -2};
    float output[2];
    CHECK(cinder_model_run(model, input, 2, output, 2, NULL) == CINDER_OK);
    CHECK(output[0] == 0.5f && output[1] == 0.5f);
    result(cinder_model_get_info(model, &summary, error), CINDER_OK, error);
    CHECK(summary.parameter_count == 22);
    cinder_model_free(model);
    CHECK(snapshot[0].parameter_count == 12 && summary.layer_count == 4);

    /* No Dense or final Softmax is required for inspection. */
    for (size_t count = 1; count <= 2; ++count) {
        for (size_t i = 0; i < count; ++i) {
            CHECK(cinder_layer_create_activation(CINDER_LAYER_RELU, 3, &layers[i], NULL) == CINDER_OK);
        }
        CHECK(cinder_model_create(layers, count, &model, NULL) == CINDER_OK);
        result(cinder_model_get_info(model, &summary, error), CINDER_OK, error);
        CHECK(summary.input_size == 3 && summary.output_size == 3);
        CHECK(summary.parameter_count == 0 && summary.layer_count == count);
        cinder_model_free(model);
    }
}

int main(void)
{
    cinder_error_t error;
    inspect(NULL);
    inspect(&error);
    puts("Model metadata: shapes, counts, errors, immutable snapshots, and activation-only models passed.");
    return 0;
}
