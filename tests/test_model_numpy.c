#include "cinder/model.h"

/* CMake selects the saved header or the newly generated build-directory copy. */
#include "numpy_compare.h"

#include <stdlib.h>
#include <string.h>

#define COUNT(array) (sizeof(array) / sizeof((array)[0]))
/* Keep checks active with NDEBUG in Release builds. */
#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

static numpy_comparison_stats_t stats;
static const char *layer_stages[] = {"layer/dense1", "layer/relu", "layer/logits", "layer/softmax"};
static const char *model_stages[] = {"model/dense1", "model/relu", "model/logits", "model/softmax"};

static void check_status(cinder_status_t status, const cinder_error_t *error,
                         const char *fixture, const char *stage)
{
    if (status != CINDER_OK) {
        fprintf(stderr, "%s/%s: Cinder status %d: %s\n", fixture, stage, status, error->message);
        exit(EXIT_FAILURE);
    }
    CHECK(error->code == CINDER_OK && error->message[0] == '\0');
}

static numpy_vector_t reference_stage(const numpy_network_case_t *fixture, size_t stage)
{
    const numpy_vector_t stages[] = {fixture->dense1, fixture->relu,
                                    fixture->logits, fixture->softmax};
    CHECK(stage < COUNT(stages));
    return stages[stage];
}

static void check_fixture_shapes(void)
{
    const numpy_network_fixture_t *network = &numpy_tiny_network;
    CHECK(network->input_size == 2 && network->hidden_size == 4 && network->output_size == 2);
    CHECK(network->weights1.count == network->hidden_size * network->input_size);
    CHECK(network->bias1.count == network->hidden_size);
    CHECK(network->weights2.count == network->output_size * network->hidden_size);
    CHECK(network->bias2.count == network->output_size);
    CHECK(COUNT(numpy_network_cases) > 0);
    for (size_t i = 0; i < COUNT(numpy_network_cases); ++i) {
        const numpy_network_case_t *fixture = &numpy_network_cases[i];
        CHECK(fixture->input.count == network->input_size);
        CHECK(fixture->dense1.count == network->hidden_size);
        CHECK(fixture->relu.count == network->hidden_size);
        CHECK(fixture->logits.count == network->output_size);
        CHECK(fixture->softmax.count == network->output_size);
    }
}

static void create_layers(cinder_layer_t **layers, size_t count)
{
    CHECK(count >= 1 && count <= 4);
    const numpy_network_fixture_t *network = &numpy_tiny_network;
    cinder_error_t error;
    check_status(cinder_layer_create_dense(network->input_size, network->hidden_size,
                 network->weights1.data, network->weights1.count,
                 network->bias1.data, network->bias1.count, &layers[0], &error),
                 &error, network->name, "create/dense1");
    if (count >= 2) {
        check_status(cinder_layer_create_activation(CINDER_LAYER_RELU, network->hidden_size,
                     &layers[1], &error), &error, network->name, "create/relu");
    }
    if (count >= 3) {
        check_status(cinder_layer_create_dense(network->hidden_size, network->output_size,
                     network->weights2.data, network->weights2.count,
                     network->bias2.data, network->bias2.count, &layers[2], &error),
                     &error, network->name, "create/dense2");
    }
    if (count == 4) {
        check_status(cinder_layer_create_activation(CINDER_LAYER_SOFTMAX, network->output_size,
                     &layers[3], &error), &error, network->name, "create/softmax");
    }
}

static void check_parameters(cinder_layer_t *const *layers, size_t count)
{
    const numpy_network_fixture_t *network = &numpy_tiny_network;
    CHECK(memcmp(layers[0]->dense->weights->data, network->weights1.data,
                 network->weights1.count * sizeof(float)) == 0);
    CHECK(memcmp(layers[0]->dense->bias->data, network->bias1.data,
                 network->bias1.count * sizeof(float)) == 0);
    if (count >= 3) {
        CHECK(memcmp(layers[2]->dense->weights->data, network->weights2.data,
                     network->weights2.count * sizeof(float)) == 0);
        CHECK(memcmp(layers[2]->dense->bias->data, network->bias2.data,
                     network->bias2.count * sizeof(float)) == 0);
    }
}

static void check_output(const numpy_network_case_t *fixture, const char *stage,
                         const float *output, size_t count, numpy_vector_t reference,
                         bool probabilities)
{
    CHECK(compare_vector(&stats, fixture->name, stage, output, count, reference, true));
    CHECK(isnan(output[count])); /* The spare element is never part of the vector. */
    if (probabilities) {
        double total = 0;
        for (size_t i = 0; i < count; ++i) {
            CHECK(output[i] >= 0 && output[i] <= 1);
            total += output[i];
        }
        CHECK(fabs(total - 1.0) < 1e-6);
    }
}

static void check_layer_sequence(void)
{
    cinder_layer_t *layers[4] = {NULL, NULL, NULL, NULL};
    create_layers(layers, 4);
    for (size_t i = 0; i < COUNT(numpy_network_cases); ++i) {
        const numpy_network_case_t *fixture = &numpy_network_cases[i];
        float input[2];
        memcpy(input, fixture->input.data, sizeof(input));
        float hidden[] = {NAN, NAN, NAN, NAN, NAN};
        float output[] = {NAN, NAN, NAN};
        cinder_error_t error;

        check_status(cinder_layer_forward(layers[0], input, COUNT(input), hidden,
                     COUNT(hidden), &error), &error, fixture->name, layer_stages[0]);
        check_output(fixture, layer_stages[0], hidden, layers[0]->output_size,
                     fixture->dense1, false);
        check_status(cinder_layer_forward(layers[1], hidden, layers[0]->output_size, hidden,
                     COUNT(hidden), &error), &error, fixture->name, layer_stages[1]);
        check_output(fixture, layer_stages[1], hidden, layers[1]->output_size,
                     fixture->relu, false);
        check_status(cinder_layer_forward(layers[2], hidden, layers[1]->output_size, output,
                     COUNT(output), &error), &error, fixture->name, layer_stages[2]);
        check_output(fixture, layer_stages[2], output, layers[2]->output_size,
                     fixture->logits, false);
        check_status(cinder_layer_forward(layers[3], output, layers[2]->output_size, output,
                     COUNT(output), &error), &error, fixture->name, layer_stages[3]);
        check_output(fixture, layer_stages[3], output, layers[3]->output_size,
                     fixture->softmax, true);

        /* Each stage consumes Cinder's previous result, never the reference. */
        CHECK(memcmp(input, fixture->input.data, sizeof(input)) == 0);
        check_parameters(layers, 4);
    }
    for (size_t i = 0; i < 4; ++i) {
        cinder_layer_free(layers[i]);
    }
}

static void check_model_prefixes(void)
{
    /* Prefixes expose intermediate results through the real model API without
     * adding tracing or workspace access to production. Count four is the full
     * network, including its final output copy. Each model reuses its buffers
     * across all inputs, in reverse order to the direct layer sequence. */
    for (size_t count = 1; count <= 4; ++count) {
        cinder_layer_t *layers[4] = {NULL, NULL, NULL, NULL};
        create_layers(layers, count);
        cinder_layer_t *borrowed[4];
        memcpy(borrowed, layers, sizeof(borrowed));
        cinder_model_t *model = NULL;
        cinder_error_t error;
        check_status(cinder_model_create(layers, count, &model, &error),
                     &error, numpy_tiny_network.name, "create/model");
        for (size_t i = 0; i < count; ++i) {
            CHECK(layers[i] == NULL);
        }
        CHECK(cinder_model_input_size(model) == numpy_tiny_network.input_size);
        CHECK(cinder_model_output_size(model) == (count <= 2 ? 4 : 2));
        for (size_t i = COUNT(numpy_network_cases); i > 0; --i) {
            const numpy_network_case_t *fixture = &numpy_network_cases[i - 1];
            float input[2];
            memcpy(input, fixture->input.data, sizeof(input));
            float storage[] = {-101, NAN, NAN, NAN, NAN, NAN, -102};
            size_t outputs = cinder_model_output_size(model);
            /* Alternate exact capacity and one spare element. */
            size_t capacity = outputs + i % 2;
            check_status(cinder_model_run(model, input, COUNT(input), storage + 1,
                         capacity, &error), &error, fixture->name, model_stages[count - 1]);
            check_output(fixture, model_stages[count - 1], storage + 1, outputs,
                         reference_stage(fixture, count - 1), count == 4);
            for (size_t tail = outputs + 1; tail <= 5; ++tail) {
                CHECK(isnan(storage[tail]));
            }
            CHECK(storage[0] == -101 && storage[6] == -102);
            CHECK(memcmp(input, fixture->input.data, sizeof(input)) == 0);
            check_parameters(borrowed, count);
        }
        cinder_model_free(model);
    }
}

int main(void)
{
    check_fixture_shapes();
    check_layer_sequence();
    check_model_prefixes();
    printf("Tiny-network NumPy references passed: %zu inputs, four stages via layers and models, "
           "%zu vectors, %zu elements.\nMax absolute error: %.9g; "
           "max fraction of tolerance: %.9g (must be <= 1).\n",
           COUNT(numpy_network_cases), stats.comparisons, stats.elements,
           stats.largest_absolute_error, stats.largest_tolerance_fraction);
    return EXIT_SUCCESS;
}
