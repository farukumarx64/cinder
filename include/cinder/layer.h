#ifndef CINDER_LAYER_H
#define CINDER_LAYER_H

#include <stddef.h>

#include "cinder/dense.h"

/* These tags describe runtime objects, not serialized model-format IDs. */
typedef enum {
    CINDER_LAYER_INVALID = 0,
    CINDER_LAYER_DENSE,
    CINDER_LAYER_RELU,
    CINDER_LAYER_SOFTMAX
} cinder_layer_type_t;

/*
 * All sizes count floats and must be positive and byte-representable.
 * DENSE: dense is owned, live, and intact; sizes equal its input/output sizes.
 * RELU/SOFTMAX: input_size == output_size and dense == NULL; no parameters.
 *
 * The type tag selects behavior through switch dispatch. A single typed payload
 * pointer suffices because only Dense has parameters; no void * or vtable.
 * After construction, callers treat all fields and nested Dense data as
 * read-only. Do not copy owning layers by value or free their payloads directly.
 * An executable/owning layer comes from a constructor and is freed once by its
 * current owner with cinder_layer_free. model.h defines transfer to a model on
 * successful model construction; its functions are not implemented yet.
 */
typedef struct {
    cinder_layer_type_t type;
    size_t input_size;
    size_t output_size;
    cinder_dense_t *dense;
} cinder_layer_t;

/*
 * Create an owning Dense layer from caller-owned parameter arrays. When
 * out_layer is non-NULL, clear *out_layer before other validation; NULL out_layer
 * returns INVALID_ARGUMENT. The slot must not already own a live object.
 * Delegate parameter validation/copying to cinder_dense_create, preserving its
 * validation order and statuses, then allocate the common layer wrapper. If
 * wrapper allocation fails, free the new Dense and return ALLOCATION.
 * Publish *out_layer only on success; every failure leaves it NULL and releases
 * all partial allocations. The original arrays remain caller-owned/unchanged.
 * The output slot and optional error object must not overlap each other or
 * either source range. Source ranges may overlap each other (read-only).
 */
cinder_status_t cinder_layer_create_dense(
    size_t input_size,
    size_t output_size,
    const float *weights,
    size_t weight_count,
    const float *bias,
    size_t bias_count,
    cinder_layer_t **out_layer,
    cinder_error_t *error
);

/*
 * Create RELU or SOFTMAX with input_size == output_size == size and dense NULL.
 * Validation order: NULL out_layer -> INVALID_ARGUMENT; otherwise clear it;
 * unsupported type (including DENSE/INVALID) or size zero -> INVALID_ARGUMENT;
 * size > SIZE_MAX / sizeof(float) -> OVERFLOW; allocation failure -> ALLOCATION.
 * Allocate only the layer wrapper, after validation. Every failure leaves the
 * output NULL. Output/error storage must be disjoint; the slot must not already
 * own a live layer. There are no activation parameter arrays or scratch buffers.
 */
cinder_status_t cinder_layer_create_activation(
    cinder_layer_type_t type,
    size_t size,
    cinder_layer_t **out_layer,
    cinder_error_t *error
);

/*
 * Validate metadata without allocating, modifying the layer, or scanning data.
 * The descriptor must be readable and fully initialized. Any non-NULL Dense
 * payload must be a live, intact cinder_dense_t; pointer validity is a caller
 * precondition. Validation alone does not create an owner or transfer ownership.
 *
 * Order/status: NULL layer or unknown/INVALID tag -> INVALID_ARGUMENT;
 * zero size -> INVALID_ARGUMENT; either vector byte extent overflow -> OVERFLOW;
 * missing DENSE payload or non-NULL activation payload -> INVALID_ARGUMENT;
 * Dense/common dimension disagreement or unequal activation sizes ->
 * DIMENSION_MISMATCH. Success -> OK. This checks one layer, not its neighbors.
 * Optional error storage must not overlap the descriptor or owned storage.
 */
cinder_status_t cinder_layer_validate(
    const cinder_layer_t *layer,
    cinder_error_t *error
);

/*
 * Release the owned Dense (DENSE only) and wrapper. NULL is a no-op. Non-NULL
 * layers must be live, intact owners returned by a constructor. No allocation;
 * no caller-pointer reset; all aliases/borrowed pointers expire on free.
 */
void cinder_layer_free(cinder_layer_t *layer);

/*
 * Execute one vector through a switch on type, with no allocations or frees.
 * layer must be a live constructor-created owner with its invariants intact.
 * Validation order: NULL layer/input/output -> INVALID_ARGUMENT; metadata via
 * cinder_layer_validate; exact input_count -> DIMENSION_MISMATCH on mismatch;
 * output_capacity >= output_size -> BUFFER_TOO_SMALL otherwise; finite sources.
 * All validation completes before writes; validation failure preserves output.
 *
 * DENSE delegates to cinder_dense_forward. RELU/SOFTMAX validate all input values
 * before copying exactly input_size floats into output, then invoke the existing
 * in-place activation on output_size floats. Return kernel status/error details.
 * Output need not be initialized. Write exactly output_size results on success;
 * spare capacity is untouched on every outcome. Arithmetic failure may leave
 * partial Dense/Softmax output: discard the result. Input and parameters remain
 * unchanged; failure does not invalidate a layer or retain caller buffers.
 *
 * For ALL types, full advertised output storage must be disjoint from input,
 * layer/Dense-owned storage, and optional error storage. Exact input/output
 * aliasing is also forbidden for activations. Read-only input may borrow Dense
 * parameter data; disjoint slices of one allocation are allowed. Error storage
 * must be disjoint from all sources/owners. These are caller preconditions, not
 * runtime overlap checks. Do not modify/free a layer or source during a call.
 * All status-returning functions follow error.h, including optional details,
 * bounded null-terminated failure messages, and cleared details on success.
 */
cinder_status_t cinder_layer_forward(
    const cinder_layer_t *layer,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);

#endif /* CINDER_LAYER_H */
