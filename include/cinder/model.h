#ifndef CINDER_MODEL_H
#define CINDER_MODEL_H

#include <stddef.h>

#include "cinder/layer.h"

/*
 * Contract and declarations only; model functions are not implemented yet.
 * A model executes a fixed, nonempty sequence of layers on one float32 vector.
 * Its representation is private: it owns the layer list, every layer, and two
 * reusable working buffers. Public size accessors expose immutable dimensions.
 * See docs/MODEL.md for ownership, construction order, and execution examples.
 */
typedef struct cinder_model cinder_model_t;

/*
 * Construct from an ordered, writable array of uniquely owned layer pointers.
 * Each non-NULL entry must be a live, intact constructor-created layer, not
 * owned by another model. Distinct entries must not share owned payloads.
 * Pointer validity and exclusive ownership are caller preconditions.
 *
 * On success, transfer ALL layers to the model, set every supplied array slot
 * to NULL, and publish *out_model. The model copies the pointer list; it does
 * not retain or own the caller's array and does not copy layer parameters.
 * Other aliases become borrowed, read-only references: never free, mutate,
 * independently execute, or transfer an adopted layer again. They expire on
 * model destruction. The ordered topology and dimensions cannot be changed.
 *
 * On any failure, leave the array and its layers unchanged and caller-owned,
 * release only new model allocations, and leave *out_model NULL. No partial
 * ownership transfer. The caller can fix/retry the list or free its layers.
 * The output slot must not already own a live model. Array storage, output slot,
 * optional error storage, and all layer-owned storage must be disjoint.
 *
 * Validation/allocation order:
 * 1. NULL out_model -> INVALID_ARGUMENT; otherwise clear *out_model.
 * 2. NULL layers or layer_count == 0 -> INVALID_ARGUMENT (no empty identity).
 * 3. layer_count > SIZE_MAX / sizeof(cinder_layer_t *) -> OVERFLOW.
 * 4. In ascending index order, call cinder_layer_validate on each entry and
 *    propagate its first error; then reject a pointer equal to any earlier
 *    entry with INVALID_ARGUMENT. NULL entries fail layer validation.
 * 5. Check connections in ascending order: previous output_size must equal
 *    next input_size; the first mismatch -> DIMENSION_MISMATCH.
 * 6. Derive external dimensions from first input/last output and working-buffer
 *    capacity from the largest input or output size of ANY layer. Each float
 *    byte extent must fit size_t (already checked by layer validation).
 * 7. Allocate the model, private pointer array, and two separate float buffers;
 *    allocation failure -> ALLOCATION with full rollback and no layer frees.
 * 8. Only after all allocations succeed, copy pointers, clear source slots,
 *    and publish the model. No remaining operation may fail during transfer.
 * No allocation occurs before all validation succeeds. No implicit activation,
 * reordering, padding, broadcasting, or dimension inference repairs a mismatch.
 */
cinder_status_t cinder_model_create(
    cinder_layer_t **layers,
    size_t layer_count,
    cinder_model_t **out_model,
    cinder_error_t *error
);

/*
 * Return float element counts: first layer's input and last layer's output.
 * A live model always has positive sizes. NULL returns zero, without error
 * storage or allocation. Non-NULL pointers must identify live models.
 * Accessors read immutable metadata; they may run during inference, but never
 * race destruction. They do not expose the layers or working buffers.
 */
size_t cinder_model_input_size(const cinder_model_t *model);
size_t cinder_model_output_size(const cinder_model_t *model);

/*
 * Run the fixed sequence on one input vector without allocations or frees.
 * model must be a live constructor-created object. It is non-const because
 * inference changes private working buffers, not topology or parameters.
 *
 * Order/status: NULL model/input/output -> INVALID_ARGUMENT;
 * input_count != model input size -> DIMENSION_MISMATCH;
 * output_capacity < model output size -> BUFFER_TOO_SMALL;
 * any non-finite input -> NONFINITE. Zero counts follow the same rules.
 * Then call cinder_layer_forward in list order, always into alternating private
 * buffers, using each layer's logical input count. Stop on the first failure
 * and preserve its status/error details. Copy exactly the model output size to
 * caller output ONLY after every layer succeeds. A single layer uses this
 * same path. Connections have already been validated during construction.
 *
 * Every returned failure leaves ALL caller output unchanged, including numeric
 * failures inside a layer. Private buffers may hold partial results; their
 * contents are unspecified and never exposed. Success writes only the logical
 * output; spare capacity is untouched on every outcome. Output need not be
 * initialized. Input and parameters remain unchanged; no caller buffers are
 * retained. Repeated calls, including after failure, require no reset and do
 * not depend on stale private-buffer contents or previous inputs/outputs.
 *
 * The caller supplies input_count readable and output_capacity writable floats.
 * Input/output ranges must be disjoint, including their full advertised extents;
 * disjoint slices of one array are allowed. Both ranges and error storage must
 * be disjoint from each other and all model-owned storage. Overlap, accessible
 * extents, object lifetime, and synchronization are caller preconditions, not
 * runtime-diagnosed errors.
 *
 * At most ONE run may be active on an instance: no simultaneous or reentrant
 * calls, even with different caller buffers. Serialize the whole call with an
 * external lock, or use separately constructed models with separate owners.
 * There is no internal lock, busy flag/status, or concurrency detection. Never
 * destroy the model while any call uses it. Independent models may run at once
 * with separate writable output/error storage and unchanged readable inputs.
 *
 * All status-returning functions follow error.h: optional error details,
 * matching code, empty message on success, bounded null-terminated failure
 * message, and no diagnostic allocations, printing, or process termination.
 */
cinder_status_t cinder_model_run(
    cinder_model_t *model,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_capacity,
    cinder_error_t *error
);

/*
 * Free every adopted layer with cinder_layer_free, both private working buffers,
 * the private pointer array, and the model. NULL is a no-op. No allocations.
 * A non-NULL model must be live, exclusively accessible, and freed exactly once.
 * The caller's pointer is not reset; borrowed layer/parameter aliases expire.
 */
void cinder_model_free(cinder_model_t *model);

#endif /* CINDER_MODEL_H */
