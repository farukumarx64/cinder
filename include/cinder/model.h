#ifndef CINDER_MODEL_H
#define CINDER_MODEL_H

#include <stddef.h>

#include "cinder/layer.h"

/*
 * Construction, connection validation, size accessors, file loading, inference,
 * and cleanup are implemented.
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
 * Load one complete .cnd v1 model from a seekable binary file. See docs/LOADER.md
 * for the validation sequence, status mapping, and constructor/cleanup rules.
 *
 * NULL out_model -> INVALID_ARGUMENT; otherwise clear *out_model first.
 * NULL/empty path -> INVALID_ARGUMENT. A non-NULL path must be a readable,
 * unchanged NUL-terminated string for the call. The writable output slot must
 * not already own a live model; it, error storage, and path must be disjoint.
 * Relative paths use the process working directory; suffix is not checked.
 * No stdin convention, caller FILE *, or memory-buffer loading API is provided.
 *
 * Check host binary32 representation, open once with "rb", and establish the
 * size on that stream before model-data allocations. Enforce MODEL_FORMAT.md's
 * v1 bounds and validate fields before dependent arithmetic, allocation, or
 * reads. Check every read, every layer connection, finite parameter bits, and
 * exact EOF. The caller must keep file contents stable during this blocking
 * call; there is no file lock or snapshot guarantee. The loader never writes
 * to the file, executes layers, retains path/file bytes, or prints diagnostics.
 *
 * FORMAT: bad magic/record ID, zero dimensions/count, wrong payload length,
 * truncation, or trailing bytes. UNSUPPORTED: version != 1 or unsupported host
 * encoding. LIMIT: .cnd layer/vector/parameter/file cap exceeded. IO: open,
 * seek/size, read, or close failure (including non-seekable input). OVERFLOW:
 * host counts/extents cannot be represented. DIMENSION_MISMATCH: header/first
 * layer, adjacent layers, or activation lengths disagree. NONFINITE: NaN/Inf
 * parameter. ALLOCATION: loader/constructor allocation failure. Constructors'
 * returned errors propagate. Short reads with ferror -> IO, clean early EOF
 * -> FORMAT. Preserve the first error; cleanup must not overwrite it.
 *
 * The loader owns its temporary pointer list, parameter staging, and every
 * constructed layer until model creation succeeds. Decode one Dense payload
 * into aligned float staging, then use cinder_layer_create_dense to copy it;
 * release staging after the call. Map file activation IDs explicitly and call
 * cinder_layer_create_activation. Never build runtime structs from raw bytes.
 * Close the file successfully AFTER exact EOF and BEFORE cinder_model_create.
 * Call model_create into a private candidate; it adopts all layers and clears
 * the temporary slots only on success. Release the empty temporary list and
 * publish *out_model only when no fallible work remains.
 *
 * Every returned failure leaves *out_model NULL and frees all partial loader
 * allocations/layers; any opened stream gets exactly one fclose attempt. A
 * close error prevents success, but does not replace an earlier failure. Do
 * not retry fclose or reuse the stream afterward, even if fclose failed.
 * On success the caller exclusively owns a normal model, including its layers,
 * parameter copies, and two work buffers, and frees it with cinder_model_free.
 * The file may then be changed/removed without affecting the model. Repeated
 * loads own independent storage. Concurrent loads need distinct output/error
 * storage; loaded models obey the existing run/free synchronization contract.
 * Optional error details follow error.h, with bounded messages and no diagnostic
 * allocation. No global last-error or errno value is part of this API contract.
 */
cinder_status_t cinder_model_load(
    const char *path,
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
 * Copy input to the first private buffer, then call cinder_layer_forward in list
 * order using each layer's logical input count. Dense writes the other buffer;
 * ReLU/Softmax update the current buffer in place. Stop on the first failure
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
