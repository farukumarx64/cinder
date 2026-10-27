#ifndef CINDER_CLI_SESSION_H
#define CINDER_CLI_SESSION_H

#include "cinder/model.h"

/* CLI-private ownership, not a public runtime API. Initialize with {0}.
 * A prepared session owns the model and both caller buffers; do not shallow-copy
 * it or replace its pointers. Model-owned layers/workspaces remain opaque.
 * Output is uninitialized until a successful cinder_model_run. */
typedef struct {
    cinder_model_t *model;
    float *input;
    float *output;
    size_t input_size;
    size_t output_size;
} cinder_cli_session_t;

/* Context for the existing CLI diagnostic formatter. stage is static text;
 * path borrows a supplied argument (or is NULL for inference/setup failures).
 * Neither string is owned. detail.code matches the returned status. */
typedef struct {
    const char *stage;
    const char *path;
    cinder_error_t detail;
} cinder_cli_failure_t;

/* Paths have already passed command-line syntax validation. Load once, read
 * and close input once, then allocate a disjoint output buffer. No inference,
 * clock queries, or printing. On success, publish all ownership to session and
 * clear optional failure details (stage="inference", path=NULL).
 * On operational failure, release partial ownership and leave session empty.
 * A NULL or nonempty destination is rejected before I/O; a nonempty session is
 * preserved. Release it before preparing again. failure may be NULL and, when
 * supplied, must be disjoint from session and all owned storage. */
cinder_status_t cinder_cli_prepare(const char *model_path, const char *input_path,
    cinder_cli_session_t *session, cinder_cli_failure_t *failure);

/* Free output, input, and model; reset every field. NULL/empty are safe no-ops.
 * After release the same object may be prepared again. No active inference may
 * use the session during preparation or release. */
void cinder_cli_release(cinder_cli_session_t *session);

#endif
