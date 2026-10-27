#include "cli_session.h"
#include "input.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static cinder_status_t preparation_error(cinder_cli_failure_t *failure,
    cinder_status_t status, const char *message)
{
    failure->detail.code = status;
    snprintf(failure->detail.message, sizeof(failure->detail.message), "%s", message);
    return status;
}

void cinder_cli_release(cinder_cli_session_t *session)
{
    if (session == NULL) return;
    if (session->output != NULL) free(session->output);
    if (session->input != NULL) free(session->input);
    if (session->model != NULL) cinder_model_free(session->model);
    *session = (cinder_cli_session_t){0};
}

cinder_status_t cinder_cli_prepare(const char *model_path, const char *input_path,
    cinder_cli_session_t *session, cinder_cli_failure_t *failure)
{
    cinder_cli_failure_t ignored;
    if (failure == NULL) failure = &ignored;
    *failure = (cinder_cli_failure_t){.stage = "inference"};
    if (session == NULL) {
        return preparation_error(failure, CINDER_ERROR_INVALID_ARGUMENT,
                                 "CLI session pointer must not be NULL.");
    }
    if (session->model != NULL || session->input != NULL || session->output != NULL ||
        session->input_size != 0 || session->output_size != 0) {
        return preparation_error(failure, CINDER_ERROR_INVALID_ARGUMENT,
                                 "CLI session must be empty before preparation.");
    }

    /* Publish only a complete session. Failure cleanup never changes the
     * destination or the borrowed diagnostic path/status/detail. */
    cinder_cli_session_t pending = {0};
    failure->stage = "model";
    failure->path = model_path;
    cinder_status_t status = cinder_model_load(model_path, &pending.model, &failure->detail);
    if (status != CINDER_OK) goto cleanup;

    failure->stage = "inference";
    failure->path = NULL;
    pending.input_size = cinder_model_input_size(pending.model);
    pending.output_size = cinder_model_output_size(pending.model);
    if (pending.input_size == 0 || pending.output_size == 0) {
        status = preparation_error(failure, CINDER_ERROR_INVALID_ARGUMENT,
                                  "Loaded model dimensions must be positive.");
        goto cleanup;
    }
    failure->stage = "input";
    failure->path = input_path;
    status = cinder_input_read(input_path, pending.input_size, &pending.input, &failure->detail);
    if (status != CINDER_OK) goto cleanup;

    failure->stage = "inference";
    failure->path = NULL;
    if (pending.output_size > SIZE_MAX / sizeof(float)) {
        status = preparation_error(failure, CINDER_ERROR_OVERFLOW,
                                  "Output byte count is not representable.");
        goto cleanup;
    }
    pending.output = malloc(pending.output_size * sizeof(float));
    if (pending.output == NULL) {
        status = preparation_error(failure, CINDER_ERROR_ALLOCATION,
                                  "Could not allocate output buffer.");
        goto cleanup;
    }
    *session = pending;
    failure->detail = (cinder_error_t){0};
    return CINDER_OK;

cleanup:
    cinder_cli_release(&pending);
    failure->detail.code = status;
    return status;
}
