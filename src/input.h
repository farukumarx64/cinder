#ifndef CINDER_INPUT_H
#define CINDER_INPUT_H

#include "cinder/error.h"
#include <stddef.h>

/* CLI-private reader: success transfers a malloc-owned vector to the caller.
 * Failure leaves *values NULL; the stream is closed before either return. */
cinder_status_t cinder_input_read(const char *path, size_t count,
                                  float **values, cinder_error_t *error);

#endif
