#ifndef CINDER_ERROR_H
#define CINDER_ERROR_H

/* Status codes currently needed by the tensor API. */
typedef enum {
    CINDER_OK = 0,
    CINDER_ERROR_INVALID_ARGUMENT,
    CINDER_ERROR_OVERFLOW,
    CINDER_ERROR_ALLOCATION,
    CINDER_ERROR_OUT_OF_BOUNDS,
    CINDER_ERROR_NONFINITE
} cinder_status_t;

#define CINDER_ERROR_MESSAGE_CAPACITY 128

/*
 * Optional, caller-owned error details. Functions returning cinder_status_t
 * accept NULL when details are not needed. Otherwise they set code to the
 * returned status and write a null-terminated message (empty on success).
 * Messages are for people; callers should branch on the status code.
 * No error-detail allocation or cleanup is required.
 */
typedef struct {
    cinder_status_t code;
    char message[CINDER_ERROR_MESSAGE_CAPACITY];
} cinder_error_t;

#endif /* CINDER_ERROR_H */
