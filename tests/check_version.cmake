execute_process(
    COMMAND "${CINDER_EXECUTABLE}" --version
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
)

if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "cinder --version failed (${result}): ${error}")
endif()

if(NOT "${output}" STREQUAL "Cinder ${EXPECTED_VERSION}\n")
    message(FATAL_ERROR
        "Expected exactly 'Cinder ${EXPECTED_VERSION}' followed by a newline; got: '${output}'"
    )
endif()

if(NOT "${error}" STREQUAL "")
    message(FATAL_ERROR "cinder --version wrote to standard error: ${error}")
endif()
