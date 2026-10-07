cmake_minimum_required(VERSION 3.16)

project(Cinder VERSION 0.1.0 LANGUAGES C)

include(CTest)

add_executable(cinder src/cli.c)

target_compile_definitions(cinder PRIVATE
    CINDER_VERSION="${PROJECT_VERSION}-dev"
)

set_target_properties(cinder PROPERTIES
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
)

target_compile_options(cinder PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(BUILD_TESTING)
    add_test(NAME cli_version
        COMMAND "${CMAKE_COMMAND}"
            "-DCINDER_EXECUTABLE=$<TARGET_FILE:cinder>"
            "-DEXPECTED_VERSION=${PROJECT_VERSION}-dev"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_version.cmake"
    )
    set_tests_properties(cli_version PROPERTIES TIMEOUT 10)
endif()
