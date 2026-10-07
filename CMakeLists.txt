cmake_minimum_required(VERSION 3.16)

project(Cinder VERSION 0.1.0 LANGUAGES C)

include(CTest)

option(CINDER_ENABLE_ASAN "Enable AddressSanitizer for development" OFF)
option(CINDER_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for development" OFF)

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

if(CINDER_ENABLE_ASAN OR CINDER_ENABLE_UBSAN)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        message(FATAL_ERROR "Cinder sanitizer builds require a GCC or Clang toolchain.")
    endif()

    target_compile_options(cinder PRIVATE -fno-omit-frame-pointer)
endif()

if(CINDER_ENABLE_ASAN)
    target_compile_options(cinder PRIVATE -fsanitize=address)
    target_link_options(cinder PRIVATE -fsanitize=address)
endif()

if(CINDER_ENABLE_UBSAN)
    target_compile_options(cinder PRIVATE
        -fsanitize=undefined
        -fno-sanitize-recover=undefined
    )
    target_link_options(cinder PRIVATE -fsanitize=undefined)
endif()

if(BUILD_TESTING)
    add_test(NAME cli_version
        COMMAND "${CMAKE_COMMAND}"
            "-DCINDER_EXECUTABLE=$<TARGET_FILE:cinder>"
            "-DEXPECTED_VERSION=${PROJECT_VERSION}-dev"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_version.cmake"
    )
    set_tests_properties(cli_version PROPERTIES TIMEOUT 10)
endif()
