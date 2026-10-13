cmake_minimum_required(VERSION 3.16)

project(Cinder VERSION 0.1.0 LANGUAGES C)

include(CTest)

option(CINDER_ENABLE_ASAN "Enable AddressSanitizer for development" OFF)
option(CINDER_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for development" OFF)

if(CINDER_ENABLE_ASAN OR CINDER_ENABLE_UBSAN)
    if(NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
        message(FATAL_ERROR "Cinder sanitizer builds require a GCC or Clang toolchain.")
    endif()
endif()

function(cinder_configure_target target)
    set_target_properties(${target} PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED YES
        C_EXTENSIONS NO
    )
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)

    if(CINDER_ENABLE_ASAN OR CINDER_ENABLE_UBSAN)
        target_compile_options(${target} PRIVATE -fno-omit-frame-pointer)
    endif()
    if(CINDER_ENABLE_ASAN)
        target_compile_options(${target} PRIVATE -fsanitize=address)
        target_link_options(${target} PRIVATE -fsanitize=address)
    endif()
    if(CINDER_ENABLE_UBSAN)
        target_compile_options(${target} PRIVATE
            -fsanitize=undefined
            -fno-sanitize-recover=undefined
        )
        target_link_options(${target} PRIVATE -fsanitize=undefined)
    endif()
endfunction()

add_library(cinder_core STATIC src/tensor.c)
target_include_directories(cinder_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
cinder_configure_target(cinder_core)

add_executable(cinder src/cli.c)
target_link_libraries(cinder PRIVATE cinder_core)
target_compile_definitions(cinder PRIVATE CINDER_VERSION="${PROJECT_VERSION}-dev")
cinder_configure_target(cinder)

if(BUILD_TESTING)
    add_test(NAME cli_version
        COMMAND "${CMAKE_COMMAND}"
            "-DCINDER_EXECUTABLE=$<TARGET_FILE:cinder>"
            "-DEXPECTED_VERSION=${PROJECT_VERSION}-dev"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/check_version.cmake"
    )
    set_tests_properties(cli_version PROPERTIES TIMEOUT 10)

    add_executable(test_tensor tests/test_tensor.c)
    target_link_libraries(test_tensor PRIVATE cinder_core)
    cinder_configure_target(test_tensor)
    add_test(NAME tensor_allocation COMMAND test_tensor)

    # This test compiles tensor.c with allocator substitutions confined to its
    # translation unit, so it must not also link the normal cinder_core copy.
    add_executable(test_tensor_alloc_failures tests/test_tensor_alloc_failures.c)
    target_include_directories(test_tensor_alloc_failures PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
    )
    cinder_configure_target(test_tensor_alloc_failures)
    add_test(NAME tensor_allocation_failures COMMAND test_tensor_alloc_failures)

    set_tests_properties(tensor_allocation tensor_allocation_failures
        PROPERTIES TIMEOUT 10
    )
endif()
