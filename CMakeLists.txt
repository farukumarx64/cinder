cmake_minimum_required(VERSION 3.16)

project(Cinder VERSION 0.1.0 LANGUAGES C)

include(CTest)

option(CINDER_ENABLE_ASAN "Enable AddressSanitizer for development" OFF)
option(CINDER_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for development" OFF)
option(CINDER_ENABLE_NUMPY_TESTS "Also test freshly generated NumPy references" OFF)

if(CINDER_ENABLE_NUMPY_TESTS AND NOT BUILD_TESTING)
    message(FATAL_ERROR "CINDER_ENABLE_NUMPY_TESTS requires BUILD_TESTING=ON.")
endif()

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

add_library(cinder_core STATIC src/tensor.c src/dense.c src/layer.c src/model.c)
target_include_directories(cinder_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
cinder_configure_target(cinder_core)
# Unix toolchains provide exp through libm (part of the system library on macOS).
if(UNIX)
    target_link_libraries(cinder_core PUBLIC m)
endif()

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

    add_executable(test_dense tests/test_dense.c)
    target_link_libraries(test_dense PRIVATE cinder_core)
    cinder_configure_target(test_dense)
    add_test(NAME dense_allocation COMMAND test_dense)

    add_executable(test_dense_forward tests/test_dense_forward.c)
    target_include_directories(test_dense_forward PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_dense_forward PRIVATE cinder_core)
    cinder_configure_target(test_dense_forward)
    add_test(NAME dense_forward COMMAND test_dense_forward)

    add_executable(test_layer tests/test_layer.c)
    target_include_directories(test_layer PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_layer PRIVATE cinder_core)
    cinder_configure_target(test_layer)
    add_test(NAME layer_interface COMMAND test_layer)

    add_executable(test_model tests/test_model.c)
    target_link_libraries(test_model PRIVATE cinder_core)
    cinder_configure_target(test_model)
    add_test(NAME model_allocation COMMAND test_model)

    add_executable(test_tensor_access tests/test_tensor_access.c)
    target_link_libraries(test_tensor_access PRIVATE cinder_core)
    cinder_configure_target(test_tensor_access)
    add_test(NAME tensor_access COMMAND test_tensor_access)

    add_executable(test_tensor_matvec tests/test_tensor_matvec.c)
    target_link_libraries(test_tensor_matvec PRIVATE cinder_core)
    cinder_configure_target(test_tensor_matvec)
    add_test(NAME tensor_matvec COMMAND test_tensor_matvec)

    add_executable(test_tensor_bias tests/test_tensor_bias.c)
    target_link_libraries(test_tensor_bias PRIVATE cinder_core)
    cinder_configure_target(test_tensor_bias)
    add_test(NAME tensor_bias COMMAND test_tensor_bias)

    add_executable(test_tensor_relu tests/test_tensor_relu.c)
    target_link_libraries(test_tensor_relu PRIVATE cinder_core)
    cinder_configure_target(test_tensor_relu)
    add_test(NAME tensor_relu COMMAND test_tensor_relu)

    add_executable(test_tensor_softmax tests/test_tensor_softmax.c)
    target_link_libraries(test_tensor_softmax PRIVATE cinder_core)
    cinder_configure_target(test_tensor_softmax)
    add_test(NAME tensor_softmax COMMAND test_tensor_softmax)

    add_executable(test_tensor_numpy tests/test_tensor_numpy.c)
    target_include_directories(test_tensor_numpy PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_tensor_numpy PRIVATE cinder_core)
    cinder_configure_target(test_tensor_numpy)
    add_test(NAME tensor_numpy_fixtures COMMAND test_tensor_numpy)

    if(CINDER_ENABLE_NUMPY_TESTS)
        find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import numpy; print(numpy.__version__)"
            RESULT_VARIABLE cinder_numpy_status
            OUTPUT_VARIABLE cinder_numpy_version
            ERROR_VARIABLE cinder_numpy_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT cinder_numpy_status EQUAL 0)
            message(FATAL_ERROR "NumPy is required for reference generation. Install tests/requirements-numpy.txt with ${Python3_EXECUTABLE}. ${cinder_numpy_error}")
        endif()
        message(STATUS "Generating reference fixtures with NumPy ${cinder_numpy_version}")
        set(cinder_numpy_header "${PROJECT_BINARY_DIR}/numpy-fixtures/numpy_fixtures.h")
        add_custom_command(
            OUTPUT "${cinder_numpy_header}"
            COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/generate_numpy_fixtures.py"
                --output "${cinder_numpy_header}"
            DEPENDS tests/generate_numpy_fixtures.py tests/requirements-numpy.txt
            COMMENT "Generating independent float32 NumPy references"
            VERBATIM
        )
        add_executable(test_tensor_numpy_generated tests/test_tensor_numpy.c "${cinder_numpy_header}")
        target_include_directories(test_tensor_numpy_generated PRIVATE "${PROJECT_BINARY_DIR}/numpy-fixtures")
        target_link_libraries(test_tensor_numpy_generated PRIVATE cinder_core)
        cinder_configure_target(test_tensor_numpy_generated)
        add_test(NAME tensor_numpy_generated COMMAND test_tensor_numpy_generated)
        set_tests_properties(tensor_numpy_generated PROPERTIES TIMEOUT 10)
    endif()

    # This test compiles tensor.c with allocator substitutions confined to its
    # translation unit, so it must not also link the normal cinder_core copy.
    add_executable(test_tensor_alloc_failures tests/test_tensor_alloc_failures.c)
    target_include_directories(test_tensor_alloc_failures PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
    )
    cinder_configure_target(test_tensor_alloc_failures)
    if(UNIX)
        target_link_libraries(test_tensor_alloc_failures PRIVATE m)
    endif()
    add_test(NAME tensor_allocation_failures COMMAND test_tensor_alloc_failures)

    # Substitute allocators in both Dense and tensor code to exercise the whole
    # ownership chain. As above, do not link another copy from cinder_core.
    add_executable(test_dense_alloc_failures tests/test_dense_alloc_failures.c)
    target_include_directories(test_dense_alloc_failures PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
    )
    cinder_configure_target(test_dense_alloc_failures)
    if(UNIX)
        target_link_libraries(test_dense_alloc_failures PRIVATE m)
    endif()
    add_test(NAME dense_allocation_failures COMMAND test_dense_alloc_failures)

    # Cover the common wrapper and its complete Dense/tensor ownership chain.
    add_executable(test_layer_alloc_failures tests/test_layer_alloc_failures.c)
    target_include_directories(test_layer_alloc_failures PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
    )
    cinder_configure_target(test_layer_alloc_failures)
    if(UNIX)
        target_link_libraries(test_layer_alloc_failures PRIVATE m)
    endif()
    add_test(NAME layer_allocation_failures COMMAND test_layer_alloc_failures)

    # Track the full model/layer/tensor ownership chain and private workspaces.
    # Allocator substitutions remain confined to this test translation unit.
    add_executable(test_model_alloc_failures tests/test_model_alloc_failures.c)
    target_include_directories(test_model_alloc_failures PRIVATE
        "${PROJECT_SOURCE_DIR}/include"
    )
    cinder_configure_target(test_model_alloc_failures)
    if(UNIX)
        target_link_libraries(test_model_alloc_failures PRIVATE m)
    endif()
    add_test(NAME model_allocation_failures COMMAND test_model_alloc_failures)

    set_tests_properties(
        tensor_allocation tensor_access tensor_matvec tensor_bias tensor_relu
        tensor_softmax tensor_numpy_fixtures tensor_allocation_failures
        dense_allocation dense_allocation_failures dense_forward
        layer_interface layer_allocation_failures
        model_allocation model_allocation_failures
        PROPERTIES TIMEOUT 10
    )
endif()
