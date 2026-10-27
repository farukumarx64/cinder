cmake_minimum_required(VERSION 3.16)

project(Cinder VERSION 0.1.0 LANGUAGES C)

include(CTest)

option(CINDER_ENABLE_ASAN "Enable AddressSanitizer for development" OFF)
option(CINDER_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer for development" OFF)
option(CINDER_ENABLE_NUMPY_TESTS "Also test freshly generated NumPy references" OFF)
option(CINDER_ENABLE_EXPORTER_TESTS "Also test Python model validation, export, and the C round trip" OFF)
option(CINDER_ENABLE_MNIST_TESTS "Also test Python MNIST training, export, fixtures, and C agreement offline" OFF)

if(CINDER_ENABLE_NUMPY_TESTS AND NOT BUILD_TESTING)
    message(FATAL_ERROR "CINDER_ENABLE_NUMPY_TESTS requires BUILD_TESTING=ON.")
endif()

if(CINDER_ENABLE_EXPORTER_TESTS AND NOT BUILD_TESTING)
    message(FATAL_ERROR "CINDER_ENABLE_EXPORTER_TESTS requires BUILD_TESTING=ON.")
endif()

if(CINDER_ENABLE_MNIST_TESTS AND NOT BUILD_TESTING)
    message(FATAL_ERROR "CINDER_ENABLE_MNIST_TESTS requires BUILD_TESTING=ON.")
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

add_library(cinder_core STATIC src/tensor.c src/dense.c src/layer.c src/model.c src/loader.c)
target_include_directories(cinder_core PUBLIC "${PROJECT_SOURCE_DIR}/include")
cinder_configure_target(cinder_core)
# Unix toolchains provide exp through libm (part of the system library on macOS).
if(UNIX)
    target_link_libraries(cinder_core PUBLIC m)
endif()

add_executable(cinder src/cli.c src/cli_session.c src/input.c)
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

    add_executable(test_input tests/test_input.c)
    target_include_directories(test_input PRIVATE "${PROJECT_SOURCE_DIR}/include")
    cinder_configure_target(test_input)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/input-test-files")
    add_test(NAME cli_input COMMAND test_input "${PROJECT_BINARY_DIR}/input-test-files/input.bin")

    add_executable(test_cli tests/test_cli.c)
    target_include_directories(test_cli PRIVATE "${PROJECT_SOURCE_DIR}/include")
    target_compile_definitions(test_cli PRIVATE CINDER_VERSION="${PROJECT_VERSION}-dev")
    if(UNIX)
        target_link_libraries(test_cli PRIVATE m)
    endif()
    cinder_configure_target(test_cli)
    add_test(NAME cli_audit COMMAND test_cli)

    # Audit the actual setup/reader/runtime with test-only allocation and I/O
    # wrappers; do not link a second runtime copy from cinder_core.
    add_executable(test_cli_setup tests/test_cli_setup.c)
    target_include_directories(test_cli_setup PRIVATE
        "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tests/fixtures")
    if(UNIX)
        target_link_libraries(test_cli_setup PRIVATE m)
    endif()
    cinder_configure_target(test_cli_setup)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cli-setup-files")
    add_test(NAME cli_setup COMMAND test_cli_setup "${PROJECT_SOURCE_DIR}"
        "${PROJECT_BINARY_DIR}/cli-setup-files")
    set_tests_properties(cli_setup PROPERTIES TIMEOUT 60)

    # Supported CLI hosts are macOS/Linux; fork/exec tests use committed files
    # and never require Python, a dataset cache or a network connection.
    if(UNIX)
        add_executable(test_cli_process tests/test_cli_process.c)
        target_link_libraries(test_cli_process PRIVATE m)
        cinder_configure_target(test_cli_process)
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/cli-test-files")
        add_test(NAME cli_process COMMAND test_cli_process "$<TARGET_FILE:cinder>"
            "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/cli-test-files")
        set_tests_properties(cli_process PROPERTIES TIMEOUT 60)
    endif()
    set_tests_properties(cli_input cli_audit PROPERTIES TIMEOUT 10)

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

    add_executable(test_model_run tests/test_model_run.c)
    target_link_libraries(test_model_run PRIVATE cinder_core)
    cinder_configure_target(test_model_run)
    add_test(NAME model_run COMMAND test_model_run)

    add_executable(test_model_load tests/test_model_load.c)
    target_include_directories(test_model_load PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_model_load PRIVATE cinder_core)
    cinder_configure_target(test_model_load)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/loader-test-files")
    add_test(NAME model_load COMMAND test_model_load
        "${PROJECT_SOURCE_DIR}/tests/fixtures/tiny/tiny.cnd"
        "${PROJECT_BINARY_DIR}/loader-test-files/model.bin"
    )
    set_tests_properties(model_load PROPERTIES TIMEOUT 60)

    # Compile the real loader and ownership chain with test-only allocation and
    # stdio substitutions. Do not link a second runtime copy from cinder_core.
    add_executable(test_loader_failures tests/test_loader_failures.c)
    target_include_directories(test_loader_failures PRIVATE
        "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tests/fixtures"
    )
    cinder_configure_target(test_loader_failures)
    if(UNIX)
        target_link_libraries(test_loader_failures PRIVATE m)
    endif()
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/loader-audit-files")
    add_test(NAME loader_failures COMMAND test_loader_failures
        "${PROJECT_SOURCE_DIR}/tests/fixtures/tiny/tiny.cnd"
        "${PROJECT_BINARY_DIR}/loader-audit-files/model.bin"
    )
    set_tests_properties(loader_failures PROPERTIES TIMEOUT 60)

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

    add_executable(test_model_numpy tests/test_model_numpy.c)
    target_include_directories(test_model_numpy PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_model_numpy PRIVATE cinder_core)
    cinder_configure_target(test_model_numpy)
    add_test(NAME model_numpy_fixtures COMMAND test_model_numpy)

    # Always consume committed .cnd/input/reference fixtures without Python.
    # The optional exporter test below reuses this executable on fresh exports.
    add_executable(test_model_round_trip tests/test_model_round_trip.c)
    target_include_directories(test_model_round_trip PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_model_round_trip PRIVATE cinder_core)
    cinder_configure_target(test_model_round_trip)
    add_test(NAME model_round_trip_fixtures COMMAND test_model_round_trip
        "${PROJECT_SOURCE_DIR}/tests/fixtures/tiny"
    )
    set_tests_properties(model_round_trip_fixtures PROPERTIES TIMEOUT 10)

    # The committed MNIST model and all saved stages are checked without Python
    # or a dataset download. Test-only prefix files live in the build directory.
    add_executable(test_mnist_inference tests/test_mnist_inference.c)
    target_include_directories(test_mnist_inference PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_mnist_inference PRIVATE cinder_core)
    cinder_configure_target(test_mnist_inference)
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/mnist-inference-files")
    add_test(NAME mnist_inference_fixtures COMMAND test_mnist_inference
        "${PROJECT_SOURCE_DIR}/examples/mnist/mnist.cnd"
        "${PROJECT_SOURCE_DIR}/tests/fixtures/mnist"
        "${PROJECT_BINARY_DIR}/mnist-inference-files"
    )
    set_tests_properties(mnist_inference_fixtures PROPERTIES TIMEOUT 30)

    # Instrument the real ownership chain only in this test translation unit.
    # Its committed fixtures require no Python or dataset cache.
    add_executable(test_mnist_lifecycle_audit tests/test_mnist_lifecycle.c)
    target_compile_definitions(test_mnist_lifecycle_audit PRIVATE CINDER_MNIST_AUDIT=1)
    target_include_directories(test_mnist_lifecycle_audit PRIVATE
        "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/tests/fixtures")
    cinder_configure_target(test_mnist_lifecycle_audit)
    if(UNIX)
        target_link_libraries(test_mnist_lifecycle_audit PRIVATE m)
    endif()
    file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/mnist-lifecycle-audit-files")
    add_test(NAME mnist_lifecycle_audit COMMAND test_mnist_lifecycle_audit
        "${PROJECT_SOURCE_DIR}/examples/mnist/mnist.cnd"
        "${PROJECT_SOURCE_DIR}/tests/fixtures/mnist"
        "${PROJECT_BINARY_DIR}/mnist-lifecycle-audit-files/model.bin")
    set_tests_properties(mnist_lifecycle_audit PROPERTIES TIMEOUT 60)

    # Test caller-managed concurrency on the supported macOS/Linux platforms.
    # Threads are a test dependency, not a dependency of the runtime library.
    if(UNIX)
        find_package(Threads REQUIRED)
        add_executable(test_mnist_lifecycle tests/test_mnist_lifecycle.c)
        target_include_directories(test_mnist_lifecycle PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
        target_link_libraries(test_mnist_lifecycle PRIVATE cinder_core Threads::Threads)
        cinder_configure_target(test_mnist_lifecycle)
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/mnist-lifecycle-files")
        add_test(NAME mnist_lifecycle COMMAND test_mnist_lifecycle
            "${PROJECT_SOURCE_DIR}/examples/mnist/mnist.cnd"
            "${PROJECT_SOURCE_DIR}/tests/fixtures/mnist"
            "${PROJECT_BINARY_DIR}/mnist-lifecycle-files/model.bin")
        set_tests_properties(mnist_lifecycle PROPERTIES TIMEOUT 60)
        add_executable(test_model_lifecycle tests/test_model_lifecycle.c)
        target_include_directories(test_model_lifecycle PRIVATE "${PROJECT_SOURCE_DIR}/tests/fixtures")
        target_link_libraries(test_model_lifecycle PRIVATE cinder_core Threads::Threads)
        cinder_configure_target(test_model_lifecycle)
        add_test(NAME model_lifecycle COMMAND test_model_lifecycle)
        file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/loaded-lifecycle-files")
        add_test(NAME model_loaded_lifecycle COMMAND test_model_lifecycle
            "${PROJECT_SOURCE_DIR}/tests/fixtures/tiny/tiny.cnd"
            "${PROJECT_BINARY_DIR}/loaded-lifecycle-files/model.bin"
        )
        set_tests_properties(model_lifecycle model_loaded_lifecycle PROPERTIES TIMEOUT 10)
    endif()

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
        # A single generation target prevents concurrent consumers from racing
        # to write the same header during a parallel build.
        add_custom_target(cinder_numpy_fixtures DEPENDS "${cinder_numpy_header}")
        add_executable(test_tensor_numpy_generated tests/test_tensor_numpy.c)
        add_dependencies(test_tensor_numpy_generated cinder_numpy_fixtures)
        target_include_directories(test_tensor_numpy_generated PRIVATE "${PROJECT_BINARY_DIR}/numpy-fixtures")
        target_link_libraries(test_tensor_numpy_generated PRIVATE cinder_core)
        cinder_configure_target(test_tensor_numpy_generated)
        add_test(NAME tensor_numpy_generated COMMAND test_tensor_numpy_generated)
        add_executable(test_model_numpy_generated tests/test_model_numpy.c)
        add_dependencies(test_model_numpy_generated cinder_numpy_fixtures)
        target_include_directories(test_model_numpy_generated PRIVATE "${PROJECT_BINARY_DIR}/numpy-fixtures")
        target_link_libraries(test_model_numpy_generated PRIVATE cinder_core)
        cinder_configure_target(test_model_numpy_generated)
        add_test(NAME model_numpy_generated COMMAND test_model_numpy_generated)
        set_tests_properties(tensor_numpy_generated model_numpy_generated PROPERTIES TIMEOUT 10)
    endif()

    if(CINDER_ENABLE_EXPORTER_TESTS OR CINDER_ENABLE_MNIST_TESTS)
        find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.__version__)"
            RESULT_VARIABLE cinder_torch_status
            OUTPUT_VARIABLE cinder_torch_version
            ERROR_VARIABLE cinder_torch_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT cinder_torch_status EQUAL 0)
            message(FATAL_ERROR "PyTorch is required for exporter/MNIST tests. Install tools/requirements-exporter.txt with ${Python3_EXECUTABLE}. ${cinder_torch_error}")
        endif()
        message(STATUS "Testing Python tools with PyTorch ${cinder_torch_version}")
    endif()

    if(CINDER_ENABLE_EXPORTER_TESTS)
        add_test(NAME exporter_validation
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_export_model.py -v
        )
        add_test(NAME exporter_binary
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_export_binary.py -v
        )
        add_test(NAME model_round_trip_exported
            COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/check_model_round_trip.py"
                --executable "$<TARGET_FILE:test_model_round_trip>"
        )
        set_tests_properties(exporter_validation exporter_binary model_round_trip_exported PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            TIMEOUT 60
        )
    endif()

    if(CINDER_ENABLE_MNIST_TESTS)
        add_test(NAME mnist_training
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_mnist_training.py -v
        )
        add_test(NAME mnist_export
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_mnist_export.py -v
        )
        add_test(NAME mnist_fixtures
            COMMAND "${Python3_EXECUTABLE}" -m unittest discover
                -s "${PROJECT_SOURCE_DIR}/tests" -p test_mnist_fixtures.py -v
        )
        add_test(NAME mnist_inference_exported
            COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tests/check_mnist_inference.py"
                --executable "$<TARGET_FILE:test_mnist_inference>"
        )
        set_tests_properties(mnist_training mnist_export mnist_fixtures PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            TIMEOUT 60
        )
        set_tests_properties(mnist_inference_exported PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            TIMEOUT 120
        )
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
        model_allocation model_allocation_failures model_run model_numpy_fixtures
        PROPERTIES TIMEOUT 10
    )
endif()
