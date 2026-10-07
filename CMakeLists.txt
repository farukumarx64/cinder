cmake_minimum_required(VERSION 3.16)

project(Cinder VERSION 0.1.0 LANGUAGES C)

add_executable(cinder src/cli.c)

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
