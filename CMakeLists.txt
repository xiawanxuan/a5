cmake_minimum_required(VERSION 3.14)
project(numcpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(NUMPCPP_BUILD_TESTS "Build tests" ON)
option(NUMPCPP_ENABLE_OPENMP "Enable OpenMP parallelization" ON)
option(NUMPCPP_ENABLE_MULTITHREADING "Enable C++ standard thread parallelization" ON)

if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

if(NUMPCPP_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        add_compile_definitions(NUMPCPP_HAVE_OPENMP)
    endif()
endif()

if(NUMPCPP_ENABLE_MULTITHREADING)
    add_compile_definitions(NUMPCPP_HAVE_THREADS)
endif()

set(NUMPCPP_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

if(NUMPCPP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
