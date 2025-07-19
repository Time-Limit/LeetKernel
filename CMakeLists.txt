cmake_minimum_required(VERSION 3.18)

project(LLMMM LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -g -lineinfo --ptxas-options=-v")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")

add_subdirectory(llmmm)
add_subdirectory(test)
add_subdirectory(examples)
