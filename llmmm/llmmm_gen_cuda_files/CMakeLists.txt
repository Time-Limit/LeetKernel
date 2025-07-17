file(GLOB_RECURSE LLMMM_SOURCES "*.cu")

add_library(llmmm_gen_cuda_files STATIC ${LLMMM_SOURCES})

set_target_properties(llmmm_gen_cuda_files PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(llmmm_gen_cuda_files PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
