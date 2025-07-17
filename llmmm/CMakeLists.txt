add_subdirectory(llmmm_gen_cuda_files)

add_library(llmmm STATIC llmmm.cc)

set_target_properties(llmmm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(llmmm PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(llmmm PRIVATE llmmm_gen_cuda_files)
