# test/CMakeLists.txt

add_library(kahan OBJECT kahan.cu)

add_executable(test_verify test_verify.cc $<TARGET_OBJECTS:kahan>)

target_link_libraries(test_verify PRIVATE llmmm)

add_executable(test_tune test_tune.cc)

target_link_libraries(test_tune PRIVATE llmmm)

add_executable(test_cublas test_cublas.cc)

find_package(CUDA REQUIRED)
target_link_libraries(test_cublas cublas ${CUDA_LIBRARIES})

add_executable(test_cublas_2 test_cublas_2.cc)

find_package(CUDA REQUIRED)
target_link_libraries(test_cublas_2 cublas ${CUDA_LIBRARIES})
