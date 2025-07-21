# test/CMakeLists.txt

add_executable(003.matrix_multiplication 003.matrix_multiplication.cu)

find_package(CUDA REQUIRED)

target_link_libraries(003.matrix_multiplication cublas ${CUDA_LIBRARIES})

add_executable(000.test 000.test.cu)

target_link_libraries(000.test ${CUDA_LIBRARIES})
