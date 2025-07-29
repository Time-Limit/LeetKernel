# test/CMakeLists.txt

find_package(CUDA REQUIRED)

add_executable(000.test 000.test.cu)
target_link_libraries(000.test PRIVATE ${CUDA_LIBRARIES})

add_executable(003.matrix_multiplication 003.matrix_multiplication.cu)
target_link_libraries(003.matrix_multiplication PRIVATE cublas ${CUDA_LIBRARIES})

add_executable(004.fp16.mm 004.fp16.mm.cu)
target_link_libraries(004.fp16.mm PRIVATE cublas cublasLt ${CUDA_LIBRARIES})

add_executable(004.fp16.cublas 004.fp16.cublas.cu)
target_link_libraries(004.fp16.cublas PRIVATE cublas cublasLt ${CUDA_LIBRARIES})
