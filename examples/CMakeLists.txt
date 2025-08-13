# test/CMakeLists.txt

find_package(CUDA REQUIRED)

add_executable(000.test 000.test.cu)
target_link_libraries(000.test PRIVATE ${CUDA_LIBRARIES})

add_executable(000.query_cuda_config 000.query_cuda_config.cu)
target_link_libraries(000.query_cuda_config PRIVATE ${CUDA_LIBRARIES})

add_executable(003.matrix_multiplication 003.matrix_multiplication.cu)
target_link_libraries(003.matrix_multiplication PRIVATE cublas ${CUDA_LIBRARIES})

add_executable(004.fp16.mm.ldmatrix 004.fp16.mm.ldmatrix.cu)
target_link_libraries(004.fp16.mm.ldmatrix PRIVATE cublas cublasLt ${CUDA_LIBRARIES})

add_executable(004.fp16.mm.ldmatrix.inner_split 004.fp16.mm.ldmatrix.inner_split.cu)
target_link_libraries(004.fp16.mm.ldmatrix.inner_split PRIVATE cublas cublasLt ${CUDA_LIBRARIES})

add_executable(004.fp16.mm.ldmatrix.notrans 004.fp16.mm.ldmatrix.notrans.cu)
target_link_libraries(004.fp16.mm.ldmatrix.notrans PRIVATE cublas cublasLt ${CUDA_LIBRARIES})

add_executable(004.fp16.mm.ldmatrix.trans 004.fp16.mm.ldmatrix.trans.cu)
target_link_libraries(004.fp16.mm.ldmatrix.trans PRIVATE cublas cublasLt ${CUDA_LIBRARIES})

add_executable(004.fp16.cublas 004.fp16.cublas.cu)
target_link_libraries(004.fp16.cublas PRIVATE cublas cublasLt ${CUDA_LIBRARIES})
