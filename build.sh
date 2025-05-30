nvcc -o 001.vector_addition 001.vector_addition.cu -O3 --generate-code arch=compute_89,code=sm_89
ncu --set full -f -o 002.matrix_transpose.ncu-rep 002.matrix_transpose
