nvcc -o ${1} ${1}.cu -O3 -g -lineinfo --generate-code arch=compute_89,code=sm_89
