#nvcc -o ${1} ${1}.cu -O3 -g -lineinfo --generate-code arch=compute_89,code=sm_89 --ptxas-options=-v --ptxas-options=-maxrregcount=255 --maxrregcount=255  -lcublas
#nvcc -o ${1} ${1}.cu -O3 -g -I/usr/local/cuda/targets/x86_64-linux/include -lineinfo --generate-code arch=compute_89,code=sm_89 --ptxas-options=-v -lcublas
#nvcc -o ${1} ${1}.cu -O3 -g -lineinfo --generate-code arch=compute_89,code=sm_89 --ptxas-options=-v -lcublas -lcublasLt -lcudart
nvcc -o ${1} ${1}.cu -O3 -g -lineinfo --generate-code arch=compute_89,code=sm_89 --ptxas-options=-v -lcublas -I./
