# Compiler and flags
CC = nvcc
CFLAGS = -I. -g
OMPFLAG=-Xcompiler -Wall -Xcompiler -fopenmp

all: grid grid_managed grid_opt grid_reorder grid_stream grid_host grid_inference

grid: create_grid.cu kernels.cu
	$(CC) -o grid create_grid.cu kernels.cu $(CFLAGS)

grid_managed: create_grid_managed.cu kernels.cu
	$(CC) -o grid_managed create_grid_managed.cu kernels.cu $(CFLAGS)

grid_opt_v1: create_grid_opt_v1.cu kernels.cu
	$(CC) -o grid_opt_v1 create_grid_opt_v1.cu kernels.cu $(CFLAGS)

grid_opt: create_grid_opt_v2.cu kernels.cu
	$(CC) -o grid_opt create_grid_opt_v2.cu kernels.cu $(CFLAGS)

grid_reorder: create_grid_reorder.cu kernels.cu
	$(CC) -o grid_reorder create_grid_reorder.cu kernels.cu $(CFLAGS)

grid_stream: create_grid_with_stream.cu kernels.cu
	$(CC) -o grid_stream create_grid_with_stream.cu kernels.cu $(CFLAGS)

grid_host: create_grid_host_alloc.cu kernels.cu
	$(CC) -o grid_host create_grid_host_alloc.cu kernels.cu $(CFLAGS)

stream_alloc: streams_host_alloc.cu kernels.cu
	$(CC) -o stream_alloc streams_host_alloc.cu kernels.cu $(CFLAGS)

stream_omp: streams_omp.cu kernels.cu
	$(CC) -o stream_omp streams_omp.cu kernels.cu $(OMPFLAG) $(CFLAGS)

grid_inference: kernels.cu kernels.cu insert.cpp delete.cpp inference.cu
	$(CC) -o grid_inference inference.cu kernels.cu insert.cpp delete.cpp $(CFLAGS)

clean:
	rm -f ./grid
	rm -f ./grid_managed
	rm -f ./grid_opt_v1
	rm -f ./grid_opt
	rm -f ./grid_reorder
	rm -f ./grid_stream
	rm -f ./grid_host
	rm -f ./stream_alloc
	rm -f ./stream_omp
	rm -f ./grid_inference
