# Compiler and flags
CC = nvcc
CFLAGS = -I. -g

all: grid grid_managed grid_opt grid_reorder grid_stream grid_host grid_inference

grid: create_grid.cu kernels.cu
	$(CC) -o grid create_grid.cu kernels.cu $(CFLAGS)

grid_managed: create_grid_managed.cu kernels.cu
	$(CC) -o grid_managed create_grid_managed.cu kernels.cu $(CFLAGS)

grid_opt: create_grid_reorder_opt.cu kernels.cu
	$(CC) -o grid_opt create_grid_reorder_opt.cu kernels.cu $(CFLAGS)

grid_reorder: create_grid_reorder.cu kernels.cu
	$(CC) -o grid_reorder create_grid_reorder.cu kernels.cu $(CFLAGS)

grid_stream: create_grid_with_stream.cu kernels.cu
	$(CC) -o grid_stream create_grid_with_stream.cu kernels.cu $(CFLAGS)

grid_host: create_grid_host_alloc.cu kernels.cu
	$(CC) -o grid_host create_grid_host_alloc.cu kernels.cu $(CFLAGS)

grid_inference: kernels.cu kernels.cu insert.cpp delete.cpp inference.cu
	$(CC) -o grid_inference inference.cu kernels.cu insert.cpp delete.cpp $(CFLAGS)

clean:
	rm -f ./grid
	rm -f ./grid_managed
	rm -f ./grid_opt
	rm -f ./grid_reorder
	rm -f ./grid_stream
	rm -f ./grid_host
	rm -f ./grid_inference
