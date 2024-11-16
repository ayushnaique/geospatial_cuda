CC=nvcc
CFLAGS=-I. -g

grid: create_grid.cu kernels.cu
	$(CC) -o create_grid create_grid.cu kernels.cu $(CFLAGS)

grid_managed: create_grid_managed.cu kernels.cu
	$(CC) -o create_grid_managed create_grid_managed.cu kernels.cu $(CFLAGS)

grid_reorder: create_grid_reorder.cu kernels.cu
	$(CC) -o create_grid_reorder create_grid_reorder.cu kernels.cu $(CFLAGS)

grid_stream: create_grid_with_stream.cu kernels.cu
	$(CC) -o create_grid_with_stream create_grid_with_stream.cu kernels.cu $(CFLAGS)

grid_host: create_grid_host_alloc.cu kernels.cu
	$(CC) -o create_grid_host create_grid_host_alloc.cu kernels.cu $(CFLAGS)

clean:
	rm -f create_grid *~
