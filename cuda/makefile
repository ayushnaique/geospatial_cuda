CC=nvcc
CFLAGS=-I. -g

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

grid_stream1: create_grid_reorder_opt_with_streams.cu kernels.cu
	$(CC) -o grid_stream1 create_grid_reorder_opt_with_streams.cu kernels.cu $(CFLAGS)

grid_alloc: streams.cu kernels.cu
	$(CC) -o grid_stream streams.cu kernels.cu $(CFLAGS)

clean:
	rm -f create_grid *~
