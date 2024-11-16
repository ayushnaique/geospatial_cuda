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

grid_ayush: create_grid_new.cu kernels_new.cu
        $(CC) -o create_grid create_grid_new.cu kernels_new.cu $(CFLAGS)

grid_host: create_grid_host_alloc.cu kernels.cu
	$(CC) -o create_grid_host create_grid_host_alloc.cu kernels.cu $(CFLAGS)

clean:
	rm -f create_grid *~
