CC=nvcc
CFLAGS=-I. -g

create_grid: create_grid.cu kernels.cu
	$(CC) -o create_grid create_grid.cu kernels.cu $(CFLAGS)

clean:
	rm -f create_grid *~