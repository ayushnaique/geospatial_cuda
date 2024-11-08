#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <kernels.h>

using namespace std;

__global__ void categorize_points(Point *d_points, int *d_categories,
								  int *grid_counts, int count, int range,
								  int middle_x, int middle_y) {
	// subgrid_counts declared outside kernel, Dynamic Shared Memory
	// Accessed using extern
	extern __shared__ int subgrid_counts[];

	int start = ((blockIdx.x * blockDim.x) + threadIdx.x) * range;

	// Initialize the subgrid counts to 0
	if (threadIdx.x == 0) {
		subgrid_counts[0] = 0;
		subgrid_counts[1] = 0;
		subgrid_counts[2] = 0;
		subgrid_counts[3] = 0;
	}
	__syncthreads();

	int first = 0, second = 0, third = 0, fourth = 0;
	for (int i = start; i < start + range; i++) {
		if (i < count) {
			// bottom left; if the point lies in bottom left, increment
			if (d_points[i].x <= middle_x and d_points[i].y <= middle_y) {
				d_categories[i] = 0;
				first++;
			}
			// bottom right; if point lies in bottom right, increment
			else if (d_points[i].x > middle_x and d_points[i].y <= middle_y) {
				d_categories[i] = 1;
				second++;
			}
			// top left; if point lies in top left, increment
			else if (d_points[i].x <= middle_x and d_points[i].y > middle_y) {
				d_categories[i] = 2;
				third++;
			}
			// top right; if point lies in top right, increment
			else if (d_points[i].x > middle_x and d_points[i].y > middle_y) {
				d_categories[i] = 3;
				fourth++;
			}
		}
	}

	// CUDA built in function to perform atomic addition at given location
	// Location : first variable
	// Store the counts of points in their respective subgrid
	atomicAdd(&subgrid_counts[0], first);
	atomicAdd(&subgrid_counts[1], second);
	atomicAdd(&subgrid_counts[2], third);
	atomicAdd(&subgrid_counts[3], fourth);
	__syncthreads();

	// Add the values of subgrid_counts to grid_counts
	if (threadIdx.x == 0) {
		atomicAdd(&grid_counts[0], subgrid_counts[0]);
		atomicAdd(&grid_counts[1], subgrid_counts[1]);
		atomicAdd(&grid_counts[2], subgrid_counts[2]);
		atomicAdd(&grid_counts[3], subgrid_counts[3]);
	}
}

__global__ void organize_points(Point *d_points, int *d_categories, Point *bl,
								Point *br, Point *tl, Point *tr, int count,
								int range) {
	extern __shared__ int subgrid_index[];

	// Initialize subgrid pointer to 0
	// Used to index the point arrays for each subgrid
	if (threadIdx.x == 0) {
		subgrid_index[0] = 0;
		subgrid_index[1] = 0;
		subgrid_index[2] = 0;
		subgrid_index[3] = 0;
	}
	__syncthreads();

	int start = threadIdx.x * range;
	for (int i = start; i < start + range; i++) {
		if (i < count) {
			// Point array will store the respective points in a contiguous
			// fashion increment subgrid index according to the category
			unsigned int category_index =
				atomicAdd(&subgrid_index[d_categories[i]], 1);
			if (d_categories[i] == 0) {
				bl[category_index] = d_points[i];
			}
			if (d_categories[i] == 1) {
				br[category_index] = d_points[i];
			}
			if (d_categories[i] == 2) {
				tl[category_index] = d_points[i];
			}
			if (d_categories[i] == 3) {
				tr[category_index] = d_points[i];
			}
		}
	}
}