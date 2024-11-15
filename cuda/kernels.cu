#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "kernels.h"

using namespace std;
namespace cg = cooperative_groups;

__inline__ __device__ int reduce_sum(int value,
									 cg::thread_block_tile<32> warp) {
	// Perform warp-wide reduction using shfl_down_sync
	// Refer https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
	for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
		value += __shfl_down_sync(0xFFFFFFFF, value, offset);
	}
	return value;
}

__global__ void categorize_points(Point *d_points, int *d_categories,
								  int *grid_counts, int count, int range,
								  float middle_x, float middle_y) {
	// subgrid_counts declared outside kernel, Dynamic Shared Memory
	// Accessed using extern
	extern __shared__ int subgrid_counts[];

	int start = ((blockIdx.x * blockDim.x) + threadIdx.x) * range;

	// create a thread group for 32 threads (warp grouping)
	cg::thread_block block = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

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

	// sum up all the sub quadrant counts inside a warp
	first = reduce_sum(first, warp);
	second = reduce_sum(second, warp);
	third = reduce_sum(third, warp);
	fourth = reduce_sum(fourth, warp);

	// Only the first thread in each warp writes to shared memory
	if (warp.thread_rank() == 0) {
		atomicAdd(&subgrid_counts[0], first);
		atomicAdd(&subgrid_counts[1], second);
		atomicAdd(&subgrid_counts[2], third);
		atomicAdd(&subgrid_counts[3], fourth);
	}
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

__global__ void reorder_points(Point *d_points, Point *grid_points,
							   int *grid_counts, int count, int range,
							   float middle_x, float middle_y, int start_pos) {
	// subgrid_counts declared outside kernel, Dynamic Shared Memory
	// Accessed using extern
	extern __shared__ int subgrid_offsets[];

	// create a thread group for 32 threads (warp grouping)
	cg::thread_block block = cg::this_thread_block();
	cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

	// Initialize the subgrid counts to 0
	if (threadIdx.x == 0) {
		subgrid_offsets[0] = 0;
		subgrid_offsets[1] = 0;
		subgrid_offsets[2] = 0;
		subgrid_offsets[3] = 0;
	}
	__syncthreads();

	// Iterate through all the points in d_points and count points in every
	// category
	int start = start_pos + threadIdx.x * range, first = 0, second = 0,
		third = 0, fourth = 0, category;
	for (int i = start; i < start + range; i++) {
		if (i < start_pos + count) {
			// bottom left; if the point lies in bottom left, increment
			if (d_points[i].x <= middle_x and d_points[i].y <= middle_y) {
				first++;
			}
			// bottom right; if point lies in bottom right, increment
			else if (d_points[i].x > middle_x and d_points[i].y <= middle_y) {
				second++;
			}
			// top left; if point lies in top left, increment
			else if (d_points[i].x <= middle_x and d_points[i].y > middle_y) {
				third++;
			}
			// top right; if point lies in top right, increment
			else if (d_points[i].x > middle_x and d_points[i].y > middle_y) {
				fourth++;
			}
		}
	}

	// sum up all the sub quadrant counts inside a warp
	first = reduce_sum(first, warp);
	second = reduce_sum(second, warp);
	third = reduce_sum(third, warp);
	fourth = reduce_sum(fourth, warp);

	// Only the first thread in each warp writes to shared memory
	if (warp.thread_rank() == 0) {
		atomicAdd(&subgrid_offsets[0], first);
		atomicAdd(&subgrid_offsets[1], second);
		atomicAdd(&subgrid_offsets[2], third);
		atomicAdd(&subgrid_offsets[3], fourth);
	}
	__syncthreads();

	// Calculate the start position for every sub grid category and store in
	// shared memory
	if (threadIdx.x == 0) {
		subgrid_offsets[4] = start_pos;
		subgrid_offsets[5] = start_pos + subgrid_offsets[0];
		subgrid_offsets[6] = subgrid_offsets[5] + subgrid_offsets[1];
		subgrid_offsets[7] = subgrid_offsets[6] + subgrid_offsets[2];
	}

	// Iterate through every point in d_points and depending on the category,
	// find latest index for category and insert point into that index within
	// grid_points
	for (int i = start; i < start + range; i++) {
		if (i < start_pos + count) {
			// bottom left; if the point lies in bottom left, increment
			if (d_points[i].x <= middle_x and d_points[i].y <= middle_y) {
				category = 0;
			}
			// bottom right; if point lies in bottom right, increment
			else if (d_points[i].x > middle_x and d_points[i].y <= middle_y) {
				category = 1;
			}
			// top left; if point lies in top left, increment
			else if (d_points[i].x <= middle_x and d_points[i].y > middle_y) {
				category = 2;
			}
			// top right; if point lies in top right, increment
			else if (d_points[i].x > middle_x and d_points[i].y > middle_y) {
				category = 3;
			}

			// atomic add at offset value and insert into grid_points
			unsigned int index = atomicAdd(&subgrid_offsets[4 + category], 1);
			grid_points[index] = d_points[i];
		}
	}

	// Assign grid_counts as the counts of subgrids
	if (threadIdx.x == 0) {
		grid_counts[0] = subgrid_offsets[0];
		grid_counts[1] = subgrid_offsets[1];
		grid_counts[2] = subgrid_offsets[2];
		grid_counts[3] = subgrid_offsets[3];
	}
}

// Validation Function
bool validate_grid(Grid *root_grid, pair<float, float> &top_right_corner,
				   pair<float, float> &bottom_left_corner) {
	if (root_grid == nullptr) return true;

	// If we have reached the bottom of the grid, we start validation
	if (root_grid->points) {
		Point *point_array = root_grid->points;
		float top_x = top_right_corner.first;
		float top_y = top_right_corner.second;

		float bot_x = bottom_left_corner.first;
		float bot_y = bottom_left_corner.second;

		float mid_x = (top_x + bot_x) / 2;
		float mid_y = (top_y + bot_y) / 2;

		int count = root_grid->count;

		for (int i = 0; i < count; i++) {
			float point_x = point_array[i].x;
			float point_y = point_array[i].y;

			if (point_x < bot_x || point_x > top_x) {
				printf(
					"Validation Error! Point (%f, %f) is plced out of bounds. "
					"Grid dimension: [(%f, %f), (%f, %f)]\n",
					point_x, point_y, bot_x, bot_y, top_x, top_y);
				return false;
			} else if (point_y < bot_y || point_y > top_y) {
				printf(
					"Validation Error! Point (%f, %f) is plced out of bounds. "
					"Grid dimension: [(%f, %f), (%f, %f)]\n",
					point_x, point_y, bot_x, bot_y, top_x, top_y);
				return false;
			} else {
				continue;
			}
		}

		return true;
	}

	// Call Recursively for all 4 quadrants
	Grid *top_left_child = nullptr;
	Grid *top_right_child = nullptr;
	Grid *bottom_left_child = nullptr;
	Grid *bottom_right_child = nullptr;

	top_left_child = root_grid->top_left;
	top_right_child = root_grid->top_right;
	bottom_left_child = root_grid->bottom_left;
	bottom_right_child = root_grid->bottom_right;

	bool check_top_left =
		validate_grid(top_left_child, top_left_child->top_right_corner,
					  top_left_child->bottom_left_corner);
	bool check_top_right =
		validate_grid(top_right_child, top_right_child->top_right_corner,
					  top_right_child->bottom_left_corner);
	bool check_bottom_left =
		validate_grid(bottom_left_child, bottom_left_child->top_right_corner,
					  bottom_left_child->bottom_left_corner);
	bool check_bottom_right =
		validate_grid(bottom_right_child, bottom_right_child->top_right_corner,
					  bottom_right_child->bottom_left_corner);

	return check_top_left && check_top_right && check_bottom_left &&
		   check_bottom_right;
}
