#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "kernels.h"

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5.0
#define MIN_DISTANCE 5.0
#define MAX_THREADS_PER_BLOCK 512
#define VERBOSE false
#define vprint(s...) \
	if (VERBOSE) {   \
		printf(s);   \
	}

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

__global__ void quadrant_search(Query *queries, int num_queries,
								QuadrantBoundary *boundaries,
								int num_boundaries, int *results) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_queries) {
		Query query = queries[idx];
		int result = -1;

		for (int i = 0; i < num_boundaries; i++) {
			QuadrantBoundary boundary = boundaries[i];
			if (query.point.x >= boundary.bottom_left.first &&
				query.point.x <= boundary.top_right.first &&
				query.point.y >= boundary.bottom_left.second &&
				query.point.y <= boundary.top_right.second) {
				result = max(result, boundary.id);
			}
		}

		results[idx] = result;
	}
}

__global__ void reorder_points(Point *d_points, Point *grid_points,
							   int *grid_counts, int count, int range,
							   float middle_x, float middle_y, int start_pos,
							   bool opt) {
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
	int start = threadIdx.x * range, end = count, first = 0, second = 0,
		third = 0, fourth = 0, category;
	if (opt) {
		start += start_pos;
		end += start_pos;
	}
	for (int i = start; i < start + range; i++) {
		if (i < end) {
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
		if (i < end) {
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

__global__ void reorder_points_h_alloc(Point *d_points, Point *grid_points,
									   int count, int range, float middle_x,
									   float middle_y, int start_pos,
									   int *d_grid_count) {
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
		d_grid_count[0] = 0;
		d_grid_count[1] = 0;
		d_grid_count[2] = 0;
		d_grid_count[3] = 0;
	}
	__syncthreads();

	// Iterate through all the points in d_points and count points in every
	// category
	int start = threadIdx.x * range, first = 0, second = 0, third = 0,
		fourth = 0, category;
	for (int i = start; i < start + range; i++) {
		if (i < count) {
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
		if (i < count) {
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
		d_grid_count[0] = subgrid_offsets[0];
		d_grid_count[1] = subgrid_offsets[1];
		d_grid_count[2] = subgrid_offsets[2];
		d_grid_count[3] = subgrid_offsets[3];
	}
}

GridArray *construct_grid_array(Point *d_grid_points0, Point *d_grid_points1,
								int count,
								pair<float, float> bottom_left_corner,
								pair<float, float> top_right_corner,
								int *h_grid_counts, int *d_grid_counts,
								int start_pos, int level, int grid_array_flag) {
	float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
		  x2 = top_right_corner.fi, y2 = top_right_corner.se;

	// Exit condition for recursion
	if (count < MIN_POINTS or
		(abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
		return new GridArray(nullptr, nullptr, nullptr, nullptr,
							 top_right_corner, bottom_left_corner, count,
							 start_pos, grid_array_flag);
	}

	vprint(
		"%d: Creating grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n",
		level, x1, y1, x2, y2, count, start_pos);

	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	// Call Kernel to reorder the points into 4 batches within d_grid_points
	float value = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
	int range = max(1.0, ceil(value));
	vprint("%d: Reorder in GPU: 1 block of %d threads each with range=%d\n",
		   level, MAX_THREADS_PER_BLOCK, range);
	reorder_points<<<1, MAX_THREADS_PER_BLOCK, 8 * sizeof(int)>>>(
		d_grid_points0, d_grid_points1, d_grid_counts, count, range, middle_x,
		middle_y, start_pos, true);

	cudaDeviceSynchronize();

	int total = 0;
	vprint("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++) {
		vprint("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
		total += h_grid_counts[i];
	}
	vprint("Total Count - %d\n", count);
	if (total == count) {
		vprint("Sum of sub grid counts matches total point count\n");
	}

	int bl_count = h_grid_counts[0], br_count = h_grid_counts[1],
		tl_count = h_grid_counts[2], tr_count = h_grid_counts[3];

	// Store the starting positions from d_grid_points for br, tl, tr
	int br_start_pos = start_pos + bl_count,
		tl_start_pos = start_pos + bl_count + br_count,
		tr_start_pos = start_pos + bl_count + br_count + tl_count;

	vprint(
		"%d: Completed grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n\n",
		level, x1, y1, x2, y2, count, start_pos);

	// Recursively call the quadtree grid function on each of the 4 sub grids
	GridArray *bl_grid, *tl_grid, *br_grid, *tr_grid;
	bl_grid = construct_grid_array(d_grid_points1, d_grid_points0, bl_count,
								   bottom_left_corner, mp(middle_x, middle_y),
								   h_grid_counts, d_grid_counts, start_pos,
								   level + 1, grid_array_flag ^ 1);
	br_grid = construct_grid_array(d_grid_points1, d_grid_points0, br_count,
								   mp(middle_x, y1), mp(x2, middle_y),
								   h_grid_counts, d_grid_counts, br_start_pos,
								   level + 1, grid_array_flag ^ 1);
	tl_grid = construct_grid_array(d_grid_points1, d_grid_points0, tl_count,
								   mp(x1, middle_y), mp(middle_x, y2),
								   h_grid_counts, d_grid_counts, tl_start_pos,
								   level + 1, grid_array_flag ^ 1);
	tr_grid = construct_grid_array(d_grid_points1, d_grid_points0, tr_count,
								   mp(middle_x, middle_y), top_right_corner,
								   h_grid_counts, d_grid_counts, tr_start_pos,
								   level + 1, grid_array_flag ^ 1);

	return new GridArray(bl_grid, br_grid, tl_grid, tr_grid, top_right_corner,
						 bottom_left_corner, count, start_pos, grid_array_flag);
}

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

Grid *assign_points(GridArray *root_grid, Point *grid_array0,
					Point *grid_array1) {
	int count = root_grid->count, start_pos = root_grid->start_pos;
	Point *points = (Point *)malloc(root_grid->count * sizeof(Point));
	Point *grid_array = grid_array0;
	if (root_grid->grid_array_flag) grid_array = grid_array1;
	for (int i = 0; i < count; i++) {
		points[i] = grid_array[start_pos + i];
	}
	Grid *bl = nullptr, *br = nullptr, *tl = nullptr, *tr = nullptr;
	if (root_grid->bottom_left)
		bl = assign_points(root_grid->bottom_left, grid_array0, grid_array1);
	if (root_grid->bottom_right)
		br = assign_points(root_grid->bottom_right, grid_array0, grid_array1);
	if (root_grid->top_left)
		tl = assign_points(root_grid->top_left, grid_array0, grid_array1);
	if (root_grid->top_right)
		tr = assign_points(root_grid->top_right, grid_array0, grid_array1);

	return new Grid(bl, br, tl, tr, points, root_grid->top_right_corner,
					root_grid->bottom_left_corner, count);
}

void prepare_boundaries(Grid *root_grid, int id, Grid *parent_grid,
						vector<QuadrantBoundary> &boundaries,
						unordered_map<int, Grid *> &grid_map) {
	root_grid->id = id;
	root_grid->parent = parent_grid;
	boundaries.push_back(
		{id, root_grid->bottom_left_corner, root_grid->top_right_corner});
	grid_map[id] = root_grid;
	if (root_grid->bottom_left)
		prepare_boundaries(root_grid->bottom_left, id * 4 + 1, root_grid,
						   boundaries, grid_map);
	if (root_grid->bottom_right)
		prepare_boundaries(root_grid->bottom_right, id * 4 + 2, root_grid,
						   boundaries, grid_map);
	if (root_grid->top_left)
		prepare_boundaries(root_grid->top_left, id * 4 + 3, root_grid,
						   boundaries, grid_map);
	if (root_grid->top_right)
		prepare_boundaries(root_grid->top_right, id * 4 + 4, root_grid,
						   boundaries, grid_map);
}

vector<int> search_quadrant(const vector<Query> &queries,
							const vector<QuadrantBoundary> &boundaries) {
	Query *d_queries;
	QuadrantBoundary *d_boundaries;
	int *d_results;

	cudaMalloc(&d_queries, queries.size() * sizeof(Query));
	cudaMalloc(&d_boundaries, boundaries.size() * sizeof(QuadrantBoundary));
	cudaMalloc(&d_results, queries.size() * sizeof(int));

	cudaMemcpy(d_queries, queries.data(), queries.size() * sizeof(Query),
			   cudaMemcpyHostToDevice);
	cudaMemcpy(d_boundaries, boundaries.data(),
			   boundaries.size() * sizeof(QuadrantBoundary),
			   cudaMemcpyHostToDevice);

	int block_size = 256;
	int num_blocks = 16;
	quadrant_search<<<num_blocks, block_size>>>(
		d_queries, queries.size(), d_boundaries, boundaries.size(), d_results);

	vector<int> results(queries.size());
	cudaMemcpy(results.data(), d_results, queries.size() * sizeof(int),
			   cudaMemcpyDeviceToHost);

	cudaFree(d_queries);
	cudaFree(d_boundaries);
	cudaFree(d_results);

	return results;	 // Return -1 if point not found in any quadrant
}