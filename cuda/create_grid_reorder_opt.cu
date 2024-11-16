#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include "kernels.h"
using namespace std;

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

GridArray *quadtree_grid(Point *d_grid_points0, Point *d_grid_points1,
						 int count, pair<float, float> bottom_left_corner,
						 pair<float, float> top_right_corner,
						 int *h_grid_counts, int *d_grid_counts, int start_pos,
						 int level, int grid_array_flag) {
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
	bl_grid =
		quadtree_grid(d_grid_points1, d_grid_points0, bl_count,
					  bottom_left_corner, mp(middle_x, middle_y), h_grid_counts,
					  d_grid_counts, start_pos, level + 1, grid_array_flag ^ 1);
	br_grid = quadtree_grid(d_grid_points1, d_grid_points0, br_count,
							mp(middle_x, y1), mp(x2, middle_y), h_grid_counts,
							d_grid_counts, br_start_pos, level + 1,
							grid_array_flag ^ 1);
	tl_grid = quadtree_grid(d_grid_points1, d_grid_points0, tl_count,
							mp(x1, middle_y), mp(middle_x, y2), h_grid_counts,
							d_grid_counts, tl_start_pos, level + 1,
							grid_array_flag ^ 1);
	tr_grid = quadtree_grid(d_grid_points1, d_grid_points0, tr_count,
							mp(middle_x, middle_y), top_right_corner,
							h_grid_counts, d_grid_counts, tr_start_pos,
							level + 1, grid_array_flag ^ 1);

	return new GridArray(bl_grid, br_grid, tl_grid, tr_grid, top_right_corner,
						 bottom_left_corner, count, start_pos, grid_array_flag);
}

int main(int argc, char *argv[]) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " file_path max_boundary"
				  << std::endl;
		return 1;
	}

	string filename = argv[1];
	float max_size = atof(argv[2]);

	ifstream file(filename);
	if (!file) {
		cerr << "Error: Could not open the file " << filename << endl;
		return 1;
	}

	string line;
	float x, y;
	vector<Point> points;
	int point_count = 0;
	while (getline(file, line)) {
		istringstream iss(line);
		if (iss >> x >> y) {
			Point p = Point((float)x, (float)y);
			points.emplace_back(p);
			point_count++;
		} else {
			cerr << "Warning: Skipping malformed line: " << line << endl;
		}
	}
	printf("Generating grid for %d points\n\n", point_count);

	file.close();

	double time_taken;
	clock_t start, end;

	// Store the d_grid_points array as points_array
	Point *d_grid_points0, *d_grid_points1;
	cudaMalloc(&d_grid_points0, point_count * sizeof(Point));
	cudaMalloc(&d_grid_points1, point_count * sizeof(Point));

	cudaMemcpy(d_grid_points0, points.data(), point_count * sizeof(Point),
			   cudaMemcpyHostToDevice);

	// Allocate page-locked memory (host memory)
	int *h_grid_counts;
	cudaHostAlloc(&h_grid_counts, 4 * sizeof(int), cudaHostAllocMapped);

	// Allocate device memory (GPU memory)
	int *d_grid_counts;
	cudaHostGetDevicePointer(&d_grid_counts, h_grid_counts, 0);

	// Allocate point arrays for storing final grid arrays from gpu
	Point *grid_array0 = (Point *)malloc(point_count * sizeof(Point));
	Point *grid_array1 = (Point *)malloc(point_count * sizeof(Point));

	start = clock();
	GridArray *root_grid = quadtree_grid(
		d_grid_points0, d_grid_points1, point_count, mp(0.0, 0.0),
		mp(max_size, max_size), h_grid_counts, d_grid_counts, 0, 0, 0);
	// copy the grid arrays from gpu to host
	cudaMemcpy(grid_array0, d_grid_points0, point_count * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(grid_array1, d_grid_points1, point_count * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	end = clock();

	// Free d_grid_points after entire grid is created
	cudaFree(d_grid_points0);
	cudaFree(d_grid_points1);

	time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken = %lf\n\n", time_taken);

	Grid *root = assign_points(root_grid, grid_array0, grid_array1);

	printf("Validating grid...\n");
	pair<float, float> lower_bound = make_pair(0.0, 0.0);
	pair<float, float> upper_bound = make_pair(max_size, max_size);
	bool check = validate_grid(root, upper_bound, lower_bound);

	if (check == true)
		printf("Grid Verification Success!\n");
	else
		printf("Grid Verification Failure!\n");

	return 0;
}
