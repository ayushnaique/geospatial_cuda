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

Grid *quadtree_grid(Point *points, int count,
					pair<float, float> bottom_left_corner,
					pair<float, float> top_right_corner, int level) {
	float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
		  x2 = top_right_corner.fi, y2 = top_right_corner.se;

	if (count < MIN_POINTS or
		(abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
		pair<float, float> upper_bound = make_pair(x2, y2);
		pair<float, float> lower_bound = make_pair(x1, y1);
		return new Grid(nullptr, nullptr, nullptr, nullptr, points, upper_bound,
						lower_bound, count);
	}

	vprint("%d: Creating grid from (%f,%f) to (%f,%f) for %d points\n", level,
		   x1, y1, x2, y2, count);

	// Array of points for the geospatial data using unified memory
	Point *d_points;
	cudaMallocManaged(&d_points, count * sizeof(Point));

	// Copy the input points to the unified memory
	memcpy(d_points, points, count * sizeof(Point));

	// Arrays for categories and grid counts using unified memory
	int *d_categories, *d_grid_counts;
	cudaMallocManaged(&d_categories, count * sizeof(int));
	cudaMallocManaged(&d_grid_counts, 4 * sizeof(int));

	// Set the number of blocks and threads per block
	int range, num_blocks, threads_per_block = MAX_THREADS_PER_BLOCK;
	if (count <= MAX_THREADS_PER_BLOCK) {
		float warps = static_cast<float>(count) / 32;
		threads_per_block = ceil(warps) * 32;
		num_blocks = 1;
	} else {
		float blocks = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
		num_blocks = min(32.0, ceil(blocks));
	}

	// Calculate the work done by each thread
	float value = static_cast<float>(count) / (num_blocks * threads_per_block);
	range = max(1.0, ceil(value));

	dim3 grid(num_blocks, 1, 1);
	dim3 block(threads_per_block, 1, 1);

	// KERNEL Function to categorize points into 4 subgrids
	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	vprint(
		"%d: Categorize in GPU: %d blocks of %d threads each with range=%d\n",
		level, num_blocks, threads_per_block, range);
	categorize_points<<<num_blocks, threads_per_block, 4 * sizeof(int)>>>(
		d_points, d_categories, d_grid_counts, count, range, middle_x,
		middle_y);

	// Synchronize to ensure kernel completion
	cudaDeviceSynchronize();

	// No need for explicit memcpy as the data is already accessible
	int total = 0;
	vprint("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++) {
		vprint("sub grid %d - %d\n", i + 1, d_grid_counts[i]);
		total += d_grid_counts[i];
	}
	vprint("Total Count - %d\n", count);
	if (total == count) {
		vprint("Sum of sub grid counts matches total point count\n");
	}

	// Declare arrays for each section of the grid using unified memory
	Point *bottom_left, *bottom_right, *top_left, *top_right;
	cudaMallocManaged(&bottom_left, d_grid_counts[0] * sizeof(Point));
	cudaMallocManaged(&bottom_right, d_grid_counts[1] * sizeof(Point));
	cudaMallocManaged(&top_left, d_grid_counts[2] * sizeof(Point));
	cudaMallocManaged(&top_right, d_grid_counts[3] * sizeof(Point));

	dim3 grid2(1, 1, 1);
	dim3 block2(threads_per_block, 1, 1);

	// KERNEL Function to assign the points to its respective array
	value = static_cast<float>(count) / threads_per_block;
	range = max(1.0, ceil(value));
	vprint("%d: Organize in GPU: 1 block of %d threads each with range=%d\n",
		   level, threads_per_block, range);
	organize_points<<<1, threads_per_block, 4 * sizeof(int)>>>(
		d_points, d_categories, bottom_left, bottom_right, top_left, top_right,
		count, range);

	// The bounds of the grid
	pair<float, float> upper_bound = make_pair(x2, y2);
	pair<float, float> lower_bound = make_pair(x1, y1);

	Grid *new_grid = new Grid(nullptr, nullptr, nullptr, nullptr, points,
							  upper_bound, lower_bound, count);

	// Synchronize to ensure kernel completion
	cudaDeviceSynchronize();
	cudaFree(d_categories);

	// Call recursive quadtree grid function directly with unified memory
	// pointers
	new_grid->bottom_left =
		quadtree_grid(bottom_left, d_grid_counts[0], bottom_left_corner,
					  mp(middle_x, middle_y), level + 1);
	new_grid->bottom_right =
		quadtree_grid(bottom_right, d_grid_counts[1], mp(middle_x, y1),
					  mp(x2, middle_y), level + 1);
	new_grid->top_left =
		quadtree_grid(top_left, d_grid_counts[2], mp(x1, middle_y),
					  mp(middle_x, y2), level + 1);
	new_grid->top_right =
		quadtree_grid(top_right, d_grid_counts[3], mp(middle_x, middle_y),
					  top_right_corner, level + 1);

	return new_grid;
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

	file.close();

	double time_taken;
	clock_t start, end;

	Point *points_array = (Point *)malloc(point_count * sizeof(Point));
	for (int i = 0; i < point_count; i++) {
		points_array[i] = points[i];
	}
	start = clock();
	Grid *root_grid = quadtree_grid(points_array, point_count, mp(0.0, 0.0),
									mp(max_size, max_size), 0);
	end = clock();

	time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken = %lf\n\n", time_taken);

	printf("Validating grid...\n");
	pair<float, float> lower_bound = make_pair(0.0, 0.0);
	pair<float, float> upper_bound = make_pair(max_size, max_size);
	bool check = validate_grid(root_grid, upper_bound, lower_bound);

	if (check == true)
		printf("Grid Verification Success!\n");
	else
		printf("Grid Verification Failure!\n");

	return 0;
}
