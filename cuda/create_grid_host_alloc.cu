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
#define VERBOSE true
#define vprint(s...) \
	if (VERBOSE) {   \
		printf(s);   \
	}

Grid* quadtree_grid(Point* d_grid_points, Point* d_points_array, int start,
					int count, int* h_grid_count, int* d_grid_count,
					pair<float, float> bottom_left_corner,
					pair<float, float> top_right_corner, int level) {
	// unzip the bounding points for faster access
	float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
		  x2 = top_right_corner.fi, y2 = top_right_corner.se;

	// Exit condition for recursion
	if (count < MIN_POINTS ||
		(abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
		Point* subset = new Point[count];
		// Get only the subset of points that the current grid requires and
		// return a grid
		if (count > 0) {
			cudaMemcpy(subset, d_points_array + start, sizeof(Point) * count,
					   cudaMemcpyDeviceToHost);
		}
		return new Grid(nullptr, nullptr, nullptr, nullptr, subset,
						top_right_corner, bottom_left_corner, count);
	}

	vprint(
		"%d: Creating grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n",
		level, x1, y1, x2, y2, count, start);

	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	// Call Kernel to reorder the points into 4 batches within d_grid_points
	float value = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
	int range = max(1.0, ceil(value));
	vprint("%d: Reorder in GPU: 1 block of %d threads each with range=%d\n",
		   level, MAX_THREADS_PER_BLOCK, range);
	reorder_points_h_alloc<<<1, MAX_THREADS_PER_BLOCK, 8 * sizeof(int)>>>(
		d_points_array, d_grid_points, count, range, middle_x, middle_y, start,
		d_grid_count);

	cudaDeviceSynchronize();

	int total = 0;
	vprint("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++) {
		vprint("sub grid %d - %d\n", i + 1, h_grid_count[i]);
		total += h_grid_count[i];
	}
	vprint("Total Count - %d\n", count);
	if (total == count) {
		vprint("Sum of sub grid counts matches total point count\n");
	}

	// Store the starting positions from d_grid_points for br, tl, tr
	int br_start_pos = start + h_grid_count[0],
		tl_start_pos = start + h_grid_count[0] + h_grid_count[1],
		tr_start_pos =
			start + h_grid_count[0] + h_grid_count[1] + h_grid_count[2];

	vprint("\n\n");

	// Recursively call the quadtree grid function on each of the 4 sub grids -
	// bl, br, tl, tr and store in Grid struct
	int g1 = h_grid_count[0], g2 = h_grid_count[1], g3 = h_grid_count[2],
		g4 = h_grid_count[3];

	Grid *bl_grid, *tl_grid, *br_grid, *tr_grid;
	bl_grid = quadtree_grid(d_grid_points, d_points_array, start, g1,
							h_grid_count, d_grid_count, bottom_left_corner,
							mp(middle_x, middle_y), level + 1);
	br_grid = quadtree_grid(d_grid_points, d_points_array, br_start_pos, g2,
							h_grid_count, d_grid_count, mp(middle_x, y1),
							mp(x2, middle_y), level + 1);
	tl_grid = quadtree_grid(d_grid_points, d_points_array, tl_start_pos, g3,
							h_grid_count, d_grid_count, mp(x1, middle_y),
							mp(middle_x, y2), level + 1);
	tr_grid = quadtree_grid(d_grid_points, d_points_array, tr_start_pos, g4,
							h_grid_count, d_grid_count, mp(middle_x, middle_y),
							top_right_corner, level + 1);

	vprint(
		"%d: Completed grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n",
		level, x1, y1, x2, y2, count, start);

	Point* subset = new Point[count];
	// Get only the subset of points that the current grid requires and return a
	// grid
	if (count > 0) {
		cudaMemcpy(subset, d_points_array + start, sizeof(Point) * count,
				   cudaMemcpyDeviceToHost);
	}
	return new Grid(bl_grid, br_grid, tl_grid, tr_grid, subset,
					top_right_corner, bottom_left_corner, count);
}

int main(int argc, char* argv[]) {
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

	Point* points_array = (Point*)malloc(point_count * sizeof(Point));
	for (int i = 0; i < point_count; i++) {
		points_array[i] = points[i];
	}

	// Store the d_grid_points array as points_array
	Point* d_grid_points;
	cudaMalloc(&d_grid_points, point_count * sizeof(Point));

	// Store all the points from the input file into the Device
	Point* d_points_arr;
	cudaMalloc(&d_points_arr, point_count * sizeof(Point));

	// Allocate page-locked memory (host memory)
	int* h_grid_count;
	cudaHostAlloc(&h_grid_count, 4 * sizeof(int), cudaHostAllocMapped);

	// Allocate device memory (GPU memory)
	int* d_grid_count;
	cudaHostGetDevicePointer(&d_grid_count, h_grid_count, 0);

	// Initialize host array
	for (int i = 0; i < 4; i++) {
		h_grid_count[i] = 0;
	}

	start = clock();

	// Transfer point data to device
	cudaMemcpy(d_points_arr, points_array, point_count * sizeof(Point),
			   cudaMemcpyHostToDevice);

	Grid* root_grid =
		quadtree_grid(d_grid_points, d_points_arr, 0, point_count, h_grid_count,
					  d_grid_count, mp(0.0, 0.0), mp(max_size, max_size), 0);

	end = clock();

	// Free d_grid_points after entire grid is created
	cudaFree(d_grid_points);
	cudaFree(d_points_arr);

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
