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

Grid *quadtree_grid(Point *points, Point *d_grid_points, int count,
					pair<float, float> bottom_left_corner,
					pair<float, float> top_right_corner, int start_pos,
					int level) {
	float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
		  x2 = top_right_corner.fi, y2 = top_right_corner.se;

	// Exit condition for recursion
	if (count < MIN_POINTS or
		(abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
		return new Grid(nullptr, nullptr, nullptr, nullptr, points,
						top_right_corner, bottom_left_corner, count);
	}

	vprint(
		"%d: Creating grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n",
		level, x1, y1, x2, y2, count, start_pos);

	// Define points for level and sub grid counts
	Point *d_points;
	int *d_grid_counts;

	// Define grid counts on host after kernel run
	vector<int> h_grid_counts(4);

	cudaMalloc(&d_points, count * sizeof(Point));
	cudaMalloc(&d_grid_counts, 4 * sizeof(int));

	cudaMemcpy(d_points, points, count * sizeof(Point), cudaMemcpyHostToDevice);

	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	// Call Kernel to reorder the points into 4 batches within d_grid_points
	float value = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
	int range = max(1.0, ceil(value));
	vprint("%d: Reorder in GPU: 1 block of %d threads each with range=%d\n",
		   level, MAX_THREADS_PER_BLOCK, range);
	reorder_points<<<1, MAX_THREADS_PER_BLOCK, 8 * sizeof(int)>>>(
		d_points, d_grid_points, d_grid_counts, count, range, middle_x,
		middle_y, start_pos, false);

	// Write the sub grid counts back to host
	cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
			   cudaMemcpyDeviceToHost);

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

	// Assign the 4 sub grid point arrays based on the returned counts
	Point *bl = (Point *)malloc(h_grid_counts[0] * sizeof(Point));
	Point *br = (Point *)malloc(h_grid_counts[1] * sizeof(Point));
	Point *tl = (Point *)malloc(h_grid_counts[2] * sizeof(Point));
	Point *tr = (Point *)malloc(h_grid_counts[3] * sizeof(Point));

	// Store the starting positions from d_grid_points for br, tl, tr
	int br_start_pos = start_pos + h_grid_counts[0],
		tl_start_pos = start_pos + h_grid_counts[0] + h_grid_counts[1],
		tr_start_pos =
			start_pos + h_grid_counts[0] + h_grid_counts[1] + h_grid_counts[2];

	// Shift the data from device's d_grid_points to host
	cudaMemcpy(bl, d_grid_points + start_pos, h_grid_counts[0] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(br, d_grid_points + br_start_pos,
			   h_grid_counts[1] * sizeof(Point), cudaMemcpyDeviceToHost);
	cudaMemcpy(tl, d_grid_points + tl_start_pos,
			   h_grid_counts[2] * sizeof(Point), cudaMemcpyDeviceToHost);
	cudaMemcpy(tr, d_grid_points + tr_start_pos,
			   h_grid_counts[3] * sizeof(Point), cudaMemcpyDeviceToHost);

	// Free data
	cudaFree(d_points);
	cudaFree(d_grid_counts);

	vprint(
		"%d: Completed grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n\n",
		level, x1, y1, x2, y2, count, start_pos);

	// Recursively call the quadtree grid function on each of the 4 sub grids -
	// bl, br, tl, tr and store in Grid struct
	Grid *bl_grid, *tl_grid, *br_grid, *tr_grid;
	bl_grid =
		quadtree_grid(bl, d_grid_points, h_grid_counts[0], bottom_left_corner,
					  mp(middle_x, middle_y), start_pos, level + 1);
	br_grid =
		quadtree_grid(br, d_grid_points, h_grid_counts[1], mp(middle_x, y1),
					  mp(x2, middle_y), br_start_pos, level + 1);
	tl_grid =
		quadtree_grid(tl, d_grid_points, h_grid_counts[2], mp(x1, middle_y),
					  mp(middle_x, y2), tl_start_pos, level + 1);
	tr_grid = quadtree_grid(tr, d_grid_points, h_grid_counts[3],
							mp(middle_x, middle_y), top_right_corner,
							tr_start_pos, level + 1);

	return new Grid(bl_grid, br_grid, tl_grid, tr_grid, points,
					top_right_corner, bottom_left_corner, count);
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

	Point *points_array = (Point *)malloc(point_count * sizeof(Point));
	for (int i = 0; i < point_count; i++) {
		points_array[i] = points[i];
	}

	// Store the d_grid_points array as points_array
	Point *d_grid_points;
	cudaMalloc(&d_grid_points, point_count * sizeof(Point));

	start = clock();
	Grid *root_grid = quadtree_grid(points_array, d_grid_points, point_count,
									mp(0.0, 0.0), mp(max_size, max_size), 0, 0);
	end = clock();

	// Free d_grid_points after entire grid is created
	cudaFree(d_grid_points);

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
