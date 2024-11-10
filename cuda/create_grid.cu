#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <kernels.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5.0
#define MIN_DISTANCE 5.0

Grid *quadtree_grid(Point *points, int count, pair<float, float> bottom_left_corner,
					pair<float, float> top_right_corner, int level) {
	float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
		x2 = top_right_corner.fi, y2 = top_right_corner.se;

	if (count < MIN_POINTS or
		(abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
			pair<float, float> upperBound = make_pair(x2, y2);
			pair<float, float> lowerBound = make_pair(x1, y1);
			return new Grid(nullptr, nullptr, nullptr, nullptr, points, upperBound, lowerBound, count);
	}

	printf("%d: Creating grid from (%f,%f) to (%f,%f) for %d points\n", level,
		   x1, y1, x2, y2, count);

	// Array of points for the geospatial data
	Point *d_points;

	// array to store the category of points (size = count) and the count of
	// points in each grid (size = 4)
	int *d_categories, *d_grid_counts;

	// Declare vectors to store the final values.
	vector<int> h_categories(count);
	vector<int> h_grid_counts(4);

	// Allocate memory to the pointers
	cudaMalloc(&d_points, count * sizeof(Point));
	cudaMalloc(&d_categories, count * sizeof(int));
	cudaMalloc(&d_grid_counts, 4 * sizeof(int));

	// Copy the point data into device
	cudaMemcpy(d_points, points, count * sizeof(Point), cudaMemcpyHostToDevice);

	// Set the number of blocks and threads per block
	int range, num_blocks = 16, threads_per_block = 256;

	// Calculate the work done by each thread
	float value = static_cast<float>(count) / (num_blocks * threads_per_block);
	range = max(1.0, ceil(value));

	dim3 grid(num_blocks, 1, 1);
	dim3 block(threads_per_block, 1, 1);

	// KERNEL Function to categorize points into 4 subgrids
	float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	printf("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

	printf(
		"%d: Categorize in GPU: %d blocks of %d threads each with range=%d\n",
		level, num_blocks, threads_per_block, range);
	categorize_points<<<grid, block, 4 * sizeof(int)>>>(
		d_points, d_categories, d_grid_counts, count, range, middle_x,
		middle_y);

	// Get back the data from device to host
	cudaMemcpy(h_categories.data(), d_categories, count * sizeof(int),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
			   cudaMemcpyDeviceToHost);

	int total = 0;
	printf("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++) {
		printf("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
		total += h_grid_counts[i];
	}
	printf("Total Count - %d\n", count);
	if (total == count) {
		printf("Sum of sub grid counts matches total point count\n");
	}

	// Declare arrays for each section of the grid and allocate memory depending
	// on the number of points found
	Point *bottom_left, *bottom_right, *top_left, *top_right;
	cudaMalloc(&bottom_left, h_grid_counts[0] * sizeof(Point));
	cudaMalloc(&bottom_right, h_grid_counts[1] * sizeof(Point));
	cudaMalloc(&top_left, h_grid_counts[2] * sizeof(Point));
	cudaMalloc(&top_right, h_grid_counts[3] * sizeof(Point));

	dim3 grid2(1, 1, 1);
	dim3 block2(threads_per_block, 1, 1);

	// KERNEL Function to assign the points to its respective array
	value = static_cast<float>(count) / threads_per_block;
	range = max(1.0, ceil(value));
	printf("%d: Organize in GPU: 1 block of %d threads each with range=%d\n",
		   level, threads_per_block, range);
	organize_points<<<grid2, block2, 4 * sizeof(int)>>>(
		d_points, d_categories, bottom_left, bottom_right, top_left, top_right,
		count, range);

	// Declare the final array in which we store the sorted points according to
	// the location in the grid
	Point *bl, *br, *tl, *tr;
	bl = (Point *)malloc(h_grid_counts[0] * sizeof(Point));
	br = (Point *)malloc(h_grid_counts[1] * sizeof(Point));
	tl = (Point *)malloc(h_grid_counts[2] * sizeof(Point));
	tr = (Point *)malloc(h_grid_counts[3] * sizeof(Point));

	// Shift the data from device to host
	cudaMemcpy(bl, bottom_left, h_grid_counts[0] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(br, bottom_right, h_grid_counts[1] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(tl, top_left, h_grid_counts[2] * sizeof(Point),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(tr, top_right, h_grid_counts[3] * sizeof(Point),
			   cudaMemcpyDeviceToHost);

	// Free data
	cudaFree(d_points);
	cudaFree(d_categories);
	cudaFree(d_grid_counts);
	cudaFree(bottom_left);
	cudaFree(bottom_right);
	cudaFree(top_left);
	cudaFree(top_right);

	printf("\n\n");

	// Recursively call the quadtree grid function on each of the 4 sub grids -
	// bl, br, tl, tr and store in Grid struct
	Grid *bl_grid, *tl_grid, *br_grid, *tr_grid;
	bl_grid = quadtree_grid(bl, h_grid_counts[0], bottom_left_corner,
							mp(middle_x, middle_y), level + 1);
	br_grid = quadtree_grid(br, h_grid_counts[1], mp(middle_x, y1),
							mp(x1, middle_y), level + 1);
	tl_grid = quadtree_grid(tl, h_grid_counts[2], mp(x1, middle_y),
							mp(middle_x, y2), level + 1);
	tr_grid = quadtree_grid(tr, h_grid_counts[3], mp(middle_x, middle_y),
							top_right_corner, level + 1);

	// The bounds of the grid
	pair<float, float> upperBound = make_pair(x2, y2);
	pair<float, float> lowerBound = make_pair(x1, y1);

	return new Grid(bl_grid, br_grid, tl_grid, tr_grid, points, upperBound, lowerBound, count);
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

	Point *points_array = (Point *)malloc(point_count * sizeof(Point));
	for (int i = 0; i < point_count; i++) {
		points_array[i] = points[i];
	}
	Grid *root_grid = quadtree_grid(points_array, point_count, mp(0.0, 0.0), mp(max_size, max_size), 0);

	printf("Validating grid...\n");
	pair<float, float> lowerBound = make_pair(0.0, 0.0);
	pair<float, float> upperBound = make_pair(max_size, max_size);
	bool check = validateGrid(root_grid, upperBound, lowerBound);
	
	if(check == true)
		printf("Grid Verification Success!\n");
	else
		printf("Grid Verification Failure!\n");

	return 0;
}
