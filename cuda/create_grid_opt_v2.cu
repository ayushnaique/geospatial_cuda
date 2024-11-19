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
	GridArray *root_grid = construct_grid_array(
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
