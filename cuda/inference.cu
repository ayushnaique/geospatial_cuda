#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "kernels.h"
using namespace std;

#define mp make_pair
#define MIN_POINTS 5.0

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

	vector<QuadrantBoundary> boundaries;
	unordered_map<int, Grid *> grid_map;

	prepare_boundaries(root, 0, nullptr, boundaries, grid_map);

	vector<Query> queries = {
		{'s', Point(637093.0, 90101.0)},
		{'i', Point(9981.0, 9979.0)},
		{'s', Point(9981.0, 9979.0)},
		{'s', Point(100.0, 100.0)},
		{'d', Point(9981.0, 9979.0)},
		{'s', Point(9981.0, 9979.0)}
		// Add more queries as needed
	};

	// Test Search
	vector<int> results = search_quadrant(queries, boundaries);
	for (int i = 0; i < results.size(); i++) {
		printf("\n");
		printf("The point to be searched (%f, %f) with a quadrant id: %d \n",
			   queries[i].point.x, queries[i].point.y, results[i]);
		if (results[i] > 0) {
			auto it = grid_map.find(results[i]);
			if (it != grid_map.end()) {
				Grid *current_grid = it->second;
				bool found = false;
				for (int j = 0; j < current_grid->count; j++) {
					if (current_grid->points[j].x == queries[i].point.x &&
						current_grid->points[j].y == queries[i].point.y) {
						found = true;
						break;
					}
				}
				printf("The type of the query is: %c \n", queries[i].type);
				switch (queries[i].type) {
					case 's':
						if (found)
							printf("Point found in quadrant with ID: %d\n",
								   results[i]);
						else
							printf("Point not found in the grid.\n");
						break;
					case 'i':
						printf("Inserting a point \n");
						if (found)
							printf(
								"Point already exists in quadrant with ID: "
								"%d\n",
								results[i]);
						else
							insert_point(queries[i].point, grid_map[results[i]],
										 boundaries);
						break;
					case 'd':
						printf("Deleting a point \n");
						if (found)
							delete_point(queries[i].point, grid_map[results[i]],
										 boundaries, grid_map);
						else
							printf("Point does not exist in the grid \n");
				}
			} else {
				printf("Quadrant with ID %d not found in the map.\n",
					   results[i]);
			}
		}
	}

	return 0;
}
