#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

#define mp make_pair
#define fi first
#define se second

struct Point {
	int x, y;

	Point(int xc, int yc) : x(xc), y(yc) {}
};

struct Grid {
	Grid bottom_left, bottom_right, top_left, top_right;
	Point *points;

	// Initialize the corresponding Point values
	Grid(Grid bl, Grid br, Grid tl, Grid tr, Point *ps)
		: bottom_left(bl),
		  bottom_right(br),
		  top_left(tl),
		  top_right(tr),
		  points(ps) {}
};

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
				d_categories[i] = 1;
				first++;
			}
			// bottom right; if point lies in bottom right, increment
			else if (d_points[i].x > middle_x and d_points[i].y <= middle_y) {
				d_categories[i] = 2;
				second++;
			}
			// top left; if point lies in top left, increment
			else if (d_points[i].x <= middle_x and d_points[i].y > middle_y) {
				d_categories[i] = 3;
				third++;
			}
			// top right; if point lies in top right, increment
			else if (d_points[i].x > middle_x and d_points[i].y > middle_y) {
				d_categories[i] = 4;
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

	for (int i = threadIdx.x; i < threadIdx.x + range; i++) {
		if (i < count) {
			// Point array will store the respective points in a contiguous
			// fashion increment subgrid index according to the category
			if (d_categories[i] == 1) {
				bl[subgrid_index[0]] = d_points[i];
				atomicAdd(&subgrid_index[0], 1);
			}
			if (d_categories[i] == 2) {
				br[subgrid_index[1]] = d_points[i];
				atomicAdd(&subgrid_index[1], 1);
			}
			if (d_categories[i] == 3) {
				tl[subgrid_index[2]] = d_points[i];
				atomicAdd(&subgrid_index[2], 1);
			}
			if (d_categories[i] == 4) {
				tr[subgrid_index[3]] = d_points[i];
				atomicAdd(&subgrid_index[3], 1);
			}
		}
	}
}

void quadtree_grid(vector<Point> points, int count,
				   pair<int, int> bottom_left_corner,
				   pair<int, int> top_right_corner) {
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
	cudaMemcpy(d_points, points.data(), count * sizeof(Point),
			   cudaMemcpyHostToDevice);

	// Set the number of blocks and threads per block
	int range, num_blocks = 16, threads_per_block = 256;

	// Calculate the work done by each thread
	if (count < num_blocks * threads_per_block)
		range = 1;
	else if (count % (num_blocks * threads_per_block) == 0)
		range = count / (threads_per_block * num_blocks);
	else {
		float value =
			static_cast<float>(count) / (num_blocks * threads_per_block);
		range = std::ceil(value);
	}
	printf("Categorize in GPU: %d blocks of %d threads each with range=%d\n",
		   num_blocks, threads_per_block, range);

	dim3 grid(num_blocks, 1, 1);
	dim3 block(threads_per_block, 1, 1);

	// KERNEL Function to categorize points into 4 subgrids
	int middle_x = (top_right_corner.fi - bottom_left_corner.fi) / 2,
		middle_y = (top_right_corner.se - bottom_left_corner.se) / 2;
	categorize_points<<<grid, block, 4 * sizeof(int)>>>(
		d_points, d_categories, d_grid_counts, count, range, middle_x,
		middle_y);

	// Get back the data from device to host
	cudaMemcpy(h_categories.data(), d_categories, count * sizeof(int),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
			   cudaMemcpyDeviceToHost);

	int total = 0;
	printf("Point counts per sub grid - \n");
	for (int i = 0; i < 4; i++) {
		printf("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
		total += h_grid_counts[i];
	}
	printf("Total Count - %d\n", count);
	if (total == count) {
		printf("Sum of sub grid counts matches total point count\n\n");
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
	float value = static_cast<float>(count) / threads_per_block;
	range = std::ceil(value);
	printf("Organize in GPU: 1 block of %d threads each with range=%d\n",
		   threads_per_block, range);
	organize_points<<<grid2, block2, 4 * sizeof(int)>>>(
		d_points, d_categories, bottom_left, bottom_right, top_left, top_right,
		count, count / threads_per_block);

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

	printf("Sample Point in bottom left - %d %d\n", bl[0].x, bl[0].y);
	printf("Sample Point in bottom right - %d %d\n", br[0].x, br[0].y);
	printf("Sample Point in top left - %d %d\n", tl[0].x, tl[0].y);
	printf("Sample Point in top right - %d %d\n", tr[0].x, tr[0].y);

	// TODO:
	// Recursively call the quadtree grid function on each of the 4 sub grids -
	// bl, br, tl, tr and store in Grid struct

	// Free data
	cudaFree(d_points);
	cudaFree(d_categories);
	cudaFree(d_grid_counts);
	cudaFree(bottom_left);
	cudaFree(bottom_right);
	cudaFree(top_left);
	cudaFree(top_right);
}

int main() {
	string filename = "points.txt";
	vector<Point> points;
	int point_count = 0;

	ifstream file(filename);
	if (!file) {
		cerr << "Error: Could not open the file " << filename << endl;
		return 1;
	}

	string line;
	int x, y;

	while (getline(file, line)) {
		istringstream iss(line);
		if (iss >> x >> y) {
			Point p = Point(x, y);
			points.emplace_back(p);
			point_count++;
		} else {
			cerr << "Warning: Skipping malformed line: " << line << endl;
		}
	}

	file.close();

	quadtree_grid(points, point_count, mp(0, 0), mp(1000, 1000));

	return 0;
}
