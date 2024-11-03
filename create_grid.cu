#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
using namespace std;

struct Point {
	int x, y;

	Point(int xc, int yc) : x(xc), y(yc) {}
};

struct Grid {
    Point *bottom_left, *bottom_right, *top_left, *top_right;

    Grid(Point* bl, Point* br, Point* tl, Point* tr)
                : bottom_left(bl), bottom_right(br), top_left(tl), top_right(tr) {}
};

__global__ void categorize_points(Point *d_points, int *d_categories, int *grid_counts, int count,
								  int range, int middle) {
    extern __shared__ int subgrid_counts[];

	int start = ((blockIdx.x * blockDim.x) + threadIdx.x) * range;
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
            if (d_points[i].x <= middle and d_points[i].y <= middle){
				d_categories[i] = 1;
                first++;
            }
            else if (d_points[i].x > middle and d_points[i].y <= middle){
				d_categories[i] = 2;
                second++;
            }
            else if (d_points[i].x <= middle and d_points[i].y > middle){
				d_categories[i] = 3;
                third++;
            }
            else if (d_points[i].x > middle and d_points[i].y > middle){
				d_categories[i] = 4;
                fourth++;
            }
		}
	}
    atomicAdd(&subgrid_counts[0], first);
    atomicAdd(&subgrid_counts[1], second);
    atomicAdd(&subgrid_counts[2], third);
    atomicAdd(&subgrid_counts[3], fourth);
    __syncthreads();


    if (threadIdx.x == 0) {
        atomicAdd(&grid_counts[0], subgrid_counts[0]);
        atomicAdd(&grid_counts[1], subgrid_counts[1]);
        atomicAdd(&grid_counts[2], subgrid_counts[2]);
        atomicAdd(&grid_counts[3], subgrid_counts[3]);
    }
}

void quadtree_grid(vector<Point> points, int count, int dimension) {
	Point *d_points;

	int *d_categories, *d_grid_counts;
    vector<int> h_categories(count);
    vector<int> h_grid_counts(4);

	cudaMalloc(&d_points, count * sizeof(Point));
	cudaMalloc(&d_categories, count * sizeof(int));
	cudaMalloc(&d_grid_counts, 4 * sizeof(int));

	cudaMemcpy(d_points, points.data(), count * sizeof(Point),
			   cudaMemcpyHostToDevice);

	int range, num_blocks = 16, threads_per_block = 256;
	if (count < num_blocks * threads_per_block)
		range = 1;
	else if (count % (num_blocks * threads_per_block) == 0)
		range = count / (threads_per_block * num_blocks);
	else {
		float value = static_cast<float>(count) / (num_blocks * threads_per_block);
		range = std::ceil(value);
	}
	printf("GPU: %d blocks of %d threads each with range=%d\n", num_blocks,
		   threads_per_block, range);

	dim3 grid(num_blocks, 1, 1);
	dim3 block(threads_per_block, 1, 1);
	categorize_points<<<grid, block, 4 * sizeof(int)>>>(d_points, d_categories, d_grid_counts, count, range,
									   dimension / 2);

	cudaMemcpy(h_categories.data(), d_categories, count * sizeof(int),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
			   cudaMemcpyDeviceToHost);

    cudaFree(d_points);
    cudaFree(d_categories);
    cudaFree(d_grid_counts);

    //for(int i = 0; i<1000; i++){
        //printf("x = %d, y = %d, category = %d\n", points[i].x, points[i].y, h_categories[i]);
    //}
    //int total = 0;
    //for(int i = 0; i<4; i++){
        //printf("sub grid %d - %d\n", i+1, h_grid_counts[i]);
        //total += h_grid_counts[i];
    //}
    //printf("Total Count - %d\n", count);
    //if(total == count){
        //printf("Matches\n");
    //}
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

	quadtree_grid(points, point_count, 1000);

	return 0;
}
