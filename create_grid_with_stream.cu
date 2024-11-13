#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include "kernels.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <time.h>
#include <thread>
using namespace std;

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5.0
#define MIN_DISTANCE 5.0

void addGrid(Grid *current_grid, Grid *quad_grid, int i)
{
    switch (i)
    {
    case 0:
        current_grid->bottom_left = quad_grid;
        break;
    case 1:
        current_grid->bottom_right = quad_grid;
        break;
    case 2:
        current_grid->top_left = quad_grid;
        break;
    case 3:
        current_grid->top_right = quad_grid;
        break;

    default:
        break;
    }
}

void quadtree_grid(Point *points, int count,
                   pair<float, float> bottom_left_corner,
                   pair<float, float> top_right_corner, cudaStream_t stream, queue<Grid *> *grid_q)
{

    float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
          x2 = top_right_corner.fi, y2 = top_right_corner.se;
    // subdivide points into quadrants only if we have enough points to split
    if (count < MIN_POINTS or (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE))
    {
        printf("exit condition reached \n");
        return;
    }

    // Array of points for the geospatial data
    Point *d_points;

    // array to store the category of points (size = count) and the count of
    // points in each grid (size = 4)
    int *d_categories, *d_grid_counts;

    // Declare vectors to store the final values.
    vector<int> h_categories(count);
    vector<int> h_grid_counts(4);

    // Allocate memory to the pointers
    cudaMallocAsync(&d_points, count * sizeof(Point), stream);
    cudaMallocAsync(&d_categories, count * sizeof(int), stream);
    cudaMallocAsync(&d_grid_counts, 4 * sizeof(int), stream);

    // Copy the point data into device
    cudaMemcpyAsync(d_points, points, count * sizeof(Point),
                    cudaMemcpyHostToDevice, stream);

    // Set the number of blocks and threads per block
    int range, num_blocks = 16, threads_per_block = 256;

    // Calculate the work done by each thread
    float value =
        static_cast<float>(count) / (num_blocks * threads_per_block);
    range = max(1.0, ceil(value));
    printf("Categorize in GPU: %d blocks of %d threads each with range=%d\n",
           num_blocks, threads_per_block, range);

    dim3 grid(num_blocks, 1, 1);
    dim3 block(threads_per_block, 1, 1);

    // KERNEL Function to categorize points into 4 subgrids
    float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
    printf("middle_x = %d, middle_y = %d \n", middle_x, middle_y);
    categorize_points<<<grid, block, 4 * sizeof(int), stream>>>(
        d_points, d_categories, d_grid_counts, count, range, middle_x,
        middle_y);

    // Get back the data from device to host
    cudaMemcpyAsync(h_categories.data(), d_categories, count * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    int total = 0;
    // printf("%d: Point counts per sub grid - \n", level);
    for (int i = 0; i < 4; i++)
    {
        printf("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
        total += h_grid_counts[i];
    }
    printf("Total Count - %d\n", count);
    if (total == count)
    {
        printf("Sum of sub grid counts matches total point count\n");
    }

    // Declare arrays for each section of the grid and allocate memory depending
    // on the number of points found
    Point *bottom_left, *bottom_right, *top_left, *top_right;
    cudaMallocAsync(&bottom_left, h_grid_counts[0] * sizeof(Point), stream);
    cudaMallocAsync(&bottom_right, h_grid_counts[1] * sizeof(Point), stream);
    cudaMallocAsync(&top_left, h_grid_counts[2] * sizeof(Point), stream);
    cudaMallocAsync(&top_right, h_grid_counts[3] * sizeof(Point), stream);

    dim3 grid2(1, 1, 1);
    dim3 block2(threads_per_block, 1, 1);

    // KERNEL Function to assign the points to its respective array
    value = static_cast<float>(count) / threads_per_block;
    range = max(1.0, ceil(value));
    printf("Organize in GPU: 1 block of %d threads each with range=%d\n",
           threads_per_block, range);
    organize_points<<<grid2, block2, 4 * sizeof(int), stream>>>(
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
    cudaMemcpyAsync(bl, bottom_left, h_grid_counts[0] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(br, bottom_right, h_grid_counts[1] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(tl, top_left, h_grid_counts[2] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(tr, top_right, h_grid_counts[3] * sizeof(Point),
                    cudaMemcpyDeviceToHost, stream);

    Grid *bottom_left_grid = new Grid(nullptr, nullptr, nullptr, nullptr, bl, mp(x1, y1), mp(middle_x, middle_y), h_grid_counts[0]);
    Grid *bottom_right_grid = new Grid(nullptr, nullptr, nullptr, nullptr, br, mp(middle_x, y1), mp(x2, middle_y), h_grid_counts[1]);
    Grid *top_left_grid = new Grid(nullptr, nullptr, nullptr, nullptr, tl, mp(x1, middle_y), mp(middle_x, y2), h_grid_counts[2]);
    Grid *top_right_grid = new Grid(nullptr, nullptr, nullptr, nullptr, tr, mp(middle_x, middle_y), mp(x2, y2), h_grid_counts[3]);

    grid_q->push(bottom_left_grid);
    grid_q->push(bottom_right_grid);
    grid_q->push(top_left_grid);
    grid_q->push(top_right_grid);

    // Free data
    cudaFreeAsync(d_points, stream);
    cudaFreeAsync(d_categories, stream);
    cudaFreeAsync(d_grid_counts, stream);
    cudaFreeAsync(bottom_left, stream);
    cudaFreeAsync(bottom_right, stream);
    cudaFreeAsync(top_left, stream);
    cudaFreeAsync(top_right, stream);

    cudaStreamSynchronize(stream);

    return;
}

Grid *build_quadtree_levels(Point *points, int point_count, queue<Grid *> *grid_q, pair<float, float> bl, pair<float, float> tr)
{

    // According to GPU documentations, 32 is the limit to the number of streams
    // but performance can not be gauranteed to be better with that many streams because of the limited number of SMs
    // We limit our streams to 4 right now
    double time_taken;
    clock_t start, end;
    start = clock();
    // current grid keeps changing depending on the stream
    Grid *current_grid;

    queue<Grid *> recursive_grids;
    Grid *root_grid = new Grid(nullptr, nullptr, nullptr, nullptr, points, bl, tr, point_count);
    recursive_grids.push(root_grid);

    quadtree_grid(points, point_count, bl, tr, nullptr, grid_q);

    while (!grid_q->empty())
    {
        // start 4 streams at a time, one for each bl, br, tl, tr points
        int batch = (grid_q->size() == 1) ? 1 : 4;
        cudaStream_t streams[batch];
        // Initialize each stream to nullptr so that we don't get segmentation faults if we exit the grid creation early
        for (int i = 0; i < batch; ++i)
        {
            streams[i] = nullptr;
        }

        if (!recursive_grids.empty())
        {
            current_grid = recursive_grids.front();
            recursive_grids.pop();
        }

        for (int i = 0; i < batch; i++)
        {
            if (grid_q->empty())
                break;

            Grid *popped_grid = grid_q->front();
            grid_q->pop();

            if (current_grid != nullptr && popped_grid)
            {
                addGrid(current_grid, popped_grid, i);
                recursive_grids.push(popped_grid);
            }

            int x1 = popped_grid->bottomLeft.fi, y1 = popped_grid->bottomLeft.se,
                x2 = popped_grid->topRight.fi, y2 = popped_grid->topRight.se;
            if (!(popped_grid->count < MIN_POINTS or (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)))
            {
                cudaStreamCreate(&(streams[i]));
                printf("Stream %d created \n", i);
                quadtree_grid(popped_grid->points, popped_grid->count, popped_grid->bottomLeft, popped_grid->topRight, streams[i], grid_q);
            }
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < batch; i++)
        {
            if (streams[i] != nullptr)
            {
                cudaStreamDestroy(streams[i]);
            }
        }
    }

    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);

    return root_grid;
}

int main(int argc, char *argv[])
{
    int initial_bl_fi;
    int initial_bl_se;
    int initial_tr_fi;
    int initial_tr_se;

    if (argc != 5)
    {
        fprintf(stderr, "usage: grid bl=(initial_bottom_left, initial_bottom_left) and grid tr=(initial_top_right, initial_top_right) \n");
        fprintf(stderr, "inital bottom left point must be mentioned (as a two space-sepaarted ints) \n");
        fprintf(stderr, "inital top right point must be mentioned (as a two space-sepaarted ints) \n");
        exit(1);
    }

    initial_bl_fi = (unsigned int)atoi(argv[1]);
    initial_bl_se = (unsigned int)atoi(argv[2]);
    initial_tr_fi = (unsigned int)atoi(argv[3]);
    initial_tr_se = (unsigned int)atoi(argv[4]);

    string filename = "points.txt";
    vector<Point> points;
    int point_count = 0;

    ifstream file(filename);
    if (!file)
    {
        cerr << "Error: Could not open the file " << filename << endl;
        return 1;
    }

    string line;
    int x, y;

    while (getline(file, line))
    {
        istringstream iss(line);
        if (iss >> x >> y)
        {
            Point p = Point(x, y);
            points.emplace_back(p);
            point_count++;
        }
        else
        {
            cerr << "Warning: Skipping malformed line: " << line << endl;
        }
    }

    file.close();

    pair<float, float> root_bl = mp(initial_bl_fi, initial_bl_se);
    pair<float, float> root_tr = mp(initial_tr_fi, initial_tr_se);

    queue<Grid *> grid_q;
    Grid *root_grid = build_quadtree_levels(&points[0], point_count, &grid_q, root_bl, root_tr);

    printf("Validating grid...\n");

    bool check = validateGrid(root_grid, root_tr, root_bl);

    if (check == true)
        printf("Grid Verification Success!\n");
    else
        printf("Grid Verification Failure!\n");

    return 0;
}
