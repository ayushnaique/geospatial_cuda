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
    if (VERBOSE)     \
    {                \
        printf(s);   \
    }

void addGrid(GridArray *current_grid, GridArray *quad_grid, int i)
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

pair<Point *, Point *> quadtree_grid(Point *d_grid_points0, Point *d_grid_points1,
                                     int count, pair<float, float> bottom_left_corner,
                                     pair<float, float> top_right_corner, int start_pos,
                                     int grid_array_flag, cudaStream_t stream,
                                     queue<GridArray *> *grid_q)
{
    float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
          x2 = top_right_corner.fi, y2 = top_right_corner.se;

    // Exit condition for recursion
    if (count < MIN_POINTS or
        (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE))
    {
        return mp(d_grid_points0, d_grid_points1);
    }

    vprint(
        "Creating grid from (%f,%f) to (%f,%f) for %d points with "
        "start_pos=%d\n",
        x1, y1, x2, y2, count, start_pos);

    // Define points for level and sub grid counts
    int *d_grid_counts;

    // Define grid counts on host after kernel run
    vector<int> h_grid_counts(4);

    cudaMallocAsync(&d_grid_counts, 4 * sizeof(int), stream);

    float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
    vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

    // Call Kernel to reorder the points into 4 batches within d_grid_points
    float value = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
    int range = max(1.0, ceil(value));
    vprint("Reorder in GPU: 1 block of %d threads each with range=%d\n",
           MAX_THREADS_PER_BLOCK, range);
    reorder_points<<<1, MAX_THREADS_PER_BLOCK, 8 * sizeof(int), stream>>>(
        d_grid_points0, d_grid_points1, d_grid_counts, count, range, middle_x,
        middle_y, start_pos, true);

    cudaDeviceSynchronize();

    // Write the sub grid counts back to host
    cudaMemcpyAsync(h_grid_counts.data(), d_grid_counts, 4 * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_grid_counts, stream);

    int total = 0;
    vprint("Point counts per sub grid - \n");
    for (int i = 0; i < 4; i++)
    {
        vprint("sub grid %d - %d\n", i + 1, h_grid_counts[i]);
        total += h_grid_counts[i];
    }
    vprint("Total Count - %d\n", count);
    if (total == count)
    {
        vprint("Sum of sub grid counts matches total point count\n");
    }

    // Store the starting positions from d_grid_points for br, tl, tr
    int br_start_pos = start_pos + h_grid_counts[0],
        tl_start_pos = start_pos + h_grid_counts[0] + h_grid_counts[1],
        tr_start_pos =
            start_pos + h_grid_counts[0] + h_grid_counts[1] + h_grid_counts[2];

    vprint(
        "Completed grid from (%f,%f) to (%f,%f) for %d points with "
        "start_pos=%d\n\n",
        x1, y1, x2, y2, count, start_pos);

    // Recursively call the quadtree grid function on each of the 4 sub grids
    GridArray *bl_grid, *tl_grid, *br_grid, *tr_grid;

    bl_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(middle_x, middle_y), bottom_left_corner, h_grid_counts[0],
                      start_pos, grid_array_flag ^ 1);

    br_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(x2, middle_y),
                      mp(middle_x, y1), h_grid_counts[1],
                      br_start_pos, grid_array_flag ^ 1);

    tl_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(middle_x, y2),
                      mp(x1, middle_y), h_grid_counts[2],
                      tl_start_pos, grid_array_flag ^ 1);

    tr_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(x2, y2),
                      mp(middle_x, middle_y), h_grid_counts[3],
                      tr_start_pos, grid_array_flag ^ 1);

    // return new GridArray(bl_grid, br_grid, tl_grid, tr_grid, top_right_corner,
    // 					 bottom_left_corner, count, start_pos, grid_array_flag);

    grid_q->push(bl_grid);
    grid_q->push(br_grid);
    grid_q->push(tl_grid);
    grid_q->push(tr_grid);

    cudaStreamSynchronize(stream);

    return mp(d_grid_points1, d_grid_points0);
}

Grid *build_quadtree_levels(vector<Point> points, int point_count,
                            queue<GridArray *> *grid_q, pair<float, float> bl,
                            pair<float, float> tr)
{
    double time_taken;
    clock_t start, end;

    start = clock();

    // Store the d_grid_points array as points_array
    Point *d_grid_points0, *d_grid_points1;
    cudaMalloc(&d_grid_points0, point_count * sizeof(Point));
    cudaMalloc(&d_grid_points1, point_count * sizeof(Point));

    cudaMemcpy(d_grid_points0, points.data(), point_count * sizeof(Point),
               cudaMemcpyHostToDevice);

    // current grid keeps changing depending on the stream
    GridArray *current_grid;

    queue<GridArray *> recursive_grids;
    pair<Point *, Point *> grid_ptrs_2d;
    GridArray *root_grid = new GridArray(nullptr, nullptr, nullptr, nullptr, tr, bl, point_count, 0, 0);
    recursive_grids.push(root_grid);

    // Pre-allocate a fixed number of streams (4 streams in this case)
    const int maxStreams = 4;
    cudaStream_t streams[maxStreams];

    // Reusing streams to avoid re-creating and destroying them
    for (int i = 0; i < maxStreams; ++i)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Initial quadtree grid call
    grid_ptrs_2d = quadtree_grid(d_grid_points0, d_grid_points1, point_count, bl,
                                 tr, 0, 0, nullptr, grid_q);

    // Stream availability tracking
    bool streamBusy[maxStreams] = {false};

    while (!grid_q->empty())
    {
        for (int i = 0; i < maxStreams; ++i)
        {
            // Check if the stream is available
            if (!streamBusy[i] || cudaStreamQuery(streams[i]) == cudaSuccess)
            {
                streamBusy[i] = false;

                if (!grid_q->empty())
                {
                    GridArray *popped_grid = grid_q->front();
                    grid_q->pop();

                    if (!recursive_grids.empty())
                    {
                        current_grid = recursive_grids.front();
                        recursive_grids.pop();
                    }

                    if (current_grid != nullptr && popped_grid != nullptr)
                    {
                        addGrid(current_grid, popped_grid, i);
                        recursive_grids.push(popped_grid);
                    }

                    float x1 = popped_grid->bottom_left_corner.fi,
                          y1 = popped_grid->bottom_left_corner.se,
                          x2 = popped_grid->top_right_corner.fi,
                          y2 = popped_grid->top_right_corner.se;

                    if (!(popped_grid->count < MIN_POINTS ||
                          (abs(x1 - x2) < MIN_DISTANCE && abs(y1 - y2) < MIN_DISTANCE)))
                    {
                        // Launch quadtree_grid asynchronously in the selected stream
                        grid_ptrs_2d = quadtree_grid(grid_ptrs_2d.fi, grid_ptrs_2d.se, popped_grid->count,
                                                     popped_grid->bottom_left_corner,
                                                     popped_grid->top_right_corner, popped_grid->start_pos, popped_grid->grid_array_flag, streams[i],
                                                     grid_q);
                        streamBusy[i] = true; // Mark stream as busy
                    }
                }
            }
        }
    }

    // Destroy streams after use
    for (int i = 0; i < maxStreams; ++i)
    {
        cudaStreamSynchronize(streams[i]); // Ensure all tasks in streams are done
        cudaStreamDestroy(streams[i]);
    }

    Point *grid_array0 = (Point *)malloc(point_count * sizeof(Point));
    Point *grid_array1 = (Point *)malloc(point_count * sizeof(Point));
    cudaMemcpy(grid_array0, d_grid_points0, point_count * sizeof(Point),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_array1, d_grid_points1, point_count * sizeof(Point),
               cudaMemcpyDeviceToHost);

    cudaFree(d_grid_points0);
    cudaFree(d_grid_points1);

    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);

    Grid *root = assign_points(root_grid, grid_array0, grid_array1);

    return root;
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " file_path max_boundary"
                  << std::endl;
        return 1;
    }

    string filename = argv[1];
    float max_size = atof(argv[2]);

    ifstream file(filename);
    if (!file)
    {
        cerr << "Error: Could not open the file " << filename << endl;
        return 1;
    }

    string line;
    float x, y;
    vector<Point> points;
    int point_count = 0;
    while (getline(file, line))
    {
        istringstream iss(line);
        if (iss >> x >> y)
        {
            Point p = Point((float)x, (float)y);
            points.emplace_back(p);
            point_count++;
        }
        else
        {
            cerr << "Warning: Skipping malformed line: " << line << endl;
        }
    }

    file.close();

    pair<float, float> root_bl = mp(0, 0);
    pair<float, float> root_tr = mp(max_size, max_size);

    queue<GridArray *> grid_q;
    Grid *root_grid = build_quadtree_levels(points, point_count, &grid_q,
                                            root_bl, root_tr);

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
