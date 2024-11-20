#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <time.h>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <omp.h>

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
                                     queue<GridArray *> *grid_q, int* h_grid_count, int* d_grid_count, int d_start, int level, omp_lock_t &lock_q) {
    // printf("streamid: %d\n", d_start);
    // unzip values for faster access
    float x1 = bottom_left_corner.fi, y1 = bottom_left_corner.se,
          x2 = top_right_corner.fi, y2 = top_right_corner.se;

    // Exit condition for recursion
    if (count < MIN_POINTS || 
        (abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
        return mp(d_grid_points0, d_grid_points1);
    }

    vprint(
		"%d: Creating grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n",
		level, x1, y1, x2, y2, count, start_pos);

    float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
    vprint("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

    // Call Kernel to reorder the points into 4 batches within d_grid_points
    float value = static_cast<float>(count) / MAX_THREADS_PER_BLOCK;
    int range = max(1.0, ceil(value));
    vprint("%d: Reorder in GPU: 1 block of %d threads each with range=%d\n",
		   level, MAX_THREADS_PER_BLOCK, range);
    reorder_points_h_alloc_stream<<<1, MAX_THREADS_PER_BLOCK, 8 * sizeof(int), stream>>>(
        d_grid_points0, d_grid_points1, count, range, middle_x,
        middle_y, start_pos, true, d_grid_count, d_start);

    // cudaDeviceSynchronize();
    cudaStreamSynchronize(stream);

    int total = 0;
    vprint("%d: Point counts per sub grid - \n", level);
    for (int i = 0; i < 4; i++)
    {
        vprint("sub grid %d - %d\n", i + 1, h_grid_count[i + d_start * 4]);
        total += h_grid_count[i + d_start * 4];
    }
    vprint("%d, Total Count - %d\n", level, count);
    if (total == count)
    {
        vprint("%d: Sum of sub grid counts matches total point count\n", level);
    }

    // Store the starting positions from d_grid_points for br, tl, tr
    int br_start_pos = start_pos + h_grid_count[d_start * 4],
        tl_start_pos = start_pos + h_grid_count[d_start * 4] + h_grid_count[d_start * 4 + 1],
        tr_start_pos =
            start_pos + h_grid_count[d_start * 4] + h_grid_count[d_start * 4 + 1] + h_grid_count[d_start * 4 + 2];

    vprint(
		"%d: Completed grid from (%f,%f) to (%f,%f) for %d points with "
		"start_pos=%d\n\n",
		level, x1, y1, x2, y2, count, start_pos);

	int g1 = h_grid_count[d_start * 4], g2 = h_grid_count[d_start * 4 + 1], g3 = h_grid_count[d_start * 4 + 2], g4 = h_grid_count[d_start * 4 + 3];
    
    // Recursively call the quadtree grid function on each of the 4 sub grids
    GridArray *bl_grid, *tl_grid, *br_grid, *tr_grid;

    bl_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(middle_x, middle_y), bottom_left_corner, g1,
                      start_pos, grid_array_flag ^ 1);

    br_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(x2, middle_y),
                      mp(middle_x, y1), g2,
                      br_start_pos, grid_array_flag ^ 1);

    tl_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(middle_x, y2),
                      mp(x1, middle_y), g3,
                      tl_start_pos, grid_array_flag ^ 1);

    tr_grid =
        new GridArray(nullptr, nullptr, nullptr, nullptr,
                      mp(x2, y2),
                      mp(middle_x, middle_y), g4,
                      tr_start_pos, grid_array_flag ^ 1);


    omp_set_lock(&lock_q); // Acquire the lock
    {
        grid_q->push(bl_grid);
        grid_q->push(br_grid);
        grid_q->push(tl_grid);
        grid_q->push(tr_grid);
    }
    omp_unset_lock(&lock_q); // Release the lock

    cudaStreamSynchronize(stream);

    return mp(d_grid_points1, d_grid_points0);
}

Grid *build_quadtree_levels(vector<Point> points, int point_count,
                            queue<GridArray *> *grid_q, pair<float, float> bl,
                            pair<float, float> tr, int level) {
    double time_taken;

	Point *points_array = (Point *)malloc(point_count * sizeof(Point));
	for (int i = 0; i < point_count; i++) {
		points_array[i] = points[i];
	}

	// Store the d_grid_points array as points_array
	Point *d_grid_points0;
	cudaMalloc(&d_grid_points0, point_count * sizeof(Point));

    // Store all the points from the input file into the Device 
    Point * d_grid_points1;
    cudaMalloc(&d_grid_points1, point_count * sizeof(Point));

    // Allocate page-locked memory (host memory)
    int* h_grid_count;
    cudaHostAlloc(&h_grid_count, 16 * sizeof(int), cudaHostAllocMapped);

    // Allocate device memory (GPU memory)
    int* d_grid_count;
    cudaHostGetDevicePointer(&d_grid_count, h_grid_count, 0);

    // Initialize host array
    for (int i = 0; i < 16; i++) {
        h_grid_count[i] = 0;
    }

	// Transfer point data to device
	cudaMemcpy(d_grid_points0, points_array, point_count * sizeof(Point), cudaMemcpyHostToDevice);

	clock_t start = clock();

    // current grid keeps changing depending on the stream
    GridArray *current_grid = nullptr;

    // Declare a queue that will store the grids in fifo order
    queue<GridArray*> recursive_grids;
    // Pair of pointers to Point str (Original Point array and the Sorted Point array for each pass)
    pair<Point*, Point*> grid_ptrs_2d;
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


    omp_lock_t lock1, lock2, lock_q;
    omp_init_lock(&lock1); // Initialize the lock
    omp_init_lock(&lock2); // Initialize the lock
    omp_init_lock(&lock_q); // Initialize the lock

    /*  
        Initial quadtree grid call
        The initial call will push 4 GridArrays of points into the queue, one for each subgrid
        These GridArrays contain the points located in each subgrid
        We use this to sort the d_grid_points 
    */
    grid_ptrs_2d = quadtree_grid(d_grid_points0, d_grid_points1, point_count, bl,
                                 tr, 0, 0, nullptr, grid_q, h_grid_count, d_grid_count, 0, level, lock_q);

    #pragma omp parallel num_threads(maxStreams) // create threads = maxStreams i.e. 4 Threads here
    {
        // Loop until all the GridArrays in the queue are taken care of
        while (!grid_q->empty()) {
            bool flg = false;

                #pragma omp for // Instruct the threads to work on each value of i independently
                for(int i = 0; i < maxStreams; i ++) {
                    // for loop here ....
                    GridArray *root = nullptr;
                    GridArray *sub_grid = nullptr;

                    // Thread-safe access to shared containers
                    omp_set_lock(&lock1);
                    {
                        if (!recursive_grids.empty()) {
                            root = recursive_grids.front();
                            flg = true;
                            // recursive_grids.pop();
                        }

                        if (!grid_q->empty()) {
                            sub_grid = grid_q->front();
                            grid_q->pop();
                        }
                    }
                    omp_unset_lock(&lock1);

                    // Process the grids only if valid
                    if (sub_grid != nullptr && root != nullptr) {
                        addGrid(root, sub_grid, i);
                        omp_set_lock(&lock2);
                        {
                            recursive_grids.push(sub_grid);
                        }
                        omp_unset_lock(&lock2);

                        float x1 = sub_grid->bottom_left_corner.fi;
                        float y1 = sub_grid->bottom_left_corner.se;
                        float x2 = sub_grid->top_right_corner.fi;
                        float y2 = sub_grid->top_right_corner.se;

                        // if (!(sub_grid->count < MIN_POINTS ||
                        //     (abs(x1 - x2) < MIN_DISTANCE && abs(y1 - y2) < MIN_DISTANCE))) {
                            // Launch quadtree_grid asynchronously in the selected stream
                            grid_ptrs_2d = quadtree_grid(
                                grid_ptrs_2d.fi, grid_ptrs_2d.se, sub_grid->count,
                                sub_grid->bottom_left_corner, sub_grid->top_right_corner,
                                sub_grid->start_pos, sub_grid->grid_array_flag,
                                streams[i], grid_q, h_grid_count,
                                d_grid_count, i, level + 1, lock_q);
                        // }
                    }
                }

            #pragma omp barrier
            // vprint("4 iterations done using 1 pass of the streams\n");
            // Ensure 'flg' is set correctly to indicate recursive_grids was processed
            #pragma omp single
            {
                if (!flg && !recursive_grids.empty()) {
                    recursive_grids.pop();
                }
                level += 1;
            }
            
        }
    }

    // vprint("GENERATION DONE!\n");

    // Destroy streams after use
    cudaStreamSynchronize(streams[0]);
    cudaStreamSynchronize(streams[1]);
    cudaStreamSynchronize(streams[2]);
    cudaStreamSynchronize(streams[3]);
    for (int i = 0; i < maxStreams; ++i)
    {
        // cudaStreamSynchronize(streams[i]); // Ensure all tasks in streams are done
        cudaStreamDestroy(streams[i]);
    }

    clock_t end = clock();

    Point *grid_array0 = (Point *)malloc(point_count * sizeof(Point));
    Point *grid_array1 = (Point *)malloc(point_count * sizeof(Point));
    cudaMemcpy(grid_array0, d_grid_points0, point_count * sizeof(Point),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(grid_array1, d_grid_points1, point_count * sizeof(Point),
               cudaMemcpyDeviceToHost);

    cudaFree(d_grid_points0);
    cudaFree(d_grid_points1);

    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken = %lf\n", time_taken);

    printf("calling assign_points\n");
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

    // This queue will store the sorted subarrays for each subgrid, we need to 
    // coalesce these points up until the entire array of points are sorted according to loc
    queue<GridArray *> grid_q;
    Grid *root_grid = build_quadtree_levels(points, point_count, &grid_q,
                                            root_bl, root_tr, 0);

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
