#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <utility>
namespace cg = cooperative_groups;
using namespace std;

struct Point {
	float x, y;

	Point(float xc, float yc) : x(xc), y(yc) {}
};

struct Grid {
	Grid *bottom_left, *bottom_right, *top_left, *top_right;
	Point *points;

	// Number of points in the grid
	int count;

	// Grid Dimension
	std ::pair<float, float> top_right_corner;
	std ::pair<float, float> bottom_left_corner;

	// Initialize the corresponding Point values
	Grid(Grid *bl, Grid *br, Grid *tl, Grid *tr, Point *ps,
		 pair<float, float> uB, pair<float, float> lB, int c)
		: bottom_left(bl),
		  bottom_right(br),
		  top_left(tl),
		  top_right(tr),
		  points(ps),
		  top_right_corner(uB),
		  bottom_left_corner(lB),
		  count(c) {}
};

struct GridArray {
	GridArray *bottom_left, *bottom_right, *top_left, *top_right;

	int count, start_pos, grid_array_flag;

	std ::pair<float, float> top_right_corner;
	std ::pair<float, float> bottom_left_corner;

	GridArray(GridArray *bl, GridArray *br, GridArray *tl, GridArray *tr, 
		  pair<float, float> uB, pair<float, float> lB, int c, int sp, int gfl)
		: bottom_left(bl),
		  bottom_right(br),
		  top_left(tl),
		  top_right(tr),
		  top_right_corner(uB),
		  bottom_left_corner(lB),
		  count(c),
          start_pos(sp),
          grid_array_flag(gfl) {}
};

__inline__ __device__ int warpReduceSum(int value,
										cg::thread_block_tile<32> warp);

__global__ void categorize_points(Point *d_points, int *d_categories,
								  int *grid_counts, int count, int range,
								  float middle_x, float middle_y);

__global__ void organize_points(Point *d_points, int *d_categories, Point *bl,
								Point *br, Point *tl, Point *tr, int count,
								int range);

__global__ void reorder_points(Point *d_points, Point *grid_points,
							   int *grid_counts, int count, int range,
							   float middle_x, float middle_y, int start_pos,
							   bool opt);

bool validate_grid(Grid *root_grid, pair<float, float> &top_right_corner,
				   pair<float, float> &bottom_left_corner);

Grid* assign_points(GridArray *root_grid, Point *grid_array1, Point *grid_array2);