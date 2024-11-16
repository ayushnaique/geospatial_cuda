#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <utility>
namespace cg = cooperative_groups;
using namespace std;

struct Point {
	float x, y;

	Point() : x(), y() {}

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

__global__ void reorder_points(Point* d_points_array, Point* d_grid_points, int count, int range, float middle_x, float middle_y, int start_pos, int* d_grid_count);

bool validate_grid(Grid *root_grid, pair<float, float> &top_right_corner, pair<float, float> &bottom_left_corner);