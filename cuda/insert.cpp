#include <bits/stdc++.h>
#include <cuda_runtime.h>

#include "kernels.h"

using namespace std;

void insert_point(Point new_point, Grid *target_grid,
				  vector<QuadrantBoundary> &boundaries) {
	// Add the point to the grid
	int prev_count = target_grid->count;
	Point *new_points = (Point *)malloc((prev_count + 1) * sizeof(Point));
	memcpy(new_points, target_grid->points, prev_count * sizeof(Point));
	new_points[prev_count] = new_point;
	free(target_grid->points);
	target_grid->points = new_points;
	target_grid->count = prev_count + 1;

	// Propagate count increment to all parent nodes
	Grid *parent_grid = target_grid->parent;
	while (parent_grid) {
		Point *new_points_parents =
			(Point *)malloc((parent_grid->count + 1) * sizeof(Point));
		memcpy(new_points_parents, parent_grid->points,
			   parent_grid->count * sizeof(Point));
		new_points_parents[parent_grid->count] = new_point;

		free(parent_grid->points);
		parent_grid->points = new_points_parents;
		parent_grid->count++;
		parent_grid = parent_grid->parent;
	}

	printf("Point inserted successfully.\n");
}
