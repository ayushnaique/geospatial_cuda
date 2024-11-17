#include <bits/stdc++.h>

#include "kernels.h"

using namespace std;

void delete_point(Point point_to_delete, Grid *target_grid,
				  vector<QuadrantBoundary> &boundaries,
				  unordered_map<int, Grid *> &grid_map) {
	// Find and remove the point from the grid
	Point *new_points =
		(Point *)malloc((target_grid->count - 1) * sizeof(Point));
	int new_count = 0;

	// Update the grid with the new points
	free(target_grid->points);
	target_grid->points = new_points;
	target_grid->count = new_count;

	// Propagate count decrement to all parent nodes
	Grid *current_grid = target_grid;
	while (current_grid) {
		current_grid->count--;

		// Remove the point from the parent's point array
		if (current_grid->parent) {
			Point *new_parent_points =
				(Point *)malloc((current_grid->parent->count) * sizeof(Point));
			int new_parent_count = 0;
			for (int i = 0; i < current_grid->parent->count + 1; i++) {
				if (current_grid->parent->points[i].x != point_to_delete.x ||
					current_grid->parent->points[i].y != point_to_delete.y) {
					new_parent_points[new_parent_count++] =
						current_grid->parent->points[i];
				}
			}
			free(current_grid->parent->points);
			current_grid->parent->points = new_parent_points;
		}

		current_grid = current_grid->parent;
	}

	// Check if the count is less than MIN_POINTS
	Grid *parent = target_grid->parent;
	if (parent->count < MIN_POINTS && parent->bottom_left) {
		printf("Removing child nodes \n");
		// Remove children nodes
		delete parent->bottom_left;
		delete parent->bottom_right;
		delete parent->top_left;
		delete parent->top_right;

		parent->bottom_left = nullptr;
		parent->bottom_right = nullptr;
		parent->top_left = nullptr;
		parent->top_right = nullptr;

		// Update boundaries vector
		int parent_id = parent->id;
		boundaries.erase(remove_if(boundaries.begin(), boundaries.end(),
								   [parent_id](const QuadrantBoundary &qb) {
									   return qb.id / 4 == parent_id;
								   }),
						 boundaries.end());

		// Update grid_map
		for (auto it = grid_map.begin(); it != grid_map.end();) {
			if (it->first / 4 == parent_id && it->first != parent_id) {
				it = grid_map.erase(it);
			} else {
				++it;
			}
		}
	}

	printf("Point deleted successfully.\n");
}