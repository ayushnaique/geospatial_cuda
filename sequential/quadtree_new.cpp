#include <bits/stdc++.h>

using namespace std;

#define mp make_pair
#define fi first
#define se second
#define MIN_POINTS 5.0
#define MIN_DISTANCE 5.0

// Structure to store point data in the grid, will be in an array
struct Point {
    int x, y;
    
    // Default constructor
    Point() : x(0), y(0) {}

    Point(int xPos, int yPos) : x(xPos), y(yPos) {}
};

// Structure for QuadTree Grid
struct Grid {
    Grid *bottom_left, *bottom_right,
         *top_left, *top_right;

    // Store all the points in the current grid level in an array
    Point* points;

    // Count the number of points in the current grid
    int count;

    // Store the bounds of the Grid
    pair<float, float> top_right_corner;
    pair<float, float> bottom_left_corner;

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

// Sequential version of categorize points kernel
void categorize_points(Point* points, int* categories, int* grid_counts, int& count, float& middle_x, float& middle_y) {
    int first = 0, second = 0, third = 0, fourth = 0;
	for (int i = 0; i < count; i++) {
        // bottom left; if the point lies in bottom left, increment
        if (points[i].x <= middle_x and points[i].y <= middle_y) {
            categories[i] = 0;
            first++;
        }
        // bottom right; if point lies in bottom right, increment
        else if (points[i].x > middle_x and points[i].y <= middle_y) {
            categories[i] = 1;
            second++;
        }
        // top left; if point lies in top left, increment
        else if (points[i].x <= middle_x and points[i].y > middle_y) {
            categories[i] = 2;
            third++;
        }
        // top right; if point lies in top right, increment
        else if (points[i].x > middle_x and points[i].y > middle_y) {
            categories[i] = 3;
            fourth++;
        }
	}

    grid_counts[0] = first;
    grid_counts[1] = second;
    grid_counts[2] = third;
    grid_counts[3] = fourth;

    return;
}

// Sequential version of organize points kernel
void organize_points(Point* points, int* categories, Point* bottom_left, Point* bottom_right, Point* top_left, Point* top_right, int& count) {
    int* subgrid_index = new int[4]();
	for (int i = 0; i < count; i++) {
        // Point array will store the respective points in a contiguous
        // fashion increment subgrid index according to the category
        unsigned int category_index = subgrid_index[categories[i]];
        if (categories[i] == 0) {
            bottom_left[category_index] = points[i];
        }
        if (categories[i] == 1) {
            bottom_right[category_index] = points[i];
        }
        if (categories[i] == 2) {
            top_left[category_index] = points[i];
        }
        if (categories[i] == 3) {
            top_right[category_index] = points[i];
        }
        // increment subgrid idx
        subgrid_index[categories[i]] += 1;
	}

    // deallocate the array
    delete []subgrid_index;

    return;
}

// Sequential version of quadtree_grid creation
Grid* quadtree_grid(Point *points, int count,
					pair<float, float> bottom_left_corner,
					pair<float, float> top_right_corner, int level) {
    // unzip the bounds for faster access
    float x1 = bottom_left_corner.first, y1 = bottom_left_corner.second,
		  x2 = top_right_corner.first, y2 = top_right_corner.second;

    // Base case, if smallest size reached or min points achieved
    if (count < MIN_POINTS or
		(abs(x1 - x2) < MIN_DISTANCE and abs(y1 - y2) < MIN_DISTANCE)) {
		pair<float, float> upper_bound = make_pair(x2, y2);
		pair<float, float> lower_bound = make_pair(x1, y1);
		return new Grid(nullptr, nullptr, nullptr, nullptr, points, upper_bound,
						lower_bound, count);
	}

	printf("%d: Creating grid from (%f,%f) to (%f,%f) for %d points\n", level,
		   x1, y1, x2, y2, count);

	// Declare array to store the quadrant in which the point will be located.
	int* categories = new int [count];
    // Stores the count of points in each sub grid
	int* grid_counts = new int [4];
    
    // Calculate the middle points of the grid
    float middle_x = (x2 + x1) / 2, middle_y = (y2 + y1) / 2;
	printf("mid_x = %f, mid_y = %f\n", middle_x, middle_y);

    // Call categorize points function
    categorize_points(points, categories, grid_counts, count, middle_x, middle_y);

    int total = 0;
	printf("%d: Point counts per sub grid - \n", level);
	for (int i = 0; i < 4; i++) {
		printf("sub grid %d - %d\n", i + 1, grid_counts[i]);
		total += grid_counts[i];
	}
	printf("Total Count - %d\n", count);
	if (total == count) {
		printf("Sum of sub grid counts matches total point count\n");
	}

    // Declare arrays for each section of the grid and allocate memory depending
	// on the number of points found
	Point *bottom_left, *bottom_right, *top_left, *top_right;
    bottom_left = new Point [grid_counts[0]];
    bottom_right = new Point [grid_counts[1]];
    top_left = new Point [grid_counts[2]];
    top_right = new Point [grid_counts[3]];

    // Call organize points function
    organize_points(points, categories, bottom_left, bottom_right, top_left, top_right, count);

    // Deallocate categories
    delete []categories;

	printf("\n\n");

    // Recursively call the quadtree grid function on each of the 4 sub grids -
	// bl, br, tl, tr and store in Grid struct
	Grid *bl_grid, *tl_grid, *br_grid, *tr_grid;
	bl_grid = quadtree_grid(bottom_left, grid_counts[0], bottom_left_corner,
							mp(middle_x, middle_y), level + 1);
	br_grid = quadtree_grid(bottom_right, grid_counts[1], mp(middle_x, y1),
							mp(x2, middle_y), level + 1);
	tl_grid = quadtree_grid(top_left, grid_counts[2], mp(x1, middle_y),
							mp(middle_x, y2), level + 1);
	tr_grid = quadtree_grid(top_right, grid_counts[3], mp(middle_x, middle_y),
							top_right_corner, level + 1);

    // deallocate grid counts
    delete []grid_counts;

    // The bounds of the grid
	pair<float, float> upper_bound = make_pair(x2, y2);
	pair<float, float> lower_bound = make_pair(x1, y1);

	return new Grid(bl_grid, br_grid, tl_grid, tr_grid, points, upper_bound,
					lower_bound, count);
}

// Validation Function
bool validate_grid(Grid *root_grid, pair<float, float> &top_right_corner,
				   pair<float, float> &bottom_left_corner) {
	if (root_grid == nullptr) return true;

	// If we have reached the bottom of the grid, we start validation
	if (root_grid->points) {
		Point *point_array = root_grid->points;
		float top_x = top_right_corner.first;
		float top_y = top_right_corner.second;

		float bot_x = bottom_left_corner.first;
		float bot_y = bottom_left_corner.second;

		float mid_x = (top_x + bot_x) / 2;
		float mid_y = (top_y + bot_y) / 2;

		int count = root_grid->count;

		for (int i = 0; i < count; i++) {
			float point_x = point_array[i].x;
			float point_y = point_array[i].y;

			if (point_x < bot_x || point_x > top_x) {
				printf(
					"Validation Error! Point (%f, %f) is plced out of bounds. "
					"Grid dimension: [(%f, %f), (%f, %f)]\n",
					point_x, point_y, bot_x, bot_y, top_x, top_y);
				return false;
			} else if (point_y < bot_y || point_y > top_y) {
				printf(
					"Validation Error! Point (%f, %f) is plced out of bounds. "
					"Grid dimension: [(%f, %f), (%f, %f)]\n",
					point_x, point_y, bot_x, bot_y, top_x, top_y);
				return false;
			} else {
				continue;
			}
		}

		return true;
	}

	// Call Recursively for all 4 quadrants
	Grid *top_left_child = nullptr;
	Grid *top_right_child = nullptr;
	Grid *bottom_left_child = nullptr;
	Grid *bottom_right_child = nullptr;

	top_left_child = root_grid->top_left;
	top_right_child = root_grid->top_right;
	bottom_left_child = root_grid->bottom_left;
	bottom_right_child = root_grid->bottom_right;

	bool check_top_left =
		validate_grid(top_left_child, top_left_child->top_right_corner,
					  top_left_child->bottom_left_corner);
	bool check_top_right =
		validate_grid(top_right_child, top_right_child->top_right_corner,
					  top_right_child->bottom_left_corner);
	bool check_bottom_left =
		validate_grid(bottom_left_child, bottom_left_child->top_right_corner,
					  bottom_left_child->bottom_left_corner);
	bool check_bottom_right =
		validate_grid(bottom_right_child, bottom_right_child->top_right_corner,
					  bottom_right_child->bottom_left_corner);

	return check_top_left && check_top_right && check_bottom_left &&
		   check_bottom_right;
}


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

	file.close();

	double time_taken;
	clock_t start, end;

	Point *points_array = (Point *)malloc(point_count * sizeof(Point));
	for (int i = 0; i < point_count; i++) {
		points_array[i] = points[i];
	}
	start = clock();
	Grid *root_grid = quadtree_grid(points_array, point_count, mp(0.0, 0.0),
									mp(max_size, max_size), 0);
	end = clock();

	time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken = %lf\n\n", time_taken);

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