#include<bits/stdc++.h>

using namespace std;

// // Create the definition of a point for the QuadTree
// struct Point {
//     // Each point is defined as (x,y)
//     int x;
//     int y;

//     Point(int xVal = 0, int yVal = 0){
//         x = xVal;
//         y = yVal;
//     }
// };

// The Nodes of a QuadTree
struct Node {
    //Point NodePos;
    pair<float,float> NodePos;
    int data;
    
    // Node contains coordinates and data
    Node(/*Point p,*/ pair<float, float> point, int d){
        NodePos = {point.first, point.second};
        data = d;
    }
};

// QUADTREE
struct QuadTree {
    // Top and Bottom point coordinates for B-Box
    // Point TopLeft;
    // Point BottomRight;
    pair<float,float> TopRight;
    pair<float,float> BottomLeft;

    // Pointer to Node -> Contains the data of the node.
    Node* n;

    // Declare the SubGrids as 4 different QuadTrees
    QuadTree* TopRightTree;
    QuadTree* BottomRightTree;
    QuadTree* TopLeftTree;
    QuadTree* BottomLeftTree;

public:
    // Inititalize Tree
    QuadTree(pair<float,float> TopR, pair<float,float> BotL){
        TopRight     = {TopR.first, TopR.second};
        BottomLeft = {BotL.first, BotL.second};

        n = nullptr;

        TopRightTree    = nullptr;
        TopLeftTree     = nullptr;
        BottomRightTree = nullptr;
        BottomLeftTree  = nullptr;
    }

    // Declare functions for insertion, search and check for whether point lies in boundary
    void insert(Node*);
    // Function to delete a node from the quadTree
    bool delete_point(Node*);
    // Return data as a Node structure -> coordinates + data
    Node* search(pair<float,float>&);
    // Returns true or false if a point lies in boundary of a grid or not
    bool inBoundary(pair<float,float>&);
};

bool QuadTree :: inBoundary(pair<float,float>& P) {
    return ((P.first >= BottomLeft.first && P.first <= TopRight.first) && (P.second <= TopRight.second && P.second >= BottomLeft.second));
}

void QuadTree :: insert(Node* node){
    // If node is empty
    if(node == nullptr)
        return;

    // Check if inBoundary, else not possible -> Return
    if(!inBoundary(node->NodePos))
        return;

    // Store the coordinates in variables for fast access
    float Top_x = TopRight.first, Top_y = TopRight.second;
    float Bottom_x = BottomLeft.first, Bottom_y = BottomLeft.second;

    // Calculate the coordinates of the mid point of the grid
    float Mid_x = (Bottom_x + Top_x) / 2.0;
    float Mid_y = (Bottom_y + Top_y) / 2.0;

    // If we have reached the smallest possible grid size
    if(abs(TopRight.first - BottomLeft.first) <= 1 && abs(BottomLeft.second - BottomLeft.second) <= 1){
        // Only if the current node is empty here, populate
        if(n == nullptr)
            n = node;

        return;
    }

    // Decision to traverse to which SubGrid in the QuadTree
    // Find the midpoint of the Parent Grid and then decide by comparing the value of the node

    // Comparing the X-Coord will tell us if in Left half or Right half of the Grid
    if(Mid_x >= node -> NodePos.first) {// In Left half

        // Compare Y-Coord to tell us if node is in Top Quarter or Bottom Quarter
        if(Mid_y <= node -> NodePos.second) { // Top Left Quarter

            if(TopLeftTree == nullptr) {
                // Create the grid according to the coordinates of the parent grid
                pair<float, float> entry1 = make_pair( Mid_x, Top_y );
                pair<float, float> entry2 = make_pair( Bottom_x, Mid_y );
                TopLeftTree = new QuadTree( entry1, entry2 );
            }

            // Recursive insert call
            TopLeftTree -> insert(node);
        }

        else{ // Bottom Left Quarter
            if(BottomLeftTree == nullptr) {
                // Create the grid according to the coordinates of the parent grid
                pair<float, float> entry1 = make_pair( Mid_x, Mid_y );
                pair<float, float> entry2 = make_pair( Bottom_x, Bottom_y );
                BottomLeftTree = new QuadTree( entry1, entry2 );
            }

            // Recursive call
            BottomLeftTree -> insert(node);
        }
    }

    else { // In Right half

        // Compare Y-Coord to tell us if node is in Top Quarter or Bottom Quarter
        if(Mid_y <= node -> NodePos.second) { // Top right Quarter

            if(TopRightTree == nullptr) {
                // Create the grid according to the coordinates of the parent grid
                pair<float, float> entry1 = make_pair( Top_x, Top_y );
                pair<float, float> entry2 = make_pair( Mid_x, Mid_y );
                TopRightTree = new QuadTree( entry1, entry2 );
            }

            // Recursive insert call
            TopRightTree -> insert(node);
        }

        else{ // Bottom Right Quarter
            if(BottomRightTree == nullptr) {
                // Create the grid according to the coordinates of the parent grid
                pair<float, float> entry1 = make_pair( Top_x, Mid_y );
                pair<float, float> entry2 = make_pair( Mid_x, Bottom_y );
                BottomRightTree = new QuadTree( entry1, entry2 );
            }

            // Recursive call
            BottomRightTree -> insert(node);
        }
    }
}

bool QuadTree :: delete_point(Node* node) {
    if(node == nullptr) {
        return false;
    }

    if(!inBoundary(node ->NodePos)) {
        return false;
    } 

    // We have reached the base and now can delete the node
    if(n != nullptr) {
        pair<float, float> n_data = n -> NodePos;
        pair<float, float> del_data = node -> NodePos;

        if(n_data.first == del_data.first && n_data.second == del_data.second) {
            Node * temp = n;
            n = nullptr;
            free(temp);
            
            return true;
        }
        else {
            printf("The node we want to delete does not exist in the QuadTree. Returning with false value!\n");
            return false;
        }
    }

    // Get the bounding box values for the 4 sub-grids
    pair<float, float> TR_lB, TR_uB, TL_lB, TL_uB, BR_lB, BR_uB, BL_lB, BL_uB;

    if(TopRightTree)
        TR_lB = TopRightTree ->BottomLeft,
        TR_uB = TopRightTree ->TopRight;
    if(TopLeftTree)
        TL_lB = TopLeftTree ->BottomLeft,
        TL_uB = TopLeftTree ->TopRight;
    if(BottomRightTree)
        BR_lB = BottomRightTree ->BottomLeft,
        BR_uB = BottomRightTree ->TopRight;
    if(BottomLeftTree)
        BL_lB = BottomLeftTree ->BottomLeft,
        BL_uB = BottomLeftTree ->TopRight;

    // Recursively call the delete function for the quadtree in which the node we want to delete is located
    if(TopRightTree && (node->NodePos.first >= TR_lB.first && node->NodePos.first <= TR_uB.first) && 
       (node->NodePos.second >= TR_lB.second && node->NodePos.second <= TR_uB.second)) {
        return TopRightTree -> delete_point(node);
    }
    else if(TopLeftTree && (node->NodePos.first >= TL_lB.first && node->NodePos.first <= TL_uB.first) && 
            (node->NodePos.second >= TL_lB.second && node->NodePos.second <= TL_uB.second)) {
        return TopLeftTree -> delete_point(node);
    }
    else if(BottomRightTree && (node->NodePos.first >= BR_lB.first && node->NodePos.first <= BR_uB.first) && 
            (node->NodePos.second >= BR_lB.second && node->NodePos.second <= BR_uB.second)) {
        return BottomRightTree -> delete_point(node);
    }
    else if(BottomLeftTree && (node->NodePos.first >= BL_lB.first && node->NodePos.first <= BL_uB.first) && 
            (node->NodePos.second >= BL_lB.second && node->NodePos.second <= BL_uB.second)) {
        return BottomLeftTree -> delete_point(node);
    }
    else {
        printf("The QuadTree has not been built for this coordinate location, returning with false value!\n");
        return false;
    }
}

// Function to find a Node in the QuadTree
Node* QuadTree :: search(pair<float, float>& P) {
    // Check if in boundary
    if(!inBoundary(P))
        return nullptr;

    // if the current node is not null, i.e. smallest grid, return the node value
    if(n != nullptr)
        return n;

    // Store the coordinates in variables for fast access
    float Top_x = TopRight.first, Top_y = TopRight.second;
    float Bottom_x = BottomLeft.first, Bottom_y = BottomLeft.second;

    // Calculate the coordinates of the mid point of the grid
    float Mid_x = (Bottom_x + Top_x) / 2.0;
    float Mid_y = (Bottom_y + Top_y) / 2.0;
    
    // Check for coordinates of the point if they lie in first half or the second
    if(Mid_x >= P.first) { // In Left

        // If in top Quarter
        if(Mid_y <= P.second) {
            // Check if the grid is empty, return NULL
            if(TopLeftTree == nullptr)
                return nullptr;

            return TopLeftTree -> search(P);
        }
        // If in bottom Quarter
        else {
            // Check if the grid is empty, return NULL
            if(BottomLeftTree == nullptr) 
                return nullptr;

            return BottomLeftTree -> search(P);
        }
    }

    else { // In Right

        // If in top Quarter
        if(Mid_y <= P.second) {
            // Check if the grid is empty, return NULL
            if(TopRightTree == nullptr)
                return nullptr;

            return TopRightTree -> search(P);
        }
        // If in bottom Quarter
        else {
            if(BottomRightTree == nullptr)
                return nullptr;

            return BottomRightTree -> search(P);
        }
    }
};

// Function to parse the .txt file and return a vector of Points
vector<pair<float,float>> parseCoordinates(const string& filename) {
    vector<pair<float,float>> coordinates;
    ifstream infile(filename);

    if (!infile) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(0);
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        float x, y;
        if (ss >> x >> y) {
            coordinates.push_back({(float)x, (float)y});
        }
    }

    infile.close();
    return coordinates;
}

// Function to validate whether the nodes in the QuadTree are in correct position or not
bool validate_grid(QuadTree* root_grid, pair<float, float>& top_right_corner, pair<float, float>& bottom_left_corner) {
    if(root_grid == nullptr) {
        return true;
    }

    // If the grid node is the leaf, i.e. n != nullptr
    if(root_grid -> n != nullptr) {
        pair<float, float> point_pos = root_grid -> n -> NodePos;
        float point_x = point_pos.first;
        float point_y = point_pos.second;

		float top_x = top_right_corner.first;
		float top_y = top_right_corner.second;

		float bot_x = bottom_left_corner.first;
		float bot_y = bottom_left_corner.second;

		float mid_x = (top_x + bot_x) / 2;
		float mid_y = (top_y + bot_y) / 2;

        if (point_x < bot_x || point_x > top_x) {
            printf(
                "Validation Error! Point (%f, %f) is plced out of bounds. "
                "Grid dimension: [(%f, %f), (%f, %f)]\n",
                point_x, point_y, bot_x, bot_y, top_x, top_y);
            return false;
		} 
        else if (point_y < bot_y || point_y > top_y) {
            printf(
                "Validation Error! Point (%f, %f) is plced out of bounds. "
                "Grid dimension: [(%f, %f), (%f, %f)]\n",
                point_x, point_y, bot_x, bot_y, top_x, top_y);
            return false;
		} 
        else {
			return true;
		}
    }

    // Call Recursively for all 4 quadrants
	QuadTree *top_left_child = nullptr;
	QuadTree *top_right_child = nullptr;
	QuadTree *bottom_left_child = nullptr;
	QuadTree *bottom_right_child = nullptr;

	top_left_child = root_grid->TopLeftTree;
	top_right_child = root_grid->TopRightTree;
	bottom_left_child = root_grid->BottomLeftTree;
	bottom_right_child = root_grid->BottomRightTree;

	bool check_top_left =
		validate_grid(top_left_child, top_left_child->TopRight,
					  top_left_child->BottomLeft);
	bool check_top_right =
		validate_grid(top_right_child, top_right_child->TopRight,
					  top_right_child->BottomLeft);
	bool check_bottom_left =
		validate_grid(bottom_left_child, bottom_left_child->TopRight,
					  bottom_left_child->BottomLeft);
	bool check_bottom_right =
		validate_grid(bottom_right_child, bottom_right_child->TopRight,
					  bottom_right_child->BottomLeft);

	return check_top_left && check_top_right && check_bottom_left &&
		   check_bottom_right;
}

int main(int argv, char* argc[]) {
    // variables for storing runtime
    double time_taken;
	clock_t start, end;

    // Input the dimension of the QuadTree
    if(argv < 5){
        printf("Command should be in the form: ./quadtree tR_x tR_y bL_x bL_y\n");
        exit(0);
    }

    float topR_x = stof(argc[1]);
    float topR_y = stof(argc[2]);
    float botL_x = stof(argc[3]);
    float botL_y = stof(argc[4]);

    // START EXECUTION
    cout<<"START\n";
    pair<float, float> TopRight = make_pair(topR_x, topR_y);
    pair<float, float> BottomLeft = make_pair(botL_x, botL_y);
    QuadTree* root = new QuadTree(TopRight, BottomLeft);
    printf("QuadTree Created\n");

    string ifile = "../points.txt";
    printf("Parsing input coordinate file\n");
    vector<pair<float,float>> Coord = parseCoordinates(ifile);
    printf("File Parsed\n");

    int len = Coord.size();
    printf("Entering Points to the QuadTree\n");

    // Begin the timer
    start = clock();
    for(int idx = 0; idx < len; idx ++){
        pair<float, float> point = make_pair(Coord[idx].first, Coord[idx].second);
        root -> insert(new Node(point, idx + 1));
    }
    // End the timer
    end = clock();

    printf("Finished entering Points to the QuadTree\n");
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken for insertion of points = %lf\n\n", time_taken);

    start = clock();
    pair<float, float> point_to_search = make_pair(Coord[1027].first, Coord[1027].second);
    Node* point = root -> search(point_to_search);
    end = clock();

    if(point != nullptr)
        printf("coords associated with point 1028 in the QuadTree is %f, %f\n", point->NodePos.first, point->NodePos.second);
    else    
        printf("The node we are searching for has either been removed or does not exist!\n");
    
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken for searching = %lf\n\n", time_taken);

    start = clock();
    root -> delete_point(new Node(point_to_search, 0));
    end = clock();
    printf("point deleted\n");

    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken for deletion = %lf\n\n", time_taken);

    start = clock();
    point = root -> search(point_to_search);
    end = clock();

    if(point != nullptr)
        printf("coords associated with point 1028 in the QuadTree is %f, %f\n", point->NodePos.first, point->NodePos.second);
    else    
        printf("The node we are searching for has either been removed or does not exist!\n");
    
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
	printf("Time taken for searching = %lf\n\n", time_taken);

	bool check = validate_grid(root, TopRight, BottomLeft);

	if (check == true)
		printf("Grid Verification Success!\n");
	else
		printf("Grid Verification Failure!\n");


    return 0;
}