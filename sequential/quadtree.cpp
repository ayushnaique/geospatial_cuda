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
class QuadTree {
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

        n = NULL;

        TopRightTree    = NULL;
        TopLeftTree     = NULL;
        BottomRightTree = NULL;
        BottomLeftTree  = NULL;
    }

    // Declare functions for insertion, search and check for whether point lies in boundary
    void insert(Node*);
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
    if(node == NULL)
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
        if(n == NULL)
            n = node;

        return;
    }

    // Decision to traverse to which SubGrid in the QuadTree
    // Find the midpoint of the Parent Grid and then decide by comparing the value of the node

    // Comparing the X-Coord will tell us if in Left half or Right half of the Grid
    if(Mid_x >= node -> NodePos.first) {// In Left half

        // Compare Y-Coord to tell us if node is in Top Quarter or Bottom Quarter
        if(Mid_y <= node -> NodePos.second) { // Top Left Quarter

            if(TopLeftTree == NULL) {
                // Create the grid according to the coordinates of the parent grid
                pair<float, float> entry1 = make_pair( Mid_x, Top_y );
                pair<float, float> entry2 = make_pair( Bottom_x, Mid_y );
                TopLeftTree = new QuadTree( entry1, entry2 );
            }

            // Recursive insert call
            TopLeftTree -> insert(node);
        }

        else{ // Bottom Left Quarter
            if(BottomLeftTree == NULL) {
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

            if(TopRightTree == NULL) {
                // Create the grid according to the coordinates of the parent grid
                pair<float, float> entry1 = make_pair( Top_x, Top_y );
                pair<float, float> entry2 = make_pair( Mid_x, Mid_y );
                TopRightTree = new QuadTree( entry1, entry2 );
            }

            // Recursive insert call
            TopRightTree -> insert(node);
        }

        else{ // Bottom Right Quarter
            if(BottomRightTree == NULL) {
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

// Function to find a Node in the QuadTree
Node* QuadTree :: search(pair<float, float>& P) {
    // Check if in boundary
    if(!inBoundary(P))
        return NULL;

    // if the current node is not null, i.e. smallest grid, return the node value
    if(n != NULL)
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
            if(TopLeftTree == NULL)
                return NULL;

            return TopLeftTree -> search(P);
        }
        // If in bottom Quarter
        else {
            // Check if the grid is empty, return NULL
            if(BottomLeftTree == NULL) 
                return NULL;

            return BottomLeftTree -> search(P);
        }
    }

    else { // In Right

        // If in top Quarter
        if(Mid_y <= P.second) {
            // Check if the grid is empty, return NULL
            if(TopRightTree == NULL)
                return NULL;

            return TopRightTree -> search(P);
        }
        // If in bottom Quarter
        else {
            if(BottomRightTree == NULL)
                return NULL;

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



int main(int argv, char* argc[]) {
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
    for(int idx = 0; idx < len; idx ++){
        pair<float, float> point = make_pair(Coord[idx].first, Coord[idx].second);
        root -> insert(new Node(point, idx + 1));
    }
    printf("Finished entering Points to the QuadTree\n");

    pair<float, float> point_to_search = make_pair(Coord[1027].first, Coord[1027].second);
    printf("coords associated with point 1028 in the QuadTree is %f, %f\n", point_to_search.first, point_to_search.second);

    return 0;
}
