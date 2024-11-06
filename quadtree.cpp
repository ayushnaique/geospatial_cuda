#include<bits/stdc++.h>

using namespace std;

// Create the definition of a point for the QuadTree
struct Point {
    // Each point is defined as (x,y)
    int x;
    int y;

    Point(int xVal = 0, int yVal = 0){
        x = xVal;
        y = yVal;
    }
};

// The Nodes of a QuadTree
struct Node {
    Point NodePos;
    int data;
    
    // Node contains coordinates and data
    Node(Point p, int d){
        NodePos = p;
        data    = d;
    }
};

// QUADTREE
class QuadTree {
    // Top and Bottom point coordinates for B-Box
    Point TopLeft;
    Point BottomRight;

    // Pointer to Node -> Contains the data of the node.
    Node* n;

    // Declare the SubGrids as 4 different QuadTrees
    QuadTree* TopRightTree;
    QuadTree* BottomRightTree;
    QuadTree* TopLeftTree;
    QuadTree* BottomLeftTree;

public:
    // Inititalize Tree
    QuadTree(Point TopL, Point BotR){
        TopLeft     = TopL;
        BottomRight = BotR;

        n = NULL;

        TopRightTree    = NULL;
        TopLeftTree     = NULL;
        BottomRightTree = NULL;
        BottomLeftTree  = NULL;
    }

    // Declare functions for insertion, search and check for whether point lies in boundary
    void insert(Node*);
    // Return data as a Node structure -> coordinates + data
    Node* search(Point);
    // Returns true or false if a point lies in boundary of a grid or not
    bool inBoundary(Point);
};

bool QuadTree :: inBoundary(Point P) {
    return ((P.x >= TopLeft.x && P.x <= BottomRight.x) && (P.y <= TopLeft.y && P.y >= BottomRight.y));
}

void QuadTree :: insert(Node* node){
    // If node is empty
    if(node == NULL)
        return;

    // Check if inBoundary, else not possible -> Return
    if(!inBoundary(node->NodePos))
        return;

    // If we have reached the smallest possible grid size
    if(abs(TopLeft.x - BottomRight.x) <= 1 && abs(TopLeft.y - BottomRight.y) <= 1){
        // Only if the current node is empty here, populate
        if(n == NULL)
            n = node;

        return;
    }

    // Decision to traverse to which SubGrid in the QuadTree
    // Find the midpoint of the Parent Grid and then decide by comparing the value of the node

    // Comparing the X-Coord will tell us if in Left half or Right half of the Grid
    if((TopLeft.x + BottomRight.x) / 2 >= node -> NodePos.x) {// In Left half

        // Compare Y-Coord to tell us if node is in Top Quarter or Bottom Quarter
        if((TopLeft.y + BottomRight.y) / 2 <= node -> NodePos.y) { // Top Left Quarter

            if(TopLeftTree == NULL) {
                // Create the grid according to the coordinates of the parent grid
                TopLeftTree = new QuadTree( Point(TopLeft.x, TopLeft.y),
                                            Point((TopLeft.x + BottomRight.x) / 2, 
                                                  (TopLeft.y + BottomRight.y) / 2)
                                          );
            }

            // Recursive insert call
            TopLeftTree -> insert(node);
        }

        else{ // Bottom Left Quarter
            if(BottomLeftTree == NULL) {
                // Create the grid according to the coordinates of the parent grid
                BottomLeftTree = new QuadTree( Point (TopLeft.x, (TopLeft.y + BottomRight.y)/ 2),
                                               Point ((TopLeft.x + BottomRight.x) / 2, BottomRight.y)
                                             );
            }

            // Recursive call
            BottomLeftTree -> insert(node);
        }
    }

    else { // In Right half

        // Compare Y-Coord to tell us if node is in Top Quarter or Bottom Quarter
        if((TopLeft.y + BottomRight.y) / 2 <= node -> NodePos.y) { // Top right Quarter

            if(TopRightTree == NULL) {
                // Create the grid according to the coordinates of the parent grid
                TopRightTree = new QuadTree( Point((TopLeft.x + BottomRight.x) / 2, TopLeft.y),
                                             Point(BottomRight.x, (TopLeft.y + BottomRight.y) / 2)
                                          );
            }

            // Recursive insert call
            TopRightTree -> insert(node);
        }

        else{ // Bottom Right Quarter
            if(BottomRightTree == NULL) {
                // Create the grid according to the coordinates of the parent grid
                BottomRightTree = new QuadTree( Point ((TopLeft.x + BottomRight.x) / 2, 
                                                       (TopLeft.y + BottomRight.y)/ 2),
                                                Point (BottomRight.x, BottomRight.y)
                                             );
            }

            // Recursive call
            BottomRightTree -> insert(node);
        }
    }
}

// Function to find a Node in the QuadTree
Node* QuadTree :: search(Point p) {
    // Check if in boundary
    if(!inBoundary(p))
        return NULL;

    // if the current node is not null, i.e. smallest grid, return the node value
    if(n != NULL)
        return n;
    
    // Check for coordinates of the point if they lie in first half or the second
    if((TopLeft.x + BottomRight.x) / 2 >= p.x) { // In Left

        // If in top Quarter
        if((TopLeft.y + BottomRight.y) / 2 <= p.y) {
            // Check if the grid is empty, return NULL
            if(TopLeftTree == NULL)
                return NULL;

            return TopLeftTree -> search(p);
        }
        // If in bottom Quarter
        else {
            // Check if the grid is empty, return NULL
            if(BottomLeftTree == NULL) 
                return NULL;

            return BottomLeftTree -> search(p);
        }
    }

    else { // In Right

        // If in top Quarter
        if((TopLeft.y + BottomRight.y) / 2 <= p.y) {
            // Check if the grid is empty, return NULL
            if(TopRightTree == NULL)
                return NULL;

            return TopRightTree -> search(p);
        }
        // If in bottom Quarter
        else {
            if(BottomRightTree == NULL)
                return NULL;

            return BottomRightTree -> search(p);
        }
    }
};

// Function to parse the .txt file and return a vector of Points
vector<pair<int,int>> parseCoordinates(const string& filename) {
    vector<pair<int,int>> coordinates;
    ifstream infile(filename);

    if (!infile) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(0);
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        int x, y;
        if (ss >> x >> y) {
            coordinates.push_back({x, y});
        }
    }

    infile.close();
    return coordinates;
}



int main() {
    // Random Grid
    cout<<"START\n";
    QuadTree* root = new QuadTree(Point(0, 1000), Point(1000, 0));
    printf("QuadTree Created\n");

    string ifile = "points.txt";
    vector<pair<int,int>> Coord = parseCoordinates(ifile);

    int len = Coord.size();
    printf("Entering Points to the QuadTree\n");
    for(int idx = 0; idx < len; idx ++){
        root -> insert(new Node(Point(Coord[idx].first, Coord[idx].second), idx + 1));
    }
    printf("Finished entering Points to the QuadTree\n");

    printf("Value associated with point 1028 in the QuadTree is %d\n", root -> search(Point(Coord[1027].first, Coord[1027].second))->data);

    return 0;
}