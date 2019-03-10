#include <opencv2/opencv.hpp>
#include <climits>
#include <queue>

using namespace cv;
using namespace std;
int width = 0;
int height = 0; 
float intensityThreshold = 0.5;

class EdgeVal{
	
	public:
	int row;
	int col;
	double weights;
	
	EdgeVal(){
		weights = 0;
	}
};

class NodeVal{
	
	public:
	int row;
	int col;
	vector<EdgeVal> edges;
	int visitedFlag;
	
	int parentRow;
	int parentCol;
	
	void setParentNode(int rowInd, int colInd){
		parentRow = rowInd;
		parentCol = colInd;
	}
	
	NodeVal(){
		visitedFlag=0;
		row=0;
		col=0;
	}
	
};

void addEdges(int row, int col, double edge_weight, NodeVal &tempVector){
	
	EdgeVal edgeObj;
	edgeObj.weights = edge_weight;
	edgeObj.row = row;
	edgeObj.col = col;
	tempVector.edges.push_back(edgeObj);
	
}

int findBFSPath(NodeVal virtualSource, vector<NodeVal> &adjList,NodeVal &sink){
	
    queue < NodeVal > nodeQueue;
	
    nodeQueue.push(virtualSource);
	virtualSource.visitedFlag = 1;
    while (!nodeQueue.empty())
    {
		
        NodeVal currentNode = nodeQueue.front();
        nodeQueue.pop();
		
        for (int v=0; v<currentNode.edges.size(); v++)
        {	
            EdgeVal edge = currentNode.edges.at(v);
			
            if(edge.row>=0 && edge.col>=0){
				
				int index = (width * edge.row) + edge.col;
                NodeVal &tempNode = adjList.at(index);
				

                if (tempNode.visitedFlag == 0 && edge.weights > 0)
                {
					
                    tempNode.visitedFlag = 1;
                    tempNode.setParentNode(currentNode.row,currentNode.col);
                    nodeQueue.push(tempNode);
                }
            }
			else if(edge.row == -2 && edge.col == -2 && edge.weights>0){
                sink.visitedFlag = 1;
                sink.setParentNode(currentNode.row,currentNode.col);
				return 1;
            }
        }
    }
    return sink.visitedFlag;
}

void clearVisited(vector<NodeVal> &adjList){
	
	for(int i =0; i< adjList.size(); i++){
		
		adjList.at(i).visitedFlag = 0;
		
	}
	
}

double getMinInPath(NodeVal &virtualSource, NodeVal &virtualSink, vector<NodeVal> &adjList){
	
	double minPath = LONG_MAX;
	NodeVal currentNode = virtualSink;
	while(currentNode.row != -1 && currentNode.col!= -1){
		NodeVal parentNode;
		if(currentNode.parentRow == -1 && currentNode.parentCol == -1){
			parentNode = virtualSource;	
		}
		else{
			int index = width * currentNode.parentRow + currentNode.parentCol;
			parentNode = adjList.at(index);
		}
		double currentEdgeCapacity = LONG_MAX;
		for(int i =0; i< parentNode.edges.size(); i++){
			if(parentNode.edges[i].row == currentNode.row && parentNode.edges[i].col == currentNode.col){
				currentEdgeCapacity = parentNode.edges.at(i).weights;
				break;
			}
		}
		
		minPath = min(minPath, currentEdgeCapacity);	
		currentNode = parentNode;
	}
	
	return minPath;
}

void findMaxFlow(vector<NodeVal> &adjList, Mat &out_image){
	
	
	NodeVal &virtualSource = adjList[adjList.size()-2];
	NodeVal &virtualSink = adjList[adjList.size()-1];
	
	int count=0;
	while(findBFSPath(virtualSource, adjList, virtualSink) == 1){
		count++;
		clearVisited(adjList);
		double minPathValue = getMinInPath(virtualSource, virtualSink, adjList);
		
		NodeVal currentNode = virtualSink;
		
		while(currentNode.parentRow != -1 && currentNode.parentCol != -1){
			
			int index = width * currentNode.parentRow + currentNode.parentCol;
		
			NodeVal &parentNode = adjList.at(index);
		
			for(int i =0; i< parentNode.edges.size(); i++){
				if(parentNode.edges[i].row == currentNode.row && parentNode.edges[i].col == currentNode.col){
					EdgeVal &parentToChild = parentNode.edges[i];
					parentToChild.weights -= minPathValue;
					break;
				}
			}
			
            currentNode = parentNode;
		}
		
	}

	for(int i = 0; i< adjList.size()-2;i++){
		
		int row = adjList[i].row;
		int col = adjList[i].col;
		
		if(adjList[i].visitedFlag == 1){
			out_image.at<Vec3b>(row, col)[0] = 0;
			out_image.at<Vec3b>(row, col)[1] = 0;
			out_image.at<Vec3b>(row, col)[2] = 255;
		}
		else{
			out_image.at<Vec3b>(row, col)[0] = 255;
			out_image.at<Vec3b>(row, col)[1] = 0;
			out_image.at<Vec3b>(row, col)[2] = 0;
		}
	}
	
}

int main( int argc, char** argv )
{
    if(argc!=4){
        cout<<"Usage: ../seg input_image initialization_file output_mask"<<endl;
        return -1;
    }
    
    // Load the input image
    // the image should be a 3 channel image by default but we will double check that in teh seam_carving
    Mat in_image;
    in_image = imread(argv[1]/*, CV_LOAD_IMAGE_COLOR*/);
   
    if(!in_image.data)
    {
        cout<<"Could not load input image!!!"<<endl;
        return -1;
    }

    if(in_image.channels()!=3){
        cout<<"Image does not have 3 channels!!! "<<in_image.depth()<<endl;
        return -1;
    }
	
	//calculate the gradient and gaussian filter outputs
	width = in_image.cols;
    height = in_image.rows;
	
	Mat gaussianOutput;
	Mat gradientOutput;
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	
	GaussianBlur( in_image, gaussianOutput, Size(3,3), 0, 0, BORDER_DEFAULT );
	cvtColor( gaussianOutput, gaussianOutput, CV_BGR2GRAY);
	
	double edge_weight;
	
	int sizeOfVector = ((height * width)+2);
	vector<NodeVal > adjList(sizeOfVector);
	
	//read the config file
	ifstream f(argv[2]);
    if(!f){
        cout<<"Could not load initial mask file!!!"<<endl;
        return -1;
    }
    
    int n;
    f>>n;
	
	vector<EdgeVal> seededPixels; //only background pixels
	
    // get the initial pixels
	NodeVal obj;
	obj.row = -1;
	obj.col = -1;
    for(int i=0;i<n;++i){
        int x, y, t;
        f>>x>>y>>t;
		EdgeVal edgeObj;
		edgeObj.row = y;
		edgeObj.col = x;
		
		if(t==0){
			edgeObj.weights = t;
			seededPixels.push_back(edgeObj);
		}
		
		else if(t==1){ //foreground - source
			edgeObj.weights = LONG_MAX;
			obj.edges.push_back(edgeObj);
		}
	}
	
	adjList.at(sizeOfVector-2) = obj;
	NodeVal nodeObj;
	nodeObj.row=-2;
	nodeObj.col=-2;
	adjList.at(sizeOfVector-1) = nodeObj;
	
	int counter = 0;
	double maxVal = 0;
	double minVal = 0;
	
	for(int i = 0; i < height; i++)
	{	
		for(int j = 0; j < width; j++){
			
			double edge_weight = 0;
			NodeVal tempVector;
			tempVector.row = i;
			tempVector.col = j;
			
			bool toSinkFlag = false;
			for(int k =0; k< seededPixels.size(); k++){
				if(seededPixels[k].row == i && seededPixels[k].col == j){
					edge_weight = LONG_MAX;
					addEdges(-2, -2, edge_weight, tempVector);
					toSinkFlag = true;
				}
			}
			//double intensityGap;
			if(j+1<width){
				double intensityGap = (gaussianOutput.at<uchar>(i,j) - gaussianOutput.at<uchar>(i, j+1));
                if(intensityGap<intensityThreshold){
                    addEdges(i, j+1, LONG_MAX, tempVector);
                }else{
                    addEdges(i, j+1, 1, tempVector);
                }	
			}
			if(j-1>=0){
				double intensityGap = (gaussianOutput.at<uchar>(i,j) - gaussianOutput.at<uchar>(i, j-1));
                if(intensityGap<intensityThreshold){
                    addEdges(i, j-1, LONG_MAX, tempVector);
                }else{
                    addEdges(i, j-1, 1, tempVector);
                }
			}
			if(i-1>=0){
				double intensityGap = (gaussianOutput.at<uchar>(i,j) - gaussianOutput.at<uchar>(i-1, j));
                if(intensityGap<intensityThreshold){
                    addEdges(i-1, j, LONG_MAX, tempVector);
                }else{
                    addEdges(i-1, j, 1, tempVector);
                }
			}
			if(i+1<height){
				
				double intensityGap = (gaussianOutput.at<uchar>(i,j) - gaussianOutput.at<uchar>(i+1, j));
                if(intensityGap<intensityThreshold){
                    addEdges(i+1, j, LONG_MAX, tempVector);
                }else{
                    addEdges(i+1, j, 1, tempVector);
                }
			}
			adjList.at(counter) = tempVector;
			counter++;
		}
	}
    
    // the output image
    Mat out_image = in_image.clone();
    
	findMaxFlow( adjList ,out_image);
    
    // write it on disk
    imwrite( argv[3], out_image);
    
    // also display them both
    namedWindow( "Original image", WINDOW_AUTOSIZE );
    namedWindow( "Show Marked Pixels", WINDOW_AUTOSIZE );
    imshow( "Original image", in_image );
    imshow( "Show Marked Pixels", out_image );
    waitKey(0);
    return 0;
}