/**
* ObjectIdentifier.cpp 
* Mark Gorewicz
* Senior Project
*/

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <string>
#include <windows.h>
using namespace cv;
using namespace std;


class Item
{
public:
	Mat shape;
	string name;
	string firstColor;
	string secondColor;
	string thirdColor;
	int nonZeros;
	

	Item(Mat pShape, string pFirstColor, string pSecondColor, string pThirdColor, int pNonZeros)
	{
		shape = pShape;
		firstColor = pFirstColor;
		secondColor = pSecondColor;
		thirdColor = pThirdColor;
		nonZeros = pNonZeros;

	}

	void setname(string pName)
	{
		name = pName;
	}

	string getName()
	{
		return name;
	}
};

/*
	GRAYSCALEIMAGE - Turns the orignal colored image into a grayscale image
       Inputs - Colored image with RGB values in each pixel
	   Return - Image in grayscale (CV_8U Type) 
*/
Mat toGrayscale(Mat &colorImage)
{
    Mat grayImage = Mat::zeros(colorImage.size(), CV_8U);

	// visit every pixel in image
	for (int y = 0;y<colorImage.rows;y++)
	{
		for (int x = 0;x<colorImage.cols;x++)
		{
			//extract pixel color data
			Vec3b intensity = colorImage.at<Vec3b>(y, x);
			int b = intensity.val[0];  //blue value of pixel
			int g = intensity.val[1];  //green value of pixel
			int r = intensity.val[2];  // red value of pixel

			// Formula to convert RGB values to grayscale intensity
			double gray = (0.1140 * b) + (0.5870 * g) + (0.2989 * r);

			// set corresponding pixel in grayscale image
			grayImage.at<uchar>(y, x) = gray;
		}
	}

	return grayImage;
}

void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--) {
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}

void medianFilter(Mat src, Mat dst)
{
	//create a sliding window of size 9
	int window[9];
    
	//mark all pixels in dst image as black
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dst.at<uchar>(y, x) = 0.0;
 
	for (int y = 1; y < src.rows - 1; y++)
	{
		for (int x = 1; x < src.cols - 1; x++) 
		{

			// Pick up window element

			window[0] = src.at<uchar>(y - 1, x - 1);
			window[1] = src.at<uchar>(y, x - 1);
			window[2] = src.at<uchar>(y + 1, x - 1);
			window[3] = src.at<uchar>(y - 1, x);
			window[4] = src.at<uchar>(y, x);
			window[5] = src.at<uchar>(y + 1, x);
			window[6] = src.at<uchar>(y - 1, x + 1);
			window[7] = src.at<uchar>(y, x + 1);
			window[8] = src.at<uchar>(y + 1, x + 1);

			// sort the window to find median
			insertionSort(window);

			// assign the median to centered element of the matrix
			dst.at<uchar>(y, x) = window[4];
		}
	}
	
}

// Computes the x component of the gradient vector
// at a given point in a image.
// returns gradient in the x direction
int xGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y - 1, x - 1) +
		2 * image.at<uchar>(y, x - 1) +
		image.at<uchar>(y + 1, x - 1) -
		image.at<uchar>(y - 1, x + 1) -
		2 * image.at<uchar>(y, x + 1) -
		image.at<uchar>(y + 1, x + 1);
}

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction

int yGradient(Mat image, int x, int y)
{
	return image.at<uchar>(y - 1, x - 1) +
		2 * image.at<uchar>(y - 1, x) +
		image.at<uchar>(y - 1, x + 1) -
		image.at<uchar>(y + 1, x - 1) -
		2 * image.at<uchar>(y + 1, x) -
		image.at<uchar>(y + 1, x + 1);
}

void sobrelFilter(Mat src, Mat mag, Mat angle)
{
	double gx, gy, sum;

	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			mag.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < src.rows - 1; y++) 
	{
		for (int x = 1; x < src.cols - 1; x++) 
		{
			gx = xGradient(src, x, y);
			gy = yGradient(src, x, y);
			sum = abs(gx) + abs(gy);
			sum = sum > 255 ? 255 : sum;
			sum = sum < 0 ? 0 : sum;
			mag.at<uchar>(y, x) = sum;
	
			double theta = (atan2(gy, gx) * 180)/ 3.14; 
			/* Convert actual edge direction to approximate value */
			if (((theta < 22.5) && (theta > -22.5)) || (theta > 157.5) || (theta < -157.5))
				angle.at<uchar>(y, x)  = 0;
			if (((theta > 22.5) && (theta < 67.5)) || ((theta < -112.5) && (theta > -157.5)))
				angle.at<uchar>(y, x)  = 45;
			if (((theta > 67.5) && (theta < 112.5)) || ((theta < -67.5) && (theta > -112.5)))
				angle.at<uchar>(y, x)  = 90;
			if (((theta > 112.5) && (theta < 157.5)) || ((theta < -22.5) && (theta > -67.5)))
				angle.at<uchar>(y, x)  = 135; 
		}
	}
}

void findEdge(Mat edges, Mat mag, Mat angle, int rowShift, int colShift, int row, int col, int dir, int lowerThreshold)
{
	int W = mag.cols;
	int H = mag.rows;
	int newRow;
	int newCol;
	bool edgeEnd = false;

	/* Find the row and column values for the next possible pixel on the edge */
	if (colShift < 0) 
	{
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < W - 1) 
	{
		newCol = col + colShift;
	}
	else
		edgeEnd = true;		// If the next pixel would be off image, don't do the while loop
	
	if (rowShift < 0) 
	{
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < H - 1) 
	{
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;

	/* Determine edge directions and gradient strengths */
	while (!edgeEnd && (mag.at<uchar>(newRow, newCol) > lowerThreshold) && (angle.at<uchar>(newRow, newCol) == dir)) 
	{
		edges.at<uchar>(newRow, newCol) = 255;

		if (colShift < 0) 
		{
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < W - 1) 
		{
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		
		if (rowShift < 0) 
		{
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < H - 1) 
		{
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
	}
}

void traceEdge(Mat mag, Mat angle, Mat edges, int upperThreshold, int lowerThreshold)
{
	int H = mag.rows;
	int W = mag.cols;

	/* Trace along all the edges in the image */
	for (int y = 1; y < H - 1; y++)
	{
		for (int x = 1; x < W - 1; x++) 
		{
			bool edgeEnd = false;
			if (mag.at<uchar>(y,x) > upperThreshold) 
			{		
				// Check to see if current pixel has a high enough gradient strength to be part of an edge
				/* Switch based on current pixel's edge direction */
				switch (angle.at<uchar>(y, x)) 
				{			
				case 0:
					findEdge(edges, mag, angle, 0, 1, y, x, 0, lowerThreshold);
					break;
				case 45:
					findEdge(edges, mag, angle, 1, 1, y, x, 45, lowerThreshold);
					break;
				case 90:
					findEdge(edges, mag, angle, 1, 0, y, x, 90, lowerThreshold);
					break;
				case 135:
					findEdge(edges, mag, angle, 1, -1, y, x, 135, lowerThreshold);
					break;
				default:
					break;
				}
			}
			
		}
	}
}

void suppressNonMax(Mat edges, Mat angles,Mat mag, int rowShift, int colShift, int row, int col, int dir)//, int lowerThreshold)
{
	int H = edges.rows;
	int W = edges.cols;
	int newRow = 0;
	int newCol = 0;
	unsigned long i;
	bool edgeEnd = false;
	float nonMax[416][3];			// Temporarily stores gradients and positions of pixels in parallel edges
	int pixelCount = 0;					// Stores the number of pixels in parallel edges
	int count;						// A for loop counter
	int max[3];						// Maximum point in a wide edge

	if (colShift < 0) 
	{
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < W - 1) 
	{
		newCol = col + colShift;
	}
	else
		edgeEnd = true;		
	
	if (rowShift < 0) 
	{
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < H - 1) 
	{
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;
	
	/* Find non-maximum parallel edges tracing up */
	while (!edgeEnd && (edges.at<uchar>(newRow,newCol) == 255) && (angles.at<uchar>(newRow, newCol) == dir))
	{
		if (colShift < 0) {
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < W - 1) {
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		if (rowShift < 0) {
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < H - 1) {
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
		nonMax[pixelCount][0] = newRow;
		nonMax[pixelCount][1] = newCol;
		nonMax[pixelCount][2] = mag.at<uchar>(newRow,newCol);
		pixelCount++;
		i = (unsigned long)(newRow * 3 * W + 3 * newCol);
	}

	/* Find non-maximum parallel edges tracing down */
	edgeEnd = false;
	colShift *= -1;
	rowShift *= -1;
	if (colShift < 0) {
		if (col > 0)
			newCol = col + colShift;
		else
			edgeEnd = true;
	}
	else if (col < W - 1) {
		newCol = col + colShift;
	}
	else
		edgeEnd = true;
	if (rowShift < 0) {
		if (row > 0)
			newRow = row + rowShift;
		else
			edgeEnd = true;
	}
	else if (row < H - 1) {
		newRow = row + rowShift;
	}
	else
		edgeEnd = true;
	i = (unsigned long)(newRow * 3 * W + 3 * newCol);
	while ((angles.at<uchar>(newRow, newCol) == dir) && !edgeEnd && (edges.at<uchar>(newRow, newCol) == 255)) 
	{
		if (colShift < 0) 
		{
			if (newCol > 0)
				newCol = newCol + colShift;
			else
				edgeEnd = true;
		}
		else if (newCol < W - 1) 
		{
			newCol = newCol + colShift;
		}
		else
			edgeEnd = true;
		
		if (rowShift < 0) 
		{
			if (newRow > 0)
				newRow = newRow + rowShift;
			else
				edgeEnd = true;
		}
		else if (newRow < H - 1)
		{
			newRow = newRow + rowShift;
		}
		else
			edgeEnd = true;
		nonMax[pixelCount][0] = newRow;
		nonMax[pixelCount][1] = newCol;
		nonMax[pixelCount][2] = mag.at<uchar>(newRow, newCol);
		pixelCount++;

	}

	/* Suppress non-maximum edges */
	max[0] = 0;
	max[1] = 0;
	max[2] = 0;
	for (count = 0; count < pixelCount; count++) {
		if (nonMax[count][2] > max[2]) {
			max[0] = nonMax[count][0];
			max[1] = nonMax[count][1];
			max[2] = nonMax[count][2];
		}
	}
	for (count = 0; count < pixelCount; count++) 
	{
		int y = nonMax[count][0];
		int x = nonMax[count][1];
		edges.at<uchar>(y,x) = 0;
	}
}

void edgeSuppression(Mat edges, Mat angles, Mat mag)
{
	int H = edges.rows;
	int W = edges.cols;
	/* Non-maximum Suppression */
	for (int y = 1; y < H - 1; y++) 
	{
		for (int x = 1; x < W - 1; x++) 
		{
			if (edges.at<uchar>(y,x) == 255) 
			{		// Check to see if current pixel is an edge
					/* Switch based on current pixel's edge direction */
				switch (angles.at<uchar>(y,x))
				{
				case 0:
					suppressNonMax(edges, angles, mag, 1, 0, y, x, 0);
					break;
				case 45:
					suppressNonMax(edges, angles, mag, 1, -1, y, x, 45);
					break;
				case 90:
					suppressNonMax(edges, angles, mag, 0, 1, y, x, 90);
					break;
				case 135:
					suppressNonMax(edges, angles, mag, 1, 1, y, x, 135);
					break;
				default:
					break;
				}
			}
		}
	}
}

Mat translateImg(Mat &img, int offsetx, int offsety)
{
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return trans_mat;
}

void topSide(Mat edges, vector<Point> &edgePoints, int pTop)
{
	vector<Point> tmpVec;
	
	for (int col = 0; col < edges.cols; col++)
	{
		for (int row = 0; row <= (edges.rows / 2); row++)
		{
			if (edges.at<uchar>(row, col)== 255)
			{
				if (edgePoints.size() != 0)
				{
					Point tmp = edgePoints.back();

					if ((abs(row - tmp.y) < 10))
						tmpVec.push_back(Point(col, row));
					break;
				}
				else
				{
					if (row - pTop < 20)
					{
						tmpVec.push_back(Point(col, row));
					}
				}
			}  
		}				   
	}

	int size = tmpVec.size();
	int prev = tmpVec.at(0).x;
	int i = 1;
	int y;
	edgePoints.push_back(tmpVec.at(0));
	while ( i < size )
	{
		int tmpX = tmpVec.at(i).x;
		
		if ((tmpX - prev) >= 10)
		{
			edgePoints.push_back(tmpVec.at(i));
			prev = tmpX;
			i++;
		}
		else
		{
			i++;
		}
	}
	edgePoints.push_back(tmpVec.at(size-1));
	
}

void rightSide(Mat edges, vector<Point> &edgePoints)
{
	vector<Point> tmpVec;
	Point start = edgePoints.back();

	for (int row = start.y; row < (edges.rows - 1); row++)
	{
		for (int col = edges.cols - 1; col >= (edges.cols/2); col--)
		{
			if (edges.at<uchar>(row, col) == 255)
			{
				Point tmp = edgePoints.back();

				if ((abs(tmp.x - col) < 30))	
				tmpVec.push_back(Point(col, row));

				break;
			}
		}
	}  

	int size = tmpVec.size();
	if (size != 0)
	{
		int prev = tmpVec.at(0).y;
		int i = 1;
		int y;
		edgePoints.push_back(tmpVec.at(0));
		while (i < size)
		{
			int tmpX = tmpVec.at(i).y;

			if ((tmpX - prev) >= 10)
			{
				edgePoints.push_back(tmpVec.at(i));
				prev = tmpX;
				i++;
			}
			else
			{
				i++;
			}
		}
		edgePoints.push_back(tmpVec.at(size - 1));
	}
}

void bottomSide(Mat edges, vector<Point> &edgePoints)
{
	vector<Point> tmpVec;
	Point start = edgePoints.back();

	for (int col = start.x ; col > 0; col--)
	{
		for (int row = edges.rows - 1; row >= (edges.rows / 2); row--)
		{
			if (edges.at<uchar>(row, col) == 255)
			{		
				Point tmp = edgePoints.back();

				if ((abs(tmp.y - row) < 30))
					tmpVec.push_back(Point(col, row));

				break;
			}
		}
	}

	int size = tmpVec.size();
	if (size != 0)
	{
		int prev = tmpVec.at(0).x;
		int i = 1;
		int y;
		edgePoints.push_back(tmpVec.at(0));
		while (i < size)
		{
			int tmpX = tmpVec.at(i).x;

			if ((prev - tmpX) >= 10)
			{
				edgePoints.push_back(tmpVec.at(i));
				prev = tmpX;
				i++;
			}
			else
			{
				i++;
			}
		}
		edgePoints.push_back(tmpVec.at(size - 1));
	}
}

void leftSide(Mat edges, vector<Point> &edgePoints)
{
	vector<Point> tmpVec;
	Point start = edgePoints.back();

	for (int row = start.y; row > 0; row--)
	{
		for (int col = 0; col <= (edges.cols / 2); col++)
		{
			if (edges.at<uchar>(row, col) == 255)
			{
				Point tmp = edgePoints.back();

				if ((abs(tmp.x- col) < 30) )
					tmpVec.push_back(Point(col, row));

				break;
			}
		}
	}

	int size = tmpVec.size();
	
	if (size != 0)
	{
		int prev = tmpVec.at(0).y;
		int i = 1;
		int y;
		edgePoints.push_back(tmpVec.at(0));
		while (i < size)
		{
			int tmpX = tmpVec.at(i).y;

			if ((prev - tmpX) >= 10)
			{
				edgePoints.push_back(tmpVec.at(i));
				prev = tmpX;
				i++;
			}
			else
			{
				i++;
			}
	}
	edgePoints.push_back(tmpVec.at(size - 1));
	}
}

Mat outline(Mat &edges, int pTop)
{
	Mat img = Mat::zeros(edges.rows, edges.cols, CV_8U);
	vector<Point> edgePoints = {};
	
	//get sides of shape
	topSide(edges, edgePoints, pTop);
	rightSide(edges, edgePoints);
	bottomSide(edges, edgePoints);
	leftSide(edges, edgePoints);
	
	const Point *pts = (const Point*) Mat(edgePoints).data;
	int npts = Mat(edgePoints).rows;

	polylines(img, &pts, &npts, 20, false, 255, 1, CV_AA, 0);

	fillPoly(img, &pts, &npts, 1, 255);

	return img;
}

int centerImage(Mat &orignialImg, Mat &img)
{
	//find highest Point
	//find lowest Point
	//find rightmost Point
	//find leftmost point
	int top, bottom, right, left;

	top = img.rows;

	for (int col = 0; col < img.cols; col++)
	{
		for (int row = 0; row < (img.rows / 2); row++)
		{
			if ((img.at<uchar>(row, col) == 255) && (row < top))
				top = row;
		}
	}

	right = 0;
	for (int row = 0; row < img.rows; row++)
	{
		for (int col = img.cols - 1; col > (img.cols / 2); col--)
		{
			if ((img.at<uchar>(row, col) == 255) && (col > right))
				right = col;
		}
	}

	bottom = 0;
	for (int col = img.cols - 1; col > 0; col--)
	{
		for (int row = img.rows - 1; row > (img.rows / 2); row--)
		{
			if ((img.at<uchar>(row, col) == 255) && (row > bottom))
				bottom = row;
		}
	}

	left = img.cols;
	for (int row = 0; row < (img.rows - 1); row++)
	{
		for (int col = 0; col < (img.cols / 2); col++)
		{
			if ((img.at<uchar>(row, col) == 255) && (col < left))
				left = col;
		}
	}


	int rowshift = ((img.rows / 2) - (abs(bottom - top) / 2)) - top;
    int colshift = ((img.cols / 2) - (abs(right - left) / 2)) - left;

	translateImg(img, colshift, rowshift);

	translateImg(orignialImg, colshift, rowshift);

	return top;	
}

string sortColor(int &r, int &y, int &g, int &c, int &b, int &m)
{
	if (r > y && r > g && r > c && r > b && r > m)
	{
		if (r == 0)
			return "none";
		r = 0;
		return "red";
	}
	if (y > r && y > g && y > c && y > b && y > m)
	{
		if (y == 0)
			return "none";
		y = 0;
		return "yellow";
	}
	if (g > y && g > r && g > c && g > b && g > m)
	{
		if (g == 0)
			return "none";
		g = 0;
		return "green";
	}
	if (c > y && c > g && c > r && c > b && c > m)
	{
		if (c == 0)
			return "none";
		c = 0;
		return "cyan";
	}
	if (b > y && b > g && b > c && b > r && b > m)
	{
		if (b == 0)
			return "none";
		b = 0;
		return "blue";
	}
	if (m > y && m > g && m > c && m > b && m > r)
	{
		if (m == 0)
			return "none";
		m = 0;
		return "magenta";
	}
}

Item getColors(Mat &img, Mat &edges)
{
	int red = 0;
	int yellow = 0;
	int green = 0;
	int cyan = 0;
	int blue = 0;
	int magenta = 0;
	Mat hsv;
	cvtColor(img, hsv, CV_BGR2HSV);
	for (int row = 0; row < edges.rows - 1; row++)
	{
		for (int col = 0; col < edges.cols - 1; col++)
		{
			if (edges.at<uchar>(row, col) == 255)
			{
				
				Vec3b color = hsv.at<Vec3b>(row, col);
				int hue = color[0];
				
				if (hue == 0)
					break;
				if (hue <= 15 || hue > 165)	
					red++;
				else if (hue > 15 && hue <= 45)
					yellow++;
				else if (hue > 45 && hue <= 75)
					green++;
				else if (hue > 75 && hue <= 105)
					cyan++;
				else if	(hue > 105 && hue <= 135)
					blue++;
				else if	(hue > 135 && hue <= 165)
					magenta++;
			}

		}
	}

	string firstColor = sortColor(red,yellow,green,cyan,blue,magenta);
	string secondColor = sortColor(red, yellow, green, cyan, blue, magenta);
	string thirdColor = sortColor(red, yellow, green, cyan, blue, magenta);
	
	int nonZeros = countNonZero(edges);
	
	Item temp = Item(edges, firstColor, secondColor, thirdColor,nonZeros);

	return temp;
}

Item imageProcessing(Mat img)
{
	destroyAllWindows();
	Mat grayImage = Mat::zeros(img.size(), CV_8U);;
	Mat blur = Mat::zeros(img.size(), CV_8U);
	Mat sobrelMag = Mat::zeros(img.size(), CV_8U);
	Mat sobrelAngle = Mat::zeros(img.size(), CV_8U);
	Mat edges = Mat::zeros(img.size(), CV_8U);
	Mat shape = Mat::zeros(img.size(), CV_8U);
	vector<Point> edgePoints;
	//Before changing to grayscale
	imshow("Orignal Image", img);			  
	grayImage = toGrayscale(img);
	//After changing to grayscale
	imshow("Grayscale Image", grayImage);
	// filter image to create blur
	medianFilter(grayImage, blur);
	//Sobrel Filter to find gradients
	sobrelFilter(blur, sobrelMag, sobrelAngle);
    // Trace the edge along gradients
	traceEdge(sobrelMag, sobrelAngle, edges, 150, 40);
	// Suppress edges to create thiner edge that follows smoother lines
	edgeSuppression(edges, sobrelAngle, sobrelMag);
	
	int top = centerImage(img,edges);
	shape = outline(edges, top);

	waitKey(60);
	Item temp = getColors(img,shape);

	return temp;
}

Mat getPicture()
{
	VideoCapture videoStream(2);   //0 is the id of video device.0 if you have only one camera.
	

	if (!videoStream.isOpened()) { //check if video device has been initialised
		cout << "cannot open camera";
	}
	Mat displayImage = Mat(Size(480, 640), CV_8UC3);;
	//unconditional loop
	while (true)
	{
		Mat cameraFrame = Mat(Size(480, 640), CV_8UC3);
		videoStream.read(cameraFrame);
		cameraFrame.copyTo(displayImage);
		rectangle(displayImage, Point(20, 20), Point(620, 460), Scalar(0, 255, 0),3);
		circle(displayImage, Point(320, 240),3,Scalar(0,255,0),3);
		
		imshow("Camera", displayImage);
		moveWindow("Camera", 0, 0);
		if (waitKey(1) == 27)
		{
			videoStream.release();
			destroyAllWindows();
			return cameraFrame;

		}

	}
}

void addItem(Item tmp, vector<Item> &items)
{
	char answer;
	string name;

	cout << "Would you like to add this item? (y/n)";
	
	cin >> answer;
	cin.ignore();

	while(answer != 'y' && answer != 'n')
	{
		cin.clear();
		cin >> answer;
	    cin.ignore();
	}

	if (answer == 'y')
	{
		cout << "What is the name of the Item? (No spaces) ";
		cin >> name;
		cin.ignore();
		tmp.setname(name);
		items.push_back(tmp);
		cout << name << " was added." << endl;
	}
	else if (answer == 'n')
		return;
}

int compareDiff(vector<int> &pDiff)
{
	int index = 0;
	int low = 50000;

	for (int i = 0; i < pDiff.size(); i++)
	{
		if ((low > pDiff.at(i)) && pDiff.at(i) != -1)
		{
			low = pDiff.at(i);
			index = i;
		}
	}

	if (pDiff.at(index) == -1)
		return -1;
	pDiff.at(index) = -1;

	return index;
}

void compareItems(Item &pItem, vector<Item> &items)
{
	Mat dst;
	vector<int> difference;
	
	if (items.size() == 0)
	{
		addItem(pItem, items);
		return;
	}

	int diff = pItem.nonZeros * 1.75;
	for (int i = 0; i < items.size(); i++)
	{
		int diff = items.at(i).nonZeros * 1.5;
					  
		compare(pItem.shape, items.at(i).shape, dst, CMP_NE);
		int j = countNonZero(dst);

	  
		if (j < diff)
		{
			cout << "diff added" << endl;
			difference.push_back(j);
		}
		else
			difference.push_back(-1);
	}

	int j;
    while(true)
	{	

		j = compareDiff(difference);
		if (j == -1)
			break;
			
		if (pItem.firstColor == items.at(j).firstColor)
		{
			if (pItem.secondColor == items.at(j).secondColor)
			{			
				cout << " This item is " << items.at(j).getName() << endl;			
				return;			
			}			
		}
	}
	addItem(pItem, items);
}

/** @function main */
int main(int argc, char** argv)
{
	vector<Item> items;
	
	char input = 'c';

	while (input != 'q')
	{
		Mat img = Mat(Size(480, 640), CV_8UC3);
		if (input == 'c')
		{
			img = getPicture();
		}
		Item tmp = imageProcessing(img);

		compareItems(tmp, items);
		
		cout << "If you wish to take another picture press (c). If you wish to quit press (q) ";
		cin >> input;
		cin.ignore();

	}
	
	 
	return 0;
}
