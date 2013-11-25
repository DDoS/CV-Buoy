#include <iostream>
#include <random>
#include <cv.h>
#include <highgui.h>
#include <math.h>

#define _USE_MATH_DEFINES

using namespace std;
using namespace cv;

// Video files
#define FILE "RoboSub 2013 Buoy.mp4"
//#define FILE "RoboSub 2013 Buoy Closeup.mp4"

// The color of a white sample in the footage for color correction
static const Scalar colorSample(170, 160, 170);
// The desired white for the sample after color correction
static const Scalar targetColor(20, 40, 220);
// The color correction to apply to get correct whites
static Scalar colorCorrection;

// A struct for the buoy color range (in HSV)
struct BuoyColorProfile {
	Scalar lowColor;
	Scalar highColor;

	BuoyColorProfile(Scalar lowColor, Scalar highColor)
		: lowColor(lowColor), highColor(highColor) {}
} ;

// The two default buoy color profiles
static const BuoyColorProfile RED_BUOY(Scalar(-15, 30, 70), Scalar(15, 255, 255));
static const BuoyColorProfile YELLOW_BUOY(Scalar(10, 30, 40), Scalar(55, 255, 255));

// A struct for the information regarding detected buoys (center position and radius)
struct DetectedBuoy {
	Point center;
	int radius;

	DetectedBuoy(Point center, int radius)
		: center(center), radius(radius) {}
} ;

// Computes some values for color correction
static void initColorCorrection() {
	// Get the gray scale equivalent of the color sample by averaging, this is the correct color
	int whiteAverage = (colorSample[0] + colorSample[1] + colorSample[2]) / 3;
	// Create a scalar for the correct color
	Scalar balancedWhite(whiteAverage, whiteAverage, whiteAverage);
	// The amount of color correction to apply is the difference between the sample color and the correct color,
	// minus the difference between the target color and the correct color
	colorCorrection = colorSample - balancedWhite - (targetColor - balancedWhite);
}

// Clamps the integer between low and high so that low <= n <= high
inline int clamp(int n, int low, int high) {
	n = n > high ? high: n;
	return n < low ? low : n;
}

// Applies color correction to the color
static void colorCorrect(uchar* b, uchar* g, uchar* r) {
	// Calculate the difference between the color and the sample color
	float bDiff = colorSample[0] - *b;
	float gDiff = colorSample[1] - *g;
	float rDiff = colorSample[2] - *r;
	// Calculate a weight, the closer the color is the sample, the closer the weight is to 1,
	// falling off linearly to 0 when nearing 0 or 255
	float bWeight = 1 - (bDiff > 0 ? bDiff / colorSample[0] : -bDiff / (255 - colorSample[0]));
	float gWeight = 1 - (gDiff > 0 ? gDiff / colorSample[1] : -gDiff / (255 - colorSample[1]));
	float rWeight = 1 - (rDiff > 0 ? rDiff / colorSample[2] : -rDiff / (255 - colorSample[2]));
	// Apply the color correction, weighting it
	*b = clamp(*b - colorCorrection[0] * bWeight, 0, 255);
	*g = clamp(*g - colorCorrection[1] * gWeight, 0, 255);
	*r = clamp(*r - colorCorrection[2] * rWeight, 0, 255);
}

// Applies the color correction to the image
static void applyColorCorrection(Mat* image) {
	// Info for image size
	int rows = image->rows;
	int cols = image->cols * image->channels();
	// Iterate over entire image, applying the color correction 
	uchar* row;
	for (int i = 0; i < rows; i++) {
		row = image->ptr<uchar>(i);
		for (int j = 0; j < cols; j += 3) {
			colorCorrect(&row[j], &row[j + 1], &row[j + 2]);
		}
	}
}

// An thresholding function that supports negative hue values (important for red)
static void inRangeWrapped(Mat* source, Scalar low, Scalar high, Mat* destination) {
	// Info for image size
	int srcRows = source->rows;
	int srcCols = source->cols * source->channels();
	int dstRows = destination->rows;
	int dstCols = destination->cols * destination->channels();
	// Ensure parameters are correct
	assert(srcCols == dstCols * 3);
	assert(srcRows == dstRows);
	// Ensure the hue range is between 0 and 180
	assert(low[0] <= high[0]);
	int lowH = (int(low[0]) + 180) % 180;
	int highH = (int(high[0]) + 180) % 180;
	// Iterate over entire image 
	uchar* srcRow;
	uchar* dstRow;
	for (int i = 0; i < srcRows; i++) {
		srcRow = source->ptr<uchar>(i);
		dstRow = destination->ptr<uchar>(i);
		for (int j = 0; j < srcCols; j += 3) {
			int h = int(srcRow[j]);
			int s = int(srcRow[j + 1]);
			int v = int(srcRow[j + 2]);
			bool hPass;
			if (lowH < highH) {
				hPass = h >= lowH && h <= highH;
			} else {
				hPass = (h >= lowH && h <= 180) || (h >= 0 && h <= highH);
			}
			if (hPass && s >= low[1] && s <= high[1] && v >= low[2] && v <= high[2]) {
				dstRow[j / 3] = 255;
			} else {
				dstRow[j / 3] = 0;
			}
		}
	}
}


// Gets the black and white thresholded image from the original color image
static void getThresholdImage(Mat* image, BuoyColorProfile buoy, Mat* destination) {
	// Smooth to reduce noise
	GaussianBlur(*image, *image, Size(3, 3), 0);
	// Convert from RGB to HSV
	Mat HSVImage(image->rows, image->cols, CV_8UC3);
	cvtColor(*image, HSVImage, CV_BGR2HSV);
	// Generate the threshold image
	inRangeWrapped(&HSVImage, buoy.lowColor, buoy.highColor, destination);
}

// Uses a Monte-Carlo method to check is the disk is at least as dense (in the sense of no missing pieces) as the desired density (as a percentage)
// It's used to discard invalid circles formed by random contours 
static bool checkDisk(Mat* image, Point* center, int radius, int sampleCount, float minDensity) {
	// Initialize the random number generator and ranges for the full disk in polar coordinates
	static mt19937 generator;
	static uniform_real_distribution<float> angleRange(0, 2 * M_PI);
	uniform_real_distribution<float> radiusRange(0, radius);
	// Total density of the samples
	float density = 0;
	// Do the sampling
 	for (int i = 0; i < sampleCount; i++) {
 		// Pick a random point on the disk
 		float angle = angleRange(generator);
 		float radius = radiusRange(generator);
 		int xx = int(radius * cos(angle) + center->x);
 		int yy = int(radius * sin(angle) + center->y);
 		// Add the normalized weight
		density += float(image->at<uchar>(yy, xx)) / 255;
 	}
 	// Normalize the total density
 	density /= sampleCount;
 	return density >= minDensity;
}

// The main method of the detection algorithm, use RED_BUOY or YELLOW_BUOY for color profiles
static vector<DetectedBuoy> detectBuoys(Mat* image, BuoyColorProfile buoy) {
	// Compute color correction data from the color sample and target color once
	static bool isColorCorrectionComputed = false;
	if (!isColorCorrectionComputed) {
		initColorCorrection();
		isColorCorrectionComputed = true;
	}
	// Apply color correction
	applyColorCorrection(image);
	// Threshold it
	Mat thresholdImage(image->rows, image->cols, CV_8UC1);
	getThresholdImage(image, buoy, &thresholdImage);	
	// Blur the image an do circle detection
	GaussianBlur(thresholdImage, thresholdImage, Size(25, 25), 0);	
	vector<Vec3f> circles;
	HoughCircles(thresholdImage, circles, CV_HOUGH_GRADIENT, 2, thresholdImage.rows / 8, 225, 75);
	// Process the circles
	vector<DetectedBuoy> buoys;
	for (int i = 0; i < circles.size(); i++) {
  		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
 		int radius = cvRound(circles[i][2]);
 		// Check that we have a well defined disk
 		if (checkDisk(&thresholdImage, &center, radius, 30, 0.90f)) {
 			buoys.push_back(DetectedBuoy(center, radius));
 		}
 	}
 	return buoys;
}

// For demonstration purposes
int main(int argc, char* argv[]) {
	// Get the first video, fail if it's not available
	VideoCapture video(FILE);
	if (!video.isOpened()) {
		cout << "No available video!" << endl;
		return -1;
	}
	// Get the delay between each frame
	int frameRate = video.get(CV_CAP_PROP_FPS);
	int frameDelay = int(round((double(1) / frameRate) * 1000));
	// Get the horizontal resolution (width) of the image to place the windows correctly
	double horizontalResolution = video.get(CV_CAP_PROP_FRAME_WIDTH);
	// Create and place the window
	namedWindow("Detection");
	moveWindow("Detection", 100, 100);	
	// Get the first frame from the video
	Mat frame;
	bool hasFrames = video.grab();
	video.retrieve(frame);
	// As long as we have a frame
	while (hasFrames) {
		// Detect the buoys
		Mat frameClone = frame.clone();
		vector<DetectedBuoy> detected = detectBuoys(&frameClone, RED_BUOY);
		// Show the detected buoys on the frame
		for (int i = 0; i < detected.size(); i++) {
	 		// Draw the circle markers
 		 	circle(frame, detected[i].center, 3, Scalar(0, 255, 0), -1, 8, 0);
 			circle(frame, detected[i].center, detected[i].radius, Scalar(0, 0, 255), 3, 8, 0);
		}
		// Show the image with the detection markers
		imshow("Detection", frame);
		// Wait 50ms for the escape key
		int key = waitKey(frameDelay);
		if (key == 27 || key == 1048603) {
			// If escape was pressed, exit
			break;
		}
		// Query the next frame
		hasFrames = video.grab();
		video.retrieve(frame);
	}
	// Destroy the windows
	destroyAllWindows();
	// Return successfully
	return 0;
}
