#include <iostream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

// The lower bound of the threshold (currently a shade of red)
#define LOW_COLOR 170, 160, 70
// The higher bound of the threshold (currently a shade of red)
#define HIGH_COLOR 180, 256, 256

// Gets the black and white thresholded image from the original color image
static IplImage* getThresholdImage(IplImage* image) {
	// Smooth to reduce noise
	cvSmooth(image, image, CV_GAUSSIAN, 3, 3);
	// Convert from RGB to HSV
	IplImage* HSVImage = cvCreateImage(cvGetSize(image), IPL_DEPTH_8U, 3);
	cvCvtColor(image, HSVImage, CV_BGR2HSV);    
	// Create the threshold image
	IplImage* thresholdImage = cvCreateImage(cvGetSize(HSVImage), IPL_DEPTH_8U, 1);
	cvInRangeS(HSVImage, cvScalar(LOW_COLOR), cvScalar(HIGH_COLOR), thresholdImage);
	// Release the temporary HSV image
	cvReleaseImage(&HSVImage);
	// Return the threshold image
	return thresholdImage;
} 

int main(int argc,char *argv[]) {
	// Get the first camera, fail if it's not available
	CvCapture* camera = cvCaptureFromCAM(0);
	if (camera == NULL) {
		cout << "No available camera!" << endl;
		return 1;
	}
	// Get the horizontal resolution (width) of the image to place the windows correctly
	double horizontalResolution = cvGetCaptureProperty(camera, CV_CAP_PROP_FRAME_WIDTH);
	// Create and place the windows
	cvNamedWindow("Input");
	cvMoveWindow("Input", 100, 100);
	cvNamedWindow("Output");
	cvMoveWindow("Output", 110 + int(horizontalResolution), 100);
	// Get the first frame from the camera
	IplImage* frame = cvQueryFrame(camera);
	// As long as we have a frame
	while (frame != NULL) {
		// Clone the frame, we can't work of the captured one
		frame = cvCloneImage(frame); 
		// Threshold it
		IplImage* thresholdImage = getThresholdImage(frame);
		// Show the original image and the thresholded one
		cvShowImage("Input", frame);
		cvShowImage("Output", thresholdImage);
		// Release the thresholded image
		cvReleaseImage(&thresholdImage);
		// Wait 50ms for the escape key
		int key = cvWaitKey(10);
		if (key == 27 || key == 1048603) {
			// If escape was pressed, exit
			break;
		}
		// Query the next frame
		frame = cvQueryFrame(camera);
	}
	// Release the frame (the clone), the windows and the camera
	cvReleaseImage(&frame);
	cvDestroyAllWindows();
	cvReleaseCapture(&camera);
	// Return successfully
	return 0;
}
