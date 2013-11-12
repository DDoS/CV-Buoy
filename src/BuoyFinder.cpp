#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <math.h>

using namespace std;
using namespace cv;

// The lower bound of the threshold (currently a shade of red)
#define LOW_COLOR 170, 160, 70
// The higher bound of the threshold (currently a shade of red)
#define HIGH_COLOR 180, 256, 256
// Video files
#define FILE "/Users/Aleksi/Desktop/RoboSub 2013 Buoy.mp4"
//#define FILE "/Users/Aleksi/Desktop/RoboSub 2013 Buoy Closeup.mp4"

static void applyColorCorrection(IplImage* image) {
	int width = image->width; 
	int height = image->height;
	int channels = image->nChannels;
	int step = image->widthStep;
	uchar minB = 255, minG = 255, minR = 255;
	uchar maxB = 0, maxG = 0, maxR = 0;
	uchar *data = (uchar*) image -> imageData;    
	for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
			uchar b = data[i * step + j * channels];
			uchar g = data[i * step + j * channels + 1];
			uchar r = data[i * step + j * channels + 2];
			if (b < minB) {
				minB = b;
			}
			if (g < minG) {
				minG = g;
			}
			if (r < minR) {
				minR = r;
			}
			if (b > maxB) {
				maxB = b;
			}
			if (g > maxG) {
				maxG = g;
			}
			if (r > maxR) {
				maxR = r;
			}
		}
	}
	//cout << int(maxB) << " " << int(minB) << endl;
    //cout << int(maxG) << " " << int(minG) << endl;
    //cout << int(maxR) << " " << int(minR) << endl;
	float deltaB = float(maxB - minB);
    float deltaG = float(maxG - minG);
    float deltaR = float(maxR - minR);
    deltaR = 150;
	for (int i = 0; i < height; i++) {
    	for (int j = 0; j < width; j++) {
    		uchar b = data[i * step + j * channels];
			uchar g = data[i * step + j * channels + 1];
			uchar r = data[i * step + j * channels + 2];
			data[i * step + j * channels] = floor((b - minB) / deltaB * 255);
			data[i * step + j * channels + 1] = floor((g - minG) / deltaG * 255);
			data[i * step + j * channels + 2] = min((r - minR) / deltaR * 255, float(255));
		}
	}
}

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
	// Get the first video, fail if it's not available
	CvCapture* video = cvCaptureFromFile(FILE);
	if (video == NULL) {
		cout << "No available video!" << endl;
		return 1;
	}
	// Get the delay between each frame
	int frameRate = cvGetCaptureProperty(video, CV_CAP_PROP_FPS);
	int frameDelay = int(round((double(1) / frameRate) * 1000));
	// Get the horizontal resolution (width) of the image to place the windows correctly
	double horizontalResolution = cvGetCaptureProperty(video, CV_CAP_PROP_FRAME_WIDTH);
	// Create and place the windows
	cvNamedWindow("Input");
	cvMoveWindow("Input", 100, 100);
	cvNamedWindow("Output");
	cvMoveWindow("Output", 110 + int(horizontalResolution), 100);
	// Get the first frame from the video
	IplImage* frame = cvQueryFrame(video);
	// As long as we have a frame
	while (frame != NULL) {
		// Clone the frame, we can't work off the captured one
		frame = cvCloneImage(frame);
		// Apply color correction
		applyColorCorrection(frame);
		// Threshold it
		IplImage* thresholdImage = getThresholdImage(frame);
		// Show the original image and the thresholded one
		cvShowImage("Input", frame);
		cvShowImage("Output", thresholdImage);
		// Release the thresholded image
		cvReleaseImage(&thresholdImage);
		// Wait 50ms for the escape key
		int key = cvWaitKey(frameDelay);
		if (key == 27 || key == 1048603) {
			// If escape was pressed, exit
			break;
		}
		// Query the next frame
		frame = cvQueryFrame(video);
	}
	// Release the frame (the clone), the windows and the video
	cvReleaseImage(&frame);
	cvDestroyAllWindows();
	cvReleaseCapture(&video);
	// Return successfully
	return 0;
}
