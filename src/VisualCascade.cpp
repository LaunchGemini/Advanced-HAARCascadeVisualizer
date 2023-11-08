#include "VisualCascade.hpp"

#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "objdetect/VisualHaar.hpp"

using namespace cv;
using namespace std;

struct getRect { Rect operator ()(const CvAvgComp& e) const { return e.rect; } };
struct getNeighbors { int operator ()(const CvAvgComp& e) const { return e.neighbors; } };

string VisualCascade::mWindowName = "Cascade Visualiser";

void VisualCascade::detectMultiScale(InputArray showImage, InputArray _image, std::vector<Rect>& objects,
	double showScale, int depth, double scaleFactor, int minNeighbors,
	int flags, Size minObjectSize, Size maxObjectSize, unsigned steps)
{
	mShowScale = showScale;
	mVisualisationDepth = depth;
	mSteps = steps;
	mStepCounter = 0;
	std::vector<int> rejectLevels;
	std::vector<double> levelWeights;
	mProgress = showImage.getMat();
	Mat image = _image.getMat();
	CV_Assert(scaleFactor > 1 && image.depth() == CV_8U);

	if (empty()) return;

	std::vector<int> numDetections;
	if (isOldFormatCascade())
	{
		CvHaarClassifierCascade* oldC = static_cast<CvHaarClassifierCascade*>(getOldCascade());
		m