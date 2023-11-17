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
		mOriginalWindowSize = Size(oldC->orig_window_size);

		std::vector<CvAvgComp> vecAvgComp;

		MemStorage storage(cvCreateMemStorage(0));
		CvMat _image = image;
		CvSeq* _objects = viscasHaarDetectObjectsForROC(&_image, oldCascade, storage, rejectLevels, levelWeights, scaleFactor,
			minNeighbors, flags, minObjectSize, maxObjectSize, false, this);
		Seq<CvAvgComp>(_objects).copyTo(vecAvgComp);
		objects.resize(vecAvgComp.size());
		std::transform(vecAvgComp.begin(), vecAvgComp.end(), objects.begin(), getRect());

		numDetections.resize(vecAvgComp.size());
		std::transform(vecAvgComp.begin(), vecAvgComp.end(), numDetections.begin(), getNeighbors());
	}
	else
	{
		cout << "New format cascade not supported for visualisation" << endl;
		detectMultiScaleNoGrouping(image, objects, rejectLevels, levelWeights, scaleFactor, minObjectSize, maxObjectSize);
		const double GROUP_EPS = 0.2;
		groupRectangles(objects, numDetections, minNeighbors, GROUP_EPS);
	}
}

int VisualCascade::getDepth() const
{
	return mVisualisationDepth;
}

void VisualCascade::setImagePath(string path)
{
	mImagePath = path;
	mFrameCounter = 0;
}

void VisualCascade::setVideo(string path)
{
	mVideoPath = path;
}

void VisualCascade::setIntegral(cv::Size integralSize, cv::Mat sum, cv::Mat sqsum)
{
	mIntegralSize = integralSize;
	mSum = sum;
	mSqsum = sqsum;
}

void VisualCascade::setWindow(int x, int y, Size detectWindowSize, Size ssz)
{
	Size showWindowSize(static_cast<int>(mShowScale * detectWindowSize.width), static_cast<int>(mShowScale * detectWindowSize.height));
	int xOffset = (mProgress.cols - showWindowSize.width)  * x / ssz.width;
	int yOffset = (mProgress.rows - showWindowSize.height) * y / ssz.height;
	mWindow = Rect(Point(xOffset, yOffset), showWindowSize);
}

void VisualCascade::show(const vector<int>& branches, int featureIndex, int nFeatures, const CvHidHaarFeature& feature)
{
	stringstream description;
	description << "Branch: ";
	for (unsigned index = 0; index < branches.size(); index++)
	{
		if (index > 0) description << "-";
		description << branches[index];
	}
	show(description.str(), feat