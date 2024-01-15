
#pragma once

#include <opencv2/objdetect.hpp>

namespace cv
{

void clipObjects(Size sz, std::vector<Rect>& objects,
                 std::vector<int>* a, std::vector<double>* b);

class FeatureEvaluator
{
public:
    enum
    {
        HAAR = 0,
        LBP  = 1,
        HOG  = 2
    };

    struct ScaleData
    {
        ScaleData() { scale = 0.f; layer_ofs = ystep = 0; }
        Size getWorkingSize(Size winSize) const
        {
            return Size(std::max(szi.width - winSize.width, 0),
                        std::max(szi.height - winSize.height, 0));
        }

        float scale;
        Size szi;
        int layer_ofs, ystep;
    };

    virtual ~FeatureEvaluator();

    virtual bool read(const FileNode& node, Size origWinSize);