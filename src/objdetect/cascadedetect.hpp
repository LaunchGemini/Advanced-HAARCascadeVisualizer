
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
    virtual Ptr<FeatureEvaluator> clone() const;
    virtual int getFeatureType() const;
    int getNumChannels() const { return nchannels; }

    virtual bool setImage(InputArray img, const std::vector<float>& scales);
    virtual bool setWindow(Point p, int scaleIdx);
    const ScaleData& getScaleData(int scaleIdx) const
    {
        CV_Assert( 0 <= scaleIdx && scaleIdx < (int)scaleData->size());
        return scaleData->at(scaleIdx);
    }
    virtual void getUMats(std::vector<UMat>& bufs);
    virtual void getMats();

    Size getLocalSize() const { return localSize; }
    Size getLocalBufSize() const { return lbufSize; }

    virtual float calcOrd(int featureIdx) const;
    virtual int calcCat(int featureIdx) const;

    static Ptr<FeatureEvaluator> create(int type);

protected:
    enum { SBUF_VALID=1, USBUF_VALID=2 };
    int sbufFlag;

    bool updateScaleData( Size imgsz, const std::vector<float>& _scales );
    virtual void computeChannels( int, InputArray ) {}
    virtual void computeOptFeatures() {}

    Size origWinSize, sbufSize, localSize, lbufSize;
    int nchannels;
    Mat sbuf, rbuf;
    UMat urbuf, usbuf, ufbuf, uscaleData;

    Ptr<std::vector<ScaleData> > scaleData;
};


class CascadeClassifierImpl : public BaseCascadeClassifier
{
public:
    CascadeClassifierImpl();
    virtual ~CascadeClassifierImpl();

    bool empty() const;
    bool load( const String& filename );
    void read( const FileNode& node );
    bool read_( const FileNode& node );
    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size() );

    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& numDetections,
                          double scaleFactor=1.1,
                          int minNeighbors=3, int flags=0,
                          Size minSize=Size(),
                          Size maxSize=Size() );

    void detectMultiScale( InputArray image,
                          CV_OUT std::vector<Rect>& objects,
                          CV_OUT std::vector<int>& rejectLevels,
                          CV_OUT std::vector<double>& levelWeights,
                          double scaleFactor = 1.1,
                          int minNeighbors = 3, int flags = 0,
                          Size minSize = Size(),
                          Size maxSize = Size(),
                          bool outputRejectLevels = false );


    bool isOldFormatCascade() const;
    Size getOriginalWindowSize() const;
    int getFeatureType() const;
    void* getOldCascade();

    void setMaskGenerator(const Ptr<MaskGenerator>& maskGenerator);
    Ptr<MaskGenerator> getMaskGenerator();

protected:
    enum { SUM_ALIGN = 64 };

    bool detectSingleScale( InputArray image, Size processingRectSize,
                            int yStep, double factor, std::vector<Rect>& candidates,
                            std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                            Size sumSize0, bool outputRejectLevels = false );
    bool ocl_detectMultiScaleNoGrouping( const std::vector<float>& scales,
                                         std::vector<Rect>& candidates );

    void detectMultiScaleNoGrouping( InputArray image, std::vector<Rect>& candidates,
                                    std::vector<int>& rejectLevels, std::vector<double>& levelWeights,
                                    double scaleFactor, Size minObjectSize, Size maxObjectSize,
                                    bool outputRejectLevels = false );

    enum { MAX_FACES = 10000 };
    enum { BOOST = 0 };
    enum { DO_CANNY_PRUNING    = CASCADE_DO_CANNY_PRUNING,
        SCALE_IMAGE         = CASCADE_SCALE_IMAGE,
        FIND_BIGGEST_OBJECT = CASCADE_FIND_BIGGEST_OBJECT,
        DO_ROUGH_SEARCH     = CASCADE_DO_ROUGH_SEARCH
    };

    friend class CascadeClassifierInvoker;
    friend class SparseCascadeClassifierInvoker;

    template<class FEval>
    friend int predictOrdered( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategorical( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictOrderedStump( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    template<class FEval>
    friend int predictCategoricalStump( CascadeClassifierImpl& cascade, Ptr<FeatureEvaluator> &featureEvaluator, double& weight);

    int runAt( Ptr<FeatureEvaluator>& feval, Point pt, int scaleIdx, double& weight );

    class Data
    {
    public:
        struct DTreeNode
        {
            int featureIdx;
            float threshold; // for ordered features only
            int left;
            int right;
        };

        struct DTree
        {
            int nodeCount;
        };

        struct Stage
        {
            int first;
            int ntrees;
            float threshold;
        };

        struct Stump
        {
            Stump() { }
            Stump(int _featureIdx, float _threshold, float _left, float _right)
            : featureIdx(_featureIdx), threshold(_threshold), left(_left), right(_right) {}

            int featureIdx;
            float threshold;
            float left;
            float right;
        };

        Data();

        bool read(const FileNode &node);

        int stageType;
        int featureType;
        int ncategories;
        int minNodesPerTree, maxNodesPerTree;
        Size origWinSize;

        std::vector<Stage> stages;
        std::vector<DTree> classifiers;
        std::vector<DTreeNode> nodes;
        std::vector<float> leaves;
        std::vector<int> subsets;
        std::vector<Stump> stumps;
    };

    Data data;
    Ptr<FeatureEvaluator> featureEvaluator;
    Ptr<CvHaarClassifierCascade> oldCascade;

    Ptr<MaskGenerator> maskGenerator;
    UMat ugrayImage;
    UMat ufacepos, ustages, unodes, uleaves, usubsets;

    Mutex mtx;
};

#define CC_CASCADE_PARAMS "cascadeParams"
#define CC_STAGE_TYPE     "stageType"
#define CC_FEATURE_TYPE   "featureType"
#define CC_HEIGHT         "height"
#define CC_WIDTH          "width"

#define CC_STAGE_NUM    "stageNum"
#define CC_STAGES       "stages"
#define CC_STAGE_PARAMS "stageParams"

#define CC_BOOST            "BOOST"
#define CC_MAX_DEPTH        "maxDepth"
#define CC_WEAK_COUNT       "maxWeakCount"
#define CC_STAGE_THRESHOLD  "stageThreshold"
#define CC_WEAK_CLASSIFIERS "weakClassifiers"
#define CC_INTERNAL_NODES   "internalNodes"
#define CC_LEAF_VALUES      "leafValues"

#define CC_FEATURES       "features"
#define CC_FEATURE_PARAMS "featureParams"
#define CC_MAX_CAT_COUNT  "maxCatCount"

#define CC_HAAR   "HAAR"
#define CC_RECTS  "rects"
#define CC_TILTED "tilted"

#define CC_LBP  "LBP"
#define CC_RECT "rect"

#define CC_HOG  "HOG"

#define CV_SUM_PTRS( p0, p1, p2, p3, sum, rect, step )                    \
    /* (x, y) */                                                          \
    (p0) = sum + (rect).x + (step) * (rect).y,                            \
    /* (x + w, y) */                                                      \
    (p1) = sum + (rect).x + (rect).width + (step) * (rect).y,             \
    /* (x + w, y) */                                                      \
    (p2) = sum + (rect).x + (step) * ((rect).y + (rect).height),          \
    /* (x + w, y + h) */                                                  \
    (p3) = sum + (rect).x + (rect).width + (step) * ((rect).y + (rect).height)

#define CV_TILTED_PTRS( p0, p1, p2, p3, tilted, rect, step )                        \
    /* (x, y) */                                                                    \
    (p0) = tilted + (rect).x + (step) * (rect).y,                                   \
    /* (x - h, y + h) */                                                            \
    (p1) = tilted + (rect).x - (rect).height + (step) * ((rect).y + (rect).height), \
    /* (x + w, y + w) */                                                            \
    (p2) = tilted + (rect).x + (rect).width + (step) * ((rect).y + (rect).width),   \
    /* (x + w - h, y + w + h) */                                                    \
    (p3) = tilted + (rect).x + (rect).width - (rect).height                         \