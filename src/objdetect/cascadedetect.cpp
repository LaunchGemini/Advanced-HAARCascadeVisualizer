
/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2008-2013, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Itseez Inc. may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <cstdio>

#include "cascadedetect.hpp"
#include "opencv2/objdetect/objdetect_c.h"

namespace cv
{

template<typename _Tp> void copyVectorToUMat(const std::vector<_Tp>& v, UMat& um)
{
    if(v.empty())
        um.release();
    Mat(1, (int)(v.size()*sizeof(v[0])), CV_8U, (void*)&v[0]).copyTo(um);
}

void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps,
                     std::vector<int>* weights, std::vector<double>* levelWeights)
{
    if( groupThreshold <= 0 || rectList.empty() )
    {
        if( weights )
        {
            size_t i, sz = rectList.size();
            weights->resize(sz);
            for( i = 0; i < sz; i++ )
                (*weights)[i] = 1;
        }
        return;
    }

    std::vector<int> labels;
    int nclasses = partition(rectList, labels, SimilarRects(eps));

    std::vector<Rect> rrects(nclasses);
    std::vector<int> rweights(nclasses, 0);
    std::vector<int> rejectLevels(nclasses, 0);
    std::vector<double> rejectWeights(nclasses, DBL_MIN);
    int i, j, nlabels = (int)labels.size();
    for( i = 0; i < nlabels; i++ )
    {
        int cls = labels[i];
        rrects[cls].x += rectList[i].x;
        rrects[cls].y += rectList[i].y;
        rrects[cls].width += rectList[i].width;
        rrects[cls].height += rectList[i].height;
        rweights[cls]++;
    }

    bool useDefaultWeights = false;

    if ( levelWeights && weights && !weights->empty() && !levelWeights->empty() )
    {
        for( i = 0; i < nlabels; i++ )
        {
            int cls = labels[i];
            if( (*weights)[i] > rejectLevels[cls] )
            {
                rejectLevels[cls] = (*weights)[i];
                rejectWeights[cls] = (*levelWeights)[i];
            }
            else if( ( (*weights)[i] == rejectLevels[cls] ) && ( (*levelWeights)[i] > rejectWeights[cls] ) )
                rejectWeights[cls] = (*levelWeights)[i];
        }
    }
    else
        useDefaultWeights = true;

    for( i = 0; i < nclasses; i++ )
    {
        Rect r = rrects[i];
        float s = 1.f/rweights[i];
        rrects[i] = Rect(saturate_cast<int>(r.x*s),
             saturate_cast<int>(r.y*s),
             saturate_cast<int>(r.width*s),
             saturate_cast<int>(r.height*s));
    }

    rectList.clear();
    if( weights )
        weights->clear();
    if( levelWeights )
        levelWeights->clear();

    for( i = 0; i < nclasses; i++ )
    {
        Rect r1 = rrects[i];
        int n1 = rweights[i];
        double w1 = rejectWeights[i];
        int l1 = rejectLevels[i];

        // filter out rectangles which don't have enough similar rectangles
        if( n1 <= groupThreshold )
            continue;
        // filter out small face rectangles inside large rectangles
        for( j = 0; j < nclasses; j++ )
        {
            int n2 = rweights[j];

            if( j == i || n2 <= groupThreshold )
                continue;
            Rect r2 = rrects[j];

            int dx = saturate_cast<int>( r2.width * eps );
            int dy = saturate_cast<int>( r2.height * eps );

            if( i != j &&
                r1.x >= r2.x - dx &&
                r1.y >= r2.y - dy &&
                r1.x + r1.width <= r2.x + r2.width + dx &&
                r1.y + r1.height <= r2.y + r2.height + dy &&
                (n2 > std::max(3, n1) || n1 < 3) )
                break;
        }

        if( j == nclasses )
        {
            rectList.push_back(r1);
            if( weights )
                weights->push_back(useDefaultWeights ? n1 : l1);
            if( levelWeights )
                levelWeights->push_back(w1);
        }
    }
}

class MeanshiftGrouping
{
public:
    MeanshiftGrouping(const Point3d& densKer, const std::vector<Point3d>& posV,
        const std::vector<double>& wV, double eps, int maxIter = 20)
    {
        densityKernel = densKer;
        weightsV = wV;
        positionsV = posV;
        positionsCount = (int)posV.size();
        meanshiftV.resize(positionsCount);
        distanceV.resize(positionsCount);
        iterMax = maxIter;
        modeEps = eps;

        for (unsigned i = 0; i<positionsV.size(); i++)
        {
            meanshiftV[i] = getNewValue(positionsV[i]);
            distanceV[i] = moveToMode(meanshiftV[i]);
            meanshiftV[i] -= positionsV[i];
        }
    }

    void getModes(std::vector<Point3d>& modesV, std::vector<double>& resWeightsV, const double eps)
    {
        for (size_t i=0; i <distanceV.size(); i++)
        {
            bool is_found = false;
            for(size_t j=0; j<modesV.size(); j++)
            {
                if ( getDistance(distanceV[i], modesV[j]) < eps)
                {
                    is_found=true;
                    break;
                }
            }
            if (!is_found)
            {
                modesV.push_back(distanceV[i]);
            }
        }

        resWeightsV.resize(modesV.size());

        for (size_t i=0; i<modesV.size(); i++)
        {
            resWeightsV[i] = getResultWeight(modesV[i]);
        }
    }

protected:
    std::vector<Point3d> positionsV;
    std::vector<double> weightsV;

    Point3d densityKernel;
    int positionsCount;

    std::vector<Point3d> meanshiftV;
    std::vector<Point3d> distanceV;
    int iterMax;
    double modeEps;

    Point3d getNewValue(const Point3d& inPt) const
    {
        Point3d resPoint(.0);
        Point3d ratPoint(.0);
        for (size_t i=0; i<positionsV.size(); i++)
        {
            Point3d aPt= positionsV[i];
            Point3d bPt = inPt;
            Point3d sPt = densityKernel;

            sPt.x *= std::exp(aPt.z);
            sPt.y *= std::exp(aPt.z);

            aPt.x /= sPt.x;
            aPt.y /= sPt.y;
            aPt.z /= sPt.z;

            bPt.x /= sPt.x;
            bPt.y /= sPt.y;
            bPt.z /= sPt.z;

            double w = (weightsV[i])*std::exp(-((aPt-bPt).dot(aPt-bPt))/2)/std::sqrt(sPt.dot(Point3d(1,1,1)));

            resPoint += w*aPt;

            ratPoint.x += w/sPt.x;
            ratPoint.y += w/sPt.y;
            ratPoint.z += w/sPt.z;
        }
        resPoint.x /= ratPoint.x;
        resPoint.y /= ratPoint.y;
        resPoint.z /= ratPoint.z;
        return resPoint;
    }

    double getResultWeight(const Point3d& inPt) const
    {
        double sumW=0;
        for (size_t i=0; i<positionsV.size(); i++)
        {
            Point3d aPt = positionsV[i];
            Point3d sPt = densityKernel;

            sPt.x *= std::exp(aPt.z);
            sPt.y *= std::exp(aPt.z);

            aPt -= inPt;

            aPt.x /= sPt.x;
            aPt.y /= sPt.y;
            aPt.z /= sPt.z;

            sumW+=(weightsV[i])*std::exp(-(aPt.dot(aPt))/2)/std::sqrt(sPt.dot(Point3d(1,1,1)));
        }
        return sumW;
    }

    Point3d moveToMode(Point3d aPt) const
    {
        Point3d bPt;
        for (int i = 0; i<iterMax; i++)
        {
            bPt = aPt;
            aPt = getNewValue(bPt);
            if ( getDistance(aPt, bPt) <= modeEps )
            {
                break;
            }
        }
        return aPt;
    }

    double getDistance(Point3d p1, Point3d p2) const
    {
        Point3d ns = densityKernel;
        ns.x *= std::exp(p2.z);
        ns.y *= std::exp(p2.z);
        p2 -= p1;
        p2.x /= ns.x;
        p2.y /= ns.y;
        p2.z /= ns.z;
        return p2.dot(p2);
    }
};
//new grouping function with using meanshift
static void groupRectangles_meanshift(std::vector<Rect>& rectList, double detectThreshold, std::vector<double>* foundWeights,
                                      std::vector<double>& scales, Size winDetSize)
{
    int detectionCount = (int)rectList.size();
    std::vector<Point3d> hits(detectionCount), resultHits;
    std::vector<double> hitWeights(detectionCount), resultWeights;
    Point2d hitCenter;

    for (int i=0; i < detectionCount; i++)
    {
        hitWeights[i] = (*foundWeights)[i];
        hitCenter = (rectList[i].tl() + rectList[i].br())*(0.5); //center of rectangles
        hits[i] = Point3d(hitCenter.x, hitCenter.y, std::log(scales[i]));
    }

    rectList.clear();
    if (foundWeights)
        foundWeights->clear();

    double logZ = std::log(1.3);
    Point3d smothing(8, 16, logZ);

    MeanshiftGrouping msGrouping(smothing, hits, hitWeights, 1e-5, 100);

    msGrouping.getModes(resultHits, resultWeights, 1);

    for (unsigned i=0; i < resultHits.size(); ++i)
    {

        double scale = std::exp(resultHits[i].z);
        hitCenter.x = resultHits[i].x;
        hitCenter.y = resultHits[i].y;
        Size s( int(winDetSize.width * scale), int(winDetSize.height * scale) );
        Rect resultRect( int(hitCenter.x-s.width/2), int(hitCenter.y-s.height/2),
            int(s.width), int(s.height) );

        if (resultWeights[i] > detectThreshold)
        {
            rectList.push_back(resultRect);
            foundWeights->push_back(resultWeights[i]);
        }
    }
}

void groupRectangles(std::vector<Rect>& rectList, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, 0, 0);
}

void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& weights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &weights, 0);
}
//used for cascade detection algorithm for ROC-curve calculating
void groupRectangles(std::vector<Rect>& rectList, std::vector<int>& rejectLevels,
                     std::vector<double>& levelWeights, int groupThreshold, double eps)
{
    groupRectangles(rectList, groupThreshold, eps, &rejectLevels, &levelWeights);
}
//can be used for HOG detection algorithm only
void groupRectangles_meanshift(std::vector<Rect>& rectList, std::vector<double>& foundWeights,
                               std::vector<double>& foundScales, double detectThreshold, Size winDetSize)
{
    groupRectangles_meanshift(rectList, detectThreshold, &foundWeights, foundScales, winDetSize);
}


FeatureEvaluator::~FeatureEvaluator() {}

bool FeatureEvaluator::read(const FileNode&, Size _origWinSize)
{
    origWinSize = _origWinSize;
    localSize = lbufSize = Size(0, 0);
    if (scaleData.empty())
        scaleData = makePtr<std::vector<ScaleData> >();
    else
        scaleData->clear();
    return true;
}

Ptr<FeatureEvaluator> FeatureEvaluator::clone() const { return Ptr<FeatureEvaluator>(); }
int FeatureEvaluator::getFeatureType() const {return -1;}
bool FeatureEvaluator::setWindow(Point, int) { return true; }
void FeatureEvaluator::getUMats(std::vector<UMat>& bufs)
{
    if (!(sbufFlag & USBUF_VALID))
    {
        sbuf.copyTo(usbuf);
        sbufFlag |= USBUF_VALID;
    }

    bufs.clear();
    bufs.push_back(uscaleData);
    bufs.push_back(usbuf);
    bufs.push_back(ufbuf);
}

void FeatureEvaluator::getMats()
{
    if (!(sbufFlag & SBUF_VALID))
    {
        usbuf.copyTo(sbuf);
        sbufFlag |= SBUF_VALID;
    }
}

float FeatureEvaluator::calcOrd(int) const { return 0.; }
int FeatureEvaluator::calcCat(int) const { return 0; }

bool FeatureEvaluator::updateScaleData( Size imgsz, const std::vector<float>& _scales )
{
    if( scaleData.empty() )
        scaleData = makePtr<std::vector<ScaleData> >();

    size_t i, nscales = _scales.size();
    bool recalcOptFeatures = nscales != scaleData->size();
    scaleData->resize(nscales);

    int layer_dy = 0;
    Point layer_ofs(0,0);
    Size prevBufSize = sbufSize;
    sbufSize.width = std::max(sbufSize.width, (int)alignSize(cvRound(imgsz.width/_scales[0]) + 31, 32));
    recalcOptFeatures = recalcOptFeatures || sbufSize.width != prevBufSize.width;

    for( i = 0; i < nscales; i++ )
    {
        FeatureEvaluator::ScaleData& s = scaleData->at(i);
        if( !recalcOptFeatures && fabs(s.scale - _scales[i]) > FLT_EPSILON*100*_scales[i] )
            recalcOptFeatures = true;
        float sc = _scales[i];
        Size sz;
        sz.width = cvRound(imgsz.width/sc);
        sz.height = cvRound(imgsz.height/sc);
        s.ystep = sc >= 2 ? 1 : 2;
        s.scale = sc;
        s.szi = Size(sz.width+1, sz.height+1);

        if( i == 0 )
        {
            layer_dy = s.szi.height;
        }

        if( layer_ofs.x + s.szi.width > sbufSize.width )
        {
            layer_ofs = Point(0, layer_ofs.y + layer_dy);
            layer_dy = s.szi.height;
        }
        s.layer_ofs = layer_ofs.y*sbufSize.width + layer_ofs.x;
        layer_ofs.x += s.szi.width;
    }

    layer_ofs.y += layer_dy;
    sbufSize.height = std::max(sbufSize.height, layer_ofs.y);
    recalcOptFeatures = recalcOptFeatures || sbufSize.height != prevBufSize.height;
    return recalcOptFeatures;
}


bool FeatureEvaluator::setImage( InputArray _image, const std::vector<float>& _scales )
{
    Size imgsz = _image.size();
    bool recalcOptFeatures = updateScaleData(imgsz, _scales);

    size_t i, nscales = scaleData->size();
    if (nscales == 0)
    {
        return false;
    }
    Size sz0 = scaleData->at(0).szi;
    sz0 = Size(std::max(rbuf.cols, (int)alignSize(sz0.width, 16)), std::max(rbuf.rows, sz0.height));

    if (recalcOptFeatures)
    {
        computeOptFeatures();
        copyVectorToUMat(*scaleData, uscaleData);
    }

    if (_image.isUMat() && localSize.area() > 0)
    {
        usbuf.create(sbufSize.height*nchannels, sbufSize.width, CV_32S);
        urbuf.create(sz0, CV_8U);

        for (i = 0; i < nscales; i++)
        {
            const ScaleData& s = scaleData->at(i);
            UMat dst(urbuf, Rect(0, 0, s.szi.width - 1, s.szi.height - 1));
            resize(_image, dst, dst.size(), 1. / s.scale, 1. / s.scale, INTER_LINEAR);
            computeChannels((int)i, dst);
        }
        sbufFlag = USBUF_VALID;
    }
    else
    {
        Mat image = _image.getMat();
        sbuf.create(sbufSize.height*nchannels, sbufSize.width, CV_32S);
        rbuf.create(sz0, CV_8U);

        for (i = 0; i < nscales; i++)
        {
            const ScaleData& s = scaleData->at(i);
            Mat dst(s.szi.height - 1, s.szi.width - 1, CV_8U, rbuf.ptr());
            resize(image, dst, dst.size(), 1. / s.scale, 1. / s.scale, INTER_LINEAR);
            computeChannels((int)i, dst);
        }
        sbufFlag = SBUF_VALID;
    }

    return true;
}

//----------------------------------------------  HaarEvaluator ---------------------------------------

bool HaarEvaluator::Feature :: read( const FileNode& node )
{
    FileNode rnode = node[CC_RECTS];
    FileNodeIterator it = rnode.begin(), it_end = rnode.end();

    int ri;
    for( ri = 0; ri < RECT_NUM; ri++ )
    {
        rect[ri].r = Rect();
        rect[ri].weight = 0.f;
    }

    for(ri = 0; it != it_end; ++it, ri++)
    {
        FileNodeIterator it2 = (*it).begin();
        it2 >> rect[ri].r.x >> rect[ri].r.y >>
            rect[ri].r.width >> rect[ri].r.height >> rect[ri].weight;
    }

    tilted = (int)node[CC_TILTED] != 0;
    return true;
}

HaarEvaluator::HaarEvaluator()
{
    optfeaturesPtr = 0;
    pwin = 0;
    localSize = Size(4, 2);
    lbufSize = Size(0, 0);
    nchannels = 0;
    tofs = 0;
}

HaarEvaluator::~HaarEvaluator()
{
}

bool HaarEvaluator::read(const FileNode& node, Size _origWinSize)
{
    if (!FeatureEvaluator::read(node, _origWinSize))
        return false;
    size_t i, n = node.size();
    CV_Assert(n > 0);
    if(features.empty())
        features = makePtr<std::vector<Feature> >();
    if(optfeatures.empty())
        optfeatures = makePtr<std::vector<OptFeature> >();
    if (optfeatures_lbuf.empty())
        optfeatures_lbuf = makePtr<std::vector<OptFeature> >();
    features->resize(n);
    FileNodeIterator it = node.begin();
    hasTiltedFeatures = false;
    std::vector<Feature>& ff = *features;
    sbufSize = Size();
    ufbuf.release();

    for(i = 0; i < n; i++, ++it)
    {
        if(!ff[i].read(*it))
            return false;
        if( ff[i].tilted )
            hasTiltedFeatures = true;
    }
    nchannels = hasTiltedFeatures ? 3 : 2;
    normrect = Rect(1, 1, origWinSize.width - 2, origWinSize.height - 2);

    localSize = lbufSize = Size(0, 0);

    return true;
}

Ptr<FeatureEvaluator> HaarEvaluator::clone() const
{
    Ptr<HaarEvaluator> ret = makePtr<HaarEvaluator>();
    *ret = *this;
    return ret;
}


void HaarEvaluator::computeChannels(int scaleIdx, InputArray img)
{
    const ScaleData& s = scaleData->at(scaleIdx);
    sqofs = hasTiltedFeatures ? sbufSize.area() * 2 : sbufSize.area();

    if (img.isUMat())
    {
        int sx = s.layer_ofs % sbufSize.width;
        int sy = s.layer_ofs / sbufSize.width;
        int sqy = sy + (sqofs / sbufSize.width);
        UMat sum(usbuf, Rect(sx, sy, s.szi.width, s.szi.height));
        UMat sqsum(usbuf, Rect(sx, sqy, s.szi.width, s.szi.height));
        sqsum.flags = (sqsum.flags & ~UMat::DEPTH_MASK) | CV_32S;

        if (hasTiltedFeatures)
        {
            int sty = sy + (tofs / sbufSize.width);
            UMat tilted(usbuf, Rect(sx, sty, s.szi.width, s.szi.height));
            integral(img, sum, sqsum, tilted, CV_32S, CV_32S);
        }
        else
        {
            UMatData* u = sqsum.u;
            integral(img, sum, sqsum, noArray(), CV_32S, CV_32S);
            CV_Assert(sqsum.u == u && sqsum.size() == s.szi && sqsum.type()==CV_32S);
        }
    }
    else
    {
        Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);
        Mat sqsum(s.szi, CV_32S, sum.ptr<int>() + sqofs, sbuf.step);

        if (hasTiltedFeatures)
        {
            Mat tilted(s.szi, CV_32S, sum.ptr<int>() + tofs, sbuf.step);
            integral(img, sum, sqsum, tilted, CV_32S, CV_32S);
        }
        else
            integral(img, sum, sqsum, noArray(), CV_32S, CV_32S);
    }
}

void HaarEvaluator::computeOptFeatures()
{
    if (hasTiltedFeatures)
        tofs = sbufSize.area();

    int sstep = sbufSize.width;
    CV_SUM_OFS( nofs[0], nofs[1], nofs[2], nofs[3], 0, normrect, sstep );

    size_t fi, nfeatures = features->size();
    const std::vector<Feature>& ff = *features;
    optfeatures->resize(nfeatures);
    optfeaturesPtr = &(*optfeatures)[0];
    for( fi = 0; fi < nfeatures; fi++ )
        optfeaturesPtr[fi].setOffsets( ff[fi], sstep, tofs );
    optfeatures_lbuf->resize(nfeatures);

    for( fi = 0; fi < nfeatures; fi++ )
        optfeatures_lbuf->at(fi).setOffsets(ff[fi], lbufSize.width > 0 ? lbufSize.width : sstep, tofs);

    copyVectorToUMat(*optfeatures_lbuf, ufbuf);
}

bool HaarEvaluator::setWindow( Point pt, int scaleIdx )
{
    const ScaleData& s = getScaleData(scaleIdx);

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= s.szi.width ||
        pt.y + origWinSize.height >= s.szi.height )
        return false;

    pwin = &sbuf.at<int>(pt) + s.layer_ofs;
    const int* pq = (const int*)(pwin + sqofs);
    int valsum = CALC_SUM_OFS(nofs, pwin);
    unsigned valsqsum = (unsigned)(CALC_SUM_OFS(nofs, pq));

    double area = normrect.area();
    double nf = area * valsqsum - (double)valsum * valsum;
    if( nf > 0. )
    {
        nf = std::sqrt(nf);
        varianceNormFactor = (float)(1./nf);
        return area*varianceNormFactor < 1e-1;
    }
    else
    {
        varianceNormFactor = 1.f;
        return false;
    }
}


void HaarEvaluator::OptFeature::setOffsets( const Feature& _f, int step, int _tofs )
{
    weight[0] = _f.rect[0].weight;
    weight[1] = _f.rect[1].weight;
    weight[2] = _f.rect[2].weight;

    if( _f.tilted )
    {
        CV_TILTED_OFS( ofs[0][0], ofs[0][1], ofs[0][2], ofs[0][3], _tofs, _f.rect[0].r, step );
        CV_TILTED_OFS( ofs[1][0], ofs[1][1], ofs[1][2], ofs[1][3], _tofs, _f.rect[1].r, step );
        CV_TILTED_OFS( ofs[2][0], ofs[2][1], ofs[2][2], ofs[2][3], _tofs, _f.rect[2].r, step );
    }
    else
    {
        CV_SUM_OFS( ofs[0][0], ofs[0][1], ofs[0][2], ofs[0][3], 0, _f.rect[0].r, step );
        CV_SUM_OFS( ofs[1][0], ofs[1][1], ofs[1][2], ofs[1][3], 0, _f.rect[1].r, step );
        CV_SUM_OFS( ofs[2][0], ofs[2][1], ofs[2][2], ofs[2][3], 0, _f.rect[2].r, step );
    }
}

Rect HaarEvaluator::getNormRect() const
{
    return normrect;
}

int HaarEvaluator::getSquaresOffset() const
{
    return sqofs;
}

//----------------------------------------------  LBPEvaluator -------------------------------------
bool LBPEvaluator::Feature :: read(const FileNode& node )
{
    FileNode rnode = node[CC_RECT];
    FileNodeIterator it = rnode.begin();
    it >> rect.x >> rect.y >> rect.width >> rect.height;
    return true;
}

LBPEvaluator::LBPEvaluator()
{
    features = makePtr<std::vector<Feature> >();
    optfeatures = makePtr<std::vector<OptFeature> >();
    scaleData = makePtr<std::vector<ScaleData> >();
}

LBPEvaluator::~LBPEvaluator()
{
}

bool LBPEvaluator::read( const FileNode& node, Size _origWinSize )
{
    if (!FeatureEvaluator::read(node, _origWinSize))
        return false;
    if(features.empty())
        features = makePtr<std::vector<Feature> >();
    if(optfeatures.empty())
        optfeatures = makePtr<std::vector<OptFeature> >();
    if (optfeatures_lbuf.empty())
        optfeatures_lbuf = makePtr<std::vector<OptFeature> >();

    features->resize(node.size());
    optfeaturesPtr = 0;
    FileNodeIterator it = node.begin(), it_end = node.end();
    std::vector<Feature>& ff = *features;
    for(int i = 0; it != it_end; ++it, i++)
    {
        if(!ff[i].read(*it))
            return false;
    }
    nchannels = 1;
    localSize = lbufSize = Size(0, 0);

    return true;
}

Ptr<FeatureEvaluator> LBPEvaluator::clone() const
{
    Ptr<LBPEvaluator> ret = makePtr<LBPEvaluator>();
    *ret = *this;
    return ret;
}

void LBPEvaluator::computeChannels(int scaleIdx, InputArray _img)
{
    const ScaleData& s = scaleData->at(scaleIdx);

    if (_img.isUMat())
    {
        int sx = s.layer_ofs % sbufSize.width;
        int sy = s.layer_ofs / sbufSize.width;
        UMat sum(usbuf, Rect(sx, sy, s.szi.width, s.szi.height));
        integral(_img, sum, noArray(), noArray(), CV_32S);
    }
    else
    {
        Mat sum(s.szi, CV_32S, sbuf.ptr<int>() + s.layer_ofs, sbuf.step);
        integral(_img, sum, noArray(), noArray(), CV_32S);
    }
}

void LBPEvaluator::computeOptFeatures()
{
    int sstep = sbufSize.width;

    size_t fi, nfeatures = features->size();
    const std::vector<Feature>& ff = *features;
    optfeatures->resize(nfeatures);
    optfeaturesPtr = &(*optfeatures)[0];
    for( fi = 0; fi < nfeatures; fi++ )
        optfeaturesPtr[fi].setOffsets( ff[fi], sstep );
    copyVectorToUMat(*optfeatures, ufbuf);
}


void LBPEvaluator::OptFeature::setOffsets( const Feature& _f, int step )
{
    Rect tr = _f.rect;
    int w0 = tr.width;
    int h0 = tr.height;

    CV_SUM_OFS( ofs[0], ofs[1], ofs[4], ofs[5], 0, tr, step );
    tr.x += 2*w0;
    CV_SUM_OFS( ofs[2], ofs[3], ofs[6], ofs[7], 0, tr, step );
    tr.y += 2*h0;
    CV_SUM_OFS( ofs[10], ofs[11], ofs[14], ofs[15], 0, tr, step );
    tr.x -= 2*w0;
    CV_SUM_OFS( ofs[8], ofs[9], ofs[12], ofs[13], 0, tr, step );
}


bool LBPEvaluator::setWindow( Point pt, int scaleIdx )
{
    CV_Assert(0 <= scaleIdx && scaleIdx < (int)scaleData->size());
    const ScaleData& s = scaleData->at(scaleIdx);

    if( pt.x < 0 || pt.y < 0 ||
        pt.x + origWinSize.width >= s.szi.width ||
        pt.y + origWinSize.height >= s.szi.height )
        return false;

    pwin = &sbuf.at<int>(pt) + s.layer_ofs;
    return true;
}


Ptr<FeatureEvaluator> FeatureEvaluator::create( int featureType )
{