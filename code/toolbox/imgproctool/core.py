#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'hkh, lih'
__version__ = '3.2'
__date__ = '26/07/2017'


import cv2
import math
import copy
import numpy as np
import scipy

__all__ = [
    'ROI_TYPE_XYWH',
    'ROI_TYPE_XYXY',
    'ROI_TYPE_ROTATED',
    'ROI_CVT_XYXY2XYWH',
    'ROI_CVT_XYWH2XYXY',
    'IPTError',
    'cvtRoi',
    'getRoiImg',
    'drawRoi',
    'inRoi',
    'gammaTransform',
    'contrastStretch',
    'findMaxAreaContours',
    'calcCentroid',
    'findMaxBoundBox',
    'calcGrayGravity',
    'getHistImg',
    'drawPoints',
    'drawLine',
    'drawContours',
    'sortDisBWPt2Pts',
    'rotateImg',
    'thinImage',
    'ULBP',
    'deskewImage'
]


ROI_TYPE_XYWH = 0L
ROI_TYPE_XYXY = 8L
ROI_TYPE_ROTATED = 64L

ROI_CVT_XYXY2XYWH = 0L
ROI_CVT_XYWH2XYXY = 8L


# Error object
class IPTError(Exception):
    """
    Generic Python-exception-derived object raised by Image Process Tool functions.

    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in Image Process Tool functions when a Image
    Processing-related condition would prevent further correct execution of the
    function.

    Parameters
    ----------
    None

    Examples
    --------
    >>> import imgproctool as IPT
    >>> imgprocesstool.getRoiImg(None, roi=[1, 2, 3, 4], roiType=imgprocesstool.ROI_TYPE_XYWH)

    """
    pass


def __cvtRoi2xyxy(roi, roiType):
    if ROI_TYPE_XYWH == roiType:
        return cvtRoi(roi=roi, flag=ROI_CVT_XYWH2XYXY)
    elif ROI_TYPE_XYXY == roiType:
        return copy.copy(roi)
    else:
        raise ValueError, 'flag is wrong!!!'


def cvtRoi(roi, flag):
    """
    convert roi type

    :param roi: list or ndarray
    :param flag: ROI_CVT_XYXY2XYWH or ROI_CVT_XYWH2XYXY
    :return: roi (xyxy or xywh,depends on what you set)
    """
    x0, y0, c, d = roi
    if ROI_CVT_XYWH2XYXY == flag:
        x1 = x0 + c
        y1 = y0 + d
        return [x0, y0, x1, y1]
    elif ROI_CVT_XYXY2XYWH == flag:
        w = c - x0
        h = d - y0
        return [x0, y0, w, h]
    else:
        raise ValueError, 'flag is wrong!!!'


def getRoiImg(img, roi, roiType, copy=True):
    """

    :param img: gray image or BGR image
    :param roi: list or ndarray
    :param roiType: flag - ROI_TYPE_XYWH or ROI_TYPE_XYXY
    :return: Roi image
    """
    if img is None:
        raise IPTError, 'img is None'
    Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
    if Roi_xyxy[0] < 0:
        Roi_xyxy[0] = 0
    if Roi_xyxy[1] < 0:
        Roi_xyxy[1] = 0
    if Roi_xyxy[2] < 0 or Roi_xyxy[3] < 0:
        raise IPTError, 'roi data invalid'
    if 3 == img.ndim:
        if copy:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2], :].copy()
        else:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2], :]
    elif 2 == img.ndim:
        if copy:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2]].copy()
        else:
            RoiImg = img[Roi_xyxy[1]:Roi_xyxy[3], Roi_xyxy[0]:Roi_xyxy[2]]
    else:
        raise IPTError, 'img data error'
    Offset_2x1 = np.array(Roi_xyxy[:2]).reshape(2, 1)
    return Offset_2x1, RoiImg


def drawRoi(img, roi, roiType, color, thickness=2, lineType=1, shift=0, offset=(0,0)):
    """
    draw roi(rectangle) in img

    :param img: gray image or BGR image
    :param roi: list or ndarray
    :param roiType: flag - ROI_TYPE_XYWH or ROI_TYPE_XYXY
    :param color: plot color you want
    :param thickness: roi(rectangle)'s thickness
    :return: None
    """
    if img is None:
        raise IPTError, 'img is None'
    Offset = np.array(offset).ravel()
    if roiType == ROI_TYPE_ROTATED:
        Points_4x2 = np.array(roi)
        if Points_4x2.shape[1] == 4:
            Points_4x2 = Points_4x2.T.reshape(4, 2)
        Contour_4x1x2 = Points_4x2.reshape(4, 1, 2).astype('int')
        cv2.drawContours(img, [Contour_4x1x2], 0, color, thickness, lineType, offset=offset)
    else:
        Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
        Roi_xyxy[0] += Offset[0]
        Roi_xyxy[1] += Offset[1]
        Roi_xyxy[2] += Offset[0]
        Roi_xyxy[3] += Offset[1]
        cv2.rectangle(img, (int(Roi_xyxy[0]), int(Roi_xyxy[1])), (int(Roi_xyxy[2]), int(Roi_xyxy[3])), color,
                      thickness=thickness, lineType=lineType, shift=shift)


def inRoi(pt, roi, roiType):
    """
    Matrix or vector norm.

    This function is able to return one of eight different matrix norms,
    or one of an infinite number of vector norms (described below), depending
    on the value of the ``ord`` parameter.

    Parameters
    ----------
    x : array_like
        Input array.  If `axis` is None, `x` must be 1-D or 2-D.
    ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
        Order of the norm (see table under ``Notes``). inf means numpy's
        `inf` object.
    axis : {int, 2-tuple of ints, None}, optional
        If `axis` is an integer, it specifies the axis of `x` along which to
        compute the vector norms.  If `axis` is a 2-tuple, it specifies the
        axes that hold 2-D matrices, and the matrix norms of these matrices
        are computed.  If `axis` is None then either a vector norm (when `x`
        is 1-D) or a matrix norm (when `x` is 2-D) is returned.
    keepdims : bool, optional
        If this is set to True, the axes which are normed over are left in the
        result as dimensions with size one.  With this option the result will
        broadcast correctly against the original `x`.

        .. versionadded:: 1.10.0

    Returns
    -------
    n : float or ndarray
        Norm of the matrix or vector(s).

    Notes
    -----
    For values of ``ord <= 0``, the result is, strictly speaking, not a
    mathematical 'norm', but it may still be useful for various numerical
    purposes.

    The following norms can be calculated:

    =====  ============================  ==========================
    ord    norm for matrices             norm for vectors
    =====  ============================  ==========================
    None   Frobenius norm                2-norm
    'fro'  Frobenius norm                --
    'nuc'  nuclear norm                  --
    inf    max(sum(abs(x), axis=1))      max(abs(x))
    -inf   min(sum(abs(x), axis=1))      min(abs(x))
    0      --                            sum(x != 0)
    1      max(sum(abs(x), axis=0))      as below
    -1     min(sum(abs(x), axis=0))      as below
    2      2-norm (largest sing. value)  as below
    -2     smallest singular value       as below
    other  --                            sum(abs(x)**ord)**(1./ord)
    =====  ============================  ==========================

    The Frobenius norm is given by [1]_:

        :math:`||A||_F = [\\sum_{i,j} abs(a_{i,j})^2]^{1/2}`

    The nuclear norm is the sum of the singular values.

    References
    ----------
    .. [1] G. H. Golub and C. F. Van Loan, *Matrix Computations*,
           Baltimore, MD, Johns Hopkins University Press, 1985, pg. 15

    """
    Point = np.array(pt).ravel()
    if roiType == ROI_TYPE_ROTATED:
        Points_4x2 = np.array(roi)
        if Points_4x2.shape[1] == 4:
            Points_4x2 = Points_4x2.T.reshape(4, 2)
        Contour_4x1x2 = Points_4x2.reshape(4, 1, 2)
        testResult = cv2.pointPolygonTest(Contour_4x1x2, tuple(Point), False)
        if testResult == -1:
            return False
        else:
            return True
    else:
        Roi_xyxy = __cvtRoi2xyxy(roi, roiType)
        if Roi_xyxy[0] <= Point[0] < Roi_xyxy[2] and Roi_xyxy[1] <= Point[1] < Roi_xyxy[3]:
            return True
    return False


def __gammaTransform(src, gamma):
    LUT = []
    C = 255.0 / (255 ** gamma)
    for i in xrange(256):
        LUT.append(C * (i**gamma))
    return cv2.LUT(src, np.array(LUT, dtype=np.uint8))


def gammaTransform(src, gamma):
    if src.ndim == 2:
        return __gammaTransform(src, gamma)
    elif src.ndim == 3:
        HSVImg = cv2.cvtColor(src=src, code=cv2.COLOR_BGR2HSV)
        H = HSVImg[:, :, 0]
        S = HSVImg[:, :, 1]
        V = HSVImg[:, :, 2]
        V = __gammaTransform(V, gamma)
        NewHSVImg = cv2.merge((H, S, V))
        return cv2.cvtColor(src=NewHSVImg, code=cv2.COLOR_HSV2BGR)
    else:
        raise IPTError('ndim must be 2 or 3')


def contrastStretch(gray, min=0):
    Hist = cv2.calcHist([gray], [0], None, [256], [0.0, 256.0])
    IdxMin = 0
    for i in xrange(Hist.shape[0]):
        if Hist[i] > min:
            IdxMin = i
            break
    IdxMax = 0
    for i in xrange(Hist.shape[0]):
        if Hist[255-i] > min:
            IdxMax = 255 - i
            break
    _, gray = cv2.threshold(gray, IdxMax, IdxMax, cv2.THRESH_TRUNC)
    gray = ((gray >= IdxMin) * gray) + ((gray < IdxMin) * IdxMin)
    Res = np.uint8(255.0 * (gray - IdxMin) / (IdxMax - IdxMin))
    return Res


def findMaxAreaContours(contours, num=1):
    ContoursNum = len(contours)
    assert (0 != ContoursNum), 'contours num is 0'

    MaxAreaContoursIndex = []
    Times = 0
    while Times < num:
        for i in xrange(ContoursNum - 1 - Times):
            if cv2.contourArea(contour=contours[i]) > cv2.contourArea(contour=contours[i+1]):
                Temp = contours[i]
                contours[i] = contours[i+1]
                contours[i+1] = Temp
        MaxAreaContoursIndex.append(ContoursNum - 1 - Times)
        Times += 1
    return MaxAreaContoursIndex


def calcCentroid(array, binaryImage=False):
    Moments = cv2.moments(array=array, binaryImage=binaryImage)
    try:
        MarkPt_2x1 = np.array([[Moments['m10'] / Moments['m00']],
                               [Moments['m01'] / Moments['m00']]])
        return True, MarkPt_2x1
    except ZeroDivisionError:
        return False, None


def findMaxBoundBox(contours, num=1):
    ContoursNum = len(contours)
    assert (0 != ContoursNum)

    MaxIndex = []
    Times = 0
    while Times < num:
        for i in xrange(ContoursNum - 1 - Times):
            _, _, w1, h1 = cv2.boundingRect(contours[i])
            _, _, w2, h2 = cv2.boundingRect(contours[i+1])
            if w1*h1 > w2*h2:
                Temp = contours[i]
                contours[i] = contours[i+1]
                contours[i+1] = Temp
        MaxIndex.append(ContoursNum - 1 - Times)
        Times += 1
    return MaxIndex


def splitImageWithHSV(src):
    HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H = HSV[:,:,0]
    B = np.uint8(((H >=90) & (H < 150)) * 255)
    G = np.uint8(((H >=30) & (H < 90)) * 255)
    R = np.uint8(((H >= 150) | (H < 30)) * 255)
    img = cv2.merge([B, G, R])
    return img


def splitImgColorWithHSV(src, chanel):
    assert chanel in 'bgr', 'must input one of b,g,r'

    HSV = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    H = HSV[:,:,0]
    if 'b' == chanel:
        return np.uint8(((H >=90) & (H < 150)) * 255)
    elif 'g' == chanel:
        return np.uint8(((H >=30) & (H < 90)) * 255)
    elif 'r' == chanel:
        return np.uint8(((H >= 150) | (H < 30)) * 255)


def calcGrayGravity(gray):
    # calculate gravity of the gray image.
    assert gray.ndim == 2, "must input a gary_img"

    Row, Col = gray.shape
    GraySum = np.sum(gray)
    if GraySum != 0:
        # np.sum(img, 0)
        # np.sum(img, 1)
        SumX = np.sum(gray, 0) * (np.array(range(Col)))
        SumY = np.sum(gray, 1) * (np.array(range(Row)))
        GravityX = np.sum(SumX) / GraySum
        GravityY = np.sum(SumY) / GraySum
    else:
        GravityX, GravityY = 0.0, 0.0
    return GravityX, GravityY


def getHistImg(imgOrHist, channels=0, mask=None, histSize=256, ranges=[0.0, 256.0]):
    """
    :param imgOrHist:
    :return:
    """
    if 1 != imgOrHist.shape[1]:
        assert 2 == imgOrHist.ndim
        Hist = cv2.calcHist(images=[imgOrHist], channels=[channels], mask=mask, histSize=[histSize], ranges=ranges)
    else:
        Hist = imgOrHist
    MaxVal = np.max(Hist)
    # init show hist image
    HistImg = np.zeros((Hist.shape[0], Hist.shape[0]))
    Hpt = Hist.shape[0] * 0.9 / MaxVal
    for i, value in enumerate(Hist):
        draw_point = (i, histSize - value * Hpt)
        cv2.line(HistImg, draw_point, (i, 0), 255, 1)
    return HistImg


def drawPoints(img, pts_2xn, color, radius=1, thickness=-1, offset=(0,0), shift=0):
    """
    draw points(circles) in img
    :param img: gray image or BGR image
    :param pts_2xn: 2xn ndarray
    :param color: plot color you want
    :param radius: points(circles)'s radius
    :param thickness: points(circles)'s thickness
    :return: None
    """
    Offset = np.array(offset).ravel()
    for idx in range(pts_2xn.shape[1]):
        cv2.circle(img, (int(pts_2xn[0, idx]+Offset[0]), int(pts_2xn[1, idx]+Offset[1])), radius, color, thickness, shift=0)


def drawLine(img, point1, point2, color, thickness=2, shift=0):
    """
    draw line in img
    :param img: gray image or BGR image
    :param point1: line's first point - list, tuple or ndarray
    :param point2: line's second point - list, tuple or ndarray
    :param color: line's color you want
    :param thickness: line's thickness
    :return: None
    """
    Point1 = np.array(point1).ravel()
    Point2 = np.array(point2).ravel()
    cv2.line(img=img, pt1=(int(Point1[0]), int(Point1[1])),
             pt2=(int(Point2[0]), int(Point2[1])), color=color, thickness=thickness, shift=shift)


def drawContours(srcImg, contours, contourIdx, color, thickness=1, lineType=8, hierarchy=None, maxLevel=1<<31-1, offset=(0,0)):
    if hierarchy is not None:
        cv2.drawContours(image=srcImg, contours=contours, contourIdx=contourIdx,
                         color=color, thickness=thickness, lineType=lineType,
                         hierarchy=hierarchy, maxLevel=maxLevel, offset=offset)
    else:
        cv2.drawContours(image=srcImg, contours=contours, contourIdx=contourIdx,
                         color=color, thickness=thickness, lineType=lineType, offset=offset)


def sortDisBWPt2Pts(point_2x1, points_2xn, ascending=True):
    """
    sort point to points by distance
    :param point_2x1: a point - ndarray
    :param points_2xn: points
    :param ascending: True or False
    :return: index sorted by calculating the distance between every point(in points_2xn) to point_2x1
    """
    assert isinstance(points_2xn, np.ndarray),  'points must be ndarray'
    assert isinstance(point_2x1, np.ndarray),   'point must be ndarray'
    assert point_2x1.shape == (2,1),   'point must be 2-by-1'
    assert points_2xn.ndim == 2,          'points must be 2*N'
    assert points_2xn.shape[0] == 2,            'points must be 2*N'

    Dis_1xn = np.linalg.norm(point_2x1 - points_2xn, axis=0)
    sortIdx = Dis_1xn.argsort()
    if not ascending:
        sortIdx[:] = sortIdx[::-1]
    return sortIdx


def rotateImg(srcImg, angle_deg):
    """
    :param numpy.ndarray src: the sra image
    :param float angle_deg:
    :param float scale:
    :return: ndarray, rotated image
    """
    w1 = math.fabs(srcImg.shape[1] * math.cos(np.deg2rad(angle_deg)))
    w2 = math.fabs(srcImg.shape[0] * math.sin(np.deg2rad(angle_deg)))
    h1 = math.fabs(srcImg.shape[1] * math.sin(np.deg2rad(angle_deg)))
    h2 = math.fabs(srcImg.shape[0] * math.cos(np.deg2rad(angle_deg)))
    width = int(w1 + w2) + 1
    height = int(h1 + h2) + 1
    dstSize = (width, height)
    x = srcImg.shape[1]
    y = srcImg.shape[0]
    center = np.array([x/2, y/2]).reshape(2, 1)
    rotateMatrix = cv2.getRotationMatrix2D(center=(0, 0),
                                           angle=angle_deg,
                                           scale=1.0)
    rotateCenter = np.dot(rotateMatrix[0:2, 0:2].reshape(2, 2), center)
    rotateMatrix[0, 2] = width / 2 - rotateCenter[0]
    rotateMatrix[1, 2] = height / 2 - rotateCenter[1]
    if 0 == angle_deg % 90:
        angle_deg = angle_deg / 90 % 4
        rotatedImg = np.ascontiguousarray(np.rot90(srcImg, angle_deg))
    else:
        rotatedImg = cv2.warpAffine(src=srcImg,
                                    M=rotateMatrix,
                                    dsize=dstSize)
    TranFormMatrix = np.vstack((rotateMatrix, np.array([0.0, 0.0, 1.0])))
    return rotatedImg, TranFormMatrix


def thinImage(img_bin, maxIteration=-1):
    imgthin = np.copy(img_bin)
    imgthin2 = np.copy(img_bin)
    count = 0
    rows = imgthin.shape[0]
    cols = imgthin.shape[1]
    while True:
        count += 1
        if maxIteration != -1 and count > maxIteration:
            break

        flag = 0
        for i in range(rows):
            for j in range(cols):
                # p9 p2 p3
                # p8 p1 p4
                # p7 p6 p5
                p1 = imgthin[i, j]
                p2 = 0 if i == 0 else imgthin[i-1, j]
                p3 = 0 if i == 0 or j == cols-1 else imgthin[i-1, j+1]
                p4 = 0 if j == cols-1 else imgthin[i, j+1]
                p5 = 0 if i == rows-1 or j == cols-1 else imgthin[i+1, j+1]
                p6 = 0 if i == rows-1 else imgthin[i+1, j]
                p7 = 0 if i == rows-1 or j == 0 else imgthin[i+1, j-1]
                p8 = 0 if j == 0 else imgthin[i, j-1]
                p9 = 0 if i == 0 or j == 0 else imgthin[i-1, j-1]

                if (p2+p3+p4+p5+p6+p7+p8+p9) >= 2 and (p2+p3+p4+p5+p6+p7+p8+p9) <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    if ap == 1:
                        if p2*p4*p6 == 0:
                            if p4*p6*p8 == 0:
                                imgthin2[i, j] = 0
                                flag = 1
        if flag == 0:
            break
        imgthin = np.copy(imgthin2)

        flag = 0
        for i in range(rows):
            for j in range(cols):
                # p9 p2 p3
                # p8 p1 p4
                # p7 p6 p5
                p1 = imgthin[i, j]
                if p1 != 1:
                    continue
                p2 = 0 if i == 0 else imgthin[i-1, j]
                p3 = 0 if i == 0 or j == cols-1 else imgthin[i-1, j+1]
                p4 = 0 if j == cols-1 else imgthin[i, j+1]
                p5 = 0 if i == rows-1 or j == cols-1 else imgthin[i+1, j+1]
                p6 = 0 if i == rows-1 else imgthin[i+1, j]
                p7 = 0 if i == rows-1 or j == 0 else imgthin[i+1, j-1]
                p8 = 0 if j == 0 else imgthin[i, j-1]
                p9 = 0 if i == 0 or j == 0 else imgthin[i-1, j-1]

                if (p2+p3+p4+p5+p6+p7+p8+p9) >= 2 and (p2+p3+p4+p5+p6+p7+p8+p9) <= 6:
                    ap = 0
                    if p2 == 0 and p3 == 1:
                        ap += 1
                    if p3 == 0 and p4 == 1:
                        ap += 1
                    if p4 == 0 and p5 == 1:
                        ap += 1
                    if p5 == 0 and p6 == 1:
                        ap += 1
                    if p6 == 0 and p7 == 1:
                        ap += 1
                    if p7 == 0 and p8 == 1:
                        ap += 1
                    if p8 == 0 and p9 == 1:
                        ap += 1
                    if p9 == 0 and p2 == 1:
                        ap += 1
                    if ap == 1:
                        if p2*p4*p8 == 0:
                            if p2*p6*p8 == 0:
                                imgthin2[i, j] = 0
                                flag = 1
        if flag == 0:
            break
        imgthin = np.copy(imgthin2)
    for i in range(rows):
        for j in range(cols):
            if imgthin2[i, j] == 1:
                imgthin2[i, j] = 255
    return imgthin2


def enhanceWithLaplacion(gray_img):
    assert 2 == gray_img.ndim

    lap_img = cv2.Laplacian(gray_img, cv2.CV_8UC1)
    enhance_img = cv2.subtract(gray_img, lap_img)
    return enhance_img


def enhanceWithLaplacion2(SrcImg):
    if 2 == SrcImg.ndim:
        type = cv2.CV_8UC1
    elif 3 == SrcImg.ndim:
        type = cv2.CV_8UC3
    else:
        raise ValueError

    kernel = np.array([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])
    return cv2.filter2D(src=SrcImg, ddepth=type, kernel=kernel)


def ULBP(src):
    assert 2 == src.ndim

    r, c = src.shape
    LBP = np.zeros(shape=src.shape, dtype=np.uint8)
    LbpHist = np.zeros(shape=(256, 1), dtype=np.float32)
    Kernel = np.array([[1,   2,  4],
                       [128, 0,  8],
                       [64, 32, 16]], dtype=np.uint8)

    for i in xrange(1, r-1):
        for j in xrange(1, c-1):
            Mask = np.zeros(shape=(3, 3), dtype=np.uint8)
            for m in xrange(-1, 2):
                for n in xrange(-1, 2):
                    Mask[m][n] = 1 if src[i+m][j+n] >= src[i][j] else 0
            LbpValue = int(np.sum(Mask * Kernel))

            if 255 == LbpValue and 0 == src[i][j]:
            # if 255 == LbpValue:
                continue
            # LBP[i][j] = LbpValue
            # ValueLeft = ((LbpValue << 1)&0xff) + (LbpValue & 0x01)
            # Temp = LbpValue ^ ValueLeft
            # JumpCount = 0
            # while Temp:
            #     Temp &= Temp - 1
            #     JumpCount += 1
            # # print '%x'%(Value), count
            # if JumpCount <= 2:
            #     LbpHist[LbpValue] += 1
            LbpHist[LbpValue] += 1
    return LBP, LbpHist


def deskewImage(img):
    m = cv2.moments(img)
    a, b = 1, 2
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, 0], [0, 1, 0]])
    dsize = img.shape[1], img.shape[0]
    img = cv2.warpAffine(src=img, M=M, dsize=dsize,flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img, M
