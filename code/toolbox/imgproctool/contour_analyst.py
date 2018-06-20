#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '3.0'
__date__ = '10/08/2016'

import cv2
import cv2.cv as cv
import numpy as np


def getConvexityDefects(contour):
    Hull = cv2.convexHull(contour, returnPoints=False)
    Defects = cv2.convexityDefects(contour, Hull)
    return Defects


def isPointInContour(contour, point):
    """
    it finds whether the point is inside or outside or on the contour
    (it returns +1, -1, 0 respectively).
    """
    Point = np.array(point).reshape(-1).tolist()
    return cv2.pointPolygonTest(contour, Point, False)


def getDistance(contour, point):
    Point = np.array(point).reshape(-1).tolist()
    return cv2.pointPolygonTest(contour, Point, True)


def matchContour(contour1, contour2, method=1):
    """
    method:   1 - use I1 formula
              2 - use I2 formula
              3 - use I3 formula
    """
    # opencv
    # return cv2.matchShapes(contour1, contour2, method, 0.0)
    HuMoment1 = cv2.HuMoments(m=cv2.moments(contour1.astype(np.float32)))
    HuMoment2 = cv2.HuMoments(m=cv2.moments(contour2.astype(np.float32)))
    HuMoment1[abs(HuMoment1) < 1e-9] = 0
    m1 = np.sign(HuMoment1) * np.log(np.abs(HuMoment1))
    m2 = np.sign(HuMoment2) * np.log(np.abs(HuMoment2))
    Valid1 = np.logical_not(np.isnan(m1))
    Valid2 = np.logical_not(np.isnan(m2))
    Valid = np.logical_and(Valid1, Valid2)
    # Valid = np.array([True]*7).reshape(HuMoment1.shape)
    if method == 1:
        DisSimilarity = np.sum(np.abs(1 / m1[Valid] - 1 / m2[Valid]))
    elif method == 2:
        DisSimilarity = np.sum(m1[Valid] - m2[Valid])
    elif method == 3:
        DisSimilarity = np.max(np.abs(m1[Valid] - m2[Valid]) / np.abs(m1[Valid]))
    elif method == 4:
        DisSimilarity = np.mean(np.abs(m1[Valid] - m2[Valid]) / np.abs(m1[Valid]))
    else:
        raise ValueError('method code error.')
    return DisSimilarity


def getAspectRatio(contour):
    [_, _, w, h] = getRoi_xywh(contour)
    return float(w) / h


def getExtentRatio(contour):
    Area = getArea(contour)
    [_, _, w, h] = getRoi_xywh(contour)
    RectArea = w * h
    return float(Area) / RectArea


def getSolidityRatio(contour):
    Area = getArea(contour)
    Hull = getConvexHull(contour)
    HullArea = getArea(Hull)
    return float(Area) / HullArea


def getEquivalentDiameter(contour):
    Area = getArea(contour)
    return np.sqrt(4 * Area / np.pi)


def getOrientation(contour):
    _, _, _, Angle_rad = fitEllipse(contour)
    return Angle_rad


def getExtremePoints(contour):
    LeftMost_2x1 = np.array(contour[contour[:, :, 0].argmin()][0]).reshape(2, 1)
    RightMost_2x1 = np.array(contour[contour[:, :, 0].argmax()][0]).reshape(2, 1)
    TopMost_2x1 = np.array(contour[contour[:, :, 1].argmin()][0]).reshape(2, 1)
    BottomMost_2x1 = np.array(contour[contour[:, :, 1].argmax()][0]).reshape(2, 1)
    return LeftMost_2x1, RightMost_2x1, TopMost_2x1, BottomMost_2x1


def cvtPoints2Contour(points_2xn):
    return points_2xn.T.reshape(-1, 1, 2).astype(np.int)


def isConvex(contour):
    return cv2.isContourConvex(contour)


def getMoment(contour):
    return cv2.moments(contour)


def getArea(contour):
    return cv2.contourArea(contour)


def getCentroid(contour):
    Moment = getMoment(contour)
    try:
        CentroidPt_2x1 = np.array([[Moment['m10'] / Moment['m00']],
                                   [Moment['m01'] / Moment['m00']]])
    except ZeroDivisionError:
        CentroidPt_2x1 = contour.mean(0).reshape(2, 1)
    return CentroidPt_2x1


def getArcLenth(contour, closed=True):
    return cv2.arcLength(curve=contour, closed=closed)


def approxPolyDP(contour, approxPercent):
    MaxDis = approxPercent * getArcLenth(contour)
    return cv2.approxPolyDP(curve=contour, epsilon=MaxDis, closed=True)


def getConvexHull(contour):
    return cv2.convexHull(contour)


def getRoi_xywh(contour):
    return cv2.boundingRect(contour)


def getRotatedRoi_xywh(contour):
    Rect = cv2.minAreaRect(contour)
    Box = cv.BoxPoints(Rect)
    return np.int(Box)


def fitEnclosingCircle(contour):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    Center_2x1 = np.array([[x],
                           [y]])
    return Center_2x1, radius


def fitLine(contour):
    return cv2.fitLine(contour, cv.CV_DIST_L2, 0, 0.01, 0.01)


def fitEllipse(contour):
    (x, y), (MajorAxisLength, MinorAxisLength), Angle_rad = cv2.fitEllipse(contour)
    Center_2x1 = np.array([[x],
                           [y]])
    return Center_2x1, MajorAxisLength, MinorAxisLength, Angle_rad
