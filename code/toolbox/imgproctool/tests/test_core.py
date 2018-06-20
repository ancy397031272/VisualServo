import numpy as np
import cv2
from imgproctool import *

if __name__ == '__main__':
    Img = np.zeros((10, 10), np.uint8)
    Roi_xyxy = [2, 2, 5, 5]
    Roi_xywh = [2, 2, 3, 3]
    CvtRoi_xywh = cvtRoi(Roi_xyxy, flag=ROI_CVT_XYXY2XYWH)
    CvtRoi_xyxy = cvtRoi(Roi_xywh, flag=ROI_CVT_XYWH2XYXY)
    print 'xyxy: ', Roi_xyxy, 'convert to xywh->', CvtRoi_xywh, 'convert back->', cvtRoi(CvtRoi_xywh, ROI_CVT_XYWH2XYXY)
    print 'xywh: ', Roi_xywh, 'convert to xyxy->', CvtRoi_xyxy, 'convert back->', cvtRoi(CvtRoi_xyxy, ROI_CVT_XYXY2XYWH)

    _, RoiImg_xyxy = getRoiImg(Img, Roi_xyxy, roiType=ROI_TYPE_XYXY)
    print 'RoiImg_xyxy shape: ', RoiImg_xyxy.shape
    _, RoiImg_xywh = getRoiImg(Img, Roi_xywh, roiType=ROI_TYPE_XYWH)
    print 'RoiImg_xywh shape: ', RoiImg_xywh.shape

    Point = [0, 0]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [2, 2]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [3, 3]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [4, 4]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)
    Point = [5, 5]
    print Point, 'in roi_xyxy?', inRoi(Point, Roi_xyxy, ROI_TYPE_XYXY), '\t/\t in roi_xywh?', inRoi(Point, Roi_xywh, ROI_TYPE_XYWH)

    cv2.namedWindow('Roi2', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Roi1', cv2.WINDOW_NORMAL)
    Img1 = np.zeros((100, 100), np.uint8)
    Img2 = np.zeros((100, 100), np.uint8)
    RotatedRoi1 = [[10, 10],
                  [50, 20],
                  [70, 70],
                  [20, 30]]
    RotatedRoi2 = [[10, 50, 70, 20],
                  [10, 20, 70, 30]]
    print 'RotatedRoi:', RotatedRoi1
    Point_2x1 = np.array([10, 20]).reshape(2, 1)
    drawPoints(Img1, Point_2x1, 255)
    print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    Point_2x1 = np.array([10, 10]).reshape(2, 1)
    drawPoints(Img1, Point_2x1, 255)
    print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    Point_2x1 = np.array([70, 70]).reshape(2, 1)
    drawPoints(Img1, Point_2x1, 255)
    drawPoints(Img1, Point_2x1, 255, offset=(-10, 10))
    print Point_2x1.ravel(), 'in RotatedRoi?', inRoi(Point_2x1, RotatedRoi1, ROI_TYPE_ROTATED)
    drawRoi(Img1, RotatedRoi1, ROI_TYPE_ROTATED, color=255, thickness=-1)
    drawRoi(Img2, RotatedRoi2, ROI_TYPE_ROTATED, color=255, offset=(-50, -50))
    Contours, _ = cv2.findContours(image=Img2.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    Rect = cv2.minAreaRect(Contours[0])
    Box = cv2.cv.BoxPoints(Rect)
    BoxImg = np.zeros((200, 200), np.uint8)
    drawRoi(Img2, Box, ROI_TYPE_ROTATED, color=255)
    drawRoi(Img1, RotatedRoi1, ROI_TYPE_ROTATED, color=255, offset=(20, 20))
    # drawRoi(Img2, RotatedRoi2, ROI_TYPE_ROTATED, color=255, offset=(20, 20))
    cv2.imshow('Roi2', Img2)
    cv2.imshow('Roi1', Img1)
    cv2.waitKey()

    # Img = (np.random.random((100, 100)) * 255).astype(np.uint8)
    # roi_xywh = [10, 10, 20, 20]
    # roi_xyxy = [10, 10, 30, 30]
    # print 'Roi_xywh:      ', roi_xywh
    # roi_xywh2xyxy = cvtRoi(roi=roi_xywh, flag=ROI_CVT_XYWH2XYXY)
    # roi_xyxy2xywh = cvtRoi(roi=roi_xyxy, flag=ROI_CVT_XYXY2XYWH)
    #
    # print 'roi_xywh2xyxy: ', roi_xywh2xyxy
    # print 'roi_xyxy2xywh: ', roi_xyxy2xywh
    #
    # _, RoiImg_xywh = getRoiImg(Img, roi_xywh, roiType=ROI_TYPE_XYWH)
    # print 'RoiImg xywh:', RoiImg_xywh.shape
    #
    # _, RoiImg_xyxy = getRoiImg(Img, roi_xyxy, roiType=ROI_TYPE_XYXY)
    # print 'RoiImg xyxy:', RoiImg_xyxy.shape
    # print np.allclose(RoiImg_xywh, RoiImg_xyxy)


    # SrcImg = cv2.imread('../Data/girl.jpg')
    # SrcImg = cv2.imread('../Data/Cam14.bmp')
    # resizeImg1 = cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5))
    # drawRoi(img=resizeImg1, roi=roi_xywh, roiType=ROI_TYPE_XYWH, color=(0,0,255))
    # cv2.imshow('roi_xywh', resizeImg1)
    # resizeImg2 = cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5))
    # drawRoi(img=resizeImg2, roi=roi_xywh2xyxy, roiType=ROI_TYPE_XYXY, color=(0,0,255))
    # cv2.imshow('roi_xywh2xyxy', resizeImg2)
    #
    # RotateImg = rotateImg(src=SrcImg, angle_deg=30)
    # cv2.namedWindow("RotateImg", cv2.WINDOW_NORMAL)
    # cv2.imshow("RotateImg", RotateImg)
    #
    # cv2.waitKey()