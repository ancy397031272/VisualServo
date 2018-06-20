#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__version__ = 1.0
__date__ = '2016/08/17'

import cv2
import math
import numpy as np

from toolbox import vgl as VGL
from toolbox.vgl.eye_hand_vision import EyeInHandStereoVision as EIHSteV
from toolbox import fileinterface as FIT
from toolbox import imgproctool as IPT
from toolbox.vgl.visual_servo_lib import VisualServoPositionBase as VSPB



if __name__ == '__main__':
    CameraCalibrationData = FIT.loadYaml(fileName='data/CameraCalibrationData_0802.yaml')

    CameraMatrix14 = np.array(CameraCalibrationData['CameraMatrix_1414'])
    CameraMatrix13 = np.array(CameraCalibrationData['CameraMatrix_1313'])
    CameraMatrixA = np.array(CameraCalibrationData['CameraMatrixA'])
    CameraMatrixB = np.array(CameraCalibrationData['CameraMatrixB'])
    DistCoeffs14 = np.array(CameraCalibrationData['DistCoeffs_1414'])
    DistCoeffs13 = np.array(CameraCalibrationData['DistCoeffs_1313'])
    DistCoeffsA = np.array(CameraCalibrationData['DistCoeffsA'])
    DistCoeffsB = np.array(CameraCalibrationData['DistCoeffsB'])

    Tc14c13 = np.array(CameraCalibrationData['Tc14c13'])
    TCACB = np.array(CameraCalibrationData['TCACB'])
    TCA14 = np.array(CameraCalibrationData['TCA14'])

    Tct14 = np.array(CameraCalibrationData['Tct_1414'])
    Tct13 = np.array(np.matrix(Tct14) * (np.matrix(Tc14c13)**-1))
    TctA = np.matrix(Tct14) * np.matrix(TCA14)

    E_AB = np.array(CameraCalibrationData['E_AB'])
    F_AB = np.array(CameraCalibrationData['F_AB'])
    E_1413 = np.array(CameraCalibrationData['E_1413'])
    F_1413 = np.array(CameraCalibrationData['F_1413'])

    Tac = np.array(CameraCalibrationData['TCA14'])

    # MyEIHSteVision = \
    #     EIHSteV(cameraMatrixA=CameraMatrixA, cameraMatrixB=CameraMatrixB,
    #             distCoeffsA=DistCoeffsA, distCoeffsB=DistCoeffsB,
    #             TcAcB=TCACB, E=E_AB, F=F_AB, TctA=TctA, TctB=None)
    MyEIHSteVision = \
        EIHSteV(cameraMatrixA=CameraMatrix14, cameraMatrixB=CameraMatrix13,
                distCoeffsA=DistCoeffs14, distCoeffsB=DistCoeffs13,
                # distCoeffsA=(), distCoeffsB=(),
                TcAcB=Tc14c13, E=E_1413, F=F_1413, TctA=Tct14, TctB=None)

    Pose = np.array([ 301.97729492,  204.28045654,   71.00037384, -153.69342041,    0.,            0.        ])
    Ttr = VGL.Pose2T(pose=Pose)

    RealPinPtsInRob = np.array([[ 299,  302, 300, 303],
                                 [ 194,  191, 190, 190],
                                 [ 128,  128, 130, 130]])[:,:1].reshape((3,-1))

    # DeltaPose = np.array([10,10,10,50,40,30]).reshape((6,1)).astype('float32')
    DeltaPose = np.array([10,10,10,10,0,0]).reshape((6,1)).astype('float32')

    T = VGL.Pose2T(pose=DeltaPose)
    RealHolePtsInRob = VGL.projectPts(pts_dxn=RealPinPtsInRob, projectMatrix=T)

    DeltaPinToHoleInRob = RealHolePtsInRob - RealPinPtsInRob
    print "DeltaPinToHoleInRob:\n", DeltaPinToHoleInRob

    RealImgPtsA_hole = MyEIHSteVision.projectPts(pts_3xn=RealHolePtsInRob, flag=EIHSteV.PROJ_ROB2IMGA, Ttr=Ttr)
    RealImgPtsB_hole = MyEIHSteVision.projectPts(pts_3xn=RealHolePtsInRob, flag=EIHSteV.PROJ_ROB2IMGB, Ttr=Ttr)
    RealImgPtsA_pin = MyEIHSteVision.projectPts(pts_3xn=RealPinPtsInRob, flag=EIHSteV.PROJ_ROB2IMGB, Ttr=Ttr)

    RealImgPtsA_hole = MyEIHSteVision.projectPts(pts_3xn=RealHolePtsInRob, flag=EIHSteV.PROJ_ROB2IMGA, Ttr=Ttr)
    RealImgPtsB_hole = MyEIHSteVision.projectPts(pts_3xn=RealHolePtsInRob, flag=EIHSteV.PROJ_ROB2IMGB, Ttr=Ttr)
    RealImgPtsA_pin = MyEIHSteVision.projectPts(pts_3xn=RealPinPtsInRob, flag=EIHSteV.PROJ_ROB2IMGA, Ttr=Ttr)
    RealImgPtsB_pin = MyEIHSteVision.projectPts(pts_3xn=RealPinPtsInRob, flag=EIHSteV.PROJ_ROB2IMGB, Ttr=Ttr)

    RealHolePtsInCamA, _ = \
        MyEIHSteVision.get3dPts(imgPtsA_2xn=RealImgPtsA_hole, imgPtsB_2xn=RealImgPtsB_hole, flag=EIHSteV.COOR_IMG2CAM)
    RealPinPtsInCamA, _ = \
        MyEIHSteVision.get3dPts(imgPtsA_2xn=RealImgPtsA_pin, imgPtsB_2xn=RealImgPtsB_pin, flag=EIHSteV.COOR_IMG2CAM)
    RealHolePtsInTool, _ = \
        MyEIHSteVision.get3dPts(imgPtsA_2xn=RealImgPtsA_hole, imgPtsB_2xn=RealImgPtsB_hole, flag=EIHSteV.COOR_IMG2TOOL)
    RealPinPtsInTool, _ = \
        MyEIHSteVision.get3dPts(imgPtsA_2xn=RealImgPtsA_pin, imgPtsB_2xn=RealImgPtsB_pin, flag=EIHSteV.COOR_IMG2TOOL)

    ShowImgA = np.zeros(shape=(5000,5000,3), dtype=np.uint8)
    ShowImgB = np.zeros(shape=(5000,5000,3), dtype=np.uint8)
    IPT.drawPoints(img=ShowImgA, pts_2xn=RealImgPtsA_hole, color=(0,0,255), radius=20, thickness=-1)
    IPT.drawPoints(img=ShowImgB, pts_2xn=RealImgPtsB_hole, color=(0,0,255), radius=20, thickness=-1)
    IPT.drawPoints(img=ShowImgA, pts_2xn=RealImgPtsA_pin, color=(0,255,0), radius=20, thickness=5)
    IPT.drawPoints(img=ShowImgB, pts_2xn=RealImgPtsB_pin, color=(0,255,0), radius=20, thickness=5)

    cv2.namedWindow('A', cv2.WINDOW_NORMAL)
    cv2.namedWindow('B', cv2.WINDOW_NORMAL)
    cv2.imshow('A', ShowImgA)
    cv2.imshow('B', ShowImgB)
    cv2.waitKey()

    Times = 0
    while True:
        Times += 1
        # if Times > 50:
        #     break
        HolePtsInCamA, _ = \
            MyEIHSteVision.get3dPts(imgPtsA_2xn=RealImgPtsA_hole, imgPtsB_2xn=RealImgPtsB_hole, flag=EIHSteV.COOR_IMG2CAM)
        PinPtsInCamA, _ = \
            MyEIHSteVision.get3dPts(imgPtsA_2xn=RealImgPtsA_pin, imgPtsB_2xn=RealImgPtsB_pin, flag=EIHSteV.COOR_IMG2CAM)
        Pin2HoleInCamA = HolePtsInCamA - PinPtsInCamA
        print "Pin2HoleInCamA: \n", Pin2HoleInCamA

        Tcr = Ttr.dot(Tct14)
        # VelocityInRob = VSPB.matchPP(objPts_3xn=HolePtsInCamA, tarPts_3xn=PinPtsInCamA, Tx2x=Tcr, mask=(1,1,1,0,0,1))
        VelocityInTool = VSPB.matchPP(objPts_3xn=HolePtsInCamA, tarPts_3xn=PinPtsInCamA, Tx2x=Tct14, mask=(1,1,1,1,1,1))
        VelocityInRob = VSPB.transVelocity(Tx2x=Ttr, velocity=VelocityInTool)
        print 'VelocityInRob: ', VelocityInRob.reshape(-1)

        MoveT = VSPB.velocityPose2Move(velocityPose=VelocityInRob, iter=10000)
        Ttr = MoveT.dot(Ttr)

        RealImgPtsA_hole = MyEIHSteVision.projectPts(pts_3xn=RealHolePtsInRob, flag=EIHSteV.PROJ_ROB2IMGA, Ttr=Ttr)
        RealImgPtsB_hole = MyEIHSteVision.projectPts(pts_3xn=RealHolePtsInRob, flag=EIHSteV.PROJ_ROB2IMGB, Ttr=Ttr)
        print (RealImgPtsA_hole - RealImgPtsA_pin)
        print (RealImgPtsB_hole - RealImgPtsB_pin)

        ShowImgA = np.zeros(shape=(5000,5000,3), dtype=np.uint8)
        ShowImgB = np.zeros(shape=(5000,5000,3), dtype=np.uint8)
        IPT.drawPoints(img=ShowImgA, pts_2xn=RealImgPtsA_hole, color=(0,0,255), radius=20, thickness=-1)
        IPT.drawPoints(img=ShowImgB, pts_2xn=RealImgPtsB_hole, color=(0,0,255), radius=20, thickness=-1)
        IPT.drawPoints(img=ShowImgA, pts_2xn=RealImgPtsA_pin, color=(0,255,0), radius=20, thickness=5)
        IPT.drawPoints(img=ShowImgB, pts_2xn=RealImgPtsB_pin, color=(0,255,0), radius=20, thickness=5)

        cv2.namedWindow('A', cv2.WINDOW_NORMAL)
        cv2.namedWindow('B', cv2.WINDOW_NORMAL)
        cv2.imshow('A', ShowImgA)
        cv2.imshow('B', ShowImgB)
        cv2.waitKey()
