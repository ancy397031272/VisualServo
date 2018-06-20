#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__version__ = 1.0
__date__ = 01/12/2015

import unittest
import numpy as np

from toolbox import vgl as VGL
from toolbox.vgl.eye_hand_vision import EyeInHandStereoVision as EIHSteV
from toolbox.vgl.stereo_vision import StereoVision as SteV
from toolbox import fileinterface as FIT

class TestEyeInHandStereoVision(unittest.TestCase):
    def setUp(self):
        ParameterYamlPath = 'data/CameraCalibrationData.yaml'
        CameraCalibrationData = FIT.loadYaml(fileName=ParameterYamlPath)

        TcAcB = np.array(CameraCalibrationData['Tc14c13'])
        CameraMatrix_14 = np.array(CameraCalibrationData['CameraMatrix_1414'])
        CameraMatrix_13 = np.array(CameraCalibrationData['CameraMatrix_1313'])
        DistCoeffs_1414 = np.array(CameraCalibrationData['DistCoeffs_1414'])
        DistCoeffs_1313 = np.array(CameraCalibrationData['DistCoeffs_1313'])
        E = np.array(CameraCalibrationData['E'])
        F = np.array(CameraCalibrationData['F'])
        TctA = np.array(CameraCalibrationData['Tct_1414'])
        TctB = np.array(CameraCalibrationData['Tct_1313'])

        ToolPose = [454.19, -202.32, 292.58, 126.09, 3.69, 177.50]
        self.Ttr = VGL.Pose2T(pose=ToolPose)

        self.PtsImgA_2xn = np.array([[6.366322275682084637e+02, 7.899394910272621928e+02],
                                     [7.416458478168997317e+02, 6.542366444992233028e+02]], dtype=np.float32)
        self.PtsImgB_2xn = np.array([[8.149478495193284289e+02, 9.375415089453317705e+02],
                                     [3.091887147386404422e+02, 3.632621606795080424e+02]], dtype=np.float32)

        MySteV = \
            SteV(cameraMatrixA=CameraMatrix_14, cameraMatrixB=CameraMatrix_13,
                 distCoeffsA=DistCoeffs_1414, distCoeffsB=DistCoeffs_1313, TcAcB=TcAcB, E=E, F=F)
        self.EIHSteV = \
            EIHSteV(cameraMatrixA=CameraMatrix_14, cameraMatrixB=CameraMatrix_13,
                    distCoeffsA=DistCoeffs_1414, distCoeffsB=DistCoeffs_1313,
                    TcAcB=TcAcB, E=E, F=F, TctA=TctA, TctB=TctB)

        self.PtsInCamA_3xn, self.PtsInCamB_3xn, _, _ = \
            MySteV.get3dPts(imgPtsA_2xn=self.PtsImgA_2xn, imgPtsB_2xn=self.PtsImgB_2xn)
        self.PtsInToolA_3xn = VGL.projectPts(pts_dxn=self.PtsInCamA_3xn, projectMatrix=TctA)
        self.PtsInToolB_3xn = VGL.projectPts(pts_dxn=self.PtsInCamB_3xn, projectMatrix=TctB)
        self.PtsInRobA_3xn = VGL.projectPts(pts_dxn=self.PtsInToolA_3xn, projectMatrix=self.Ttr)
        self.PtsInRobB_3xn = VGL.projectPts(pts_dxn=self.PtsInToolB_3xn, projectMatrix=self.Ttr)

        self.ProjImgPtsA = MySteV.projectPts(pts_3xn=self.PtsInCamA_3xn, flag=MySteV.CAM_A)
        self.ProjImgPtsB = MySteV.projectPts(pts_3xn=self.PtsInCamB_3xn, flag=MySteV.CAM_B)

    def assertAlmostEqualArray(self, first, second, delta=1.0):
        assert first.shape == second.shape
        assert (abs(first - second) < delta).all()

    def testGet3dPts(self):
        PtsInCamA_3xn, PtsInCamB_3xn = \
            self.EIHSteV.get3dPts(imgPtsA_2xn=self.PtsImgA_2xn, imgPtsB_2xn=self.PtsImgB_2xn,
                                  flag=self.EIHSteV.COOR_IMG2CAM)
        self.assertAlmostEqualArray(first=PtsInCamA_3xn, second=self.PtsInCamA_3xn, delta=0.001)
        self.assertAlmostEqualArray(first=PtsInCamB_3xn, second=self.PtsInCamB_3xn, delta=0.001)

        PtsInToolA_3xn, PtsInToolB_3xn = \
            self.EIHSteV.get3dPts(imgPtsA_2xn=self.PtsImgA_2xn, imgPtsB_2xn=self.PtsImgB_2xn,
                                  flag=self.EIHSteV.COOR_IMG2TOOL)
        self.assertAlmostEqualArray(first=PtsInToolA_3xn, second=self.PtsInToolA_3xn, delta=0.001)
        self.assertAlmostEqualArray(first=PtsInToolB_3xn, second=self.PtsInToolB_3xn, delta=0.001)

        PtsInRobA_3xn, PtsInRobB_3xn = \
            self.EIHSteV.get3dPts(imgPtsA_2xn=self.PtsImgA_2xn, imgPtsB_2xn=self.PtsImgB_2xn,
                                  Ttr=self.Ttr, flag=self.EIHSteV.COOR_IMG2ROB)
        self.assertAlmostEqualArray(first=PtsInRobA_3xn, second=self.PtsInRobA_3xn, delta=0.001)
        self.assertAlmostEqualArray(first=PtsInRobB_3xn, second=self.PtsInRobB_3xn, delta=0.001)

    def testProjectPts(self):
        PtsInImgA_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInRobA_3xn, flag=self.EIHSteV.PROJ_ROB2IMGA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgA_2xn, second=self.ProjImgPtsA, delta=0.001)

        PtsInImgA_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInToolA_3xn, flag=self.EIHSteV.PROJ_TOOL2IMGA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgA_2xn, second=self.ProjImgPtsA, delta=0.001)

        PtsInImgA_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInCamA_3xn, flag=self.EIHSteV.PROJ_CAMA2IMGA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgA_2xn, second=self.ProjImgPtsA, delta=0.001)

        PtsInImgA_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInCamB_3xn, flag=self.EIHSteV.PROJ_CAMB2IMGA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgA_2xn, second=self.ProjImgPtsA, delta=0.001)

        PtsInImgB_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInRobB_3xn, flag=self.EIHSteV.PROJ_ROB2IMGB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgB_2xn, second=self.ProjImgPtsB, delta=0.001)

        PtsInImgB_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInToolB_3xn, flag=self.EIHSteV.PROJ_TOOL2IMGB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgB_2xn, second=self.ProjImgPtsB, delta=0.001)

        PtsInImgB_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInCamB_3xn, flag=self.EIHSteV.PROJ_CAMB2IMGB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInImgB_2xn, second=self.ProjImgPtsB, delta=0.001)

        PtsInImgB_2xn = \
            self.EIHSteV.projectPts(pts_3xn=self.PtsInCamA_3xn, flag=self.EIHSteV.PROJ_CAMA2IMGB)
        self.assertAlmostEqualArray(first=PtsInImgB_2xn, second=self.ProjImgPtsB, delta=0.001)

    def testCvtCoods(self):
        PtsInTool_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInRobA_3xn, flag=self.EIHSteV.COOR_ROB2TOOL, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInTool_3xn, second=self.PtsInToolA_3xn, delta=0.001)

        PtsInCamA_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInRobA_3xn, flag=self.EIHSteV.COOR_ROB2CAMA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInCamA_3xn, second=self.PtsInCamA_3xn, delta=0.001)

        PtsInCamB_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInRobB_3xn, flag=self.EIHSteV.COOR_ROB2CAMB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInCamB_3xn, second=self.PtsInCamB_3xn, delta=0.001)

        PtsInRob_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInToolA_3xn, flag=self.EIHSteV.COOR_TOOL2ROB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInRob_3xn, second=self.PtsInRobA_3xn, delta=0.001)

        PtsInCamA_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInToolA_3xn, flag=self.EIHSteV.COOR_TOOL2CAMA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInCamA_3xn, second=self.PtsInCamA_3xn, delta=0.001)

        PtsInCamB_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInToolA_3xn, flag=self.EIHSteV.COOR_TOOL2CAMB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInCamB_3xn, second=self.PtsInCamB_3xn, delta=0.001)

        PtsInRob_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInCamA_3xn, flag=self.EIHSteV.COOR_CAMA2ROB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInRob_3xn, second=self.PtsInRobA_3xn, delta=0.001)

        PtsInTool_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInCamA_3xn, flag=self.EIHSteV.COOR_CAMA2TOOL, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInTool_3xn, second=self.PtsInToolA_3xn, delta=0.001)

        PtsInCamB_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInCamA_3xn, flag=self.EIHSteV.COOR_CAMA2CAMB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInCamB_3xn, second=self.PtsInCamB_3xn, delta=0.001)

        PtsInRob_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInCamB_3xn, flag=self.EIHSteV.COOR_CAMB2ROB, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInRob_3xn, second=self.PtsInRobB_3xn, delta=0.001)

        PtsInTool_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInCamB_3xn, flag=self.EIHSteV.COOR_CAMB2TOOL, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInTool_3xn, second=self.PtsInToolB_3xn, delta=0.001)

        PtsInCamA_3xn = \
            self.EIHSteV.cvtCoords(pts_3xn=self.PtsInCamB_3xn, flag=self.EIHSteV.COOR_CAMB2CAMA, Ttr=self.Ttr)
        self.assertAlmostEqualArray(first=PtsInCamA_3xn, second=self.PtsInCamA_3xn, delta=0.001)

    def testGetNewPoseOrCoords(self):
        SrcT = np.array([[ 1, 0, 0, 1],
                         [ 0, 1, 0, 1],
                         [ 0, 0, 1, 1],
                         [ 0, 0, 0, 1]], dtype=np.float32)
        SrcPose = VGL.T2Pose(T_4x4=SrcT)
        DeltaPose = [0,0,0,90,90,90]
        DeltaT = VGL.Pose2T(DeltaPose)

        RefNewT_ByBase = np.dot(DeltaT, SrcT)
        RefNewPose_ByBase = VGL.T2Pose(RefNewT_ByBase)
        RefNewT_BySelf = np.dot(SrcT, DeltaT)
        RefNewPose_BySelf = VGL.T2Pose(RefNewT_BySelf)

        NewT_ByBase = \
            self.EIHSteV.getNewPoseOrCoords(poseOrCoords=SrcT, deltaPose=DeltaPose, flag=EIHSteV.TRAN_BY_BASE)
        self.assertAlmostEqualArray(first=NewT_ByBase, second=RefNewT_ByBase, delta=0.00001)

        NewPose_ByBase = \
            self.EIHSteV.getNewPoseOrCoords(poseOrCoords=SrcPose, deltaPose=DeltaPose, flag=EIHSteV.TRAN_BY_BASE)
        self.assertAlmostEqualArray(first=NewPose_ByBase, second=RefNewPose_ByBase, delta=0.00001)

        NewT_BySelf = \
            self.EIHSteV.getNewPoseOrCoords(poseOrCoords=SrcT, deltaPose=DeltaPose, flag=EIHSteV.TRAN_BY_SELF)
        self.assertAlmostEqualArray(first=NewT_BySelf, second=RefNewT_BySelf, delta=0.00001)

        NewPose_BySelf = \
            self.EIHSteV.getNewPoseOrCoords(poseOrCoords=SrcPose, deltaPose=DeltaPose, flag=EIHSteV.TRAN_BY_SELF)
        self.assertAlmostEqualArray(first=NewPose_BySelf, second=RefNewPose_BySelf, delta=0.00001)

    def testGetNewToolPose(self):
        pass


if __name__ == '__main__':
    unittest.main()


# if __name__ == "__main__":
#     ParameterYamlPath = './Data/CameraCalibrationData.yaml'
#     CameraCalibrationData = FIT.loadYaml(fileName=ParameterYamlPath)
#
#     TcAcB = np.array(CameraCalibrationData['Tc14c13'])
#     CameraMatrix_1414 = np.array(CameraCalibrationData['CameraMatrix_1414'])
#     CameraMatrix_1313 = np.array(CameraCalibrationData['CameraMatrix_1313'])
#     DistCoeffs_1414 = np.array(CameraCalibrationData['DistCoeffs_1414'])
#     DistCoeffs_1313 = np.array(CameraCalibrationData['DistCoeffs_1313'])
#     E = np.array(CameraCalibrationData['E'])
#     F = np.array(CameraCalibrationData['F'])
#     TctA = np.array(CameraCalibrationData['Tct_1414'])
#     TctB = np.array(CameraCalibrationData['Tct_1313'])
#
#
#     ToolPose = [454.20, -202.29, 292.58, 123.52, 3.7, 177.51]
#     Ttr = VGL.Pose2T(pose=ToolPose)
#
#     Pins_14 = np.array([[1429, 1553],
#                         [ 858,  792]], dtype=np.float32)
#     Pins_13 = np.array([[1568, 1766],
#                         [ 594,  724]], dtype=np.float32)
#     Holes_14 = np.array([[1554, 1676],
#                          [ 868,  801]], dtype=np.float32)
#     Holes_13 = np.array([[1673, 1876],
#                          [ 801,  938]], dtype=np.float32)
#
#     MyEIHSteV = \
#         EyeInHandStereoVision(cameraMatrixA=CameraMatrix_1414, cameraMatrixB=CameraMatrix_1313,
#                               distCoeffsA=DistCoeffs_1414, distCoeffsB=DistCoeffs_1313,
#                               TcAcB=TcAcB, E=E, F=F, TctA=TctA, TctB=TctB)
#
#     print "=================get3dPts=========================="
#     PinPtsInRobA_3xn, PinPtsInRobB_3xn = \
#         MyEIHSteV.get3dPts(imgPtsA_2xn=Pins_14, imgPtsB_2xn=Pins_13, flag=MyEIHSteV.COOR_IMG2ROB, Ttr=Ttr)
#     print "PinPtsInRobA:\n", PinPtsInRobA_3xn
#     print "PinPtsInRobB:\n", PinPtsInRobB_3xn
#
#     HolePtsInRobA_3xn, HolePtsInRobB_3xn = \
#         MyEIHSteV.get3dPts(imgPtsA_2xn=Holes_14, imgPtsB_2xn=Holes_13, flag=MyEIHSteV.COOR_IMG2ROB, Ttr=Ttr)
#     print "HolePtsInRobA:\n", HolePtsInRobA_3xn
#     print "HolePtsInRobB:\n", HolePtsInRobB_3xn
#
#     PinDis = np.linalg.norm(PinPtsInRobA_3xn[:,0] - PinPtsInRobA_3xn[:,1])
#     print "PinDis:", PinDis
#
#     HoleDis = np.linalg.norm(HolePtsInRobA_3xn[:,0] - HolePtsInRobA_3xn[:,1])
#     print "HoleDis:", HoleDis
#
#
#     print "===============getNewToolPose======================"
#     PinVec  = PinPtsInRobA_3xn[:,1]  - PinPtsInRobA_3xn[:,0]
#     HoleVec = HolePtsInRobA_3xn[:,1] - HolePtsInRobA_3xn[:,0]
#
#     PinVec[2] = 0
#     HoleVec[2] = 0
#
#     # PinHoleAng = VGL.computeVectorAngle_rad(vec0=PinVec, vec1=HoleVec) / math.pi * 180
#     RotateVec = VGL.computeRotateVec(vec0=PinVec, vec1=HoleVec)
#     PinHoleAng = np.linalg.norm(RotateVec) / math.pi * 180
#     if RotateVec[2] < 0:
#         PinHoleAng = -PinHoleAng
#     print "PinHoleAng: ", PinHoleAng
#
#     DeltaU = PinHoleAng
#     # DeltaX = PinPtsInRobA_3xn[0,0] - HolePtsInRobA_3xn[0,0]
#     DeltaX = HolePtsInRobA_3xn[0,0] - PinPtsInRobA_3xn[0,0]
#     DeltaPose = [DeltaX, 0, 0, DeltaU, 0, 0]
#     print "DeltaPose: ", DeltaPose
#
#     PinCoords = np.vstack((np.hstack((np.eye(3),PinPtsInRobA_3xn[:,0].reshape(3,1))), np.array([0,0,0,1])))
#     print "PinCoords:\n", PinCoords
#
#     NewToolPose = \
#         MyEIHSteV.getNewToolPose(srcCoordInRob=PinCoords, deltaPose=DeltaPose, curTtr=Ttr, flag=MyEIHSteV.TRAN_BY_SELF)
#     print "NewToolPose: ", NewToolPose
