#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__version__ = '1.0'
__date__ = '25/12/2015'

import numpy as np
from . import core
from .stereo_vision import StereoVision

__all__ = ['EyeHandVision', 'EyeToHandStereoVision', 'EyeInHandStereoVision']


class EyeHandVision(object):
    CAM_A = 0L
    CAM_B = 1L

    COOR_ROB        = 1L
    COOR_TOOL       = 2L
    COOR_CAMA       = 3L
    COOR_CAMB       = 4L
    COOR_IMGA       = 5L
    COOR_IMGB       = 6L

    COOR_ROB2TOOL   = 7L
    COOR_ROB2CAMA   = 8L
    COOR_ROB2CAMB   = 9L
    COOR_TOOL2ROB   = 12L
    COOR_TOOL2CAMA  = 13L
    COOR_TOOL2CAMB  = 14L
    COOR_CAMA2ROB   = 17L
    COOR_CAMA2TOOL  = 18L
    COOR_CAMA2CAMB  = 19L
    COOR_CAMB2ROB   = 21L
    COOR_CAMB2TOOL  = 22L
    COOR_CAMB2CAMA  = 23L

    COOR_IMG2ROB    = 26L
    COOR_IMG2TOOL   = 27L
    COOR_IMG2CAM    = 28L

    PROJ_ROB2IMGA   = COOR_ROB2CAMA
    PROJ_ROB2IMGB   = COOR_ROB2CAMB
    PROJ_TOOL2IMGA  = COOR_TOOL2CAMA
    PROJ_TOOL2IMGB  = COOR_TOOL2CAMB
    PROJ_CAMA2IMGB  = COOR_CAMA2CAMB
    PROJ_CAMB2IMGA  = COOR_CAMB2CAMA
    PROJ_CAMA2IMGA  = 29L
    PROJ_CAMB2IMGB  = 30L

    TRAN_BY_BASE    = 31L
    TRAN_BY_SELF    = 32L
    def __init__(self):
        object.__init__(self)

    def getNewPoseOrCoords(self, poseOrCoords, deltaPose, flag):
        """
        :param poseOrCoords: 4x4 array, 6-element array, list, tuple
        :param deltaPose: 6-element array, list, tuple
        :param flag: must be one in (TRAN_BY_BASE, TRAN_BY_SELF)
        :return: new poseOrCoords, same type of the poseOrCoords
        """
        assert 6 == np.array(deltaPose).size

        if 6 == np.array(poseOrCoords).size:
            SrcT = core.Pose2T(pose=poseOrCoords)
        elif core.isArray(poseOrCoords, checkSize=(4,4)):
            SrcT = poseOrCoords.copy()
        else:
            raise ValueError, "src must be 6-element pose or 4x4 array"

        DeltaT = core.Pose2T(pose=deltaPose)
        if flag == self.TRAN_BY_BASE:
            NewT = np.dot(DeltaT, SrcT)
        elif flag == self.TRAN_BY_SELF:
            NewT = np.dot(SrcT, DeltaT)
        else:
            raise ValueError, "flag must be TRAN_BY_BASE or TRAN_BY_SELF"

        if core.isArray(poseOrCoords, checkSize=(4,4)):
            return NewT
        else:
            NewPose = core.T2Pose(T_4x4=NewT)
            return NewPose

    def getNewToolPose(self, srcCoordInRob, deltaPose, curTtr, flag):
        """
        :param srcCoordInRob: 4x4 array
        :param deltaPose: 6-element array, list, tuple
        :param curTtr: 4x4 array
        :param flag: must be one in (TRAN_BY_BASE, TRAN_BY_SELF)
        :return: new pose, 6-element array
        """
        assert core.isArray(srcCoordInRob, checkSize=(4,4))
        assert 6 == np.array(deltaPose).size
        assert core.isArray(curTtr, checkSize=(4,4))
        assert self.TRAN_BY_BASE == flag or self.TRAN_BY_SELF == flag

        Tot_Fix = np.matrix(curTtr)**-1 * np.matrix(srcCoordInRob)
        NewTor = self.getNewPoseOrCoords(poseOrCoords=srcCoordInRob, deltaPose=deltaPose, flag=flag)
        NewTtr = np.matrix(NewTor) * (Tot_Fix ** -1)
        return core.T2Pose(np.array(NewTtr))


class EyeToHandStereoVision(EyeHandVision):
    def __init__(self):
        object.__init__(self)


class EyeInHandStereoVision(EyeHandVision):
    def __init__(self, cameraMatrixA, cameraMatrixB, distCoeffsA, distCoeffsB, TcAcB, E=None, F=None, TctA=None, TctB=None):
        """
        :param cameraMatrixA: 3x3 array
        :param cameraMatrixB: 3x3 array
        :param distCoeffsA: list tuple or array
        :param distCoeffsB: list tuple or array
        :param TcAcB: 4x4 array
        :param E: 3x3 array or None
        :param F: 3x3 array or None
        :param TctA: 4x4 array or None
        :param TctB: 4x4 array or None
        :return: None
        """
        assert core.isArray(cameraMatrixA, checkSize=(3,3))
        assert core.isArray(cameraMatrixB, checkSize=(3,3))
        assert isinstance(distCoeffsA, (tuple, list, np.ndarray))
        assert isinstance(distCoeffsB, (tuple, list, np.ndarray))
        assert core.isArray(TcAcB, checkSize=(4,4))


        EyeHandVision.__init__(self)
        if (TctA is None) and (TctB is None):
            raise ValueError, "TctA and TctB can't all be None"

        if TctA is None:
            TctA = np.dot(TctB, TcAcB)
        elif TctB is None:
            TctB = np.dot(TctA, np.linalg.inv(TcAcB))

        self.__MyStereoVision = \
            StereoVision(cameraMatrixA=cameraMatrixA, cameraMatrixB=cameraMatrixB,
                         distCoeffsA=distCoeffsA, distCoeffsB=distCoeffsB, TcAcB=TcAcB, E=E, F=F)

        self.__TctA = TctA
        self.__TctB = TctB
        self.__Ttr = None
        self.__Tx2x = {self.COOR_ROB2TOOL : None,
                       self.COOR_ROB2CAMA : None,
                       self.COOR_ROB2CAMB : None,
                       self.COOR_TOOL2ROB : None,
                       self.COOR_CAMA2ROB : None,
                       self.COOR_CAMB2ROB : None,
                       self.COOR_TOOL2CAMA: np.linalg.inv(self.__TctA),
                       self.COOR_TOOL2CAMB: np.linalg.inv(self.__TctB),
                       self.COOR_CAMA2TOOL: self.__TctA.copy(),
                       self.COOR_CAMB2TOOL: self.__TctB.copy(),
                       self.COOR_CAMB2CAMA: np.linalg.inv(TcAcB),
                       self.COOR_CAMA2CAMB: TcAcB.copy(),
                       }

    def __updateTx2x(self, Ttr):
        """
        :param Ttr: 4x4 array
        :return: None
        """
        assert core.isArray(Ttr, checkSize=(4,4))

        self.__Ttr = Ttr
        self.__Tx2x[self.COOR_ROB2TOOL]  = np.linalg.inv(self.__Ttr)
        self.__Tx2x[self.COOR_ROB2CAMA]  = np.linalg.inv(self.__TctA).dot(np.linalg.inv(self.__Ttr))
        self.__Tx2x[self.COOR_ROB2CAMB]  = np.linalg.inv(self.__TctB).dot(np.linalg.inv(self.__Ttr))
        self.__Tx2x[self.COOR_TOOL2ROB]  = self.__Ttr.copy()
        self.__Tx2x[self.COOR_CAMA2ROB]  = self.__Ttr.dot(self.__TctA)
        self.__Tx2x[self.COOR_CAMB2ROB]  = self.__Ttr.dot(self.__TctB)

    def cvtCoords(self, pts_3xn, flag, Ttr=None):
        """
        :param srcPts: 3xn array
        :param flag: must one of in (COOR_ROB2TOOL, COOR_ROB2CAMA, COOR_ROB2CAMB,\
                                     COOR_TOOL2ROB, COOR_TOOL2CAMA, COOR_TOOL2CAMB,\
                                     COOR_CAMA2ROB, COOR_CAMA2TOOL COOR_CAMA2CAMB,\
                                     COOR_CAMB2ROB, COOR_CAMB2TOOL,  COOR_CAMB2CAMA)
        :param Ttr: when flag is relevant to Robot, it must be 4x4 array
        :return: 3xn array
        """
        assert flag in (self.COOR_ROB2TOOL, self.COOR_ROB2CAMA, self.COOR_ROB2CAMB,
                        self.COOR_TOOL2ROB, self.COOR_TOOL2CAMA, self.COOR_TOOL2CAMB,
                        self.COOR_CAMA2ROB, self.COOR_CAMA2TOOL, self.COOR_CAMA2CAMB,
                        self.COOR_CAMB2ROB, self.COOR_CAMB2TOOL, self.COOR_CAMB2CAMA)


        if flag in (self.COOR_ROB2TOOL, self.COOR_ROB2CAMA, self.COOR_ROB2CAMB,
                    self.COOR_TOOL2ROB, self.COOR_CAMA2ROB, self.COOR_CAMB2ROB):
            assert core.isArray(Ttr, checkSize=(4,4))
            self.__updateTx2x(Ttr=Ttr)

        ProjectPts_3xn = core.projectPts(pts_dxn=pts_3xn, projectMatrix=self.__Tx2x[flag])
        return ProjectPts_3xn

    def get3dPts(self, imgPtsA_2xn, imgPtsB_2xn, flag, Ttr=None):
        """
        :param imgPtsA_2xn: 2xn array
        :param imgPtsB_2xn: 2xn array
        :param flag: must be one in (COOR_IMG2CAM, COOR_IMG2TOOL, COOR_IMG2ROB)
        :param Ttr: when flag is relevant to Robot, it must be 4x4 array
        :return: 3xn array
        """
        assert 2 == imgPtsA_2xn.shape[0]
        assert 2 == imgPtsA_2xn.ndim
        assert 2 == imgPtsB_2xn.shape[0]
        assert 2 == imgPtsB_2xn.ndim
        assert flag in (self.COOR_IMG2CAM, self.COOR_IMG2TOOL, self.COOR_IMG2ROB)

        if flag  == self.COOR_IMG2ROB:
            assert core.isArray(Ttr, checkSize=(4,4))
            self.__updateTx2x(Ttr=Ttr)

        PtsInCamA_3xn, PtsInCamB_3xn, _, _ = \
            self.__MyStereoVision.get3dPts(imgPtsA_2xn=imgPtsA_2xn, imgPtsB_2xn=imgPtsB_2xn)

        if self.COOR_IMG2CAM == flag:
            return PtsInCamA_3xn, PtsInCamB_3xn
        elif self.COOR_IMG2TOOL == flag:
            PtsInToolA_3xn = self.cvtCoords(pts_3xn=PtsInCamA_3xn, Ttr=Ttr, flag=self.COOR_CAMA2TOOL)
            PtsInToolB_3xn = self.cvtCoords(pts_3xn=PtsInCamB_3xn, Ttr=Ttr, flag=self.COOR_CAMB2TOOL)
            return PtsInToolA_3xn, PtsInToolB_3xn
        elif self.COOR_IMG2ROB == flag:
            PtsInRobA_3xn = self.cvtCoords(pts_3xn=PtsInCamA_3xn, Ttr=Ttr, flag=self.COOR_CAMA2ROB)
            PtsInRobB_3xn = self.cvtCoords(pts_3xn=PtsInCamB_3xn, Ttr=Ttr, flag=self.COOR_CAMB2ROB)
            return PtsInRobA_3xn, PtsInRobB_3xn

    def projectPts(self, pts_3xn, flag, Ttr=None):
        """
        :param pts_3xn: 3xn array
        :param flag: must be one in (COOR_IMG2CAM, COOR_IMG2TOOL, COOR_IMG2ROB)
        :param Ttr: when flag is relevant to Robot, it must be 4x4 array
        :return:
        """
        assert 3 == pts_3xn.shape[0]
        assert 2 == pts_3xn.ndim
        assert flag in (self.PROJ_ROB2IMGA, self.PROJ_ROB2IMGB, self.PROJ_TOOL2IMGA, self.PROJ_TOOL2IMGB,
                        self.PROJ_CAMA2IMGA, self.PROJ_CAMA2IMGB, self.PROJ_CAMB2IMGA, self.PROJ_CAMB2IMGB)

        if flag in (self.PROJ_ROB2IMGA, self.PROJ_ROB2IMGB):
            assert core.isArray(Ttr, checkSize=(4,4))
            self.__updateTx2x(Ttr=Ttr)

        if flag in (self.PROJ_CAMA2IMGA, self.PROJ_CAMB2IMGB):
            PtsInCam_3xn = pts_3xn.copy()
        else:
            PtsInCam_3xn = self.cvtCoords(pts_3xn=pts_3xn, Ttr=Ttr, flag=flag)

        if flag in (self.PROJ_ROB2IMGA, self.PROJ_TOOL2IMGA, self.PROJ_CAMA2IMGA, self.PROJ_CAMB2IMGA):
            ProjImgPts_2xn = self.__MyStereoVision.projectPts(pts_3xn=PtsInCam_3xn, flag=self.__MyStereoVision.CAM_A, distortFlag=True)
        else:
            ProjImgPts_2xn = self.__MyStereoVision.projectPts(pts_3xn=PtsInCam_3xn, flag=self.__MyStereoVision.CAM_B, distortFlag=True)
        return ProjImgPts_2xn

    def calEpilineError(self, imgPtsA_2xn, imgPtsB_2xn):
        return self.__MyStereoVision.calEpilineError(imgPtsA_2xn=imgPtsA_2xn, imgPtsB_2xn=imgPtsB_2xn)

    def unDistort(self, img, flag):
        return self.__MyStereoVision.unDistort(img, flag)

    def unDistortPts(self, imgPts_2xn, flag):
        return self.__MyStereoVision.unDistortPts(imgPts_2xn, flag)

    def getEpiline(self, imgPts_2xn, flag):
        return self.__MyStereoVision.getEpiline(imgPts_2xn=imgPts_2xn, flag=flag)