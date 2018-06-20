#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '25/12/2015'

import numpy as np
import cv2

from . import core
from .single_vision import SingleVision

__all__ = ['StereoVision']


class StereoVision(object):
    CAM_A = 0L
    CAM_B = 1L
    CAMA_2_CAMB = 2L
    CAMB_2_CAMA = 3L
    def __init__(self, cameraMatrixA, cameraMatrixB, distCoeffsA, distCoeffsB, TcAcB, E, F,
                 imgSizeA=(), imgSizeB=(), resolutionA=None, resolutionB=None):
        object.__init__(self)
        self.__TcAcB = TcAcB
        self.__TcBCA = np.linalg.inv(TcAcB)
        self.__E = E
        self.__F = F
        SingleVisionA = SingleVision(cameraMatrix=cameraMatrixA, distCoeffs=distCoeffsA, imgSize=imgSizeA, resolution=resolutionA)
        SingleVisionB = SingleVision(cameraMatrix=cameraMatrixB, distCoeffs=distCoeffsB, imgSize=imgSizeB, resolution=resolutionB)
        self.__SingleVision = {self.CAM_A: SingleVisionA,
                               self.CAM_B: SingleVisionB}
        self.__CameraMatrix = {self.CAM_A: cameraMatrixA,
                               self.CAM_B: cameraMatrixB}
        self.__DistCoeffs   = {self.CAM_A: distCoeffsA,
                               self.CAM_B: distCoeffsB}
        self.__ImgSize      = {self.CAM_A: imgSizeA,
                               self.CAM_B: imgSizeB}
        self.__Resolution   = {self.CAM_A: resolutionA,
                               self.CAM_B: resolutionB}

    def cvtCoords(self, Pts_3xn, flag):
        assert 3 == Pts_3xn.shape[0]
        assert 2 == Pts_3xn.ndim
        assert flag in (self.CAMA_2_CAMB, self.CAMB_2_CAMA)

        if self.CAMA_2_CAMB == flag:
            return core.projectPts(pts_dxn=Pts_3xn, projectMatrix=self.__TcAcB)
        else:
            return core.projectPts(pts_dxn=Pts_3xn, projectMatrix=self.__TcBCA)

    def get3dPts(self, imgPtsA_2xn, imgPtsB_2xn, unDistortFlag=True, calcReprojErr=False):
        assert imgPtsA_2xn.shape == imgPtsB_2xn.shape
        assert 2 == imgPtsA_2xn.shape[0]

        DistCoeffsDicA = {True : self.__DistCoeffs[self.CAM_A],
                          False: ()}
        DistCoeffsDicB = {True : self.__DistCoeffs[self.CAM_B],
                          False: ()}
        PtsInCamA, RpErrA, RpErrB = \
            core.reconstruct3DPts(imgPtsA_2xn=imgPtsA_2xn, imgPtsB_2xn=imgPtsB_2xn,
                                  cameraMatrixA=self.__CameraMatrix[self.CAM_A], cameraMatrixB=self.__CameraMatrix[self.CAM_B],
                                  distCoeffsA=DistCoeffsDicA[unDistortFlag], distCoeffsB=DistCoeffsDicB[unDistortFlag],
                                  Tx2CamA=np.eye(4), Tx2CamB=self.__TcAcB, calcReprojErr=calcReprojErr)
        PtsInCamB = core.projectPts(pts_dxn=PtsInCamA, projectMatrix=self.__TcAcB)
        return PtsInCamA, PtsInCamB, RpErrA, RpErrB

    def projectPts(self, pts_3xn, flag, distortFlag=True):
        DistCoeffsDicA = {True : self.__DistCoeffs[self.CAM_A],
                          False: ()}
        DistCoeffsDicB = {True : self.__DistCoeffs[self.CAM_B],
                          False: ()}
        DisCoeffs = {self.CAM_A: DistCoeffsDicA,
                     self.CAM_B: DistCoeffsDicB}
        ProjImgPts_2xn = \
            core.projectPtsToImg(pts_3xn=pts_3xn, Tx2Cam=np.eye(4),
                                cameraMatrix=self.__CameraMatrix[flag], distCoeffs=DisCoeffs[flag][distortFlag])
        return ProjImgPts_2xn

    def unDistort(self, img, flag):
        UnDistortImg = self.__SingleVision[flag].unDistortImg(img=img)
        return UnDistortImg

    def unDistortPts(self, imgPts_2xn, flag):
        UnDistortPts_2xn, UnDistortRay_2xn = self.__SingleVision[flag].unDistortPts(imgPts_2xn=imgPts_2xn)
        return UnDistortPts_2xn, UnDistortRay_2xn

    def calEpilineError(self, imgPtsA_2xn, imgPtsB_2xn):
        imgPtsA_2xn_unDistor, _ = self.unDistortPts(imgPts_2xn=imgPtsA_2xn, flag=self.CAM_A)
        imgPtsB_2xn_unDistor, _ = self.unDistortPts(imgPts_2xn=imgPtsB_2xn, flag=self.CAM_B)

        imgPtsA_nx1x2 = imgPtsA_2xn_unDistor.T.reshape(-1, 1, 2)
        imgPtsB_nx3 = core.Homo(imgPtsB_2xn_unDistor).T.reshape(-1, 3)

        CamAPts2CamBLines_nx1x3 = cv2.computeCorrespondEpilines(np.float32(imgPtsA_nx1x2), 1, np.float32(self.__F))
        CamAPts2CamBLines_nx3 = CamAPts2CamBLines_nx1x3.reshape(-1, 3)
        Error_nx3 = imgPtsB_nx3 * CamAPts2CamBLines_nx3
        Error_1xn = np.abs(Error_nx3.sum(axis=1)).reshape(1, -1)
        return Error_1xn

    def getEpiline(self, imgPts_2xn, flag):
        imgPts_2xn_unDistor, _ = self.unDistortPts(imgPts_2xn=imgPts_2xn, flag=flag)
        imgPts_nx1x2 = imgPts_2xn_unDistor.T.reshape(-1, 1, 2)
        Lines_nx1x3 = cv2.computeCorrespondEpilines(np.float32(imgPts_nx1x2), 1, np.float32(self.__F))
        Lines_nx3 = Lines_nx1x3.reshape(-1, 3)
        return Lines_nx3
