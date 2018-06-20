#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'lh'
__version__ = '1.0'
__date__ = '30/11/2015'

import numpy as np
from . import core

__all__ = ['SingleVision']


class SingleVision(object):
    def __init__(self, cameraMatrix, distCoeffs, imgSize=(), resolution=None):
        """
        :param cameraMatrix: intrinsic matrix
        :param distCoeffs: distortion coeffs
        :param imgSize: image size (col x row)
        :param resolution:
        :return:
        """
        object.__init__(self)
        self.__CameraMatrix = cameraMatrix
        self.__DistCoeffs = distCoeffs
        self.__ImgSize = imgSize
        self.__Resolution = resolution

    def get3DPts(self, imgPts_2xn, z_mm, unDistortFlag=True):
        assert imgPts_2xn.ndim == 2,        "imgPts must be 2xn"
        assert imgPts_2xn.shape[0] == 2,    "imgPts must be 2xn"
        DistCoeffDic = {True : self.__DistCoeffs,
                        False: ()}
        UnDistortPts_2xn, UnDistortRay_2xn = core.unDistortPts(imgPts_2xn=imgPts_2xn,
                                                              cameraMatrix=self.__CameraMatrix,
                                                              distCoeffs=DistCoeffDic[unDistortFlag])
        Pts3D_3xn = core.Homo(UnDistortRay_2xn) * z_mm
        return Pts3D_3xn

    def projectPts2Img(self, pts_3xn, distortFlag=True):
        assert pts_3xn.ndim == 2,        "pts must be 3xn"
        assert pts_3xn.shape[0] == 3,    "pts must be 3xn"
        DistCoeffDic = {True : self.__DistCoeffs,
                        False: ()}
        ImgPts_2xn = core.projectPtsToImg(pts_3xn=pts_3xn, Tx2Cam=np.eye(4),
                                         cameraMatrix=self.__CameraMatrix,
                                         distCoeffs=DistCoeffDic[distortFlag])
        return ImgPts_2xn

    def unDistortImg(self, img):
        UnDistortImg = core.unDistortImg(img=img, cameraMatrix=self.__CameraMatrix, distCoeffs=self.__DistCoeffs)
        return UnDistortImg

    def unDistortPts(self, imgPts_2xn):
        assert imgPts_2xn.ndim == 2,        "imgPts must be 2xn"
        assert imgPts_2xn.shape[0] == 2,    "imgPts must be 2xn"
        UnDistortPts_2xn, UnDistortRay_2xn = core.unDistortPts(imgPts_2xn=imgPts_2xn,
                                                              cameraMatrix=self.__CameraMatrix,
                                                              distCoeffs=self.__DistCoeffs)
        return UnDistortPts_2xn, UnDistortRay_2xn
