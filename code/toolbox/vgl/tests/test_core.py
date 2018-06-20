#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '2017.07.12'
__copyright__ = "Copyright 2017, PI"

import os
import sys
__current_path = os.path.dirname(__file__) or '.'
sys.path.insert(0, os.path.abspath(os.path.join(__current_path, os.path.pardir, os.path.pardir)))

import cv2
import math
import numpy as np
import numpy.testing as testNumpy
from unittest import TestCase

import vgl as VGL


# ------------------------------------
# isArray
# ------------------------------------
class TestIsArray_list(TestCase):
    TypeFunc = list
    GT = False
    def test_Array(self):
        Data = self.TypeFunc([[1, 2], [3, 4]])
        Result = VGL.isArray(Data)
        assert self.GT == Result

class TestIsArray_tuple(TestIsArray_list):
    TypeFunc = tuple
    GT = False

class TestIsArray_array(TestIsArray_list):
    TypeFunc = np.array
    GT = True

# ------------------------------------
# T2Rt & Rt2T
# ------------------------------------
class TestTAndRt(TestCase):
    TypeFunc = np.array
    def test_Rt2T(self):
        R = self.TypeFunc([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
        t = self.TypeFunc([1, 2, 3])
        T_GT = np.array([[1, 0, 0, 1],
                         [0, 1, 0, 2],
                         [0, 0, 1, 3],
                         [0, 0, 0, 1]], dtype=np.float)
        T = VGL.Rt2T(R, t)
        testNumpy.assert_array_equal(T_GT, T)
        assert T.dtype == np.float

    def test_Rt2T_copy(self):
        R = self.TypeFunc([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = self.TypeFunc([1, 2, 3])
        T_GT = np.array([[1, 0, 0, 1],
                         [0, 1, 0, 2],
                         [0, 0, 1, 3],
                         [0, 0, 0, 1]], dtype=np.float)
        T = VGL.Rt2T(R, t)
        if isinstance(R, np.ndarray):
            R[0, 0] = 0
            t[0] = 0
        elif isinstance(R, list):
            R[0][0] = 0
            t[0] = 0
        else:
            pass
        testNumpy.assert_array_equal(T_GT, T)

    def test_T2Rt(self):
        R_GT = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.float)
        t_GT = np.array([1, 2, 3], dtype=np.float).reshape(3, 1)
        T = self.TypeFunc([[1, 0, 0, 1],
                           [0, 1, 0, 2],
                           [0, 0, 1, 3],
                           [0, 0, 0, 1]])
        R, t = VGL.T2Rt(T)
        testNumpy.assert_array_equal(R, R_GT)
        testNumpy.assert_array_equal(t, t_GT)
        assert t.dtype == np.float
        assert R.dtype == np.float

    def test_T2Rt_copy(self):
        R_GT = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.float)
        t_GT = np.array([1, 2, 3], dtype=np.float).reshape(3, 1)
        T = self.TypeFunc([[1, 0, 0, 1],
                           [0, 1, 0, 2],
                           [0, 0, 1, 3],
                           [0, 0, 0, 1]])
        R, t = VGL.T2Rt(T)
        if isinstance(T, np.ndarray):
            T[0, 0] = 0
            T[2, 3] = 0
        elif isinstance(T, list):
            T[0][0] = 0
            T[2][3] = 0
        else:
            pass
        testNumpy.assert_array_equal(R, R_GT)
        testNumpy.assert_array_equal(t, t_GT)
        assert t.dtype == np.float
        assert R.dtype == np.float

    def test_T2Rt_invalid(self):
        T = self.TypeFunc([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.T2Rt(T)

        T = None
        with np.testing.assert_raises(VGL.VGLError):
            VGL.T2Rt(T)

    def test_Rt2T_invalid(self):
        R = self.TypeFunc([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = self.TypeFunc([1, 2])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.Rt2T(R, t)

        R = self.TypeFunc([[1, 0],
                      [0, 1],
                      [0, 0]])
        t = self.TypeFunc([1, 2, 3])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.Rt2T(R, t)

        R = None
        t = self.TypeFunc([1, 2, 3])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.Rt2T(R, t)

        R = self.TypeFunc([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        t = None
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.Rt2T(R, t)

class TestTAndRt_list(TestTAndRt):
    TypeFunc = list

class TestTAndRt_tuple(TestTAndRt):
    TypeFunc = tuple

# ------------------------------------
# T2rt & rt2T
# ------------------------------------
class TestTAndrt(TestCase):
    TypeFunc = np.array
    def test_rt2T(self):
        r = self.TypeFunc([[0],
                           [0],
                           [0.78539816]])
        t = self.TypeFunc([1, 2, 3])
        T_GT = np.array([[ 0.70710678, -0.70710678,  0.        , 1.0],
                         [ 0.70710678,  0.70710678,  0.        , 2.0],
                         [ 0.        ,  0.        ,  1.        , 3.0],
                         [ 0.        ,  0.        ,  0.        , 1.0]], dtype=np.float)
        T = VGL.rt2T(r, t)
        testNumpy.assert_allclose(T_GT, T)
        assert T.dtype == np.float

    def test_rt2T_copy(self):
        r = self.TypeFunc([[0],
                           [0],
                           [0.78539816]])
        t = self.TypeFunc([1, 2, 3])
        T_GT = np.array([[ 0.70710678, -0.70710678,  0.        , 1.0],
                         [ 0.70710678,  0.70710678,  0.        , 2.0],
                         [ 0.        ,  0.        ,  1.        , 3.0],
                         [ 0.        ,  0.        ,  0.        , 1.0]], dtype=np.float)
        T = VGL.rt2T(r, t)
        if isinstance(r, np.ndarray):
            r[0, 0] = 0
            t[0] = 0
        elif isinstance(r, list):
            r[0][0] = 0
            t[0] = 0
        else:
            pass
        testNumpy.assert_allclose(T_GT, T)

    def test_T2rt(self):
        r_GT = np.array([0, 0, 0.78539816]).reshape(3, 1)
        t_GT = np.array([1, 2, 3]).reshape(3, 1)
        T = self.TypeFunc([[ 0.70710678, -0.70710678,  0.        , 1.0],
                           [ 0.70710678,  0.70710678,  0.        , 2.0],
                           [ 0.        ,  0.        ,  1.        , 3.0],
                           [ 0.        ,  0.        ,  0.        , 1.0]])
        r, t = VGL.T2rt(T)
        testNumpy.assert_allclose(r, r_GT)
        testNumpy.assert_allclose(t, t_GT)
        assert t.dtype == np.float
        assert r.dtype == np.float

    def test_T2rt_copy(self):
        r_GT = np.array([0, 0, 0.78539816]).reshape(3, 1)
        t_GT = np.array([1, 2, 3]).reshape(3, 1)
        T = self.TypeFunc([[ 0.70710678, -0.70710678,  0.        , 1.0],
                           [ 0.70710678,  0.70710678,  0.        , 2.0],
                           [ 0.        ,  0.        ,  1.        , 3.0],
                           [ 0.        ,  0.        ,  0.        , 1.0]])
        r, t = VGL.T2rt(T)

        if isinstance(T, np.ndarray):
            T[0, 0] = 0
            T[2, 3] = 0
        elif isinstance(T, list):
            T[0][0] = 0
            T[2][3] = 0
        else:
            pass
        testNumpy.assert_allclose(r, r_GT)
        testNumpy.assert_allclose(t, t_GT)
        assert t.dtype == np.float
        assert r.dtype == np.float

    def test_T2rt_invalid(self):
        T = self.TypeFunc([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 0]])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.T2rt(T)

        T = None
        with np.testing.assert_raises(VGL.VGLError):
            VGL.T2Rt(T)

    def test_rt2T_invalid(self):
        r = self.TypeFunc([[0],
                           [0],
                           [0.78539816]])
        t = self.TypeFunc([1, 2])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.rt2T(r, t)

        r = self.TypeFunc([[1, 0],
                           [0, 1],
                           [0, 0]])
        t = self.TypeFunc([1, 2, 3])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.rt2T(r, t)

        r = None
        t = self.TypeFunc([1, 2, 3])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.rt2T(r, t)

        r = self.TypeFunc([[0],
                           [0],
                           [0.78539816]])
        t = None
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.rt2T(r, t)

class TestTAndrt_list(TestTAndrt):
    TypeFunc = list

class TestTAndrt_tuple(TestTAndrt):
    TypeFunc = tuple

# ------------------------------------
# getRx & getRy & getRz & Euler2R
# ------------------------------------
class TestRxyz(TestCase):
    def test_getRx(self):
        Angle_rad = np.pi / 2
        R_GT = np.array([[1.0, 0.0,  0.0],
                         [0.0, 0.0, -1.0],
                         [0.0, 1.0,  0.0]])
        R = VGL.getRx(Angle_rad)
        testNumpy.assert_allclose(R, R_GT, atol=1e-7)

    def test_getRy(self):
        Angle_rad = np.pi / 2
        R_GT = np.array([[ 0.0, 0.0, 1.0],
                         [ 0.0, 1.0, 0.0],
                         [-1.0, 0.0, 0.0]])
        R = VGL.getRy(Angle_rad)
        testNumpy.assert_allclose(R, R_GT, atol=1e-7)

    def test_getRz(self):
        Angle_rad = np.pi / 2
        R_GT = np.array([[0.0, -1.0, 0.0],
                         [1.0,  0.0, 0.0],
                         [0.0,  0.0, 1.0]])
        R = VGL.getRz(Angle_rad)
        testNumpy.assert_allclose(R, R_GT, atol=1e-7)

class TestEuler2ROrder_xzx(TestCase):
    Order = 'xzx'
    mode = 'intrinsic'
    def test_Euler2R(self):
        Alpha, Beta, Gamma = [np.pi/10, np.pi/7, np.pi/2]
        cosValue = map(math.cos, [Alpha, Beta, Gamma])
        sinValue = map(math.sin, [Alpha, Beta, Gamma])
        if self.mode == 'intrinsic':
            R_GT = self.calcR_intrinsic(sinValue, cosValue)
        elif self.mode == 'extrinsic':
            R_GT = self.calcR_intrinsic(sinValue, cosValue)
        else:
            raise ValueError, 'mode error'
        R = VGL.Euler2R(Alpha, Beta, Gamma, self.Order, self.mode)
        testNumpy.assert_allclose(R, R_GT)

    def calcR_intrinsic(self, sinValueList, cosValue):
        s1, s2, s3 = sinValueList
        c1, c2, c3 = cosValue
        if self.Order == 'xzx':
            R = np.array([[   c2,           -c3*s2,             s2*s3],
                          [c1*s2, c1*c2*c3 - s1*s3, -c3*s1 - c1*c2*s3],
                          [s1*s2, c1*s3 + c2*c3*s1,  c1*c3 - c2*s1*s3]])
        elif self.Order == 'zyx':
            R = np.array([[c1*c2, c1*s2*s3 - c3*s1,  s1*s3 + c1*c3*s2],
                          [c2*s1, c1*c3 + s1*s2*s3,  c3*s1*s2 - c1*s3],
                          [  -s2,            c2*s3,             c2*c3]])
        elif self.Order == 'xyx':
            R = np.array([[    c2,            s2*s3,             s2*c3],
                          [ s1*s2, c1*c3 - c2*s1*s3, -c1*s3 - s1*c2*c3],
                          [-c1*s2, s1*c3 + c2*s3*c1,  c1*c2*c3 - s1*s3]])
        elif self.Order == 'yxz':
            R = np.array([[ c1*c3 + s1*s2*s3, c3*s1*s2 - c1*s3,  c2*s1],
                          [            c2*s3,            c2*c3,    -s2],
                          [ c1*s2*s3 - c3*s1, c1*c3*s2 + s1*s3,  c1*c2]])
        else:
            raise ValueError
        return R

    def test_IntrinsicEQExtrinsic(self):
        Alpha, Beta, Gamma = [np.pi/10, np.pi/7, np.pi/2]
        R_i = VGL.Euler2R(Alpha, Beta, Gamma, self.Order, mode='intrinsic')
        R_e = VGL.Euler2R(Gamma, Beta, Alpha, self.Order[::-1], mode='extrinsic')
        Coord = np.eye(3)
        RotatedByI = np.dot(R_i, Coord)
        RotatedByE = np.dot(Coord, R_e)
        testNumpy.assert_allclose(RotatedByE, RotatedByI)

class TestEuler2ROrder_zyx(TestEuler2ROrder_xzx):
    Order = 'zyx'

class TestEuler2ROrder_xyx(TestEuler2ROrder_xzx):
    Order = 'xyx'

class TestEuler2ROrder_yxz(TestEuler2ROrder_xzx):
    Order = 'yxz'

# ------------------------------------
# R2Euler
# ------------------------------------
class TestR2Euler(TestCase):
    TypeFunc = np.array
    def test_R2Euler(self):
        R = self.TypeFunc([[1,  0,  0],
                           [0, -1,  0],
                           [0,  0, -1]])
        a, b, c = VGL.R2Euler_zyx(R)
        testNumpy.assert_allclose([a, b, c], [0, 0, np.pi])

    def test_R2EulerErr(self):
        R = self.TypeFunc([[ 1,  0,  0],
                           [ 0,  1,  0],
                           [ 0,  0, -1]])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.R2Euler_zyx(R)

    def test_R2EulerHandleSingular(self):
        GT = [0, 90, 30]
        Spe = math.sqrt(3) / 2.0
        R = self.TypeFunc([[  0,  0.5,  Spe],
                           [  0,  Spe, -0.5],
                           [ -1,    0,    0]])
        z, y, x = VGL.R2Euler_zyx(R)
        testNumpy.assert_allclose([z, y, x], np.deg2rad(GT))

class TestR2Euler_list(TestR2Euler):
    TypeFunc = list

class TestR2Euler_tuple(TestR2Euler):
    TypeFunc = tuple

# ------------------------------------
# T2Pose & Pose2T
# ------------------------------------
class TestTAndPose(TestCase):
    TypeFunc = np.array
    def test_Pose2T(self):
        Pose = self.TypeFunc([10, 20, 30, 90, 90, 90])
        T_GT = np.array([[ 0, 0, 1, 10],
                         [ 0, 1, 0, 20],
                         [-1, 0, 0, 30],
                         [ 0, 0, 0,  1]], dtype=np.float)
        T = VGL.Pose2T(Pose)
        testNumpy.assert_allclose(T_GT, T, atol=1e-7)

    def test_T2Pose(self):
        Pose_GT = np.array([[10], [20], [30], [30], [0], [90]])
        Spe = 0.8660254037844
        T = np.array([[ Spe, 0.0,  0.5, 10],
                      [ 0.5, 0.0, -Spe, 20],
                      [ 0.0,   1,  0.0, 30],
                      [ 0.0,   0,    0,  1]], dtype=np.float)
        Pose = VGL.T2Pose(T)
        testNumpy.assert_allclose(Pose_GT, Pose, atol=1e-7)

class TestTAndPose_list(TestTAndPose):
    TypeFunc = list

class TestTAndPose_tuple(TestTAndPose):
    TypeFunc = tuple

# ------------------------------------
# Homo & unHomo
# ------------------------------------
class TestHomoAndUnHomo(TestCase):
    TypeFunc = np.array
    def test_Homo(self):
        Point_dxn = self.TypeFunc([[1, 2],
                                   [1, 2]])
        Homo_GT = np.array([[1, 2],
                            [1, 2],
                            [1, 1]])
        HomoPoint = VGL.Homo(Point_dxn)
        testNumpy.assert_allclose(Homo_GT, HomoPoint)

    def test_unHomo(self):
        unHomo_GT = self.TypeFunc([[1, 1],
                                   [1, 1]])
        Point_dxn = np.array([[1, 2],
                              [1, 2],
                              [1, 2]])
        unHomoPoint = VGL.unHomo(Point_dxn)
        testNumpy.assert_allclose(unHomoPoint, unHomo_GT)

    def test_HomoCopy(self):
        Point_dxn = self.TypeFunc([[1, 2],
                                   [1, 2]])
        Homo_GT = np.array([[1, 2],
                            [1, 2],
                            [1, 1]])
        HomoPoint = VGL.Homo(Point_dxn)
        if isinstance(Point_dxn, np.ndarray):
            Point_dxn[0, 0] = 5
        elif isinstance(Point_dxn, list):
            Point_dxn[0][0] = 5
        else:
            pass
        testNumpy.assert_allclose(Homo_GT, HomoPoint)

    def test_unHomoCopy(self):
        unHomo_GT = self.TypeFunc([[1, 2],
                                   [1, 2]])
        Point_dxn = np.array([[1, 2],
                            [1, 2],
                            [1, 1]])
        unHomoPoint = VGL.unHomo(Point_dxn)
        if isinstance(Point_dxn, np.ndarray):
            Point_dxn[0, 0] = 5
        elif isinstance(Point_dxn, list):
            Point_dxn[0][0] = 5
        else:
            pass
        testNumpy.assert_allclose(unHomoPoint, unHomo_GT)

    def test_Homo_invalid(self):
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.Homo([1, 2, 3])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.Homo(None)

    def test_unHomo_invalid(self):
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.unHomo([1, 2, 3])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.unHomo(None)

class TestHomoAndUnHomo_tuple(TestHomoAndUnHomo):
    TypeFunc = tuple

class TestHomoAndUnHomo_list(TestHomoAndUnHomo):
    TypeFunc = list

# -----------------------------------------
# computeVectorAngle_rad & computeRotateVec
# -----------------------------------------
class TestVectorRotateAndAngle(TestCase):
    TypeFunc = np.array
    def test_computeVectorAngle2D(self):
        Vec0 = self.TypeFunc([0, 1])
        Vec1 = self.TypeFunc([0.5, 0.5])
        Angle_rad = VGL.computeVectorAngle_rad(Vec0, Vec1)
        testNumpy.assert_allclose(Angle_rad, np.pi/4)

    def test_computeRotateVec(self):
        Vec0 = self.TypeFunc([0, 0, 5])
        Vec1 = self.TypeFunc([3, 0, 0])
        GT = np.array([0, np.pi/2, 0]).reshape(3, 1)
        AxisAngle = VGL.computeRotateVec(Vec0, Vec1)
        testNumpy.assert_allclose(AxisAngle, GT)

    def test_computeRotateVec_180(self):
        Vec0 = self.TypeFunc([2, 2, 2])
        Vec1 = self.TypeFunc([-2, -2, -2])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.computeRotateVec(Vec0, Vec1)

    def test_computeRotateVec_180RotateAxisInvalid(self):
        Vec0 = self.TypeFunc([[0], [0], [1]])
        Vec1 = self.TypeFunc([[0], [0], [-1]])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.computeRotateVec(Vec0, Vec1)

    def test_computeRotateVec_0(self):
        Vec0 = self.TypeFunc([[0], [0], [1]])
        Vec1 = self.TypeFunc([[0], [0], [1]])
        GT = np.array([[0], [0], [0]])
        AxisAngle = VGL.computeRotateVec(Vec0, Vec1)
        testNumpy.assert_allclose(GT, AxisAngle)

    def test_computeRotateVec_180RotateAxisValid(self):
        Vec0 = self.TypeFunc([[1], [0], [0]])
        Vec1 = self.TypeFunc([[-1], [0], [0]])
        AxisAngle = VGL.computeRotateVec(Vec0, Vec1, rotateAxis=[0, 0, 1])
        R = cv2.Rodrigues(AxisAngle)[0]
        NewVec = VGL.projectPts(pts_dxn=Vec0, projectMatrix=R)
        testNumpy.assert_allclose(Vec1, NewVec, atol=1e-7)

    def test_rotateVec0ToVec1(self):
        Vec0 = self.TypeFunc([0.5, 0.7, 5.6])
        Vec1 = self.TypeFunc([3.8, 1.5, 3.5])
        AxisAngle = VGL.computeRotateVec(Vec0, Vec1)
        R = cv2.Rodrigues(AxisAngle)[0]
        Rotated = np.dot(R, Vec0)
        RotatedNormed = Rotated / np.linalg.norm(Vec0)
        Vec1Normed = Vec1 / np.linalg.norm(Vec1)
        testNumpy.assert_allclose(Vec1Normed, RotatedNormed)

class TestVectorRotateAndAngle_list(TestVectorRotateAndAngle):
    TypeFunc = list

class TestVectorRotateAndAngle_tuple(TestVectorRotateAndAngle):
    TypeFunc = tuple

# ----------------------------------
# distBW2lines
# ----------------------------------
class TestLineDisAndTransform(TestCase):
    TypeFunc = np.array
    def test_distBW2Line2D_Parallel(self):
        Line0 = self.TypeFunc([[2, 3],
                               [2, 3]])
        Line1 = self.TypeFunc([[1, 20],
                               [0, 0]])
        GTDist = 0.0
        GT_Intersect0 = np.array([0, 0]).reshape(2, 1)
        GT_Intersect1 = np.array([0, 0]).reshape(2, 1)
        Dist, Intersect0, Intersect1 = VGL.distBW2Lines(Line0, Line1)
        testNumpy.assert_allclose(Dist, GTDist)
        testNumpy.assert_allclose(Intersect0, GT_Intersect0)
        testNumpy.assert_allclose(Intersect1, GT_Intersect1)

    def test_distBW2Line2D_Intersect(self):
        Line0 = self.TypeFunc([[-2, -3],
                               [-2, -3]])
        Line1 = self.TypeFunc([[1, 2],
                               [0, 1]])
        GTDist = math.sqrt(2) / 2
        GT_Intersect0 = np.array([-2, -2]).reshape(2, 1)
        GT_Intersect1 = np.array([-1.5, -2.5]).reshape(2, 1)
        Dist, Intersect0, Intersect1 = VGL.distBW2Lines(Line0, Line1)
        testNumpy.assert_allclose(Dist, GTDist)
        testNumpy.assert_allclose(Intersect0, GT_Intersect0)
        testNumpy.assert_allclose(Intersect1, GT_Intersect1)


    def test_distBw2Line3D_Intersect(self):
        Line0 = self.TypeFunc([[2, 3],
                               [2, 3],
                               [2, 3]])
        Line1 = self.TypeFunc([[12,  19],
                               [15,  25],
                               [ 0, -5]])
        GTDist = 0.0
        GT_Intersect0 = np.array([5, 5, 5]).reshape(3, 1)
        GT_Intersect1 = np.array([5, 5, 5]).reshape(3, 1)
        Dist, Intersect0, Intersect1 = VGL.distBW2Lines(Line0, Line1)
        testNumpy.assert_allclose(Dist, GTDist)
        testNumpy.assert_allclose(Intersect0, GT_Intersect0)
        testNumpy.assert_allclose(Intersect1, GT_Intersect1)

    def test_distBW2Line3D_Parallel(self):
        Line0 = self.TypeFunc([[2, 3],
                               [2, 3],
                               [2, 3]])
        Line1 = self.TypeFunc([[0, 1],
                               [0, 1],
                               [8, 9]])
        GTDist = 8 / math.sqrt(3) * math.sqrt(2)
        GT_Intersect0 = np.array([2, 2, 2]).reshape(3, 1)
        GT_Intersect1 = np.array([-2.0/3.0, -2.0/3.0, 22.0/3.0]).reshape(3, 1)
        Dist, Intersect0, Intersect1 = VGL.distBW2Lines(Line0, Line1)
        testNumpy.assert_allclose(Dist, GTDist)
        testNumpy.assert_allclose(Intersect0, GT_Intersect0)
        testNumpy.assert_allclose(Intersect1, GT_Intersect1)

    def test_distBW2Line3D_Normal(self):
        Line0 = self.TypeFunc([[2, 3],
                               [2, 3],
                               [0, 0]])
        Line1 = self.TypeFunc([[5, 9],
                               [0, 0],
                               [5, 5]])
        GTDist = 5
        GT_Intersect0 = np.array([0, 0, 0]).reshape(3, 1)
        GT_Intersect1 = np.array([0, 0, 5]).reshape(3, 1)
        Dist, Intersect0, Intersect1 = VGL.distBW2Lines(Line0, Line1)
        testNumpy.assert_allclose(Dist, GTDist)
        testNumpy.assert_allclose(Intersect0, GT_Intersect0)
        testNumpy.assert_allclose(Intersect1, GT_Intersect1)

class TestLineDisAndTransform_list(TestLineDisAndTransform):
    TypeFunc = list

class TestLineDisAndTransform_tuple(TestLineDisAndTransform):
    TypeFunc = tuple

# ----------------------------------
# getTransformWith2LineSegment
# ----------------------------------
class TestTransformWith2LineSegment(TestCase):
    TypeFunc = np.array
    def testTransform(self):
        Line0 = self.TypeFunc([[3, 5],
                               [3, 5],
                               [3, 5]])
        Line1 = self.TypeFunc([[ 3,  5],
                               [ 3,  5],
                               [-3, -5]])
        T = VGL.getTransformWith2LineSegment(Line0, Line1)
        TransLine0 = VGL.projectPts(Line0, T)
        testNumpy.assert_allclose(TransLine0, Line1)

    def testTransform_NotAlign(self):
        Line0 = self.TypeFunc([[3, 6],
                               [3, 6],
                               [3, 6]])
        Line1 = self.TypeFunc([[ 3,  5],
                               [ 3,  5],
                               [-3, -5]])
        GTTransLine = np.array([[ 3,  6],
                                [ 3,  6],
                                [-3, -6]])
        T = VGL.getTransformWith2LineSegment(Line0, Line1)
        TransLine0 = VGL.projectPts(Line0, T)
        testNumpy.assert_allclose(TransLine0, GTTransLine)

class TestTransformWith2LineSegment_list(TestTransformWith2LineSegment):
    TypeFunc = list

class TestTransformWith2LineSegment_tuple(TestTransformWith2LineSegment):
    TypeFunc = tuple

# ----------------------------------
# projectPts
# ----------------------------------
class TestProjectPts(TestCase):
    TypeFunc = np.array
    def test_projectPts_SameD(self):
        Pts = self.TypeFunc([[0, 2],
                             [1, 3]])
        ProjectMatrix = self.TypeFunc([[0, -1],
                                       [1,  0]])
        GT = np.array([[-1, -3],
                       [ 0,  2]])
        ProjectedPts = VGL.projectPts(pts_dxn=Pts, projectMatrix=ProjectMatrix)
        testNumpy.assert_allclose(GT, ProjectedPts)

    def test_projectPts_D1(self):
        Pts = self.TypeFunc([[0, 2],
                             [1, 3]])
        ProjectMatrix = self.TypeFunc([[0, -1, 10],
                                       [1,  0, 20],
                                       [0,  0,  1]])
        GT = np.array([[ 9,  7],
                       [20, 22]])
        ProjectedPts = VGL.projectPts(pts_dxn=Pts, projectMatrix=ProjectMatrix)
        testNumpy.assert_allclose(GT, ProjectedPts)

    def test_projectPts_MatrixErr(self):
        Pts = self.TypeFunc([[0, 2],
                             [1, 3]])
        ProjectMatrix = self.TypeFunc([[0, -1, 10],
                                       [1,  0, 20]])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.projectPts(pts_dxn=Pts, projectMatrix=ProjectMatrix)

    def test_projectPts_NotMatch(self):
        Pts = self.TypeFunc([[0, 2],
                             [1, 3]])
        ProjectMatrix = self.TypeFunc([[0, -1,  0, 11],
                                       [1,  0,  0, 22],
                                       [0,  0,  1, 33],
                                       [0,  0,  0,  1]])
        with testNumpy.assert_raises(VGL.VGLError):
            VGL.projectPts(pts_dxn=Pts, projectMatrix=ProjectMatrix)

class TestProjectPts_list(TestProjectPts):
    TypeFunc = list

class TestProjectPts_tuple(TestProjectPts):
    TypeFunc = tuple

# ----------------------------------
# projectPtsToImg
# ----------------------------------
class TestProjectPtsToImg(TestCase):
    TypeFunc = np.array
    def test_projectPtsToImg(self):
        Pts_3xn = self.TypeFunc([[1, 2],
                                 [2, 3],
                                 [4, 5]])
        Tx2Cam = self.TypeFunc([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        CameraMatrix = self.TypeFunc([[100,   0, 320],
                                      [  0, 100, 240],
                                      [  0,   0,   1]])
        Dist = self.TypeFunc([])
        GT = np.array([[345, 360],
                       [290, 300]])
        ImgPts_2xn = VGL.projectPtsToImg(Pts_3xn, Tx2Cam, CameraMatrix, Dist)
        testNumpy.assert_allclose(GT, ImgPts_2xn)

class TestProjectPtsToImg_list(TestProjectPtsToImg):
    TypeFunc = list

class TestProjectPtsToImg_tuple(TestProjectPtsToImg):
    TypeFunc = tuple

# ----------------------------------
# unDistortPts & unDistortImg
# ----------------------------------
class TestUnDistortImgAndPts(TestCase):
    TypeFunc = np.array
    def test_unDistortImg(self):
        Img = np.zeros((30, 30), dtype=np.uint8)
        Img[10:20, 10:20] = 255
        CameraMatrix = self.TypeFunc([[2, 0, 15],
                                      [0, 2, 15],
                                      [0, 0,  1]])
        Dist = self.TypeFunc([0.1, 0.2, 0.3, 0.5])
        NewImg = VGL.unDistortImg(Img, CameraMatrix, Dist)

    def test_unDistortImgNoDist(self):
        Img = np.zeros((30, 30), dtype=np.uint8)
        Img[10:20, 10:20] = 1
        CameraMatrix = self.TypeFunc([[2, 0, 15],
                                      [0, 2, 15],
                                      [0, 0,  1]])
        Dist = self.TypeFunc([])
        NewImg = VGL.unDistortImg(Img, CameraMatrix, Dist)
        testNumpy.assert_allclose(NewImg, Img)

    def test_unDistortImgOpt(self):
        ImgSize = self.TypeFunc([30, 30])
        Img = np.zeros((30, 30), dtype=np.uint8)
        Img[10:20, 10:20] = 255
        CameraMatrix = self.TypeFunc([[2, 0, 15],
                                      [0, 2, 15],
                                      [0, 0,  1]])
        Dist = self.TypeFunc([0.1, 0.2, 0.3, 0.5])
        NewImg, NewCameraMatrix, Roi_xywh = VGL.unDistortImgOptimal(Img, CameraMatrix, Dist, imgSize=ImgSize, alpha=1)

    def test_unDistortImgOptNoDist(self):
        ImgSize = self.TypeFunc([30, 30])
        Img = np.zeros((30, 30), dtype=np.uint8)
        Img[10:20, 10:20] = 2
        CameraMatrix = self.TypeFunc([[2, 0, 15],
                                      [0, 2, 15],
                                      [0, 0,  1]])
        Dist = self.TypeFunc([])
        NewImg, NewCameraMatrix, Roi_xywh = VGL.unDistortImgOptimal(Img, CameraMatrix, Dist, imgSize=ImgSize)

    def test_unDistortPts(self):
        Points = []
        for i in xrange(10, 20):
            for j in xrange(10, 20):
                Points.append([i, j])
        ImgPts_2xn = np.array(Points).T.reshape(2, -1)
        Img = np.zeros((30, 30), dtype=np.uint8)
        Img[10:20, 10:20] = 255
        CameraMatrix = self.TypeFunc([[2, 0, 15],
                                      [0, 2, 15],
                                      [0, 0,  1]])
        Dist = self.TypeFunc([0.01, 0.01, 0.01, 0.01])
        UnDistImgPts_2xn, Rays_2xn = VGL.unDistortPts(ImgPts_2xn, CameraMatrix, Dist)

    def test_unDistortPtsNoDist(self):
        Points = []
        for i in xrange(10, 20):
            for j in xrange(10, 20):
                Points.append([i, j])
        ImgPts_2xn = np.array(Points).T.reshape(2, -1)
        Img = np.zeros((30, 30), dtype=np.uint8)
        Img[10:20, 10:20] = 255
        CameraMatrix = self.TypeFunc([[2, 0, 15],
                                      [0, 2, 15],
                                      [0, 0,  1]])
        Dist = self.TypeFunc([])
        UnDistImgPts_2xn, Rays_2xn = VGL.unDistortPts(ImgPts_2xn, CameraMatrix, Dist)

class TestUnDistortImgAndPts_list(TestUnDistortImgAndPts):
    TypeFunc = list
#
class TestUnDistortImgAndPts_tuple(TestUnDistortImgAndPts):
    TypeFunc = tuple

# ----------------------------------
# reconstruct3DPts
# ----------------------------------
class TestReconstruct3D(TestCase):
    TypeFunc = np.array
    def test_reconstruct3D(self):
        ImgPtsA = self.TypeFunc([[320],
                                 [240]])
        ImgPtsB = self.TypeFunc([[320],
                                 [240]])
        CameraMatrixA = self.TypeFunc([[100, 0, 320],
                                       [0, 100, 240],
                                       [0, 0,   1]])
        CameraMatrixB = self.TypeFunc([[100, 0, 320],
                                       [0, 100, 240],
                                       [0, 0,   1]])
        Tx2CamA = self.TypeFunc([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])

        Deg45 = np.sqrt(2) / 2
        Tx2CamB = self.TypeFunc([[    1,     0,      0,  0],
                                 [    0, Deg45, -Deg45, 10],
                                 [    0, Deg45,  Deg45,  0],
                                 [    0,      0,     0,  1]])
        DistA = self.TypeFunc([])
        DistB = self.TypeFunc([])
        GTPInX = np.array([0, 0, np.sqrt(2)*10]).reshape(3, 1)
        GTPInB = np.array([0, 0, 10]).reshape(3, 1)
        PtsInX_3xn, ErrA, ErrB = \
            VGL.reconstruct3DPts(imgPtsA_2xn=ImgPtsA, imgPtsB_2xn=ImgPtsB, cameraMatrixA=CameraMatrixA,
                                 cameraMatrixB=CameraMatrixB, distCoeffsA=DistA, distCoeffsB=DistB,
                                 Tx2CamA=Tx2CamA, Tx2CamB=Tx2CamB, calcReprojErr=True)
        PtsInB_3xn = VGL.projectPts(PtsInX_3xn, Tx2CamB)
        testNumpy.assert_allclose(PtsInX_3xn, GTPInX, atol=1e-7)
        testNumpy.assert_allclose(PtsInB_3xn, GTPInB, atol=1e-7)
        testNumpy.assert_allclose(ErrA, 0, atol=1e-7)
        testNumpy.assert_allclose(ErrB, 0, atol=1e-7)

class TestReconstruct3D_list(TestReconstruct3D):
    TypeFunc = list

class TestReconstruct3D_tuple(TestReconstruct3D):
    TypeFunc = tuple


if __name__ == '__main__':
    os.chdir(__current_path)
    VGL.test(doctests=True)
    # print VGL.test(doctests=False)
