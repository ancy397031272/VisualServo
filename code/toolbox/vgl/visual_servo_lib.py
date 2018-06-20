#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__version__ = '1.0'
__date__ = '23/08/2016'

import copy
import numpy as np
from . import core

__all__ = ['VisualServoImageBase', 'VisualServoPositionBase']


class VisualServo(object):
    @classmethod
    def _calImageJacobian(cls, intrinsic, x, y, z):
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        u = x - cx
        v = y - cy
        J = np.array([[fx/z,    0, -u/z,        -(u*v)/fy, (fx**2+u**2)/fx, -(fx*v)/fy],
                      [   0, fy/z, -v/z, -(fy**2+v**2)/fy,        (u*v)/fx, (fy*u)/fx]])
        return J

    @classmethod
    def filterMoveDimension(cls, JMatrix, mask=(1,1,1,1,1,1)):
        assert isinstance(JMatrix, np.ndarray)
        assert JMatrix.shape[1] == len(mask), 'The column of JMatrix == len(Mask)'

        NonZero = np.nonzero(mask)
        FilterMatrix = np.zeros_like(JMatrix)
        FilterMatrix[:, NonZero] = JMatrix[:, NonZero]
        return FilterMatrix

    @classmethod
    def skewMatrix(cls, t):
        '''
        sk(p) = [0, -z, y
                 z, 0, -x
                 -y, x, 0]
        :param t:
        :return:
        '''
        assert t.shape == (3,1), 't must be a 3 by 1 vector'
        sk = np.array([[      0, -t[2,0],  t[1,0]],
                       [ t[2,0],       0, -t[0,0]],
                       [-t[1,0],  t[0,0],       0]])
        return sk

    @classmethod
    def _velocityTranMatrix(cls, Tx2x):
        R_cr = Tx2x[0:3, 0:3]
        t_cr = Tx2x[0:3, 3].reshape(3,1)
        sk_t = cls.skewMatrix(t_cr)

        VMatrix_c2r = np.hstack((R_cr, np.dot(sk_t, R_cr)))
        temp = np.hstack((np.zeros((3,3)), R_cr))
        VMatrix_c2r = np.vstack((VMatrix_c2r, temp))

        return VMatrix_c2r

    @classmethod
    def transVelocity(cls, Tx2x, velocity):
        return cls._velocityTranMatrix(Tx2x=Tx2x).dot(velocity)

    # @classmethod
    # def velocityPose2T(cls, velocityPose):
    #     DeltaPose = np.zeros((6,1)).astype('float32')
    #     DeltaPose[:3] = velocityPose[:3]
    #     DeltaPose[3]  = velocityPose[5]/math.pi*180
    #     DeltaPose[4]  = velocityPose[4]/math.pi*180
    #     DeltaPose[5]  = velocityPose[3]/math.pi*180
    #     T = core.Pose2T(pose=DeltaPose)
    #     return T

    @classmethod
    def velocityPose2Move(cls, velocityPose, iter=1):
        assert isinstance(velocityPose, np.ndarray)
        assert velocityPose.shape == (6,1)

        VelocityPoseStep = velocityPose / float(iter)
        TStep = np.matrix(cls._velocityPose2T(velocityPose=VelocityPoseStep))
        T = TStep ** iter
        return np.array(T)

    @classmethod
    def _velocityPose2T(cls, velocityPose):
        assert velocityPose.shape == (6,1)

        v = velocityPose[0:3]
        Omega = velocityPose[3:6]
        sk_Omega = cls.skewMatrix(Omega)
        T_rot = np.identity(4)
        T_rot[0:3, 0:3] += sk_Omega
        T_rot[0:3, 3] += v.reshape(-1,)
        return T_rot


class VisualServoPositionBase(VisualServo):

    @classmethod
    def matchPP(cls, objPts_3xn, tarPts_3xn, Tx2x, mask=(1,1,1,1,1,1)):
        """
        :param objPts_3xn:
        :param tarPts_3xn:
        :param Tx2x:
        :param mask: x,y,z,wx,wy,wz
        :return:
        """
        ObjPts_3xn = core.projectPts(pts_dxn=objPts_3xn, projectMatrix=Tx2x)
        TarPts_3xn = core.projectPts(pts_dxn=tarPts_3xn, projectMatrix=Tx2x)

        ObjNum = ObjPts_3xn.shape[1]
        A = None
        for i in xrange(ObjNum):
            Temp = np.hstack((np.eye(3), -cls.skewMatrix(ObjPts_3xn[:,i].reshape(3,1))))
            if A is None:
                A = Temp.copy()
            else:
                A = np.vstack((A, Temp))
        A = cls.filterMoveDimension(A, mask=mask)

        U3 = (ObjPts_3xn - TarPts_3xn).T.reshape((-1,1))
        U6 = np.linalg.pinv(A).dot(U3)
        return U6

    @classmethod
    def matchPL(cls):
        pass


class VisualServoImageBase(VisualServo):

    @classmethod
    def matchPP(cls, intrinsic, objPts_2xn, tarPts_2xn, z_1xn, Tc2x, mask=(1,1,1,1,1,1)):
        assert len(objPts_2xn.shape)==2 and objPts_2xn.shape[0] == 2, 'ObjPts is not a 2 by N matrix'
        assert len(objPts_2xn.shape)==2 and objPts_2xn.shape[0] == 2, 'TarPts is not a 2 by N matrix'
        assert isinstance(z_1xn, np.ndarray), 'z_1byN must be ndarray'
        assert objPts_2xn.shape[1] == tarPts_2xn.shape[1] and objPts_2xn.shape[1] == z_1xn.shape[1], 'N must be the same'
        assert objPts_2xn.shape[1] > 0, 'N must bigger than 0'

        velocity_matrix_c2r = cls._velocityTranMatrix(Tx2x=Tc2x)
        J = None
        ef = None
        for i in range(0, objPts_2xn.shape[1]):
            Jtemp = cls._calImageJacobian(intrinsic=intrinsic, x=objPts_2xn[0,i], y=objPts_2xn[1, i], z=z_1xn[0, i])
            Jtemp = np.dot(Jtemp, np.linalg.inv(velocity_matrix_c2r))
            eftemp = (objPts_2xn[:, i] - tarPts_2xn[:, i]).reshape(2,1)
            if i == 0:
                J  = copy.deepcopy(Jtemp)
                ef = copy.deepcopy(eftemp)
            else:
                J  = np.vstack((J, Jtemp))
                ef = np.vstack((ef, eftemp))
        J = cls.filterMoveDimension(JMatrix=J, mask=mask)
        return J, ef

    @classmethod
    def matchPPs(cls, intrinsic_list, objPts_2xn_list, tarPts_2xn_list, z_1xn_list, Tc2x_list, mask=(1,1,1,1,1,1)):
        assert len(intrinsic_list) == len(objPts_2xn_list) == len(tarPts_2xn_list) == len(z_1xn_list) == len(Tc2x_list)

        Num = len(intrinsic_list)
        J, Ef = None, None
        for i in xrange(Num):
            TempJ, TempEf = \
                cls.matchPP(intrinsic=intrinsic_list[i],
                            objPts_2xn=objPts_2xn_list[i],
                            tarPts_2xn=tarPts_2xn_list[i],
                            z_1xn=z_1xn_list[i],
                            Tc2x=Tc2x_list[i],
                            mask=mask)
            if 0 == i:
                J = TempJ.copy()
                Ef = TempEf.copy()
            else:
                J = np.vstack((J, TempJ))
                Ef = np.vstack((Ef, TempEf))
        Velocity = -np.linalg.pinv(J).dot(Ef)
        return Velocity,Ef

    @classmethod
    def matchPL(cls):
        pass