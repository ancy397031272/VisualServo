#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
__author__ = 'LH'
__version__ = 1.0
__date__ = 23/10/2015

import abc
import numpy as np


class PIDKernel(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def clearStatus(self):
        return

    @abc.abstractmethod
    def calPidOutput(self, curErr):
        return


class DeltaPIDKernel(PIDKernel):
    """
    Incremental PID
    """
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.preUk = np.zeros((6,1))
        self.deltaUk = np.zeros((6,1))
        self.curUk = np.zeros((6,1))
        self.curErr = np.zeros((6,1))
        self.preErr = np.zeros((6,1))
        self.pre_preErr = np.zeros((6,1))

    def clearStatus(self):
        self.preUk = np.zeros((6,1))
        self.deltaUk = np.zeros((6,1))
        self.curUk = np.zeros((6,1))
        self.curErr = np.zeros((6,1))
        self.preErr = np.zeros((6,1))
        self.pre_preErr = np.zeros((6,1))

    def calPidOutput(self, curErr):
        self.curErr = curErr
        self.deltaUk = self.Kp * (self.curErr - self.preErr) + \
                       self.Ki * self.curErr + \
                       self.Kd * (self.curErr-2*self.preErr+self.pre_preErr)
        self.pre_preErr = self.preErr
        self.preErr = self.curErr
        return self.deltaUk


class AbsPIDKernel(PIDKernel):
    """
    Position PID
    """
    def __init__(self, P, I, D):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.curUk = np.zeros((6,1))
        self.curErr = np.zeros((6,1))
        self.preErr = np.zeros((6,1))
        self.IntErr = np.zeros((6,1))
        self.DeltaErr = np.zeros((6,1))

    def clearStatus(self):
        self.curUk = np.zeros((6,1))
        self.curErr = np.zeros((6,1))
        self.preErr = np.zeros((6,1))
        self.IntErr = np.zeros((6,1))
        self.DeltaErr = np.zeros((6,1))

    def calPidOutput(self, curErr):
        self.curErr = curErr
        self.IntErr += self.curErr
        self.DeltaErr = self.curErr - self.preErr

        self.curUk = self.Kp * self.curErr + \
                     self.Ki * self.IntErr + \
                     self.Kd * self.DeltaErr

        self.preErr = self.curErr
        return self.curUk
