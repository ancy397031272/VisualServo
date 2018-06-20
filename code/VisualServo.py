#!/usr/bin/python2.7
# -*- coding:utf-8 -*-
import cv2
import time
import numpy as np
import GazeboInterface
from toolbox import vgl as VGL
from toolbox.vgl.eye_hand_vision import EyeInHandStereoVision as EIHSteV
from toolbox import fileinterface as FIT
from toolbox import imgproctool as IPT
from toolbox.vgl.visual_servo_lib import VisualServoImageBase as VSIB
import matplotlib.pyplot as plt


class ImageBasedVisualServo(object):
    def __init__(self):
        self.CameraMatrixA = None  # init in __init_camera
        self.CameraMatrixB = None  # init in __init_camera
        self.MyEIHSteVision = self.__init_camera()
        self.vs_gazebo_env = self.__init_robot()
        # self.imgPtsA_hole, self.imgPtsB_hole = self.vs_gazebo_env.getRealImgPts_hole()
        self.imgPtsA_hole, self.imgPtsB_hole = self.vs_gazebo_env.getVirtualPointsInImg(10)
        self.RealImgPtsA_pin, self.RealImgPtsB_pin = self.vs_gazebo_env.getRealImgPts_pin()
        initial_pose = self.vs_gazebo_env.getRobotPose()
        self.Ttr = VGL.Pose2T(initial_pose)
        self.single_E = []
        self.E = []
        # self.draw_pins_holes(self.imgPtsA_hole, self.imgPtsB_hole, self.RealImgPtsA_pin, self.RealImgPtsB_pin, True)

    def __init_camera(self):
        CameraCalibrationData = FIT.loadYaml(fileName='./CameraCalibrationData.yaml')
        self.CameraMatrixA = np.array(CameraCalibrationData['CameraMatrixA'])
        self.CameraMatrixB = np.array(CameraCalibrationData['CameraMatrixB'])
        TCACB = np.array(CameraCalibrationData['TCACB'])
        TCA14 = np.array(CameraCalibrationData['TCA14'])  # not used
        Tct14 = np.array(CameraCalibrationData['Tct_1414'])  # not used
        TctA = np.matrix(Tct14) * np.matrix(TCA14)  # not used
        E_AB = np.array(CameraCalibrationData['E_AB'])  # not used
        F_AB = np.array(CameraCalibrationData['F_AB'])  # not used

        MyEIHSteVision = EIHSteV(cameraMatrixA=self.CameraMatrixA, cameraMatrixB=self.CameraMatrixB, distCoeffsA=(),
                                 distCoeffsB=(), TcAcB=TCACB, E=E_AB, F=F_AB, TctA=TctA, TctB=None)
        return MyEIHSteVision

    def __init_robot(self):
        ROBOT_BASE = [400, 39.5, 380, 0, 0, 180]  # note it is relative to robot base
        # BOARD_REF = [400, 39.5, 162, 0, 0, 0]
        BOARD_REF = [400 + 20, 39.5 + 20, 761, 0, 0, 90]  # note it is relative to world coodinate
        vs_gazebo_env = GazeboInterface.VSGazeboEnv()
        vs_gazebo_env.reset(BOARD_REF, ROBOT_BASE)
        return vs_gazebo_env

    def track(self, track_times, theta,delta_t):
        TcA2r = self.vs_gazebo_env.getTcA2r()
        TcB2r = self.vs_gazebo_env.getTcB2r()
        RA = self.get_R(theta, theta, theta)
        RB = self.get_R(-theta, -theta, -theta)
        # TcA2r[0:3, 0:3] = RA.dot(TcA2r[0:3, 0:3])
        # TcB2r[0:3, 0:3] = RB.dot(TcB2r[0:3, 0:3])
        Tx, Ty, Tz = delta_t, delta_t, delta_t
        TcA2r[:, 3] += [Tx, Ty, Tz, 1]
        TcB2r[:, 3] += [Tx, Ty, Tz, 1]
        Times = 0
        while Times < track_times:
            Times += 1
            PinPtsInCamA, PinPtsInCamB = self.MyEIHSteVision.get3dPts(imgPtsA_2xn=self.RealImgPtsA_pin,
                                                                      imgPtsB_2xn=self.RealImgPtsB_pin,
                                                                      flag=EIHSteV.COOR_IMG2CAM)
            HolePtsInCamA, HolePtsInCamB = self.MyEIHSteVision.get3dPts(imgPtsA_2xn=self.imgPtsA_hole,
                                                                        imgPtsB_2xn=self.imgPtsB_hole,
                                                                        flag=EIHSteV.COOR_IMG2CAM)

            VelocityInRob, Ef = VSIB.matchPPs(intrinsic_list=[self.CameraMatrixA, self.CameraMatrixB],
                                              objPts_2xn_list=[self.RealImgPtsA_pin, self.RealImgPtsB_pin],
                                              # tarPts_2xn_list=[RealImgPtsA_hole, RealImgPtsB_hole],
                                              tarPts_2xn_list=[self.imgPtsA_hole, self.imgPtsB_hole],
                                              z_1xn_list=[PinPtsInCamA[2, :].reshape(1, -1),
                                                          # +HolePtsInCamA[2,:].reshape(1,-1))/2,
                                                          PinPtsInCamB[2, :].reshape(1, -1)],
                                              # +HolePtsInCamB[2,:].reshape(1,-1))/2],
                                              Tc2x_list=[TcA2r, TcB2r],
                                              mask=(1, 1, 1, 0, 0, 1)  # note: dimension is [x,y,z,w,v,u]
                                              )

            self.single_E.append(np.mean(np.abs(Ef)))
            VelocityInRob = VSIB.transVelocity(Tx2x=np.eye(4), velocity=VelocityInRob)
            MoveT = VSIB.velocityPose2Move(velocityPose=VelocityInRob, iter=100000)
            self.Ttr = MoveT.dot(self.Ttr)
            new_pose = VGL.T2Pose(self.Ttr, False)
            self.vs_gazebo_env.setRobotPose(new_pose)  # note that setrobotpose in set tool pose relative to robot base
            self.RealImgPtsA_pin, self.RealImgPtsB_pin = self.vs_gazebo_env.getRealImgPts_pin()
            # self.draw_pins_holes(self.imgPtsA_hole, self.imgPtsB_hole, self.RealImgPtsA_pin, self.RealImgPtsB_pin)

    def draw_pins_holes(self, ImgPtsA_hole, ImgPtsB_hole, RealImgPtsA_pin, RealImgPtsB_pin, is_save=False):
        ShowImgA = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
        ShowImgB = np.zeros(shape=(960, 1280, 3), dtype=np.uint8)
        IPT.drawPoints(img=ShowImgA, pts_2xn=ImgPtsA_hole, color=(0, 0, 255), radius=7, thickness=1)
        IPT.drawPoints(img=ShowImgB, pts_2xn=ImgPtsB_hole, color=(0, 0, 255), radius=7, thickness=1)
        IPT.drawPoints(img=ShowImgA, pts_2xn=RealImgPtsA_pin, color=(0, 255, 0), radius=7, thickness=-1)
        IPT.drawPoints(img=ShowImgB, pts_2xn=RealImgPtsB_pin, color=(0, 255, 0), radius=7, thickness=-1)
        if is_save:
            cv2.imwrite('./pics/A_blackBG.png', ShowImgA)
            cv2.imwrite('./pics/B_blackBG.png', ShowImgB)

        cv2.namedWindow('A', cv2.WINDOW_NORMAL)
        cv2.namedWindow('B', cv2.WINDOW_NORMAL)
        cv2.imshow('A', ShowImgA)
        cv2.imshow('B', ShowImgB)
        cv2.waitKey()
        return

    def save_img(self):
        ShowImgA = self.vs_gazebo_env.getPinHoleLeftImage()
        ShowImgB = self.vs_gazebo_env.getPinHoleRightImage()
        IPT.drawPoints(img=ShowImgA, pts_2xn=self.imgPtsA_hole, color=(0, 0, 255), radius=7, thickness=1)
        IPT.drawPoints(img=ShowImgB, pts_2xn=self.imgPtsB_hole, color=(0, 0, 255), radius=7, thickness=1)
        IPT.drawPoints(img=ShowImgA, pts_2xn=self.RealImgPtsA_pin, color=(0, 255, 0), radius=7, thickness=-1)
        IPT.drawPoints(img=ShowImgB, pts_2xn=self.RealImgPtsB_pin, color=(0, 255, 0), radius=7, thickness=-1)
        cv2.imwrite('./pics/A_realBG.png', ShowImgA)
        cv2.imwrite('./pics/B_realBG.png', ShowImgB)

    def get_R(self, alpha, beta, gama):
        alpha = alpha / 360.
        beta = beta / 360.
        gama = gama / 360.
        ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gama)
        sa, sb, sg = np.sin(alpha), np.sin(beta), np.sin(gama)
        R = np.array([[ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
                      [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
                      [-sb, cb * sg, cb * cg]])
        return R

    def testDepthZ(self):
        """
        To verify if calculating depth z is right or not
        """
        PinPtsInCamA, PinPtsInCamB = self.MyEIHSteVision.get3dPts(imgPtsA_2xn=self.RealImgPtsA_pin,
                                                                  imgPtsB_2xn=self.RealImgPtsB_pin,
                                                                  flag=EIHSteV.COOR_IMG2CAM)
        TcA2ref = self.vs_gazebo_env.getTcA2ref()
        TcB2ref = self.vs_gazebo_env.getTcB2ref()
        PinPtsInRefA = VGL.projectPts(PinPtsInCamA, TcA2ref)
        PinPtsInRefB = VGL.projectPts(PinPtsInCamB, TcB2ref)
        print 'PinPtsInRefA', PinPtsInRefA
        print 'PinPtsInRefB', PinPtsInRefB
        left_pin_lead_pose_mm_deg, right_pin_lead_pose_mm_deg = self.vs_gazebo_env.getPinLeadPoses_mm_deg()
        print 'left_pin_lead_pose_mm_deg', left_pin_lead_pose_mm_deg
        print 'right_pin_lead_pose_mm_deg', right_pin_lead_pose_mm_deg

    def repeat_track(self, repeat_times, track_times, delta_angle,delta_t):
        for i in range(repeat_times):
            print "repeat track ", i + 1
            self.single_E = []
            if i != 0:
                self.__init_robot()
            self.imgPtsA_hole, self.imgPtsB_hole = self.vs_gazebo_env.getVirtualPointsInImg(10)
            self.RealImgPtsA_pin, self.RealImgPtsB_pin = self.vs_gazebo_env.getRealImgPts_pin()
            initial_pose = self.vs_gazebo_env.getRobotPose()
            self.Ttr = VGL.Pose2T(initial_pose)
            self.track(track_times, i * delta_angle,i *delta_t)
            print self.single_E
            self.E.append(self.single_E)

    def insert(self, insert_depth=8):
        cur_pose = self.vs_gazebo_env.getRobotPose()
        insert_pose = cur_pose + np.array([0., 0., -insert_depth, 0., 0., 0.])
        self.vs_gazebo_env.setRobotPose(insert_pose)

    def go_virtualPts_then_insert(self, insert_depth=8, virtualPts_height=10):
        self.insert()
        cur_pose = self.vs_gazebo_env.getRobotPose()
        insert_pose = cur_pose + np.array([0., 0., -virtualPts_height, 0., 0., 0.])
        time.sleep(1)
        self.vs_gazebo_env.setRobotPose(insert_pose)

    def plot_error(self, repeat_times, track_times, delta_angle):
        plt.figure()
        for i in range(repeat_times):
            plt.plot(self.E[i])
            plt.scatter(np.arange(0, track_times), self.E[i])
            # plt.legend("rotation angle:" + str(i * 0.5))
        plt.xlabel("#track")
        plt.ylabel("error/pixel")

        plt.legend((np.arange(0, repeat_times) * delta_angle).tolist())
        plt.xlim([0, 30])
        plt.ylim([0, 50])
        plt.show()


if __name__ == '__main__':

    E_list = []

    IBVS = ImageBasedVisualServo()
    # IBVS.save_img()
    # IBVS.testDepthZ()
    # IBVS.track(track_times=50)
    IBVS.repeat_track(repeat_times=10, track_times=20,delta_angle=1,delta_t=1)
    # IBVS.go_virtualPts_then_insert(10)
    IBVS.plot_error(repeat_times=10, track_times=20,delta_angle=1)
    # IBVS.insert()
