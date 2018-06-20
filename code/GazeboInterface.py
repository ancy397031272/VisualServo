__author__ = 'ZHYP'
__version__ = '0.1'
__date__ = '12/12/2017'
__copyright__ = "Copyright 2017, RR"

import time
import numpy as np

# from rr_robot_plugin.pythonInterface import GazeboInterface
# from rr_robot_plugin.pythonInterface.toolbox.vgl import Pose2T, T2Pose, projectPtsToImg
import sys
sys.path.append('../')
from RobotControl import RobotControl
from WorldControl import WorldControl
from Camera import Camera
from toolbox.vgl import Pose2T, T2Pose, projectPtsToImg,projectPts

ROBOT_NAME = 'kent'
DOF = 6
PIN_HOLE_LEFT_CAMERA_TOPIC = '/pin_hole_left_cam/my_sensor/rgb/image'
PIN_HOLE_RIGHT_CAMERA_TOPIC = '/pin_hole_right_cam/my_sensor/rgb/image'

PIN_HOLE_LEFT_CAMERA_LINK = 'pin_hole_left_cam::camera::link'
PIN_HOLE_RIGHT_CAMERA_LINK = 'pin_hole_right_cam::camera::link'

BOARD_LINK = 'board::board::link'

# BOARD_LINK = 'board'

LEFT_PIN_LINK_3 = 'kent::lf_link_3'
RIGHT_PIN_LINK_3 = 'kent::ri_link_3'
C_BASE_LINK = 'kent::c_base'

# no !
# SET_PIN_JOINT_ANGLES_SERVICE = 'kent/SetAngle'
# GET_PIN_JOINT_ANGLES_SERVICE = 'kent/GetAngle'

ROBOT_TIMEOUT_SEC = 5

PIN_LEAD2LINK3_MM_DEG = [0, 0, -7, 0, 0, 0]
LEFT_HOLE2BOARD_MM_DEG = [2.715, 0, 11, 0, 0, 0]
RIGHT_HOLE2BOARD_MM_DEG = [-2.715, 0, 11, 0, 0, 0]
LEFT_PIN_LINK3_TO_CBASE_MM_DEG = [2.71624339358, 0, 9.5, 0, 0, 0]
RIGHT_PIN_LINK3_TO_CBASE_MM_DEG = [-2.71624339358, 0, 9.5, 0, 0, 0]

LEAD_INSPECTION_POSE_MM_DEG = [316.7, -350, 380, 0, 0, 180]

GET_LINK3_POSE_BY_C_BASE = True
FOV = 1.0471820914712345
IMAGE_WIDTH_PX = 1280
IMAGE_HEIGHT_PX = 960
ERROR_TOLERANCE_XY_MM = 0.35
ERROR_TOLERANCE_Z_MM = 1.0
ERROR_TOLERANCE_OBLIQUITY_DEG = 1.0
MAX_STRAIGHTEN_COUNT = 10


class VSGazeboEnvError(Exception):
    pass


class VSGazeboEnv(object):
    def __init__(self):
        self._robot = RobotControl(robotName=ROBOT_NAME, dof=DOF)
        self._world = WorldControl()
        self._pin_hole_left_cam = Camera(rgbImageTopic=PIN_HOLE_LEFT_CAMERA_TOPIC)
        self._pin_hole_right_cam = Camera(rgbImageTopic=PIN_HOLE_RIGHT_CAMERA_TOPIC)

        self._inspected_pin_leads_in_cameras = None
        self._board_pose_mm_deg = None
        self._robot_init_pose_mm_deg = None
        self._left_pin_lead_pose_init_mm_deg = None
        self._right_pin_lead_pose_init_mm_deg = None

    # ----- get images ------
    def getPinHoleLeftImage(self):
        return self._pin_hole_left_cam.getRGBImage()

    def getPinHoleRightImage(self):
        return self._pin_hole_right_cam.getRGBImage()

    # ----- robot manipulation -----
    def getRobotPose(self):
        return self._robot.getRobotPos()

    def setRobotPose(self, pose_mm_deg, wait=True):
        self._robot.setRobotPos(pose_mm_deg)
        timeout = 0
        while wait and self._robot.isMoving:
            time.sleep(0.01)
            timeout += 0.01
            if timeout > ROBOT_TIMEOUT_SEC:
                raise VSGazeboEnvError('Robot moving timeout')

    def setRobotJointAngles(self, angles_deg, wait=True):
        self._robot.setJointAngle(angles_deg)
        timeout = 0
        while wait and self._robot.isMoving:
            time.sleep(0.01)
            timeout += 0.01
            if timeout > ROBOT_TIMEOUT_SEC:
                raise VSGazeboEnvError('Robot moving timeout')

    def resetRobotToHome(self):
        self.setRobotJointAngles([0] * 6)

    def goToLeadInspectionPose(self):
        self.setRobotPose(LEAD_INSPECTION_POSE_MM_DEG)

    def setRobotSpeed(self, speed_percentage):
        self._robot.setSpeedRate(speed_percentage)

    # ----- model manipulation -----
    def _getModelPose_mm_deg(self, link_name):
        return self._world.getLinkPos(link_name)

    def _setModelPose(self, link_name, pose_mm_deg):
        self._world.setLinkPos(link_name, pose_mm_deg)

    # ----- board related actions -----
    def getBoardPose_mm_deg(self):
        return self._getModelPose_mm_deg(BOARD_LINK)

    def setBoardPose(self, pose_mm_deg):
        self._setModelPose(BOARD_LINK, pose_mm_deg)

    def getHolePoses_mm_deg(self):
        board_pose_mm_deg = self.getBoardPose_mm_deg()

        t_board_to_ref = Pose2T(board_pose_mm_deg)
        t_left_hole_to_board = Pose2T(LEFT_HOLE2BOARD_MM_DEG)
        t_right_hole_to_board = Pose2T(RIGHT_HOLE2BOARD_MM_DEG)

        t_left_hole_to_ref = t_board_to_ref.dot(t_left_hole_to_board)
        t_right_hole_to_ref = t_board_to_ref.dot(t_right_hole_to_board)

        left_hole_pose_mm_deg = T2Pose(t_left_hole_to_ref)
        right_hole_pose_mm_deg = T2Pose(t_right_hole_to_ref)

        return left_hole_pose_mm_deg.flatten(), right_hole_pose_mm_deg.flatten()

    # ----- pin related operations -----
    def getPinLink3PoseByCBase_mm_deg(self, pin_link3_to_c_base):
        c_base_pose_mm_deg = self._getModelPose_mm_deg(C_BASE_LINK)
        t_c_base_to_ref = Pose2T(c_base_pose_mm_deg)
        t_pin_link3_to_c_base = Pose2T(pin_link3_to_c_base)

        t_pin_link3_to_ref = t_c_base_to_ref.dot(t_pin_link3_to_c_base)
        pin_link3_pose_mm_deg = T2Pose(t_pin_link3_to_ref)

        return pin_link3_pose_mm_deg.flatten()

    def getLeftPinLink3Pose_mm_deg(self, by_c_base=False):
        return self.getPinLink3PoseByCBase_mm_deg(LEFT_PIN_LINK3_TO_CBASE_MM_DEG) if by_c_base else \
            self._getModelPose_mm_deg(LEFT_PIN_LINK_3)

    def getRightPinLink3Pose_mm_deg(self, by_c_base=False):
        return self.getPinLink3PoseByCBase_mm_deg(RIGHT_PIN_LINK3_TO_CBASE_MM_DEG) if by_c_base else \
            self._getModelPose_mm_deg(RIGHT_PIN_LINK_3)

    def getPinLeadPoses_mm_deg(self, by_c_base=False):
        left_pin_link3_pose_mm_deg = self.getLeftPinLink3Pose_mm_deg(by_c_base)
        right_pin_link3_pose_mm_deg = self.getRightPinLink3Pose_mm_deg(by_c_base)

        t_left_pin_link3_to_ref = Pose2T(left_pin_link3_pose_mm_deg)
        t_right_pin_link3_to_ref = Pose2T(right_pin_link3_pose_mm_deg)
        t_lead_to_link3 = Pose2T(PIN_LEAD2LINK3_MM_DEG)

        t_left_pin_lead_to_ref = t_left_pin_link3_to_ref.dot(t_lead_to_link3)
        t_right_pin_lead_to_ref = t_right_pin_link3_to_ref.dot(t_lead_to_link3)

        left_pin_lead_pose_mm_deg = T2Pose(t_left_pin_lead_to_ref)
        right_pin_lead_pose_mm_deg = T2Pose(t_right_pin_lead_to_ref)

        return left_pin_lead_pose_mm_deg.flatten(), right_pin_lead_pose_mm_deg.flatten()

    def getPinLeadsHolesInImage(self, camera_link):
        f_width = IMAGE_WIDTH_PX / (np.tan(FOV / 2.0) * 2.0)
        cameraMatrix = np.array([
            [f_width, 0, IMAGE_WIDTH_PX / 2.0],
            [0, f_width, IMAGE_HEIGHT_PX / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        camera_pose_mm_deg = self._getModelPose_mm_deg(camera_link)
        t_camera_to_ref = Pose2T(camera_pose_mm_deg)
        t_ref_to_camera = np.linalg.inv(t_camera_to_ref)

        # -----pin leads in image -----
        left_pin_lead_pose_mm_deg, right_pin_lead_pose_mm_deg = self.getPinLeadPoses_mm_deg()

        left_pin_in_camera_px = projectPtsToImg(left_pin_lead_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])
        right_pin_in_camera_px = projectPtsToImg(right_pin_lead_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])

        # -----holes in image -----
        left_hole_pose_mm_deg, right_hole_pose_mm_deg = self.getHolePoses_mm_deg()

        left_hole_in_camera_px = projectPtsToImg(left_hole_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])
        right_hole_in_camera_px = projectPtsToImg(right_hole_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])

        return left_pin_in_camera_px.flatten(), right_pin_in_camera_px.flatten(), \
               left_hole_in_camera_px.flatten(), right_hole_in_camera_px.flatten()

    def getTcA2ref(self):
        camera_pose_mm_deg = self._getModelPose_mm_deg(PIN_HOLE_LEFT_CAMERA_LINK )
        t_camera_to_ref = Pose2T(camera_pose_mm_deg)
        return t_camera_to_ref

    def getTcB2ref(self):
        camera_pose_mm_deg = self._getModelPose_mm_deg(PIN_HOLE_RIGHT_CAMERA_LINK)
        t_camera_to_ref = Pose2T(camera_pose_mm_deg)
        return t_camera_to_ref

    def getPinLeadsHolesInLeftCamera(self):
        return self.getPinLeadsHolesInImage(PIN_HOLE_LEFT_CAMERA_LINK)

    def getPinLeadsHolesInRightCamera(self):
        return self.getPinLeadsHolesInImage(PIN_HOLE_RIGHT_CAMERA_LINK)

    def getPinLeadsHolesInCameras(self):
        # ----- pin leads and holes in left camera -----
        left_pin_in_left_camera_px, right_pin_in_left_camera_px, \
        left_hole_in_left_camera_px, right_hole_in_left_camera_px = self.getPinLeadsHolesInLeftCamera()

        # ----- pin leads and holes in right camera -----
        left_pin_in_right_camera_px, right_pin_in_right_camera_px, \
        left_hole_in_right_camera_px, right_hole_in_right_camera_px = self.getPinLeadsHolesInRightCamera()
        return np.r_[left_pin_in_left_camera_px, right_pin_in_left_camera_px,
                     left_pin_in_right_camera_px, right_pin_in_right_camera_px], \
               np.r_[left_hole_in_left_camera_px, right_hole_in_left_camera_px,
                     left_hole_in_right_camera_px, right_hole_in_right_camera_px]


        # def setPinJointAngles(self, joint_angles_deg):

    def getTc2r(self, camera_link):
        camera_pose_mm_deg = self._getModelPose_mm_deg(camera_link)
        t_camera_to_ref = Pose2T(camera_pose_mm_deg)
        robot_base_pose=[0.,0.,600.,0.,0.,0.]
        t_robot_base_to_ref=Pose2T(robot_base_pose)
        t_ref_to_robot_base=np.linalg.inv(t_robot_base_to_ref)
        t_camera_to_robot_base=np.dot(t_ref_to_robot_base,t_camera_to_ref)
        return t_camera_to_robot_base

    def getTcA2r(self):
        return self.getTc2r(PIN_HOLE_LEFT_CAMERA_LINK)

    def getTcB2r(self):
        return self.getTc2r(PIN_HOLE_RIGHT_CAMERA_LINK)

    def getTcA2cB(self):
        TcA2r=self.getTcA2r()
        TcB2r=self.getTcB2r()
        Tr2cB=np.linalg.inv(TcB2r)
        TcA2cB=np.dot(Tr2cB,TcA2r)
        return TcA2cB

    def getTref2r(self):
        robot_base_pose = [0., 0., 600., 0., 0., 0.]
        t_robot_base_to_ref = Pose2T(robot_base_pose)
        t_ref_to_robot_base = np.linalg.inv(t_robot_base_to_ref)
        return t_ref_to_robot_base

    def setVirtualPointsInImg(self,delta_z,camera_link):
        f_width = IMAGE_WIDTH_PX / (np.tan(FOV / 2.0) * 2.0)
        cameraMatrix = np.array([
            [f_width, 0, IMAGE_WIDTH_PX / 2.0],
            [0, f_width, IMAGE_HEIGHT_PX / 2.0],
            [0, 0, 1]
        ], dtype=np.float32)
        camera_pose_mm_deg = self._getModelPose_mm_deg(camera_link)
        t_camera_to_ref = Pose2T(camera_pose_mm_deg)
        t_ref_to_camera = np.linalg.inv(t_camera_to_ref)

        # -----pin leads in image -----
        left_pin_lead_pose_mm_deg, right_pin_lead_pose_mm_deg = self.getPinLeadPoses_mm_deg()

        left_pin_in_camera_px = projectPtsToImg(left_pin_lead_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])
        right_pin_in_camera_px = projectPtsToImg(right_pin_lead_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])

        # -----holes in image -----
        left_hole_pose_mm_deg, right_hole_pose_mm_deg = self.getHolePoses_mm_deg()
        left_hole_pose_mm_deg[2] += delta_z
        right_hole_pose_mm_deg[2] += delta_z

        left_hole_in_camera_px = projectPtsToImg(left_hole_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])
        right_hole_in_camera_px = projectPtsToImg(right_hole_pose_mm_deg[:3], t_ref_to_camera, cameraMatrix, [])

        return left_pin_in_camera_px.flatten(), right_pin_in_camera_px.flatten(), \
               left_hole_in_camera_px.flatten(), right_hole_in_camera_px.flatten()

    def getVirtualPointsInImg(self,delta_z):
        left_pin_in_left_camera, right_pin_in_left_camera, left_hole_in_left_camera, right_hole_in_left_camera = self.setVirtualPointsInImg(delta_z,PIN_HOLE_LEFT_CAMERA_LINK)
        left_pin_in_right_camera, right_pin_in_right_camera, left_hole_in_right_camera, right_hole_in_right_camera = self.setVirtualPointsInImg(delta_z,PIN_HOLE_RIGHT_CAMERA_LINK)
        virtualImgPtsA_hole = np.hstack(
            (left_hole_in_left_camera.reshape(2, -1), right_hole_in_left_camera.reshape(2, -1)))
        virtualImgPtsB_hole = np.hstack(
            (left_hole_in_right_camera.reshape(2, -1), right_hole_in_right_camera.reshape(2, -1)))
        return virtualImgPtsA_hole,virtualImgPtsB_hole

    def getRealImgPts_hole(self):
        _, _, left_hole_in_left_camera, right_hole_in_left_camera = self.getPinLeadsHolesInLeftCamera()
        _, _, left_hole_in_right_camera, right_hole_in_right_camera = self.getPinLeadsHolesInRightCamera()
        realImgPtsA_hole = np.hstack((left_hole_in_left_camera.reshape(2, -1), right_hole_in_left_camera.reshape(2, -1)))
        realImgPtsB_hole = np.hstack((left_hole_in_right_camera.reshape(2, -1), right_hole_in_right_camera.reshape(2, -1)))
        return realImgPtsA_hole,realImgPtsB_hole
    def getRealImgPts_pin(self):
        left_pin_in_left_camera, right_pin_in_left_camera, _, _, = self.getPinLeadsHolesInLeftCamera()
        left_pin_in_right_camera, right_pin_in_right_camera, _, _, = self.getPinLeadsHolesInRightCamera()
        RealImgPtsA_pin = np.hstack((left_pin_in_left_camera.reshape(2, -1), right_pin_in_left_camera.reshape(2, -1)))
        RealImgPtsB_pin = np.hstack((left_pin_in_right_camera.reshape(2, -1), right_pin_in_right_camera.reshape(2, -1)))
        return RealImgPtsA_pin,RealImgPtsB_pin

    def getRealPinPtsInRob(self):
        left_pin_lead_pose_mm_deg, right_pin_lead_pose_mm_deg=self.getPinLeadPoses_mm_deg()
        t_ref_to_robot_base = self.getTref2r()
        RealPinPtsInRob_left=projectPts(left_pin_lead_pose_mm_deg[:3].reshape(3,-1),t_ref_to_robot_base)
        RealPinPtsInRob_right=projectPts(right_pin_lead_pose_mm_deg[:3].reshape(3,-1),t_ref_to_robot_base)

        RealPinPtsInRob = np.hstack((RealPinPtsInRob_left.reshape(3, -1), RealPinPtsInRob_right.reshape(3, -1)))
        return RealPinPtsInRob
    # ----- operations for RL algorithms -----
    def reset(self, board_pose_mm_deg, robot_pose_mm_deg):
        self._board_pose_mm_deg = board_pose_mm_deg
        self._robot_init_pose_mm_deg = robot_pose_mm_deg
        self.setBoardPose(board_pose_mm_deg)
        self.goToLeadInspectionPose()
        self._inspected_pin_leads_in_cameras = self.getPinLeadsHolesInCameras()
        self.setRobotPose(robot_pose_mm_deg)
        self._left_pin_lead_pose_init_mm_deg, self._right_pin_lead_pose_init_mm_deg = \
            self.getPinLeadPoses_mm_deg(GET_LINK3_POSE_BY_C_BASE)

        return True
        # return False if not self.straighten_pins() else True

    def getInspectedPinLeadsInCameras(self):
        return self._inspected_pin_leads_in_cameras


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)

    ROBOT_REF = [400, 39.5, 380, 0, 0, 180]
    # BOARD_REF = [400, 39.5, 162, 0, 0, 0]
    BOARD_REF = [400, 39.5, 761, 0, 0, 0]

    vs_gazebo_env = VSGazeboEnv()
    vs_gazebo_env.reset(BOARD_REF, ROBOT_REF)



    left_pin_link3_pose_mm_deg = vs_gazebo_env.getLeftPinLink3Pose_mm_deg()
    right_pin_link3_pose_mm_deg = vs_gazebo_env.getRightPinLink3Pose_mm_deg()

    left_pin_lead_pose_mm_deg, right_pin_lead_pose_mm_deg = vs_gazebo_env.getPinLeadPoses_mm_deg()

    print 'link3 pose of left pin and right pin'
    print left_pin_link3_pose_mm_deg
    print right_pin_link3_pose_mm_deg
    print 'lead pose of left pin and right pin'
    print left_pin_lead_pose_mm_deg
    print right_pin_lead_pose_mm_deg

    left_pin_left_camera_px, right_pin_left_camera_px, \
    left_hole_left_camera, right_hole_left_camera = vs_gazebo_env.getPinLeadsHolesInLeftCamera()
    left_pin_right_camera_px, right_pin_right_camera_px, \
    left_hole_right_camera, right_hole_right_camera = vs_gazebo_env.getPinLeadsHolesInRightCamera()

    print 'left camera, left pin and right pin'
    print left_pin_left_camera_px
    print right_pin_left_camera_px
    print 'right camera, left pin and right pin'
    print left_pin_right_camera_px
    print right_pin_right_camera_px

    board_pose = vs_gazebo_env.getBoardPose_mm_deg()
    left_hole_pose, right_hole_pose = vs_gazebo_env.getHolePoses_mm_deg()

    print 'board pose'
    print board_pose
    print 'left hole, right hole'
    print left_hole_pose
    print right_hole_pose

    import cv2

    pin_hole_left_img = vs_gazebo_env.getPinHoleLeftImage()
    pin_hole_right_img = vs_gazebo_env.getPinHoleRightImage()

    RED = [0, 0, 255, 0]
    YELLOW = [0, 255, 255, 0]
    cv2.circle(pin_hole_left_img, tuple(np.round(left_pin_left_camera_px).astype(int)),
               radius=3, thickness=1, color=RED)
    cv2.circle(pin_hole_left_img, tuple(np.round(right_pin_left_camera_px).astype(int)),
               radius=3, thickness=1, color=YELLOW)
    cv2.circle(pin_hole_right_img, tuple(np.round(left_pin_right_camera_px).astype(int)),
               radius=3, thickness=1, color=RED)
    cv2.circle(pin_hole_right_img, tuple(np.round(right_pin_right_camera_px).astype(int)),
               radius=3, thickness=1, color=YELLOW)

    cv2.circle(pin_hole_left_img, tuple(np.round(left_hole_left_camera).astype(int)),
               radius=3, thickness=1, color=RED)
    cv2.circle(pin_hole_left_img, tuple(np.round(right_hole_left_camera).astype(int)),
               radius=3, thickness=1, color=YELLOW)
    cv2.circle(pin_hole_right_img, tuple(np.round(left_hole_right_camera).astype(int)),
               radius=3, thickness=1, color=RED)
    cv2.circle(pin_hole_right_img, tuple(np.round(right_hole_right_camera).astype(int)),
               radius=3, thickness=1, color=YELLOW)

    # cv2.imshow('left', pin_hole_left_img)
    # cv2.imshow('right', pin_hole_right_img)
    cv2.imwrite('./left.png', pin_hole_left_img)
    cv2.imwrite('./right.png', pin_hole_right_img)
    cv2.waitKey()

    # vs_gazebo_env.resetRobotToHome()

