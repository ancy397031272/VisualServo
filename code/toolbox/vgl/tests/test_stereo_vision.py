# # if __name__ == '__main__':
# #     ParameterYamlPath = './Data/CameraCalibrationData.yaml'
# #     YamlFile = file(ParameterYamlPath, 'r')
# #     CameraCalibrationData = yaml.load(YamlFile)
# #     YamlFile.close()
# #
# #     SrcImg_cam13 = cv2.imread('./Data/cam13.png')
# #     SrcImg_cam14 = cv2.imread('./Data/cam14.png')
# #     Tc1c2 = np.array(CameraCalibrationData['Tc14c13'])
# #     Intrinsic_cam13 = np.array(CameraCalibrationData['CameraMatrix_1313'])
# #     Intrinsic_cam14 = np.array(CameraCalibrationData['CameraMatrix_1414'])
# #     DistCoeffs_1313 = np.array(CameraCalibrationData['DistCoeffs_1313'])
# #     DistCoeffs_1414 = np.array(CameraCalibrationData['DistCoeffs_1414'])
# #     E = np.array(CameraCalibrationData['E'])
# #     F = np.array(CameraCalibrationData['F'])
# #
# #     ImgPts1313_nx2 = np.loadtxt('./Data/1313.txt')
# #     ImgPts1414_nx2 = np.loadtxt('./Data/1414.txt')
# #     ImgPts1313_2xn = ImgPts1313_nx2.T.reshape(2, -1)
# #     ImgPts1414_2xn = ImgPts1414_nx2.T.reshape(2, -1)
# #     MyStereoSys = StereoVision(cameraMatrixA=Intrinsic_cam14, cameraMatrixB=Intrinsic_cam13,
# #                                distCoeffsA=DistCoeffs_1414, distCoeffsB=DistCoeffs_1313, TcAcB=Tc1c2, E=E, F=F)
# #     PtsInCamA_3xn, PtsInCamB_3xn, RpErrA, RpErrB = \
# #         MyStereoSys.get3dPts(imgPtsA_2xn=ImgPts1414_2xn, imgPtsB_2xn=ImgPts1313_2xn, calcReprojErr=True)
# #     print '================ get3dPts ================'
# #     print "PtsInCamA:\n", PtsInCamA_3xn
# #     print "PtsInCamB:\n", PtsInCamB_3xn
# #     print 'DistanceInCamA: ', np.linalg.norm(PtsInCamA_3xn[:,3] - PtsInCamA_3xn[:,4])
# #     print 'DistanceInCamB: ', np.linalg.norm(PtsInCamB_3xn[:,3] - PtsInCamB_3xn[:,4])
# #     # print 'GroundTruthDis_mm: ', GroundTruthDis_mm
# #     print "RpErrA:\n", RpErrA
# #     print "RpErrB:\n", RpErrB
#
# if __name__ == "__main__":
#     ParameterYamlPath = './Data/CameraCalibrationData.yaml'
#     YamlFile = file(ParameterYamlPath, 'r')
#     CameraCalibrationData = yaml.load(YamlFile)
#     YamlFile.close()
#
#     # SrcImg_cam13 = cv2.imread('./Data/cam13.png')
#     # SrcImg_cam14 = cv2.imread('./Data/cam14.png')
#     Tc1c2 = np.array(CameraCalibrationData['Tc14c13'])
#     Intrinsic_cam13 = np.array(CameraCalibrationData['CameraMatrix_1313'])
#     Intrinsic_cam14 = np.array(CameraCalibrationData['CameraMatrix_1414'])
#     DistCoeffs_1313 = np.array(CameraCalibrationData['DistCoeffs_1313'])
#     DistCoeffs_1414 = np.array(CameraCalibrationData['DistCoeffs_1414'])
#     E = np.array(CameraCalibrationData['E'])
#     F = np.array(CameraCalibrationData['F'])
#
#     PointCam14_2xn = np.array([[6.366322275682084637e+02, 7.899394910272621928e+02],
#                                [7.416458478168997317e+02, 6.542366444992233028e+02]], dtype=np.float32)
#     PointCam13_2xn = np.array([[8.149478495193284289e+02, 9.375415089453317705e+02],
#                                [3.091887147386404422e+02, 3.632621606795080424e+02]], dtype=np.float32)
#     GroundTruthDis_mm = 4
#
#     MyStereoSys = StereoVision(cameraMatrixA=Intrinsic_cam14, cameraMatrixB=Intrinsic_cam13,
#                                distCoeffsA=DistCoeffs_1414, distCoeffsB=DistCoeffs_1313, TcAcB=Tc1c2, E=E, F=F)
#     PtsInCamA_3xn, PtsInCamB_3xn, RpErrA, RpErrB = \
#         MyStereoSys.get3dPts(imgPtsA_2xn=PointCam14_2xn, imgPtsB_2xn=PointCam13_2xn, calcReprojErr=True)
#     print '================ get3dPts ================'
#     print "PtsInCamA:\n", PtsInCamA_3xn
#     print "PtsInCamB:\n", PtsInCamB_3xn
#     print 'DistanceInCamA: ', np.linalg.norm(PtsInCamA_3xn[:,0] - PtsInCamA_3xn[:,1])
#     print 'DistanceInCamB: ', np.linalg.norm(PtsInCamB_3xn[:,0] - PtsInCamB_3xn[:,1])
#     print 'GroundTruthDis_mm: ', GroundTruthDis_mm
#     print "RpErrA:\n", RpErrA
#     print "RpErrB:\n", RpErrB
#
#     TctA = np.array(CameraCalibrationData['Tct_1414'])
#     TctB = np.array(CameraCalibrationData['Tct_1313'])
#
#     PtsInToolA_3xn = VGL.projectPts(pts=PtsInCamA_3xn, projectMatrix=TctA)
#     PtsInToolB_3xn = VGL.projectPts(pts=PtsInCamB_3xn, projectMatrix=TctB)
#
#     ToolPose = [454.19, -202.32, 292.58, 126.09, 3.69, 177.50]
#     Ttr = VGL.Pose2T(pose=ToolPose)
#     PtsInRobA_3xn = VGL.projectPts(pts=PtsInToolA_3xn, projectMatrix=Ttr)
#     PtsInRobB_3xn = VGL.projectPts(pts=PtsInToolB_3xn, projectMatrix=Ttr)
#
#     distortProjPtsCamA_2xn = MyStereoSys.projectPts(pts_3xn=PtsInCamA_3xn, flag=StereoVision.CAM_A, distortFlag=True)
#     projPtsCamA_2xn = MyStereoSys.projectPts(pts_3xn=PtsInCamA_3xn, flag=StereoVision.CAM_A, distortFlag=False)
#     print '================ projectPts ================'
#     print 'projPtsCamA_2xn:\n', projPtsCamA_2xn
#     print 'distortProjPtsCamA_2xn:\n', distortProjPtsCamA_2xn
#     print 'GroundTruthPtsCamA_2xn:\n', PointCam14_2xn
#
#     UnDistortPtsCamA_2xn, UnDistortRaysCamA_2xn = MyStereoSys.unDistortPts(imgPts_2xn=distortProjPtsCamA_2xn, flag=StereoVision.CAM_A)
#     print '================ unDistortPts ================'
#     print 'UnDistortRaysCamA_2xn:\n', UnDistortRaysCamA_2xn
#     print 'UnDistortPtsCamA_2xn:\n', UnDistortPtsCamA_2xn
#     print 'GroundTruthUnDistortPtsCamA_2xn:\n', projPtsCamA_2xn
#
#     # UnDistortImg_Cam14 = MyStereoSys.unDistort(img=SrcImg_cam14, flag=StereoVision.CAM_A)
#     # UnDistortImg_Cam13 = MyStereoSys.unDistort(img=SrcImg_cam13, flag=StereoVision.CAM_B)
#     # showShape = (SrcImg_cam14.shape[1]/5, SrcImg_cam14.shape[0]/5)
#     # cv2.imshow('Cam14_delta', cv2.resize(SrcImg_cam14 - UnDistortImg_Cam14, showShape))
#     # cv2.imshow('Cam13_delta', cv2.resize(SrcImg_cam13 - UnDistortImg_Cam13, showShape))
#     # cv2.waitKey()
#     # cv2.destroyAllWindows()
#
#     # ImgA = cv2.imread('./data/FTestImgA.png')
#     # ImgB = cv2.imread('./data/FTestImgB.png')
#     ImgPtsA_nx2 = np.loadtxt('./Data/FTestImgPtsCam14.txt')
#     ImgPtsB_nx2 = np.loadtxt('./Data/FTestImgPtsCam13.txt')
#     ImgPtsA_2xn = ImgPtsA_nx2.T.reshape(2, -1)
#     ImgPtsB_2xn = ImgPtsB_nx2.T.reshape(2, -1)
#
#     Error_1xn = MyStereoSys.calEpilineError(imgPtsA_2xn=ImgPtsA_2xn, imgPtsB_2xn=ImgPtsB_2xn)
#     print 'Error_1xn:\n', Error_1xn
#     print 'mean: ', Error_1xn.mean()
#     print 'std: ', Error_1xn.std()
#
#
#
