# if __name__ == '__main__':
#     ImgPts_2xn = np.array([[6.366322275682084637e+02, 7.899394910272621928e+02],
#                            [7.416458478168997317e+02, 6.542366444992233028e+02]], dtype=np.float32)
#     GroundTruth3DPts_3xn = np.array([[-13.52018097, -10.11538314],
#                                      [ -5.64607480,  -7.66223813],
#                                      [ 86.82860645,  87.40814098]])
#     CameraMatrix = np.array([[  3.86200788e+03,   0.00000000e+00,   1.23487131e+03],
#                              [  0.00000000e+00,   3.86121201e+03,   9.90865811e+02],
#                              [  0.00000000e+00,   0.00000000e+00,   1.00000000e+00]])
#     DistCoeff = np.array([-7.34537009e-02,   3.57639028e+01,  -8.57163220e-05,  -1.95640238e-05,
#                            1.85238425e+02,   8.06021246e-02,   3.53983086e+01,   1.93034174e+02])
#
#     MySingleVision = SingleVision(cameraMatrix=CameraMatrix, distCoeffs=DistCoeff, imgSize=(2592, 1944))
#
#     Pts3D_3xn = MySingleVision.get3DPts(ImgPts_2xn, z_mm=GroundTruth3DPts_3xn[2,1], unDistortFlag=False)
#     unDistortPts3D_3xn = MySingleVision.get3DPts(ImgPts_2xn, z_mm=GroundTruth3DPts_3xn[2,1], unDistortFlag=True)
#     print '================ get3DPts ================'
#     print 'Pts3D_3xn:\n', Pts3D_3xn
#     print 'unDistortPts3D_3xn:\n', unDistortPts3D_3xn
#     print '***GroundTruthPts3D_3xn***\n', GroundTruth3DPts_3xn
#
#     print '================ projectPts2Img ================'
#     distortProjectImgPts_2xn = MySingleVision.projectPts2Img(pts_3xn=unDistortPts3D_3xn, distortFlag=True)
#     projectImgPts_2xn = MySingleVision.projectPts2Img(pts_3xn=unDistortPts3D_3xn, distortFlag=False)
#     print 'distortProjectImgPts_2xn:\n', distortProjectImgPts_2xn
#     print 'projectImgPts_2xn:\n', projectImgPts_2xn
#     print '***GroundTruthImgPts_2xn***\n', ImgPts_2xn
#
#     print '================ unDistortPts ================'
#     UnDisTortPts, UnDisTortRay = MySingleVision.unDistortPts(imgPts_2xn=distortProjectImgPts_2xn)
#     print 'UnDisTortPts:\n', UnDisTortPts
#     print 'UnDisTortRay:\n', UnDisTortRay
#     print '***GroundTruthImgPts_2xn***\n', projectImgPts_2xn
#
#     SrcImg = cv2.imread('./Data/cam4.png')
#     unDistortImg = MySingleVision.unDistort(img=SrcImg)
#     cv2.imshow('unDistortImg', cv2.resize(unDistortImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5)))
#     cv2.imshow('SrcImg', cv2.resize(SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5)))
#     cv2.imshow('distort', cv2.resize(unDistortImg-SrcImg, (SrcImg.shape[1]/5, SrcImg.shape[0]/5)))
#     cv2.waitKey()