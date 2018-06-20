#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '07/09/2016'
__copyright__ = "Copyright 2016, PI"

import cv2
import math
import numpy as np

__all__ = [
    'VGLError',
    'isArray',
    'isRotationMatrix',
    'Rt2T',
    'T2Rt',
    'T2rt',
    'rt2T',
    'getRx',
    'getRy',
    'getRz',
    'R2Euler_zyx',
    'Euler2R',
    'Pose2T',
    'T2Pose',
    'Homo',
    'unHomo',
    'computeVectorAngle_rad',
    'computeRotateVec',
    'getTransformWith2LineSegment',
    'distBW2Lines',
    'projectPts',
    'projectPtsToImg',
    'unDistortImg',
    'unDistortImgOptimal',
    'unDistortPts',
    'reconstruct3DPts',
]


# Error object
class VGLError(Exception):
    """
    Generic Python-exception-derived object raised by Vision Geometry Lib functions.

    General purpose exception class, derived from Python's exception.Exception
    class, programmatically raised in Vision Geometry Lib functions when a Image
    Processing-related condition would prevent further correct execution of the
    function.

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> VGL.T2Rt(None)
    Traceback (most recent call last):
    ...
    VGLError: ...
    """
    pass

def __raiseError(msg=None):
    if msg is None:
        raise VGLError
    else:
        raise VGLError, msg

def __checkNone(array):
    if array is None:
        raise VGLError('array can not be [None]')

def __checkSize(array, size, msg=None):
    if array.size != size:
        if msg is None:
            raise VGLError('%d size array given. Array must be '
                    '%d size' % (array.size, size))
        else:
            raise VGLError(msg)

def __checkShape(array, shape, msg=None):
    if array.shape != shape:
        if msg is None:
            raise VGLError('%s shape array given. Array must be '
                    '%s shape' % (str(array.shape), str(shape)))
        else:
            raise VGLError(msg)

def __checkNdim(array, ndim, msg=None):
    if array.ndim != ndim:
        if msg is None:
            raise VGLError('%d-dimensional array given. Array must be '
                    'at %d-dimensional' % (array.ndim, ndim))
        else:
            raise VGLError(msg)

def __toArray(array_like, copy=True):
    __checkNone(array_like)
    if isinstance(array_like, np.ndarray):
        if copy:
            return array_like.copy()
        else:
            return array_like
    return np.array(array_like)

def isArray(array, checkSize=None):
    """
    Judge the input whether a np.ndarray.

    If checkSize is not None, will check if the array size is correct.

    Parameters
    ----------
    array : array_like
        Input data
    checkSize : tuple or list, optional
        Check mat's size, if None(default), only check mat whether ndarray

    Returns
    -------
    Judgement : bool

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> a = np.zeros((3, 3))
    >>> VGL.isArray(a, (3, 3))
    True

    """
    if not isinstance(array, np.ndarray):
        return False
    if checkSize is not None \
            and array.shape != tuple(checkSize):
        return False
    return True

def isRotationMatrix(R_3x3):
    """
    Check whether rotation matrix is correct.

    Parameters
    ----------
    R_3x3 : array_like
        Rotation matrix, (3x3)

    Returns
    -------
    Judgement : bool

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Rotation_matrix

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> VGL.isRotationMatrix(R)
    True
    >>> R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> VGL.isRotationMatrix(R)
    False
    """
    R = __toArray(R_3x3)
    __checkShape(R, (3, 3))
    ShouldBeNormOne = np.allclose(np.linalg.norm(R, axis=0), np.ones(shape=(3)))
    ShouldBePerpendicular = \
        np.allclose(np.cross(R[:, 0], R[:, 1]), R[:, 2]) \
        and np.allclose(np.cross(R[:, 1], R[:, 2]), R[:, 0]) \
        and np.allclose(np.cross(R[:, 2], R[:, 0]), R[:, 1])
    return ShouldBePerpendicular and ShouldBeNormOne

def Rt2T(R_3x3, t):
    """
    Compose rotation matrix and translation vector to transform matrix.

    Parameters
    ----------
    R_3x3 : array_like
        Rotation matrix, (3x3)
    t : array_like
        Translation vector, (3x1 or 1x3 or 3)

    Returns
    -------
    T : ndarray
        Transformation matrix, (4x4)

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> R = np.eye(3)
    >>> t = np.array([1, 2, 3]) # shape can be (3, 1), (1, 3) or (3,) or any array_like which has 3 elements
    >>> VGL.Rt2T(R, t)
    array([[ 1.,  0.,  0.,  1.],
           [ 0.,  1.,  0.,  2.],
           [ 0.,  0.,  1.,  3.],
           [ 0.,  0.,  0.,  1.]])

    See Also
    --------
    T2rt
    rt2T
    T2Rt
    """
    R = __toArray(R_3x3, copy=True)
    t_vec = __toArray(t, copy=True).ravel()
    __checkShape(R, (3, 3))
    __checkSize(t_vec, 3)

    T = np.eye(4, 4 , dtype=np.float)
    T[0:3,0:3] = R
    T[:3, -1] = t_vec
    return T

def T2Rt(T_4x4):
    """
    Decompose transform matrix to R, t.

    Parameters
    ----------
    T_4x4 - array_like
        Transformation matrix, (4x4)

    Returns
    -------
    R : ndarray
        Rotation matrix, (3x3)
    t : ndarray
        Translation vector, (3x1)

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> T = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    >>> R, t = VGL.T2Rt(T)
    >>> R
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    >>> t
    array([[ 1.],
           [ 2.],
           [ 3.]])

    See Also
    --------
    Rt2T
    T2rt
    rt2T
    """
    T = __toArray(T_4x4, copy=True).astype(np.float)
    __checkShape(T, (4, 4))
    R_3x3 = T[0:3, 0:3]
    t_3x1 = T[0:3, -1].reshape(3, 1)
    return R_3x3, t_3x1

def T2rt(T_4x4):
    """
    Convert transform matrix to rodrigues rotation vector and translation vector.

    Parameters
    ----------
    T_4x4 : array_like
        Transformation matrix, (4x4)

    Returns
    -------
    r : ndarray
        Rodrigues rotation vector, (3x1)
    t : ndarray
        Translation vector, (3x1)

    See Also
    --------
    Rt2T
    rt2T
    T2Rt

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> T = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
    >>> rVec, tVec = VGL.T2rt(T)
    >>> rVec
    array([[ 0.],
           [ 0.],
           [ 0.]])
    >>> tVec
    array([[ 1.],
           [ 2.],
           [ 3.]])
    """
    R_3x3, t_3x1 = T2Rt(T_4x4)
    r_3x1, _ = cv2.Rodrigues(R_3x3)
    r_3x1 = r_3x1.reshape(3, 1)
    return r_3x1, t_3x1

def rt2T(rVec, tVec):
    """
    Convert rodrigues rotation vector and translation vector to transform matrix.

    Parameters
    ----------
    rVec : array_like
        Rodrigues rotation vector, 3x1 or 1x3 or 3.
    tVec : array_like
        Translation vector, 3x1 or 1x3 or 3.

    Returns
    -------
    T : ndarray
        Transformation matrix, 4x4.

    See Also
    --------
    Rt2T
    T2rt
    T2Rt

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> rVec = np.array([0.78539816, 0, 0])   # shape can be (3, 1), (1, 3) or (3,) or any array_like which has 3 elements
    >>> tVec = np.array([7, 8, 9])            # shape can be (3, 1), (1, 3) or (3,) or any array_like which has 3 elements
    >>> VGL.rt2T(rVec, tVec)
    array([[ 1.        ,  0.        ,  0.        ,  7.        ],
           [ 0.        ,  0.70710678, -0.70710678,  8.        ],
           [ 0.        ,  0.70710678,  0.70710678,  9.        ],
           [ 0.        ,  0.        ,  0.        ,  1.        ]])
    """
    r_vector = __toArray(rVec)
    t_vector = __toArray(tVec)
    __checkSize(r_vector, 3)
    __checkSize(t_vector, 3)
    R,_ = cv2.Rodrigues(r_vector)
    T_4x4 = Rt2T(R, t_vector)
    return T_4x4

def getRx(angle_rad):
    """
    Computes rotation matrix which rotate around x-axis.

    Parameters
    ----------
    angle_rad : float
        rotation angle, radian.

    Returns
    -------
    R : ndarray
        Rotation matrix, 3x3

    See Also
    --------
    getRy
    getRz
    Euler2R
    R2Euler_zyx

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> Angle_rad = 3.141592 / 4
    >>> VGL.getRx(Angle_rad)
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.7071069 , -0.70710667],
           [ 0.        ,  0.70710667,  0.7071069 ]])
    """
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    Rx = np.array([[1, 0,    0],
                   [0, ca, -sa],
                   [0, sa,  ca]], np.float)
    return Rx

def getRy(angle_rad):
    """
    Computes rotation matrix which rotate around y-axis.

    Parameters
    ----------
    angle_rad : float
        rotation angle, radian.

    Returns
    -------
    R : ndarray
        Rotation matrix, 3x3

    See Also
    --------
    getRx
    getRz
    Euler2R
    R2Euler_zyx

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> Angle_rad = 3.141592 / 4
    >>> VGL.getRy(Angle_rad)
    array([[ 0.7071069 ,  0.        ,  0.70710667],
           [ 0.        ,  1.        ,  0.        ],
           [-0.70710667,  0.        ,  0.7071069 ]])
    """
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    Ry = np.array([ [ca,  0,  sa],
                    [0,   1,   0],
                    [-sa, 0,  ca]], np.float)
    return Ry

def getRz(angle_rad):
    """
    Computes rotation matrix which rotate around z-axis.

    Parameters
    ----------
    angle_rad : float
        rotation angle, radian.

    Returns
    -------
    R : ndarray
        Rotation matrix, 3x3

    See Also
    --------
    getRx
    getRy
    Euler2R
    R2Euler_zyx

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> Angle_rad = 3.141592 / 4
    >>> VGL.getRz(Angle_rad)
    array([[ 0.7071069 , -0.70710667,  0.        ],
           [ 0.70710667,  0.7071069 ,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])
    """
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    Rz = np.array([ [ca,  -sa,  0],
                    [sa,   ca,  0],
                    [0,    0,   1]], np.float)
    return Rz

def R2Euler_zyx(R_3x3, checkR=True):
    """
    Convert rotation matrix to 3 Euler angle.

    The way of decomposition is according to z-y-x Euler angles, rotated by intrinsic mode.
    If Beta is 90 deg, singular, also called "Gimbal-lock", fix Alpha = 0 to calculate Gamma.

    Parameters
    ----------
    R_3x3 : array_like
        Rotation matrix, (3x3)
    checkR : bool, optional
        Flag indicating whether check rotation matrix is valid or not, if not, raise error. Default is True.

    Returns
    -------
    Alpha : float
        Rotate Alpha-angle around z firstly, radian.
    Beta : float
        Rotate Beta-angle around y secondly, radian.
    Gamma : float
        Rotate Gamma-angle around x lastly, radian.

    Raises
    ------
    VGLError:
        If checkR is True and rotation matrix R_3x3 is invalid.

    See Also
    --------
    getRx
    getRy
    getRz
    Euler2R

    References
    ----------
    .. [1] https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    .. [2] https://en.wikipedia.org/wiki/Gimbal_lock

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> import numpy as np
    >>> GT = np.array([np.pi/2, np.pi/4, np.pi/4])
    >>> Spe = np.sqrt(2) / 2
    >>> R = np.array([[0, -Spe, Spe], [Spe, 0.5, 0.5], [-Spe, 0.5, 0.5]], np.float)
    >>> Alpha, Beta, Gamma = VGL.R2Euler_zyx(R)
    >>> np.allclose(GT, np.array([Alpha, Beta, Gamma]), atol=1e-6)
    True
    """
    if checkR and not isRotationMatrix(R_3x3):
        __raiseError('R_3x3 must be rotation matrix')
    R = __toArray(R_3x3)
    Sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    Singular = Sy < 1e-6
    if Singular:
        Alpha_rad = 0
        Beta_rad = math.atan2(-R[2, 0], Sy)
        Gamma_rad = math.atan2(-R[1, 2], R[1, 1])
    else:
        Alpha_rad = math.atan2(R[1, 0], R[0, 0])
        Beta_rad = math.atan2(-R[2, 0], Sy)
        Gamma_rad = math.atan2(R[2, 1], R[2, 2])
    return Alpha_rad, Beta_rad, Gamma_rad

def Euler2R(alpha_rad, beta_rad, gamma_rad, sequence='zyx', mode='intrinsic'):
    """
    Convert 3 Euler angle to rotation matrix.

    The way of convert is according to z-y-x Euler angles.

    Parameters
    ----------
    alpha_rad : float
        Rotation alpha-angle around order[0] firstly, radian.
    beta_rad : float
        Rotation alpha-angle around order[1] secondly, radian.
    gamma_rad : float
        Rotation alpha-angle around order[2] finally, radian.
    sequence : str, optional
        Axis rotation sequence for the Euler angles.
        Default is 'zyx' - The order of rotation angles is z-axis, y-axis, x-axis.
    mode : str, optional
        - 'intrinsic' (default), rotation about the rotating coordinate frame.
        - 'extrinsic', rotation about the fix coordinate frame.

    Returns
    -------
    R_3x3 : ndarray
        Rotation matrix, 3x3

    Raises
    ------
    VGLError:
        Sequence or mode invalid.

    See Also
    --------
    getRx
    getRy
    getRz
    R2Euler_zyx

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Euler_angles

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> import numpy as np
    >>> a = np.pi/2
    >>> b = np.pi/4
    >>> c = np.pi/4
    >>> Frame = np.eye(3)
    >>> GT = np.array([[0, -0.707, 0.707], [0.707, 0.5, 0.5], [-0.707, 0.5, 0.5]], np.float)
    >>> R = VGL.Euler2R(a, b, c, sequence='zyx', mode='intrinsic')
    >>> Rotated = np.dot(Frame, R)  # rotate
    >>> np.allclose(R, GT, atol=1e-3)
    True
    >>> R = VGL.Euler2R(c, b, a, sequence='xyz', mode='extrinsic')
    >>> Rotated = np.dot(R, Frame)  # rotate
    >>> np.allclose(R, GT, atol=1e-3)
    True
    """
    Mode_lower = mode.lower()
    if not isinstance(sequence, str):
        __raiseError('order must be string')
    if len(sequence) != 3:
        __raiseError('order must contain 3 element')
    for c in list(sequence):
        if c not in ('x', 'y', 'z'):
            __raiseError("order must be 'x', 'y' or 'z'")
    MapDic = {'x': getRx,
              'y': getRy,
              'z': getRz}
    R1 = MapDic[sequence[0]](alpha_rad)
    R2 = MapDic[sequence[1]](beta_rad)
    R3 = MapDic[sequence[2]](gamma_rad)
    if Mode_lower == 'extrinsic':
        return R3.dot(R2).dot(R1)
    elif Mode_lower == 'intrinsic':
        return R1.dot(R2).dot(R3)
    else:
        __raiseError('mode must be [intrinsic] or [extrinsic]')

def Pose2T(pose):
    """
    Convert pose to transform matrix.

    Parameters
    ----------
    pose : array_like
        Pose of coordinate {B} in coordinate {A}, [x, y, z, u, v, w], 6x1 or 1x6 or 6.

    Returns
    -------
    T_4x4 : ndarray
        Transform matrix :math:`T_B^A` which map point from coordinate {B} to coordinate {A}, 4x4.

    See Also
    --------
    T2Pose

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> import numpy as np
    >>> Pose = [10, 20, 30, 30, 45, 30]  # 6x1 or 1x6 or 6 array_like
    >>> GT = np.array([[ 0.61237242, -0.12682648,  0.78033010, 10], [ 0.35355339,  0.92677670, -0.12682648, 20], \
[-0.70710680,  0.35355339,  0.61237242, 30], [ 0.00000000,  0.00000000,  0.00000000,  1]])
    >>> T = VGL.Pose2T(Pose)
    >>> np.allclose(GT, T, atol=1e-6)
    True
    """
    Pose = __toArray(pose, copy=True).astype(np.float).ravel()
    __checkSize(Pose, 6)

    X, Y, Z, U_deg, V_deg, W_deg = Pose
    U_rad = np.deg2rad(U_deg)
    V_rad = np.deg2rad(V_deg)
    W_rad = np.deg2rad(W_deg)
    R = Euler2R(U_rad, V_rad, W_rad, sequence='zyx', mode='intrinsic')
    t = np.array([[X], [Y], [Z]], np.float)
    T = Rt2T(R_3x3=R, t=t)
    return T

def T2Pose(T_4x4, checkR=True):
    """
    Convert transform matrix to pose.

    Parameters
    ----------
    T_4x4 : array_like
        Transform matrix T_BA which map point from coordinate {B} to coordinate {A}, 4x4.
    checkR : bool, optional
        If this is set to True, will check rotation matrix, if rotation matrix is invalid, raise error.

    Returns
    -------
    Pose_6x1 : ndarray
        Pose of coordinate {B} in coordinate {A}, [x, y, z, u, v, w], (6x1).

    See Also
    --------
    Pose2T

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> import numpy as np
    >>> GT = np.array([[50], [60], [70], [30], [45], [30]])
    >>> T = np.array([[ 0.61237242, -0.12682648,  0.78033010, 50], [ 0.35355339,  0.92677670, -0.12682648, 60], \
[-0.70710680,  0.35355339,  0.61237242, 70], [ 0.00000000,  0.00000000,  0.00000000,  1]])
    >>> Pose = VGL.T2Pose(T)   # 6x1
    >>> np.allclose(Pose, GT, atol=1e-6)
    True
    """
    R, t = T2Rt(T_4x4=T_4x4)
    AngleZ_rad, AngleY_rad, AngleX_rad = R2Euler_zyx(R, checkR=checkR)
    AngleXYZ_deg = np.rad2deg([AngleZ_rad, AngleY_rad, AngleX_rad])
    Pose = np.hstack((t.ravel(), AngleXYZ_deg))
    return np.array(Pose, dtype=np.float).reshape(6, 1)

def Homo(points_dxn):
    """
    Converts points from Euclidean to homogeneous space.

    Parameters
    ----------
    points_dxn : array_like
        Points in euclidean space, dxn.(d - dimension,
                                        n - points number)

    Returns
    -------
    points_(d-1)xn : ndarray
        Points in homogeneous space, (d-1)xn.

    See Also
    --------
    unHomo

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> Pts = np.array([[10, 20, 50], [20, 30, 70]])
    >>> VGL.Homo(Pts)
    array([[ 10.,  20.,  50.],
           [ 20.,  30.,  70.],
           [  1.,   1.,   1.]])
    """
    Points_dxn = __toArray(points_dxn, copy=True)
    __checkNdim(Points_dxn, 2)
    return np.vstack((Points_dxn, np.ones((1, Points_dxn.shape[1]))))

def unHomo(points_dxn):
    """
    Converts points from homogeneous to Euclidean space.

    Parameters
    ----------
    points_dxn : array_like
        Points in homogeneous space, dxn.(d - dimension,
                                          n - points number)

    Returns
    -------
    points_(d+1)xn : ndarray
        Points in Euclidean space, (d+1)xn.

    See Also
    --------
    Homo

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Homogeneous_coordinates

    Examples
    --------
    >>> import numpy as np
    >>> from toolbox import vgl as VGL
    >>> Pts = np.array([[10, 20, 50], [20, 30, 70], [ 5, 10,  5]])
    >>> VGL.unHomo(Pts)
    array([[  2.,   2.,  10.],
           [  4.,   3.,  14.]])
    """
    Points_dxn = __toArray(points_dxn, copy=True)
    __checkNdim(Points_dxn, 2)

    Temp = Points_dxn.astype(np.float)
    Temp /= Temp[-1]
    return Temp[0:-1]

def computeVectorAngle_rad(vec0, vec1):
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    vec0 : array_like
        First Vector, nx1 or 1xn or n
    vec1 : array_like
        Second Vector, nx1 or 1xn or n

    Returns
    -------
    Angle_rad : float
        Angle between vec0 and vec1, radian.

    See Also
    --------
    computeRotateVec

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> Vec0 = [0, 1]
    >>> Vec1 = [1, 0]
    >>> VGL.computeVectorAngle_rad(Vec0, Vec1)
    1.57...
    >>> Vec0 = [0.5, 0, 0.5]
    >>> Vec1 = [0.5, 0.5, 0]
    >>> VGL.computeVectorAngle_rad(Vec0, Vec1)
    1.047...
    """
    Vec0 = __toArray(vec0, copy=False).ravel()
    Vec1 = __toArray(vec1, copy=False).ravel()
    __checkSize(Vec0, Vec1.size, 'vec0.size != vec1.size')

    VecNorm0 = np.linalg.norm(Vec0)
    VecNorm1 = np.linalg.norm(Vec1)
    Acos = np.inner(Vec0, Vec1) / (VecNorm0 * VecNorm1)
    if Acos > 1:
        Acos = 1.0
    elif Acos < -1:
        Acos = -1.0
    Angle_rad =  math.acos(Acos)
    return Angle_rad

def computeRotateVec(vec0, vec1, rotateAxis=None):
    """
    Compute the Rodrigues rotation vector between two vectors.

    Parameters
    ----------
    vec0 : array_like
        First Vector, 3x1 or 1x3 or 3.
    vec1 : array_like
        Second Vector, 3x1 or 1x3 or 3
    rotateAxis : array_like, optional
        Default is None, when vec0 and vec1 are in a line with opposite direction, use given rotate axis, 3x1 or 1x3 or 3.

    Returns
    -------
    RodriguesVec : ndarray
        Rodrigues rotation vector, 3x1.

    Raises
    ------
    VGLError:
        One of vec0 and vec1 is zero vector, or vec0 and vec1 are in a line with opposite direction.

    See Also
    --------
    computeVectorAngle_rad

    References
    ----------
    .. [1] http://www.euclideanspace.com/maths/geometry/rotations/axisAngle/index.htm

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> Vec0 = [0, 0, 1]
    >>> Vec1 = [0, 1, 0]
    >>> VGL.computeRotateVec(Vec0, Vec1)
    array([[-1.57...],
           [ 0.        ],
           [ 0.        ]])
    """
    Vec0 = __toArray(vec0, copy=False).ravel()
    Vec1 = __toArray(vec1, copy=False).ravel()
    __checkSize(Vec0, Vec1.size, 'vec0.size != vec1.size')
    if np.linalg.norm(Vec0) == 0:
        __raiseError('vec0 can not be zero vector')
    if np.linalg.norm(Vec1) == 0:
        __raiseError('vec1 can not be zero vector')

    Theta = computeVectorAngle_rad(Vec0, Vec1)
    AbsTheta = math.fabs(Theta)
    # is theta near 0
    if AbsTheta < 1e-7:
        return np.zeros((3, 1), dtype=np.float)
    # is theta near +- pi?
    if np.pi - 1e-7 < AbsTheta < np.pi + 1e-7:
        # given rotate axis?
        if rotateAxis is not None:
            RotateAxis = __toArray(rotateAxis, copy=False).ravel().astype(np.float)
            # is rotate axis valid?
            if math.fabs(np.inner(RotateAxis, Vec0)) < 1e-7:
                RotateAxisNorm = RotateAxis / np.linalg.norm(RotateAxis)
                RodriguesVec = RotateAxisNorm * np.pi
                return RodriguesVec
            else:
                __raiseError('vec0 vec1 are in a line with opposite direction,given rotate axis is invalid.')
        else:
            __raiseError('vec0 vec1 are in a line with opposite direction,but no given rotate axis.')
    # normal
    OrthogonalVec = np.cross(Vec0, Vec1)
    NormOrthogonalVec = OrthogonalVec / np.linalg.norm(OrthogonalVec)
    RodriguesVec = (NormOrthogonalVec * Theta).reshape(3, 1)
    return RodriguesVec

def getTransformWith2LineSegment(lineSegPts0_3x2, lineSegPts1_3x2):
    """
    Compute the movement from line-segment-0 to line-segment-1.

    Parameters
    ----------
    lineSegPts0_3x2 : array_like
        Line presented by two 3x1 points.
    lineSegPts1_3x2 : array_like
        Line presented by two 3x1 points

    Returns
    -------
    T_4x4 : ndarray
        Transform matrix, 4x4.

    Raises
    ------
    VGLError:
        If vec0 or vec1 is zero vector, or vec0 and vec1 lie in the same line.

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> import numpy as np
    >>> a = [[0, 5], [0, 0], [0, 0]]
    >>> b = [[0, 0], [0, 0], [0, 5]]
    >>> GT = np.array([[0, 0, -1, 0], [0, 1,  0, 0], [1, 0,  0, 0], [0, 0,  0, 1]])
    >>> T = VGL.getTransformWith2LineSegment(a, b)
    >>> np.allclose(T, GT)
    True
    >>> TransLine = np.dot(T, np.vstack((a, np.ones((1, 2)))))[:3, :]
    >>> np.allclose(TransLine, b)
    True
    """
    LineSegPts0_3x2 = __toArray(lineSegPts0_3x2, copy=False)
    LineSegPts1_3x2 = __toArray(lineSegPts1_3x2, copy=False)
    __checkShape(LineSegPts0_3x2, (3, 2))
    __checkShape(LineSegPts1_3x2, (3, 2))

    Vec0 = LineSegPts0_3x2[:, 1] - LineSegPts0_3x2[:, 0]
    Vec1 = LineSegPts1_3x2[:, 1] - LineSegPts1_3x2[:, 0]
    RotateVec = computeRotateVec(vec0=Vec0, vec1=Vec1)
    RotateMatrix, _ = cv2.Rodrigues(RotateVec)
    t = LineSegPts1_3x2[:, 0] - np.dot(RotateMatrix , LineSegPts0_3x2[:, 0])
    T_4x4 = Rt2T(RotateMatrix, t)
    return T_4x4

def distBW2Lines(line0_dx2, line1_dx2):
    """
    Compute shortest distance between two lines.

    Parameters
    ----------
    line0_dx2 : array_like
        Line presented by two dimension-by-1 points, dx2.
    line1_dx2 : array_like
        Line presented by two dimension-by-1 points, dx2.

    Returns
    -------
    Distance: float
        Distance between two lines.
    IntersectPt0: ndarray
        Intersect point lie in line0, dx1.
    IntersectPt1: ndarray
        Intersect point lie in line1, dx1.

    References
    ----------
    .. [1] http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> a = [[0, 0], [1, 5]]
    >>> b = [[10, 10], [-1, -5]]
    >>> Dist, Point0, Point1 = VGL.distBW2Lines(a, b)
    >>> Dist
    10.0
    >>> Point0
    array([[ 0.],
           [ 1.]])
    >>> Point1
    array([[ 10.],
           [  1.]])
    >>> a = [[0, 0], [1, 5], [0, 0]]
    >>> b = [[10, 10], [-1, -1], [7, 9]]
    >>> Dist, Point0, Point1 = VGL.distBW2Lines(a, b)
    >>> Dist
    10.0
    >>> Point0
    array([[ 0.],
           [-1.],
           [ 0.]])
    >>> Point1
    array([[ 10.],
           [ -1.],
           [  0.]])
    """
    Line0 = __toArray(line0_dx2).astype(np.float)
    Line1 = __toArray(line1_dx2).astype(np.float)
    __checkNdim(Line0, 2)
    __checkNdim(Line1, 2)
    if Line0.shape[1] != 2:
        __raiseError('Line0 must be nx2')
    if Line1.shape[1] != 2:
        __raiseError('Line1 must be nx2')
    P0 = Line0[:, 0]
    P1 = Line0[:, 1]
    Q0 = Line1[:, 0]
    Q1 = Line1[:, 1]
    u = P1 - P0
    v = Q1 - Q0
    w0 = P0 - Q0

    a = np.inner(u, u) # always >= 0
    b = np.inner(u, v)
    c = np.inner(v, v) # always >= 0
    d = np.inner(u, w0)
    e = np.inner(v, w0)
    D = a*c - b*b      # always >= 0
    # compute the line parameters of the two closest points
    if (D < 1e-7):
        sc = 0.0
        if b > c:
            tc = d / b
        else:
            tc = e / c
    else:
        sc = (b*e - c*d) / D
        tc = (a*e - b*d) / D

    IntersectPt0 = (P0 + sc * u).reshape(Line0.shape[0], 1)
    IntersectPt1 = (Q0 + tc * v).reshape(Line1.shape[0], 1)
    Dist = np.linalg.norm(IntersectPt0 - IntersectPt1)
    return Dist, IntersectPt0, IntersectPt1

def projectPts(pts_dxn, projectMatrix):
    """
    Project n-dimension points using project matrix.\n
    :math:`ProjectedPts_{d*n} = projectMatrix_{d*d}*pts_{d*n}`, or\n
    :math:`ProjectedPts_{d*n} = unHomo(projectMatrix_{(d+1)*(d+1)}*Homo(pts_{d*n}))`

    Parameters
    ----------
    pts_dxn : array_like
        Points which are dimension-by-number shape in coordinate {A}, dxn.
    projectMatrix : array_like
        Project matrix which project points from coordinate {A} to anther space, dxd or d+1xd+1.

    Raises
    ------
    VGLError:
        When project matrix is not as square matrix, or shapes of points and project matrix are not aligned.

    Returns
    -------
    ProjectedPts_dxn : ndarray
        Points which in projected space, dxn.

    Examples
    --------
    >>> from toolbox import vgl as VGL
    >>> PtsInA_2xn = [[0, 1, 2, 3], [4, 5, 6, 7]]
    >>> TAB = [[0, 1, 1], [-1, 0, 2], [0, 0, 1]]
    >>> PtsInB_2xn = VGL.projectPts(PtsInA_2xn, TAB)
    >>> PtsInB_2xn
    array([[ 5.,  6.,  7.,  8.],
           [ 2.,  1.,  0., -1.]])
    >>> PtsInA_3xn = [[0, 1, 2, 3], [2, 3, 4, 5], [5, 6, 7, 8]]
    >>> TAB = [[0, 1, 0, 1], [-1, 0, 0, 2], [0, 0, 1, -1], [0, 0, 0, 1]]
    >>> PtsInB_3xn = VGL.projectPts(PtsInA_3xn, TAB)
    >>> PtsInB_3xn
    array([[ 3.,  4.,  5.,  6.],
           [ 2.,  1.,  0., -1.],
           [ 4.,  5.,  6.,  7.]])
    """
    ProjectMat = __toArray(projectMatrix, copy=False)
    Pts = __toArray(pts_dxn, copy=False)
    __checkNdim(ProjectMat, 2)
    __checkNdim(Pts, 2)
    Shape1 = ProjectMat.shape
    Shape2 = Pts.shape
    if Shape1[0] != Shape1[1]:
        __raiseError('projectMatrix must be square matrix.')
    if Shape1[1] == Shape2[0]:
        return ProjectMat.dot(Pts)
    elif Shape1[1] == (Shape2[0] + 1):
        return unHomo(ProjectMat.dot(Homo(Pts)))
    else:
        __raiseError('shapes %s and %s not aligned!' % (str(Shape1), str(Shape2)))

def projectPtsToImg(pts_3xn, Tx2Cam, cameraMatrix, distCoeffs):
    """
    Project 3D points from coordinate {x} to camera image.

    Parameters
    ----------
    pts_3xn : array_like
        3D points in {x} coordinate, 3xn.
    Tx2Cam : array_like
        Transform matrix from coordinate {x} to camera coordinate, 4x4.
    cameraMatrix : array_like
        Camera matrix / Intrinsic matrix, 3x3.
    distCoeffs : array_like
        Input vector of distortion coefficients :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])` \
        of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    Returns
    -------
    ProjImgPts_2xn : ndarray
        Points in image coord (u, v), 2xn.
    """
    CameraMatrix = __toArray(cameraMatrix, copy=False).astype(np.float)
    DistCoeffs = __toArray(distCoeffs, copy=False).astype(np.float)
    Pts_3xn = __toArray(pts_3xn, copy=False).astype(np.float)
    Pts_nx3 = Pts_3xn.T.reshape(-1, 3)
    __checkShape(CameraMatrix, (3, 3))

    R, t = T2Rt(T_4x4=Tx2Cam)
    Rvec, _ = cv2.Rodrigues(R)
    ProjImgPts_nx2, _ = \
        cv2.projectPoints(objectPoints=Pts_nx3, rvec=Rvec.astype(np.float),
                          tvec=t.astype(np.float), cameraMatrix=CameraMatrix, distCoeffs=DistCoeffs)
    ProjImgPts_2xn = ProjImgPts_nx2.T.reshape(2, -1)
    return ProjImgPts_2xn

def unDistortPts(imgPts_2xn, cameraMatrix, distCoeffs):
    """
    Transforms points in image to compensate for lens distortion.

    Parameters
    ----------
    imgPts_2xn : array_like
        Points in image, 2xn
    cameraMatrix : array_like
        Camera matrix / Intrinsic matrix, 3x3
    distCoeffs : array_like
        Input vector of distortion coefficients :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])` \
        of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    Returns
    -------
    UnDistortPts_2xn : ndarray
        Points in image after compensated.
    UnDistortRay_2xn : ndarray
        Rays in camera coordinate after compensated.

    See Also
    --------
    unDistortImg
    """
    ImgPts_2xn = __toArray(imgPts_2xn, copy=False).astype(np.float)
    CameraMatrix = __toArray(cameraMatrix, copy=False).astype(np.float)
    DistCoeffs = __toArray(distCoeffs, copy=False).astype(np.float)
    if 0 == DistCoeffs.size:
        DistCoeffs = (0, 0, 0, 0)
    __checkNdim(ImgPts_2xn, 2)
    __checkShape(CameraMatrix, (3, 3))
    if 2 != ImgPts_2xn.shape[0]:
        __raiseError('imgPts_2xn must be 2xn')

    ImgPts_1xnx2 = ImgPts_2xn.T.reshape(1, -1, 2)
    UnDistortRay_nx2 = cv2.undistortPoints(src=ImgPts_1xnx2, cameraMatrix=CameraMatrix, distCoeffs=DistCoeffs)
    UnDistortRay_2xn = UnDistortRay_nx2[0].T
    UnDistortPts_2xn = unHomo(CameraMatrix.dot(Homo(UnDistortRay_2xn)))
    return UnDistortPts_2xn, UnDistortRay_2xn

def unDistortImg(img, cameraMatrix, distCoeffs):
    """
    Transforms an image to compensate for lens distortion.

    Parameters
    ----------
    img : ndarray
        Original image.
    cameraMatrix : array_like
        Camera matrix / Intrinsic matrix, 3x3
    distCoeffs : array_like
        Input vector of distortion coefficients :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])` \
        of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.

    Returns
    -------
    unDistortImg : ndarray
        Image after compensated.

    See Also
    --------
    unDistortImgOptimal
    unDistortPts
    """
    Img = __toArray(img, copy=False)
    CameraMatrix = __toArray(cameraMatrix, copy=False).astype(np.float)
    DistCoeffs = __toArray(distCoeffs, copy=False).astype(np.float)
    __checkShape(CameraMatrix, (3, 3))
    if 0 == DistCoeffs.size:
        DistCoeffs = (0, 0, 0, 0)

    unDistortImg = cv2.undistort(src=Img, cameraMatrix=CameraMatrix, distCoeffs=DistCoeffs)
    return unDistortImg

def unDistortImgOptimal(img, cameraMatrix, distCoeffs, imgSize, alpha=None):
    """
    Transforms an image to compensate for lens distortion.

    Parameters
    ----------
    img : ndarray
        Original image.
    cameraMatrix : array_like
        Camera matrix / Intrinsic matrix, 3x3
    distCoeffs : array_like
        Input vector of distortion coefficients :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])` \
        of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
    imgSize : array_like
        Image size which contain 2 element: width and height.
    alpha : float, optional
        Free scaling parameter. If it is None, the function performs the default scaling. Otherwise,
        the parameter should be between 0 and 1.
        alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible
        (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted
        so that all the pixels from the original images from the cameras are retained in the rectified images
        (no source image pixels are lost).
        Obviously, any intermediate value yields an intermediate result between those two extreme cases.

    Returns
    -------
    unDistortImg : ndarray
        Image after compensated.
    NewCameraMatrix : ndarray
        New camera matrix based on the free scaling parameter.
    Roi_xywh : ndarray
        Valid pixel Roi, shape 4.

    See Also
    --------
    unDistortImg
    unDistortPts
    """
    Img = __toArray(img, copy=False)
    CameraMatrix = __toArray(cameraMatrix, copy=False).astype(np.float)
    DistCoeffs = __toArray(distCoeffs, copy=False).astype(np.float)
    ImgSize = __toArray(imgSize, copy=False).ravel()
    __checkShape(CameraMatrix, (3, 3))
    __checkSize(ImgSize, 2)
    if 0 == DistCoeffs.size:
        DistCoeffs = (0, 0, 0, 0)
    w, h = ImgSize
    if alpha is not None:
        if alpha > 1:
            Alpha = 1.0
        elif alpha < 0:
            Alpha = 0.0
        else:
            Alpha = float(alpha)
    else:
        Alpha = -1

    NewCameraMatrix, Roi = \
        cv2.getOptimalNewCameraMatrix(cameraMatrix=CameraMatrix, distCoeffs=DistCoeffs, imageSize=(w, h), alpha=Alpha)
    unDistortImg = cv2.undistort(Img, CameraMatrix, distCoeffs=DistCoeffs, newCameraMatrix=NewCameraMatrix)
    Roi_xywh = __toArray(Roi, copy=False)
    return unDistortImg, NewCameraMatrix, Roi_xywh

def reconstruct3DPts(imgPtsA_2xn, imgPtsB_2xn, cameraMatrixA, cameraMatrixB,
                     distCoeffsA, distCoeffsB, Tx2CamA, Tx2CamB, calcReprojErr=False):
    """
    Reconstructs points by triangulation.

    Parameters
    ----------
    imgPtsA_2xn : array_like
        Points in image, 2xn.
    imgPtsB_2xn : array_like
        Points in image, 2xn.
    cameraMatrixA : array_like
        Camera matrix / Intrinsic matrix of camera-A, 3x3.
    cameraMatrixB : array_like
        Camera matrix / Intrinsic matrix of camera-B, 3x3.
    distCoeffsA : array_like
        Input vector of distortion coefficients :math:`(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6]])` \
        of 4, 5, or 8 elements. If the vector is NULL/empty, the zero distortion coefficients are assumed.
        Distortion coefficients of camera-A.
    distCoeffsB : array_like
        Distortion coefficients of camera-A.
    Tx2CamA : array_like
        Transform matrix from coordinate {x} to camera-A coordinate, 4x4.
    Tx2CamB : array_like
        Transform matrix from coordinate {x} to camera-B coordinate, 4x4.
    calcReprojErr : bool, optional
        Calculate reprojected error or not.(default is False)

    Returns
    -------
    PtsInCoordinateX_3xn : ndarray
        Reconstructed points in {X} coordinate, 3xn.
    ReprojedErrA : float
        If calcReprojErr is True, return reprojected error of camera-A.
        Else, return None.
    ReprojedErrB : float
        Same as ReprojedErrA
    """
    Tx2CamA_4x4 = __toArray(Tx2CamA, copy=False).astype(np.float)
    Tx2CamB_4x4 = __toArray(Tx2CamB, copy=False).astype(np.float)
    CameraMatrixA = __toArray(cameraMatrixA, copy=False)
    CameraMatrixB = __toArray(cameraMatrixB, copy=False)
    ImgPtsA_2xn = __toArray(imgPtsA_2xn, copy=False)
    ImgPtsB_2xn = __toArray(imgPtsB_2xn, copy=False)
    DistCoeffsA = __toArray(distCoeffsA, copy=False)
    DistCoeffsB = __toArray(distCoeffsB, copy=False)
    UnDistortPtsA_2xn, UnDistortRaysA_2xn\
        = unDistortPts(imgPts_2xn=ImgPtsA_2xn, cameraMatrix=CameraMatrixA, distCoeffs=DistCoeffsA)
    UnDistortPtsB_2xn, UnDistortRaysB_2xn\
        = unDistortPts(imgPts_2xn=ImgPtsB_2xn, cameraMatrix=CameraMatrixB, distCoeffs=DistCoeffsB)
    PtsInCoordinateX_4xn = \
        cv2.triangulatePoints(projMatr1=Tx2CamA_4x4[0:3],
                              projMatr2=Tx2CamB_4x4[0:3],
                              projPoints1=UnDistortRaysA_2xn.astype(np.float),
                              projPoints2=UnDistortRaysB_2xn.astype(np.float))
    PtsInCoordinateX_3xn = unHomo(PtsInCoordinateX_4xn)

    if calcReprojErr:
        PtsInCoordinateX_3xn_Norm = PtsInCoordinateX_3xn
        ProjImgPtsA_2xn = \
            projectPtsToImg(pts_3xn=PtsInCoordinateX_3xn_Norm,
                                Tx2Cam=Tx2CamA_4x4, cameraMatrix=CameraMatrixA, distCoeffs=DistCoeffsA)
        RpErrA = np.linalg.norm(UnDistortPtsA_2xn - ProjImgPtsA_2xn, axis=0).mean()

        ProjImgPtsB_2xn = \
            projectPtsToImg(pts_3xn=PtsInCoordinateX_3xn_Norm,
                                Tx2Cam=Tx2CamB_4x4, cameraMatrix=CameraMatrixB, distCoeffs=DistCoeffsB)
        RpErrB = np.linalg.norm(UnDistortPtsB_2xn - ProjImgPtsB_2xn, axis=0).mean()
        return PtsInCoordinateX_3xn, RpErrA, RpErrB
    return PtsInCoordinateX_3xn, None, None
