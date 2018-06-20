#!/usr/bin/env python
from __future__ import division, print_function

"""
VGL (Vision Geometry Library) is a library that includes several functions of vision project.
    It contains:
        1. Coordinate transformation.
        2. Camera geometry.
        3. Other functions...
"""

from .core import *
from . import single_vision
from . import stereo_vision
from . import eye_hand_vision
from . import visual_servo_lib

from numpy.testing.nosetester import NoseTester
test = NoseTester().test
bench = NoseTester().bench
