#!/usr/bin/env python
from __future__ import division, print_function

"""
IPT (Image Process Tool) is a library that includes several functions of image process.
"""

from .core import *
from . import contour_analyst

from numpy.testing.nosetester import NoseTester
test = NoseTester().test
bench = NoseTester().bench
