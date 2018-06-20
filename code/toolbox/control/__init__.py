#!/usr/bin/env python
from __future__ import division, print_function

"""
control (File Interface Tool) is a library that includes several functions of control.
"""

from . import pid

from numpy.testing.nosetester import NoseTester
test = NoseTester().test
bench = NoseTester().bench
