#!/usr/bin/env python
from __future__ import division, print_function

__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '24/10/2016'
__copyright__ = "Copyright 2016, PI"


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('vgl', parent_package, top_path)
    config.add_data_dir('tests')
    config.name = 'vgl'
    config.version = '3.0.0'
    config.description='Vision Geometry Library'
    config.url=''
    config.author='Li Hao'
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
