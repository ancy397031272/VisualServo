from __future__ import division, print_function

__author__ = 'Li Hao'
__version__ = '3.0'
__date__ = '07/09/2016'
__copyright__ = "Copyright 2016, PI"


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('toolbox', parent_package, top_path)
    config.add_subpackage('vgl')
    config.add_subpackage('systool')
    config.add_subpackage('fileinterface')
    config.add_subpackage('imgproctool')
    config.add_subpackage('control')
    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)
