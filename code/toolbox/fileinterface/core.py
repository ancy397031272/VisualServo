#!/usr/bin/env python
from __future__ import division, print_function
# -*- coding:utf-8 -*-
__author__ = 'hkh'
__date__ = '25/10/2016'
__version__ = '3.0'

import yaml
import pickle as pk

__all__ = [
    'loadYaml',
    'loadAllYaml',
    'dumpYaml',
    'dumpAllYaml',
    'loadPk',
    'dumpPk',
]

def loadYaml(fileName, method='r'):
    """
    Parse the first YAML document in a stream
    and produce the corresponding Python object.
    """
    with open(fileName, method) as file:
        return  yaml.load(stream=file)

def loadAllYaml(fileName, method='r'):
    """
    Parse all YAML documents in a stream
    and produce corresponding Python objects.
    """
    with open(fileName, method) as file:
        return yaml.load_all(stream=file)

def dumpYaml(data, fileName, method='w'):
    """
    Serialize a Python object into a YAML stream.
    If stream is None, return the produced string instead.
    """
    with open(fileName, method) as file:
        yaml.dump(data=data, stream=file)

def dumpAllYaml(data, fileName, method='w'):
    """
    Serialize a sequence of Python objects into a YAML stream.
    If stream is None, return the produced string instead.
    """
    with open(fileName, method) as file:
        yaml.dump_all(documents=data, stream=file)

def loadPk(fileName, method='r'):
    """
    Read a pickled object representation from the open file.
    Return the reconstituted object hierarchy specified in the file.
    """
    with open(fileName, method) as File:
        return pk.load(File)

def dumpPk(data, fileName, method='w'):
    """
    Write a pickled representation of obj to the open file.
    """
    with open(fileName, method) as File:
        pk.dump(obj=data, file=File)