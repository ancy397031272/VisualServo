#!/bin/bash
python conf.py
sphinx-apidoc -o ./ ../../ setup -e -f -d 5
rm *setup.rst
rm *Test*.rst
cd ..
make clean
make html
