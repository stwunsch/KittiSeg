#!/bin/bash

virtualenv py2_virtualenv --system-site-packages
source py2_virtualenv/bin/activate
pip install pip --upgrade
pip install tensorflow_gpu==1.1.0
pip install --upgrade protobuf
pip install --upgrade scipy
pip install commentjson
