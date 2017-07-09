#! /bin/bash

source py2_virtualenv/bin/activate
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cudnn-v5/cuda/lib64:$LD_LIBRARY_PATH
