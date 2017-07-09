#!/bin/bash

# Download VGG16 weights
mkdir -p DATA
wget ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy
mv vgg16.py DATA

# Download KittiSeg pretrained weights
wget ftp://mi.eng.cam.ac.uk/pub/mttt2/models/KittiSeg_pretrained.zip
unzip KittiSeg_pretrained.zip

# Make directory for graph with renamed tensors
mkdir -p renamed
