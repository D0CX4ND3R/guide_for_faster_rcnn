# Guide for building faster rcnn
Reproduction of Faster R-CNN by tensorflow. Guide to build an own Faster R-CNN. 
For learning and communication.
Train a toy dataset generates circles, rectangles and triangles.

# Requires
Tensorflow >= 1.11
opencv >= 3.4.0

# Usage
python train.py

# Others
The backbone is ResNeXt-50, now only supporting batch_size=1.

# Need to fix
Nan appeares during training the RPN.

# Next
* Add test codes.
* Fix Nan in training the RPN.
* Training with other datasets as COCO or VOC.
