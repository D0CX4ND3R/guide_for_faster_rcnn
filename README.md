# Guide for building faster rcnn
Reproduction of Faster R-CNN by tensorflow. Guide to build an own Faster R-CNN.
Train a toy dataset generates circles, rectangles and triangles.

# Requires
Tensorflow >= 1.11
opencv >= 3.4.0

# Usage
## Training
python train.py

## Use Tensorboard
tensorboard --logdir=./logs

# Others
The backbone is ResNeXt-50, now only support batch_size=1.

# Need to fix
* ~~Nan will appeares during training the RPN.~~
* Maybe has problems in RPN generating rois and labels.
* For the toy dataset, maybe the resnext-50 is too large to train.
* When training, rcnn_bbox_loss reduces to zero quickly, accuracy increses to 1.0 quickly.

# Next
* Add references.
* Add commits.
* Add different backbones.
* Add test codes.
* ~~Fix Nan in training the RPN.~~
* Training with other datasets as COCO or VOC.
