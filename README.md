# Guide for building faster rcnn
Reproduction of Faster R-CNN by tensorflow. Guide to build an own Faster R-CNN.
Aiming to build up faster rcnn for beginner. Easy to read and understand with comments.
Train a toy dataset generates circles, rectangles and triangles.

# Requires
Tensorflow >= 1.11
opencv >= 3.4.0

# Usage
## Training
`
python train.py
`
## Use Tensorboard
`
tensorboard --logdir=./logs
`

# Others
Batch size only supported 1.
Only the toy_dataset available.


# Next
* Add references.
* Add comments.
* Add different backbones: ~~VGG~~, MobileNet.
* Add test codes. (Complete after validate the toy dataset.)
* Training with other datasets as COCO or VOC. (Complete after validate the toy dataset.)
