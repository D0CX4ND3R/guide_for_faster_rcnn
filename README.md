#Faster R-CNN guide

Faster R-CNN is a really classical two-stage framework for object detection. Common tensorflow versions are often 
in complicated class implementation. For beginner or fresher, reading those codes is very hard. So I build this repo
reproducing the Faster R-CNN by tensorflow without any complicated class, just functions and sufficient comments. 
It is easy to read and understand. The project can guide you to build your own Faster R-CNN framework.
Hope you like this. You also can give any advise for this repo, welcome to technical exchange (as well as my English
mistakes). :)

#Features

* No class implementations
* Sufficient comments
* Easy to read and understand
* Tensorboard shows rpn and rcnn results
* Training in a toy dataset, no download large datasets
* Training in large dataset(late support)

#Requires

Tensorflow >= 1.11
opencv >= 3.4.0

#Usage

##Training toy dataset

1. Batch size == 1
`
python train.py
`
2. Batch size > 1
`
python train_batch.py
`
* COCO is not available.

##Testing

`
python test.py
`

##Use Tensorboard

`
tensorboard --logdir=./logs
`

# Next

* Add references.
* Add MobileNet.
* Training with other datasets as COCO or VOC.
