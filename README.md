# I3D models trained on Kinetics

## Overview

This repository contains trained models reported in the paper "[Quo Vadis,
Action Recognition? A New Model and the Kinetics
Dataset](https://arxiv.org/abs/1705.07750)" by Joao Carreira and Andrew
Zisserman.

Tensorflow code is from Deepmind's [Kinetics-I3D](https://github.com/deepmind/kinetics-i3d).

Pytorch code is from [Kinetics-I3D](https://github.com/piergiaj/pytorch-i3d)

## Fine-tuning and Feature Extraction
These models were pretrained on imagenet and kinetics (see original repo) for details).

## Something to say
You need to down load the checkpoint from the original repo
1. default load the kinetics pre-trained model
2. extract features in thumos14 validation and test dataset
the extract way is segment the video at uniform interval.

frames      | interval        | video fps
----------- | :-------------: | -----------
<=15000     | 24              | 30
<=30000     | 48              | 30
\>30000     | 96              | 30

In order to reduce the redundancy in frames, we choose to subsample the video to 10fps.
A clip includes 48 frames, we sample 16 frames and send to the I3D network to extract [1,1024] features

Feature is generated after Mix_5c and avg_pool layer:

input -> output:

rgb: [1, 16, 224, 224, 3] -> [1024,]

flow:[1, 16, 224, 224, 2] -> [1024,]


