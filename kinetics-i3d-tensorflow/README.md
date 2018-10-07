# I3D models trained on Kinetics

## From here

[repo](https://github.com/deepmind/kinetics-i3d)

## Running the code

### Setup

First follow the instructions for [installing
Sonnet](https://github.com/deepmind/sonnet).

Then, clone this repository using

`$ git clone https://github.com/deepmind/kinetics-i3d`

### code

Run the code please check the multi_evaluate.sh

### Something to say
You need to down load the checkpoint from the original repo
1. default load the kinetics pre-trained model
2. extract features in thumos14 validation and test dataset
the extract way is segment the video at uniform interval.

   frames     |     interval    | video fps
------------- | :-------------: | -----------
   <=15000    |       24        |     30
   <=30000    |       48        |     30
   > 30000    |       96        |     30

In order to reduce the redundancy in frames, we choose to subsample the video to 10fps.
A clip includes 48 frames, we sample 16 frames and send to the I3D network to extract [1,1024] features

The feature is generate after Mix_5c and avg_pool layer:

rgb: [1, 16, 224, 224, 3] -> [1024,]

flow:[1, 16, 224, 224, 2] -> [1024,]


