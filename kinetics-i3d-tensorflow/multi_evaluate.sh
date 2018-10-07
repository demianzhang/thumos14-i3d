#!/usr/bin/env bash

python evaluate_sample.py -mode flow  -gpu 0 -root /media/zjg/workspace/video/test -split /media/zjg/workspace/action/data/thumos14_test.json -save_dir /media/zjg/workspace/action/npy/
python evaluate_sample.py -mode flow  -gpu 1 -root /media/zjg/workspace/video/validation -split /media/zjg/workspace/action/data/thumos14_validation.json -save_dir /media/zjg/workspace/action/npy/
