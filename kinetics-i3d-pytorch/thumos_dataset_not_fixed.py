import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random

import os
import os.path

import cv2

# test "video_test_0001292": {"actions": [], "frame": 6974},

image_tmpl="img_{:05d}.jpg"
flow_tmpl="{}_{:05d}.jpg"

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num, 3):
    img = cv2.imread(os.path.join(image_dir, vid, image_tmpl.format(i)))#[:, :, [2, 1, 0]]
    # print(os.path.join(image_dir, vid, image_tmpl.format(i)))
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num, 3):
    imgx = cv2.imread(os.path.join(image_dir, vid, flow_tmpl.format('flow_x', i)), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, flow_tmpl.format('flow_y', i)), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(split_file, split, root, mode, num_classes=20):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if not os.path.exists(os.path.join(root, vid)):
            continue
        # num_frames = len(os.listdir(os.path.join(root, vid)))
        num_frames = data[vid]["frame"]
        label = np.zeros(1, np.int64)

        fps = 30
        # for ann in data[vid]['actions']:
        #     for fr in range(0,num_frames,1):
        #         if fr/fps > ann[1] and fr/fps < ann[2]:
        #             label[ann[0], fr] = 1 # binary classification

        tmp = data[vid]['actions'][0][0]
        label[0]=tmp-1
        dataset.append((vid, label, num_frames))
        i += 1
    
    return dataset


class Thumos(data_utl.Dataset):
    # split in [validation,test]
    def __init__(self, split_file, split, root, mode, transforms=None, save_dir='', num=0):
        
        self.data = make_dataset(split_file, split, root, mode)
        self.split_file = split_file
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label, nf = self.data[index]
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid

        clips=[]
        start_gt = []
        during = 24
        if nf>15000: # over 15000 frames
            during = 48
        if nf>30000:
            during = 96
        for i in range(0,nf+1,during): # not fixed segment number
            if i+49>nf:
               continue
            else:
                start_frame = max(i,1)
            if self.mode == 'rgb':
                imgs = load_rgb_frames(self.root, vid, start_frame, 48)
            else:
                imgs = load_flow_frames(self.root, vid, start_frame, 48)

            imgs = self.transforms(imgs)
            clips.append(video_to_tensor(imgs))
            start_gt.append(start_frame)
#         f=open("/media/zjg/workspace/action/data/start_frame/"+vid+".txt",'w')
#         for item in start_gt:
#             f.write("%s " % item)
        return clips, torch.from_numpy(label), vid

    def __len__(self):
        return len(self.data)
