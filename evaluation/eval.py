import os
import numpy as np
import torch
import torch.nn.functional as F
from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import cPickle as pickle
import math
import sys
import cv2

sys.path.insert(0,'./ref_utils/')
sys.path.insert(0,'./Model/')

from Model import MultiscaleWarpingNet8


torch.cuda.set_device(0)
net = MultiscaleWarpingNet8()

net.load_state_dict(torch.load("pretrained_models/dual_camera/CP250000.pth"))
net.eval()
net.cuda()

H,W = (640,1024)

input_img1_SR = np.zeros((1,3,H,W),dtype=np.float32)
input_img2_HR = np.zeros((1,3,H,W),dtype=np.float32)

lr_img = np.array(cv2.imread("evaluation/test_imgs/lr.png"),dtype=np.float32).transpose(2,0,1) / 255.0

ref_img = np.array(cv2.imread("evaluation/test_imgs/ref.png"),dtype=np.float32).transpose(2,0,1) / 255.0

input_img1_SR[0] = lr_img[::-1,:,:]
input_img2_HR[0] = ref_img[::-1,:,:]

buff = {}
buff["input_img1_SR"] = input_img1_SR
buff['input_img2_HR'] = input_img2_HR
buff['input_img1_LR'] = input_img1_SR

with torch.no_grad():

    coarse_img,fine_img = net(buff)

    sr_img = np.array(np.clip(fine_img.cpu().numpy()[0],0.0,1.0) * 255.0, dtype=np.uint8).transpose(1,2,0)
    cv2.imwrite("evaluation/test_imgs/sr.png",sr_img[:,:,::-1])
