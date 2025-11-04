"""
python trainv6.py --backbone mobilenetv2 --weights_dir Weights
loss: Scale and Shift Invariant Loss
model: root
"""

import os
import cv2
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import albumentations as A
import matplotlib.pyplot as plt

# import model for traning
# from model_v4 import FastDepthV2, FastDepth, weights_init
from depth_model.fdepth_resnet_v2 import FastDepthV2
# from depth_model.fdepth_resnet_v3 import FastDepthV2

# from depth_model.depth_mobile import FastDepthV2, weights_init

import dataloader_v6
from load_pretrained import load_pretrained_encoder, load_pretrained_fastdepth
import torch.optim as optim


import utils, loss_func
from metric_depth.util.loss import SiLogLoss, DepthLoss, RelativeL1Loss, L1Loss
from metric_depth.util import loss as loss_fn
from metric_depth.util import loss_v3


from torch.optim.lr_scheduler import LambdaLR

import math
from tqdm import tqdm
import torch.nn.functional as F
import json

import glob

import time

from support.dataloader import nyuv2_dataloader_v2, cross_dataset, hyp_dataloader_v3, outdoor_v1, outdoor_v2
from torch.utils.data import ConcatDataset, DataLoader


import torch
import gc

import autoclip
from autoclip.torch import QuantileClip

import random, torch, numpy as np
import torch_pruning as tp


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # thêm dòng này nữa!
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

model = FastDepthV2()

checkpoint = torch.load("/home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/43/best_checkpoint.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
# optim.load_state_dict(checkpoint["optim"])

# model.load_state_dict(checkpoint)
model = model.to(device)

example_input = torch.rand(1, 3, 128, 160).to(device)
macs, parameters = tp.utils.count_ops_and_params(model, example_input)

print(f"[full model] \t\tMACs: {macs/1e9:.2f} G, \tParameters: {parameters/1e6:.2f} M")