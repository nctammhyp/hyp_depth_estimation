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

model = FastDepthV2()


checkpoint = torch.load(
    r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\checkpoints_depth\best_checkpoint.pth",
    map_location='cpu',
    weights_only=False  # <-- allow loading the full pickled file
)

model.load_state_dict(checkpoint["model"])
# optim.load_state_dict(checkpoint["optim"])

# model.load_state_dict(checkpoint)
model = model.to('cpu')

print(model)

# path = r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\checkpoints_depth\best_checkpoint.pth"

# with open(path, "rb") as f:
#     print(f.read(20))
