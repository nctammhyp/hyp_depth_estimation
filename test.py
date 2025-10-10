import os
import torch 

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
from depth_model.depth_resnet_v2_distill import FastDepthV2


from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# model_weights_path =  '/kaggle/working/depth_anything_v2_vitl.pth'
model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
model_encoder = 'vitl'

model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
model.load_state_dict(torch.load(r'depth_anything_v2\checkpoint\depth_anything_v2_vitl.pth', map_location='cpu'))

model.to('cpu')
model.eval()
dummy_input = torch.randn(1, 3, 224, 224)
output, fea_distill = model(dummy_input)

print("Output shape:", output.shape)  # [1, 1000]
print("fea_distill shape:", fea_distill.shape)  # [1, 1000]
# print(f"diff: ", (output-fea_distill.squeeze(0)).sum())


print("---------------------------   student        ----------------------------------------")
# Tạo model custom từ scratch
model = FastDepthV2()
print("Pretrained ImageNet weights đã được load thành công!")

output, student_feature_map = model(dummy_input)
print("Output shape:", output.shape)  # [1, 1000]
print(f"student_feature_map: {student_feature_map.shape}")
