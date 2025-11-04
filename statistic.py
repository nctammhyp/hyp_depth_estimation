
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ======================
# Import dataloader từ file bạn có
# ======================
from support.dataloader import nyuv2_dataloader_v2, cross_dataset, hyp_dataloader_v3, outdoor_v1, outdoor_v2
from torch.utils.data import Subset
from torch.utils.data import DataLoader

# ----------------------
# 1. Load dataset
# ----------------------
train_loader, val_loader = outdoor_v2.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/outdoor_2", batch_size=16, size=(322, 196))

from analysis_dataset.maxMin import find_extreme_depths, visualize_depth_pair

# Gọi hàm
result = find_extreme_depths(train_loader, val_loader, save_dir="dataset_infor/outdoor_2")

# In kết quả
print(result)