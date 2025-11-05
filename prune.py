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
import copy


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # thêm dòng này nữa!
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

# model = FastDepthV2()

# checkpoint = torch.load("/home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/43/best_checkpoint.pth", map_location=device)
# model.load_state_dict(checkpoint["model"])
# # optim.load_state_dict(checkpoint["optim"])

# # model.load_state_dict(checkpoint)
# model = model.to(device)

# example_input = torch.rand(1, 3, 196, 322).to(device)
# macs, parameters = tp.utils.count_ops_and_params(model, example_input)

# print(f"[full model] \t\tMACs: {macs/1e9:.2f} G, \tParameters: {parameters/1e6:.2f} M")

# # clone full model before pruning
# pruned_model = copy.deepcopy(model)
# pruned_model = pruned_model.to(device)

# # set which layers to skip pruning. important to keep final classifier layer
# ignored_layers = []
# # for m in pruned_model.modules():
# #     if isinstance(m, torch.nn.Linear) and m.out_features == 10:
# #         ignored_layers.append(m)

# # iterative pruning
# iterative_steps = 20
# pruner = tp.pruner.MagnitudePruner(
#     model = pruned_model,
#     example_inputs = example_input,
#     importance = tp.importance.MagnitudeImportance(p=2),
#     pruning_ratio = 1,
#     iterative_steps = iterative_steps,
#     ignored_layers = ignored_layers,
#     round_to = 2,
# )

class Timer:
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.starter = torch.cuda.Event(enable_timing=True)
            self.ender = torch.cuda.Event(enable_timing=True)

    def start(self):
        if self.use_cuda:
            self.starter.record()
        else:
            self.start_time = time.time()

    def stop(self):
        if self.use_cuda:
            self.ender.record()
            torch.cuda.synchronize()
            return self.starter.elapsed_time(self.ender)  # ms
        else:
            return (time.time() - self.start_time) * 1000  # ms

def estimate_latency(model, example_inputs, repetitions=50):
    timer = Timer()
    timings = np.zeros((repetitions, 1))

    # warm-up
    for _ in range(5):
        _ = model(example_inputs)

    with torch.no_grad():
        for rep in range(repetitions):
            timer.start()
            _ = model(example_inputs)
            elapsed = timer.stop()
            timings[rep] = elapsed

    return np.mean(timings), np.std(timings)


def eval_depth(pred, target):
    eps = 1e-6  # tránh chia 0, log 0
    assert pred.shape == target.shape

    pred_safe = torch.clamp(pred, min=eps)
    target_safe = torch.clamp(target, min=eps)

    thresh = torch.max(target_safe / pred_safe, pred_safe / target_safe)
    # d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d1 = torch.sum(thresh < 1.25).float() / thresh.numel()

    diff = pred_safe - target_safe

    # print("NaN in pred:", torch.isnan(pred).any().item())
    # print("NaN in target:", torch.isnan(target).any().item())

    diff_log = torch.log(pred_safe) - torch.log(target_safe)
    # print(f"errrrrrrrrrrrrrrrrr: {torch.log(pred_safe)}")

    abs_rel = torch.mean(torch.abs(diff) / target_safe)
    rmse = torch.sqrt(torch.mean(diff ** 2))
    mae = torch.mean(torch.abs(diff))

    silog = torch.sqrt(
        torch.mean(diff_log ** 2) - 0.5 * (torch.mean(diff_log) ** 2)
    )

    return {
        'd1': d1.detach(),
        'abs_rel': abs_rel.detach(),
        'rmse': rmse.detach(),
        'mae': mae.detach(),
        'silog': silog.detach()
    }


def train_fn(model, device = "cpu", load_state = False, state_path = './'):
    # params
    num_epochs = 5
    learning_rate=0.01


    # optim = torch.optim.AdamW(
    #       model.parameters(),  # lấy toàn bộ parameter của model
    #       lr=3e-4,
    #       weight_decay=0.01
    #   )
    
    optim = torch.optim.SGD(model.parameters(), lr = learning_rate ,weight_decay=1e-4, momentum=0.9)


    print('Model created')

    # criterion = SiLogLoss() # author's loss
    # criterion = DepthLoss()
    criterion = L1Loss()

    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))

    # train_loader, val_loader = dataloader_v6.create_data_loaders("/content/hypdataset_v2_subtest", batch_size=3, size=(160, 128))

    train_loader, val_loader = None, None

    use_cross_dataset = True
    if use_cross_dataset:
        # train_loader_v1, val_loader_v1 = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=16, size=(322, 196))
        # train_loader_v2 = hyp_dataloader_v3.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v3", batch_size=16, size=(322, 196))
        # train_loader_v3, val_loader_v3 = nyuv2_dataloader_v2.create_data_loaders()
        # train_loader_v4 = cross_dataset.create_train_loader(batch_size=8, size=(322, 196))

        # train_loader_v5 = outdoor_v1.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_outdoor_v1", batch_size=16, size=(322, 196))
        train_loader_v6, val_loader_v6 = outdoor_v2.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/outdoor_2", batch_size=16, size=(322, 196))


        # print(f"train_loader_v1: {len(train_loader_v1.dataset)} samples ({len(train_loader_v1)} batches)")
        # print(f"val_loader_v1:   {len(val_loader_v1.dataset)} samples ({len(val_loader_v1)} batches)")
        # print(f"train_loader_v2: {len(train_loader_v2.dataset)} samples ({len(train_loader_v2)} batches)")
        # print(f"train_loader_v3: {len(train_loader_v3.dataset)} samples ({len(train_loader_v3)} batches)")
        # print(f"val_loader_v3:   {len(val_loader_v3.dataset)} samples ({len(val_loader_v3)} batches)")
        # print(f"train_loader_v4: {len(train_loader_v4.dataset)} samples ({len(train_loader_v4)} batches)")
        # print(f"outdoor v1: {len(train_loader_v5.dataset)} samples ({len(train_loader_v5)} batches)")
        print(f"train_loader_v6: {len(train_loader_v6.dataset)} samples ({len(train_loader_v6)} batches)")



        # Gom tất cả dataset lại (kể cả val)
        datasets = [
            # train_loader_v1.dataset,
            # train_loader_v2.dataset,
            # train_loader_v3.dataset,
            # val_loader_v3.dataset,   # thêm val_loader_v3
            # train_loader_v4.dataset,
            # train_loader_v5.dataset
            train_loader_v6.dataset
        ]

        # Gộp chúng lại
        combined_train_dataset = ConcatDataset(datasets)

        # Tạo DataLoader chung
        combined_train_loader = DataLoader(
            combined_train_dataset,
            batch_size=8,
            shuffle=True,       # quan trọng để trộn toàn bộ data
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        train_loader, val_loader = combined_train_loader, val_loader_v6
        # train_loader, val_loader = combined_train_loader, val_loader_v3
        # train_loader, val_loader = combined_train_loader, train_loader_v5

    else:
        # train_loader, val_loader = nyuv2_dataloader_v2.create_data_loaders()
        train_loader, val_loader = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=16, size=(160, 128))



    print(f"size of train loader: {len(train_loader)}; val loader: {len(val_loader)}")
 


    best_val = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}


    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0

        for i , (input,target) in enumerate(tqdm(train_loader, total=len(train_loader))):
            img, depth = input.to(device), target.to(device)

            optim.zero_grad()
            pred = model(img)

            # loss = criterion('l1',pred,depth,epoch)

            # mask = (depth > 1e-3) & (depth <= 1) & torch.isfinite(depth)
            mask = (depth > 5) & (depth < 1000)


            # print("pred shape:", pred.shape)
            # print("target shape:", target.shape)
            # print("valid_mask shape:", mask.shape)
            loss = criterion(pred, depth, mask)

            loss.backward()
            optim.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        # results = {'d1': 0, 'rmse': 0}
        results = {'d1': 0, 'abs_rel': 0, 'rmse': 0, 'mae': 0, 'silog': 0}
        # test_loss = 0

        with torch.no_grad():
            for i , (input,target) in tqdm(enumerate(val_loader)):
                img, depth = input.to(device), target.to(device)

                pred = model(img)

                # test_loss += criterion('l1',pred, depth).item()
                # pred = pred.squeeze(1).squeeze(0)

                # mask = (depth >= 0.001)
                # cur_results = eval_depth(pred, depth)

                # print(depth)


                # mask = (depth <= 1) & (depth >= 0.001)
                mask = (depth > 5) & (depth < 1000)


                # print(mask)

                # valid_pixels = mask.sum().item()
                # print(f"mask: {valid_pixels}")

                # print("pred shape:", pred.shape)
                # print("target shape:", target.shape)
                # print("valid_mask shape:", mask.shape)
                cur_results = eval_depth(pred[mask], depth[mask])


                for k in results:
                    results[k] += cur_results[k]


        # val_loss = test_loss/len(val_loader)

        # for k in results:
        #    results[k] = round(results[k] / len(val_loader), 3)
        for k in results:
            results[k] = round((results[k] / len(val_loader)).item(), 3)

        print(f"epoch_{epoch}, val_metrics={results}")        

        # ===== Save Checkpoint =====
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict()
            # "scheduler": scheduler.state_dict()
        }, f"{state_path}/prune_last_checkpoint_{epoch}.pth")


def prune():
    device = "cuda:0"
    model = FastDepthV2().to(device)
    checkpoint = torch.load("/home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/44/best_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model"])

    example_input = torch.rand(1, 3, 196, 322).to(device)

    macs, parameters = tp.utils.count_ops_and_params(model, example_input)

    print(f"[full model] \t\tMACs: {macs/1e9:.2f} G, \tParameters: {parameters/1e6:.2f} M")

    # clone full model before pruning
    pruned_model = copy.deepcopy(model)
    pruned_model = pruned_model.to(device)

    # set which layers to skip pruning. important to keep final classifier layer
    ignored_layers = []
    for m in pruned_model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == 10:
            ignored_layers.append(m)

    # iterative pruning
    iterative_steps = 10
    pruner = tp.pruner.MagnitudePruner(
        model = pruned_model,
        example_inputs = example_input,
        importance = tp.importance.MagnitudeImportance(p=2),
        pruning_ratio = 1,
        iterative_steps = iterative_steps,
        ignored_layers = ignored_layers,
        round_to = 8,
    )

    for iter in range(iterative_steps):
        # prune
        pruner.step()
        # fine-tune pruned model
        train_fn(pruned_model, device = "cuda:0", load_state = False, state_path = 'ours_checkpoints')
        # count MACs and parameters
        macs, parameters = tp.utils.count_ops_and_params(pruned_model, example_input)
        latency_mu, latency_std = estimate_latency(pruned_model, example_input)
        current_pruning_ratio = 1 / iterative_steps * (iter + 1)
        print(f"[pruned model] \tPruning ratio: {current_pruning_ratio:.2f}, \tMACs: {macs/1e9:.2f} G, \tParameters: {parameters/1e6:.2f} M, \tLatency: {latency_mu:.2f} ± {latency_std:.2f} ms")


        torch.save(pruned_model, f"ours_checkpoints/46/pruned_model_full_{iter}.pth")
        print(f"Saved full pruned model to ours_checkpoints/46/pruned_model_full_{iter}.pth")


if __name__ == "__main__":
    prune()