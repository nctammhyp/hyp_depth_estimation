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

from model_v4 import FastDepthV2, FastDepth, weights_init
import dataloader_v5
from load_pretrained import load_pretrained_encoder, load_pretrained_fastdepth
import torch.optim as optim


import utils, loss_func
from metric_depth.util.loss import SiLogLoss, DepthLoss
from torch.optim.lr_scheduler import LambdaLR

import math
from tqdm import tqdm
import torch.nn.functional as F
import json

import glob


args = utils.parse_args()



def eval_depth(pred, target):
    valid_mask = ((target>0) + (pred>0)) > 0
    pred_output = pred[valid_mask]
    target_masked =  target[valid_mask]
    abs_diff = (pred_output - target_masked).abs() 
    RMSE = torch.sqrt(torch.mean((abs_diff).pow(2)))

    maxRatio = torch.max(pred_output / target_masked, target_masked / pred_output)
    d1 = float((maxRatio < 1.25).float().mean())

    return {
        'd1': d1,
        'rmse': RMSE.item(),
    }





def get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float = 0.5, 
    last_epoch: int = -1
):
    """
    Tạo cosine learning rate schedule với warmup.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer cần điều chỉnh LR
        num_warmup_steps (int): Số step để warmup từ 0 → lr gốc
        num_training_steps (int): Tổng số training step (epoch * steps_per_epoch)
        num_cycles (float): Số chu kỳ cosine (mặc định 0.5 = 1 lần từ lr → 0)
        last_epoch (int): Epoch cuối cùng (dùng khi resume training)

    Returns:
        torch.optim.lr_scheduler.LambdaLR
    """

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 
            0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)




def train_fn(device = "cpu", load_state = False, state_path = './'):
    # params
    num_epochs = 50

    warmup_epochs = 3
    accum_steps = 8

    train_loader, val_loader = dataloader_v5.create_data_loaders("/kaggle/input/hypdataset-v1-0/hypdataset_v1", batch_size=256)


    model = FastDepthV2().to("cuda:0")
    model.encoder = load_pretrained_encoder(model.encoder,args.weights_dir,args.backbone)
    model.decoder.apply(weights_init)

    # optim = torch.optim.SGD(model.parameters(), lr = 3e-4 ,weight_decay=1e-4)
    optim = torch.optim.AdamW(
          model.parameters(),  # lấy toàn bộ parameter của model
          lr=3e-4
      )
    
    # --- Scheduler ---
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )



    print('Model created')

    # criterion = SiLogLoss() # author's loss
    criterion = DepthLoss()

    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))
 

    best_val_loss = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if load_state:
        # checkpoint = torch.load("/kaggle/working/kgl_src/ours_checkpoints/checkpoint_best_30.pth", map_location=device)
        # # model.load_state_dict(checkpoint["model"])
        # # optim.load_state_dict(checkpoint["optim"])

        # model.load_state_dict(checkpoint)
        # model = model.to("cuda")
        checkpoint = torch.load(
            f"{state_path}/checkpoint_latest.pth", map_location=device
        )
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")


    scaler = torch.amp.GradScaler()

    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0

        optim.zero_grad()

        for i, (input, target) in tqdm(enumerate(train_loader)):
            img, depth = input.to(device, non_blocking=True), target.to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda'):
                pred = model(img)
                loss = criterion('l1', pred, depth, epoch) / accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                scheduler.step()

            total_loss += loss.item() * accum_steps  # nhân lại để logging đúng

        avg_loss = total_loss / len(train_loader)


        # ===== Validation =====
        model.eval()
        results = {'d1': 0, 'rmse': 0}
        test_loss = 0

        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            for i , (input,target) in tqdm(enumerate(val_loader)):
                img, depth = input.to(device), target.to(device)

                pred = model(img)

                test_loss += criterion('l1',pred, depth).item()

                # mask = (depth >= 0.001)
                cur_results = eval_depth(pred, depth)


                for k in results:
                    results[k] += cur_results[k]

        
        val_loss = test_loss/len(val_loader)

        # for k in results:
        #    results[k] = round(results[k] / len(val_loader), 3)
        for k in results:
            results[k] = round((results[k] / len(val_loader)), 3)



        # ===== Save Checkpoint =====
        # torch.save({
        #     "model": model.state_dict(),
        #     "optim": optim.state_dict()
        #     # "scheduler": scheduler.state_dict()
        # }, f"{state_path}/checkpoint_{epoch}.pth")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            new_ckpt = f"{state_path}/checkpoint_best_{epoch}.pth"

            # torch.save(model.state_dict(), new_ckpt)
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optim": optim.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"{new_ckpt}",
            )
            print(f"----- 🔥 Saved new best checkpoint at epoch {epoch}")

            # 2. Xóa tất cả checkpoint cũ (trừ file vừa lưu)
            for ckpt in glob.glob(f"{state_path}/checkpoint_best_*.pth"):
                if ckpt != new_ckpt:
                    os.remove(ckpt)

        # Cập nhật history
        history["train_loss"].append(avg_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append(results)

        # Lưu log JSON
        with open(f"{state_path}/history.json", "w") as f:
            json.dump(history, f, indent=2)


        print(f"epoch_{epoch}, train_loss={avg_loss:.5f}, val loss: {val_loss:.5f}, val_metrics={results}")

        # ==== Vẽ biểu đồ ====
        # epochs = range(1, num_epochs+1)
        epochs = range(1, len(history["train_loss"])+1)

        plt.figure(figsize=(8,5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{state_path}/train_loss_curve.png")
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{state_path}/val_loss_curve.png")
        plt.close()



if __name__ == "__main__":
    train_fn(device='cuda:0', load_state=False, state_path="/kaggle/working/ours_checkpoints")