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
from depth_model.fdepth_resnet import FastDepthV2

import dataloader_v6
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

import time


# args = utils.parse_args()



def eval_depth(pred, target):
    eps = 1e-6  # tránh chia 0, log 0
    assert pred.shape == target.shape

    pred_safe = torch.clamp(pred, min=eps)
    target_safe = torch.clamp(target, min=eps)

    thresh = torch.max(target_safe / pred_safe, pred_safe / target_safe)
    d1 = torch.sum(thresh < 1.25).float() / len(thresh)

    # thresh = torch.max(target_safe / pred_safe, pred_safe / target_safe)
    # d1 = (thresh < 1.25).float().mean()

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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)



def inference_sample(model, state_path, device, model_type="last"):
    """
    Load model checkpoint and run inference on test dataset.

    Args:
        model: PyTorch model
        state_path: Path chứa checkpoint
        device: torch.device
        model_type: "last" (checkpoint mới nhất) hoặc "best" (checkpoint tốt nhất)
    """
    # =========================
    # 1. Load checkpoint
    # =========================
    if model_type == "last":
        # Load checkpoint mới nhất
        ckpts = glob.glob(os.path.join(state_path, "last_checkpoint_*.pth"))
        if len(ckpts) == 0:
            raise FileNotFoundError("No last_checkpoint_*.pth found in the state_path!")

        ckpts.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
        latest_ckpt = ckpts[-1]

        checkpoint = torch.load(latest_ckpt, map_location=device)
        print(f"[INFO] Loaded latest checkpoint: {latest_ckpt}")

    elif model_type == "best":
        # Load best checkpoint
        ckpts = glob.glob(os.path.join(state_path, "checkpoint_best_*.pth"))
        if len(ckpts) == 0:
            raise FileNotFoundError("No checkpoint_best_*.pth found in the state_path!")

        ckpts.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[-1]))
        best_ckpt = ckpts[-1]

        checkpoint = torch.load(best_ckpt, map_location=device)
        print(f"[INFO] Loaded best checkpoint: {best_ckpt}")

    else:
        raise ValueError("model_type must be either 'last' or 'best'")

    # Load weights vào model
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()

    # =========================
    # 2. Paths setup
    # =========================
    data_test = "/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1/test"
    
    save_test = "predict_sample"
    os.makedirs(save_test, exist_ok=True)

    images_root = os.path.join(data_test, "images")
    labels_root = os.path.join(data_test, "labels_npy")  # <-- Load từ thư mục chứa file .npy

    # Lấy danh sách tất cả scene
    scene_list = sorted(os.listdir(images_root))

    epsilon = 1e-8
    total_images = 0

    # =========================
    # 3. Inference từng scene
    # =========================
    for scene_name in scene_list:
        scene_img_dir = os.path.join(images_root, scene_name)
        scene_label_dir = os.path.join(labels_root, scene_name)

        if not os.path.isdir(scene_img_dir):
            continue

        # Lấy danh sách file ảnh trong scene
        image_paths = sorted(glob.glob(os.path.join(scene_img_dir, "*.png")))
        print(f"[INFO] Scene {scene_name}: Found {len(image_paths)} images.")

        for img_path in image_paths:
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]

            # Đường dẫn file .npy tương ứng
            label_path = os.path.join(scene_label_dir, base_name + ".npy")

            if not os.path.exists(label_path):
                print(f"Warning: No GT .npy found for {file_name} in scene {scene_name}, skipping...")
                continue

            # ----- Read RGB -----
            rgb = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB

            # ----- Load Depth từ file .npy -----
            gt_depth = np.load(label_path).astype(np.float32)

            # Resize về input size của model (160x128)
            rgb_resized = cv2.resize(rgb, (160, 128))
            gt_resized = cv2.resize(gt_depth, (160, 128))

            # Normalize GT depth để visualize
            gt_resized = (gt_resized - gt_resized.min()) / (gt_resized.max() - gt_resized.min() + epsilon)

            # ----- Chuẩn bị tensor -----
            rgb_tensor = torch.from_numpy(rgb_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)

            # ----- Model inference -----
            with torch.no_grad():
                pred_depth = model(rgb_tensor).cpu().squeeze(0).squeeze(0).numpy()

            # Normalize predicted depth để hiển thị
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + epsilon)

            # =========================
            # 4. Convert to color maps
            # =========================
            gt_colormap = (plt.cm.plasma(gt_resized)[:, :, :3] * 255).astype(np.uint8)
            pred_colormap = (plt.cm.plasma(pred_depth)[:, :, :3] * 255).astype(np.uint8)

            # Resize RGB gốc về cùng kích thước
            rgb_show = cv2.resize(rgb, (160, 128))

            # =========================
            # 5. Horizontal concat
            # =========================
            concat_img = np.concatenate([rgb_show, gt_colormap, pred_colormap], axis=1)

            # =========================
            # 6. Save result
            # =========================
            save_name = f"{scene_name}_{file_name}"  # thêm tiền tố scene
            save_path = os.path.join(save_test, f"{model_type}_{save_name}")
            cv2.imwrite(save_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))

            total_images += 1

    print(f"[INFO] Inference completed. Total processed images: {total_images}")





# def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
#     return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=2.0, last_epoch=-1):
    """
    Scheduler LR: Warmup + Cosine Annealing with Restarts.
    Args:
        optimizer: Optimizer đang dùng (AdamW, SGD,...)
        num_warmup_steps: Số step warmup (LR tăng từ 0 -> max LR)
        num_training_steps: Tổng số step trong toàn bộ training
        num_cycles: Số lần restart (>=0.5). Mặc định 2.0 => 2 lần restart
        last_epoch: Dùng khi resume training
    """
    def lr_lambda(current_step):
        # Phase 1: Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Phase 2: Cosine Annealing with Restart
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def train_fn(device = "cpu", load_state = False, state_path = './'):
    # params
    num_epochs = 500
    warmup_epochs = 8
    num_cycles = 2


    model = FastDepthV2().to(device)

    optim = torch.optim.Adam(
          model.parameters(),  # lấy toàn bộ parameter của model
          lr=3e-4,
          weight_decay=0.01
      )

    print('Model created')

    criterion = SiLogLoss() # author's loss
    # criterion = DepthLoss()
    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))

    train_loader, val_loader = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=512, size=(160, 128))

    # total_steps = num_epochs * len(train_loader)
    # warmup_steps = warmup_epochs * len(train_loader)
    # scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps, num_cycles)


    print(f"size of train loader: {len(train_loader)}; val loader: {len(val_loader)}")
 

    best_val_absrel = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if load_state:
        checkpoint = torch.load("/home/gremsy_guest/hyp_workspa", map_location=device)
        # model.load_state_dict(checkpoint["model"])
        # optim.load_state_dict(checkpoint["optim"])

        model.load_state_dict(checkpoint)
        model = model.to(device)


    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0

        for i , (input,target) in enumerate(tqdm(train_loader, total=len(train_loader))):
            img, depth = input.to(device), target.to(device)

            optim.zero_grad()
            pred = model(img)

            # loss = criterion('l1',pred,depth,epoch)

            mask = (depth > 1e-3)

            # print("pred shape:", pred.shape)
            # print("target shape:", target.shape)
            # print("valid_mask shape:", mask.shape)
            loss = criterion(pred, depth, mask)

            loss.backward()
            optim.step()
            # scheduler.step()


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


                mask = (depth >= 0.001)

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

        # ===== Save Checkpoint =====
        torch.save({
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            # "scheduler": scheduler.state_dict()
        }, f"{state_path}/last_checkpoint_{epoch}.pth")

        if results['abs_rel'] < best_val_absrel:
            best_val_absrel = results['abs_rel']
            new_ckpt = f"{state_path}/checkpoint_best_{epoch}.pth"

            # 1. Lưu checkpoint mới
            # torch.save(model.state_dict(), new_ckpt)
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                # "scheduler": scheduler.state_dict()
            }, new_ckpt)

            # 2. Xóa tất cả best checkpoint cũ (trừ file vừa lưu)
            for ckpt in glob.glob(f"{state_path}/checkpoint_best_*.pth"):
                print(ckpt)
                print(new_ckpt)
                print("--------------")
                if ckpt != new_ckpt:
                    os.remove(ckpt)


            # inference cho best checkpoint
            inference_sample(model, state_path, device, model_type="best")

        inference_sample(model, state_path, device, model_type="last")


        # Cập nhật history
        history["train_loss"].append(avg_loss)
        # history["val_loss"].append(val_loss)
        history["val_metrics"].append(results)

        # Lưu log JSON
        with open(f"{state_path}/history.json", "w") as f:
            json.dump(history, f, indent=2)


        print(f"epoch_{epoch}, train_loss={avg_loss:.5f}, val_metrics={results}")

        # ==== Vẽ biểu đồ ====
        # epochs = range(1, num_epochs+1)
        epochs = range(1, len(history["train_loss"]) + 1)
        loss_val = [m["silog"] for m in history["val_metrics"]]  # lấy metric silog từ val_metrics

        plt.figure(figsize=(8, 5))

        # Train loss
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker='o')

        # Validation loss
        plt.plot(epochs, loss_val, label="Val Loss", marker='s')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)

        # Lưu biểu đồ
        plt.savefig(f"{state_path}/train_val_loss_curve.png", dpi=150)
        plt.close()

        absrel = [m["abs_rel"] for m in history["val_metrics"]]
        plt.figure(figsize=(8,5))
        plt.plot(epochs, absrel, label="AbsRel (val)")
        plt.xlabel("Epoch")
        plt.ylabel("AbsRel")
        plt.legend()
        plt.savefig(f"{state_path}/val_absrel_curve.png")
        plt.close()



if __name__ == "__main__":
    train_fn(device='cuda:0', load_state=False, state_path="/home/gremsy_guest/hyp_workspace/depth_v2/ours_checkpoints/9")