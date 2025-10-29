import os
import cv2
import torch
import random
import numpy as np
import gc
import json
import math
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import torch.nn.functional as F

from depth_model.fdepth_resnet_v3 import FastDepthV2
from metric_depth.util.loss import L1Loss
from support.dataloader import outdoor_v2

# ============================================================
#  🔧 Utility Functions
# ============================================================

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.cuda.empty_cache()
gc.collect()


def eval_depth(pred, target, criterion):
    eps = 1e-6
    assert pred.shape == target.shape

    pred_safe = torch.clamp(pred, min=eps)
    target_safe = torch.clamp(target, min=eps)

    thresh = torch.max(target_safe / pred_safe, pred_safe / target_safe)
    d1 = torch.sum(thresh < 1.25).float() / thresh.numel()

    diff = pred_safe - target_safe
    diff_log = torch.log(pred_safe) - torch.log(target_safe)

    abs_rel = torch.mean(torch.abs(diff) / target_safe)
    rmse = torch.sqrt(torch.mean(diff ** 2))
    mae = torch.mean(torch.abs(diff))
    silog = torch.sqrt(torch.mean(diff_log ** 2) - 0.5 * (torch.mean(diff_log) ** 2))

    return {
        'd1': d1.detach(),
        'abs_rel': abs_rel.detach(),
        'rmse': rmse.detach(),
        'mae': mae.detach(),
        'loss': torch.tensor(0.0)
    }


def adjust_learning_rate(optimizer, epoch, learning_rate=0.01):
    if epoch < 15:
        lr = learning_rate
    elif epoch < 60:
        lr = learning_rate / 2
    elif epoch < 120:
        lr = learning_rate / 4
    elif epoch < 160:
        lr = learning_rate / 8
    else:
        lr = learning_rate / 16
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def inference_sample(model, state_path, device, model_type="best"):
    if model_type == "last":
        ckpt_path = os.path.join(state_path, "last_checkpoint.pth")
    else:
        ckpt_path = os.path.join(state_path, "best_checkpoint.pth")

    if not os.path.exists(ckpt_path):
        print(f"[WARN] No checkpoint found at {ckpt_path}")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    data_test = "/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1/test"
    save_test = os.path.join(state_path, f"predict_sample_{model_type}")
    os.makedirs(save_test, exist_ok=True)

    images_root = os.path.join(data_test, "images")
    labels_root = os.path.join(data_test, "labels_npy_322_196")
    scene_list = sorted(os.listdir(images_root))

    total_images = 0
    epsilon = 1e-8

    for scene_name in scene_list:
        scene_img_dir = os.path.join(images_root, scene_name)
        scene_label_dir = os.path.join(labels_root, scene_name)
        if not os.path.isdir(scene_img_dir):
            continue
        image_paths = sorted(
            glob.glob(os.path.join(scene_img_dir, "*.png")) +
            glob.glob(os.path.join(scene_img_dir, "*.jpg"))
        )
        print(f"[INFO] Scene {scene_name}: {len(image_paths)} imgs")
        for img_path in image_paths:
            file_name = os.path.basename(img_path)
            base_name = os.path.splitext(file_name)[0]
            label_path = os.path.join(scene_label_dir, base_name + ".npy")
            if not os.path.exists(label_path):
                continue
            rgb = cv2.imread(img_path)[:, :, ::-1]
            gt_depth = np.load(label_path).astype(np.float32)
            rgb_resized = cv2.resize(rgb, (322, 196))
            gt_resized = cv2.resize(gt_depth, (322, 196))
            gt_resized = (gt_resized - gt_resized.min()) / (gt_resized.max() - gt_resized.min() + epsilon)
            rgb_tensor = torch.from_numpy(rgb_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_depth = model(rgb_tensor).cpu().squeeze(0).squeeze(0).numpy()
            pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + epsilon)
            gt_colormap = (plt.cm.plasma(gt_resized)[:, :, :3] * 255).astype(np.uint8)
            pred_colormap = (plt.cm.plasma(pred_depth)[:, :, :3] * 255).astype(np.uint8)
            concat_img = np.concatenate([rgb_resized, gt_colormap, pred_colormap], axis=1)
            save_path = os.path.join(save_test, f"{scene_name}_{file_name}")
            cv2.imwrite(save_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
            total_images += 1
    print(f"[INFO] Inference done. Total: {total_images}")


# ============================================================
#  🧠 Training One Fold
# ============================================================

def train_single_fold(device, train_loader, val_loader, fold_path, num_epochs=30, learning_rate=0.01):
    model = FastDepthV2(training=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    criterion = L1Loss()

    history = {"train_loss": [], "val_metrics": []}
    best_rmse = float("inf")

    for epoch in range(num_epochs):
        model.train()
        adjust_learning_rate(optimizer, epoch, learning_rate)
        total_loss = 0

        for imgs, depths in tqdm(train_loader, desc=f"[Fold Train] Epoch {epoch+1}/{num_epochs}"):
            imgs, depths = imgs.to(device), depths.to(device)
            optimizer.zero_grad()
            pred, disp1, disp2, disp3 = model(imgs)
            mask = (depths >= 0.0001)
            loss = sum(criterion(x, depths, mask) for x in [pred, disp1, disp2, disp3])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        model.eval()
        results = {'d1': 0, 'abs_rel': 0, 'rmse': 0, 'mae': 0, 'loss': 0}
        with torch.no_grad():
            for imgs, depths in val_loader:
                imgs, depths = imgs.to(device), depths.to(device)
                pred = model(imgs)
                mask = (depths >= 0.0001)
                r = eval_depth(pred[mask], depths[mask], criterion)
                for k in results:
                    results[k] += r[k]
        for k in results:
            results[k] = round((results[k] / len(val_loader)).item(), 4)

        print(f"[Fold Epoch {epoch+1}] TrainLoss={avg_loss:.4f} | RMSE={results['rmse']:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch}, f"{fold_path}/last_checkpoint.pth")

        if results['rmse'] < best_rmse:
            best_rmse = results['rmse']
            torch.save({"model": model.state_dict(), "epoch": epoch}, f"{fold_path}/best_checkpoint.pth")
            inference_sample(model, fold_path, device, model_type="best")

        history["train_loss"].append(avg_loss)
        history["val_metrics"].append(results)
        with open(f"{fold_path}/history.json", "w") as f:
            json.dump(history, f, indent=2)

    return history["val_metrics"][-1]


# ============================================================
#  🔁 K-Fold Cross Validation
# ============================================================

def train_kfold(device="cuda:0", n_splits=5, state_path="./checkpoints_kfold"):
    os.makedirs(state_path, exist_ok=True)

    # dataset root
    dataset_path = "/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/outdoor_2"
    full_dataset, _ = outdoor_v2.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/outdoor_2", batch_size=16, size=(322, 196))
    print(f"[INFO] Dataset path: {dataset_path}")
    print(f"[INFO] Total samples in dataset: {len(full_dataset)}")


    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n========== Fold {fold+1}/{n_splits} ==========")
        fold_path = os.path.join(state_path, f"fold_{fold+1}")
        os.makedirs(fold_path, exist_ok=True)

        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True)

        fold_metrics = train_single_fold(device, train_loader, val_loader, fold_path)
        all_metrics.append(fold_metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = round(sum(m[key] for m in all_metrics) / len(all_metrics), 4)

    with open(f"{state_path}/summary.json", "w") as f:
        json.dump(avg_metrics, f, indent=2)

    print("\n========== K-FOLD SUMMARY ==========")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")


# ============================================================
#  🚀 Entry Point
# ============================================================

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_kfold(
        device=device,
        n_splits=5,
        state_path="/home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/kfold"
    )
