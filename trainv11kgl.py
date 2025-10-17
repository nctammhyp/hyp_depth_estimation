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
# from depth_model.fdepth_resnet_v2 import FastDepthV2
from depth_model.depth_mobile import FastDepthV2, weights_init

import dataloader_v6
from load_pretrained import load_pretrained_encoder, load_pretrained_fastdepth
import torch.optim as optim


import utils, loss_func
from metric_depth.util.loss import SiLogLoss, DepthLoss, RelativeL1Loss, L1Loss, CustomLoss
from torch.optim.lr_scheduler import LambdaLR

import math
from tqdm import tqdm
import torch.nn.functional as F
import json

import glob

import time

from support.dataloader import nyuv2_dataloader_v2, cross_dataset, hyp_dataloader_v3
from torch.utils.data import ConcatDataset, DataLoader


import torch
import gc

# 1. Xóa cache
torch.cuda.empty_cache()

# 2. Xóa các object không dùng nữa
gc.collect()

# args = utils.parse_args()
# ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# # ===============================
# # 1️⃣ CUDA / cuDNN / TensorCore
# # ===============================
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"          # đảm bảo mapping GPU ổn định
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"               # async kernel launch
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"      # tối ưu workspace kernel
# os.environ["CUBLAS_FORCE_TF32_TENSOR_OP_MATH"] = "1"  # bật TF32 TensorCore
# os.environ["NVIDIA_TF32_OVERRIDE"] = "1"              # ép TF32 khi FP32 compute

# os.environ["CUDNN_BENCHMARK"] = "1"                   # chọn kernel nhanh nhất
# os.environ["CUDNN_DETERMINISTIC"] = "0"               # cho phép non-deterministic kernel

# # ===============================
# # 2️⃣ PyTorch memory / DSA
# # ===============================
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.8"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"                # bật dynamic shape allocation (PyTorch 2.x)

# # ===============================
# # 3️⃣ CPU / Thread / I/O
# # ===============================
# os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())    # max CPU cores
# os.environ["UV_THREADPOOL_SIZE"] = "64"               # tăng threadpool cho async I/O

# # ===============================
# # 4️⃣ Torch runtime flags
# # ===============================
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = False
# torch.set_float32_matmul_precision("high")           # bật TF32 trên Ada GPUs



def eval_depth(pred, target, criterion):
    eps = 1e-6  # tránh chia 0, log 0
    assert pred.shape == target.shape

    pred_safe = torch.clamp(pred, min=eps)
    target_safe = torch.clamp(target, min=eps)

    thresh = torch.max(target_safe / pred_safe, pred_safe / target_safe)
    # d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d1 = torch.sum(thresh < 1.25).float() / thresh.numel()


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

    mask = (target >= 1e-3)
    loss = criterion(pred, target, mask)

    return {
        'd1': d1.detach(),
        'abs_rel': abs_rel.detach(),
        'rmse': rmse.detach(),
        'mae': mae.detach(),
        'loss': loss.detach()
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

    if model_type == "last":
        ckpt_path = os.path.join(state_path, "last_checkpoint.pth")
    elif model_type == "best":
        ckpt_path = os.path.join(state_path, "best_checkpoint.pth")
    else:
        raise ValueError("model_type must be either 'last' or 'best'")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint file not found: {ckpt_path}")
    else:
        
        print(f"*****      infer: {ckpt_path}    ******")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        model.to(device)

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


# def adjust_learning_rate(optimizer, epoch, learning_rate=0.005):
#     if epoch < 60:
#         lr = learning_rate
#     elif epoch < 120:
#         lr = learning_rate / 2   # 0.0025
#     elif epoch < 160:
#         lr = learning_rate / 4   # 0.00125
#     else:
#         lr = learning_rate / 8   # 0.000625
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, learning_rate=0.01):
    if epoch < 30:
        lr = learning_rate
    elif epoch < 60:
        lr = learning_rate / 2
    elif epoch < 120:
        lr = learning_rate / 4   # 0.0025
    elif epoch < 160:
        lr = learning_rate / 8   # 0.00125
    else:
        lr = learning_rate / 16   # 0.000625
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fn(device = "cuda:0", load_state = False, state_path = './'):
    # params
    num_epochs = 5000
    warmup_epochs = 8
    num_cycles = 2
    max_depth = 600
    learning_rate=0.01



    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))



    model = FastDepthV2()

    model.encoder = load_pretrained_encoder(model.encoder,'Weights','mobilenetv2')
    model.decoder.apply(weights_init)
    
    
    model.to(device)

    # optim = torch.optim.Adam(
    #       model.parameters(),  # lấy toàn bộ parameter của model
    #       lr=0.01,
    #       weight_decay=0.01
    #   )

    # optim = torch.optim.SGD(model.parameters(), lr = 0.01 ,weight_decay=1e-4)
    # optim = torch.optim.Adam(model.parameters(), lr = 0.01 ,weight_decay=1e-4)

    # optim = torch.optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

    optim = torch.optim.ASGD(
        model.parameters(),
        lr=learning_rate,            # max LR ban đầu
        lambd=0.0001,
        alpha=0.75,
        t0=1e6,
        weight_decay=0
    )


    # optim = torch.optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    

    
    # backbone_params = model.encoder.parameters()
    # decoder_params = model.decoder.parameters()

    # optim = torch.optim.Adam([
    #     {"params": backbone_params, "lr": 0.01},  # backbone LR nhỏ
    #     {"params": decoder_params, "lr": 0.01}    # decoder LR lớn
    # ], weight_decay=1e-5)


    print('Model created')

    # criterion = SiLogLoss() # author's loss
    criterion = CustomLoss()
    # criterion = SiLogL1Loss()
    # criterion = DepthLoss()
    # criterion = RelativeL1Loss()
    # criterion = L1Loss()
    # scheduler = transformers.get_cosine_schedule_with_warmup(optim, len(train_dataloader)*warmup_epochs, num_epochs*scheduler_rate*len(train_dataloader))

    # train_loader, val_loader = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=512, size=(160, 128))
    # train_loader, val_loader = nyuv2_dataloader_v2.create_data_loaders()
    train_loader, val_loader = None, None

    use_cross_dataset = False
    if use_cross_dataset:
        train_loader_v1, val_loader_v1 = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=16, size=(160, 128))
        train_loader_v2 = hyp_dataloader_v3.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v3", batch_size=16, size=(160, 128))
        train_loader_v3, val_loader_v3 = nyuv2_dataloader_v2.create_data_loaders()
        train_loader_v4 = cross_dataset.create_train_loader(batch_size=16, size=(160, 128))

        print(f"train_loader_v1: {len(train_loader_v1.dataset)} samples ({len(train_loader_v1)} batches)")
        print(f"val_loader_v1:   {len(val_loader_v1.dataset)} samples ({len(val_loader_v1)} batches)")
        print(f"train_loader_v2: {len(train_loader_v2.dataset)} samples ({len(train_loader_v2)} batches)")
        print(f"train_loader_v3: {len(train_loader_v3.dataset)} samples ({len(train_loader_v3)} batches)")
        print(f"val_loader_v3:   {len(val_loader_v3.dataset)} samples ({len(val_loader_v3)} batches)")
        print(f"train_loader_v4: {len(train_loader_v4.dataset)} samples ({len(train_loader_v4)} batches)")


        # Gom tất cả dataset lại (kể cả val)
        datasets = [
            train_loader_v1.dataset,
            train_loader_v2.dataset,
            train_loader_v3.dataset,
            val_loader_v3.dataset,   # thêm val_loader_v3
            train_loader_v4.dataset
        ]

        # Gộp chúng lại
        combined_train_dataset = ConcatDataset(datasets)

        # Tạo DataLoader chung
        combined_train_loader = DataLoader(
            combined_train_dataset,
            batch_size=16,
            shuffle=True,       # quan trọng để trộn toàn bộ data
            num_workers=8,
            pin_memory=True,
            drop_last=True
        )

        train_loader, val_loader = combined_train_loader, val_loader_v1
    else:
        train_loader, val_loader = nyuv2_dataloader_v2.create_data_loaders()
        # train_loader, val_loader = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=16, size=(160, 128))



    print(f"size of train loader: {len(train_loader)}; val loader: {len(val_loader)}")
 
    # best val monitor: loss silog
    best_val = 7092
    # best_loss = 1e9
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    if load_state:
        print("----------   load checkpoint -------------")
        print("checkpoint: /home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/18/last_checkpoint.pth")
        checkpoint = torch.load("/home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/18/last_checkpoint.pth", map_location=device)
        model.load_state_dict(checkpoint["model"])
        # optim.load_state_dict(checkpoint["optim"])

        # model.load_state_dict(checkpoint)
        model = model.to(device)


    # model = torch.compile(model)  

    

    # Chọn device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # for name, param in model.named_parameters():
    #     print(name, param.device)

    # print("------------------------------------------------------------------")


    for epoch in range(0, num_epochs):
        model.train()
        total_loss = 0
        adjust_learning_rate(optim, epoch, learning_rate)

        for i , (input,target) in enumerate(tqdm(train_loader, total=len(train_loader))):
            img, depth = input.to(device), target.to(device)

            optim.zero_grad()
            pred = model(img)


            mask = (depth > 1e-3)
            # mask = (depth > 1e-3) & torch.isfinite(depth)

            # mask = (depth > 1e-3) & (depth <= max_depth)

            # print("pred shape:", pred.shape)
            # print("target shape:", target.shape)
            # print("valid_mask shape:", mask.shape)

            loss = criterion(pred,depth,mask)

            # loss = criterion(pred, depth, mask)

            loss.backward()
            optim.step()
            # scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== Validation =====
        model.eval()
        # results = {'d1': 0, 'rmse': 0}
        results = {'d1': 0, 'abs_rel': 0, 'rmse': 0, 'mae': 0, 'loss': 0}
        # test_loss = 0

        with torch.no_grad():
            for i , (input,target) in tqdm(enumerate(val_loader)):
                img, depth = input.to(device), target.to(device)

                pred = model(img)

                # test_loss += criterion('l1',pred, depth).item()
                # pred = pred.squeeze(1).squeeze(0)

                mask = (depth >= 0.001)
                # cur_results = eval_depth(pred, depth)

                # print(depth)


                # mask = (depth > 1e-3) & (depth <= max_depth)
                # mask = (depth > 1e-3) & torch.isfinite(depth)


                # print(mask)

                # valid_pixels = mask.sum().item()
                # print(f"mask: {valid_pixels}")

                # print("pred shape:", pred.shape)
                # print("target shape:", target.shape)
                # print("valid_mask shape:", mask.shape)
                cur_results = eval_depth(pred[mask], depth[mask], criterion)


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
            "epoch": epoch

            # "scheduler": scheduler.state_dict()
        }, f"{state_path}/last_checkpoint.pth")

        # if results['abs_rel'] < best_val_absrel:
        if results['loss'] < best_val:

            best_val = results['loss']
            new_ckpt = f"{state_path}/best_checkpoint.pth"

            # 1. Lưu checkpoint mới
            # torch.save(model.state_dict(), new_ckpt)
            torch.save({
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "epoch": epoch
                # "scheduler": scheduler.state_dict()
            }, new_ckpt)


        #     # inference cho best checkpoint
        #     inference_sample(model, state_path, device, model_type="best")

        # inference_sample(model, state_path, device, model_type="last")


        # Cập nhật history
        history["train_loss"].append(avg_loss)
        # history["val_loss"].append(val_loss)
        history["val_metrics"].append(results)

        # Lưu log JSON
        with open(f"{state_path}/history.json", "w") as f:
            json.dump(history, f, indent=2)


        print(f"epoch_{epoch}, train_loss={avg_loss:.5f}, val_metrics={results}, - LR = {optim.param_groups[0]['lr']:.6f}")

        # ==== Vẽ biểu đồ ====
        # epochs = range(1, num_epochs+1)
        epochs = range(1, len(history["train_loss"]) + 1)
        loss_val = [m["loss"] for m in history["val_metrics"]]  # lấy metric silog từ val_metrics

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
    train_fn(device='cuda:0', load_state=False, state_path="/kaggle/working/hyp_depth_estimation/ours_checkpoints/16")
    # train_fn(device='cuda:0', load_state=False, state_path="/home/gremsy_guest/hyp_workspace/hyp_depth_estimation/ours_checkpoints/20")
