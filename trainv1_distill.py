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
import dataloader_v6

from depth_anything_v2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

import torch
import torch.nn as nn
from tqdm import tqdm
from metric_depth.util.loss import SiLogLoss, DepthLoss, SiLogL1Loss

import glob
import json

torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

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


def train_mse_loss(teacher, student, train_loader, val_loader, epochs, learning_rate, feature_map_weight, ce_loss_weight, device):

    state_path = ""
    distill_loss = nn.MSELoss()
    main_loss = SiLogLoss() # author's loss

    max_depth = 255



    optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

    teacher.to(device)
    student.to(device)
    teacher.eval()  # Teacher set to evaluation mode
    student.train() # Student to train mode

    history = {"train_loss": [], "val_loss": [], "val_metrics": []}



    for epoch in range(epochs):
        total_loss = 0.0
        student.train()
        for i , (inputs,labels) in enumerate(tqdm(train_loader, total=len(train_loader))):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Again ignore teacher logits
            with torch.no_grad():
                _, teacher_feature_map = teacher(inputs)

            # Forward pass with the student model
            student_logits, regressor_feature_map = student(inputs)

            # Calculate the loss
            hidden_rep_loss = distill_loss(regressor_feature_map, teacher_feature_map)

            mask = (labels > 1e-3) & (labels <= max_depth) & torch.isfinite(labels)
            # Calculate the true label loss
            label_loss = main_loss(student_logits, labels, mask)

            # Weighted sum of the two losses
            loss = feature_map_weight * hidden_rep_loss + ce_loss_weight * label_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ===== Validation =====
        student.eval()
        # results = {'d1': 0, 'rmse': 0}
        results = {'d1': 0, 'abs_rel': 0, 'rmse': 0, 'mae': 0, 'silog': 0}
        # test_loss = 0

        with torch.no_grad():
            for i , (input,target) in tqdm(enumerate(val_loader)):
                img, depth = input.to(device), target.to(device)

                pred = student(img)

                # test_loss += criterion('l1',pred, depth).item()
                # pred = pred.squeeze(1).squeeze(0)

                # mask = (depth >= 0.001)
                # cur_results = eval_depth(pred, depth)

                # print(depth)


                mask = (depth > 1e-3) & (depth <= max_depth) & torch.isfinite(depth)

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
            "model": student.state_dict(),
            "optim": optimizer.state_dict(),
            # "scheduler": scheduler.state_dict()
        }, f"{state_path}/last_checkpoint_{epoch}.pth")

        # if results['abs_rel'] < best_val_absrel:
        if results['silog'] < best_val:

            best_val = results['silog']
            new_ckpt = f"{state_path}/checkpoint_best_{epoch}.pth"

            # 1. Lưu checkpoint mới
            # torch.save(model.state_dict(), new_ckpt)
            torch.save({
                "model": student.state_dict(),
                "optim": optimizer.state_dict(),
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
            inference_sample(student, state_path, device, model_type="best")

        inference_sample(student, state_path, device, model_type="last")


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



def trainer():
    # teacher

    # model_weights_path =  '/kaggle/working/depth_anything_v2_vitl.pth'
    # model_configs = {
    #         'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    #         'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    #         'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    #         'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    #     }
    # model_encoder = 'vitl'

    device = "cpu"

    teacher_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
    teacher_model.load_state_dict(torch.load(r'depth_anything_v2\checkpoint\depth_anything_v2_vitl.pth', map_location='cpu'))

    # student
    student_model = FastDepthV2()
    print("Pretrained ImageNet weights đã được load thành công!")

    train_loader, val_loader = dataloader_v6.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/hyp_dataset_v1", batch_size=512, size=(160, 128))

    # Train and test once again
    train_mse_loss(teacher=teacher_model, student=student_model, train_loader=train_loader, val_loader=val_loader, epochs=10, learning_rate=0.001, feature_map_weight=0.25, ce_loss_weight=0.75, device=device)

if __name__ == "__main__":
    trainer()