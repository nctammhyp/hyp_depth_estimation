import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def find_extreme_depths(train_loader, val_loader=None, save_dir="dataset_infor/outdoor_2"):
    """
    Quét toàn bộ dataset (train + val) để tìm:
      - Ảnh có giá trị depth lớn nhất
      - Ảnh có giá trị depth nhỏ nhất
    Sau đó lưu cặp RGB–Depth tương ứng ra file.

    Args:
        train_loader: DataLoader của tập train
        val_loader: DataLoader của tập val (optional)
        save_dir (str): Thư mục lưu kết quả

    Returns:
        dict chứa thông tin:
        {
            "max_val": float,
            "min_val": float,
            "max_depth_path": str,
            "min_depth_path": str,
            "max_rgb_path": str,
            "min_rgb_path": str
        }
    """
    os.makedirs(save_dir, exist_ok=True)

    max_val = -float('inf')
    min_val = float('inf')
    max_depth_map = None
    min_depth_map = None
    max_rgb_img = None
    min_rgb_img = None

    all_loaders = [train_loader]
    if val_loader is not None:
        all_loaders.append(val_loader)

    for loader in all_loaders:
        for batch in tqdm(loader, desc="🔍 Scanning dataset"):
            # Giả sử batch = (rgb, depth)
            rgb, depth = batch[0], batch[1]

            depth = depth.float()
            rgb = rgb.float()

            batch_max_val = depth.max().item()
            batch_min_val = depth.min().item()

            # ---- Max depth ----
            if batch_max_val > max_val:
                max_val = batch_max_val
                flat_idx = depth.argmax().item()
                batch_idx = flat_idx // (depth.shape[2] * depth.shape[3])
                max_depth_map = depth[batch_idx].squeeze().cpu().numpy()
                max_rgb_img = rgb[batch_idx].permute(1, 2, 0).cpu().numpy()
                max_rgb_img = np.clip(max_rgb_img / max_rgb_img.max(), 0, 1)

            # ---- Min depth ----
            if batch_min_val < min_val:
                min_val = batch_min_val
                flat_idx = depth.argmin().item()
                batch_idx = flat_idx // (depth.shape[2] * depth.shape[3])
                min_depth_map = depth[batch_idx].squeeze().cpu().numpy()
                min_rgb_img = rgb[batch_idx].permute(1, 2, 0).cpu().numpy()
                min_rgb_img = np.clip(min_rgb_img / min_rgb_img.max(), 0, 1)

    # ---- Save files ----
    paths = {}
    if max_depth_map is not None:
        paths["max_depth_path"] = os.path.join(save_dir, "max_depth.npy")
        paths["max_rgb_path"] = os.path.join(save_dir, "max_rgb.png")
        np.save(paths["max_depth_path"], max_depth_map)
        plt.imsave(paths["max_rgb_path"], max_rgb_img)

    if min_depth_map is not None:
        paths["min_depth_path"] = os.path.join(save_dir, "min_depth.npy")
        paths["min_rgb_path"] = os.path.join(save_dir, "min_rgb.png")
        np.save(paths["min_depth_path"], min_depth_map)
        plt.imsave(paths["min_rgb_path"], min_rgb_img)

    print(f"✅ Done!\nMax depth = {max_val:.4f}\nMin depth = {min_val:.4f}")
    print(f"Files saved in: {save_dir}")

    return {
        "max_val": max_val,
        "min_val": min_val,
        **paths
    }


def visualize_depth_pair(rgb_path, depth_path, title="Depth visualization"):
    """
    Hiển thị ảnh RGB và Depth tương ứng để kiểm tra kết quả.
    """
    rgb = plt.imread(rgb_path)
    depth = np.load(depth_path)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(rgb)
    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(depth, cmap="plasma")
    plt.title("Depth Map")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.suptitle(title)
    plt.show()
