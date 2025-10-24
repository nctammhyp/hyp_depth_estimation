import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import albumentations as A
from torchvision.transforms import Compose
from transform import Resize, NormalizeImage, PrepareForNet, Crop  # Giữ nguyên

def depth_to_norm_log(depth, d_min=0.001, d_max=600.0):
    """
    Normalize depth to [0, 1] (log-space)
    - Input depth: giá trị càng xa càng nhỏ
    - Output s: càng xa -> càng lớn
    - d_min, d_max: hiểu theo nghĩa vật lý (gần nhỏ, xa lớn)
    """
    depth = np.clip(depth, d_min, d_max)
    log_d = np.log(depth)
    log_d_min, log_d_max = np.log(d_min), np.log(d_max)
    # đảo chiều để xa -> lớn
    s = (log_d_max - log_d) / (log_d_max - log_d_min)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


class DepthDataset(Dataset):
    def __init__(self, paths, mode="train", size=(224, 224)):
        self.paths = paths
        self.mode = mode
        self.size = size

        net_w, net_h = size

        # Data augmentation
        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast()
        ])

    def __len__(self):
        return len(self.paths)
    
    def normalize_depth(self, depth):
        return depth_to_norm_log(depth, d_min=0.001, d_max=600.0)

    def __getitem__(self, index):
        try:

            rgb_path, depth_path = self.paths[index]

            # Load RGB (BGR -> RGB)
            rgb = cv2.imread(rgb_path)[:, :, ::-1]  

            # Load Depth từ file .npy
            depth = np.load(depth_path).astype(np.float32)  # <-- Đọc file npy

            # Resize
            rgb = cv2.resize(rgb, self.size)
            depth = cv2.resize(depth, self.size)  # giữ nguyên giá trị float của depth

            # Augment cho ảnh RGB và Depth
            if self.mode == "train":
                augmented = self.augs(image=rgb, mask=depth)
                rgb, depth = augmented["image"] / 255.0, augmented["mask"]
            else:
                rgb, depth = rgb / 255.0, depth


            depth = self.normalize_depth(depth)

            # Chuyển sang tensor
            rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)  # [C, H, W]
            depth = torch.from_numpy(depth).float().unsqueeze(0)   # [1, H, W]

            return rgb, depth
        except Exception as e:
            print(f"[WARNING] Skip corrupted data: {rgb_path}, {depth_path}, error: {e}")
            return None  # thử lại ảnh khác




# ===================== Lấy danh sách file ảnh/label =====================
import os

def get_image_label_pairs(directory, img_ext=".png", label_ext=".npy"):
    img_root = os.path.join(directory, "images")
    # lbl_root = os.path.join(directory, "labels_npy")
    lbl_root = os.path.join(directory, "labels_npy_322_196")

    # print(f"img root: {img_root}")
    # print(f"lbl_root: {lbl_root}")
    
    pairs = []
    if not os.path.exists(img_root) or not os.path.exists(lbl_root):
        print("Thư mục rỗng!")
        return pairs

    for scene_name in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, scene_name)
        lbl_dir = os.path.join(lbl_root, scene_name)
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.endswith(img_ext)}
        lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir) if f.endswith(label_ext)}

        # Lấy giao nhau theo tên file, bỏ phần mở rộng
        common_keys = sorted(set(img_files.keys()) & set(lbl_files.keys()))
        for key in common_keys:
            rgb_path = os.path.join(img_dir, img_files[key])
            depth_path = os.path.join(lbl_dir, lbl_files[key])
            if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                pairs.append((rgb_path, depth_path))

    return pairs



# def subdataset_get_image_label_pairs(directory, img_ext=".png", label_ext=".png", phase = "train"):
#     img_root = os.path.join(directory, "images")
#     lbl_root = os.path.join(directory, "labels_npy")

#     pairs = []
#     if not os.path.exists(img_root) or not os.path.exists(lbl_root):
#         return pairs  # Không có thư mục thì trả rỗng

#     # Danh sách folder được phép lấy
#     allowed_scenes = {
#         # "scene_1_3MP",
#         # "scene_1_12MP",
#         # "scene_2_12MP",
#         # "scene_3_12MP",
#         # "scene_4_12MP",
#         # "scene_5_12MP",
#         "scene_6_12MP",
#         "scene_7_12MP",
#         "scene_8_12MP",
#         "scene_9_3MP",
#         "scene_10_3MP"}

#     for scene_name in sorted(os.listdir(img_root)):
#         if phase == 'train' and scene_name not in allowed_scenes:
#             continue  # Bỏ qua folder không nằm trong danh sách

#         img_dir = os.path.join(img_root, scene_name)
#         lbl_dir = os.path.join(lbl_root, scene_name)

#         if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
#             continue

#         # Lọc file ảnh và depth
#         img_files = set(f for f in os.listdir(img_dir) if f.endswith(img_ext))
#         lbl_files = set(f for f in os.listdir(lbl_dir) if f.endswith(label_ext))

#         # Lấy giao nhau để chắc chắn có cả RGB và depth
#         common_files = sorted(img_files & lbl_files)

#         for f in common_files:
#             rgb_path = os.path.join(img_dir, f)
#             depth_path = os.path.join(lbl_dir, f)
#             if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
#                 pairs.append((rgb_path, depth_path))

#     return pairs


from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # Skip if entire batch failed
    return default_collate(batch)

# ===================== Hàm tạo DataLoader =====================
def create_data_loaders(data_root, batch_size=64, size=(160, 128)):

    train_paths = get_image_label_pairs(os.path.join(data_root, "train"))
    val_paths   = get_image_label_pairs(os.path.join(data_root, "val"))

    train_dir = os.path.join(data_root, "train")
    # print(f"val path: {train_dir}")
    print(f"val_paths: {train_paths}")


    val_dir = os.path.join(data_root, "val")
    # print(f"val path: {val_dir}")
    print(f"val_paths: {val_paths}")




    # train_paths = subdataset_get_image_label_pairs(os.path.join(data_root, "train"), phase = "train")
    # val_paths   = subdataset_get_image_label_pairs(os.path.join(data_root, "val"), phase = "val")


    random.shuffle(train_paths)

    train_dataset = DepthDataset(train_paths, mode="train", size=size)
    val_dataset   = DepthDataset(val_paths, mode="val", size=size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=8,
                            pin_memory=True)

    return train_loader, val_loader


# ===================== Example =====================
if __name__ == "__main__":
    data_root = "/path/to/dataset"
    train_loader, val_loader = create_data_loaders(data_root, batch_size=8, size=(224, 224))

    for rgb, depth in train_loader:
        print(rgb.shape, depth.shape, rgb.min().item(), rgb.max().item(), depth.min().item(), depth.max().item())
        break