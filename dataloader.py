import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import albumentations as A

from transform import Resize, NormalizeImage, PrepareForNet, Crop  # Giữ nguyên

# =========================
# LẤY FILE RGB + DEPTH
# =========================
def get_all_files_rgbd(directory):
    """
    Tạo list tuple (rgb_path, depth_path) cho dataset_ours/rgbd-scenes-v2
    """
    rgb_files = sorted([f for f in os.listdir(os.path.join(directory, "images")) if f.endswith("-color.png")])
    depth_files = sorted([f for f in os.listdir(os.path.join(directory, "labels")) if f.endswith("-depth.png")])

    all_paths = []
    for rgb, depth in zip(rgb_files, depth_files):
        rgb_path = os.path.join(directory, "images", rgb)
        depth_path = os.path.join(directory, "labels", depth)
        all_paths.append((rgb_path, depth_path))
    return all_paths

def get_all_files_nyu(directory):
    """
    Tạo list tuple (rgb_path, depth_path) cho dataset NYU (giống code cũ)
    """
    rgb_files = sorted([f for f in os.listdir(os.path.join(directory, "images")) if f.endswith(".jpg")])
    depth_files = sorted([f for f in os.listdir(os.path.join(directory, "labels")) if f.endswith(".png")])

    all_paths = []
    for rgb, depth in zip(rgb_files, depth_files):
        rgb_path = os.path.join(directory, "images", rgb)
        depth_path = os.path.join(directory, "labels", depth)
        all_paths.append((rgb_path, depth_path))
    return all_paths

# =========================
# LẤY FILE (ảnh, label) TỪ BẤT KỲ DATASET NÀO
# =========================
# ===================== Lấy danh sách file ảnh/label =====================
def get_image_label_pairs(directory, img_ext=".png", label_ext=".png"):
    img_root = os.path.join(directory, "images")
    lbl_root = os.path.join(directory, "labels")

    pairs = []
    if not os.path.exists(img_root) or not os.path.exists(lbl_root):
        return pairs  # Không có thư mục thì trả rỗng

    for scene_name in sorted(os.listdir(img_root)):
        img_dir = os.path.join(img_root, scene_name)
        lbl_dir = os.path.join(lbl_root, scene_name)

        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        # Lọc file ảnh và depth
        img_files = set(f for f in os.listdir(img_dir) if f.endswith(img_ext))
        lbl_files = set(f for f in os.listdir(lbl_dir) if f.endswith(label_ext))

        # Lấy giao nhau để chắc chắn có cả RGB và depth
        common_files = sorted(img_files & lbl_files)

        for f in common_files:
            rgb_path = os.path.join(img_dir, f)
            depth_path = os.path.join(lbl_dir, f)
            if os.path.isfile(rgb_path) and os.path.isfile(depth_path):
                pairs.append((rgb_path, depth_path))

    return pairs


# =========================
# DATASET CHUNG
# =========================
class DepthDataset(Dataset):
    def __init__(self, paths, mode="train", size=(224, 224)):
        self.paths = paths
        self.mode = mode
        self.size = size

        net_w, net_h = size
        self.transform = Compose([
            NormalizeImage(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])


        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(hue=0.1, contrast=0.1, brightness=0.1,
                          saturation=0.1, p=0.5),
            A.GaussNoise(
                std_range=[0.03, 0.07],
                mean_range=[0, 0.2],
                per_channel=True,
                noise_scale_factor=1,
                p=0.3
            )
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rgb_path, depth_path = self.paths[index]

        # print(f"rgb path: {rgb_path}")
        # print(f"depth path: {depth_path}")

        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR -> RGB
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

        rgb = cv2.resize(rgb, self.size)
        depth = cv2.resize(depth, self.size)  # giữ giá trị nguyên depth

        if self.mode == "train":
            augmented = self.augs(image=rgb, mask=depth)
            rgb, depth = augmented["image"]/255.0, augmented["mask"]/255.0
        else:
            rgb, depth = rgb/255.0, depth/255.0

        sample = self.transform({"image": rgb, "depth": depth})
        sample["image"] = torch.from_numpy(sample["image"]).float()
        sample["depth"] = torch.from_numpy(sample["depth"]).float()
        return sample


# =========================
# LẤY DATALOADER CHO 2 DATASET
# =========================
def get_combined_dataloaders(batch_size=16):
    # NYU Depth v2
    # nyu_train = get_all_files_nyu("dataset_ours/nyudepthv2/train")
    # nyu_val = get_all_files_nyu("dataset_ours/nyudepthv2/val")
    # nyu_test = get_all_files_nyu("dataset_ours/nyudepthv2/test")

    # RGBD Scenes v2
    # rgbd_train = get_all_files_rgbd("dataset_ours/rgbd-scenes-v2/train")
    # rgbd_val = get_all_files_rgbd("dataset_ours/rgbd-scenes-v2/val")
    # rgbd_test = get_all_files_rgbd("dataset_ours/rgbd-scenes-v2/test")

    # dan_train = get_image_label_pairs(r"D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hypdataset_v1_subtest\train")
    # dan_val   = get_image_label_pairs(r"D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hypdataset_v1_subtest\val")

    # for windows testing
    data_root = r"D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hypdataset_v1_subtest"

    train_paths = get_image_label_pairs(os.path.join(data_root, "train"))
    val_paths   = get_image_label_pairs(os.path.join(data_root, "val"))

    # Shuffle train
    random.shuffle(train_paths)

    train_dataset = DepthDataset(train_paths, mode="train", size=(160, 128))
    val_dataset = DepthDataset(val_paths, mode="val", size=(160, 128))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4,
                              pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=1,
                            shuffle=False, num_workers=4,
                            pin_memory=True, drop_last=False)
    return train_loader, val_loader