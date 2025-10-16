import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


# ===================== Dataset Class =====================
class DepthDataset(Dataset):
    def __init__(self, paths, size=(224, 224)):
        self.paths = paths
        self.size = size

        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        try:
            rgb_path, depth_path = self.paths[index]

            # Load RGB
            rgb = cv2.imread(rgb_path)
            if rgb is None:
                raise FileNotFoundError(f"Không thể đọc ảnh: {rgb_path}")
            rgb = rgb[:, :, ::-1]  # BGR -> RGB

            # Load Depth
            depth = np.load(depth_path).astype(np.float32)

            # Resize
            rgb = cv2.resize(rgb, self.size)
            depth = cv2.resize(depth, self.size)

            # Augmentation
            augmented = self.augs(image=rgb, mask=depth)
            rgb, depth = augmented["image"], augmented["mask"]

            # Normalize
            rgb = rgb / 255.0

            # To Tensor
            rgb = torch.from_numpy(rgb).float().permute(2, 0, 1)  # [C,H,W]
            depth = torch.from_numpy(depth).float().unsqueeze(0)   # [1,H,W]

            return rgb, depth
        
        except Exception as e:
            print(f"[WARNING] Skip corrupted data: {rgb_path}, {depth_path}, error: {e}")
            return None  # thử lại ảnh khác


# ===================== Pairing Function =====================
def get_image_label_pairs(img_root, lbl_root, img_exts=(".png", ".jpg", ".jpeg"), label_ext=".npy"):
    pairs = []

    # Duyệt toàn bộ file ảnh trong thư mục con
    for root, _, files in os.walk(img_root):
        for fname in files:
            if not fname.lower().endswith(img_exts):
                continue

            img_path = os.path.join(root, fname)

            # Đường dẫn tương đối so với img_root
            rel_path = os.path.relpath(img_path, img_root)
            rel_no_ext = os.path.splitext(rel_path)[0]

            # Ghép label tương ứng trong lbl_root
            depth_path = os.path.join(lbl_root, rel_no_ext + label_ext)
            if os.path.exists(depth_path):
                pairs.append((img_path, depth_path))
            else:
                # print(f"⚠️ Missing label for {rel_path}")
                pass

    return pairs

from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # Skip if entire batch failed
    return default_collate(batch)

# ===================== Create train loader =====================
def create_loader(dataset_paths, batch_size=16, size=(160, 128)):
    """
    dataset_paths: list of tuples (image_dir, label_dir)
    """
    all_pairs = []
    for img_dir, lbl_dir in dataset_paths:
        pairs = get_image_label_pairs(img_dir, lbl_dir)
        all_pairs.extend(pairs)
        print(f"✔️ Loaded {len(pairs)} pairs from {os.path.basename(img_dir)}")

    random.shuffle(all_pairs)
    print(f"👉 Tổng cộng {len(all_pairs)} cặp ảnh-depth")

    train_set = DepthDataset(all_pairs, size=size)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=8,
                              pin_memory=True, drop_last=True, collate_fn=collate_fn)
    return train_loader


def create_train_loader(batch_size=16, size=(160, 128)):
    base = "/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/cross_dataset_v3"

    dataset_paths = [
        (os.path.join(base, "images/eth3d"),        os.path.join(base, "labels_npy/eth3d")),
        (os.path.join(base, "images/small_diode"),  os.path.join(base, "labels_npy/small_diode")),
        # Bạn có thể thêm nhiều dataset khác ở đây
    ]

    train_loader = create_loader(dataset_paths, batch_size=batch_size, size=size)

    return train_loader





# ===================== Example =====================
if __name__ == "__main__":
    base = "/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/cross_dataset_v3"

    dataset_paths = [
        (os.path.join(base, "images/eth3d"),        os.path.join(base, "labels_npy/eth3d")),
        (os.path.join(base, "images/small_diode"),  os.path.join(base, "labels_npy/small_diode")),
        # Bạn có thể thêm nhiều dataset khác ở đây
    ]

    train_loader = create_train_loader(dataset_paths, batch_size=4, size=(224, 224))

    for rgb, depth in train_loader:
        print(f"rgb: {rgb.shape}, depth: {depth.shape}, rgb_range=({rgb.min():.3f},{rgb.max():.3f}), depth_range=({depth.min():.3f},{depth.max():.3f})")
        break
