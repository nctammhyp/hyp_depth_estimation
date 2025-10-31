import os
import random
import numpy as np
import h5py
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import albumentations as A

iheight, iwidth = 480, 640


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f["rgb"])
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, C)
    # depth = np.array(h5f["anythingv2l_322_196"])
    depth = np.array(h5f['depth'])
    h5f.close()
    return rgb, depth


class NYUDataset(Dataset):
    def __init__(self, root_dir, train, loader=h5_loader):
        self.loader = loader
        self.output_size = (196, 322)  # (H, W)
        self.root_dir = root_dir
        self.train = train

        classes, class_to_idx = self.get_classes(root_dir)
        self.images = self.build_dataset(root_dir, class_to_idx)
        self.classes = classes
        self.class_to_idx = class_to_idx

        # Step 1: Resize (same for RGB and depth)
        self.resize = A.Resize(height=self.output_size[0], width=self.output_size[1])

        # Step 2: Augmentation (applied AFTER resize)
        self.augs = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5)
        ])

    def __len__(self):
        return len(self.images)

    def __getraw__(self, index):
        path, _ = self.images[index]
        return self.loader(path)

    def __getitem__(self, index):
        try:

            rgb, depth = self.__getraw__(index)
            rgb_tensor, depth_tensor = self.transform(rgb, depth)
            return rgb_tensor, depth_tensor
        
        except Exception as e:

            print(f"[WARNING] Skip corrupted file nuyv2: error: {e}")
            return None

    def build_dataset(self, root_dir, class_to_idx):
        images = []
        for class_name in sorted(os.listdir(root_dir)):
            dir_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(dir_path):
                continue
            for root, _, files in sorted(os.walk(dir_path)):
                for f in sorted(files):
                    if f.endswith(".h5"):
                        current_path = os.path.join(root, f)
                        images.append((current_path, class_to_idx[class_name]))
        return images

    def get_classes(self, root_dir):
        classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def transform(self, rgb, depth):
        # Step 1: Resize
        resized = self.resize(image=rgb, mask=depth)
        rgb_resized = resized["image"]
        depth_resized = resized["mask"]

        # Step 2: Augmentation (chỉ RGB)
        if self.train:
            rgb_aug = self.augs(image=rgb_resized)["image"]
        else:
            rgb_aug = rgb_resized

        # ✅ Step 3: Chia 255 trực tiếp
        rgb_aug = rgb_aug.astype(np.float32) / 255.0

        # ✅ Step 4: Chuyển sang tensor
        rgb_tensor = torch.from_numpy(rgb_aug).permute(2, 0, 1).contiguous()  # [C,H,W]
        depth_tensor = torch.from_numpy(depth_resized.astype(np.float32)).unsqueeze(0)  # [1,H,W]

        return rgb_tensor, depth_tensor

from torch.utils.data.dataloader import default_collate

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # Skip if entire batch failed
    return default_collate(batch)

def create_data_loaders(batch_size=8, subset=False):
    print("Creating dataset... patience.")
    base_dir = "/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/nyu_v2"
    # base_dir = r"D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\nyuv2_partial_406"

    train_path = os.path.join(base_dir, "train")
    val_path = os.path.join(base_dir, "val")

    if not os.path.isdir(train_path) or not os.path.isdir(val_path):
        raise RuntimeError("Dataset directory not found.")

    train_dataset = NYUDataset(train_path, train=True)
    val_dataset = NYUDataset(val_path, train=False)

    

    # args.train_set = train_dataset
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
        worker_init_fn=lambda work_id: np.random.seed(work_id)
    )

    # args.val_set = val_dataset
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    print("Finish loading datasets")

    # if subset:
    #     train_size = len(train_dataset)
    #     val_size = len(val_dataset)
    #     train_subset = Subset(train_dataset, np.random.choice(train_size, size=max(1, train_size // 100), replace=False))
    #     val_subset = Subset(val_dataset, np.random.choice(val_size, size=max(1, val_size // 100), replace=False))

    #     args.train_set = train_subset
    #     args.val_set = val_subset

    #     train_loader = DataLoader(train_subset, batch_size=args.bsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    #     val_loader = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader
