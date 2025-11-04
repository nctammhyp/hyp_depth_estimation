import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ======================
# Import dataloader từ file bạn có
# ======================
from support.dataloader import nyuv2_dataloader_v2, cross_dataset, hyp_dataloader_v3, outdoor_v1, outdoor_v2

# ----------------------
# 1. Load dataset
# ----------------------
train_loader, val_loader = outdoor_v2.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/outdoor_2", batch_size=16, size=(322, 196))

# ----------------------
# 2. Trích xuất đặc trưng đơn giản cho từng ảnh
# ----------------------
def extract_features(rgb_batch):
    """
    rgb_batch: Tensor [B,3,H,W] (0-1)
    Output: list[feature vector]
    """
    rgb_np = rgb_batch.numpy()
    feats = []
    for img in rgb_np:
        # Flatten [3,H,W] -> [H,W,3]
        img = np.transpose(img, (1,2,0))
        f = []
        # mean/std mỗi kênh
        for c in range(3):
            f.append(img[...,c].mean())
            f.append(img[...,c].std())
        # histogram (16-bin mỗi kênh)
        for c in range(3):
            hist, _ = np.histogram(img[...,c], bins=16, range=(0,1))
            f.extend(hist / np.sum(hist))
        feats.append(f)
    return feats

# ----------------------
# 3. Thu thập dữ liệu đặc trưng
# ----------------------
X, y = [], []

# Train = 0
for rgb, depth in tqdm(train_loader, desc="Extracting train feats", total=len(train_loader)):
    feats = extract_features(rgb)
    X.extend(feats)
    y.extend([0]*len(feats))
    # if len(X) > 2000:  # Giới hạn để chạy nhanh
    #     break

# Val = 1
for rgb, depth in tqdm(train_loader, desc="Extracting val feats", total=len(train_loader)):
    feats = extract_features(rgb)
    X.extend(feats)
    y.extend([1]*len(feats))
    # if len(X) > 4000:
    #     break

X = np.array(X)
y = np.array(y)

print("Feature shape:", X.shape, "Labels:", np.bincount(y))

# ----------------------
# 4. Train model phân biệt train/val
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:,1]

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
