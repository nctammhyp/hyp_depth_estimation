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
# train_loader, val_loader = outdoor_v2.create_data_loaders("/home/gremsy_guest/hyp_workspace/depth_dataset/datasets/outdoor_2", batch_size=16, size=(322, 196))
train_loader, val_loader = nyuv2_dataloader_v2.create_data_loaders()


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
for rgb, depth in tqdm(val_loader, desc="Extracting val feats", total=len(val_loader)):
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


# ========== Tạo thư mục lưu ==========
save_dir = "dataset_infor/nyuv2"
os.makedirs(save_dir, exist_ok=True)

# ========== Hàm tính đặc trưng thống kê ==========
def collect_image_stats(loader, n_max=1000):
    """
    Trích xuất thống kê cơ bản của RGB images từ loader.
    """
    means, stds, brightness = [], [], []
    hist_all = np.zeros((3, 16))
    
    for rgb, depth in tqdm(loader, total=len(loader)):
        rgb_np = rgb.numpy()
        for img in rgb_np:
            img = np.transpose(img, (1,2,0))
            means.append(img.mean(axis=(0,1)))
            stds.append(img.std(axis=(0,1)))
            brightness.append(img.mean())
            for c in range(3):
                h, _ = np.histogram(img[...,c], bins=16, range=(0,1))
                hist_all[c] += h
        if len(means) > n_max:
            break
    means = np.array(means)
    stds = np.array(stds)
    brightness = np.array(brightness)
    hist_all = hist_all / hist_all.sum(axis=1, keepdims=True)
    return means, stds, brightness, hist_all

# ========== Thu thập thống kê ==========
print("Collecting TRAIN stats...")
train_means, train_stds, train_brightness, train_hist = collect_image_stats(train_loader)

print("Collecting VAL stats...")
val_means, val_stds, val_brightness, val_hist = collect_image_stats(val_loader)

# ========== 1. Biểu đồ brightness ==========
plt.figure(figsize=(6,4))
plt.hist(train_brightness, bins=50, alpha=0.6, label='train', color='blue')
plt.hist(val_brightness, bins=50, alpha=0.6, label='val', color='orange')
plt.xlabel("Brightness (mean pixel value)")
plt.ylabel("Count")
plt.legend()
plt.title("Brightness Distribution (Train vs Val)")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "brightness_distribution.png"))
plt.close()

# ========== 2. Mean per channel ==========
plt.figure(figsize=(6,4))
for i, c in enumerate(['R','G','B']):
    plt.hist(train_means[:,i], bins=40, alpha=0.5, label=f'Train {c}')
    plt.hist(val_means[:,i], bins=40, alpha=0.5, label=f'Val {c}')
plt.xlabel("Mean pixel value per channel")
plt.legend()
plt.title("Color Channel Mean Distribution")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "color_mean_distribution.png"))
plt.close()

# ========== 3. Std per channel ==========
plt.figure(figsize=(6,4))
for i, c in enumerate(['R','G','B']):
    plt.hist(train_stds[:,i], bins=40, alpha=0.5, label=f'Train {c}')
    plt.hist(val_stds[:,i], bins=40, alpha=0.5, label=f'Val {c}')
plt.xlabel("Std per channel")
plt.legend()
plt.title("Color Channel Std Distribution")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "color_std_distribution.png"))
plt.close()

# ========== 4. Average histogram RGB ==========
plt.figure(figsize=(6,4))
bins = np.linspace(0,1,16)
for i, c in enumerate(['R','G','B']):
    plt.plot(bins[:-1], train_hist[i], label=f'Train {c}', linestyle='--')
    plt.plot(bins[:-1], val_hist[i], label=f'Val {c}', linestyle='-')
plt.title("Normalized RGB Histograms (Train vs Val)")
plt.xlabel("Pixel value")
plt.ylabel("Normalized frequency")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "rgb_histogram_distribution.png"))
plt.close()

# ========== 5. Lưu vài ảnh minh hoạ ==========
import matplotlib.pyplot as plt

def save_sample_images(train_loader, val_loader, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    train_rgb, _ = next(iter(train_loader))
    val_rgb, _ = next(iter(val_loader))
    for i in range(min(4, len(train_rgb))):
        fig, axes = plt.subplots(1,2, figsize=(6,3))
        axes[0].imshow(np.transpose(train_rgb[i].numpy(), (1,2,0)))
        axes[0].set_title("Train Sample")
        axes[1].imshow(np.transpose(val_rgb[i].numpy(), (1,2,0)))
        axes[1].set_title("Val Sample")
        for ax in axes: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_pair_{i}.png"))
        plt.close()

save_sample_images(train_loader, val_loader, os.path.join(save_dir, "samples"))

print(f"✅ Visualization saved to: {save_dir}")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ========== PCA 2D visualization ==========
print("Computing PCA...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(7,6))
plt.scatter(
    X_pca[y==0,0], X_pca[y==0,1], 
    s=5, alpha=0.4, label='Train', color='blue'
)
plt.scatter(
    X_pca[y==1,0], X_pca[y==1,1], 
    s=5, alpha=0.4, label='Val', color='orange'
)
plt.title(f"PCA 2D Feature Space (Train vs Val)\nExplained Var: {pca.explained_variance_ratio_.sum():.2f}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "pca_scatter.png"))
plt.close()

print("✅ PCA scatter saved:", os.path.join(save_dir, "pca_scatter.png"))
