import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from depth_model.fdepth_resnet_v2 import FastDepthV2

def load_image(img_path, size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, size)
    img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)
    return img_tensor, img_resized

def normalize_depth(depth):
    depth -= depth.min()
    depth /= (depth.max() + 1e-8)
    return depth

def inference(image_path, gt_path):
    # Load model
    model = FastDepthV2()
    epsilon = 1e-8

    print("----------   load checkpoint -------------")
    checkpoint = torch.load(
        r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\checkpoint_for_optimize\checkpoint_118\44\best_checkpoint.pth", 
        map_location="cuda:0"
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to("cuda:0")
    model.eval()

    # Load and preprocess image
    img_rgb = cv2.imread(image_path)[:, :, ::-1]  # BGR -> RGB
    rgb_resized = cv2.resize(img_rgb, (322, 196))
    rgb_tensor = torch.from_numpy(rgb_resized / 255.0).float().permute(2, 0, 1).unsqueeze(0).to("cuda:0")

    with torch.no_grad():
        pred_depth = model(rgb_tensor).cpu().squeeze(0).squeeze(0).numpy()
    
    # Normalize predicted depth
    pred_depth = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + epsilon)

    # Load ground truth depth
    gt_depth = np.load(gt_path)
    # Normalize gt depth for display
    gt_depth_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + epsilon)

    # Show input, predicted depth, and ground truth depth
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img_rgb)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Predicted Depth")
    plt.imshow(pred_depth, cmap='inferno')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Ground Truth Depth")
    plt.imshow(gt_depth_norm, cmap='inferno')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = r"D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hyp_outdoor_v2\val\images\IMG_0041\005929.jpg"
    gt_path = r"D:\ubuntu\test_algorithm\deep_learning\hyp_dataset\hyp_outdoor_v2\val\labels_npy_322_196\IMG_0041\005929.npy"
    inference(image_path, gt_path)
