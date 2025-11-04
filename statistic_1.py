import cv2
import numpy as np

# Đường dẫn
rgb_path = r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\dataset_infor\outdoor_2\min_rgb.png"
depth_path = r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\dataset_infor\outdoor_2\min_depth.npy"

def show_rgb_depth_cv2(rgb_path, depth_path, window_name="RGB-Depth Viewer", depth_min=5, depth_max=500):
    """
    Đọc ảnh RGB (.png, .jpg) và depth (.npy),
    chỉ hiển thị vùng depth trong [depth_min, depth_max].
    """
    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"Không thể đọc ảnh RGB: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # ---- Đọc depth ----
    depth = np.load(depth_path)
    if depth is None:
        raise FileNotFoundError(f"Không thể đọc depth: {depth_path}")

    # ---- Giới hạn giá trị depth ----
    depth = np.clip(depth, depth_min, depth_max)

    # ---- Chuẩn hóa để hiển thị ----
    depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_PLASMA)

    # ---- Ghép cạnh nhau và hiển thị ----
    combined = np.hstack((cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_colored))
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


show_rgb_depth_cv2(
    rgb_path=rgb_path,
    depth_path=depth_path,
    depth_min=5,
    depth_max=1000
)
