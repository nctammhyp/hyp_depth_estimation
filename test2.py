import os

import numpy as np

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

def norm_log_to_depth(s, d_min=0.001, d_max=600.0):
    log_d_min, log_d_max = np.log(d_min), np.log(d_max)
    log_d = log_d_max - s * (log_d_max - log_d_min)
    return np.exp(log_d).astype(np.float32)

print("-------------------------------------------------------------------------------")

def depth_to_inv_norm_log(depth, d_min=0.001, d_max=600.0):
    """
    Normalize inverse depth (1/depth) to [0, 1] in log space.
    
    - Input depth: càng xa -> giá trị càng nhỏ
    - Output s: càng xa -> càng nhỏ (tức depth xa -> inv_depth nhỏ)
    - d_min, d_max: giới hạn vật lý (mét)
    """
    depth = np.clip(depth, d_min, d_max)
    inv_d = 1.0 / depth

    inv_d_min = 1.0 / d_max  # nhỏ nhất (xa nhất)
    inv_d_max = 1.0 / d_min  # lớn nhất (gần nhất)

    log_inv_d = np.log(inv_d)
    log_inv_d_min, log_inv_d_max = np.log(inv_d_min), np.log(inv_d_max)

    s = (log_inv_d - log_inv_d_min) / (log_inv_d_max - log_inv_d_min)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def inv_norm_log_to_depth(s, d_min=0.001, d_max=600.0):
    inv_d_min, inv_d_max = 1.0 / d_max, 1.0 / d_min
    log_inv_d_min, log_inv_d_max = np.log(inv_d_min), np.log(inv_d_max)
    log_inv_d = log_inv_d_min + s * (log_inv_d_max - log_inv_d_min)
    inv_d = np.exp(log_inv_d)
    depth = 1.0 / inv_d
    return depth

print("-------------------------------------------------------------------------------")

def depth_to_inv_norm(depth, d_min=0.001, d_max=600.0):
    """
    Normalize inverse depth (1/depth) linearly to [0, 1].
    - Input depth: càng xa → càng nhỏ
    - Output s: càng xa → càng nhỏ (0), càng gần → càng lớn (1)
    """
    depth = np.clip(depth, d_min, d_max)
    inv_d = 1.0 / depth
    inv_d_min = 1.0 / d_max
    inv_d_max = 1.0 / d_min
    s = (inv_d - inv_d_min) / (inv_d_max - inv_d_min)
    return np.clip(s, 0.0, 1.0).astype(np.float32)


def inv_norm_to_depth(s, d_min=0.001, d_max=600.0):
    inv_d_min = 1.0 / d_max
    inv_d_max = 1.0 / d_min
    inv_d = inv_d_min + s * (inv_d_max - inv_d_min)
    depth = 1.0 / inv_d
    return depth


print("-----------------------------------------------------")


def depth_to_inv_norm(depth, d_min=0.001, d_max=600.0):
    """
    Normalize inverse depth (1/depth) linearly to [0, 1],
    nhưng KHÔNG clip depth > d_max — để giữ thông tin vùng ngoài.
    """
    depth = np.maximum(depth, d_min)  # chỉ tránh chia 0
    inv_d = 1.0 / depth

    inv_d_min = 1.0 / d_max
    inv_d_max = 1.0 / d_min

    # Chuẩn hóa tuyến tính theo khoảng [inv_d_min, inv_d_max]
    s = (inv_d - inv_d_min) / (inv_d_max - inv_d_min)
    # không clip ngay, chỉ đảm bảo giới hạn
    return s.astype(np.float32)


# d = np.array([0.001, 0.1, 1.0, 10.0, 600.0])
# s = depth_to_norm_log(d)
# print("Depth:", d)
# print("Normed:", s)
# print("Recovered:", norm_log_to_depth(s))


d = np.array([0.001, 0.1, 1.0, 10.0, 600.0, 1000.0])
s = depth_to_inv_norm(d)
print("Depth:", d)
print("Normed:", s)
print("Recovered:", inv_norm_to_depth(s))