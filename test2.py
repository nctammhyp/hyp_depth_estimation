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


d = np.array([0.001, 0.1, 1.0, 10.0, 600.0])
s = depth_to_norm_log(d)
print("Depth:", d)
print("Normed:", s)
print("Recovered:", norm_log_to_depth(s))
