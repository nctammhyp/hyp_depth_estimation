import os
import torch
import numpy as np
from depth_model.fdepth_resnet_v2 import FastDepthV2

# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRUNED_MODEL_PATH = r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\checkpoint_for_optimize\checkpoint_118\47\pruned_model_full_13.pth"
ONNX_PATH = r"D:\ubuntu\test_algorithm\deep_learning\FastDepth_src\checkpoint_for_optimize\opt_118\pruned_model_full_13.onnx"
INPUT_SHAPE = (3, 196, 322)  # C,H,W
BATCH_SIZE = 1

print(f"Using device: {DEVICE}")

# =========================
# 1. Load FULL MODEL (weights_only=False)
# =========================
checkpoint = torch.load(PRUNED_MODEL_PATH, map_location=DEVICE, weights_only=False)

# Nếu checkpoint là model -> dùng trực tiếp
if isinstance(checkpoint, torch.nn.Module):
    model = checkpoint.to(DEVICE)
    print("Loaded FULL MODEL checkpoint")

# Nếu checkpoint là state_dict -> khởi tạo model rồi load
else:
    model = FastDepthV2().to(DEVICE)
    model.load_state_dict(checkpoint)
    print("Loaded STATE_DICT checkpoint")

model.eval()
print(f"Loaded pruned model from {PRUNED_MODEL_PATH}")

# =========================
# 2. Export to ONNX
# =========================
dummy_input = torch.randn(BATCH_SIZE, *INPUT_SHAPE, device=DEVICE)

torch.onnx.export(
    model,
    dummy_input,
    ONNX_PATH,
    export_params=True,
    opset_version=17,
    input_names=["input"],
    output_names=["output"]
)

print(f"✅ Exported pruned model to: {ONNX_PATH}")
