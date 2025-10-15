import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y1 = F.relu(x)
y2 = F.relu(y1)

print(y1)  # tensor([0., 0., 0., 1., 2.])
print(y2)  # tensor([0., 0., 0., 1., 2.]) → không đổi
