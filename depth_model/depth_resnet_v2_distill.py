"""
Model có sử dụng maxdepth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import math,time

# import resnet18
from depth_model import resnet18

def ConvBlock(in_channels,out_channels,kernel_size,stride,padding):
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

def DWConvBlock(in_channels,out_channels,kernel_size,stride,padding = None):
  if padding == None:
    padding = (kernel_size - 1) // 2
  return nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

class NNConv5_DecoderV2(nn.Module):
  def __init__(self, kernel_size, depthwise=True):
    super(NNConv5_DecoderV2, self).__init__()
    if (depthwise):
      self.conv1 = nn.Sequential(DWConvBlock(512,512,kernel_size,1),ConvBlock(512,256,1,1,0)) #14X14
      self.conv2 = nn.Sequential(DWConvBlock(256,256,kernel_size,1),ConvBlock(256,128,1,1,0)) #28 X 28
      self.conv3 = nn.Sequential(DWConvBlock(128,128,kernel_size,1),ConvBlock(128,64,1,1,0)) # 56X56
      self.conv4 = nn.Sequential(DWConvBlock(64,64,kernel_size,1),ConvBlock(64,64,1,1,0)) #112 X 112
      self.conv5 = nn.Sequential(DWConvBlock(64,64,kernel_size,1),ConvBlock(64,64,1,1,0)) #224 X 224

    self.output = ConvBlock(64,1,1,1,0)
  def forward(self,x):
    x = F.interpolate(self.conv1(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv2(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv3(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv4(x), scale_factor=2, mode='nearest')
    x = F.interpolate(self.conv5(x), scale_factor=2, mode='nearest')
    return self.output(x)


class FastDepthV2(nn.Module):
  def __init__(self, kernel_size=5, max_depth = 600):
    super(FastDepthV2,self).__init__()
    resnet = resnet18.load_resnet18()
    # Bỏ avgpool và fc
    self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # lấy từ conv1 -> layer4
    # print(self.encoder)
    # print("-----------------------------------------")
    # print(self.encoder[0])

    self.decoder = NNConv5_DecoderV2(kernel_size)
    self.max_depth = max_depth
  def forward(self,x):
    # print("debug 1:", x.min().item(), x.max().item(), "NaN:", torch.isnan(x).any().item())
    x = self.encoder[0](x)
    # print(f"fea 1: {x.size()}")

    x = self.encoder[1](x)
    # print(f"fea 2: {x.size()}")

    x = self.encoder[2](x)
    # print(f"fea 3: {x.size()}")

    layer1 = x
    
    x = self.encoder[3](x)

    # layer2 = x

    # print(f"fea 4: {x.size()}")

    x = self.encoder[4](x)

    layer2 = x

    # print(f"fea 5: {x.size()}")

    x = self.encoder[5](x)

    layer3 = x

    # print(f"fea 6: {x.size()}")

    x = self.encoder[6](x)

    # layer3 = x

    # print(f"fea 7: {x.size()}")

    x = self.encoder[7](x)
    # print(f"fea 8: {x.size()}")

    # x = self.decoder.conv1(x)
    x = F.interpolate(self.decoder.conv1(x), scale_factor=2, mode='nearest')

    # print(f"dec 1: {x.size()}")

    # x = self.decoder.conv2(x)

    # print(f"size before: {self.decoder.conv2(x).size()}") # result torch.Size([1, 128, 10, 8])
    x = F.interpolate(self.decoder.conv2(x), scale_factor=2, mode='nearest')
    # print(f"size after: {x.size()}", flush=True) # result: torch.Size([1, 128, 20, 16])
    # print(f"size layer: {layer3.size()}", flush=True) # result: torch.Size([1, 128, 20, 16])
    # layer3 = F.interpolate(layer3, size=x.shape[2:], mode='bilinear', align_corners=False)


    x = x + layer3

    # print(f"dec 2: {x.size()}")

    x = F.interpolate(self.decoder.conv3(x), scale_factor=2, mode='nearest')
    # layer2 = F.interpolate(layer2, size=x.shape[2:], mode='bilinear', align_corners=False)


    # print(f"dec 3: {x.size()}")

    x = x + layer2

    x= F.interpolate(self.decoder.conv4(x), scale_factor=2, mode='nearest')
    # layer1 = F.interpolate(layer1, size=x.shape[2:], mode='bilinear', align_corners=False)

    # print(f"dec 4: {x.size()}")

    student_feature_map = x
    
    x = x+layer1

    x= F.interpolate(self.decoder.conv5(x), scale_factor=2, mode='nearest')

    # print(f"dec 5: {x.size()}")


    # print("debug 2:", x.min().item(), x.max().item(), "NaN:", torch.isnan(x).any().item())

    # print(x.size())

    # x = F.interpolate(self.decoder.conv1(x), scale_factor=2, mode='nearest')

    # print(f"fea 23: {x.size()}")

    x = self.decoder.output(x)          # output raw logits
    x = torch.sigmoid(x) * self.max_depth  # scale về [0, max_depth]
    # x = F.relu(x)

    return x, student_feature_map
  
if __name__ == "__main__":
  # Tạo model custom từ scratch
  model = FastDepthV2()
  print("Pretrained ImageNet weights đã được load thành công!")

  # 168, 126
  dummy_input = torch.randn(1, 3, 224, 224)
  output, student_feature_map = model(dummy_input)
  print("Output shape:", output.shape)  # [1, 1000]
  print(f"student_feature_map: {student_feature_map.shape}")
