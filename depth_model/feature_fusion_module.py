import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_ch, out_ch, bias=False):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

def depthwise_conv3x3(ch, stride=1, padding=1):
    return nn.Conv2d(ch, ch, kernel_size=3, stride=stride, padding=padding, groups=ch, bias=False)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_ch_enc, in_ch_dec, out_ch, reduction=1):
        super().__init__()
        reduced_ch = max(out_ch // reduction, 8)

        self.enc_proj = conv1x1(in_ch_enc, reduced_ch, bias=False)
        self.dec_proj = conv1x1(in_ch_dec, reduced_ch, bias=False)

        self.dec_dw = nn.Sequential(
            nn.ReLU(inplace=True),
            depthwise_conv3x3(reduced_ch),
            nn.BatchNorm2d(reduced_ch),
            nn.ReLU(inplace=True)
        )

        self.merge_conv = nn.Sequential(
            conv1x1(reduced_ch * 2, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.refine_dw = nn.Sequential(
            depthwise_conv3x3(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        e = self.enc_proj(enc_feat)                # (B, reduced_ch, H, W)
        d = self.dec_proj(dec_feat)                # (B, reduced_ch, h, w)
        d = F.interpolate(d, size=e.shape[2:], mode='bilinear', align_corners=False)
        d = self.dec_dw(d)                         # (B, reduced_ch, H, W)
        x = torch.cat([e, d], dim=1)               # (B, 2*reduced_ch, H, W)
        x = self.merge_conv(x)                     # (B, out_ch, H, W)
        x = self.refine_dw(x)                      # (B, out_ch, H, W)
        return x

class FastFFM(nn.Module):
    def __init__(self, in_ch_enc, in_ch_dec, out_ch, reduction=1):
        super().__init__()
        reduced_ch = max(out_ch // reduction, 8)

        self.enc_proj = conv1x1(in_ch_enc, reduced_ch, bias=False)
        self.dec_proj = conv1x1(in_ch_dec, reduced_ch, bias=False)

        self.merge_conv = nn.Sequential(
            conv1x1(reduced_ch * 2, out_ch, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.refine_dw = nn.Sequential(
            depthwise_conv3x3(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        e = self.enc_proj(enc_feat)
        d = self.dec_proj(dec_feat)
        x = torch.cat([e, d], dim=1)
        x = self.merge_conv(x)
        x = self.refine_dw(x)
        return x

class AddFFM(nn.Module):
    def __init__(self, in_ch_enc, in_ch_dec, out_ch):
        super().__init__()
        self.enc_proj = conv1x1(in_ch_enc, out_ch, bias=False)
        self.dec_proj = conv1x1(in_ch_dec, out_ch, bias=False)
        self.refine_dw = nn.Sequential(
            depthwise_conv3x3(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        e = self.enc_proj(enc_feat)
        d = self.dec_proj(dec_feat)
        x = e + d  # elementwise add fusion
        x = self.refine_dw(x)
        return x

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x)
        return x * w

class SEFusionFFM(nn.Module):
    def __init__(self, in_ch_enc, in_ch_dec, out_ch, reduction=4):
        super().__init__()
        # Project both inputs to out_ch
        self.enc_proj = conv1x1(in_ch_enc, out_ch, bias=False)
        self.dec_proj = conv1x1(in_ch_dec, out_ch, bias=False)

        # SE block for channel attention
        self.se = SEBlock(out_ch, reduction=reduction)

        # Depthwise refine
        self.refine_dw = nn.Sequential(
            depthwise_conv3x3(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, enc_feat, dec_feat):
        e = self.enc_proj(enc_feat)
        d = self.dec_proj(dec_feat)
        x = e + d  # sum fusion
        x = self.se(x)  # reweight channels
        x = self.refine_dw(x)
        return x


# -------------------------
# Test với input giả lập
# -------------------------
if __name__ == "__main__":
    B = 2
    enc_feat = torch.randn(B, 128, 20, 16)   # encoder feature (B,C,H,W)
    dec_feat = torch.randn(B, 128, 20, 16)   # decoder feature nhỏ hơn

    # ffm = FeatureFusionModule(in_ch_enc=128, in_ch_dec=128, out_ch=64)
    # ffm = FeatureFusionModule(in_ch_enc=128, in_ch_dec=128, out_ch=128)
    # ffm = FastFFM(in_ch_enc=128, in_ch_dec=128, out_ch=128)
    # ffm = AddFFM(in_ch_enc=128, in_ch_dec=128, out_ch=128)
    ffm = SEFusionFFM(in_ch_enc=128, in_ch_dec=128, out_ch=128)

    out = ffm(enc_feat, dec_feat)

    print("Encoder input :", enc_feat.shape)
    print("Decoder input :", dec_feat.shape)
    print("Output        :", out.shape)
