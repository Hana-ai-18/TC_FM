# """TCNM/Unet3D_merge_tiny.py - 3D U-Net for satellite images"""
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np


# class Conv3dBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
#         super().__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, k, s, p, bias=True),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(out_ch, out_ch, k, s, p, bias=True),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(True)
#         )
#         self.residual = nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False)
    
#     def forward(self, x):
#         return self.conv2(self.conv1(x)) + self.residual(x)


# class Down(nn.Module):
#     def __init__(self, in_ch, out_ch, k, s):
#         super().__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool3d(k, s),
#             Conv3dBlock(in_ch, out_ch)
#         )
    
#     def forward(self, x):
#         return self.maxpool_conv(x)


# class Up(nn.Module):
#     def __init__(self, x1_in, x2_in, out_ch, k, s, p):
#         super().__init__()
#         self.up = nn.Sequential(
#             nn.ConvTranspose3d(x1_in, x1_in, k, s, p, bias=True),
#             nn.ReLU(True)
#         )
#         self.conv = Conv3dBlock(x1_in + x2_in, out_ch)
    
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         diffT = x2.size()[2] - x1.size()[2]
#         diffH = x2.size()[3] - x1.size()[3]
#         diffW = x2.size()[4] - x1.size()[4]
        
#         x1 = F.pad(x1, [
#             diffW // 2, diffW - diffW // 2,
#             diffH // 2, diffH - diffH // 2,
#             diffT // 2, diffT - diffT // 2
#         ])
        
#         return self.conv(torch.cat([x2, x1], dim=1))


# class OutConv(nn.Module):
#     def __init__(self, in_ch_list, out_ch):
#         super().__init__()
#         self.up_list = nn.ModuleList()
        
#         for i, ch in enumerate(in_ch_list[:-1]):
#             scale = int(np.power(2, len(in_ch_list) - 1 - i))
#             self.up_list.append(nn.Sequential(
#                 nn.ConvTranspose3d(ch, ch, [1, scale, scale], [1, scale, scale]),
#                 nn.ReLU(True),
#                 nn.Conv3d(ch, in_ch_list[-1], 3, 1, 1),
#                 nn.BatchNorm3d(in_ch_list[-1]),
#                 nn.ReLU(True)
#             ))
        
#         self.final_conv = nn.Sequential(
#             nn.Conv3d(in_ch_list[-1], out_ch, 3, 1, 1),
#             nn.BatchNorm3d(out_ch),
#             nn.ReLU(True),
#             nn.AdaptiveAvgPool3d((1, 64, 64)),
#             nn.Conv3d(out_ch, out_ch, 1, 1, 0)
#         )
    
#     def forward(self, x_list):
#         x6, x7, x8, x9 = x_list
#         x6 = self.up_list[0](x6)
#         x7 = self.up_list[1](x7)
#         x8 = self.up_list[2](x8)
        
#         target_t = x9.size(2)
#         x6 = F.interpolate(x6, size=(target_t, 64, 64))
#         x7 = F.interpolate(x7, size=(target_t, 64, 64))
#         x8 = F.interpolate(x8, size=(target_t, 64, 64))
        
#         return self.final_conv(torch.cat([x6, x7, x8, x9], dim=2))


# class Unet3D(nn.Module):
#     def __init__(self, in_channel=1, out_channel=1):
#         super().__init__()
#         self.inc = Conv3dBlock(in_channel, 16)
#         self.down1 = Down(16, 32, [1, 2, 2], [1, 2, 2])
#         self.down2 = Down(32, 64, [1, 2, 2], [1, 2, 2])
#         self.down3 = Down(64, 128, [2, 2, 2], [2, 2, 2])
#         self.down4 = Down(128, 128, [2, 2, 2], [2, 2, 2])
        
#         self.up1 = Up(128, 128, 64, [2, 2, 2], [2, 2, 2], 0)
#         self.up2 = Up(64, 64, 32, [2, 2, 2], [2, 2, 2], 0)
#         self.up3 = Up(32, 32, 16, [1, 2, 2], [1, 2, 2], 0)
#         self.up4 = Up(16, 16, 16, [1, 2, 2], [1, 2, 2], 0)
        
#         self.outc = OutConv([64, 32, 16, 16], out_channel)
    
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
        
#         x6 = self.up1(x5, x4)
#         x7 = self.up2(x6, x3)
#         x8 = self.up3(x7, x2)
#         x9 = self.up4(x8, x1)
        
#         return self.outc([x6, x7, x8, x9])

"""
TCNM/Unet3D_merge_tiny.py  ── v9
===================================
3D U-Net for Data3d input.

Input tensor shape: [B, 1, T_obs, 81, 81]
  Channels (13 per timestep, stacked along C or T axis depending on loader):
    GPH  at 200, 500, 850, 925 hPa  (4)
    U    at 200, 500, 850, 925 hPa  (4)
    V    at 200, 500, 850, 925 hPa  (4)
    SST  (1 layer, surface)          (1)
  Total: 13 channels per timestep

The model treats the time dimension as depth (T) and spatial as H×W.
Output: [B, C_out, T_out, H_out, W_out] — used by VelocityField._context()

Two outputs:
  e_3d_En : encoder bottleneck features (for 1D-Data Encoder fusion)
  e_3d_De : decoder output (for De-LSTM future Data3d prediction)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

class Conv3dBlock(nn.Module):
    """Double Conv3d with residual skip."""

    def __init__(self, in_ch: int, out_ch: int, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int,
                 pool_k=(1, 2, 2), pool_s=(1, 2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(pool_k, pool_s),
            Conv3dBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    def __init__(self, x1_ch: int, x2_ch: int, out_ch: int,
                 up_k=(1, 2, 2), up_s=(1, 2, 2)):
        super().__init__()
        self.up   = nn.ConvTranspose3d(x1_ch, x1_ch, up_k, up_s, bias=False)
        self.conv = Conv3dBlock(x1_ch + x2_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad to match x2
        diff = [x2.size(i) - x1.size(i) for i in (2, 3, 4)]
        x1 = F.pad(x1, [
            diff[2] // 2, diff[2] - diff[2] // 2,
            diff[1] // 2, diff[1] - diff[1] // 2,
            diff[0] // 2, diff[0] - diff[0] // 2,
        ])
        return self.conv(torch.cat([x2, x1], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  Main Unet3D
# ══════════════════════════════════════════════════════════════════════════════

class Unet3D(nn.Module):
    """
    3D U-Net for spatiotemporal TC satellite data.

    Input  : [B, in_channel, T, 81, 81]
    Output : [B, out_channel, T_out, H_out, W_out]

    in_channel = 13 (GPH×4 + U×4 + V×4 + SST×1) when all channels stacked,
                 or 1 when a single field is used.
    """

    def __init__(self, in_channel: int = 1, out_channel: int = 1):
        super().__init__()

        # Encoder
        self.inc    = Conv3dBlock(in_channel, 16)
        self.down1  = Down(16,  32,  pool_k=(1, 2, 2), pool_s=(1, 2, 2))
        self.down2  = Down(32,  64,  pool_k=(1, 2, 2), pool_s=(1, 2, 2))
        self.down3  = Down(64,  128, pool_k=(2, 2, 2), pool_s=(2, 2, 2))
        self.down4  = Down(128, 128, pool_k=(2, 2, 2), pool_s=(2, 2, 2))

        # Decoder
        self.up1    = Up(128, 128, 64,  up_k=(2, 2, 2), up_s=(2, 2, 2))
        self.up2    = Up(64,   64, 32,  up_k=(2, 2, 2), up_s=(2, 2, 2))
        self.up3    = Up(32,   32, 16,  up_k=(1, 2, 2), up_s=(1, 2, 2))
        self.up4    = Up(16,   16, 16,  up_k=(1, 2, 2), up_s=(1, 2, 2))

        # Final conv → pool → output
        self.outc   = nn.Sequential(
            nn.Conv3d(16, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),   # [B, C, T, 1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, in_channel, T, H, W]
        Returns [B, out_channel, T, 1, 1]
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)

        return self.outc(x)   # [B, out_channel, T, 1, 1]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return both encoder bottleneck (e_3d_En) and decoder output (e_3d_De).

        e_3d_En : [B, 128, T//4, H//16, W//16]  — bottleneck for 1D fusion
        e_3d_De : [B, out_channel, T, 1, 1]     — decoder output for De-LSTM
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)          # bottleneck → e_3d_En

        xd = self.up1(x5, x4)
        xd = self.up2(xd, x3)
        xd = self.up3(xd, x2)
        xd = self.up4(xd, x1)
        e_3d_De = self.outc(xd)      # [B, out_ch, T, 1, 1]

        return x5, e_3d_De           # (e_3d_En, e_3d_De)