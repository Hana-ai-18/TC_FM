
"""
TCNM/Unet3D_merge_tiny.py  ── v9-fixed
=========================================
3D U-Net for Data3d input.

Input : [B, 13, T_obs, 81, 81]
  13 channels per timestep (already z-score normalised by TrajectoryDataset):
    GPH @200, 500, 850, 925 hPa   (4 channels)
    U   @200, 500, 850, 925 hPa   (4 channels)
    V   @200, 500, 850, 925 hPa   (4 channels)
    SST (surface)                  (1 channel)

Two outputs from encode():
  e_3d_En : [B, 128, T//4, H', W'] — bottleneck for 1D-Data Encoder fusion
  e_3d_De : [B, out_ch, T, 1, 1]  — decoder output (spatial summary)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
#  Building blocks
# ══════════════════════════════════════════════════════════════════════════════

class Conv3dBlock(nn.Module):
    """Double Conv3d with residual skip connection."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
        # 1×1×1 skip to match channel dimension
        self.skip = nn.Conv3d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.skip(x)


class Down(nn.Module):
    """MaxPool3d followed by a Conv3dBlock."""

    def __init__(
        self,
        in_ch:  int,
        out_ch: int,
        pool_k: tuple = (1, 2, 2),
        pool_s: tuple = (1, 2, 2),
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool3d(pool_k, pool_s),
            Conv3dBlock(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """Transposed conv upsampling + skip concat + Conv3dBlock."""

    def __init__(
        self,
        x1_ch: int,
        x2_ch: int,
        out_ch: int,
        up_k:  tuple = (1, 2, 2),
        up_s:  tuple = (1, 2, 2),
    ):
        super().__init__()
        self.up   = nn.ConvTranspose3d(x1_ch, x1_ch, up_k, up_s, bias=False)
        self.conv = Conv3dBlock(x1_ch + x2_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad x1 to match x2 spatial/temporal dims if they differ
        diff = [x2.size(i) - x1.size(i) for i in (2, 3, 4)]
        x1 = F.pad(x1, [
            diff[2] // 2, diff[2] - diff[2] // 2,
            diff[1] // 2, diff[1] - diff[1] // 2,
            diff[0] // 2, diff[0] - diff[0] // 2,
        ])
        return self.conv(torch.cat([x2, x1], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
#  Unet3D
# ══════════════════════════════════════════════════════════════════════════════

class Unet3D(nn.Module):
    """
    3D U-Net for spatiotemporal TC satellite data.

    Input  : [B, in_channel, T, H, W]   (in_channel=13 for full Data3d)
    Output : [B, out_channel, T, 1, 1]  (via AdaptiveAvgPool3d)

    encode() additionally returns the bottleneck feature map for fusion
    with the 1D-Data Encoder (Eq. 8 in paper).
    """

    def __init__(self, in_channel: int = 13, out_channel: int = 1):
        super().__init__()

        # Encoder
        self.inc   = Conv3dBlock(in_channel, 16)
        self.down1 = Down(16,  32,  pool_k=(1, 2, 2), pool_s=(1, 2, 2))
        self.down2 = Down(32,  64,  pool_k=(1, 2, 2), pool_s=(1, 2, 2))
        self.down3 = Down(64,  128, pool_k=(2, 2, 2), pool_s=(2, 2, 2))
        self.down4 = Down(128, 128, pool_k=(2, 2, 2), pool_s=(2, 2, 2))

        # Decoder
        self.up1 = Up(128, 128, 64,  up_k=(2, 2, 2), up_s=(2, 2, 2))
        self.up2 = Up(64,   64, 32,  up_k=(2, 2, 2), up_s=(2, 2, 2))
        self.up3 = Up(32,   32, 16,  up_k=(1, 2, 2), up_s=(1, 2, 2))
        self.up4 = Up(16,   16, 16,  up_k=(1, 2, 2), up_s=(1, 2, 2))

        # Final output: project to out_channel, then pool spatial dims to 1×1
        self.outc = nn.Sequential(
            nn.Conv3d(16, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((None, 1, 1)),  # [B, out_ch, T, 1, 1]
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

        return self.outc(x)  # [B, out_ch, T, 1, 1]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full encoder+decoder pass returning both bottleneck and decoder output.

        Parameters
        ----------
        x : [B, in_channel, T, H, W]

        Returns
        -------
        e_3d_En : [B, 128, T//4, H//16, W//16]
            Bottleneck features for fusion with 1D-Data Encoder (Eq. 8).
        e_3d_De : [B, out_channel, T, 1, 1]
            Decoder output (spatial summary, used for De-LSTM placeholder).
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)   # ← bottleneck = e_3d_En

        xd = self.up1(x5, x4)
        xd = self.up2(xd, x3)
        xd = self.up3(xd, x2)
        xd = self.up4(xd, x1)
        e_3d_De = self.outc(xd)  # [B, out_ch, T, 1, 1]

        return x5, e_3d_De