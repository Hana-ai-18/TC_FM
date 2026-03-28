"""
TCNM/env_net_transformer_gphsplit.py
=====================================
Environmental feature Transformer — fixed to match ACTUAL env_data structure.

Feature vector (90 dims total when flattened):
── Data1d (84 dims) ─────────────────────────────────────────────────────────
  wind                  float (1)    /110
  intensity_class       (6,)         0=TD..5=SuperTY  one-hot
  move_velocity         float (1)    /1219.84
  month                 (12,)        one-hot
  location_lon_scs      (10,)        2.5°/bin [100–125°E]  one-hot
  location_lat_scs      (8,)         2.5°/bin [5–25°N]     one-hot
  bearing_to_scs_center (16,)        16 compass directions 22.5° step, one-hot
  dist_to_scs_boundary  (5,)         outside/very_far/far/mid/near  one-hot
  delta_velocity        (5,)         decrease_strong → increase_strong  one-hot
  history_direction12   (8,)  |−1    8-direction one-hot, pad −1 if missing
  history_direction24   (8,)  |−1    same
  history_inte_change24 (4,)  |−1    intensity change 24h, pad −1 if missing
── Data3d / 500 hPa GPH+UV (6 dims) ─────────────────────────────────────────
  gph500_mean           float (1)    z-score (μ=5900, σ=200 m)
  gph500_center         float (1)    z-score
  u500_mean             float (1)    /30 m/s, clip[−1,1]
  u500_center           float (1)    /30 m/s, clip[−1,1]
  v500_mean             float (1)    /30 m/s, clip[−1,1]
  v500_center           float (1)    /30 m/s, clip[−1,1]

ENV_LSTM architecture:
  Per-timestep feature fusion → LSTM(input=90, hidden=128, layers=2) → d_model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Feature spec (key → expected dim) ────────────────────────────────────────
ENV_FEATURE_DIMS: dict[str, int] = {
    "wind":                   1,
    "intensity_class":        6,
    "move_velocity":          1,
    "month":                 12,
    "location_lon_scs":      10,
    "location_lat_scs":       8,
    "bearing_to_scs_center": 16,
    "dist_to_scs_boundary":   5,
    "delta_velocity":         5,
    "history_direction12":    8,
    "history_direction24":    8,
    "history_inte_change24":  4,
    # Data3d scalars
    "gph500_mean":            1,
    "gph500_center":          1,
    "u500_mean":              1,
    "u500_center":            1,
    "v500_mean":              1,
    "v500_center":            1,
}

ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: build feature vector [B, T, ENV_DIM_TOTAL] from env_data dict
# ══════════════════════════════════════════════════════════════════════════════

def build_env_vector(
    env_data: dict | None,
    B: int,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Assemble the 90-dim feature vector from an env_data dict.

    Any missing key → zero-filled slot of the correct dimension.
    Handles shapes [B,T,d], [B,d], [B,T], scalar, 1-D tensor.

    Returns
    -------
    tensor : [B, T, 90]
    """
    parts: list[torch.Tensor] = []

    for key, dim in ENV_FEATURE_DIMS.items():
        slot = torch.zeros(B, T, dim, device=device)

        if env_data is None or not isinstance(env_data, dict) or key not in env_data:
            parts.append(slot)
            continue

        v = env_data[key]
        if v is None:
            parts.append(slot)
            continue

        if not torch.is_tensor(v):
            try:
                v = torch.tensor(v, dtype=torch.float, device=device)
            except Exception:
                parts.append(slot)
                continue
        v = v.float().to(device)

        # ── Normalise shape to [B, T, dim] ───────────────────────────────
        try:
            if v.dim() == 0:                        # scalar
                slot = v.expand(B, T, 1) if dim == 1 else slot
            elif v.dim() == 1:
                if v.numel() == dim:                # [dim] → broadcast
                    slot = v.unsqueeze(0).unsqueeze(0).expand(B, T, dim)
                elif v.numel() == B * T * dim:
                    slot = v.view(B, T, dim)
                # else leave zeros
            elif v.dim() == 2:
                if v.shape == (B, T):               # [B,T] scalar-per-step
                    slot = v.unsqueeze(-1).expand(-1, -1, dim)
                elif v.shape == (B, dim):           # [B,dim] per-sample
                    slot = v.unsqueeze(1).expand(-1, T, -1)
                elif v.shape[0] == T and v.shape[1] == dim:  # [T,dim]
                    slot = v.unsqueeze(0).expand(B, -1, -1)
                elif v.shape[0] == B:
                    slot = v.unsqueeze(1).expand(-1, T, -1)[..., :dim]
                    if slot.shape[-1] < dim:
                        slot = F.pad(slot, (0, dim - slot.shape[-1]))
            elif v.dim() == 3:
                if v.shape[:2] == (B, T):           # [B,T,?]
                    d_in = v.shape[-1]
                    if d_in == dim:
                        slot = v
                    elif d_in < dim:
                        slot = F.pad(v, (0, dim - d_in))
                    else:
                        slot = v[..., :dim]
                elif v.shape[0] == T:               # [T,B,dim]
                    slot = v.permute(1, 0, 2)[..., :dim]
                    if slot.shape[-1] < dim:
                        slot = F.pad(slot, (0, dim - slot.shape[-1]))
        except Exception:
            pass  # keep zeros

        parts.append(slot.float())

    return torch.cat(parts, dim=-1)   # [B, T, 90]


# ══════════════════════════════════════════════════════════════════════════════
#  ENV-LSTM model
# ══════════════════════════════════════════════════════════════════════════════

class Env_net(nn.Module):
    """
    ENV-LSTM encoder that processes the 90-dim environmental feature vector
    (Data1d + Data3d scalars) via a 2-layer LSTM and outputs a context vector.

    Architecture
    ------------
    input (per timestep) : [B, T, 90]
    LSTM                 : input=90 → hidden=128, 2 layers, dropout=0.2
    projection           : linear(128 → d_model)
    output               : ([B, d_model],  0,  0)   ← 0s for API compat

    The UNet3D spatial encoder is called externally in flow_matching_model.py
    (VelocityField._context) — Env_net only handles the 1-D env features.
    """

    def __init__(
        self,
        obs_len:   int = 8,
        embed_dim: int = 16,     # kept for API compat, not used internally
        d_model:   int = 64,
    ):
        super().__init__()
        self.obs_len   = obs_len
        self.d_model   = d_model
        self.input_dim = ENV_DIM_TOTAL  # 90

        # Lightweight projection before LSTM (optional but helps convergence)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )

        self.lstm = nn.LSTM(
            input_size  = 128,
            hidden_size = 128,
            num_layers  = 2,
            batch_first = True,
            dropout     = 0.2,
        )

        self.out_proj = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(
        self,
        env_data: dict | None,
        gph:      torch.Tensor,   # [B,1,T,H,W] — used only for B, T, device
    ) -> tuple[torch.Tensor, int, int]:
        """
        Args
        ----
        env_data : dict of env feature tensors (see ENV_FEATURE_DIMS) or None
        gph      : [B, C, T, H, W]  spatial image — provides B, T, device

        Returns
        -------
        (context [B, d_model],  0,  0)
        """
        # normalise gph shape
        if gph.dim() == 4:
            gph = gph.unsqueeze(1)
        B, C, T, H, W = gph.shape
        device = gph.device

        feat = build_env_vector(env_data, B, T, device)    # [B, T, 90]
        x    = self.input_proj(feat)                       # [B, T, 128]
        _, (h_n, _) = self.lstm(x)                        # h_n [2, B, 128]
        ctx  = self.out_proj(h_n[-1])                     # [B, d_model]
        return ctx, 0, 0


# ══════════════════════════════════════════════════════════════════════════════
#  Stand-alone feature engineering helpers
#  (called from TrajectoryDataset / data pre-processing)
# ══════════════════════════════════════════════════════════════════════════════

SCS_BBOX = dict(lon_min=100.0, lon_max=125.0,
                lat_min=5.0,   lat_max=20.0)
SCS_CENTER = (112.5, 15.0)           # approximate SCS centre (lon, lat)
SCS_DIAGONAL_KM = 3100.0             # diagonal length used for normalisation

COMPASS_ANGLES = [i * 22.5 for i in range(16)]  # 0°=N, 22.5°=NNE, … 337.5°=NNW

BOUNDARY_THRESHOLDS = [
    0.05,   # near:       0–5%   of diagonal (~0–155 km)
    0.15,   # mid:        5–15%
    0.30,   # far:       15–30%
]
# classes: [outside(neg), very_far(≥30%), far(15-30%), mid(5-15%), near(0-5%)]
N_BOUNDARY_CLASSES = 5

DELTA_VEL_BINS = [-20, -5, 5, 20]   # km/h thresholds for 5 classes


import math


def _haversine_deg(lon1, lat1, lon2, lat2) -> float:
    """Great-circle distance in km between two (lon,lat) degree points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
    """
    16-class one-hot encoding of compass bearing from (lon,lat) to SCS centre.

    Compass rose (22.5° bins):
      idx 0  = N    (337.5–22.5°)
      idx 1  = NNE  (22.5–45°)
      …
      idx 15 = NNW  (315–337.5°)

    Returns list of 16 ints (one-hot).
    """
    c_lon, c_lat = SCS_CENTER
    dy = c_lat - lat_deg
    dx = (c_lon - lon_deg) * math.cos(math.radians((lat_deg + c_lat) / 2))
    bearing = math.degrees(math.atan2(dx, dy)) % 360   # 0=N, clockwise
    idx = int((bearing + 11.25) / 22.5) % 16
    v = [0] * 16
    v[idx] = 1
    return v


def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
    """
    5-class one-hot for distance to SCS boundary rectangle
    [100–125°E, 5–20°N].

    Distance to boundary = min distance to any of the 4 rectangle edges.
    Normalised by SCS diagonal (3100 km).

    Classes (bit string 00001 = near):
      [0] outside  (distance < 0, i.e. storm is outside the box)
      [1] very_far (≥ 30% diagonal, ~930 km from boundary)
      [2] far      (15–30%)
      [3] mid      (5–15%)
      [4] near     (0–5%, < ~155 km)

    Returns list of 5 ints (one-hot).
    """
    lon_min, lon_max = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
    lat_min, lat_max = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]

    # Check inside/outside
    inside = (lon_min <= lon_deg <= lon_max) and (lat_min <= lat_deg <= lat_max)

    if not inside:
        return [1, 0, 0, 0, 0]   # class 0: outside

    # Min distance to each of the 4 edges (rough, in km)
    def edge_km(delta_deg):
        return delta_deg * 111.0   # 1° ≈ 111 km

    d_west  = edge_km(lon_deg  - lon_min)
    d_east  = edge_km(lon_max  - lon_deg)
    d_south = edge_km(lat_deg  - lat_min)
    d_north = edge_km(lat_max  - lat_deg)
    d_min   = min(d_west, d_east, d_south, d_north)

    ratio = d_min / SCS_DIAGONAL_KM

    if ratio < BOUNDARY_THRESHOLDS[0]:   # < 5%
        idx = 4   # near
    elif ratio < BOUNDARY_THRESHOLDS[1]:  # 5–15%
        idx = 3   # mid
    elif ratio < BOUNDARY_THRESHOLDS[2]:  # 15–30%
        idx = 2   # far
    else:                                  # ≥ 30%
        idx = 1   # very_far

    v = [0] * 5
    v[idx] = 1
    return v


def delta_velocity_onehot(delta_km_h: float) -> list[int]:
    """
    5-class one-hot for TC translation speed change (km/h per 6h).

    Bins: ≤-20 | -20…-5 | -5…5 | 5…20 | >20
    Returns list of 5 ints.
    """
    if delta_km_h <= DELTA_VEL_BINS[0]:
        idx = 0
    elif delta_km_h <= DELTA_VEL_BINS[1]:
        idx = 1
    elif delta_km_h <= DELTA_VEL_BINS[2]:
        idx = 2
    elif delta_km_h <= DELTA_VEL_BINS[3]:
        idx = 3
    else:
        idx = 4
    v = [0] * 5
    v[idx] = 1
    return v


def intensity_class_onehot(wind_kt: float) -> list[int]:
    """
    6-class one-hot TC intensity (Beaufort/Saffir-Simpson hybrid, WMO):
      0=TD (<34kt), 1=TS (34–48), 2=TY (48–64),
      3=SevTY (64–84), 4=ViSevTY (84–115), 5=SuperTY (>115)
    """
    thresholds = [34, 48, 64, 84, 115]
    idx = sum(wind_kt >= t for t in thresholds)
    v = [0] * 6
    v[min(idx, 5)] = 1
    return v