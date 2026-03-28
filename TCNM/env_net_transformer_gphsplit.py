# """
# TCNM/env_net_transformer_gphsplit.py
# =====================================
# Environmental feature Transformer — fixed to match ACTUAL env_data structure.

# Feature vector (90 dims total when flattened):
# ── Data1d (84 dims) ─────────────────────────────────────────────────────────
#   wind                  float (1)    /110
#   intensity_class       (6,)         0=TD..5=SuperTY  one-hot
#   move_velocity         float (1)    /1219.84
#   month                 (12,)        one-hot
#   location_lon_scs      (10,)        2.5°/bin [100–125°E]  one-hot
#   location_lat_scs      (8,)         2.5°/bin [5–25°N]     one-hot
#   bearing_to_scs_center (16,)        16 compass directions 22.5° step, one-hot
#   dist_to_scs_boundary  (5,)         outside/very_far/far/mid/near  one-hot
#   delta_velocity        (5,)         decrease_strong → increase_strong  one-hot
#   history_direction12   (8,)  |−1    8-direction one-hot, pad −1 if missing
#   history_direction24   (8,)  |−1    same
#   history_inte_change24 (4,)  |−1    intensity change 24h, pad −1 if missing
# ── Data3d / 500 hPa GPH+UV (6 dims) ─────────────────────────────────────────
#   gph500_mean           float (1)    z-score (μ=5900, σ=200 m)
#   gph500_center         float (1)    z-score
#   u500_mean             float (1)    /30 m/s, clip[−1,1]
#   u500_center           float (1)    /30 m/s, clip[−1,1]
#   v500_mean             float (1)    /30 m/s, clip[−1,1]
#   v500_center           float (1)    /30 m/s, clip[−1,1]

# ENV_LSTM architecture:
#   Per-timestep feature fusion → LSTM(input=90, hidden=128, layers=2) → d_model
# """

# from __future__ import annotations

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ── Feature spec (key → expected dim) ────────────────────────────────────────
# ENV_FEATURE_DIMS: dict[str, int] = {
#     "wind":                   1,
#     "intensity_class":        6,
#     "move_velocity":          1,
#     "month":                 12,
#     "location_lon_scs":      10,
#     "location_lat_scs":       8,
#     "bearing_to_scs_center": 16,
#     "dist_to_scs_boundary":   5,
#     "delta_velocity":         5,
#     "history_direction12":    8,
#     "history_direction24":    8,
#     "history_inte_change24":  4,
#     # Data3d scalars
#     "gph500_mean":            1,
#     "gph500_center":          1,
#     "u500_mean":              1,
#     "u500_center":            1,
#     "v500_mean":              1,
#     "v500_center":            1,
# }

# ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90


# # ══════════════════════════════════════════════════════════════════════════════
# #  Helper: build feature vector [B, T, ENV_DIM_TOTAL] from env_data dict
# # ══════════════════════════════════════════════════════════════════════════════

# def build_env_vector(
#     env_data: dict | None,
#     B: int,
#     T: int,
#     device: torch.device,
# ) -> torch.Tensor:
#     """
#     Assemble the 90-dim feature vector from an env_data dict.

#     Any missing key → zero-filled slot of the correct dimension.
#     Handles shapes [B,T,d], [B,d], [B,T], scalar, 1-D tensor.

#     Returns
#     -------
#     tensor : [B, T, 90]
#     """
#     parts: list[torch.Tensor] = []

#     for key, dim in ENV_FEATURE_DIMS.items():
#         slot = torch.zeros(B, T, dim, device=device)

#         if env_data is None or not isinstance(env_data, dict) or key not in env_data:
#             parts.append(slot)
#             continue

#         v = env_data[key]
#         if v is None:
#             parts.append(slot)
#             continue

#         if not torch.is_tensor(v):
#             try:
#                 v = torch.tensor(v, dtype=torch.float, device=device)
#             except Exception:
#                 parts.append(slot)
#                 continue
#         v = v.float().to(device)

#         # ── Normalise shape to [B, T, dim] ───────────────────────────────
#         try:
#             if v.dim() == 0:                        # scalar
#                 slot = v.expand(B, T, 1) if dim == 1 else slot
#             elif v.dim() == 1:
#                 if v.numel() == dim:                # [dim] → broadcast
#                     slot = v.unsqueeze(0).unsqueeze(0).expand(B, T, dim)
#                 elif v.numel() == B * T * dim:
#                     slot = v.view(B, T, dim)
#                 # else leave zeros
#             elif v.dim() == 2:
#                 if v.shape == (B, T):               # [B,T] scalar-per-step
#                     slot = v.unsqueeze(-1).expand(-1, -1, dim)
#                 elif v.shape == (B, dim):           # [B,dim] per-sample
#                     slot = v.unsqueeze(1).expand(-1, T, -1)
#                 elif v.shape[0] == T and v.shape[1] == dim:  # [T,dim]
#                     slot = v.unsqueeze(0).expand(B, -1, -1)
#                 elif v.shape[0] == B:
#                     slot = v.unsqueeze(1).expand(-1, T, -1)[..., :dim]
#                     if slot.shape[-1] < dim:
#                         slot = F.pad(slot, (0, dim - slot.shape[-1]))
#             elif v.dim() == 3:
#                 if v.shape[:2] == (B, T):           # [B,T,?]
#                     d_in = v.shape[-1]
#                     if d_in == dim:
#                         slot = v
#                     elif d_in < dim:
#                         slot = F.pad(v, (0, dim - d_in))
#                     else:
#                         slot = v[..., :dim]
#                 elif v.shape[0] == T:               # [T,B,dim]
#                     slot = v.permute(1, 0, 2)[..., :dim]
#                     if slot.shape[-1] < dim:
#                         slot = F.pad(slot, (0, dim - slot.shape[-1]))
#         except Exception:
#             pass  # keep zeros

#         parts.append(slot.float())

#     return torch.cat(parts, dim=-1)   # [B, T, 90]


# # ══════════════════════════════════════════════════════════════════════════════
# #  ENV-LSTM model
# # ══════════════════════════════════════════════════════════════════════════════

# class Env_net(nn.Module):
#     """
#     ENV-LSTM encoder that processes the 90-dim environmental feature vector
#     (Data1d + Data3d scalars) via a 2-layer LSTM and outputs a context vector.

#     Architecture
#     ------------
#     input (per timestep) : [B, T, 90]
#     LSTM                 : input=90 → hidden=128, 2 layers, dropout=0.2
#     projection           : linear(128 → d_model)
#     output               : ([B, d_model],  0,  0)   ← 0s for API compat

#     The UNet3D spatial encoder is called externally in flow_matching_model.py
#     (VelocityField._context) — Env_net only handles the 1-D env features.
#     """

#     def __init__(
#         self,
#         obs_len:   int = 8,
#         embed_dim: int = 16,     # kept for API compat, not used internally
#         d_model:   int = 64,
#     ):
#         super().__init__()
#         self.obs_len   = obs_len
#         self.d_model   = d_model
#         self.input_dim = ENV_DIM_TOTAL  # 90

#         # Lightweight projection before LSTM (optional but helps convergence)
#         self.input_proj = nn.Sequential(
#             nn.Linear(self.input_dim, 128),
#             nn.LayerNorm(128),
#             nn.GELU(),
#         )

#         self.lstm = nn.LSTM(
#             input_size  = 128,
#             hidden_size = 128,
#             num_layers  = 2,
#             batch_first = True,
#             dropout     = 0.2,
#         )

#         self.out_proj = nn.Sequential(
#             nn.Linear(128, d_model),
#             nn.LayerNorm(d_model),
#         )

#     def forward(
#         self,
#         env_data: dict | None,
#         gph:      torch.Tensor,   # [B,1,T,H,W] — used only for B, T, device
#     ) -> tuple[torch.Tensor, int, int]:
#         """
#         Args
#         ----
#         env_data : dict of env feature tensors (see ENV_FEATURE_DIMS) or None
#         gph      : [B, C, T, H, W]  spatial image — provides B, T, device

#         Returns
#         -------
#         (context [B, d_model],  0,  0)
#         """
#         # normalise gph shape
#         if gph.dim() == 4:
#             gph = gph.unsqueeze(1)
#         B, C, T, H, W = gph.shape
#         device = gph.device

#         feat = build_env_vector(env_data, B, T, device)    # [B, T, 90]
#         x    = self.input_proj(feat)                       # [B, T, 128]
#         _, (h_n, _) = self.lstm(x)                        # h_n [2, B, 128]
#         ctx  = self.out_proj(h_n[-1])                     # [B, d_model]
#         return ctx, 0, 0


# # ══════════════════════════════════════════════════════════════════════════════
# #  Stand-alone feature engineering helpers
# #  (called from TrajectoryDataset / data pre-processing)
# # ══════════════════════════════════════════════════════════════════════════════

# SCS_BBOX = dict(lon_min=100.0, lon_max=125.0,
#                 lat_min=5.0,   lat_max=20.0)
# SCS_CENTER = (112.5, 15.0)           # approximate SCS centre (lon, lat)
# SCS_DIAGONAL_KM = 3100.0             # diagonal length used for normalisation

# COMPASS_ANGLES = [i * 22.5 for i in range(16)]  # 0°=N, 22.5°=NNE, … 337.5°=NNW

# BOUNDARY_THRESHOLDS = [
#     0.05,   # near:       0–5%   of diagonal (~0–155 km)
#     0.15,   # mid:        5–15%
#     0.30,   # far:       15–30%
# ]
# # classes: [outside(neg), very_far(≥30%), far(15-30%), mid(5-15%), near(0-5%)]
# N_BOUNDARY_CLASSES = 5

# DELTA_VEL_BINS = [-20, -5, 5, 20]   # km/h thresholds for 5 classes


# import math


# def _haversine_deg(lon1, lat1, lon2, lat2) -> float:
#     """Great-circle distance in km between two (lon,lat) degree points."""
#     R = 6371.0
#     dlat = math.radians(lat2 - lat1)
#     dlon = math.radians(lon2 - lon1)
#     a = (math.sin(dlat / 2) ** 2
#          + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
#          * math.sin(dlon / 2) ** 2)
#     return 2 * R * math.asin(min(1.0, math.sqrt(a)))


# def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
#     """
#     16-class one-hot encoding of compass bearing from (lon,lat) to SCS centre.

#     Compass rose (22.5° bins):
#       idx 0  = N    (337.5–22.5°)
#       idx 1  = NNE  (22.5–45°)
#       …
#       idx 15 = NNW  (315–337.5°)

#     Returns list of 16 ints (one-hot).
#     """
#     c_lon, c_lat = SCS_CENTER
#     dy = c_lat - lat_deg
#     dx = (c_lon - lon_deg) * math.cos(math.radians((lat_deg + c_lat) / 2))
#     bearing = math.degrees(math.atan2(dx, dy)) % 360   # 0=N, clockwise
#     idx = int((bearing + 11.25) / 22.5) % 16
#     v = [0] * 16
#     v[idx] = 1
#     return v


# def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
#     """
#     5-class one-hot for distance to SCS boundary rectangle
#     [100–125°E, 5–20°N].

#     Distance to boundary = min distance to any of the 4 rectangle edges.
#     Normalised by SCS diagonal (3100 km).

#     Classes (bit string 00001 = near):
#       [0] outside  (distance < 0, i.e. storm is outside the box)
#       [1] very_far (≥ 30% diagonal, ~930 km from boundary)
#       [2] far      (15–30%)
#       [3] mid      (5–15%)
#       [4] near     (0–5%, < ~155 km)

#     Returns list of 5 ints (one-hot).
#     """
#     lon_min, lon_max = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
#     lat_min, lat_max = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]

#     # Check inside/outside
#     inside = (lon_min <= lon_deg <= lon_max) and (lat_min <= lat_deg <= lat_max)

#     if not inside:
#         return [1, 0, 0, 0, 0]   # class 0: outside

#     # Min distance to each of the 4 edges (rough, in km)
#     def edge_km(delta_deg):
#         return delta_deg * 111.0   # 1° ≈ 111 km

#     d_west  = edge_km(lon_deg  - lon_min)
#     d_east  = edge_km(lon_max  - lon_deg)
#     d_south = edge_km(lat_deg  - lat_min)
#     d_north = edge_km(lat_max  - lat_deg)
#     d_min   = min(d_west, d_east, d_south, d_north)

#     ratio = d_min / SCS_DIAGONAL_KM

#     if ratio < BOUNDARY_THRESHOLDS[0]:   # < 5%
#         idx = 4   # near
#     elif ratio < BOUNDARY_THRESHOLDS[1]:  # 5–15%
#         idx = 3   # mid
#     elif ratio < BOUNDARY_THRESHOLDS[2]:  # 15–30%
#         idx = 2   # far
#     else:                                  # ≥ 30%
#         idx = 1   # very_far

#     v = [0] * 5
#     v[idx] = 1
#     return v


# def delta_velocity_onehot(delta_km_h: float) -> list[int]:
#     """
#     5-class one-hot for TC translation speed change (km/h per 6h).

#     Bins: ≤-20 | -20…-5 | -5…5 | 5…20 | >20
#     Returns list of 5 ints.
#     """
#     if delta_km_h <= DELTA_VEL_BINS[0]:
#         idx = 0
#     elif delta_km_h <= DELTA_VEL_BINS[1]:
#         idx = 1
#     elif delta_km_h <= DELTA_VEL_BINS[2]:
#         idx = 2
#     elif delta_km_h <= DELTA_VEL_BINS[3]:
#         idx = 3
#     else:
#         idx = 4
#     v = [0] * 5
#     v[idx] = 1
#     return v


# def intensity_class_onehot(wind_kt: float) -> list[int]:
#     """
#     6-class one-hot TC intensity (Beaufort/Saffir-Simpson hybrid, WMO):
#       0=TD (<34kt), 1=TS (34–48), 2=TY (48–64),
#       3=SevTY (64–84), 4=ViSevTY (84–115), 5=SuperTY (>115)
#     """
#     thresholds = [34, 48, 64, 84, 115]
#     idx = sum(wind_kt >= t for t in thresholds)
#     v = [0] * 6
#     v[min(idx, 5)] = 1
#     return v

"""
TCNM/env_net_transformer_gphsplit.py  ── v9
=============================================
Environmental feature Transformer — fixed to match ACTUAL env_data structure
seen in the uploaded env .npy files (image reference).

Feature vector (90 dims total when flattened):
── Data1d (84 dims) ─────────────────────────────────────────────────────────
  wind                  float (1)    normalised /110
  intensity_class       (6,)         0=TD..5=SuperTY  one-hot
  move_velocity         float (1)    /1219.84
  month                 (12,)        one-hot
  location_lon_scs      (10,)        2.5°/bin [100–125°E]  one-hot
  location_lat_scs      (8,)         2.5°/bin [5–25°N]     one-hot
  bearing_to_scs_center (16,)        16 compass dirs 22.5° step, one-hot
  dist_to_scs_boundary  (5,)         outside/very_far/far/mid/near  one-hot
  delta_velocity        (5,)         ≤-20 | -20…-5 | -5…5 | 5…20 | >20 km/h
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

Total: 1+6+1+12+10+8+16+5+5+8+8+4 + 6 = 90 dims

ENV_LSTM architecture:
  input_proj: Linear(90→128) + LayerNorm + GELU
  LSTM(input=128, hidden=128, layers=2, dropout=0.2)
  out_proj: Linear(128→d_model) + LayerNorm
"""

from __future__ import annotations

import math
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
    # Data3d scalars from 500 hPa fields
    "gph500_mean":            1,
    "gph500_center":          1,
    "u500_mean":              1,
    "u500_center":            1,
    "v500_mean":              1,
    "v500_center":            1,
}

ENV_DIM_TOTAL = sum(ENV_FEATURE_DIMS.values())   # 90


# ══════════════════════════════════════════════════════════════════════════════
#  SCS geography constants
# ══════════════════════════════════════════════════════════════════════════════

SCS_BBOX = dict(lon_min=100.0, lon_max=125.0, lat_min=5.0, lat_max=20.0)
SCS_CENTER = (112.5, 12.5)          # centre of [100–125°E, 5–20°N]
SCS_DIAGONAL_KM = 3100.0

# 16 compass directions, 22.5° apart, starting from N (0°)
# N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW
COMPASS_LABELS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]

# Distance-to-boundary thresholds (fraction of SCS diagonal)
BOUNDARY_THRESHOLDS = [0.05, 0.15, 0.30]   # near / mid / far / very_far

# Delta-velocity bins (km/h change per 6h)
DELTA_VEL_BINS = [-20.0, -5.0, 5.0, 20.0]


# ══════════════════════════════════════════════════════════════════════════════
#  Stand-alone feature engineering helpers  (used by dataset + data processing)
# ══════════════════════════════════════════════════════════════════════════════

def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def bearing_to_scs_center_onehot(lon_deg: float, lat_deg: float) -> list[int]:
    """
    16-class one-hot of compass bearing from (lon,lat) → SCS centre (112.5°E, 12.5°N).

    Bins are 22.5° wide:
      idx 0  = N    ([-11.25, 11.25)° — i.e. bearing 348.75–11.25°)
      idx 1  = NNE  (11.25–33.75°)
      …
      idx 15 = NNW  (326.25–348.75°)

    Returns list[int] length 16 (one-hot).
    """
    c_lon, c_lat = SCS_CENTER
    # latitude-corrected Δx, Δy
    mid_lat = math.radians((lat_deg + c_lat) / 2.0)
    dx = (c_lon - lon_deg) * math.cos(mid_lat)
    dy = c_lat - lat_deg
    bearing = math.degrees(math.atan2(dx, dy)) % 360.0   # 0=N, clockwise
    idx = int((bearing + 11.25) / 22.5) % 16
    v = [0] * 16
    v[idx] = 1
    return v


def dist_to_scs_boundary_onehot(lon_deg: float, lat_deg: float) -> list[int]:
    """
    5-class one-hot for minimum distance to SCS boundary rectangle
    [100–125°E, 5–20°N], normalised by SCS diagonal (3100 km).

    Classes:
      [0] outside  — storm is outside the rectangle (distance negative)
      [1] very_far — ≥ 30% diagonal (~930 km from nearest edge)
      [2] far      — 15–30%  (~465–930 km)
      [3] mid      — 5–15%   (~155–465 km)
      [4] near     — 0–5%    (< ~155 km)  →  bit string 00001 = near

    Returns list[int] length 5 (one-hot).
    """
    lon_min, lon_max = SCS_BBOX["lon_min"], SCS_BBOX["lon_max"]
    lat_min, lat_max = SCS_BBOX["lat_min"], SCS_BBOX["lat_max"]

    inside = (lon_min <= lon_deg <= lon_max) and (lat_min <= lat_deg <= lat_max)
    if not inside:
        return [1, 0, 0, 0, 0]   # class 0: outside

    # Distance to each of the 4 edges (in km, rough equirectangular)
    d_west  = (lon_deg  - lon_min) * 111.0 * math.cos(math.radians(lat_deg))
    d_east  = (lon_max  - lon_deg) * 111.0 * math.cos(math.radians(lat_deg))
    d_south = (lat_deg  - lat_min) * 111.0
    d_north = (lat_max  - lat_deg) * 111.0
    d_min   = min(d_west, d_east, d_south, d_north)

    ratio = d_min / SCS_DIAGONAL_KM

    if ratio < BOUNDARY_THRESHOLDS[0]:    # < 5% → near
        idx = 4
    elif ratio < BOUNDARY_THRESHOLDS[1]:  # 5–15% → mid
        idx = 3
    elif ratio < BOUNDARY_THRESHOLDS[2]:  # 15–30% → far
        idx = 2
    else:                                  # ≥ 30% → very_far
        idx = 1

    v = [0] * 5
    v[idx] = 1
    return v


def delta_velocity_onehot(delta_km_h: float) -> list[int]:
    """
    5-class one-hot for TC translation speed change (km/h per 6-h step).

    Bins: ≤ -20 | -20…-5 | -5…+5 | +5…+20 | > +20
    """
    thresholds = DELTA_VEL_BINS   # [-20, -5, 5, 20]
    if delta_km_h <= thresholds[0]:
        idx = 0
    elif delta_km_h <= thresholds[1]:
        idx = 1
    elif delta_km_h <= thresholds[2]:
        idx = 2
    elif delta_km_h <= thresholds[3]:
        idx = 3
    else:
        idx = 4
    v = [0] * 5
    v[idx] = 1
    return v


def intensity_class_onehot(wind_kt: float) -> list[int]:
    """
    6-class one-hot TC intensity (WMO / JTWC hybrid):
      0 = TD        (< 34 kt)
      1 = TS        (34–47 kt)
      2 = TY        (48–63 kt)
      3 = Sev TY    (64–83 kt)
      4 = ViSev TY  (84–114 kt)
      5 = Super TY  (≥ 115 kt)
    """
    thresholds = [34, 48, 64, 84, 115]
    idx = sum(wind_kt >= t for t in thresholds)
    v = [0] * 6
    v[min(idx, 5)] = 1
    return v


def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
    """One-hot bin encoding for a scalar in [lo, hi] into n equal bins."""
    idx = int((val - lo) / (hi - lo) * n)
    idx = max(0, min(n - 1, idx))
    v = [0] * n
    v[idx] = 1
    return v


# ══════════════════════════════════════════════════════════════════════════════
#  Build full env feature dict for one timestep
# ══════════════════════════════════════════════════════════════════════════════

def build_env_features_one_step(
    lon_norm: float,
    lat_norm: float,
    wind_norm: float,
    timestamp: str,
    env_npy: dict | None,
    prev_speed_kmh: float | None,
) -> dict:
    """
    Build the complete 90-dim feature dictionary for one observation step.

    Parameters
    ----------
    lon_norm, lat_norm : TCND normalised coordinates
    wind_norm          : normalised wind speed
    timestamp          : e.g. '2019073106'
    env_npy            : raw dict loaded from .npy file (or None)
    prev_speed_kmh     : TC translation speed at the previous step (km/h)
    """
    # Denorm to physical units
    lon_01  = lon_norm * 50.0 + 1800.0
    lat_01  = lat_norm * 50.0
    lon_deg = lon_01 / 10.0
    lat_deg = lat_01 / 10.0
    wind_kt = wind_norm * 25.0 + 40.0

    feat: dict = {}

    # ── 1. wind scalar ───────────────────────────────────────────────────
    feat["wind"] = [wind_kt / 110.0]

    # ── 2. intensity class ───────────────────────────────────────────────
    feat["intensity_class"] = intensity_class_onehot(wind_kt)

    # ── 3. move_velocity ─────────────────────────────────────────────────
    mv = 0.0
    if isinstance(env_npy, dict):
        v = env_npy.get("move_velocity", 0.0)
        mv = 0.0 if (v is None or v == -1) else float(v)
    feat["move_velocity"] = [mv / 1219.84]

    # ── 4. month one-hot ─────────────────────────────────────────────────
    try:
        month_idx = int(timestamp[4:6]) - 1
    except Exception:
        month_idx = 0
    month_oh = [0] * 12
    month_oh[max(0, min(11, month_idx))] = 1
    feat["month"] = month_oh

    # ── 5. location lon/lat bins ─────────────────────────────────────────
    feat["location_lon_scs"] = _pos_onehot(lon_deg, 100.0, 125.0, 10)
    feat["location_lat_scs"] = _pos_onehot(lat_deg,   5.0,  25.0,  8)

    # ── 6. bearing to SCS centre (NEW) ───────────────────────────────────
    feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)

    # ── 7. distance to SCS boundary (NEW) ────────────────────────────────
    feat["dist_to_scs_boundary"] = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

    # ── 8. delta_velocity (NEW) ──────────────────────────────────────────
    cur_speed = mv   # km/h
    delta = (cur_speed - prev_speed_kmh) if prev_speed_kmh is not None else 0.0
    feat["delta_velocity"] = delta_velocity_onehot(delta)

    # ── 9. history directions ─────────────────────────────────────────────
    for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
        if isinstance(env_npy, dict) and key in env_npy:
            v = env_npy[key]
            if isinstance(v, (list,)) or hasattr(v, "__iter__"):
                v = list(v)[:dim]
                v = v + [0] * (dim - len(v))
                if all(x == -1 for x in v):
                    v = [-1] * dim
            else:
                v = [-1] * dim
        else:
            v = [-1] * dim
        feat[key] = v

    # ── 10. history intensity change ──────────────────────────────────────
    key = "history_inte_change24"
    if isinstance(env_npy, dict) and key in env_npy:
        v = env_npy[key]
        if isinstance(v, (list,)) or hasattr(v, "__iter__"):
            v = list(v)[:4]
            v = v + [0] * (4 - len(v))
            if all(x == -1 for x in v):
                v = [-1] * 4
        else:
            v = [-1] * 4
    else:
        v = [-1] * 4
    feat["history_inte_change24"] = v

    # ── 11. Data3d scalars from env_npy ──────────────────────────────────
    d3d_keys = [
        ("gph500_mean",   5900.0, 200.0,  False),
        ("gph500_center", 5900.0, 200.0,  False),
        ("u500_mean",     0.0,    30.0,   True),
        ("u500_center",   0.0,    30.0,   True),
        ("v500_mean",     0.0,    30.0,   True),
        ("v500_center",   0.0,    30.0,   True),
    ]
    for k, mu, sigma, clip in d3d_keys:
        if isinstance(env_npy, dict) and k in env_npy:
            v = float(env_npy[k])
            if v == -1:
                v = 0.0
            v = (v - mu) / (sigma + 1e-8)
            if clip:
                v = max(-1.0, min(1.0, v))
        else:
            v = 0.0
        feat[k] = [v]

    return feat


def feat_to_tensor(feat: dict) -> "torch.Tensor":
    """Convert feature dict → flat float tensor [90]."""
    import torch
    import torch.nn.functional as F
    parts = []
    for key in ENV_FEATURE_DIMS:
        dim = ENV_FEATURE_DIMS[key]
        v = feat.get(key, None)
        if v is None:
            parts.append(torch.zeros(dim))
        else:
            if not isinstance(v, (list, torch.Tensor)):
                v = [float(v)]
            t = torch.tensor(v, dtype=torch.float)
            if t.numel() < dim:
                t = F.pad(t, (0, dim - t.numel()))
            elif t.numel() > dim:
                t = t[:dim]
            parts.append(t)
    return torch.cat(parts)   # [90]


# ══════════════════════════════════════════════════════════════════════════════
#  Helper: build feature vector [B, T, 90] from env_data dict
# ══════════════════════════════════════════════════════════════════════════════

def build_env_vector(
    env_data: dict | None,
    B: int,
    T: int,
    device: "torch.device",
) -> "torch.Tensor":
    """
    Assemble the 90-dim feature vector from an env_data dict.
    Handles shapes [B,T,d], [B,d], [T,d], scalar, etc.
    Returns [B, T, 90].
    """
    parts: list = []

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

        try:
            if v.dim() == 0:
                slot = v.expand(B, T, 1) if dim == 1 else slot
            elif v.dim() == 1:
                if v.numel() == dim:
                    slot = v.view(1, 1, dim).expand(B, T, dim)
                elif v.numel() == B * T * dim:
                    slot = v.view(B, T, dim)
            elif v.dim() == 2:
                if v.shape == (B, T):
                    slot = v.unsqueeze(-1).expand(-1, -1, dim)
                elif v.shape == (B, dim):
                    slot = v.unsqueeze(1).expand(-1, T, -1)
                elif v.shape[0] == T and v.shape[1] == dim:
                    slot = v.unsqueeze(0).expand(B, -1, -1)
                elif v.shape[0] == B:
                    d_in = v.shape[-1]
                    slot = v.unsqueeze(1).expand(-1, T, -1)[..., :dim]
                    if slot.shape[-1] < dim:
                        slot = F.pad(slot, (0, dim - slot.shape[-1]))
            elif v.dim() == 3:
                if v.shape[:2] == (B, T):
                    d_in = v.shape[-1]
                    if d_in == dim:
                        slot = v
                    elif d_in < dim:
                        slot = F.pad(v, (0, dim - d_in))
                    else:
                        slot = v[..., :dim]
                elif v.shape[0] == T:
                    slot = v.permute(1, 0, 2)[..., :dim]
                    if slot.shape[-1] < dim:
                        slot = F.pad(slot, (0, dim - slot.shape[-1]))
        except Exception:
            pass

        parts.append(slot.float())

    return torch.cat(parts, dim=-1)   # [B, T, 90]


# ══════════════════════════════════════════════════════════════════════════════
#  ENV-LSTM model
# ══════════════════════════════════════════════════════════════════════════════

class Env_net(nn.Module):
    """
    ENV-LSTM: processes 90-dim environmental feature vector via a 2-layer LSTM
    and outputs a context vector of dimension d_model.

    Architecture
    ------------
    input_proj : Linear(90→128) + LayerNorm + GELU
    LSTM       : 128 → hidden=128, 2 layers, dropout=0.2
    out_proj   : Linear(128→d_model) + LayerNorm
    output     : ([B, d_model], 0, 0)   ← 0s for API compat
    """

    def __init__(self, obs_len: int = 8, embed_dim: int = 16, d_model: int = 64):
        super().__init__()
        self.obs_len   = obs_len
        self.d_model   = d_model
        self.input_dim = ENV_DIM_TOTAL  # 90

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
        gph: "torch.Tensor",   # [B, C, T, H, W] — provides B, T, device
    ) -> tuple:
        if gph.dim() == 4:
            gph = gph.unsqueeze(1)
        B, C, T, H, W = gph.shape
        device = gph.device

        feat = build_env_vector(env_data, B, T, device)   # [B, T, 90]
        x    = self.input_proj(feat)                      # [B, T, 128]
        _, (h_n, _) = self.lstm(x)
        ctx  = self.out_proj(h_n[-1])                     # [B, d_model]
        return ctx, 0, 0