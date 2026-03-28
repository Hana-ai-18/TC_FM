# """
# TCNM/data/trajectoriesWithMe_unet_training.py
# ==============================================
# TC trajectory dataset — fixed to build the FULL 90-dim env feature vector
# matching the actual .npy env data structure (see env_net_transformer_gphsplit).

# New features added per timestep:
#   bearing_to_scs_center  (16,) one-hot  — compass bearing to SCS centre
#   dist_to_scs_boundary   (5,)  one-hot  — distance class to SCS rectangle
#   delta_velocity         (5,)  one-hot  — speed change between consecutive steps

# Data layout in .txt files (TCND_VN):
#   col0  row_id (float, typically 1.0)
#   col1  ped_id (float, 1.0 for single TC)
#   col2  lon_norm  = (lon_01E - 1800) / 50
#   col3  lat_norm  = lat_01N / 50
#   col4  pres_norm = (pres_hPa - 960) / 50
#   col5  wind_norm = (wind_kt - 40) / 25
#   col-2 date   (e.g. 2019073106)
#   col-1 name   (e.g. WIPHA)
# """
# from __future__ import annotations

# import logging
# import math
# import os

# import cv2
# import netCDF4 as nc
# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset

# from TCNM.env_net_transformer_gphsplit import (
#     bearing_to_scs_center_onehot,
#     dist_to_scs_boundary_onehot,
#     delta_velocity_onehot,
#     intensity_class_onehot,
#     ENV_FEATURE_DIMS,
# )

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ── Physical conversion constants ────────────────────────────────────────────
# NORM_TO_01DEG_LON = 50.0    # lon_01E = norm * 50 + 1800
# NORM_OFFSET_LON   = 1800.0
# NORM_TO_01DEG_LAT = 50.0    # lat_01N = norm * 50
# _KNOT_TO_KMH      = 1.852
# _01DEG_TO_KM      = 111.0 / 10.0   # 1 unit of 0.1° ≈ 11.1 km


# # ══════════════════════════════════════════════════════════════════════════════
# #  env_data_processing — exported (used by inference dataset)
# # ══════════════════════════════════════════════════════════════════════════════

# def env_data_processing(env_dict: dict) -> dict:
#     """Replace sentinel −1 values with 0.0 and return cleaned dict."""
#     if not isinstance(env_dict, dict):
#         return {}
#     return {k: (0.0 if v == -1 else v) for k, v in env_dict.items()}


# # ══════════════════════════════════════════════════════════════════════════════
# #  Build full env feature dict for one timestep
# # ══════════════════════════════════════════════════════════════════════════════

# def _build_env_features(
#     lon_norm: float, lat_norm: float,
#     wind_norm: float,
#     timestamp: str,
#     env_npy: dict | None,
#     prev_speed_kmh: float | None,
# ) -> dict:
#     """
#     Build the complete 90-dim feature dictionary for one observation step.

#     Parameters
#     ----------
#     lon_norm, lat_norm : normalised coordinates (TCND convention)
#     wind_norm          : normalised wind speed
#     timestamp          : e.g. '2019073106'
#     env_npy            : raw dict loaded from .npy file (or None)
#     prev_speed_kmh     : TC translation speed at the previous step (km/h)

#     Returns
#     -------
#     dict with keys matching ENV_FEATURE_DIMS
#     """
#     # Denorm to physical units
#     lon_01  = lon_norm * NORM_TO_01DEG_LON + NORM_OFFSET_LON
#     lat_01  = lat_norm * NORM_TO_01DEG_LAT
#     lon_deg = lon_01 / 10.0
#     lat_deg = lat_01 / 10.0
#     wind_kt = wind_norm * 25.0 + 40.0

#     feat: dict = {}

#     # ── Data1d features ──────────────────────────────────────────────────────
#     feat["wind"]           = [wind_kt / 110.0]
#     feat["intensity_class"] = intensity_class_onehot(wind_kt)

#     # move_velocity: use env_npy if present, else zero
#     mv = 0.0
#     if isinstance(env_npy, dict) and "move_velocity" in env_npy:
#         mv = float(env_npy.get("move_velocity", 0.0))
#         if mv == -1:
#             mv = 0.0
#     feat["move_velocity"] = [mv / 1219.84]

#     # month one-hot from timestamp
#     try:
#         month_idx = int(timestamp[4:6]) - 1   # 0-indexed
#     except Exception:
#         month_idx = 0
#     month_oh = [0] * 12
#     month_oh[max(0, min(11, month_idx))] = 1
#     feat["month"] = month_oh

#     # location binning inside SCS
#     lon_bins = list(np.arange(100.0, 126.0, 2.5))   # 10 bins
#     lat_bins = list(np.arange(5.0, 26.0, 2.5))       # 8 bins
#     lon_oh = [0] * 10
#     lat_oh = [0] * 8
#     for i, b in enumerate(lon_bins[:-1]):
#         if lon_deg >= b:
#             lon_oh[i] = 1
#         else:
#             break
#     for i, b in enumerate(lat_bins[:-1]):
#         if lat_deg >= b:
#             lat_oh[i] = 1
#         else:
#             break
#     # Use proper one-hot (not cumulative)
#     lon_oh = _pos_onehot(lon_deg, 100.0, 125.0, 10)
#     lat_oh = _pos_onehot(lat_deg, 5.0,   25.0,  8)
#     feat["location_lon_scs"] = lon_oh
#     feat["location_lat_scs"] = lat_oh

#     feat["bearing_to_scs_center"] = bearing_to_scs_center_onehot(lon_deg, lat_deg)
#     feat["dist_to_scs_boundary"]  = dist_to_scs_boundary_onehot(lon_deg, lat_deg)

#     # delta_velocity
#     cur_speed = mv  # km/h
#     if prev_speed_kmh is not None:
#         delta = cur_speed - prev_speed_kmh
#     else:
#         delta = 0.0
#     feat["delta_velocity"] = delta_velocity_onehot(delta)

#     # history direction (from env_npy or pad with −1)
#     for key, dim in [("history_direction12", 8), ("history_direction24", 8)]:
#         if isinstance(env_npy, dict) and key in env_npy:
#             v = env_npy[key]
#             if isinstance(v, (list, np.ndarray)):
#                 v = list(v)[:dim]
#                 v = v + [0] * (dim - len(v))
#                 if all(x == -1 for x in v):
#                     v = [-1] * dim
#             else:
#                 v = [-1] * dim
#         else:
#             v = [-1] * dim
#         feat[key] = v

#     # history_inte_change24
#     key = "history_inte_change24"
#     if isinstance(env_npy, dict) and key in env_npy:
#         v = env_npy[key]
#         if isinstance(v, (list, np.ndarray)):
#             v = list(v)[:4]
#             v = v + [0] * (4 - len(v))
#             if all(x == -1 for x in v):
#                 v = [-1] * 4
#         else:
#             v = [-1] * 4
#     else:
#         v = [-1] * 4
#     feat["history_inte_change24"] = v

#     # ── Data3d scalars (from env_npy or zero) ────────────────────────────────
#     d3d_keys = ["gph500_mean", "gph500_center",
#                 "u500_mean", "u500_center",
#                 "v500_mean", "v500_center"]
#     for k in d3d_keys:
#         if isinstance(env_npy, dict) and k in env_npy:
#             v = float(env_npy[k])
#             if v == -1:
#                 v = 0.0
#         else:
#             v = 0.0
#         feat[k] = [v]

#     return feat


# def _pos_onehot(val: float, lo: float, hi: float, n: int) -> list[int]:
#     """One-hot bin encoding for a scalar in [lo, hi] into n equal bins."""
#     idx = int((val - lo) / (hi - lo) * n)
#     idx = max(0, min(n - 1, idx))
#     v = [0] * n
#     v[idx] = 1
#     return v


# def _feat_to_tensor(feat: dict) -> torch.Tensor:
#     """Convert feature dict to flat float tensor [90]."""
#     parts = []
#     for key in ENV_FEATURE_DIMS:
#         v = feat.get(key, None)
#         if v is None:
#             parts.append(torch.zeros(ENV_FEATURE_DIMS[key]))
#         else:
#             if not isinstance(v, (list, np.ndarray, torch.Tensor)):
#                 v = [float(v)]
#             t = torch.tensor(v, dtype=torch.float)
#             d = ENV_FEATURE_DIMS[key]
#             if t.numel() < d:
#                 t = F.pad(t, (0, d - t.numel()))
#             elif t.numel() > d:
#                 t = t[:d]
#             parts.append(t)
#     return torch.cat(parts)   # [90]


# # ══════════════════════════════════════════════════════════════════════════════
# #  seq_collate
# # ══════════════════════════════════════════════════════════════════════════════

# def seq_collate(data):
#     """
#     Collate a list of dataset items.

#     Each item: 16-element list (see TrajectoryDataset.__getitem__)
#     Key output indices:
#       0  obs_traj   [T_obs, B, 2]
#       1  pred_traj  [T_pred, B, 2]
#       7  obs_Me     [T_obs, B, 2]
#       8  pred_Me    [T_pred, B, 2]
#       11 img_obs    [B, 1, T_obs, 64, 64]  (UNet3D format)
#       12 img_pred   [B, 1, T_pred, 64, 64]
#       13 env_data   dict of tensors  (each [B, T, dim])
#       15 tyID       list
#     """
#     (obs_traj, pred_traj, obs_rel, pred_rel,
#      non_linear_ped, loss_mask,
#      obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
#      obs_date, pred_date,
#      img_obs, img_pred,
#      env_data_raw, tyID) = zip(*data)

#     def traj_TBC(lst):
#         cat = torch.cat(lst, dim=0)          # [total_ped, 2, T]
#         return cat.permute(2, 0, 1)          # [T, total_ped, 2]

#     obs_traj_out    = traj_TBC(obs_traj)
#     pred_traj_out   = traj_TBC(pred_traj)
#     obs_rel_out     = traj_TBC(obs_rel)
#     pred_rel_out    = traj_TBC(pred_rel)
#     obs_Me_out      = traj_TBC(obs_Me)
#     pred_Me_out     = traj_TBC(pred_Me)
#     obs_Me_rel_out  = traj_TBC(obs_Me_rel)
#     pred_Me_rel_out = traj_TBC(pred_Me_rel)

#     nlp_out = torch.tensor(
#         [v for sl in non_linear_ped for v in
#          (sl if hasattr(sl, "__iter__") else [sl])],
#         dtype=torch.float,
#     )
#     mask_out = torch.cat(list(loss_mask), dim=0).permute(1, 0)

#     counts = torch.tensor([t.shape[0] for t in obs_traj])
#     cum    = torch.cumsum(counts, dim=0)
#     starts = torch.cat([torch.tensor([0]), cum[:-1]])
#     seq_start_end = torch.stack([starts, cum], dim=1)

#     # Images: [B, 1, T_obs, 64, 64]
#     img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
#     img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

#     # env_data: merge per-key, each [B, T, dim]
#     B = len(env_data_raw)
#     env_out: dict | None = None
#     valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
#     if valid_envs:
#         env_out = {}
#         all_keys = set()
#         for d in valid_envs:
#             all_keys.update(d.keys())
#         for key in all_keys:
#             vals = []
#             for d in env_data_raw:
#                 if isinstance(d, dict) and key in d:
#                     v = d[key]
#                     if not torch.is_tensor(v):
#                         v = torch.tensor(v, dtype=torch.float)
#                     vals.append(v.float())
#                 else:
#                     # placeholder: use shape from first available
#                     ref = next((d[key] for d in valid_envs if key in d), None)
#                     if ref is not None:
#                         ref_t = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref
#                         vals.append(torch.zeros_like(ref_t.float()))
#                     else:
#                         vals.append(torch.zeros(1))
#             try:
#                 mx = max(v.numel() for v in vals)
#                 padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
#                 env_out[key] = torch.stack(padded, dim=0)
#             except Exception:
#                 pass

#     return (
#         obs_traj_out,       # 0
#         pred_traj_out,      # 1
#         obs_rel_out,        # 2
#         pred_rel_out,       # 3
#         nlp_out,            # 4
#         mask_out,           # 5
#         seq_start_end,      # 6
#         obs_Me_out,         # 7
#         pred_Me_out,        # 8
#         obs_Me_rel_out,     # 9
#         pred_Me_rel_out,    # 10
#         img_obs_out,        # 11  [B, 1, T_obs,  64, 64]
#         img_pred_out,       # 12  [B, 1, T_pred, 64, 64]
#         env_out,            # 13  dict
#         None,               # 14
#         list(tyID),         # 15
#     )


# # ══════════════════════════════════════════════════════════════════════════════
# #  TrajectoryDataset
# # ══════════════════════════════════════════════════════════════════════════════

# class TrajectoryDataset(Dataset):
#     """
#     TC trajectory dataset for the TCND_VN data structure.

#     File format (Data1d/*.txt):
#       STT  1.0  LONG_norm  LAT_norm  PRES_norm  WIND_norm  DATE  NAME

#     Expected directory tree:
#       <root>/
#         Data1d/
#           train/  *.txt
#           test/   *.txt
#         Data3d/   (satellite images, .nc or .npy)
#         ENV_DATA/ (env feature dicts, .npy)
#     """

#     def __init__(
#         self,
#         data_dir,
#         obs_len:    int   = 8,
#         pred_len:   int   = 12,
#         skip:       int   = 1,
#         threshold:  float = 0.002,
#         min_ped:    int   = 1,
#         delim:      str   = " ",
#         other_modal: str  = "gph",
#         test_year:  int | None = None,
#         type:       str   = "train",
#         is_test:    bool  = False,
#         **kwargs,
#     ):
#         super().__init__()

#         # ── Path resolution ──────────────────────────────────────────────
#         if isinstance(data_dir, dict):
#             root  = data_dir["root"]
#             dtype = data_dir.get("type", type)
#         else:
#             root  = data_dir
#             dtype = type
#         if is_test:
#             dtype = "test"

#         root = os.path.abspath(root)
#         bn   = os.path.basename(root)

#         if bn in ("train", "test", "val"):
#             self.root_path = os.path.dirname(os.path.dirname(root))
#         elif bn == "Data1d":
#             self.root_path = os.path.dirname(root)
#         else:
#             self.root_path = root

#         self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
#         self.data3d_path = os.path.join(self.root_path, "Data3d")
#         self.env_path    = os.path.join(self.root_path, "ENV_DATA")

#         logger.info(f"root       : {self.root_path}")
#         logger.info(f"Data1d ({dtype}): {self.data1d_path}")
#         logger.info(f"Data3d     : {self.data3d_path}")
#         logger.info(f"ENV_DATA   : {self.env_path}")

#         self.obs_len   = obs_len
#         self.pred_len  = pred_len
#         self.seq_len   = obs_len + pred_len
#         self.skip      = skip

#         if not os.path.exists(self.data1d_path):
#             logger.error(f"Missing: {self.data1d_path}")
#             self.num_seq = 0
#             self.seq_start_end = []
#             return

#         all_files = [
#             os.path.join(self.data1d_path, f)
#             for f in os.listdir(self.data1d_path)
#             if f.endswith(".txt")
#             and (test_year is None or str(test_year) in f)
#         ]
#         logger.info(f"{len(all_files)} files (year filter={test_year})")

#         self.obs_traj_raw  = []
#         self.pred_traj_raw = []
#         self.obs_Me_raw    = []
#         self.pred_Me_raw   = []
#         self.obs_rel_raw   = []
#         self.pred_rel_raw  = []
#         self.non_linear_ped = []
#         self.tyID          = []
#         num_peds_in_seq    = []

#         for path in all_files:
#             base  = os.path.splitext(os.path.basename(path))[0]
#             parts = base.split("_")
#             f_year = parts[0] if len(parts) > 0 else "unknown"
#             f_name = parts[1] if len(parts) > 1 else base

#             d    = self._read_file(path, delim)
#             data = d["main"]
#             add  = d["addition"]
#             if len(data) < self.seq_len:
#                 continue

#             frames     = np.unique(data[:, 0]).tolist()
#             frame_data = [data[data[:, 0] == f] for f in frames]
#             n_seq      = int(math.ceil(
#                 (len(frames) - self.seq_len + 1) / skip))

#             for idx in range(0, n_seq * skip, skip):
#                 if idx + self.seq_len > len(frame_data):
#                     break
#                 seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
#                 peds = np.unique(seg[:, 1])
#                 cnt  = 0

#                 for pid in peds:
#                     ps = seg[seg[:, 1] == pid]
#                     if len(ps) != self.seq_len:
#                         continue
#                     ps  = np.transpose(ps[:, 2:])       # [4, seq_len]
#                     rel = np.zeros_like(ps)
#                     rel[:, 1:] = ps[:, 1:] - ps[:, :-1]

#                     self.obs_traj_raw.append(
#                         torch.from_numpy(ps[:2, :obs_len]).float())
#                     self.pred_traj_raw.append(
#                         torch.from_numpy(ps[:2, obs_len:]).float())
#                     self.obs_Me_raw.append(
#                         torch.from_numpy(ps[2:, :obs_len]).float())
#                     self.pred_Me_raw.append(
#                         torch.from_numpy(ps[2:, obs_len:]).float())
#                     self.obs_rel_raw.append(
#                         torch.from_numpy(rel[:2, :obs_len]).float())
#                     self.pred_rel_raw.append(
#                         torch.from_numpy(rel[:2, obs_len:]).float())
#                     self.non_linear_ped.append(
#                         self._poly_fit(ps, pred_len, threshold))
#                     cnt += 1

#                 if cnt >= min_ped:
#                     num_peds_in_seq.append(cnt)
#                     self.tyID.append({
#                         "old":    [f_year, f_name, idx],
#                         "tydate": [add[i][0]
#                                    for i in range(idx, idx + self.seq_len)],
#                     })

#         self.num_seq = len(self.tyID)
#         cum = np.cumsum(num_peds_in_seq).tolist()
#         self.seq_start_end = list(zip([0] + cum[:-1], cum))
#         logger.info(f"{self.num_seq} sequences loaded")

#     # ── File reading ─────────────────────────────────────────────────────────

#     def _read_file(self, path: str, delim: str) -> dict:
#         data, add = [], []
#         with open(path) as f:
#             for line in f:
#                 p = line.strip().split(delim)
#                 if len(p) < 5:
#                     continue
#                 add.append(p[-2:])
#                 nums = [
#                     1.0 if i == 1
#                     else (float(v) if v.lower() not in ("null", "nan", "") else 0.0)
#                     for i, v in enumerate(p[:-2])
#                 ]
#                 data.append(nums)
#         return {"main": np.asarray(data), "addition": add}

#     def _poly_fit(self, traj, tlen, threshold):
#         t  = np.linspace(0, tlen - 1, tlen)
#         rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
#         ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
#         return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

#     # ── Image loading ─────────────────────────────────────────────────────────

#     def _load_nc(self, path: str) -> torch.Tensor | None:
#         try:
#             with nc.Dataset(path) as ds:
#                 key = list(ds.variables.keys())[-1]
#                 arr = np.array(ds.variables[key][:])
#             if arr.ndim == 3:
#                 arr = arr[0]
#             arr = cv2.resize(arr.astype(np.float32), (64, 64))
#             arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
#             return torch.from_numpy(arr).float().unsqueeze(-1)
#         except Exception:
#             return None

#     def _load_npy_img(self, path: str) -> torch.Tensor | None:
#         try:
#             arr = np.load(path)
#             if arr.ndim == 3:
#                 arr = arr[0]
#             arr = cv2.resize(arr.astype(np.float32), (64, 64))
#             arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
#             return torch.from_numpy(arr).float().unsqueeze(-1)
#         except Exception:
#             return None

#     def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
#         """Load satellite image → [64, 64, 1] float tensor."""
#         folder = os.path.join(self.data3d_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return torch.zeros(64, 64, 1)
#         prefix = f"WP{year}{ty_name}_{timestamp}"
#         for ext, fn in [(".nc", self._load_nc), (".npy", self._load_npy_img)]:
#             p = os.path.join(folder, prefix + ext)
#             if os.path.exists(p):
#                 r = fn(p)
#                 if r is not None:
#                     return r
#         try:
#             for fname in sorted(os.listdir(folder)):
#                 if timestamp in fname:
#                     p  = os.path.join(folder, fname)
#                     fn = self._load_nc if fname.endswith(".nc") else self._load_npy_img
#                     r  = fn(p)
#                     if r is not None:
#                         return r
#         except Exception:
#             pass
#         return torch.zeros(64, 64, 1)

#     # ── Env data loading ──────────────────────────────────────────────────────

#     def _load_env_npy(self, year, ty_name, timestamp) -> dict | None:
#         """Load raw .npy env dict for one timestep."""
#         folder = os.path.join(self.env_path, str(year), str(ty_name))
#         if not os.path.exists(folder):
#             return None
#         fname = f"WP{year}{ty_name}_{timestamp}.npy"
#         path  = os.path.join(folder, fname)
#         if not os.path.exists(path):
#             cands = [f for f in os.listdir(folder)
#                      if timestamp in f and f.endswith(".npy")]
#             path  = os.path.join(folder, cands[0]) if cands else None
#         if path and os.path.exists(path):
#             try:
#                 raw = np.load(path, allow_pickle=True).item()
#                 return env_data_processing(raw)
#             except Exception:
#                 pass
#         return None

#     def _get_env_features(
#         self,
#         year, ty_name,
#         dates: list[str],
#         obs_traj: np.ndarray,   # [2, T_obs] normalised
#     ) -> dict:
#         """
#         Build env feature dict for the full observation window.
#         Each key → tensor [T_obs, dim].
#         """
#         T = len(dates)
#         all_feats: list[dict] = []
#         prev_speed = None

#         for t in range(T):
#             lon_n = float(obs_traj[0, t])
#             lat_n = float(obs_traj[1, t])
#             env_npy = self._load_env_npy(year, ty_name, dates[t])
#             wind_n = 0.0   # we don't have direct access here; use env_npy

#             feat = _build_env_features(
#                 lon_norm    = lon_n,
#                 lat_norm    = lat_n,
#                 wind_norm   = wind_n,
#                 timestamp   = dates[t],
#                 env_npy     = env_npy,
#                 prev_speed_kmh = prev_speed,
#             )
#             all_feats.append(feat)

#             # compute move_velocity for next step's delta
#             if isinstance(env_npy, dict):
#                 mv = float(env_npy.get("move_velocity", 0.0))
#                 if mv == -1:
#                     mv = 0.0
#                 prev_speed = mv

#         # Stack per key: each [T_obs, dim]
#         env_out: dict = {}
#         for key in ENV_FEATURE_DIMS:
#             rows = []
#             for feat in all_feats:
#                 v = feat.get(key, [0.0] * ENV_FEATURE_DIMS[key])
#                 t = torch.tensor(v, dtype=torch.float)
#                 d = ENV_FEATURE_DIMS[key]
#                 if t.numel() < d:
#                     t = F.pad(t, (0, d - t.numel()))
#                 rows.append(t[:d])
#             env_out[key] = torch.stack(rows, dim=0)   # [T_obs, dim]
#         return env_out

#     def _embed_time(self, date_list: list[str]) -> torch.Tensor:
#         rows = []
#         for d in date_list:
#             try:
#                 rows.append([
#                     (float(d[:4]) - 1949) / 70.0 - 0.5,
#                     (float(d[4:6]) - 1)  / 11.0 - 0.5,
#                     (float(d[6:8]) - 1)  / 30.0 - 0.5,
#                     float(d[8:10])       / 18.0 - 0.5,
#                 ])
#             except Exception:
#                 rows.append([0., 0., 0., 0.])
#         arr = torch.tensor(rows, dtype=torch.float).t()
#         return arr.unsqueeze(0)                        # [1, 4, T]

#     # ── Dataset interface ─────────────────────────────────────────────────────

#     def __len__(self):
#         return self.num_seq

#     def __getitem__(self, index):
#         if self.num_seq == 0:
#             raise IndexError("Empty dataset")

#         s, e   = self.seq_start_end[index]
#         info   = self.tyID[index]
#         year   = str(info["old"][0])
#         tyname = str(info["old"][1])
#         dates  = info["tydate"]

#         # Images: [T_obs, 64, 64, 1]
#         imgs    = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
#         img_obs  = torch.stack(imgs, dim=0)
#         img_pred = torch.zeros(self.pred_len, 64, 64, 1)

#         # Per-ped trajectory tensors [n_ped, 2, T]
#         obs_traj  = torch.stack([self.obs_traj_raw[i]  for i in range(s, e)])
#         pred_traj = torch.stack([self.pred_traj_raw[i] for i in range(s, e)])
#         obs_rel   = torch.stack([self.obs_rel_raw[i]   for i in range(s, e)])
#         pred_rel  = torch.stack([self.pred_rel_raw[i]  for i in range(s, e)])
#         obs_Me    = torch.stack([self.obs_Me_raw[i]    for i in range(s, e)])
#         pred_Me   = torch.stack([self.pred_Me_raw[i]   for i in range(s, e)])
#         n         = e - s
#         nlp       = [self.non_linear_ped[i] for i in range(s, e)]
#         mask      = torch.ones(n, self.seq_len)

#         # Build env features using first ped's trajectory
#         obs_traj_np = obs_traj[0].numpy()   # [2, T_obs]
#         env_out = self._get_env_features(
#             year, tyname, dates[:self.obs_len], obs_traj_np)

#         return [
#             obs_traj,  pred_traj, obs_rel,  pred_rel,
#             nlp,       mask,
#             obs_Me,    pred_Me,   obs_rel,  pred_rel,
#             self._embed_time(dates[:self.obs_len]),
#             self._embed_time(dates[self.obs_len:]),
#             img_obs,   img_pred,
#             env_out,   info,
#         ]

"""
TCNM/data/trajectoriesWithMe_unet_training.py  ── v9
======================================================
TC trajectory dataset — TRAINING VERSION.
Fixed to build the FULL 90-dim env feature vector matching actual .npy data.

New features vs v8:
  bearing_to_scs_center  (16,) — 16-compass-direction one-hot to SCS centre
  dist_to_scs_boundary   (5,)  — distance class to [100–125°E, 5–20°N] bbox
  delta_velocity         (5,)  — speed change between consecutive steps

Data layout in .txt files (TCND_VN):
  col0  row_id (float)
  col1  ped_id (float, 1.0)
  col2  lon_norm  = (lon_01E − 1800) / 50
  col3  lat_norm  = lat_01N / 50
  col4  pres_norm = (pres_hPa − 960) / 50
  col5  wind_norm = (wind_kt − 40) / 25
  col-2 date   (e.g. 2019073106)
  col-1 name   (e.g. WIPHA)
"""
from __future__ import annotations

import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import netCDF4 as nc
    HAS_NC = True
except ImportError:
    HAS_NC = False

from TCNM.env_net_transformer_gphsplit import (
    bearing_to_scs_center_onehot,
    dist_to_scs_boundary_onehot,
    delta_velocity_onehot,
    intensity_class_onehot,
    build_env_features_one_step,
    feat_to_tensor,
    ENV_FEATURE_DIMS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Physical conversion constants ─────────────────────────────────────────────
NORM_TO_01DEG_LON = 50.0
NORM_OFFSET_LON   = 1800.0
NORM_TO_01DEG_LAT = 50.0


# ══════════════════════════════════════════════════════════════════════════════
#  env_data_processing  (exported — shared with inference dataset)
# ══════════════════════════════════════════════════════════════════════════════

def env_data_processing(env_dict: dict) -> dict:
    """Replace sentinel −1 with 0.0 and return cleaned dict."""
    if not isinstance(env_dict, dict):
        return {}
    cleaned = {}
    for k, v in env_dict.items():
        if isinstance(v, (list, np.ndarray)):
            cleaned[k] = v   # keep arrays as-is; build_env_features handles -1
        else:
            cleaned[k] = 0.0 if v == -1 else v
    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
#  seq_collate
# ══════════════════════════════════════════════════════════════════════════════

def seq_collate(data):
    """
    Collate a list of dataset items into batch tensors.

    Item layout (16 elements):
      0  obs_traj   [n_ped, 2, T_obs]
      1  pred_traj  [n_ped, 2, T_pred]
      2  obs_rel    [n_ped, 2, T_obs]
      3  pred_rel   [n_ped, 2, T_pred]
      4  nlp        list[float]
      5  mask       [n_ped, seq_len]
      6  obs_Me     [n_ped, 2, T_obs]
      7  pred_Me    [n_ped, 2, T_pred]
      8  obs_Me_rel [n_ped, 2, T_obs]
      9  pred_Me_rel[n_ped, 2, T_pred]
      10 obs_date   embed [1, 4, T_obs]
      11 pred_date  embed [1, 4, T_pred]
      12 img_obs    [T_obs, 64, 64, 1]
      13 img_pred   [T_pred, 64, 64, 1]
      14 env_data   dict of tensors [T_obs, dim]
      15 tyID       dict

    Batch outputs (same indices, stacked):
      0  obs_traj   [T_obs, B, 2]
      1  pred_traj  [T_pred, B, 2]
      7  obs_Me     [T_obs, B, 2]
      8  pred_Me    [T_pred, B, 2]
      11 img_obs    [B, 1, T_obs, 64, 64]
      12 img_pred   [B, 1, T_pred, 64, 64]
      13 env_data   dict — each key: [B, T, dim]
      15 tyID       list[dict]
    """
    (obs_traj, pred_traj, obs_rel, pred_rel,
     nlp, mask,
     obs_Me, pred_Me, obs_Me_rel, pred_Me_rel,
     obs_date, pred_date,
     img_obs, img_pred,
     env_data_raw, tyID) = zip(*data)

    def traj_TBC(lst):
        cat = torch.cat(lst, dim=0)         # [total_ped, 2, T]
        return cat.permute(2, 0, 1)         # [T, total_ped, 2]

    obs_traj_out    = traj_TBC(obs_traj)
    pred_traj_out   = traj_TBC(pred_traj)
    obs_rel_out     = traj_TBC(obs_rel)
    pred_rel_out    = traj_TBC(pred_rel)
    obs_Me_out      = traj_TBC(obs_Me)
    pred_Me_out     = traj_TBC(pred_Me)
    obs_Me_rel_out  = traj_TBC(obs_Me_rel)
    pred_Me_rel_out = traj_TBC(pred_Me_rel)

    nlp_out = torch.tensor(
        [v for sl in nlp for v in (sl if hasattr(sl, "__iter__") else [sl])],
        dtype=torch.float,
    )
    mask_out = torch.cat(list(mask), dim=0).permute(1, 0)

    counts = torch.tensor([t.shape[0] for t in obs_traj])
    cum    = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0]), cum[:-1]])
    seq_start_end = torch.stack([starts, cum], dim=1)

    # Images: [B, 1, T_obs, 64, 64]
    img_obs_out  = torch.stack(list(img_obs),  dim=0).permute(0, 4, 1, 2, 3).float()
    img_pred_out = torch.stack(list(img_pred), dim=0).permute(0, 4, 1, 2, 3).float()

    # env_data: merge — each key [B, T, dim]
    B = len(env_data_raw)
    env_out: dict | None = None
    valid_envs = [d for d in env_data_raw if isinstance(d, dict)]
    if valid_envs:
        env_out = {}
        all_keys: set[str] = set()
        for d in valid_envs:
            all_keys.update(d.keys())
        for key in all_keys:
            vals = []
            for d in env_data_raw:
                if isinstance(d, dict) and key in d:
                    v = d[key]
                    if not torch.is_tensor(v):
                        v = torch.tensor(v, dtype=torch.float)
                    vals.append(v.float())
                else:
                    ref = next((d[key] for d in valid_envs if key in d), None)
                    if ref is not None:
                        rt = torch.tensor(ref, dtype=torch.float) if not torch.is_tensor(ref) else ref.float()
                        vals.append(torch.zeros_like(rt))
                    else:
                        vals.append(torch.zeros(1))
            try:
                # Stack along batch dim; each is [T, dim]
                env_out[key] = torch.stack(vals, dim=0)   # [B, T, dim]
            except Exception:
                try:
                    mx = max(v.numel() for v in vals)
                    padded = [F.pad(v.flatten(), (0, mx - v.numel())) for v in vals]
                    env_out[key] = torch.stack(padded, dim=0)
                except Exception:
                    pass

    return (
        obs_traj_out,       # 0  [T_obs, B, 2]
        pred_traj_out,      # 1  [T_pred, B, 2]
        obs_rel_out,        # 2
        pred_rel_out,       # 3
        nlp_out,            # 4
        mask_out,           # 5
        seq_start_end,      # 6
        obs_Me_out,         # 7  [T_obs, B, 2]
        pred_Me_out,        # 8  [T_pred, B, 2]
        obs_Me_rel_out,     # 9
        pred_Me_rel_out,    # 10
        img_obs_out,        # 11 [B, 1, T_obs, 64, 64]
        img_pred_out,       # 12 [B, 1, T_pred, 64, 64]
        env_out,            # 13 dict
        None,               # 14
        list(tyID),         # 15
    )


# ══════════════════════════════════════════════════════════════════════════════
#  TrajectoryDataset (Training)
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryDataset(Dataset):
    """
    TC trajectory dataset for TCND_VN.

    Directory tree expected:
      <root>/
        Data1d/
          train/  *.txt
          val/    *.txt
          test/   *.txt
        Data3d/   (satellite images, .nc or .npy)
        ENV_DATA/ (env feature dicts, .npy)
    """

    def __init__(
        self,
        data_dir,
        obs_len:     int   = 8,
        pred_len:    int   = 12,
        skip:        int   = 1,
        threshold:   float = 0.002,
        min_ped:     int   = 1,
        delim:       str   = " ",
        other_modal: str   = "gph",
        test_year:   int | None = None,
        type:        str   = "train",
        is_test:     bool  = False,
        **kwargs,
    ):
        super().__init__()

        # ── Path resolution ───────────────────────────────────────────────
        if isinstance(data_dir, dict):
            root  = data_dir["root"]
            dtype = data_dir.get("type", type)
        else:
            root  = data_dir
            dtype = type
        if is_test:
            dtype = "test"

        root = os.path.abspath(root)
        bn   = os.path.basename(root)
        if bn in ("train", "test", "val"):
            self.root_path = os.path.dirname(os.path.dirname(root))
        elif bn == "Data1d":
            self.root_path = os.path.dirname(root)
        else:
            self.root_path = root

        self.data1d_path = os.path.join(self.root_path, "Data1d", dtype)
        self.data3d_path = os.path.join(self.root_path, "Data3d")
        self.env_path    = os.path.join(self.root_path, "ENV_DATA")

        logger.info(f"root        : {self.root_path}")
        logger.info(f"Data1d ({dtype}): {self.data1d_path}")
        logger.info(f"ENV_DATA    : {self.env_path}")

        self.obs_len    = obs_len
        self.pred_len   = pred_len
        self.seq_len    = obs_len + pred_len
        self.skip       = skip
        self.modal_name = other_modal

        if not os.path.exists(self.data1d_path):
            logger.error(f"Missing: {self.data1d_path}")
            self.num_seq = 0
            self.seq_start_end = []
            self.tyID = []
            return

        all_files = [
            os.path.join(self.data1d_path, f)
            for f in os.listdir(self.data1d_path)
            if f.endswith(".txt")
            and (test_year is None or str(test_year) in f)
        ]
        logger.info(f"{len(all_files)} files (year={test_year})")

        self.obs_traj_raw   = []
        self.pred_traj_raw  = []
        self.obs_Me_raw     = []
        self.pred_Me_raw    = []
        self.obs_rel_raw    = []
        self.pred_rel_raw   = []
        self.non_linear_ped = []
        self.tyID           = []
        num_peds_in_seq     = []

        for path in all_files:
            base   = os.path.splitext(os.path.basename(path))[0]
            parts  = base.split("_")
            f_year = parts[0] if parts else "unknown"
            f_name = parts[1] if len(parts) > 1 else base

            d    = self._read_file(path, delim)
            data = d["main"]
            add  = d["addition"]
            if len(data) < self.seq_len:
                continue

            frames     = np.unique(data[:, 0]).tolist()
            frame_data = [data[data[:, 0] == f] for f in frames]
            n_seq      = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, n_seq * skip, skip):
                if idx + self.seq_len > len(frame_data):
                    break
                seg  = np.concatenate(frame_data[idx: idx + self.seq_len])
                peds = np.unique(seg[:, 1])
                cnt  = 0

                for pid in peds:
                    ps = seg[seg[:, 1] == pid]
                    if len(ps) != self.seq_len:
                        continue
                    ps  = np.transpose(ps[:, 2:])      # [4, seq_len]
                    rel = np.zeros_like(ps)
                    rel[:, 1:] = ps[:, 1:] - ps[:, :-1]

                    self.obs_traj_raw.append(
                        torch.from_numpy(ps[:2, :obs_len]).float())
                    self.pred_traj_raw.append(
                        torch.from_numpy(ps[:2, obs_len:]).float())
                    self.obs_Me_raw.append(
                        torch.from_numpy(ps[2:, :obs_len]).float())
                    self.pred_Me_raw.append(
                        torch.from_numpy(ps[2:, obs_len:]).float())
                    self.obs_rel_raw.append(
                        torch.from_numpy(rel[:2, :obs_len]).float())
                    self.pred_rel_raw.append(
                        torch.from_numpy(rel[:2, obs_len:]).float())
                    self.non_linear_ped.append(
                        self._poly_fit(ps, pred_len, threshold))
                    cnt += 1

                if cnt >= min_ped:
                    num_peds_in_seq.append(cnt)
                    self.tyID.append({
                        "old":    [f_year, f_name, idx],
                        "tydate": [add[i][0] for i in range(idx, idx + self.seq_len)],
                    })

        self.num_seq = len(self.tyID)
        cum = np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = list(zip([0] + cum[:-1], cum))
        logger.info(f"{self.num_seq} sequences loaded")

    # ── File reading ──────────────────────────────────────────────────────────

    def _read_file(self, path: str, delim: str) -> dict:
        data, add = [], []
        with open(path) as f:
            for line in f:
                p = line.strip().split(delim)
                if len(p) < 5:
                    continue
                add.append(p[-2:])
                nums = [
                    1.0 if i == 1
                    else (float(v) if v.lower() not in ("null", "nan", "") else 0.0)
                    for i, v in enumerate(p[:-2])
                ]
                data.append(nums)
        return {"main": np.asarray(data), "addition": add}

    def _poly_fit(self, traj, tlen, threshold):
        t  = np.linspace(0, tlen - 1, tlen)
        rx = np.polyfit(t, traj[0, -tlen:], 2, full=True)[1]
        ry = np.polyfit(t, traj[1, -tlen:], 2, full=True)[1]
        return 1.0 if (len(rx) > 0 and rx[0] + ry[0] >= threshold) else 0.0

    # ── Image loading ─────────────────────────────────────────────────────────

    def _resize_norm(self, arr: np.ndarray) -> np.ndarray:
        arr = arr.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if HAS_CV2:
            arr = cv2.resize(arr, (64, 64))
        else:
            # simple nearest-neighbor resize fallback
            h, w = arr.shape[:2]
            row_idx = (np.arange(64) * h // 64).astype(int)
            col_idx = (np.arange(64) * w // 64).astype(int)
            arr = arr[np.ix_(row_idx, col_idx)]
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
        return arr

    def img_read(self, year, ty_name, timestamp) -> torch.Tensor:
        folder = os.path.join(self.data3d_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return torch.zeros(64, 64, 1)
        prefix = f"WP{year}{ty_name}_{timestamp}"
        # Try exact match
        for ext in (".nc", ".npy"):
            p = os.path.join(folder, prefix + ext)
            if os.path.exists(p):
                arr = self._load_array(p)
                if arr is not None:
                    return torch.from_numpy(self._resize_norm(arr)).unsqueeze(-1)
        # Fuzzy match
        try:
            for fname in sorted(os.listdir(folder)):
                if timestamp in fname:
                    arr = self._load_array(os.path.join(folder, fname))
                    if arr is not None:
                        return torch.from_numpy(self._resize_norm(arr)).unsqueeze(-1)
        except Exception:
            pass
        return torch.zeros(64, 64, 1)

    def _load_array(self, path: str):
        try:
            if path.endswith(".npy"):
                return np.load(path)
            elif path.endswith(".nc") and HAS_NC:
                with nc.Dataset(path) as ds:
                    key = list(ds.variables.keys())[-1]
                    return np.array(ds.variables[key][:])
        except Exception:
            pass
        return None

    # ── Env loading ───────────────────────────────────────────────────────────

    def _load_env_npy(self, year, ty_name, timestamp) -> dict | None:
        folder = os.path.join(self.env_path, str(year), str(ty_name))
        if not os.path.exists(folder):
            return None
        for fname in [f"WP{year}{ty_name}_{timestamp}.npy", f"{timestamp}.npy"]:
            path = os.path.join(folder, fname)
            if os.path.exists(path):
                try:
                    raw = np.load(path, allow_pickle=True).item()
                    return env_data_processing(raw)
                except Exception:
                    pass
        # fuzzy
        try:
            cands = [f for f in os.listdir(folder)
                     if timestamp in f and f.endswith(".npy")]
            if cands:
                raw = np.load(os.path.join(folder, cands[0]),
                              allow_pickle=True).item()
                return env_data_processing(raw)
        except Exception:
            pass
        return None

    def _get_env_features(
        self,
        year: str,
        ty_name: str,
        dates: list[str],
        obs_traj: np.ndarray,  # [2, T_obs] normalised (lon, lat)
        obs_Me:   np.ndarray,  # [2, T_obs] normalised (pres, wind)
    ) -> dict:
        """Build env feature dict for full observation window. Each key → [T_obs, dim]."""
        T = len(dates)
        all_feats: list[dict] = []
        prev_speed = None

        for t in range(T):
            lon_n  = float(obs_traj[0, t])
            lat_n  = float(obs_traj[1, t])
            wind_n = float(obs_Me[1, t])   # wind_norm from Me
            env_npy = self._load_env_npy(year, ty_name, dates[t])

            feat = build_env_features_one_step(
                lon_norm       = lon_n,
                lat_norm       = lat_n,
                wind_norm      = wind_n,
                timestamp      = dates[t],
                env_npy        = env_npy,
                prev_speed_kmh = prev_speed,
            )
            all_feats.append(feat)

            # update prev_speed for next step's delta_velocity
            if isinstance(env_npy, dict):
                mv = float(env_npy.get("move_velocity", 0.0) or 0.0)
                prev_speed = mv if mv != -1 else 0.0

        # Stack per key: each [T_obs, dim]
        env_out: dict = {}
        for key in ENV_FEATURE_DIMS:
            dim  = ENV_FEATURE_DIMS[key]
            rows = []
            for feat in all_feats:
                v = feat.get(key, [0.0] * dim)
                t = torch.tensor(v, dtype=torch.float)
                if t.numel() < dim:
                    t = F.pad(t, (0, dim - t.numel()))
                rows.append(t[:dim])
            env_out[key] = torch.stack(rows, dim=0)   # [T_obs, dim]
        return env_out

    def _embed_time(self, date_list: list[str]) -> torch.Tensor:
        rows = []
        for d in date_list:
            try:
                rows.append([
                    (float(d[:4]) - 1949) / 70.0 - 0.5,
                    (float(d[4:6]) - 1)   / 11.0 - 0.5,
                    (float(d[6:8]) - 1)   / 30.0 - 0.5,
                    float(d[8:10])        / 18.0 - 0.5,
                ])
            except Exception:
                rows.append([0., 0., 0., 0.])
        arr = torch.tensor(rows, dtype=torch.float).t()
        return arr.unsqueeze(0)   # [1, 4, T]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        if self.num_seq == 0:
            raise IndexError("Empty dataset")

        s, e   = self.seq_start_end[index]
        info   = self.tyID[index]
        year   = str(info["old"][0])
        tyname = str(info["old"][1])
        dates  = info["tydate"]

        # Images
        imgs     = [self.img_read(year, tyname, ts) for ts in dates[:self.obs_len]]
        img_obs  = torch.stack(imgs, dim=0)                          # [T_obs, 64, 64, 1]
        img_pred = torch.zeros(self.pred_len, 64, 64, 1)

        # Trajectory tensors [n_ped, 2, T]
        obs_traj  = torch.stack([self.obs_traj_raw[i]  for i in range(s, e)])
        pred_traj = torch.stack([self.pred_traj_raw[i] for i in range(s, e)])
        obs_rel   = torch.stack([self.obs_rel_raw[i]   for i in range(s, e)])
        pred_rel  = torch.stack([self.pred_rel_raw[i]  for i in range(s, e)])
        obs_Me    = torch.stack([self.obs_Me_raw[i]    for i in range(s, e)])
        pred_Me   = torch.stack([self.pred_Me_raw[i]   for i in range(s, e)])
        n         = e - s
        nlp       = [self.non_linear_ped[i] for i in range(s, e)]
        mask      = torch.ones(n, self.seq_len)

        # Build env features using first ped
        obs_traj_np = obs_traj[0].numpy()   # [2, T_obs]
        obs_Me_np   = obs_Me[0].numpy()     # [2, T_obs]
        env_out = self._get_env_features(
            year, tyname, dates[:self.obs_len], obs_traj_np, obs_Me_np)

        return [
            obs_traj,   pred_traj,  obs_rel,  pred_rel,
            nlp,        mask,
            obs_Me,     pred_Me,    obs_rel,  pred_rel,
            self._embed_time(dates[:self.obs_len]),
            self._embed_time(dates[self.obs_len:]),
            img_obs,    img_pred,
            env_out,    info,
        ]