
# """
# TCNM/data/trajectoriesWithMe_unet.py — TEST / INFERENCE VERSION
# ================================================================
# Đồng bộ với bản Training. Hỗ trợ lọc theo năm.

# FIX: import env_data_processing từ training module (đã được export).
#      Thứ tự tọa độ: [LONG, LAT, PRES, WIND] (giữ nguyên).
# """
# import os
# import logging
# import math
# import numpy as np
# import cv2
# import torch
# from torch.utils.data import Dataset

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ── Reuse from training module ────────────────────────────────────────────────
# from TCNM.data.trajectoriesWithMe_unet_training import (
#     env_data_processing,   # FIX: was missing in original
#     seq_collate,
# )


# class TrajectoryDataset(Dataset):
#     """
#     Test/Inference Dataset.
#     Thứ tự tọa độ: [LONG, LAT, PRES, WIND]
#     """
#     def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1,
#                  threshold=0.002, min_ped=1, delim=' ',
#                  other_modal='gph', test_year=None, **kwargs):
#         super().__init__()

#         # ── Path resolution ────────────────────────────────
#         if isinstance(data_dir, dict):
#             input_root = data_dir['root']
#             dtype      = data_dir.get('type', 'test')
#         else:
#             input_root = data_dir
#             dtype      = kwargs.get('type', 'test')

#         if 'Data1d' in input_root:
#             self.data1d_path = (input_root if input_root.endswith(dtype)
#                                 else os.path.join(input_root, dtype))
#         else:
#             self.data1d_path = os.path.join(input_root, 'Data1d', dtype)

#         self.root_path   = os.path.dirname(os.path.dirname(self.data1d_path))
#         self.data3d_path = os.path.join(self.root_path, 'Data3d')
#         self.env_path    = os.path.join(self.root_path, 'ENV_DATA')

#         self.obs_len    = obs_len
#         self.pred_len   = pred_len
#         self.skip       = skip
#         self.seq_len    = obs_len + pred_len
#         self.modal_name = other_modal

#         if not os.path.exists(self.data1d_path):
#             logger.warning(f"Data path not found: {self.data1d_path}")
#             self.num_seq = 0
#             self.tyID    = []
#             return

#         all_files = [
#             os.path.join(self.data1d_path, f)
#             for f in os.listdir(self.data1d_path)
#             if f.endswith('.txt')
#             and (not test_year or str(test_year) in f)
#         ]
#         logger.info(f"Found {len(all_files)} test files  (year={test_year})")

#         self.seq_list       = []
#         self.seq_list_rel   = []
#         self.non_linear_ped = []
#         self.tyID           = []
#         num_peds_in_seq     = []

#         for path in all_files:
#             filename = os.path.basename(path).replace('.txt', '')
#             parts    = filename.split('_')
#             f_year   = parts[0] if len(parts) >= 2 else 'Unknown'
#             f_name   = parts[1] if len(parts) >= 2 else filename

#             data_dict = self._read_track_file(path)
#             addinf, data = data_dict['addition'], data_dict['main']

#             if len(data) < self.seq_len:
#                 continue

#             frames       = np.unique(data[:, 0]).tolist()
#             frame_data   = [data[data[:, 0] == f, :] for f in frames]
#             num_sequences = int(math.ceil(
#                 (len(frames) - self.seq_len + 1) / self.skip))

#             for idx in range(0, num_sequences * self.skip, self.skip):
#                 if idx + self.seq_len > len(frame_data):
#                     break

#                 curr = np.concatenate(frame_data[idx:idx + self.seq_len])
#                 peds = np.unique(curr[:, 1])
#                 cnt  = 0

#                 c_seq = np.zeros((len(peds), 4, self.seq_len))
#                 c_rel = np.zeros((len(peds), 4, self.seq_len))

#                 for p_id in peds:
#                     p_seq = curr[curr[:, 1] == p_id, :]
#                     if len(p_seq) != self.seq_len:
#                         continue
#                     p_seq        = np.transpose(p_seq[:, 2:])   # [4, seq_len]
#                     rel_seq      = np.zeros(p_seq.shape)
#                     rel_seq[:, 1:] = p_seq[:, 1:] - p_seq[:, :-1]
#                     c_seq[cnt]   = p_seq
#                     c_rel[cnt]   = rel_seq
#                     self.non_linear_ped.append(1.0)
#                     cnt += 1

#                 if cnt >= min_ped:
#                     num_peds_in_seq.append(cnt)
#                     self.seq_list.append(c_seq[:cnt])
#                     self.seq_list_rel.append(c_rel[:cnt])
#                     self.tyID.append({
#                         'old':    [f_year, f_name, idx],
#                         'tydate': [addinf[i][0] for i in range(idx, idx + self.seq_len)]
#                     })

#         self.num_seq = len(self.seq_list)

#         if self.num_seq > 0:
#             all_seq = np.concatenate(self.seq_list, axis=0)
#             all_rel = np.concatenate(self.seq_list_rel, axis=0)

#             self.obs_traj  = torch.from_numpy(all_seq[:, :2, :self.obs_len]).float()
#             self.pred_traj = torch.from_numpy(all_seq[:, :2, self.obs_len:]).float()
#             self.obs_Me    = torch.from_numpy(all_seq[:, 2:, :self.obs_len]).float()
#             self.pred_Me   = torch.from_numpy(all_seq[:, 2:, self.obs_len:]).float()
#             self.obs_rel   = torch.from_numpy(all_rel[:, :2, :self.obs_len]).float()
#             self.pred_rel  = torch.from_numpy(all_rel[:, :2, self.obs_len:]).float()

#             cumsum = np.cumsum(num_peds_in_seq).tolist()
#             self.seq_start_end = list(zip([0] + cumsum[:-1], cumsum))

#             logger.info(f"✅ {self.num_seq} sequences for inference")

#     # ── File reader ───────────────────────────────────────

#     def _read_track_file(self, path):
#         data, add = [], []
#         with open(path, 'r') as f:
#             for i, line in enumerate(f):
#                 parts = line.strip().split()
#                 if len(parts) < 6:
#                     continue
#                 row = [float(i), 1.0]
#                 for val in parts[2:6]:
#                     try:
#                         row.append(float(val) if val.lower() != 'null' else 0.0)
#                     except Exception:
#                         row.append(0.0)
#                 data.append(row)
#                 add.append(parts[-2:])
#         return {'main': np.asarray(data), 'addition': add}

#     # ── Data loaders ──────────────────────────────────────

#     def _load_data3d(self, year, ty_name, timestamp):
#         """Load satellite image → [64, 64, 1]"""
#         for name in [f"WP{year}{ty_name}_{timestamp}.npy",
#                      f"{timestamp}.npy"]:
#             path = os.path.join(self.data3d_path, str(year), ty_name, name)
#             if os.path.exists(path):
#                 try:
#                     img = np.load(path)
#                     img = cv2.resize(img, (64, 64))
#                     img = self._transforms(img)
#                     return torch.from_numpy(img[:, :, np.newaxis]).float()
#                 except Exception:
#                     pass
#         return torch.zeros(64, 64, 1)

#     def _load_env_data(self, year, ty_name, timestamp):
#         """Load env dict → clean via env_data_processing."""
#         for name in [f"WP{year}{ty_name}_{timestamp}.npy",
#                      f"{timestamp}.npy"]:
#             path = os.path.join(self.env_path, str(year), ty_name, name)
#             if os.path.exists(path):
#                 try:
#                     d = np.load(path, allow_pickle=True).item()
#                     return env_data_processing(d)   # FIX: use shared function
#                 except Exception:
#                     pass
#         return {'wind': 0.0, 'move_velocity': 0.0, 'month': np.zeros(12)}

#     def _transforms(self, img):
#         modal_range = {
#             'gph': (44490.5, 58768.4),
#             'sst': (273, 312),
#         }
#         vmin, vmax = modal_range.get(self.modal_name, (img.min(), img.max() + 1e-6))
#         return np.clip((img - vmin) / (vmax - vmin), 0, 1)

#     def _embed_time(self, date_list):
#         rows = []
#         for d in date_list:
#             try:
#                 rows.append([
#                     (float(d[:4]) - 1949) / 76.0 - 0.5,
#                     (float(d[4:6]) - 1)  / 11.0 - 0.5,
#                     (float(d[6:8]) - 1)  / 30.0 - 0.5,
#                     float(d[8:10])       / 18.0 - 0.5,
#                 ])
#             except Exception:
#                 rows.append([0., 0., 0., 0.])
#         return torch.tensor(rows).transpose(1, 0).unsqueeze(0).float()

#     # ── Dataset interface ─────────────────────────────────

#     def __len__(self):
#         return self.num_seq

#     def __getitem__(self, index):
#         s, e   = self.seq_start_end[index]
#         info   = self.tyID[index]
#         dates  = info['tydate']
#         year   = info['old'][0]
#         tyname = info['old'][1]

#         obs_traj  = self.obs_traj[s:e].squeeze(0)
#         pred_traj = self.pred_traj[s:e].squeeze(0)
#         obs_Me    = self.obs_Me[s:e].squeeze(0)
#         pred_Me   = self.pred_Me[s:e].squeeze(0)
#         obs_rel   = self.obs_rel[s:e].squeeze(0)
#         pred_rel  = self.pred_rel[s:e].squeeze(0)

#         img_obs_list = [self._load_data3d(year, tyname, d)
#                         for d in dates[:self.obs_len]]
#         img_obs  = torch.stack(img_obs_list, dim=0)        # [T_obs, 64, 64, 1]
#         env_dict = self._load_env_data(year, tyname, dates[self.obs_len - 1])

#         # 16-element tuple matching seq_collate
#         return [
#             obs_traj,  pred_traj,  obs_rel, pred_rel,
#             1.0,       torch.ones((self.seq_len,)),
#             obs_Me,    pred_Me,    obs_rel, pred_rel,
#             self._embed_time(dates[:self.obs_len]),
#             self._embed_time(dates[self.obs_len:]),
#             img_obs,
#             torch.zeros((self.pred_len, 64, 64, 1)),
#             env_dict,
#             info,
#         ]

"""
TCNM/data/trajectoriesWithMe_unet.py  ── TEST / INFERENCE VERSION  v9
=======================================================================
Synced with training dataset. Supports year filtering.
Uses shared env_data_processing and seq_collate from training module.
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from TCNM.data.trajectoriesWithMe_unet_training import (
    env_data_processing,
    seq_collate,
    TrajectoryDataset as _TrainingDataset,
)
from TCNM.env_net_transformer_gphsplit import (
    build_env_features_one_step,
    ENV_FEATURE_DIMS,
)


class TrajectoryDataset(_TrainingDataset):
    """
    Test/Inference dataset — inherits from training dataset.
    Overrides default dtype to 'test'. Everything else identical.
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
        type:        str   = "test",
        is_test:     bool  = True,
        **kwargs,
    ):
        super().__init__(
            data_dir    = data_dir,
            obs_len     = obs_len,
            pred_len    = pred_len,
            skip        = skip,
            threshold   = threshold,
            min_ped     = min_ped,
            delim       = delim,
            other_modal = other_modal,
            test_year   = test_year,
            type        = type,
            is_test     = is_test,
            **kwargs,
        )