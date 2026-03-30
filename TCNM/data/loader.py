# """
# TCNM/data/loader.py  ── v9
# ============================
# Smart data loader — resolves TCND_vn root automatically.
# Compatible with Kaggle (Google Drive mount) and local paths.
# """
# import os
# import sys

# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# sys.path.insert(0, project_root)

# from torch.utils.data import DataLoader
# from TCNM.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate


# def _find_tcnd_root(path: str) -> str:
#     """Walk up directory tree to find folder containing Data1d/."""
#     path = os.path.abspath(path)
#     check = path
#     for _ in range(6):
#         if os.path.exists(os.path.join(check, "Data1d")):
#             return check
#         parent = os.path.dirname(check)
#         if parent == check:
#             break
#         check = parent
#     # Try common Kaggle/Colab mount names
#     for sub in ("TCND_vn", "tcnd_vn", "data"):
#         candidate = os.path.join(path, sub)
#         if os.path.exists(os.path.join(candidate, "Data1d")):
#             return candidate
#     return path


# def data_loader(args, path_config, test: bool = False, test_year=None):
#     if isinstance(path_config, dict):
#         raw_path  = path_config.get("root", "")
#         dset_type = path_config.get("type", "test" if test else "train")
#     else:
#         raw_path  = str(path_config)
#         dset_type = "test" if test else "train"

#     root = _find_tcnd_root(raw_path)
#     print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

#     dataset = TrajectoryDataset(
#         data_dir    = root,
#         obs_len     = args.obs_len,
#         pred_len    = args.pred_len,
#         skip        = args.skip,
#         threshold   = args.threshold,
#         min_ped     = args.min_ped,
#         delim       = args.delim,
#         other_modal = getattr(args, "other_modal", "gph"),
#         test_year   = test_year,
#         type        = dset_type,
#         is_test     = test,
#     )

#     loader = DataLoader(
#         dataset,
#         batch_size  = args.batch_size,
#         shuffle     = not test,
#         collate_fn  = seq_collate,
#         num_workers = 0,
#         drop_last   = False,
#         pin_memory  = True if not test else False,
#     )

#     print(f"  ✅  {len(dataset)} sequences")
#     return dataset, loader
"""
TCNM/data/loader.py  ── v9-fixed (patched)
============================================
Smart data loader — resolves TCND_vn root automatically.

FIX: persistent_workers=True chỉ được set khi num_workers > 0.
     Trên Kaggle dùng num_workers=0 (default) để tránh deadlock.
"""
from __future__ import annotations

import os
import sys

_here = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _here not in sys.path:
    sys.path.insert(0, _here)

from torch.utils.data import DataLoader

from TCNM.data.trajectoriesWithMe_unet_training import TrajectoryDataset, seq_collate


def _find_tcnd_root(path: str) -> str:
    """Walk up the directory tree to find the folder containing Data1d/."""
    path  = os.path.abspath(path)
    check = path
    for _ in range(6):
        if os.path.exists(os.path.join(check, "Data1d")):
            return check
        parent = os.path.dirname(check)
        if parent == check:
            break
        check = parent

    for sub in ("TCND_vn", "tcnd_vn", "data", "TCND"):
        candidate = os.path.join(path, sub)
        if os.path.exists(os.path.join(candidate, "Data1d")):
            return candidate

    return path


def data_loader(
    args,
    path_config,
    test:       bool      = False,
    test_year:  int | None = None,
    batch_size: int | None = None,
) -> tuple:
    """
    Create a DataLoader for the given split.

    Parameters
    ----------
    args        : argparse Namespace with obs_len, pred_len, batch_size, etc.
    path_config : str | dict — {'root': ..., 'type': 'train'|'val'|'test'}
    test        : bool — if True, shuffle=False
    test_year   : int | None — filter Data1d files by year string
    batch_size  : override args.batch_size if provided

    Returns
    -------
    (dataset, loader)
    """
    if isinstance(path_config, dict):
        raw_path  = path_config.get("root", "")
        dset_type = path_config.get("type", "test" if test else "train")
    else:
        raw_path  = str(path_config)
        dset_type = "test" if test else "train"

    root = _find_tcnd_root(raw_path)
    print(f"DataLoader | root={root} | type={dset_type} | year={test_year}")

    dataset = TrajectoryDataset(
        data_dir    = root,
        obs_len     = args.obs_len,
        pred_len    = args.pred_len,
        skip        = getattr(args, "skip",        1),
        threshold   = getattr(args, "threshold",   0.002),
        min_ped     = getattr(args, "min_ped",      1),
        delim       = getattr(args, "delim",        " "),
        other_modal = getattr(args, "other_modal",  "gph"),
        test_year   = test_year,
        type        = dset_type,
        is_test     = test,
    )

    # FIX: num_workers mặc định 0 (Kaggle-safe)
    # persistent_workers và prefetch_factor CHỈ dùng khi num_workers > 0
    num_workers = getattr(args, "num_workers", 0)
    use_persistent = num_workers > 0
    prefetch       = 2 if num_workers > 0 else None

    loader = DataLoader(
        dataset,
        batch_size         = batch_size or args.batch_size,
        shuffle            = not test,
        collate_fn         = seq_collate,
        num_workers        = num_workers,
        persistent_workers = use_persistent,   # FIX: False khi num_workers=0
        prefetch_factor    = prefetch,         # FIX: None khi num_workers=0
        drop_last          = False,
        pin_memory         = _cuda_available() and num_workers > 0,
    )

    print(f"  {len(dataset)} sequences loaded  (workers={num_workers})")
    return dataset, loader


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False