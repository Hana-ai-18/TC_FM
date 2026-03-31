
"""
TCNM/data/trajectoriesWithMe_unet.py  ── v9-fixed (TEST / INFERENCE)
======================================================================
Test dataset — inherits TrajectoryDataset from training module.
Identical logic, defaults to dtype='test' and is_test=True.
"""
from __future__ import annotations

from TCNM.data.trajectoriesWithMe_unet_training import (
    TrajectoryDataset as _TrainingDataset,
    seq_collate,                   # re-export so callers can import from here
    env_data_processing,
)


class TrajectoryDataset(_TrainingDataset):
    """
    Test/Inference dataset — wraps training dataset with test defaults.
    Overrides type to 'test' and is_test=True.
    Everything else (normalisation, env loading, Data3d loading) is identical.
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


__all__ = ["TrajectoryDataset", "seq_collate", "env_data_processing"]