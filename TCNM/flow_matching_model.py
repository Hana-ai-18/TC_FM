
"""
TCNM/flow_matching_model.py  ── v8
====================================
OT-CFM Flow Matching + PINN for TC trajectory prediction.

Changes from v7
---------------
- Loss computation delegated to TCNM/losses.py (no duplication)
- afCRPS (M=4 ensemble) replaces MSE for FM loss — Lang et al. 2026
- heading_loss uses Greer 2021 + Runge 2021 formulation
- sample() returns (traj, Me, per_sample_trajs) for CRPS evaluation
- save_predict_csv() writes per-prediction CSV after every call to sample()
- Bug fixes from v7 retained: ERA5 ch7/ch11, PINN scale ×100, ensemble=1 train
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import math

from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from TCNM.Unet3D_merge_tiny import Unet3D
from TCNM.env_net_transformer_gphsplit import Env_net
from TCNM.losses import compute_total_loss, WEIGHTS

# ── Shape convention ──────────────────────────────────────────────────────────
# trajectory : [T, B, 2]   T=steps, B=batch, 2=(lon_norm, lat_norm)
# ─────────────────────────────────────────────────────────────────────────────

NORM_TO_DEG = 5.0


# ══════════════════════════════════════════════════════════════════════════════
#  Velocity field (denoiser / flow network)
# ══════════════════════════════════════════════════════════════════════════════

class VelocityField(nn.Module):
    """
    Predicts the OT-CFM velocity field  v_θ(x_t, t, context).

    Architecture: 3-D U-Net (spatial) + Env-Transformer + LSTM (obs history)
    fused via a context MLP, then decoded with a Transformer + linear head.
    """

    def __init__(
        self,
        pred_len:  int   = 12,
        obs_len:   int   = 8,
        ctx_dim:   int   = 128,
        sigma_min: float = 0.02,
    ):
        super().__init__()
        self.pred_len  = pred_len
        self.obs_len   = obs_len
        self.sigma_min = sigma_min

        # ── Encoders ─────────────────────────────────────────────────────
        self.spatial_enc  = Unet3D(in_channel=1, out_channel=1)
        self.env_enc      = Env_net(obs_len=obs_len, d_model=64)
        self.obs_lstm     = nn.LSTM(
            input_size=4, hidden_size=128, num_layers=3,
            batch_first=True, dropout=0.2,
        )
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))

        # ── Context fusion ────────────────────────────────────────────────
        # 16 (spatial) + 64 (env) + 128 (obs) = 208
        self.ctx_fc1  = nn.Linear(16 + 64 + 128, 512)
        self.ctx_ln   = nn.LayerNorm(512)
        self.ctx_drop = nn.Dropout(0.15)
        self.ctx_fc2  = nn.Linear(512, ctx_dim)

        # ── Time embedding ────────────────────────────────────────────────
        self.time_fc1 = nn.Linear(128, 256)
        self.time_fc2 = nn.Linear(256, 128)

        # ── Trajectory decoder ────────────────────────────────────────────
        self.traj_embed = nn.Linear(4, 128)
        self.pos_enc    = nn.Parameter(torch.randn(1, pred_len, 128) * 0.02)

        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=128, nhead=8, dim_feedforward=512,
                dropout=0.15, activation="gelu", batch_first=True,
            ),
            num_layers=4,
        )
        self.out_fc1 = nn.Linear(128, 256)
        self.out_fc2 = nn.Linear(256, 4)

    # ── Time embedding (sinusoidal) ──────────────────────────────────────

    def _time_emb(self, t: torch.Tensor, dim: int = 128) -> torch.Tensor:
        half = dim // 2
        freq = torch.exp(
            torch.arange(half, dtype=torch.float32, device=t.device)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        emb = t.float().unsqueeze(1) * 1_000.0 * freq.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return F.pad(emb, (0, dim % 2))

    # ── Context vector ───────────────────────────────────────────────────

    def _context(self, batch_list: List) -> torch.Tensor:
        obs_traj  = batch_list[0]    # [T_obs, B, 2]
        obs_Me    = batch_list[7]    # [T_obs, B, 2]
        image_obs = batch_list[11]
        env_data  = batch_list[13]

        f_s = self.spatial_enc(image_obs).mean(dim=2)
        f_s = self.spatial_pool(f_s).flatten(1)           # [B, 16]

        f_e, _, _ = self.env_enc(env_data, image_obs)     # [B, 64]

        obs_in    = torch.cat([obs_traj, obs_Me], dim=2).permute(1, 0, 2)
        _, (h_n, _) = self.obs_lstm(obs_in)
        f_h       = h_n[-1]                               # [B, 128]

        ctx = torch.cat([f_s, f_e, f_h], dim=-1)          # [B, 208]
        ctx = F.gelu(self.ctx_ln(self.ctx_fc1(ctx)))
        ctx = self.ctx_drop(ctx)
        return self.ctx_fc2(ctx)                           # [B, ctx_dim]

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        x_t:        torch.Tensor,   # [B, T, 4]
        t:          torch.Tensor,   # [B]
        batch_list: List,
    ) -> torch.Tensor:              # [B, T, 4]
        ctx   = self._context(batch_list)                  # [B, ctx_dim]
        t_emb = F.gelu(self.time_fc1(self._time_emb(t)))
        t_emb = self.time_fc2(t_emb)                      # [B, 128]

        x_emb  = self.traj_embed(x_t) + self.pos_enc + t_emb.unsqueeze(1)
        memory = torch.cat([t_emb.unsqueeze(1), ctx.unsqueeze(1)], dim=1)

        out = self.transformer(x_emb, memory)
        return self.out_fc2(F.gelu(self.out_fc1(out)))


# ══════════════════════════════════════════════════════════════════════════════
#  Main model
# ══════════════════════════════════════════════════════════════════════════════

class TCFlowMatching(nn.Module):
    """
    TC trajectory prediction via Optimal-Transport Conditional Flow Matching
    with PINN-BVE physics regularisation.

    Training loss  (Eq. 60 in paper):
        L_total = 1.0·L_FM + 2.0·L_dir + 0.5·L_step
                + 1.0·L_disp + 2.0·L_heading
                + 0.2·L_smooth + 0.5·L_PINN

    Outputs: distribution p(Y | context), not a single trajectory.
    """

    def __init__(
        self,
        pred_len:   int   = 12,
        obs_len:    int   = 8,
        sigma_min:  float = 0.02,
        n_train_ens: int  = 4,     # ensemble size during training (afCRPS)
        **kwargs,
    ):
        super().__init__()
        self.pred_len    = pred_len
        self.obs_len     = obs_len
        self.sigma_min   = sigma_min
        self.n_train_ens = n_train_ens
        self.net = VelocityField(pred_len, obs_len, sigma_min=sigma_min)

    # ── Coordinate helpers ────────────────────────────────────────────────

    @staticmethod
    def _to_rel(traj_gt, Me_gt, last_pos, last_Me):
        """Absolute normalised → relative offset from last observed."""
        return torch.cat(
            [traj_gt - last_pos.unsqueeze(0),
             Me_gt   - last_Me.unsqueeze(0)],
            dim=-1,
        ).permute(1, 0, 2)    # [B, T, 4]

    @staticmethod
    def _to_abs(rel, last_pos, last_Me):
        """Relative offset → absolute normalised."""
        d = rel.permute(1, 0, 2)   # [T, B, 4]
        return (last_pos.unsqueeze(0) + d[:, :, :2],
                last_Me.unsqueeze(0)  + d[:, :, 2:])

    # ── OT-CFM noise schedule ─────────────────────────────────────────────

    def _cfm_noisy(self, x1: torch.Tensor) -> Tuple:
        """
        Sample interpolant x_t = t·x₁ + (1 − t·(1−σ))·x₀
        and target velocity.
        """
        B, device = x1.shape[0], x1.device
        sm  = self.sigma_min
        x0  = torch.randn_like(x1) * sm
        t   = torch.rand(B, device=device)
        te  = t.view(B, 1, 1)
        x_t = te * x1 + (1.0 - te * (1.0 - sm)) * x0
        denom      = (1.0 - (1.0 - sm) * te).clamp(min=1e-5)
        target_vel = (x1 - (1.0 - sm) * x_t) / denom
        return x_t, t, te, denom, target_vel

    # ── Training forward ──────────────────────────────────────────────────

    def get_loss(self, batch_list: List) -> torch.Tensor:
        return self.get_loss_breakdown(batch_list)["total"]

    def get_loss_breakdown(self, batch_list: List) -> Dict:
        """
        Compute total loss + all component values for logging.

        Returns
        -------
        dict with keys: total, fm, dir, step, disp, heading, smooth, pinn
        """
        traj_gt = batch_list[1]    # [T, B, 2]
        Me_gt   = batch_list[8]
        obs_t   = batch_list[0]
        obs_Me  = batch_list[7]

        lp, lm = obs_t[-1], obs_Me[-1]    # [B, 2]
        x1 = self._to_rel(traj_gt, Me_gt, lp, lm)   # [B, T, 4]

        x_t, t, te, denom, target_vel = self._cfm_noisy(x1)
        pred_vel = self.net(x_t, t, batch_list)       # [B, T, 4]

        # ── Build M ensemble samples for afCRPS (L1) ─────────────────────
        # Each sample: add OT-CFM noise independently, run forward pass
        samples_list = []
        for _ in range(self.n_train_ens):
            x_t_s, t_s, te_s, denom_s, _ = self._cfm_noisy(x1)
            pv_s = self.net(x_t_s, t_s, batch_list)
            x1_s = x_t_s + denom_s * pv_s
            pred_abs_s, _ = self._to_abs(x1_s, lp, lm)
            samples_list.append(pred_abs_s)

        pred_samples = torch.stack(samples_list)   # [M, T, B, 2]

        # Mean prediction for geometric losses
        x1_pred = x_t + denom * pred_vel
        pred_abs, _ = self._to_abs(x1_pred, lp, lm)   # [T, B, 2]

        return compute_total_loss(
            pred_abs     = pred_abs,
            gt           = traj_gt,
            ref          = lp,
            batch_list   = batch_list,
            pred_samples = pred_samples,
            weights      = WEIGHTS,
        )

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        batch_list:   List,
        num_ensemble: int = 50,
        ddim_steps:   int = 10,
        predict_csv:  Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Euler integration of the learned velocity field.

        Args
        ----
        batch_list    : standard batch list
        num_ensemble  : number of independent samples (50 for CRPS evaluation)
        ddim_steps    : number of ODE integration steps
        predict_csv   : if not None, write per-sample predictions to this path

        Returns
        -------
        traj_mean  : [T, B, 2]   ensemble mean trajectory (normalised)
        Me_mean    : [T, B, 2]   ensemble mean intensity
        all_trajs  : [S, T, B, 2]  all individual samples (for CRPS)
        """
        lp  = batch_list[0][-1]    # [B, 2]
        lm  = batch_list[7][-1]
        B, device = lp.shape[0], lp.device
        dt  = 1.0 / ddim_steps

        traj_s = []
        me_s   = []

        for _ in range(num_ensemble):
            x_t = torch.randn(B, self.pred_len, 4, device=device) * self.sigma_min
            for step in range(ddim_steps):
                t_b = torch.full((B,), step * dt, device=device)
                x_t = x_t + dt * self.net(x_t, t_b, batch_list)
                x_t[:, :, :2].clamp_(-5.0, 5.0)
            tr, me = self._to_abs(x_t, lp, lm)
            traj_s.append(tr)
            me_s.append(me)

        all_trajs = torch.stack(traj_s)    # [S, T, B, 2]
        all_me    = torch.stack(me_s)

        traj_mean = all_trajs.mean(0)      # [T, B, 2]
        me_mean   = all_me.mean(0)

        if predict_csv is not None:
            self._write_predict_csv(
                predict_csv, traj_mean, all_trajs,
                batch_list=batch_list,
            )

        return traj_mean, me_mean, all_trajs

    # ── CSV prediction export ──────────────────────────────────────────────

    @staticmethod
    def _write_predict_csv(
        csv_path:   str,
        traj_mean:  torch.Tensor,   # [T, B, 2]  normalised
        all_trajs:  torch.Tensor,   # [S, T, B, 2]  normalised
        batch_list: Optional[List] = None,
    ) -> None:
        """
        Write per-prediction CSV:
        one row per (batch_item, step) with mean + ensemble spread.

        Columns
        -------
        timestamp, batch_idx, step_idx, lead_h,
        lon_mean_deg, lat_mean_deg,
        lon_std_deg,  lat_std_deg,
        ens_spread_km
        """
                
        from utils.metrics import haversine_km
        import numpy as np

        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        T, B, _ = traj_mean.shape
        S       = all_trajs.shape[0]

        # Denorm to degrees
        def _deg(t_norm):
            lon = (t_norm[..., 0] * 5.0 + 180.0).cpu().numpy()
            lat = (t_norm[..., 1] * 5.0).cpu().numpy()
            return lon, lat

        mean_lon, mean_lat = _deg(traj_mean)   # [T, B]
        all_lon  = (all_trajs[..., 0] * 5.0 + 180.0).cpu().numpy()  # [S, T, B]
        all_lat  = (all_trajs[..., 1] * 5.0).cpu().numpy()

        write_header = not os.path.exists(csv_path)
        fields = [
            "timestamp", "batch_idx", "step_idx", "lead_h",
            "lon_mean_deg", "lat_mean_deg",
            "lon_std_deg",  "lat_std_deg",
            "ens_spread_km",
        ]

        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if write_header:
                writer.writeheader()

            for b in range(B):
                for k in range(T):
                    ens_pts_01 = np.stack([
                        all_lon[:, k, b] * 10.0,
                        all_lat[:, k, b] * 10.0,
                    ], axis=-1)    # [S, 2]  0.1° units
                    mean_pt_01 = np.array([
                        mean_lon[k, b] * 10.0,
                        mean_lat[k, b] * 10.0,
                    ])
                    mean_rep = mean_pt_01[np.newaxis].repeat(S, axis=0)
                    spread = float(haversine_km(ens_pts_01, mean_rep).mean())

                    writer.writerow(dict(
                        timestamp     = ts,
                        batch_idx     = b,
                        step_idx      = k,
                        lead_h        = (k + 1) * 6,
                        lon_mean_deg  = f"{mean_lon[k, b]:.4f}",
                        lat_mean_deg  = f"{mean_lat[k, b]:.4f}",
                        lon_std_deg   = f"{all_lon[:, k, b].std():.4f}",
                        lat_std_deg   = f"{all_lat[:, k, b].std():.4f}",
                        ens_spread_km = f"{spread:.2f}",
                    ))

        print(f"  📍  Predictions → {csv_path}  (B={B}, T={T}, S={S})")


# Backward-compatibility alias
TCDiffusion = TCFlowMatching