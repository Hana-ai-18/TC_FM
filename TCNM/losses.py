"""
TCNM/losses.py  ── v9
========================
Loss functions — unchanged algorithm, updated env key access.

ERA5 wind fields are now under keys 'u500_mean'/'v500_mean' in env_data dict
(matching the actual 90-dim feature vector structure).

All loss formulas identical to v8.
"""
from __future__ import annotations
import math
from typing import Optional

import torch
import torch.nn.functional as F

OMEGA       = 7.2921e-5
R_EARTH     = 6.371e6
DT_6H       = 6 * 3600
NORM_TO_DEG = 5.0
NORM_TO_M   = NORM_TO_DEG * 111_000.0
ERA5_RES_DEG = 0.25
DELTA_DEG    = 0.10
PINN_SCALE   = 100.0

WEIGHTS = dict(fm=1.0, dir=2.0, step=0.5, disp=1.0,
               heading=2.0, smooth=0.2, pinn=0.5)


# ── Haversine ────────────────────────────────────────────────────────────────

def _haversine(p1, p2, unit_01deg=True):
    scale = 10.0 if unit_01deg else 1.0
    lat1  = torch.deg2rad(p1[..., 1] / scale)
    lat2  = torch.deg2rad(p2[..., 1] / scale)
    dlon  = torch.deg2rad((p2[..., 0] - p1[..., 0]) / scale)
    dlat  = torch.deg2rad((p2[..., 1] - p1[..., 1]) / scale)
    a = (torch.sin(dlat / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
    return 2.0 * 6371.0 * torch.asin(a.clamp(0, 1).sqrt())


def _denorm_deg(x):
    out = x.clone()
    out[..., 0] = x[..., 0] * NORM_TO_DEG + 180.0
    out[..., 1] = x[..., 1] * NORM_TO_DEG
    return out


# ── L1 afCRPS ─────────────────────────────────────────────────────────────

def fm_afcrps_loss(pred_samples, gt, unit_01deg=False):
    M, T, B, _ = pred_samples.shape
    if M == 1:
        return _haversine(pred_samples[0], gt, unit_01deg).mean()
    total = gt.new_zeros(())
    n_pairs = 0
    for s in range(M):
        for sp in range(M):
            if s == sp:
                continue
            d_sy  = _haversine(pred_samples[s],  gt,               unit_01deg)
            d_spy = _haversine(pred_samples[sp], gt,               unit_01deg)
            d_ssp = _haversine(pred_samples[s],  pred_samples[sp], unit_01deg)
            total = total + (d_sy + d_spy - d_ssp).clamp(min=0).mean()
            n_pairs += 1
    return total / (2.0 * n_pairs)


# ── L2 overall direction ──────────────────────────────────────────────────

def overall_dir_loss(pred, gt, ref):
    p_disp = pred[-1] - ref
    g_disp = gt[-1]   - ref
    pn = p_disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = g_disp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (1.0 - ((p_disp / pn) * (g_disp / gn)).sum(-1)).mean()


# ── L3 step direction ─────────────────────────────────────────────────────

def step_dir_loss(pred, gt):
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pv = pred[1:] - pred[:-1]
    gv = gt[1:]   - gt[:-1]
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (1.0 - ((pv / pn) * (gv / gn)).sum(-1)).mean()


# ── L4 displacement ───────────────────────────────────────────────────────

def disp_loss(pred, gt):
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pred_disp = (pred[1:] - pred[:-1]).norm(dim=-1).mean(0)
    gt_disp   = (gt[1:]   - gt[:-1]).norm(dim=-1).mean(0)
    return ((pred_disp - gt_disp) ** 2).mean()


# ── L5 heading ────────────────────────────────────────────────────────────

def heading_loss(pred, gt):
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    pv = pred[1:] - pred[:-1]
    gv = gt[1:]   - gt[:-1]
    pn = pv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    gn = gv.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    cos_phi   = ((pv / pn) * (gv / gn)).sum(-1)
    wrong_dir = F.relu(-cos_phi).mean()

    if pred.shape[0] >= 3:
        def _curv(v):
            cross = v[1:,:,0]*v[:-1,:,1] - v[1:,:,1]*v[:-1,:,0]
            n1 = v[1:].norm(dim=-1).clamp(min=1e-6)
            n2 = v[:-1].norm(dim=-1).clamp(min=1e-6)
            return cross / (n1 * n2)
        curv_mse = F.mse_loss(_curv(pv), _curv(gv))
    else:
        curv_mse = pred.new_zeros(())
    return wrong_dir + curv_mse


# ── L6 smoothness ─────────────────────────────────────────────────────────

def smooth_loss(pred):
    if pred.shape[0] < 3:
        return pred.new_zeros(())
    acc = pred[2:] - 2.0 * pred[1:-1] + pred[:-2]
    return (acc ** 2).mean()


# ── L7 PINN BVE ───────────────────────────────────────────────────────────

def _parse_era5_v9(batch_list):
    """
    Extract 500hPa U/V from env_data dict (keys: u500_mean, v500_mean).
    These are scalar per-timestep values — not full 2D fields.
    Falls back to simplified BVE.
    """
    obs_traj = batch_list[0]
    last     = obs_traj[-1]
    c_lon    = last[:, 0] * NORM_TO_DEG + 180.0
    c_lat    = last[:, 1] * NORM_TO_DEG

    env = batch_list[13]
    if isinstance(env, dict):
        uk = next((k for k in ("u500_mean", "u500_center", "u850")
                   if k in env), None)
        vk = next((k for k in ("v500_mean", "v500_center", "v850")
                   if k in env), None)
        if uk and vk:
            u = env[uk].float()
            v = env[vk].float()
            # u, v are [B, T] or [B, T, 1] — use last timestep as scalar field proxy
            if u.dim() == 3:
                u = u[:, -1, 0]
                v = v[:, -1, 0]
            elif u.dim() == 2:
                u = u[:, -1]
                v = v[:, -1]
            return u, v, c_lon, c_lat, True
    return None, None, c_lon, c_lat, False


def _pinn_simplified(pred_abs):
    T = pred_abs.shape[0]
    if T < 4:
        return pred_abs.new_zeros(())
    v   = pred_abs[1:] - pred_abs[:-1]
    vx, vy = v[..., 0], v[..., 1]
    zeta   = vx[1:] * vy[:-1] - vy[1:] * vx[:-1]
    if zeta.shape[0] < 2:
        return pred_abs.new_zeros(())
    dzeta   = zeta[1:] - zeta[:-1]
    lat_rad = pred_abs[2:T-1, :, 1] * NORM_TO_DEG * (math.pi / 180)
    beta_n  = (2.0 * OMEGA * NORM_TO_M * DT_6H / R_EARTH) * torch.cos(lat_rad)
    residual = dzeta + beta_n * vy[1:T-2]
    return (residual ** 2).mean() * PINN_SCALE


def pinn_bve_loss(pred_abs, batch_list):
    """PINN BVE — simplified β-plane (ERA5 fields are scalar, not 2D grids)."""
    # With only scalar u500/v500 from env_data, use simplified BVE
    return _pinn_simplified(pred_abs)


# ── Combined ──────────────────────────────────────────────────────────────

def compute_total_loss(pred_abs, gt, ref, batch_list,
                       pred_samples=None, weights=WEIGHTS):
    if pred_samples is not None:
        l_fm = fm_afcrps_loss(pred_samples, gt, unit_01deg=False)
    else:
        l_fm = _haversine(pred_abs, gt, unit_01deg=False).mean()

    l_dir     = overall_dir_loss(pred_abs, gt, ref)
    l_step    = step_dir_loss(pred_abs, gt)
    l_disp    = disp_loss(pred_abs, gt)
    l_heading = heading_loss(pred_abs, gt)
    l_smooth  = smooth_loss(pred_abs)
    l_pinn    = pinn_bve_loss(pred_abs, batch_list)

    total = (
        weights["fm"]      * l_fm
      + weights["dir"]     * l_dir
      + weights["step"]    * l_step
      + weights["disp"]    * l_disp
      + weights["heading"] * l_heading
      + weights["smooth"]  * l_smooth
      + weights["pinn"]    * l_pinn
    )

    return dict(
        total   = total,
        fm      = l_fm.item(),
        dir     = l_dir.item(),
        step    = l_step.item(),
        disp    = l_disp.item(),
        heading = l_heading.item(),
        smooth  = l_smooth.item(),
        pinn    = l_pinn.item(),
    )


# ── Legacy ───────────────────────────────────────────────────────────────────

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=None):
        super().__init__()
        self.margin  = margin
        self.loss_fn = (torch.nn.SoftMarginLoss() if margin is None
                        else torch.nn.TripletMarginLoss(margin=margin, p=2))

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            y = torch.ones(anchor.shape[0], device=anchor.device)
            return self.loss_fn(
                torch.norm(anchor - neg, 2, dim=1) -
                torch.norm(anchor - pos, 2, dim=1), y)
        return self.loss_fn(anchor, pos, neg)


def bce_loss(input, target):
    neg_abs = -input.abs()
    return (input.clamp(min=0) - input * target
            + (1 + neg_abs.exp()).log()).mean()


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode="average"):
    loss = (loss_mask.unsqueeze(2) *
            (pred_traj_gt.permute(1,0,2) - pred_traj.permute(1,0,2)) ** 2)
    if mode == "sum":     return torch.sum(loss)
    if mode == "average": return torch.sum(loss) / torch.numel(loss_mask.data)
    return loss.sum(dim=2).sum(dim=1)


def toNE(pred_traj, pred_Me):
    rt = pred_traj.clone()
    rm = pred_Me.clone()
    if rt.dim() == 2:
        rt = rt.unsqueeze(1)
        rm = rm.unsqueeze(1)
    rt[:,:,0] = rt[:,:,0] * 50.0 + 1800.0
    rt[:,:,1] = rt[:,:,1] * 50.0
    rm[:,:,0] = rm[:,:,0] * 50.0 + 960.0
    rm[:,:,1] = rm[:,:,1] * 25.0 + 40.0
    return rt, rm


def trajectory_displacement_error(pred, gt, mode="sum"):
    _gt   = gt.permute(1,0,2)
    _pred = pred.permute(1,0,2)
    diff  = _gt - _pred
    lon_km = diff[:,:,0]/10.0*111.0*torch.cos(_gt[:,:,1]/10.0*torch.pi/180.0)
    lat_km = diff[:,:,1]/10.0*111.0
    loss   = torch.sqrt(lon_km**2 + lat_km**2)
    return torch.sum(loss) if mode == "sum" else loss


def value_error(pred, gt, mode="sum"):
    loss = torch.abs(pred.permute(1,0,2) - gt.permute(1,0,2))
    return torch.sum(loss) if mode == "sum" else loss


def evaluate_diffusion_output(best_traj, best_Me, gt_traj, gt_Me):
    rt, rm = toNE(best_traj.clone(), best_Me.clone())
    rg, rgm = toNE(gt_traj.clone(), gt_Me.clone())
    return (trajectory_displacement_error(rt, rg, mode="raw"),
            value_error(rm, rgm, mode="raw"))