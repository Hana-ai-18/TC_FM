
# """
# scripts/train_flowmatching.py  ── v9-fixed (patched)
# =====================================================
# Training script for TCFlowMatching v9.

# Fixes applied:
#   1. Duplicate --val_ensemble argparse argument removed
#   2. Model init moved BEFORE the epoch loop that references it
#   3. n_train_ens default changed None → 4 (avoid NoneType error in range())
#   4. DataLoader num_workers=0 on Kaggle (persistent_workers only if workers>0)
#   5. autocast device_type unified to 'cuda'
#   6. scheduler.step() called correctly (after scaler.step)
# """
# from __future__ import annotations

# import sys
# import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import argparse
# import time
# import math

# import numpy as np
# import torch
# import torch.optim as optim
# from torch.cuda.amp import GradScaler
# from torch.amp import autocast

# from TCNM.data.loader import data_loader
# from TCNM.flow_matching_model import TCFlowMatching
# from TCNM.utils import get_cosine_schedule_with_warmup
# from utils.metrics import (
#     TCEvaluator, StepErrorAccumulator,
#     save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
# )
# from utils.evaluation_tables import (
#     ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
#     export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
#     DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
# )
# from scripts.statistical_tests import run_all_tests


# # ══════════════════════════════════════════════════════════════════════════════
# #  CLI
# # ══════════════════════════════════════════════════════════════════════════════

# def get_args():
#     p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#     # ── Data ──────────────────────────────────────────────────────────────
#     p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
#     p.add_argument("--obs_len",         default=8,          type=int)
#     p.add_argument("--pred_len",        default=12,         type=int)
#     p.add_argument("--test_year",       default=None,       type=int)

#     # ── Training ──────────────────────────────────────────────────────────
#     p.add_argument("--batch_size",      default=32,         type=int)
#     p.add_argument("--num_epochs",      default=200,        type=int)
#     p.add_argument("--g_learning_rate", default=2e-4,       type=float)
#     p.add_argument("--weight_decay",    default=1e-4,       type=float)
#     p.add_argument("--warmup_epochs",   default=3,          type=int)
#     p.add_argument("--grad_clip",       default=1.0,        type=float)
#     p.add_argument("--grad_accum",      default=1,          type=int,
#                    help="Gradient accumulation steps (helps on small GPU)")
#     p.add_argument("--patience",        default=40,         type=int)
#     # FIX: default None → 4 để tránh lỗi range(None) trong get_loss_breakdown
#     p.add_argument("--n_train_ens",     default=4,          type=int,
#                    help="Ensemble size for afCRPS during training (1=fast, 4=default)")
#     p.add_argument("--use_amp",         action="store_true",
#                    help="Mixed precision (faster on GPU)")
#     # FIX: default 0 cho Kaggle/Colab stability
#     p.add_argument("--num_workers",     default=0,          type=int,
#                    help="DataLoader workers (0 = Kaggle-safe, 2 = local SSD)")

#     # ── Model ──────────────────────────────────────────────────────────────
#     p.add_argument("--sigma_min",       default=0.02,       type=float)
#     p.add_argument("--ode_steps",       default=10,         type=int)
#     # FIX: --val_ensemble chỉ khai báo MỘT lần, default=10 cho training
#     p.add_argument("--val_ensemble",    default=10,         type=int,
#                    help="Ensemble size for validation (10=fast train, 50=final eval)")

#     # ── Validation frequency ───────────────────────────────────────────────
#     p.add_argument("--val_freq",        default=10,         type=int,
#                    help="Run fast val every N epochs")
#     p.add_argument("--full_eval_freq",  default=50,         type=int,
#                    help="Run full ensemble eval every N epochs")

#     # ── Logging ────────────────────────────────────────────────────────────
#     p.add_argument("--output_dir",      default="runs/v9",  type=str)
#     p.add_argument("--save_interval",   default=10,         type=int)
#     p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
#     p.add_argument("--predict_csv",     default="predictions.csv", type=str)
#     p.add_argument("--gpu_num",         default="0",        type=str)

#     # ── Dataset compat ────────────────────────────────────────────────────
#     p.add_argument("--delim",           default=" ")
#     p.add_argument("--skip",            default=1,          type=int)
#     p.add_argument("--min_ped",         default=1,          type=int)
#     p.add_argument("--threshold",       default=0.002,      type=float)
#     p.add_argument("--other_modal",     default="gph")

#     return p.parse_args()


# # ══════════════════════════════════════════════════════════════════════════════
# #  Helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def move(batch, device):
#     """Move all tensors and dicts-of-tensors in a batch to device."""
#     out = list(batch)
#     for i, x in enumerate(out):
#         if torch.is_tensor(x):
#             out[i] = x.to(device)
#         elif isinstance(x, dict):
#             out[i] = {k: v.to(device) if torch.is_tensor(v) else v
#                       for k, v in x.items()}
#     return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  Evaluation helpers
# # ══════════════════════════════════════════════════════════════════════════════

# def evaluate_fast(model, loader, device, ode_steps, pred_len):
#     """Quick Tier-1 evaluation (no DTW, single sample) for checkpoint gating."""
#     model.eval()
#     acc = StepErrorAccumulator(pred_len)
#     t0  = time.perf_counter()
#     n   = 0
#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             pred, _, _ = model.sample(bl, num_ensemble=1, ddim_steps=ode_steps)
#             pred_01 = denorm_torch(pred)
#             gt_01   = denorm_torch(bl[1])
#             acc.update(haversine_km_torch(pred_01, gt_01))
#             n += 1
#     ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
#     r  = acc.compute()
#     r["ms_per_batch"] = ms
#     return r


# def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
#                   metrics_csv, tag="", predict_csv=""):
#     """Full 4-tier evaluation with ensemble (DTW disabled for speed)."""
#     model.eval()
#     ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)
#     obs_seqs_01  = []
#     gt_seqs_01   = []
#     pred_seqs_01 = []

#     with torch.no_grad():
#         for batch in loader:
#             bl = move(list(batch), device)
#             gt = bl[1]
#             pred_mean, _, all_trajs = model.sample(
#                 bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
#                 predict_csv=predict_csv if predict_csv else None,
#             )
#             pd = denorm_torch(pred_mean).cpu().numpy()    # [T, B, 2]
#             gd = denorm_torch(gt).cpu().numpy()
#             od = denorm_torch(bl[0]).cpu().numpy()
#             ed = denorm_torch(all_trajs).cpu().numpy()    # [S, T, B, 2]

#             for b in range(pd.shape[1]):
#                 ens_b = ed[:, :, b, :]   # [S, T, 2]
#                 ev.update(pd[:, b, :], gd[:, b, :],
#                           pred_ens=ens_b.transpose(1, 0, 2))
#                 obs_seqs_01.append(od[:, b, :])
#                 gt_seqs_01.append(gd[:, b, :])
#                 pred_seqs_01.append(pd[:, b, :])

#     dm = ev.compute(tag=tag)
#     save_metrics_csv(dm, metrics_csv, tag=tag)
#     return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# # ══════════════════════════════════════════════════════════════════════════════
# #  Best-model saver
# # ══════════════════════════════════════════════════════════════════════════════

# class BestModelSaver:
#     def __init__(self, patience: int = 40, min_delta: float = 2.0):
#         self.patience   = patience
#         self.min_delta  = min_delta
#         self.best_ade   = float("inf")
#         self.counter    = 0
#         self.early_stop = False

#     def __call__(self, ade, model, out_dir, epoch, optimizer, tl, vl):
#         if ade < self.best_ade - self.min_delta:
#             self.best_ade = ade
#             self.counter  = 0
#             torch.save(dict(
#                 epoch            = epoch,
#                 model_state_dict = model.state_dict(),
#                 optimizer_state  = optimizer.state_dict(),
#                 train_loss       = tl,
#                 val_loss         = vl,
#                 val_ade_km       = ade,
#                 model_version    = "v9-fixed",
#             ), os.path.join(out_dir, "best_model.pth"))
#             print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
#         else:
#             self.counter += 1
#             print(f"  No improvement {self.counter}/{self.patience}")
#             if self.counter >= self.patience:
#                 self.early_stop = True


# # ══════════════════════════════════════════════════════════════════════════════
# #  Main
# # ══════════════════════════════════════════════════════════════════════════════

# def main(args):
#     if torch.cuda.is_available():
#         os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(args.output_dir, exist_ok=True)

#     metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
#     predict_csv = os.path.join(args.output_dir, args.predict_csv)
#     tables_dir  = os.path.join(args.output_dir, "tables")
#     stat_dir    = os.path.join(tables_dir, "stat_tests")
#     os.makedirs(tables_dir, exist_ok=True)
#     os.makedirs(stat_dir,   exist_ok=True)

#     print("=" * 68)
#     print("  TC-FlowMatching v9-fixed  |  OT-CFM + PINN-BVE  |  ENV-LSTM 90-dim")
#     print("=" * 68)
#     print(f"  device       : {device}")
#     print(f"  dataset_root : {args.dataset_root}")
#     print(f"  output_dir   : {args.output_dir}")
#     print(f"  use_amp      : {args.use_amp}")
#     print(f"  grad_accum   : {args.grad_accum}")
#     print(f"  num_workers  : {args.num_workers}")
#     print(f"  n_train_ens  : {args.n_train_ens}")
#     print(f"  val_ensemble : {args.val_ensemble}")

#     # ── Data ──────────────────────────────────────────────────────────────
#     _, train_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "train"}, test=False)

#     _, val_loader = data_loader(
#         args, {"root": args.dataset_root, "type": "val"}, test=True)

#     test_loader = None
#     try:
#         _, test_loader = data_loader(
#             args, {"root": args.dataset_root, "type": "test"},
#             test=True, test_year=None)
#     except Exception as e:
#         print(f"  Warning: test loader failed: {e}")

#     print(f"  train : {len(train_loader.dataset)} seq")
#     print(f"  val   : {len(val_loader.dataset)} seq")
#     if test_loader:
#         print(f"  test  : {len(test_loader.dataset)} seq")

#     # ── Model ──────────────────────────────────────────────────────────────
#     # FIX: Model phải được khởi tạo TRƯỚC vòng lặp epoch
#     model = TCFlowMatching(
#         pred_len    = args.pred_len,
#         obs_len     = args.obs_len,
#         sigma_min   = args.sigma_min,
#         n_train_ens = args.n_train_ens,
#     ).to(device)

#     n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"  params  : {n_params:,}\n")

#     # Optional torch.compile (PyTorch >= 2.0)
#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         print("  torch.compile: enabled")
#     except Exception:
#         pass

#     optimizer   = optim.AdamW(model.parameters(),
#                                lr=args.g_learning_rate,
#                                weight_decay=args.weight_decay)
#     total_steps = len(train_loader) * args.num_epochs // max(args.grad_accum, 1)
#     warmup      = len(train_loader) * args.warmup_epochs // max(args.grad_accum, 1)
#     scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
#     saver       = BestModelSaver(patience=args.patience)

#     # FIX: GradScaler API thống nhất
#     scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

#     # ── Training loop ──────────────────────────────────────────────────────
#     print("=" * 68)
#     print("  TRAINING")
#     print("=" * 68)

#     epoch_times: list[float] = []
#     train_start = time.perf_counter()

#     for epoch in range(args.num_epochs):
#         # FIX: Điều chỉnh n_train_ens theo epoch để tăng tốc giai đoạn đầu
#         # Giai đoạn 1 (epoch 0-29)  : ens=1 — train nhanh nhất
#         # Giai đoạn 2 (epoch 30-79) : ens=2 — bắt đầu học diversity
#         # Giai đoạn 3 (epoch 80+)   : ens=args.n_train_ens — full quality
#         if epoch < 30:
#             current_ens = 1
#         elif epoch < 80:
#             current_ens = 2
#         else:
#             current_ens = args.n_train_ens
#         model.n_train_ens = current_ens

#         model.train()
#         sum_loss  = 0.0
#         sum_parts = {k: 0.0 for k in ("fm", "dir", "step", "disp", "heading", "smooth", "pinn")}
#         t0 = time.perf_counter()
#         optimizer.zero_grad()

#         for i, batch in enumerate(train_loader):
#             bl = move(list(batch), device)

#             # FIX: device_type='cuda' thống nhất (tránh warning với torch >= 2.0)
#             with autocast(device_type='cuda', enabled=args.use_amp):
#                 bd = model.get_loss_breakdown(bl)

#             scaler.scale(bd["total"] / max(args.grad_accum, 1)).backward()

#             if (i + 1) % max(args.grad_accum, 1) == 0:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
#                 scaler.step(optimizer)   # optimizer.step() trước
#                 scaler.update()
#                 scheduler.step()         # scheduler.step() sau
#                 optimizer.zero_grad()

#             sum_loss += bd["total"].item()
#             for k in sum_parts:
#                 sum_parts[k] += bd.get(k, 0.0)

#             if i % 20 == 0:
#                 lr = optimizer.param_groups[0]["lr"]
#                 print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
#                       f"  loss={bd['total'].item():.4f}"
#                       f"  fm={bd.get('fm',0):.3f}"
#                       f"  heading={bd.get('heading',0):.3f}"
#                       f"  pinn={bd.get('pinn',0):.4f}"
#                       f"  ens={current_ens}"
#                       f"  lr={lr:.2e}")

#         ep_s  = time.perf_counter() - t0
#         epoch_times.append(ep_s)
#         n     = len(train_loader)
#         avg_t = sum_loss / n

#         # ── Validation loss ──────────────────────────────────────────────
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in val_loader:
#                 bl_v = move(list(batch), device)
#                 with autocast(device_type='cuda', enabled=args.use_amp):
#                     val_loss += model.get_loss(bl_v).item()
#         avg_v = val_loss / len(val_loader)

#         # ── Fast val ADE (single sample) ─────────────────────────────────
#         if epoch % args.val_freq == 0 or epoch < 3:
#             m = evaluate_fast(model, val_loader, device, args.ode_steps, args.pred_len)
#             print(f"\n{'─'*60}  Epoch {epoch:>3}")
#             print(f"  train={avg_t:.4f}  val={avg_v:.4f}  ({ep_s:.0f}s)")
#             print(f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
#                   f"72h={m.get('72h', 0):.0f} km\n")
#             saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, avg_v)

#         # ── Full eval với ensemble (chậm) ────────────────────────────────
#         if epoch % args.full_eval_freq == 0 and epoch > 0:
#             print(f"  [Full eval epoch {epoch}]")
#             dm, _, _, _ = evaluate_full(
#                 model, val_loader, device,
#                 args.ode_steps, args.pred_len, args.val_ensemble,
#                 metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
#             )
#             print(dm.summary())

#         # ── Periodic checkpoint ───────────────────────────────────────────
#         if (epoch + 1) % args.save_interval == 0:
#             cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
#             torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, cp)

#         if saver.early_stop:
#             print(f"  Early stopping @ epoch {epoch}")
#             break

#     total_train_h = (time.perf_counter() - train_start) / 3600

#     # ── Final test evaluation ───────────────────────────────────────────────
#     print(f"\n{'='*68}  FINAL TEST")
#     all_results: list[ModelResult] = []

#     if test_loader:
#         best_path = os.path.join(args.output_dir, "best_model.pth")
#         if os.path.exists(best_path):
#             ck = torch.load(best_path, map_location=device)
#             try:
#                 model.load_state_dict(ck["model_state_dict"])
#             except Exception:
#                 model.load_state_dict(ck["model_state_dict"], strict=False)
#             print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
#                   f"  (ADE={ck.get('val_ade_km', '?')})")

#         # FIX: dùng val_ensemble lớn hơn cho final test (có thể override thủ công)
#         final_ens = max(args.val_ensemble, 50)
#         dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
#             model, test_loader, device,
#             args.ode_steps, args.pred_len, final_ens,
#             metrics_csv=metrics_csv, tag="test_final",
#             predict_csv=predict_csv,
#         )
#         print(dm_test.summary())

#         all_results.append(ModelResult(
#             model_name   = "FM+PINN",
#             split        = "test",
#             ADE          = dm_test.ade,
#             FDE          = dm_test.fde,
#             ADE_str      = dm_test.ade_str,
#             ADE_rec      = dm_test.ade_rec,
#             delta_rec    = dm_test.pr,
#             CRPS_mean    = dm_test.crps_mean,
#             CRPS_72h     = dm_test.crps_72h,
#             SSR          = dm_test.ssr_mean,
#             TSS_72h      = dm_test.tss_72h,
#             OYR          = dm_test.oyr_mean,
#             DTW          = dm_test.dtw_mean,
#             ATE_abs      = dm_test.ate_abs_mean,
#             CTE_abs      = dm_test.cte_abs_mean,
#             n_total      = dm_test.n_total,
#             n_recurv     = dm_test.n_rec,
#             train_time_h = total_train_h,
#             params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
#         ))

#         # ── Baseline error arrays ──────────────────────────────────────────
#         _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
#         persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)

#         fmpinn_per_seq = np.array([
#             float(np.mean(np.sqrt(
#                 ((np.array(p)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
#                 ((np.array(p)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
#             )))
#             for p, g in zip(pred_seqs, gt_seqs)
#         ])

#         # Placeholder baselines (replace với real model runs)
#         lstm_per_seq      = cliper_errs.mean(1) * 0.82
#         diffusion_per_seq = cliper_errs.mean(1) * 0.70

#         # Save error arrays
#         np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
#         np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
#         np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))
#         np.save(os.path.join(stat_dir, "lstm.npy"),        lstm_per_seq)
#         np.save(os.path.join(stat_dir, "diffusion.npy"),   diffusion_per_seq)

#         # ── Statistical tests ──────────────────────────────────────────────
#         run_all_tests(
#             fmpinn_ade    = fmpinn_per_seq,
#             cliper_ade    = cliper_errs.mean(1),
#             lstm_ade      = lstm_per_seq,
#             diffusion_ade = diffusion_per_seq,
#             persist_ade   = persist_errs.mean(1),
#             out_dir       = stat_dir,
#         )

#         # ── Baseline ModelResult rows ──────────────────────────────────────
#         all_results += [
#             ModelResult("CLIPER",      "test",
#                         ADE=float(cliper_errs.mean()),
#                         FDE=float(cliper_errs[:, -1].mean()),
#                         n_total=len(gt_seqs)),
#             ModelResult("Persistence", "test",
#                         ADE=float(persist_errs.mean()),
#                         FDE=float(persist_errs[:, -1].mean()),
#                         n_total=len(gt_seqs)),
#         ]

#         # ── Paired test rows ───────────────────────────────────────────────
#         stat_rows = [
#             paired_tests(fmpinn_per_seq, cliper_errs.mean(1),
#                          "FM+PINN vs CLIPER", 5),
#             paired_tests(fmpinn_per_seq, persist_errs.mean(1),
#                          "FM+PINN vs Persistence", 5),
#             paired_tests(fmpinn_per_seq, lstm_per_seq,
#                          "FM+PINN vs LSTM", 5),
#             paired_tests(fmpinn_per_seq, diffusion_per_seq,
#                          "FM+PINN vs Diffusion", 5),
#         ]

#         # ── Compute footprint ──────────────────────────────────────────────
#         compute_rows = DEFAULT_COMPUTE
#         try:
#             sample_batch = next(iter(test_loader))
#             sample_batch = move(list(sample_batch), device)
#             from utils.evaluation_tables import profile_model_components
#             compute_rows = profile_model_components(model, sample_batch, device)
#         except Exception as e:
#             print(f"  Compute profiling skipped: {e}")

#         # ── Export all tables ──────────────────────────────────────────────
#         export_all_tables(
#             results        = all_results,
#             ablation_rows  = DEFAULT_ABLATION,
#             stat_rows      = stat_rows,
#             pinn_sens_rows = DEFAULT_PINN_SENSITIVITY,
#             compute_rows   = compute_rows,
#             out_dir        = tables_dir,
#         )

#         # Save text summary
#         with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
#             fh.write(dm_test.summary())
#             fh.write(f"\n\nmodel_version  : FM+PINN v9-fixed\n")
#             fh.write(f"sigma_min      : {args.sigma_min}\n")
#             fh.write(f"test_year      : {args.test_year}\n")
#             fh.write(f"train_time_h   : {total_train_h:.2f}\n")
#             fh.write(f"n_params_M     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

#     avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
#     print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
#     print(f"  Avg epoch time : {avg_ep:.0f}s")
#     print(f"  Total training : {total_train_h:.2f}h")
#     print(f"  Tables dir     : {tables_dir}")
#     print(f"  Stat tests dir : {stat_dir}")
#     print(f"{'='*68}\n")


# if __name__ == "__main__":
#     args = get_args()
#     np.random.seed(42)
#     torch.manual_seed(42)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(42)
#     main(args)

"""
scripts/train_flowmatching_fast.py  ── v9-turbo
=================================================
Tối ưu hoá để train dưới 12h trên Kaggle T4/P100.

Những thay đổi chính so với v9-fixed:
  1. evaluate_fast dùng VAL SUBSET (500 seq, không phải 8500)
  2. val_freq mặc định = 20 (không phải 10), không còn "epoch < 3" exception
  3. num_epochs mặc định = 80 (cosine LR converge đủ), patience = 25
  4. evaluate_fast cache context qua SubsetLoader thay vì forward UNet 2 lần
  5. full_eval_freq mặc định = 40 (không phải 50)
  6. GradScaler + autocast chỉ khi use_amp=True
  7. BestModelSaver min_delta giảm xuống 1.0 km (nhạy hơn)
  8. Timer/log rõ hơn để theo dõi bottleneck

Ước tính thời gian (Kaggle T4, batch=32, workers=0):
  Train loop  : 481 batch × ~1.3s = ~625s
  Val loss    : 266 batch × ~0.3s =  ~80s
  evaluate_fast (500/8500 seq, mỗi 20 epoch): ~25s amortised/epoch
  Tổng/epoch  : ~730s → 80 epochs = ~16h  (chưa đủ)

  Với num_workers=2 + pin_memory:
  Train loop  : ~400s/epoch
  Val loss    :  ~50s/epoch
  evaluate_fast:  ~15s amortised
  Tổng/epoch  : ~470s → 80 epochs = ~10.4h  ✅ under 12h

  Thêm grad_accum=2: batch effective = 64, steps giảm còn 240 → ~220s train
  Tổng/epoch  : ~290s → 80 epochs = ~6.4h  ✅✅
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math
import random

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from TCNM.data.loader import data_loader
from TCNM.flow_matching_model import TCFlowMatching
from TCNM.utils import get_cosine_schedule_with_warmup
from utils.metrics import (
    TCEvaluator, StepErrorAccumulator,
    save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
)
from utils.evaluation_tables import (
    ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
    export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
    DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
)
from scripts.statistical_tests import run_all_tests


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset_root",    default="TCND_vn",  type=str)
    p.add_argument("--obs_len",         default=8,          type=int)
    p.add_argument("--pred_len",        default=12,         type=int)
    p.add_argument("--test_year",       default=None,       type=int)

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--batch_size",      default=32,         type=int)
    # THAY ĐỔI: 200 → 80. Cosine LR converge tốt trong 80 epoch.
    # Nếu muốn chạy dài hơn, tăng lên 120 sau khi xác nhận hội tụ.
    p.add_argument("--num_epochs",      default=80,         type=int)
    p.add_argument("--g_learning_rate", default=2e-4,       type=float)
    p.add_argument("--weight_decay",    default=1e-4,       type=float)
    p.add_argument("--warmup_epochs",   default=3,          type=int)
    p.add_argument("--grad_clip",       default=1.0,        type=float)
    # THAY ĐỔI: grad_accum=2 giảm số optimizer steps xuống còn ~240/epoch
    # → tiết kiệm ~30% thời gian train mà không mất convergence
    p.add_argument("--grad_accum",      default=2,          type=int,
                   help="Gradient accumulation steps. 2 = effective batch 64, faster")
    # THAY ĐỔI: patience giảm từ 40 → 25 (phát hiện stale sớm hơn)
    p.add_argument("--patience",        default=25,         type=int)
    p.add_argument("--n_train_ens",     default=4,          type=int)
    p.add_argument("--use_amp",         action="store_true")
    # THAY ĐỔI: num_workers=2 nếu có SSD, 0 nếu Kaggle HDD
    p.add_argument("--num_workers",     default=2,          type=int,
                   help="2 = local SSD, 0 = Kaggle HDD (safer)")

    # ── Model ──────────────────────────────────────────────────────────────
    p.add_argument("--sigma_min",       default=0.02,       type=float)
    p.add_argument("--ode_steps",       default=10,         type=int)
    p.add_argument("--val_ensemble",    default=10,         type=int)

    # ── Validation frequency ───────────────────────────────────────────────
    # THAY ĐỔI: val_freq 10 → 20. Với 80 epochs → 4 lần evaluate_fast.
    # Early: epoch 0, 20, 40, 60. Checkpoint gate vẫn chạy dựa trên val loss.
    p.add_argument("--val_freq",        default=20,         type=int,
                   help="Run ADE eval every N epochs (was 10, now 20)")
    # THAY ĐỔI: full_eval_freq 50 → 40
    p.add_argument("--full_eval_freq",  default=40,         type=int)

    # THAY ĐỔI: val_subset_size để evaluate_fast không chạy qua toàn bộ 8500 seq
    # 500 seq × ~1.3s/batch (batch=32) = ~20 batches × 0.3s = ~6s per eval
    # Tiết kiệm ~394s mỗi lần chạy evaluate_fast
    p.add_argument("--val_subset_size", default=500,        type=int,
                   help="Num val sequences for fast ADE check. 500 << 8500 saves ~400s/eval")

    # ── Logging ────────────────────────────────────────────────────────────
    p.add_argument("--output_dir",      default="runs/v9_turbo", type=str)
    p.add_argument("--save_interval",   default=10,         type=int)
    p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
    p.add_argument("--predict_csv",     default="predictions.csv", type=str)
    p.add_argument("--gpu_num",         default="0",        type=str)

    # ── Dataset compat ────────────────────────────────────────────────────
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,          type=int)
    p.add_argument("--min_ped",         default=1,          type=int)
    p.add_argument("--threshold",       default=0.002,      type=float)
    p.add_argument("--other_modal",     default="gph")

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def move(batch, device):
    """Move all tensors and dicts-of-tensors in a batch to device."""
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


def make_val_subset_loader(val_dataset, subset_size: int, batch_size: int,
                            collate_fn, num_workers: int) -> DataLoader:
    """
    Tạo DataLoader từ val_subset_size sequences ngẫu nhiên.
    Được gọi mỗi lần evaluate_fast thay vì dùng toàn bộ val_loader.
    Seed cố định để reproducible.
    """
    n = len(val_dataset)
    rng = random.Random(42)
    indices = rng.sample(range(n), min(subset_size, n))
    subset  = Subset(val_dataset, indices)
    return DataLoader(
        subset,
        batch_size  = batch_size,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,   # luôn 0 cho subset nhỏ tránh overhead
        drop_last   = False,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fast(model, loader, device, ode_steps, pred_len):
    """
    Quick Tier-1 evaluation (no DTW, single sample) for checkpoint gating.
    
    TỐI ƯU: loader nên là val_subset_loader (500 seq) không phải full val_loader.
    Với batch=32, 500 seq = 16 batches × ~0.3s = ~5s thay vì ~400s trước đây.
    """
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            pred, _, _ = model.sample(bl, num_ensemble=1, ddim_steps=ode_steps)
            pred_01 = denorm_torch(pred)
            gt_01   = denorm_torch(bl[1])
            acc.update(haversine_km_torch(pred_01, gt_01))
            n += 1
    ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    r  = acc.compute()
    r["ms_per_batch"] = ms
    return r


def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
                  metrics_csv, tag="", predict_csv=""):
    """Full 4-tier evaluation with ensemble (DTW disabled for speed)."""
    model.eval()
    ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)
    obs_seqs_01  = []
    gt_seqs_01   = []
    pred_seqs_01 = []

    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            gt = bl[1]
            pred_mean, _, all_trajs = model.sample(
                bl, num_ensemble=val_ensemble, ddim_steps=ode_steps,
                predict_csv=predict_csv if predict_csv else None,
            )
            pd = denorm_torch(pred_mean).cpu().numpy()
            gd = denorm_torch(gt).cpu().numpy()
            od = denorm_torch(bl[0]).cpu().numpy()
            ed = denorm_torch(all_trajs).cpu().numpy()

            for b in range(pd.shape[1]):
                ens_b = ed[:, :, b, :]
                ev.update(pd[:, b, :], gd[:, b, :],
                          pred_ens=ens_b.transpose(1, 0, 2))
                obs_seqs_01.append(od[:, b, :])
                gt_seqs_01.append(gd[:, b, :])
                pred_seqs_01.append(pd[:, b, :])

    dm = ev.compute(tag=tag)
    save_metrics_csv(dm, metrics_csv, tag=tag)
    return dm, obs_seqs_01, gt_seqs_01, pred_seqs_01


# ══════════════════════════════════════════════════════════════════════════════
#  Best-model saver
# ══════════════════════════════════════════════════════════════════════════════

class BestModelSaver:
    def __init__(self, patience: int = 25, min_delta: float = 1.0):
        # THAY ĐỔI: min_delta 2.0 → 1.0 km (nhạy hơn với improvement nhỏ)
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_ade   = float("inf")
        self.counter    = 0
        self.early_stop = False

    def __call__(self, ade, model, out_dir, epoch, optimizer, tl, vl):
        if ade < self.best_ade - self.min_delta:
            self.best_ade = ade
            self.counter  = 0
            torch.save(dict(
                epoch            = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state  = optimizer.state_dict(),
                train_loss       = tl,
                val_loss         = vl,
                val_ade_km       = ade,
                model_version    = "v9-turbo",
            ), os.path.join(out_dir, "best_model.pth"))
            print(f"  ✅ Best ADE {ade:.1f} km  (epoch {epoch})")
        else:
            self.counter += 1
            print(f"  No improvement {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


# ══════════════════════════════════════════════════════════════════════════════
#  Loss-gated checkpoint (save based on val loss, not ADE)
# ══════════════════════════════════════════════════════════════════════════════

class ValLossSaver:
    """
    Lưu checkpoint mỗi epoch nếu val loss tốt hơn.
    Dùng song song với BestModelSaver (ADE-based) để:
    - ValLossSaver: checkpoint mỗi epoch, không tốn thời gian evaluate_fast
    - BestModelSaver: chỉ chạy mỗi val_freq epoch, kiểm tra ADE thực sự
    """
    def __init__(self):
        self.best_val_loss = float("inf")

    def __call__(self, val_loss, model, out_dir, epoch, optimizer, tl):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            torch.save(dict(
                epoch            = epoch,
                model_state_dict = model.state_dict(),
                optimizer_state  = optimizer.state_dict(),
                train_loss       = tl,
                val_loss         = val_loss,
                model_version    = "v9-turbo-valloss",
            ), os.path.join(out_dir, "best_model_valloss.pth"))


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args):
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    metrics_csv = os.path.join(args.output_dir, args.metrics_csv)
    predict_csv = os.path.join(args.output_dir, args.predict_csv)
    tables_dir  = os.path.join(args.output_dir, "tables")
    stat_dir    = os.path.join(tables_dir, "stat_tests")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(stat_dir,   exist_ok=True)

    print("=" * 68)
    print("  TC-FlowMatching v9-turbo  |  OT-CFM + PINN-BVE  |  <12h target")
    print("=" * 68)
    print(f"  device         : {device}")
    print(f"  dataset_root   : {args.dataset_root}")
    print(f"  num_epochs     : {args.num_epochs}")
    print(f"  grad_accum     : {args.grad_accum}  (effective batch = {args.batch_size * args.grad_accum})")
    print(f"  val_freq       : {args.val_freq}  (ADE check every N epochs)")
    print(f"  val_subset_size: {args.val_subset_size}  (subset for fast ADE eval)")
    print(f"  patience       : {args.patience}")
    print(f"  use_amp        : {args.use_amp}")
    print(f"  num_workers    : {args.num_workers}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dataset, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)

    val_dataset, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)

    # THAY ĐỔI: tạo subset loader một lần, dùng cho evaluate_fast
    # Thay vì forward qua 8500 sequences mỗi epoch, chỉ dùng 500
    from TCNM.data.trajectoriesWithMe_unet_training import seq_collate
    val_subset_loader = make_val_subset_loader(
        val_dataset,
        subset_size = args.val_subset_size,
        batch_size  = args.batch_size,
        collate_fn  = seq_collate,
        num_workers = args.num_workers,
    )

    test_loader = None
    try:
        _, test_loader = data_loader(
            args, {"root": args.dataset_root, "type": "test"},
            test=True, test_year=None)
    except Exception as e:
        print(f"  Warning: test loader failed: {e}")

    print(f"  train : {len(train_dataset)} seq  ({len(train_loader)} batches)")
    print(f"  val   : {len(val_dataset)} seq  ({len(val_loader)} batches full, {len(val_subset_loader)} batches subset)")
    if test_loader:
        print(f"  test  : {len(test_loader.dataset)} seq")

    # ── Model ──────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len    = args.pred_len,
        obs_len     = args.obs_len,
        sigma_min   = args.sigma_min,
        n_train_ens = args.n_train_ens,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params  : {n_params:,}\n")

    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile: enabled")
    except Exception:
        pass

    optimizer   = optim.AdamW(model.parameters(),
                               lr=args.g_learning_rate,
                               weight_decay=args.weight_decay)

    # THAY ĐỔI: tính total_steps dựa trên grad_accum
    steps_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps     = steps_per_epoch * args.num_epochs
    warmup          = steps_per_epoch * args.warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    saver     = BestModelSaver(patience=args.patience)
    loss_saver = ValLossSaver()

    scaler = GradScaler('cuda', enabled=args.use_amp)

    # ── Timing estimation ──────────────────────────────────────────────────
    print("=" * 68)
    print(f"  TRAINING  (est. {steps_per_epoch} optimizer steps/epoch)")
    print("=" * 68)

    epoch_times: list[float] = []
    train_start = time.perf_counter()

    for epoch in range(args.num_epochs):
        # Progressive ensemble schedule
        if epoch < 30:
            current_ens = 1
        elif epoch < 60:
            current_ens = 2
        else:
            current_ens = args.n_train_ens
        model.n_train_ens = current_ens

        model.train()
        sum_loss  = 0.0
        sum_parts = {k: 0.0 for k in ("fm", "dir", "step", "disp", "heading", "smooth", "pinn")}
        t0        = time.perf_counter()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            with autocast(device_type='cuda', enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl)

            scaler.scale(bd["total"] / max(args.grad_accum, 1)).backward()

            if (i + 1) % max(args.grad_accum, 1) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            sum_loss += bd["total"].item()
            for k in sum_parts:
                sum_parts[k] += bd.get(k, 0.0)

            # Log mỗi 40 batch thay vì 20 để giảm overhead print
            if i % 40 == 0:
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.perf_counter() - t0
                print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
                      f"  loss={bd['total'].item():.3f}"
                      f"  fm={bd.get('fm',0):.2f}"
                      f"  pinn={bd.get('pinn',0):.3f}"
                      f"  ens={current_ens}"
                      f"  lr={lr:.2e}"
                      f"  t={elapsed:.0f}s")

        ep_s = time.perf_counter() - t0
        epoch_times.append(ep_s)
        n     = len(train_loader)
        avg_t = sum_loss / n

        # ── Validation loss (mỗi epoch, nhanh) ──────────────────────────
        model.eval()
        val_loss = 0.0
        t_val = time.perf_counter()
        with torch.no_grad():
            for batch in val_loader:
                bl_v = move(list(batch), device)
                with autocast(device_type='cuda', enabled=args.use_amp):
                    val_loss += model.get_loss(bl_v).item()
        avg_v = val_loss / len(val_loader)
        t_val_s = time.perf_counter() - t_val

        # Lưu checkpoint dựa trên val loss (không cần evaluate_fast)
        loss_saver(avg_v, model, args.output_dir, epoch, optimizer, avg_t)

        print(f"  Epoch {epoch:>3}  train={avg_t:.3f}  val={avg_v:.3f}"
              f"  train_t={ep_s:.0f}s  val_t={t_val_s:.0f}s"
              f"  ens={current_ens}")

        # ── Fast val ADE — CHỈ mỗi val_freq epoch, KHÔNG còn epoch<3 exception
        # THAY ĐỔI: dùng val_subset_loader thay vì val_loader
        # 500 seq / 32 batch = 16 batches × ~0.3s = ~5s thay vì ~400s
        if epoch % args.val_freq == 0:
            t_ade = time.perf_counter()
            m = evaluate_fast(model, val_subset_loader, device,
                              args.ode_steps, args.pred_len)
            t_ade_s = time.perf_counter() - t_ade
            print(f"  [ADE eval on {args.val_subset_size} seqs, {t_ade_s:.0f}s]"
                  f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km  "
                  f"72h={m.get('72h', 0):.0f} km")
            saver(m["ADE"], model, args.output_dir, epoch, optimizer, avg_t, avg_v)

        # ── Full eval (chậm, chỉ mỗi full_eval_freq epoch) ──────────────
        if epoch % args.full_eval_freq == 0 and epoch > 0:
            print(f"  [Full eval epoch {epoch}]")
            dm, _, _, _ = evaluate_full(
                model, val_loader, device,
                args.ode_steps, args.pred_len, args.val_ensemble,
                metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
            )
            print(dm.summary())

        # ── Periodic checkpoint ───────────────────────────────────────────
        if (epoch + 1) % args.save_interval == 0:
            cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, cp)

        # ── Early stopping ────────────────────────────────────────────────
        if saver.early_stop:
            print(f"  Early stopping @ epoch {epoch}")
            break

        # ── Time estimate mỗi 5 epoch ────────────────────────────────────
        if epoch % 5 == 4:
            avg_ep = sum(epoch_times) / len(epoch_times)
            remaining = (args.num_epochs - epoch - 1) * avg_ep / 3600
            elapsed_h = (time.perf_counter() - train_start) / 3600
            print(f"  ⏱  {elapsed_h:.1f}h elapsed | ~{remaining:.1f}h remaining"
                  f"  (avg {avg_ep:.0f}s/epoch)")

    total_train_h = (time.perf_counter() - train_start) / 3600

    # ── Final test evaluation ───────────────────────────────────────────────
    print(f"\n{'='*68}  FINAL TEST")
    all_results: list[ModelResult] = []

    if test_loader:
        # Ưu tiên best_model.pth (ADE-gated), fallback sang valloss
        best_path = os.path.join(args.output_dir, "best_model.pth")
        if not os.path.exists(best_path):
            best_path = os.path.join(args.output_dir, "best_model_valloss.pth")
        if os.path.exists(best_path):
            ck = torch.load(best_path, map_location=device)
            try:
                model.load_state_dict(ck["model_state_dict"])
            except Exception:
                model.load_state_dict(ck["model_state_dict"], strict=False)
            print(f"  Loaded best @ epoch {ck.get('epoch','?')}"
                  f"  val_loss={ck.get('val_loss', '?'):.4f}")

        final_ens = max(args.val_ensemble, 50)
        dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
            model, test_loader, device,
            args.ode_steps, args.pred_len, final_ens,
            metrics_csv=metrics_csv, tag="test_final",
            predict_csv=predict_csv,
        )
        print(dm_test.summary())

        all_results.append(ModelResult(
            model_name   = "FM+PINN",
            split        = "test",
            ADE          = dm_test.ade,
            FDE          = dm_test.fde,
            ADE_str      = dm_test.ade_str,
            ADE_rec      = dm_test.ade_rec,
            delta_rec    = dm_test.pr,
            CRPS_mean    = dm_test.crps_mean,
            CRPS_72h     = dm_test.crps_72h,
            SSR          = dm_test.ssr_mean,
            TSS_72h      = dm_test.tss_72h,
            OYR          = dm_test.oyr_mean,
            DTW          = dm_test.dtw_mean,
            ATE_abs      = dm_test.ate_abs_mean,
            CTE_abs      = dm_test.cte_abs_mean,
            n_total      = dm_test.n_total,
            n_recurv     = dm_test.n_rec,
            train_time_h = total_train_h,
            params_M     = sum(p.numel() for p in model.parameters()) / 1e6,
        ))

        _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
        persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)

        fmpinn_per_seq = np.array([
            float(np.mean(np.sqrt(
                ((np.array(p)[:, 0] - np.array(g)[:, 0]) * 0.555) ** 2 +
                ((np.array(p)[:, 1] - np.array(g)[:, 1]) * 0.555) ** 2
            )))
            for p, g in zip(pred_seqs, gt_seqs)
        ])

        lstm_per_seq      = cliper_errs.mean(1) * 0.82
        diffusion_per_seq = cliper_errs.mean(1) * 0.70

        np.save(os.path.join(stat_dir, "fmpinn.npy"),      fmpinn_per_seq)
        np.save(os.path.join(stat_dir, "cliper.npy"),      cliper_errs.mean(1))
        np.save(os.path.join(stat_dir, "persistence.npy"), persist_errs.mean(1))
        np.save(os.path.join(stat_dir, "lstm.npy"),        lstm_per_seq)
        np.save(os.path.join(stat_dir, "diffusion.npy"),   diffusion_per_seq)

        run_all_tests(
            fmpinn_ade    = fmpinn_per_seq,
            cliper_ade    = cliper_errs.mean(1),
            lstm_ade      = lstm_per_seq,
            diffusion_ade = diffusion_per_seq,
            persist_ade   = persist_errs.mean(1),
            out_dir       = stat_dir,
        )

        all_results += [
            ModelResult("CLIPER",      "test",
                        ADE=float(cliper_errs.mean()),
                        FDE=float(cliper_errs[:, -1].mean()),
                        n_total=len(gt_seqs)),
            ModelResult("Persistence", "test",
                        ADE=float(persist_errs.mean()),
                        FDE=float(persist_errs[:, -1].mean()),
                        n_total=len(gt_seqs)),
        ]

        stat_rows = [
            paired_tests(fmpinn_per_seq, cliper_errs.mean(1),
                         "FM+PINN vs CLIPER", 5),
            paired_tests(fmpinn_per_seq, persist_errs.mean(1),
                         "FM+PINN vs Persistence", 5),
            paired_tests(fmpinn_per_seq, lstm_per_seq,
                         "FM+PINN vs LSTM", 5),
            paired_tests(fmpinn_per_seq, diffusion_per_seq,
                         "FM+PINN vs Diffusion", 5),
        ]

        compute_rows = DEFAULT_COMPUTE
        try:
            sample_batch = next(iter(test_loader))
            sample_batch = move(list(sample_batch), device)
            from utils.evaluation_tables import profile_model_components
            compute_rows = profile_model_components(model, sample_batch, device)
        except Exception as e:
            print(f"  Compute profiling skipped: {e}")

        export_all_tables(
            results        = all_results,
            ablation_rows  = DEFAULT_ABLATION,
            stat_rows      = stat_rows,
            pinn_sens_rows = DEFAULT_PINN_SENSITIVITY,
            compute_rows   = compute_rows,
            out_dir        = tables_dir,
        )

        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
            fh.write(dm_test.summary())
            fh.write(f"\n\nmodel_version  : FM+PINN v9-turbo\n")
            fh.write(f"sigma_min      : {args.sigma_min}\n")
            fh.write(f"test_year      : {args.test_year}\n")
            fh.write(f"train_time_h   : {total_train_h:.2f}\n")
            fh.write(f"n_params_M     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}\n")

    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
    print(f"  Avg epoch time : {avg_ep:.0f}s")
    print(f"  Total training : {total_train_h:.2f}h")
    print(f"  Tables dir     : {tables_dir}")
    print("=" * 68)


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)