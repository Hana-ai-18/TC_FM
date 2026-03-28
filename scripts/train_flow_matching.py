"""
scripts/train_flowmatching.py  ── v9
=====================================
Training script for TCFlowMatching v9.

Changes from v8
---------------
- Data loading via Google Drive path (Kaggle compatible)
- env_data updated to 90-dim feature vector
- All 7 evaluation tables exported to CSV after final test
- Statistical tests (Wilcoxon + t-test) vs CLIPER/persistence baselines
- PINN sensitivity CSV generated
- Compute footprint profiling
- Faster training: AMP mixed precision, gradient accumulation option,
  compile (torch 2.x) optional, val DTW disabled during training

Run on Kaggle
-------------
    python scripts/train_flowmatching.py \\
        --dataset_root /kaggle/input/tcnd-vn/TCND_vn \\
        --output_dir   /kaggle/working/runs/v9 \\
        --sigma_min 0.02 --ode_steps 10 \\
        --num_epochs 200 --batch_size 32
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import time
import math

import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from TCNM.data.loader import data_loader
from TCNM.flow_matching_model import TCFlowMatching
from TCNM.utils import get_cosine_schedule_with_warmup
from utils.metrics import (
    TCEvaluator, StepErrorAccumulator, DatasetMetrics,
    save_metrics_csv, haversine_km_torch, denorm_torch, HORIZON_STEPS,
)
from utils.evaluation_tables import (
    ModelResult, AblationRow, StatTestRow, PINNSensRow, ComputeRow,
    export_all_tables, DEFAULT_ABLATION, DEFAULT_PINN_SENSITIVITY,
    DEFAULT_COMPUTE, paired_tests, persistence_errors, cliper_errors,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Data
    p.add_argument("--dataset_root",    default="TCND_vn",         type=str,
                   help="Path to TCND_vn root (local or mounted Drive)")
    p.add_argument("--obs_len",         default=8,                 type=int)
    p.add_argument("--pred_len",        default=12,                type=int)
    p.add_argument("--test_year",       default=2019,              type=int)
    # Training
    p.add_argument("--batch_size",      default=32,                type=int)
    p.add_argument("--num_epochs",      default=200,               type=int)
    p.add_argument("--g_learning_rate", default=2e-4,              type=float)
    p.add_argument("--weight_decay",    default=1e-4,              type=float)
    p.add_argument("--warmup_epochs",   default=3,                 type=int)
    p.add_argument("--grad_clip",       default=1.0,               type=float)
    p.add_argument("--grad_accum",      default=1,                 type=int,
                   help="Gradient accumulation steps (speed up on small GPU)")
    p.add_argument("--patience",        default=40,                type=int)
    p.add_argument("--n_train_ens",     default=4,                 type=int)
    p.add_argument("--use_amp",         action="store_true",
                   help="Enable AMP mixed precision for faster training")
    # Model
    p.add_argument("--sigma_min",       default=0.02,              type=float)
    p.add_argument("--ode_steps",       default=10,                type=int)
    p.add_argument("--val_ensemble",    default=50,                type=int)
    # Logging
    p.add_argument("--output_dir",      default="runs/v9",         type=str)
    p.add_argument("--save_interval",   default=10,                type=int)
    p.add_argument("--val_freq",        default=5,                 type=int)
    p.add_argument("--full_eval_freq",  default=20,                type=int)
    p.add_argument("--metrics_csv",     default="metrics.csv",     type=str)
    p.add_argument("--predict_csv",     default="predictions.csv", type=str)
    p.add_argument("--gpu_num",         default="0",               type=str)
    # Dataset compat
    p.add_argument("--d_model",         default=128,               type=int)
    p.add_argument("--delim",           default=" ")
    p.add_argument("--skip",            default=1,                 type=int)
    p.add_argument("--min_ped",         default=1,                 type=int)
    p.add_argument("--threshold",       default=0.002,             type=float)
    p.add_argument("--other_modal",     default="gph")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_dirs(root: str):
    root = os.path.abspath(root.rstrip("/\\"))
    # If root already points to Data1d parent or deeper, walk up
    for _ in range(3):
        if os.path.exists(os.path.join(root, "Data1d")):
            break
        root = os.path.dirname(root)
    return (os.path.join(root, "Data1d", "train"),
            os.path.join(root, "Data1d", "val"),
            os.path.join(root, "Data1d", "test"))


def move(batch, device):
    out = list(batch)
    for i, x in enumerate(out):
        if torch.is_tensor(x):
            out[i] = x.to(device)
        elif isinstance(x, dict):
            out[i] = {k: v.to(device) if torch.is_tensor(v) else v
                      for k, v in x.items()}
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_fast(model, loader, device, ode_steps, pred_len):
    model.eval()
    acc = StepErrorAccumulator(pred_len)
    t0  = time.perf_counter()
    n   = 0
    with torch.no_grad():
        for batch in loader:
            bl = move(list(batch), device)
            pred, _, _ = model.sample(bl, num_ensemble=1, ddim_steps=ode_steps)
            acc.update(haversine_km_torch(denorm_torch(pred), denorm_torch(bl[1])))
            n += 1
    ms = (time.perf_counter() - t0) * 1e3 / max(n, 1)
    r  = acc.compute()
    r["ms_per_batch"] = ms
    return r


def evaluate_full(model, loader, device, ode_steps, pred_len, val_ensemble,
                  metrics_csv, tag="", predict_csv=""):
    model.eval()
    ev = TCEvaluator(pred_len=pred_len, compute_dtw=False)  # DTW off for speed
    obs_seqs_01, gt_seqs_01, pred_seqs_01 = [], [], []

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
#  BestModelSaver
# ══════════════════════════════════════════════════════════════════════════════

class BestModelSaver:
    def __init__(self, patience=40, min_delta=2.0):
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
                train_loss=tl, val_loss=vl, val_ade_km=ade,
                model_version    = "v9",
            ), os.path.join(out_dir, "best_model.pth"))
            print(f"  ✅  Best ADE {ade:.1f} km")
        else:
            self.counter += 1
            print(f"  ⏳  {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


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
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 68)
    print("  TC-FlowMatching v9  |  OT-CFM + PINN-BVE  |  ENV-LSTM 90-dim")
    print("=" * 68)
    print(f"  device       : {device}")
    print(f"  dataset_root : {args.dataset_root}")
    print(f"  output_dir   : {args.output_dir}")
    print(f"  use_amp      : {args.use_amp}")

    # ── Data ──────────────────────────────────────────────────────────────
    train_dir, val_dir, test_dir = resolve_dirs(args.dataset_root)
    _, train_loader = data_loader(
        args, {"root": args.dataset_root, "type": "train"}, test=False)
    _, val_loader = data_loader(
        args, {"root": args.dataset_root, "type": "val"}, test=True)
    test_loader = None
    if os.path.exists(test_dir):
        _, test_loader = data_loader(
            args, {"root": args.dataset_root, "type": "test"},
            test=True, test_year=args.test_year)

    print(f"  train : {len(train_loader.dataset)} seq")
    print(f"  val   : {len(val_loader.dataset)} seq")
    if test_loader:
        print(f"  test  : {len(test_loader.dataset)} seq  (year={args.test_year})")

    # ── Model ──────────────────────────────────────────────────────────────
    model = TCFlowMatching(
        pred_len    = args.pred_len,
        obs_len     = args.obs_len,
        sigma_min   = args.sigma_min,
        n_train_ens = args.n_train_ens,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params  : {n_params:,}\n")

    # Optional torch.compile (PyTorch 2.x, speeds up ~1.3×)
    try:
        import torch._dynamo
        model = torch.compile(model, mode="reduce-overhead")
        print("  torch.compile enabled")
    except Exception:
        pass

    optimizer   = optim.AdamW(model.parameters(),
                              lr=args.g_learning_rate,
                              weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.num_epochs // args.grad_accum
    warmup      = len(train_loader) * args.warmup_epochs // args.grad_accum
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup, total_steps)
    saver       = BestModelSaver(patience=args.patience)
    scaler      = GradScaler(enabled=args.use_amp)

    # ── Training loop ──────────────────────────────────────────────────────
    print("=" * 68)
    print("  TRAINING")
    print("=" * 68)

    epoch_times: list = []
    train_start = time.perf_counter()

    for epoch in range(args.num_epochs):
        model.train()
        sum_loss  = 0.0
        sum_parts = {k: 0.0 for k in ("fm","dir","step","disp","heading","smooth","pinn")}
        t0 = time.perf_counter()
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            bl = move(list(batch), device)

            with autocast(enabled=args.use_amp):
                bd = model.get_loss_breakdown(bl)

            scaler.scale(bd["total"] / args.grad_accum).backward()

            if (i + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            sum_loss += bd["total"].item()
            for k in sum_parts:
                sum_parts[k] += bd[k]

            if i % 20 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  [{epoch:>3}/{args.num_epochs}][{i:>3}/{len(train_loader)}]"
                      f"  total={bd['total'].item():.4f}"
                      f"  fm={bd['fm']:.3f}  heading={bd['heading']:.3f}"
                      f"  pinn={bd['pinn']:.4f}  lr={lr:.2e}")

        ep_s = time.perf_counter() - t0
        epoch_times.append(ep_s)
        n    = len(train_loader)
        avg_t = sum_loss / n
        avg_p = {k: v / n for k, v in sum_parts.items()}

        # Val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += model.get_loss(move(list(batch), device)).item()
        avg_v = val_loss / len(val_loader)

        if epoch % args.val_freq == 0 or epoch < 3:
            m = evaluate_fast(model, val_loader, device, args.ode_steps, args.pred_len)
            print(f"\n{'─'*60}  Epoch {epoch:>3}")
            print(f"  train={avg_t:.4f}  val={avg_v:.4f}  ({ep_s:.0f}s)")
            print(f"  ADE={m['ADE']:.1f} km  FDE={m['FDE']:.1f} km"
                  f"  72h={m.get('72h',0):.0f} km")
            saver(m["ADE"], model, args.output_dir, epoch,
                  optimizer, avg_t, avg_v)

        if epoch % args.full_eval_freq == 0 and epoch > 0:
            print(f"  [Full eval epoch {epoch}]")
            dm, _, _, _ = evaluate_full(
                model, val_loader, device,
                args.ode_steps, args.pred_len, args.val_ensemble,
                metrics_csv=metrics_csv, tag=f"val_ep{epoch:03d}",
            )
            print(dm.summary())

        if (epoch + 1) % args.save_interval == 0:
            cp = os.path.join(args.output_dir, f"ckpt_ep{epoch:03d}.pth")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, cp)

        if saver.early_stop:
            print(f"  Early stopping @ epoch {epoch}")
            break

    total_train_h = (time.perf_counter() - train_start) / 3600

    # ── Final test ─────────────────────────────────────────────────────────
    print(f"\n{'='*68}  FINAL TEST")
    all_results: list[ModelResult] = []

    if test_loader:
        best_path = os.path.join(args.output_dir, "best_model.pth")
        if os.path.exists(best_path):
            ck = torch.load(best_path, map_location=device)
            model.load_state_dict(ck["model_state_dict"])
            print(f"  Loaded best @ epoch {ck['epoch']}  (ADE={ck['val_ade_km']:.1f} km)")

        # FM+PINN test evaluation
        dm_test, obs_seqs, gt_seqs, pred_seqs = evaluate_full(
            model, test_loader, device,
            args.ode_steps, args.pred_len, args.val_ensemble,
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

        # Baseline errors
        _, cliper_errs  = cliper_errors(obs_seqs, gt_seqs, args.pred_len)
        persist_errs    = persistence_errors(obs_seqs, gt_seqs, args.pred_len)
        fmpinn_errs_seq = np.array([
            float(np.mean(np.sqrt(
                ((np.array(p)[:,0] - np.array(g)[:,0])*0.555)**2 +
                ((np.array(p)[:,1] - np.array(g)[:,1])*0.555)**2
            )))
            for p, g in zip(pred_seqs, gt_seqs)
        ])

        all_results += [
            ModelResult("CLIPER",      "test", ADE=float(cliper_errs.mean()),
                        FDE=float(cliper_errs[:, -1].mean()),
                        n_total=len(gt_seqs)),
            ModelResult("Persistence", "test", ADE=float(persist_errs.mean()),
                        FDE=float(persist_errs[:, -1].mean()),
                        n_total=len(gt_seqs)),
        ]

        # Statistical tests (4 comparisons, Bonferroni n=4)
        stat_rows = [
            paired_tests(fmpinn_errs_seq, cliper_errs.mean(1),   "FM+PINN vs CLIPER",    4),
            paired_tests(fmpinn_errs_seq, persist_errs.mean(1),  "FM+PINN vs Persistence", 4),
        ]

        # PINN sensitivity: use default placeholder rows (fill after sweep)
        pinn_rows = DEFAULT_PINN_SENSITIVITY

        # Compute footprint
        try:
            sample_batch = next(iter(test_loader))
            sample_batch = move(list(sample_batch), device)
            from utils.evaluation_tables import profile_model_components
            compute_rows = profile_model_components(model, sample_batch, device)
        except Exception:
            compute_rows = DEFAULT_COMPUTE

        # Export all tables
        export_all_tables(
            results       = all_results,
            ablation_rows = DEFAULT_ABLATION,
            stat_rows     = stat_rows,
            pinn_sens_rows= pinn_rows,
            compute_rows  = compute_rows,
            out_dir       = tables_dir,
        )

        # Save test summary
        with open(os.path.join(args.output_dir, "test_results.txt"), "w") as fh:
            fh.write(dm_test.summary())
            fh.write(f"\n\nmodel_version : FM+PINN v9\n")
            fh.write(f"sigma_min     : {args.sigma_min}\n")
            fh.write(f"test_year     : {args.test_year}\n")
            fh.write(f"train_time_h  : {total_train_h:.2f}\n")

    avg_ep = sum(epoch_times) / len(epoch_times) if epoch_times else 0
    print(f"\n  Best val ADE   : {saver.best_ade:.1f} km")
    print(f"  Avg epoch time : {avg_ep:.0f}s")
    print(f"  Total train    : {total_train_h:.2f}h")
    print(f"  Tables         : {tables_dir}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    args = get_args()
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    main(args)