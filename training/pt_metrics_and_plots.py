"""
Extract MAE and F1 per .pt (TCN_SA) model and produce comparison plots.

1) Metrics: scans checkpoint dir for *.pt, loads each, reads 'metrics' (mae, f1)
   from saved checkpoint; appliance from filename. Writes pt_metrics.csv.
   Note: .pt in apps/backend/models/tcn_sa are export-only (state_dict) and have
   no "metrics" key; run training (train_tcn_sa) to get checkpoints with metrics.
2) Bar plot: when MAE/F1 are present, saves metrics_bar.png (MAE and F1 per appliance).
3) Pred vs GT plots: for each appliance runs inference on val slice, picks periods
   with activity, saves plot_<Appliance>_period_1.png; plus plot_all_appliances.png.

Usage:
  python -m training.pt_metrics_and_plots
  python -m training.pt_metrics_and_plots --plots --data path/to/nilm_ready_1sec_new.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_checkpoint_safe(path: Path):
    import torch
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ckpt


def parse_appliance_from_pt_name(name: str) -> str:
    # TCN_SA_HeatPump_best.pt -> HeatPump
    # TCN_SA_Dishwasher_20260126_225619_best.pt -> Dishwasher
    stem = Path(name).stem
    if not stem.startswith("TCN_SA_"):
        return stem
    rest = stem[7:]  # drop "TCN_SA_"
    if "_best" in rest:
        rest = rest.replace("_best", "")
    # rest = "HeatPump" or "Dishwasher_20260126_225619"
    return rest.split("_")[0]


def extract_metrics_from_pt(path: Path) -> dict | None:
    ckpt = _load_checkpoint_safe(path)
    if ckpt is None or not isinstance(ckpt, dict):
        return None
    m = ckpt.get("metrics")
    if m is None:
        return {"mae": None, "f1": None, "mae_on": None, "source": "no_metrics"}
    return {
        "mae": m.get("mae"),
        "f1": m.get("f1"),
        "mae_on": m.get("mae_on"),
        "P_MAX_kw": ckpt.get("P_MAX"),
        "epoch": ckpt.get("epoch"),
    }


def run_metrics(checkpoint_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Scan *.pt in checkpoint_dir, extract MAE/F1, print table and save CSV."""
    pts = list(checkpoint_dir.glob("*.pt"))
    rows = []
    for p in sorted(pts):
        app = parse_appliance_from_pt_name(p.name)
        info = extract_metrics_from_pt(p)
        if info is None:
            rows.append({"appliance": app, "path": p.name, "mae": None, "f1": None, "mae_on": None})
            continue
        rows.append({
            "appliance": app,
            "path": p.name,
            "mae": info.get("mae"),
            "f1": info.get("f1"),
            "mae_on": info.get("mae_on"),
            "P_MAX_kw": info.get("P_MAX_kw"),
            "epoch": info.get("epoch"),
        })
    df = pd.DataFrame(rows)
    print("PT model metrics (MAE in W, F1 in [0,1]):")
    print(df.to_string(index=False))
    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    has_metrics = df["mae"].notna().any() or df["f1"].notna().any()
    if not has_metrics:
        print("(No metrics in checkpoints — backend .pt are state_dict only. Run train_tcn_sa to save checkpoints with metrics.)")
    return df


def run_metrics_bar_plot(df: pd.DataFrame, out_dir: Path) -> None:
    """Bar plot of MAE and F1 per appliance when values exist."""
    if df.empty or (df["mae"].isna().all() and df["f1"].isna().all()):
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apps = df["appliance"].tolist()
    mae = df["mae"].fillna(0).values
    f1 = df["f1"].fillna(0).values
    has_mae = df["mae"].notna()
    has_f1 = df["f1"].notna()
    if not (has_mae.any() or has_f1.any()):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if has_mae.any():
        ax = axes[0]
        m = np.where(has_mae, mae, np.nan)
        ax.bar(apps, m, color="steelblue", alpha=0.8)
        ax.set_title("MAE (W) per appliance")
        ax.set_ylabel("MAE [W]")
        ax.tick_params(axis="x", rotation=45)
    if has_f1.any():
        ax = axes[1]
        f = np.where(has_f1, f1, np.nan)
        ax.bar(apps, f, color="coral", alpha=0.8)
        ax.set_title("F1 per appliance")
        ax.set_ylabel("F1")
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_bar.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'metrics_bar.png'}")


def run_plots(
    checkpoint_dir: Path,
    data_parquet: Path,
    out_dir: Path,
    appliances: list[str] | None = None,
    num_active_days: int = 2,
    max_plots_per_app: int = 2,
) -> None:
    """
    For each appliance with a .pt, run inference on validation slice,
    pick periods with activity, plot pred vs GT per appliance + one combined.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from training.tcn_sa import WaveNILMv6STAFN
    import torch
    import pyarrow.parquet as pq

    # Resolve paths
    checkpoint_dir = Path(checkpoint_dir)
    data_parquet = Path(data_parquet)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = {parse_appliance_from_pt_name(p.name): p for p in checkpoint_dir.glob("*.pt")}
    if appliances:
        pts = {k: v for k, v in pts.items() if k in appliances}
    if not pts:
        print("No .pt files found (or no matching appliances).")
        return

    # Load parquet: Time, Aggregate, and appliance columns we have .pt for
    import pyarrow as pa
    try:
        schema = pq.read_schema(data_parquet)
    except Exception as e:
        print(f"Cannot read parquet: {e}")
        return
    snames = set(schema.names)
    load_cols = ["Time", "Aggregate"] + [a for a in pts if a in snames]
    df = pd.read_parquet(data_parquet, columns=load_cols)
    df["Time"] = pd.to_datetime(df["Time"]).dt.tz_localize(None)
    df = df.set_index("Time").sort_index()

    # Use last 10% of data as "val" for plotting (or user-configurable)
    n = len(df)
    val_start = int(n * 0.9)
    val_df = df.iloc[val_start:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window = 4096  # tutti i modelli TCN_SA
    stride_plot = 3600  # 1 hour step for plot windows

    # Build features: Aggregate + temporal (simplified: hour_sin/cos from index)
    times = val_df.index
    hour = times.hour + times.minute / 60.0 + times.second / 3600.0
    agg = val_df["Aggregate"].astype(np.float32).values
    hour_sin = np.sin(2 * np.pi * hour / 24).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24).astype(np.float32)
    dow = times.dayofweek + hour / 24.0
    dow_sin = np.sin(2 * np.pi * dow / 7).astype(np.float32)
    dow_cos = np.cos(2 * np.pi * dow / 7).astype(np.float32)
    month = times.month + (times.day / 31.0)
    month_sin = np.sin(2 * np.pi * month / 12).astype(np.float32)
    month_cos = np.cos(2 * np.pi * month / 12).astype(np.float32)
    delta_p = np.zeros_like(agg, dtype=np.float32)
    delta_p[1:] = np.diff(agg)
    delta_p = np.clip(delta_p / 5.0, -1, 1)
    # Normalize aggregate by p95
    p95 = np.percentile(agg, 95)
    agg_n = (np.clip(agg / p95, 0, 2) * 2 - 1).astype(np.float32)
    X = np.stack([agg_n, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, delta_p], axis=1)
    # X: (T, 8)

    all_preds = {}
    all_gts = {}
    for app, pt_path in pts.items():
        if app not in val_df.columns:
            continue
        ckpt = _load_checkpoint_safe(pt_path)
        if ckpt is None:
            continue
        P_MAX = float(ckpt.get("P_MAX", 2.0))
        n_features = 8
        model = WaveNILMv6STAFN(
            n_features=n_features,
            n_appliances=1,
            hidden_channels=64,
            n_blocks=11,
            use_psa=True,
            use_inception=True,
            lookahead=0,
        )
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        if state is None and isinstance(ckpt, dict) and "metrics" not in ckpt:
            # Export-only .pt: whole dict is state_dict
            state = ckpt
        if state is not None:
            model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        preds_w = []
        gts_w = []
        gt_vals = val_df[app].astype(np.float32).fillna(0).values
        for start in range(0, len(X) - window, stride_plot):
            end = start + window
            x = X[start:end]
            x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.no_grad():
                gate, power = model(x_t, target_timestep=-1)
            p_w = power.squeeze().cpu().numpy() * P_MAX * 1000
            pw = float(np.asarray(p_w).ravel()[0])
            preds_w.append(pw)
            gts_w.append(float(gt_vals[end - 1]) * 1000)
        all_preds[app] = np.array(preds_w)
        all_gts[app] = np.array(gts_w)

    if not all_preds:
        print("No predictions produced.")
        return

    # Build time index for plot (one point per stride_plot)
    n_win = min(len(v) for v in all_preds.values())
    t_index = val_df.index[(window - 1) + np.arange(n_win) * stride_plot]

    # Per-appliance plots: pick segments with activity (top by sum(gt) over 24h-like bins)
    seg_len = max(1, 24 * 3600 // stride_plot)
    for app in all_preds:
        gt = all_gts[app]
        pred = all_preds[app]
        seg_sums = [np.sum(gt[i : i + seg_len]) for i in range(0, len(gt) - seg_len + 1, seg_len)]
        if not seg_sums:
            seg_sums = [np.sum(gt)]
        order = np.argsort(seg_sums)[::-1]
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in app)
        for plot_i, seg_idx in enumerate(order[:max_plots_per_app]):
            i0 = int(seg_idx) * seg_len
            i1 = min(i0 + seg_len, len(gt), len(t_index))
            i0 = min(i0, i1 - 1)
            if i0 < 0 or i1 <= i0:
                continue
            t_plot = t_index[i0:i1]
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(t_plot, gt[i0:i1], label="Ground truth (W)", color="steelblue", alpha=0.8)
            ax.plot(t_plot, pred[i0:i1], label="Predicted (W)", color="coral", alpha=0.8, linestyle="--")
            ax.set_title(f"{app} — pred vs GT (active period {plot_i+1})")
            ax.set_xlabel("Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / f"plot_{safe}_period_{plot_i+1}.png", dpi=150)
            plt.close(fig)
        print(f"  Saved plot(s) for {app}")

    # Combined plot: all appliances in one figure (overlaid)
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_preds)))
    for (app, c) in zip(all_preds, colors):
        L = min(len(t_index), len(all_preds[app]), len(all_gts[app]))
        ax.plot(t_index[:L], all_gts[app][:L], label=f"{app} GT", color=c, alpha=0.7)
        ax.plot(t_index[:L], all_preds[app][:L], label=f"{app} pred", color=c, linestyle="--", alpha=0.7)
    ax.set_title("All appliances — predicted vs ground truth")
    ax.set_xlabel("Time")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "plot_all_appliances.png", dpi=150)
    plt.close(fig)
    print("  Saved plot_all_appliances.png")


def main():
    ap = argparse.ArgumentParser(description="PT model metrics + pred vs GT plots")
    ap.add_argument("--checkpoint-dir", type=Path, default=ROOT / "apps" / "backend" / "models" / "tcn_sa", help="Dir with .pt files")
    ap.add_argument("--out-csv", type=Path, default=ROOT / "training" / "pt_metrics.csv", help="Output CSV for metrics")
    ap.add_argument("--plots", action="store_true", help="Generate pred vs GT plots")
    ap.add_argument("--data", type=Path, default=None, help="Parquet path for plots (default: data/processed/1sec_new/nilm_ready_1sec_new.parquet)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "training" / "plots_pt", help="Output dir for plots")
    ap.add_argument("--appliances", nargs="*", default=None, help="Restrict to these appliances")
    ap.add_argument("--num-active-days", type=int, default=2)
    ap.add_argument("--max-plots-per-app", type=int, default=2)
    args = ap.parse_args()

    df = run_metrics(args.checkpoint_dir, args.out_csv)

    if args.plots:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        run_metrics_bar_plot(df, args.out_dir)
        data_path = args.data or (ROOT / "data" / "processed" / "1sec_new" / "nilm_ready_1sec_new.parquet")
        if not data_path.exists():
            print(f"Data not found: {data_path}. Pred vs GT plots skipped; set --data to generate.")
        else:
            run_plots(
                checkpoint_dir=args.checkpoint_dir,
                data_parquet=data_path,
                out_dir=args.out_dir,
                appliances=args.appliances,
                num_active_days=args.num_active_days,
                max_plots_per_app=args.max_plots_per_app,
            )


if __name__ == "__main__":
    main()
