"""
Semplice: per ogni appliance, un giorno a caso con attivita.
Carica il modello .pt, predice, e plotta segnale originale vs segnale predetto.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq

    from training.tcn_sa import WaveNILMv6STAFN

    checkpoint_dir = ROOT / "apps" / "backend" / "models" / "tcn_sa"
    parquet_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (ROOT / "data" / "processed" / "1sec_new" / "nilm_ready_1sec_new.parquet")
    out_dir = ROOT / "training" / "plots_pt"
    out_dir.mkdir(parents=True, exist_ok=True)

    window = 4096
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stride = 300  # un punto ogni 5 min (plot leggibile, inferenza veloce)

    def app_from_name(name):
        stem = Path(name).stem
        if not stem.startswith("TCN_SA_"):
            return stem
        rest = stem[7:].replace("_best", "")
        return rest.split("_")[0]

    pts = {app_from_name(p.name): p for p in checkpoint_dir.glob("*.pt")}
    if not parquet_path.exists():
        print(f"Parquet non trovato: {parquet_path}")
        return

    schema = pq.read_schema(parquet_path)
    cols = ["Time", "Aggregate"] + [a for a in pts if a in schema.names]
    df = pd.read_parquet(parquet_path, columns=cols)
    df["Time"] = pd.to_datetime(df["Time"]).dt.tz_localize(None)
    df = df.set_index("Time").sort_index()

    for app, pt_path in pts.items():
        if app not in df.columns:
            continue
        gt_all = df[app].astype(float).fillna(0)
        # giorni con attivita: media > 5 W
        daily = gt_all.resample("D").mean()
        active_days = daily[daily > 0.005].index
        if len(active_days) == 0:
            print(f"  {app}: nessun giorno con attivita, skip")
            continue
        day = active_days[np.random.randint(0, len(active_days))]
        day_str = pd.Timestamp(day).strftime("%Y-%m-%d")
        mask = (df.index.date == pd.Timestamp(day).date())
        day_df = df.loc[mask]
        if len(day_df) < window:
            print(f"  {app}: giorno {day_str} troppo corto ({len(day_df)}), skip")
            continue

        # Features per quel giorno (7 o 8: il checkpoint decide)
        t = day_df.index
        h = t.hour + t.minute / 60.0 + t.second / 3600.0
        agg = day_df["Aggregate"].astype(np.float32).values
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        n_in = state["stem.0.conv.weight"].shape[1]
        if n_in == 8:
            X = np.stack([
                np.clip(agg / np.percentile(agg, 95), 0, 2).astype(np.float32) * 2 - 1,
                np.sin(2 * np.pi * h / 24).astype(np.float32),
                np.cos(2 * np.pi * h / 24).astype(np.float32),
                np.sin(2 * np.pi * (t.dayofweek + h / 24) / 7).astype(np.float32),
                np.cos(2 * np.pi * (t.dayofweek + h / 24) / 7).astype(np.float32),
                np.sin(2 * np.pi * (t.month + t.day / 31.0) / 12).astype(np.float32),
                np.cos(2 * np.pi * (t.month + t.day / 31.0) / 12).astype(np.float32),
                np.clip(np.diff(agg, prepend=agg[0]) / 5.0, -1, 1).astype(np.float32),
            ], axis=1)
        else:
            X = np.stack([
                np.clip(agg / np.percentile(agg, 95), 0, 2).astype(np.float32) * 2 - 1,
                np.sin(2 * np.pi * h / 24).astype(np.float32),
                np.cos(2 * np.pi * h / 24).astype(np.float32),
                np.sin(2 * np.pi * (t.dayofweek + h / 24) / 7).astype(np.float32),
                np.cos(2 * np.pi * (t.dayofweek + h / 24) / 7).astype(np.float32),
                np.sin(2 * np.pi * (t.month + t.day / 31.0) / 12).astype(np.float32),
                np.cos(2 * np.pi * (t.month + t.day / 31.0) / 12).astype(np.float32),
            ], axis=1)

        P_MAX = float(ckpt.get("P_MAX", 2.0))
        model = WaveNILMv6STAFN(
            n_features=n_in, n_appliances=1, hidden_channels=64, n_blocks=11,
            use_psa=True, use_inception=True, lookahead=0,
        )
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()

        gt_vals = day_df[app].astype(float).fillna(0).values * 1000  # W
        pred_w = []
        time_ix = []
        for start in range(0, len(X) - window, stride):
            x = torch.from_numpy(X[start : start + window].astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                _, p = model(x, target_timestep=-1)
            pw = float(p.squeeze().cpu().numpy()) * P_MAX * 1000
            pred_w.append(pw)
            time_ix.append(start + window - 1)
        pred_w = np.array(pred_w)
        time_ix = np.array(time_ix)
        t_plot = day_df.index[time_ix]
        gt_plot = gt_vals[time_ix]

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_plot, gt_plot, label="Originale (W)", color="steelblue", alpha=0.9)
        ax.plot(t_plot, pred_w, label="Predetto (W)", color="coral", alpha=0.9, linestyle="--")
        ax.set_title(f"{app} — {day_str} (un giorno con attivita)")
        ax.set_xlabel("Ora")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in app)
        fig.savefig(out_dir / f"originale_vs_predetto_{safe}_{day_str}.png", dpi=150)
        plt.close(fig)
        print(f"  {app}: {day_str} -> originale_vs_predetto_{safe}_{day_str}.png")

    print(f"\nPlot salvati in {out_dir}")


if __name__ == "__main__":
    main()
