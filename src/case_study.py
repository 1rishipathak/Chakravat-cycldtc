# replay the trained forecaster over individual named cyclones

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build, imd_category          # noqa: E402
from features.build import build_dataset, HORIZONS      # noqa: E402
from dataset import load_dataset                        # noqa: E402
from eval.metrics import great_circle_km                # noqa: E402

RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
ART = ROOT / "artifacts"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# chosen to span the failure modes: an extreme RI case, a shear-limited case,
# and a landfalling storm with a sharp recurve.
CASE_STORMS = ["AMPHAN", "MOCHA", "TAUKTAE", "BIPARJOY", "REMAL"]


def storm_frame(data: pd.DataFrame, name: str) -> pd.DataFrame:
    sub = data[data["NAME"].str.upper().str.startswith(name)]
    if sub.empty:
        raise KeyError(f"no storm named {name} in the dataset")
    sid = sub.groupby("SID")["vmax_kt"].max().idxmax()
    return data[data["SID"] == sid].sort_values("ISO_TIME").reset_index(drop=True)


def replay(storm: pd.DataFrame, residual, horizon: int) -> pd.DataFrame:
    # forecast at horizon hours from every synoptic time in the storm
    mask = storm[f"y_dlat_{horizon}"].notna()
    sub = storm[mask]
    if sub.empty:
        return pd.DataFrame()

    pred_lat = sub["LAT"].to_numpy() + residual.predict(sub, f"y_dlat_{horizon}")
    pred_lon = sub["LON"].to_numpy() + residual.predict(sub, f"y_dlon_{horizon}")
    pred_dv = residual.predict(sub, f"y_dv_{horizon}")

    return pd.DataFrame({
        "time": sub["ISO_TIME"].to_numpy(),
        "true_lat": sub[f"true_lat_{horizon}"].to_numpy(),
        "true_lon": sub[f"true_lon_{horizon}"].to_numpy(),
        "pred_lat": pred_lat,
        "pred_lon": pred_lon,
        "err_km": great_circle_km(sub[f"true_lat_{horizon}"], sub[f"true_lon_{horizon}"],
                                  pred_lat, pred_lon),
        "true_vmax": sub["vmax_kt"].to_numpy() + sub[f"y_dv_{horizon}"].to_numpy(),
        "pred_vmax": sub["vmax_kt"].to_numpy() + pred_dv,
    })


def plot_storm(storm: pd.DataFrame, rep: pd.DataFrame, name: str, horizon: int) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot(storm["LON"], storm["LAT"], "-o", color="#0B6E7F", ms=4, lw=2,
             label="Best track", zorder=3)
    for _, r in rep.iterrows():
        ax1.plot([r["true_lon"], r["pred_lon"]], [r["true_lat"], r["pred_lat"]],
                 "-", color="#DD7526", lw=0.8, alpha=0.6, zorder=2)
    ax1.scatter(rep["pred_lon"], rep["pred_lat"], s=22, color="#CE4630",
                label=f"{horizon} h forecast", zorder=4)
    ax1.set_xlabel("Longitude (E)")
    ax1.set_ylabel("Latitude (N)")
    ax1.set_title(f"{name} - track, {horizon} h forecasts")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)

    ax2.plot(storm["ISO_TIME"], storm["vmax_kt"], "-o", color="#0B6E7F", ms=4,
             lw=2, label="Best track intensity")
    ax2.plot(rep["time"] + pd.Timedelta(hours=horizon), rep["pred_vmax"], "--s",
             color="#CE4630", ms=4, lw=1.6, label=f"{horizon} h forecast")
    for thr, lab in [(34, "CS"), (64, "VSCS"), (90, "ESCS"), (120, "SuCS")]:
        ax2.axhline(thr, color="#7C949D", lw=0.7, ls=":")
        ax2.text(storm["ISO_TIME"].iloc[0], thr + 1.5, lab, fontsize=8, color="#4A626C")
    ax2.set_ylabel("Max sustained wind (kt, 3-min)")
    ax2.set_title(f"{name} - intensity")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(alpha=0.25)
    fig.autofmt_xdate()

    fig.suptitle(f"Chakravat replay: {name} ({int(storm['SEASON'].iloc[0])})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    path = FIG / f"case_{name.lower()}_{horizon}h.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    bundle = joblib.load(ART / "prediction_models.joblib")
    residual = bundle["residual"]

    data = load_dataset()

    print("=" * 78)
    print("CASE STUDY REPLAYS - 24 h forecasts, test seasons")
    print("=" * 78)

    rows = []
    for name in CASE_STORMS:
        try:
            storm = storm_frame(data, name)
        except KeyError:
            print(f"  {name}: not found, skipping")
            continue

        rep = replay(storm, residual, 24)
        if rep.empty:
            continue

        peak = storm["vmax_kt"].max()
        rows.append({
            "storm": name,
            "season": int(storm["SEASON"].iloc[0]),
            "peak_kt": peak,
            "peak_category": imd_category(peak),
            "fixes": len(rep),
            "track_mae_km": float(rep["err_km"].mean()),
            "track_max_km": float(rep["err_km"].max()),
            "vmax_mae_kt": float(np.mean(np.abs(rep["pred_vmax"] - rep["true_vmax"]))),
        })
        path = plot_storm(storm, rep, name, 24)
        print(f"  {name:<10} plotted -> {path.name}")

    print()
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    summary.to_csv(ROOT / "reports" / "case_studies.csv", index=False)
    print(f"\nWrote {ROOT / 'reports' / 'case_studies.csv'}")


if __name__ == "__main__":
    main()
