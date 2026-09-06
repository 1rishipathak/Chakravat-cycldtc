# operational model selection and the definitive results table

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build                                # noqa: E402
from features.build import build_dataset, HORIZONS              # noqa: E402
from dataset import load_dataset                              # noqa: E402
from eval.splits import storm_split                             # noqa: E402
from eval import metrics                                        # noqa: E402
from models.baselines import Cliper                             # noqa: E402
from models.sequence import (                                   # noqa: E402
    build_sequences, GruForecaster, SEQ_FEATURES, STATIC_FEATURES, TARGETS,
)

RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
ART = ROOT / "artifacts"
OUT = ROOT / "reports"
N_H = len(HORIZONS)


def gru_predictions(pack: dict, ckpt: dict, device: str) -> np.ndarray:
    # de-normalised GRU outputs for a prepared sequence pack
    seq = ((pack["seq"] - ckpt["seq_mean"]) / ckpt["seq_std"]) * pack["seq_mask"]
    static = (pack["static"] - ckpt["stat_mean"]) / ckpt["stat_std"]

    n_static = ckpt.get("n_static", len(STATIC_FEATURES))
    model = GruForecaster(len(SEQ_FEATURES), n_static, len(TARGETS)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        out = model(
            torch.tensor(seq, dtype=torch.float32, device=device),
            torch.tensor(pack["seq_mask"], dtype=torch.float32, device=device),
            torch.tensor(static, dtype=torch.float32, device=device),
        ).cpu().numpy()
    return out * ckpt["y_std"] + ckpt["y_mean"]


def score_split(rows: pd.DataFrame, pack: dict, pred: np.ndarray,
                residual, cliper) -> dict[tuple[str, int], dict[str, float]]:
    # error for each model on every (task, horizon), on identical rows
    tmask = pack["y_mask"]
    scores: dict[tuple[str, int], dict[str, float]] = {}

    for i, h in enumerate(HORIZONS):
        m = (tmask[:, i] > 0) & (tmask[:, N_H + i] > 0)
        if m.sum() >= 20:
            sub = rows[m]
            tlat, tlon = sub[f"true_lat_{h}"].to_numpy(), sub[f"true_lon_{h}"].to_numpy()
            lat0, lon0 = sub["LAT"].to_numpy(), sub["LON"].to_numpy()
            b_dlat, b_dlon = residual.predict(sub, f"y_dlat_{h}"), residual.predict(sub, f"y_dlon_{h}")
            g_dlat, g_dlon = pred[m, i], pred[m, N_H + i]
            scores[("track", h)] = {
                "n": int(m.sum()),
                "cliper": metrics.track_error(
                    tlat, tlon,
                    lat0 + cliper.predict(sub, f"y_dlat_{h}"),
                    lon0 + cliper.predict(sub, f"y_dlon_{h}"))["mean_km"],
                "boosted": metrics.track_error(tlat, tlon, lat0 + b_dlat, lon0 + b_dlon)["mean_km"],
                "gru": metrics.track_error(tlat, tlon, lat0 + g_dlat, lon0 + g_dlon)["mean_km"],
                # equal-weight blend. Averaging two decorrelated forecasters is
                # more robust than picking one when the validation block is too
                # small to rank them reliably.
                "blend": metrics.track_error(
                    tlat, tlon,
                    lat0 + 0.5 * (b_dlat + g_dlat),
                    lon0 + 0.5 * (b_dlon + g_dlon))["mean_km"],
            }

        col = 2 * N_H + i
        m = tmask[:, col] > 0
        if m.sum() >= 20:
            sub = rows[m]
            truth = pack["y"][m, col]
            b_dv, g_dv = residual.predict(sub, f"y_dv_{h}"), pred[m, col]
            scores[("intensity", h)] = {
                "n": int(m.sum()),
                "cliper": metrics.intensity_error(
                    truth, cliper.predict(sub, f"y_dv_{h}"))["mae_kt"],
                "boosted": metrics.intensity_error(truth, b_dv)["mae_kt"],
                "gru": metrics.intensity_error(truth, g_dv)["mae_kt"],
                "blend": metrics.intensity_error(truth, 0.5 * (b_dv + g_dv))["mae_kt"],
            }
    return scores


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset()
    splits = storm_split(data)

    bundle = joblib.load(ART / "prediction_models.joblib")
    residual, cliper = bundle["residual"], bundle["cliper"]
    ckpt = torch.load(ART / "gru_forecaster.pt", weights_only=False)

    packs, rows, preds, scores = {}, {}, {}, {}
    for name in ("val", "test"):
        packs[name] = build_sequences(splits[name])
        idx = pd.DataFrame(packs[name]["index"], columns=["SID", "ISO_TIME"])
        rows[name] = idx.merge(splits[name], on=["SID", "ISO_TIME"], how="left")
        preds[name] = gru_predictions(packs[name], ckpt, device)
        scores[name] = score_split(rows[name], packs[name], preds[name], residual, cliper)

    print("=" * 78)
    print("MODEL SELECTION - decided on validation seasons 2016-2019")
    print("=" * 78)
    selection: dict[str, str] = {}
    for (task, h), s in sorted(scores["val"].items()):
        # the blend competes as a third candidate. It wins whenever the two
        # base models are close, which is exactly when picking one is a
        # coin-flip on 35 validation storms.
        winner = min(("boosted", "gru", "blend"), key=lambda k: s[k])
        selection[f"{task}_{h}"] = winner
        unit = "km" if task == "track" else "kt"
        print(f"  {task:<9} {h:>2} h   boosted {s['boosted']:>7.2f}   "
              f"gru {s['gru']:>7.2f}   blend {s['blend']:>7.2f} {unit}   -> {winner}")

    print("\n" + "=" * 78)
    print("FINAL RESULTS - test seasons 2020-2025, selection applied unchanged")
    print("=" * 78)

    final = []
    print("\nTrack - mean position error (km)")
    print(f"  {'horizon':>8} {'n':>6} {'CLIPER':>9} {'selected':>10} {'model':>9} {'skill':>8}")
    for h in HORIZONS:
        s = scores["test"].get(("track", h))
        if not s:
            continue
        chosen = selection[f"track_{h}"]
        val, skill = s[chosen], 100 * (1 - s[chosen] / s["cliper"])
        print(f"  {h:>6} h {s['n']:>6} {s['cliper']:>9.1f} {val:>10.1f} "
              f"{chosen:>9} {skill:>7.1f}%")
        final.append({"task": "track", "horizon_h": h, "n": s["n"], "unit": "km",
                      "cliper": s["cliper"], "selected_model": chosen,
                      "selected_error": val, "skill_vs_cliper_pct": skill})

    print("\nIntensity - MAE (kt)")
    print(f"  {'horizon':>8} {'n':>6} {'CLIPER':>9} {'selected':>10} {'model':>9} {'skill':>8}")
    for h in HORIZONS:
        s = scores["test"].get(("intensity", h))
        if not s:
            continue
        chosen = selection[f"intensity_{h}"]
        val, skill = s[chosen], 100 * (1 - s[chosen] / s["cliper"])
        print(f"  {h:>6} h {s['n']:>6} {s['cliper']:>9.2f} {val:>10.2f} "
              f"{chosen:>9} {skill:>7.1f}%")
        final.append({"task": "intensity", "horizon_h": h, "n": s["n"], "unit": "kt",
                      "cliper": s["cliper"], "selected_model": chosen,
                      "selected_error": val, "skill_vs_cliper_pct": skill})

    (OUT / "final_results.json").write_text(
        json.dumps({"selection": selection, "results": final,
                    "val_scores": {f"{k[0]}_{k[1]}": v for k, v in scores["val"].items()},
                    "test_scores": {f"{k[0]}_{k[1]}": v for k, v in scores["test"].items()}},
                   indent=2, default=float)
    )
    print(f"\nWrote {OUT / 'final_results.json'}")


if __name__ == "__main__":
    main()
