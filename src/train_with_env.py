# A/B test: does ERA5 environment actually fix the track error?

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build                                # noqa: E402
from features.build import build_dataset, HORIZONS              # noqa: E402
from features.environment import attach, ENV_FEATURES           # noqa: E402
from eval.splits import storm_split, describe                   # noqa: E402
from eval import metrics                                        # noqa: E402
from models.baselines import Cliper                             # noqa: E402
from models.residual import ResidualForecaster                  # noqa: E402

RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
ERA5 = ROOT / "data" / "era5"
CACHE = ROOT / "data" / "processed" / "dataset_with_env.pkl"
OUT = ROOT / "reports"

TRACK_TARGETS = [f"y_dlat_{h}" for h in HORIZONS] + [f"y_dlon_{h}" for h in HORIZONS]
INTENSITY_TARGETS = [f"y_dv_{h}" for h in HORIZONS]
ALL_TARGETS = TRACK_TARGETS + INTENSITY_TARGETS


def load_dataset(refresh: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        print(f"loading cached dataset  {CACHE.name}")
        return pd.read_pickle(CACHE)

    data = build_dataset(build(str(RAW)))
    n_years = len(list(ERA5.glob("era5_pl_*.nc")))
    if n_years == 0:
        raise SystemExit(
            "No ERA5 files in data/era5.\n"
            "Accept the two licences, then run:\n"
            "  python src/ingest/era5.py"
        )
    print(f"attaching ERA5 environment from {n_years} seasons ...")
    data = attach(data, ERA5)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    data.to_pickle(CACHE)
    print(f"cached -> {CACHE}")
    return data


def evaluate(test: pd.DataFrame, model, cliper) -> dict[tuple[str, int], float]:
    scores: dict[tuple[str, int], float] = {}
    for h in HORIZONS:
        m = test[f"y_dlat_{h}"].notna() & test[f"y_dlon_{h}"].notna()
        sub = test[m]
        if len(sub) >= 20:
            scores[("track", h)] = metrics.track_error(
                sub[f"true_lat_{h}"], sub[f"true_lon_{h}"],
                sub["LAT"] + model.predict(sub, f"y_dlat_{h}"),
                sub["LON"] + model.predict(sub, f"y_dlon_{h}"),
            )["mean_km"]
        sub = test[test[f"y_dv_{h}"].notna()]
        if len(sub) >= 20:
            scores[("intensity", h)] = metrics.intensity_error(
                sub[f"y_dv_{h}"], model.predict(sub, f"y_dv_{h}")
            )["mae_kt"]
    return scores


def main() -> None:
    data = load_dataset(refresh="--refresh" in sys.argv)

    have = [c for c in ENV_FEATURES if c in data.columns]
    coverage = data[have].notna().all(axis=1).mean() if have else 0.0
    print(f"\nenvironmental predictors present: {len(have)}/{len(ENV_FEATURES)}")
    print(f"rows with complete environment:   {100 * coverage:.1f}%")

    splits = storm_split(data)
    print("\n" + describe(splits))
    train, test = splits["train"], splits["test"]

    print("\ntraining WITHOUT environment ...")
    plain = data.drop(columns=have)
    p_splits = storm_split(plain)
    cliper_p = Cliper().fit(p_splits["train"], ALL_TARGETS)
    model_p = ResidualForecaster().fit(p_splits["train"], ALL_TARGETS)
    before = evaluate(p_splits["test"], model_p, cliper_p)

    print("training WITH environment ...")
    cliper_e = Cliper().fit(train, ALL_TARGETS)
    model_e = ResidualForecaster().fit(train, ALL_TARGETS)
    after = evaluate(test, model_e, cliper_e)

    print("\n" + "=" * 78)
    print("ERA5 ABLATION - test seasons 2020-2025")
    print("=" * 78)
    rows = []
    for task, unit, fmt in (("track", "km", "{:.1f}"), ("intensity", "kt", "{:.2f}")):
        print(f"\n{task.title()} ({unit})")
        print(f"  {'horizon':>8} {'history only':>13} {'+ ERA5':>10} {'change':>10}")
        for h in HORIZONS:
            b, a = before.get((task, h)), after.get((task, h))
            if b is None or a is None:
                continue
            delta = 100 * (b - a) / b
            arrow = "better" if delta > 0 else "worse"
            print(f"  {h:>6} h {fmt.format(b):>13} {fmt.format(a):>10} "
                  f"{delta:>+9.1f}% {arrow}")
            rows.append({"task": task, "horizon_h": h, "unit": unit,
                         "history_only": b, "with_era5": a, "pct_change": delta})

    (OUT / "era5_ablation.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nWrote {OUT / 'era5_ablation.json'}")


if __name__ == "__main__":
    main()
