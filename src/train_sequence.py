# train the GRU forecaster and score it against the boosted stack

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest.ibtracs import build                                  # noqa: E402
from features.build import build_dataset, HORIZONS                # noqa: E402
from dataset import load_dataset                              # noqa: E402
from eval.splits import storm_split, describe                     # noqa: E402
from eval import metrics                                          # noqa: E402
from models.sequence import (                                     # noqa: E402
    build_sequences, GruForecaster, masked_mse,
    SEQ_FEATURES, STATIC_FEATURES, TARGETS,
)

RAW = ROOT / "data" / "raw" / "ibtracs.NI.csv"
OUT = ROOT / "reports"
ART = ROOT / "artifacts"

SEED = 0
EPOCHS = 300
PATIENCE = 30
BATCH = 128
LR = 2e-3


def standardise(train_arr: np.ndarray, *others: np.ndarray, axis=(0,)):
    mean = train_arr.mean(axis=axis, keepdims=True)
    std = train_arr.std(axis=axis, keepdims=True)
    std[std < 1e-6] = 1.0
    return [(a - mean) / std for a in (train_arr,) + others], mean, std


def to_tensors(pack: dict, device: str) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(pack["seq"], dtype=torch.float32, device=device),
        torch.tensor(pack["seq_mask"], dtype=torch.float32, device=device),
        torch.tensor(pack["static"], dtype=torch.float32, device=device),
        torch.tensor(pack["y"], dtype=torch.float32, device=device),
        torch.tensor(pack["y_mask"], dtype=torch.float32, device=device),
    )


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print(f"CHAKRAVAT - GRU sequence forecaster  [device: {device}]")
    print("=" * 78)

    data = load_dataset()
    splits = storm_split(data)
    print(describe(splits))

    packs = {name: build_sequences(part) for name, part in splits.items()}
    for name, p in packs.items():
        print(f"  {name:<6} sequences: {len(p['seq']):,}")

    # normalisation statistics come from the training split only.
    seq_flat = packs["train"]["seq"].reshape(-1, len(SEQ_FEATURES))
    seq_valid = packs["train"]["seq_mask"].reshape(-1) > 0
    seq_mean = seq_flat[seq_valid].mean(axis=0)
    seq_std = seq_flat[seq_valid].std(axis=0)
    seq_std[seq_std < 1e-6] = 1.0

    stat_mean = packs["train"]["static"].mean(axis=0)
    stat_std = packs["train"]["static"].std(axis=0)
    stat_std[stat_std < 1e-6] = 1.0

    y_train = packs["train"]["y"]
    y_mask_train = packs["train"]["y_mask"]
    y_mean = (y_train * y_mask_train).sum(0) / np.clip(y_mask_train.sum(0), 1, None)
    y_var = ((y_train - y_mean) ** 2 * y_mask_train).sum(0) / np.clip(y_mask_train.sum(0), 1, None)
    y_std = np.sqrt(y_var)
    y_std[y_std < 1e-6] = 1.0

    for p in packs.values():
        p["seq"] = ((p["seq"] - seq_mean) / seq_std) * p["seq_mask"]
        p["static"] = (p["static"] - stat_mean) / stat_std
        p["y_norm"] = (p["y"] - y_mean) / y_std

    tensors = {}
    for name, p in packs.items():
        tensors[name] = (
            torch.tensor(p["seq"], dtype=torch.float32),
            torch.tensor(p["seq_mask"], dtype=torch.float32),
            torch.tensor(p["static"], dtype=torch.float32),
            torch.tensor(p["y_norm"], dtype=torch.float32),
            torch.tensor(p["y_mask"], dtype=torch.float32),
        )

    train_ds = TensorDataset(*tensors["train"])
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=False)

    # size the static branch from the data, not the constant: build_sequences
    # appends whatever ERA5 columns are present, so the width varies with
    # whether the environment cache exists.
    n_static = packs["train"]["static"].shape[1]
    print(f"  static features: {n_static} "
          f"({', '.join(packs['train']['static_cols'][:4])}, ...)")
    model = GruForecaster(
        n_seq=len(SEQ_FEATURES), n_static=n_static, n_out=len(TARGETS)
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

    val = [t.to(device) for t in tensors["val"]]
    best_val, best_state, bad = np.inf, None, 0

    print(f"\nparameters: {sum(p.numel() for p in model.parameters()):,}")
    print("training ...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for seq, smask, stat, y, ymask in loader:
            seq, smask, stat = seq.to(device), smask.to(device), stat.to(device)
            y, ymask = y.to(device), ymask.to(device)
            opt.zero_grad()
            loss = masked_mse(model(seq, smask, stat), y, ymask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vloss = masked_mse(model(val[0], val[1], val[2]), val[3], val[4]).item()
        sched.step(vloss)

        if vloss < best_val - 1e-5:
            best_val, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if epoch % 25 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  val masked-MSE {vloss:.4f}  (best {best_val:.4f})")
        if bad >= PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()

    test = [t.to(device) for t in tensors["test"]]
    with torch.no_grad():
        pred_norm = model(test[0], test[1], test[2]).cpu().numpy()
    pred = pred_norm * y_std + y_mean

    truth = packs["test"]["y"]
    tmask = packs["test"]["y_mask"]
    idx = pd.DataFrame(packs["test"]["index"], columns=["SID", "ISO_TIME"])
    test_rows = idx.merge(splits["test"], on=["SID", "ISO_TIME"], how="left")

    print("\n" + "=" * 78)
    print("GRU vs BOOSTED STACK - test seasons 2020-2025")
    print("=" * 78)

    bundle = joblib.load(ART / "prediction_models.joblib")
    residual = bundle["residual"]
    n_h = len(HORIZONS)
    comparison = []

    print("\nTrack - mean position error (km)")
    print(f"  {'horizon':>8} {'n':>6} {'chakravat':>11} {'gru':>9}   verdict")
    for i, h in enumerate(HORIZONS):
        m = (tmask[:, i] > 0) & (tmask[:, n_h + i] > 0)
        if m.sum() < 20:
            continue
        sub = test_rows[m]
        true_lat = sub[f"true_lat_{h}"].to_numpy()
        true_lon = sub[f"true_lon_{h}"].to_numpy()
        gru_lat = sub["LAT"].to_numpy() + pred[m, i]
        gru_lon = sub["LON"].to_numpy() + pred[m, n_h + i]
        res_lat = sub["LAT"].to_numpy() + residual.predict(sub, f"y_dlat_{h}")
        res_lon = sub["LON"].to_numpy() + residual.predict(sub, f"y_dlon_{h}")

        g = metrics.track_error(true_lat, true_lon, gru_lat, gru_lon)["mean_km"]
        r = metrics.track_error(true_lat, true_lon, res_lat, res_lon)["mean_km"]
        verdict = "GRU wins" if g < r else "boosted wins"
        print(f"  {h:>6} h {int(m.sum()):>6} {r:>11.1f} {g:>9.1f}   {verdict}")
        comparison.append({"task": "track", "horizon_h": h, "n": int(m.sum()),
                           "chakravat_km": r, "gru_km": g})

    print("\nIntensity - MAE (kt)")
    print(f"  {'horizon':>8} {'n':>6} {'chakravat':>11} {'gru':>9}   verdict")
    for i, h in enumerate(HORIZONS):
        col = 2 * n_h + i
        m = tmask[:, col] > 0
        if m.sum() < 20:
            continue
        sub = test_rows[m]
        truth_dv = truth[m, col]
        g = metrics.intensity_error(truth_dv, pred[m, col])["mae_kt"]
        r = metrics.intensity_error(truth_dv, residual.predict(sub, f"y_dv_{h}"))["mae_kt"]
        verdict = "GRU wins" if g < r else "boosted wins"
        print(f"  {h:>6} h {int(m.sum()):>6} {r:>11.2f} {g:>9.2f}   {verdict}")
        comparison.append({"task": "intensity", "horizon_h": h, "n": int(m.sum()),
                           "chakravat_kt": r, "gru_kt": g})

    (OUT / "sequence_comparison.json").write_text(json.dumps(comparison, indent=2))
    torch.save({"state_dict": model.state_dict(), "n_static": n_static,
                "static_cols": packs["train"]["static_cols"],
                "seq_mean": seq_mean, "seq_std": seq_std,
                "stat_mean": stat_mean, "stat_std": stat_std,
                "y_mean": y_mean, "y_std": y_std}, ART / "gru_forecaster.pt")
    print(f"\nWrote {OUT / 'sequence_comparison.json'}")
    print(f"Saved {ART / 'gru_forecaster.pt'}")


if __name__ == "__main__":
    main()
