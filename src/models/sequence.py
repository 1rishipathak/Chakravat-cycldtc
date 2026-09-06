# GRU sequence forecaster - the deep-learning arm of the comparison

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from features.build import HORIZONS

SEQ_LEN = 8          # 8 synoptic steps = 48 h of history
STEP_HOURS = 6

# per-timestep channels. Kept deliberately small - more channels than the
# training set can support is how a sequence model memorises storms.
SEQ_FEATURES = ["LAT", "LON", "vmax_kt", "pres_hpa", "u_deg_h", "v_deg_h", "dv_prev6"]

# storm-level context, concatenated to the encoder output rather than repeated
# at every timestep.
STATIC_FEATURES = [
    "age_h", "vmax_max_so_far", "DIST2LAND",
    "doy_sin", "doy_cos", "is_bob", "NEWDELHI_CI",
]

TARGETS = [f"y_dlat_{h}" for h in HORIZONS] \
        + [f"y_dlon_{h}" for h in HORIZONS] \
        + [f"y_dv_{h}" for h in HORIZONS]


def static_features(df) -> list[str]:
    # storm context, extended with ERA5 columns when the cache supplied them
    from features.environment import ENV_FEATURES

    return STATIC_FEATURES + [c for c in ENV_FEATURES if c in df.columns]


def build_sequences(df: pd.DataFrame) -> dict[str, np.ndarray]:
    # assemble (N, SEQ_LEN, C) history tensors aligned to each forecast time
    seqs, masks, statics, targets, target_masks, index = [], [], [], [], [], []
    static_cols = static_features(df)

    for sid, storm in df.groupby("SID", sort=False):
        storm = storm.sort_values("ISO_TIME")
        # index every fix by its offset in 6-hour steps from the storm's first
        # fix, so gaps become absent slots instead of shifted history.
        t0 = storm["ISO_TIME"].iloc[0]
        slot = ((storm["ISO_TIME"] - t0).dt.total_seconds() / 3600 / STEP_HOURS).round().astype(int)
        by_slot = {int(s): row for s, row in zip(slot, storm.to_dict("records"))}

        for s, row in zip(slot, storm.to_dict("records")):
            s = int(s)
            window = np.zeros((SEQ_LEN, len(SEQ_FEATURES)), dtype=np.float32)
            valid = np.zeros((SEQ_LEN, 1), dtype=np.float32)
            for k in range(SEQ_LEN):
                past = by_slot.get(s - (SEQ_LEN - 1 - k))
                if past is None:
                    continue
                vals = [past[c] for c in SEQ_FEATURES]
                if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in vals):
                    continue
                window[k] = np.array(vals, dtype=np.float32)
                valid[k] = 1.0

            # A forecast needs the current fix at minimum.
            if valid[-1, 0] == 0.0:
                continue

            stat = np.array([row.get(c, np.nan) for c in static_cols], dtype=np.float32)
            stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)

            y = np.array([row.get(t, np.nan) for t in TARGETS], dtype=np.float32)
            ymask = np.isfinite(y).astype(np.float32)
            if ymask.sum() == 0:
                continue

            seqs.append(window)
            masks.append(valid)
            statics.append(stat)
            targets.append(np.nan_to_num(y, nan=0.0))
            target_masks.append(ymask)
            index.append((sid, row["ISO_TIME"]))

    return {
        "seq": np.stack(seqs),
        "seq_mask": np.stack(masks),
        "static": np.stack(statics),
        "y": np.stack(targets),
        "y_mask": np.stack(target_masks),
        "index": index,
        "static_cols": static_cols,
    }


class GruForecaster(nn.Module):
    # small bidirectional-free GRU encoder with a multi-horizon head

    def __init__(self, n_seq: int, n_static: int, n_out: int, hidden: int = 64,
                 layers: int = 1, dropout: float = 0.2):
        super().__init__()
        # mask is concatenated as an extra channel so the net can tell an
        # absent fix from a genuine zero.
        self.gru = nn.GRU(n_seq + 1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_out),
        )

    def forward(self, seq: torch.Tensor, seq_mask: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        x = torch.cat([seq, seq_mask], dim=-1)
        _, h = self.gru(x)
        return self.head(torch.cat([h[-1], static], dim=-1))


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # MSE over observed targets only - horizons the storm did not reach are
    diff = (pred - target) ** 2 * mask
    return diff.sum() / mask.sum().clamp(min=1.0)
