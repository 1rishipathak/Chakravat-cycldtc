# summary figures for the report: skill curves, RI reliability, leakage

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TEAL, ORANGE, RED, GREY, BLUE = "#0B6E7F", "#DD7526", "#CE4630", "#7C949D", "#3E7CA8"


def main() -> None:
    results = json.loads((ROOT / "reports" / "prediction_results.json").read_text())
    track = pd.DataFrame(results["track"])
    intensity = pd.DataFrame(results["intensity"])
    final = pd.DataFrame(json.loads(
        (ROOT / "reports" / "final_results.json").read_text())["results"])
    ft = final[final["task"] == "track"].set_index("horizon_h")
    fi = final[final["task"] == "intensity"].set_index("horizon_h")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- track error vs horizon ---
    ax = axes[0]
    t = track.pivot(index="horizon_h", columns="model", values="mean_km")
    for name, colour, style in (("persistence", GREY, ":"), ("cliper", BLUE, "--")):
        ax.plot(t.index, t[name], style, color=colour, marker="o", ms=5, lw=2, label=name)
    ax.plot(ft.index, ft["selected_error"], "-", color=TEAL, marker="o", ms=5, lw=2.4,
            label="chakravat (selected)")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Mean position error (km)")
    ax.set_title("Track error", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xticks(t.index)

    # --- intensity error vs horizon ---
    ax = axes[1]
    i = intensity.pivot(index="horizon_h", columns="model", values="mae_kt")
    for name, colour, style in (("persistence", GREY, ":"), ("cliper", BLUE, "--")):
        ax.plot(i.index, i[name], style, color=colour, marker="o", ms=5, lw=2, label=name)
    ax.plot(fi.index, fi["selected_error"], "-", color=TEAL, marker="o", ms=5, lw=2.4,
            label="chakravat (selected)")
    ax.set_xlabel("Forecast horizon (h)")
    ax.set_ylabel("Intensity MAE (kt)")
    ax.set_title("Intensity error", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    ax.set_xticks(i.index)

    # --- skill of the selected operational model, over CLIPER ---
    ax = axes[2]
    pos = np.arange(len(HORIZ := list(ft.index)))
    width = 0.38
    ax.bar(pos - width / 2, ft["skill_vs_cliper_pct"], width=width, color=TEAL, label="track")
    ax.bar(pos + width / 2, fi["skill_vs_cliper_pct"], width=width, color=ORANGE, label="intensity")
    ax.axhline(0, color="#0D1B21", lw=1)
    # mark where the GRU was selected over the boosted stack.
    for k, h in enumerate(HORIZ):
        if ft.loc[h, "selected_model"] == "gru" or fi.loc[h, "selected_model"] == "gru":
            ax.text(k, max(ft.loc[h, "skill_vs_cliper_pct"],
                           fi.loc[h, "skill_vs_cliper_pct"]) + 0.6,
                    "GRU", ha="center", fontsize=8, color="#4A626C", fontweight="bold")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{h} h" for h in HORIZ])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Skill vs CLIPER (%)")
    ax.set_title("Skill of the selected model", fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Chakravat prediction stack - held-out seasons 2020-2025, North Indian Ocean",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG / "skill_summary.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # --- leakage bar ---
    leak = results["leakage_demo"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    names = ["Random frame split\n(leaky)", "Held-out seasons\n(honest)"]
    vals = [leak["frame_split_leaky"]["mae_kt"], leak["storm_split_honest"]["mae_kt"]]
    bars = ax.bar(names, vals, color=[ORANGE, TEAL], width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.12, f"{v:.2f} kt",
                ha="center", fontweight="bold")
    ax.set_ylabel("24 h intensity MAE (kt)")
    ax.set_title("Same model, same data, two evaluation protocols",
                 fontweight="bold", fontsize=11)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    out = FIG / "leakage_demo.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
