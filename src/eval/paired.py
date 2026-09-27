"""Compare two models the way an operational centre does, not by eye.

We have been deciding whether a change helped by printing two confidence
intervals and seeing whether they overlap. That test is far too blunt, and it is
not what anyone who does this professionally uses.

The National Hurricane Center's verification compares two models with a
**two-sided paired t-test on a homogeneous sample** - the same cases for both -
and adjusts the degrees of freedom for serial correlation, treating forecasts
less than 18 hours apart as not independent.

The distinction matters because it changes conclusions. Two models scored on the
same 943 patches are not two independent samples; they are one sample measured
twice. Their overall intervals can overlap almost completely while the
*difference* between them is consistent and clearly non-zero, because the storms
that are hard for one are hard for the other and that shared difficulty cancels
when you subtract.

Two things here, and both matter:

  paired_bootstrap  resamples whole storms and reports an interval on the
                    difference, which is what should have been reported all
                    along

  effective_n       how many independent observations 943 correlated patches
                    are actually worth, which is what a t-test needs and what
                    makes the difference between a real result and a coincidence
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SEPARATION_H = 18.0          # NHC's threshold for treating fixes as independent


def effective_n(times, storms, min_sep_h: float = MIN_SEPARATION_H) -> int:
    """Independent observations in a set of serially correlated fixes.

    Our patches are three hours apart within a storm and a cyclone does not
    reinvent itself in three hours, so 943 patches are nowhere near 943
    independent facts. Counting how many survive an 18-hour separation rule
    gives an honest denominator.
    """
    t = pd.to_datetime(pd.Series(list(times)))
    s = pd.Series(list(storms)).astype(str)
    total = 0
    for _, idx in s.groupby(s).groups.items():
        ts = t.loc[idx].sort_values()
        last, kept = None, 0
        for v in ts:
            if last is None or (v - last).total_seconds() / 3600.0 >= min_sep_h:
                kept += 1
                last = v
        total += kept
    return int(total)


def paired_bootstrap(storms, diff, reps: int = 4000, seed: int = 0) -> dict:
    """Interval on the mean paired difference, resampling whole storms.

    Resampling storms rather than rows keeps every fix of a chosen storm
    together, which is the only way to respect the fact that they are not
    independent of each other.
    """
    rng = np.random.default_rng(seed)
    storms = np.asarray(storms)
    diff = np.asarray(diff, dtype=float)
    uniq = np.unique(storms)
    where = {s: np.flatnonzero(storms == s) for s in uniq}

    means = np.empty(reps)
    for i in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([where[s] for s in pick])
        means[i] = diff[rows].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(diff.mean()), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "share_of_storms_improved": float(np.mean(
                [diff[where[s]].mean() > 0 for s in uniq]))}


def compare(y, a, b, storms, times=None, label_a="A", label_b="B",
            metric: str = "squared") -> dict:
    """Is b better than a on the same cases, and by how much?

    metric "squared" compares squared errors, whose mean is the MSE behind an
    RMSE; "absolute" compares absolute errors, which is what an intensity MAE
    is. The paired difference is a-error minus b-error, so a positive mean means
    b is the better model.
    """
    y, a, b = np.asarray(y, float), np.asarray(a, float), np.asarray(b, float)
    ea, eb = np.abs(a - y), np.abs(b - y)
    if metric == "squared":
        ea, eb = ea ** 2, eb ** 2
    diff = ea - eb

    out = {"label_a": label_a, "label_b": label_b, "n": int(len(y)),
           "metric": metric,
           "rmse_a": float(np.sqrt(np.mean((a - y) ** 2))),
           "rmse_b": float(np.sqrt(np.mean((b - y) ** 2))),
           "mae_a": float(np.mean(np.abs(a - y))),
           "mae_b": float(np.mean(np.abs(b - y)))}
    out.update(paired_bootstrap(storms, diff))

    if times is not None:
        n_eff = effective_n(times, storms)
        out["n_effective"] = n_eff
        # a paired t on the effective sample, which is the NHC construction
        sd = diff.std(ddof=1)
        if sd > 0 and n_eff > 2:
            from scipy import stats
            t = diff.mean() / (sd / np.sqrt(n_eff))
            out["t"] = float(t)
            out["p_value"] = float(2 * (1 - stats.t.cdf(abs(t), df=n_eff - 1)))
            out["significant_5pct"] = bool(out["p_value"] < 0.05)
    return out


def report(res: dict) -> str:
    unit = "kt^2" if res["metric"] == "squared" else "kt"
    lines = [
        f"{res['label_a']} -> {res['label_b']}  on {res['n']} paired cases",
        f"  RMSE {res['rmse_a']:.2f} -> {res['rmse_b']:.2f} kt"
        f"   MAE {res['mae_a']:.2f} -> {res['mae_b']:.2f} kt",
        f"  paired difference in {res['metric']} error: {res['mean']:+.3f} {unit} "
        f"[{res['ci95'][0]:+.3f}, {res['ci95'][1]:+.3f}]",
        f"  interval excludes zero: {res['excludes_zero']}",
        f"  storms improved: {100*res['share_of_storms_improved']:.0f}%",
    ]
    if "n_effective" in res:
        lines.append(f"  effective sample {res['n_effective']} of {res['n']} "
                     f"after an {MIN_SEPARATION_H:.0f} h separation rule")
    if "p_value" in res:
        lines.append(f"  paired t = {res['t']:+.2f}, p = {res['p_value']:.4f}, "
                     f"significant at 5%: {res['significant_5pct']}")
    return "\n".join(lines)
