# every headline number with the report file it came from
#
#     python src/eval/headline_numbers.py
#
# the docs and the deck take their numbers from here rather than from
# memory. if a claim is not in this output it is not measured, and if it
# disagrees with this output the claim is wrong: reports/ is written by the
# training and evaluation scripts and by nothing else.

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[2] / "reports"


def load(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def line(label, value, source):
    print(f"{label:<52} {value:<34} {source}")


print("=" * 130)
print("CHAKRAVAT NUMBERS - from reports/ and, where marked, from the raw best track")
print("=" * 130)

IB = R.parent / "data" / "raw" / "ibtracs.NI.csv"
if IB.exists():
    import pandas as pd
    _ib = pd.read_csv(IB, low_memory=False, skiprows=[1])
    _ib["SEASON"] = pd.to_numeric(_ib["SEASON"], errors="coerce")
    _ib = _ib[(_ib.SEASON >= 1990) & (_ib.SEASON <= 2025)]
    _n_seasons = _ib.SEASON.nunique()
    _named = _ib[pd.to_numeric(_ib["WMO_WIND"], errors="coerce") >= 34]["SID"].nunique()
    print("\nBASIN CONTEXT (why the test sets are small)")
    line("  named storms a season, 1990-2025",
         f"{_named / _n_seasons:.1f}  ({_named} storms over {_n_seasons} seasons, IMD 3-minute wind)",
         "data/raw/ibtracs.NI.csv")
    line("  all systems a season",
         f"{_ib['SID'].nunique() / _n_seasons:.1f}  ({_ib['SID'].nunique()} total)",
         "data/raw/ibtracs.NI.csv")

ci = load("forecast_skill_ci.json")
if ci:
    print("\nT4 FORECAST, AS SERVED (held-out seasons 2020-2025, 51 storms)")
    for r in ci["results"]:
        lo, hi = r["skill_ci95"]
        line(f"  {r['task']} {r['horizon_h']} h",
             f"{r['served']:.2f} vs CLIPER {r['cliper']:.2f} {r['unit']}  "
             f"{r['skill_pct']:+.1f}% [{lo:+.1f},{hi:+.1f}]", "forecast_skill_ci.json")

cone = load("cone_calibration.json")
if cone:
    print("\nCONE COVERAGE (track)")
    for c in cone["coverage"]:
        if c["task"] == "track":
            line(f"  {c['horizon_h']} h",
                 f"50% {100*c['coverage_50']:.0f}  67% {100*c['coverage_67']:.0f}  "
                 f"90% {100*c['coverage_90']:.0f}", "cone_calibration.json")

lf = load("landfall_results.json")
if lf:
    print("\nLANDFALL")
    line("  timing MAE", f"{lf['timing']['model_mae_h']:.2f} h vs {lf['timing']['baseline_mae_h']:.2f} h",
         "landfall_results.json")
    line("  position mean / median",
         f"{lf['position']['hybrid_mean_km']:.0f} / {lf['position']['hybrid_median_km']:.0f} km",
         "landfall_results.json")
    line("  intensity at coast", f"{lf['intensity']['model_mae_kt']:.2f} kt", "landfall_results.json")
    line("  will it land in 72 h",
         f"POD {lf['will_land']['pod']:.2f} FAR {lf['will_land']['far']:.2f} CSI {lf['will_land']['csi']:.2f}",
         "landfall_results.json")

d = load("t1_detection.json")
if d:
    print("\nT1 DETECTION (held-out seasons, threshold %.2f)" % d["threshold"])
    line("  all systems", f"F1 {d['f1']:.3f}  recall {d['recall']:.3f}  median fix {d['median_km']:.0f} km",
         "t1_detection.json")
    n = d.get("named_only")
    if n:
        line("  34 kt and above", f"F1 {n['f1']:.3f}  recall {n['recall']:.3f}  median fix {n['median_km']:.0f} km",
             "t1_detection.json")
    line("  coldest-pixel baseline", f"F1 {d['baseline_f1']:.3f}  median {d['baseline_median_km']:.0f} km",
         "t1_detection.json")
    for band, v in d["recall_by_intensity"].items():
        if v["n"]:
            line(f"  recall {band}", f"{v['recall']:.3f}  (n={v['n']})", "t1_detection.json")

# two checkpoints serve: detector.pt answers for GridSat imagery and
# detector_insat.pt for INSAT and live, because neither wins on both.
DETS = [("detector.pt", "t1_detection_by_sensor.json", "gridsat"),
        ("detector_insat.pt", "t1_detection_insat_by_sensor.json", "insat")]
print("\nT1 BY SENSOR (the checkpoint that serves each source is marked <-)")
for ckpt, fname, serves in DETS:
    bs = load(fname)
    if not bs:
        continue
    for src in ("gridsat", "insat"):
        if src in bs:
            r = bs[src]
            line(f"  {ckpt} on {src}" + ("  <-" if src == serves else ""),
                 f"F1 {r['f1']:.3f}  named F1 {r['named_only']['f1']:.3f}  "
                 f"median fix {r['median_km']:.0f} km  ({r['n_scenes']} scenes)", fname)

cvs = load("cv/t2_hybrid_reflect.json")
if cvs:
    print("\nT2 DVORAK SCENE (5-fold by storm, %d patches, %d storms)" % (cvs["patches"], cvs["storms"]))
    for k in ("cold_cloud_logistic", "cnn", "hybrid"):
        v = cvs[k]
        line(f"  {k}", f"macro-F1 {v['macro_f1']:.3f} [{v['macro_f1_ci95'][0]:.3f},{v['macro_f1_ci95'][1]:.3f}]"
             f"  acc {v['accuracy']:.3f}", "cv/t2_hybrid_reflect.json")
    line("  hybrid minus cnn", str([round(x, 3) for x in cvs["hybrid_minus_cnn_ci95"]]),
         "cv/t2_hybrid_reflect.json")
    line("  per-class F1 (hybrid)", str({k: round(v, 2) for k, v in cvs["hybrid"]["per_class_f1"].items()}),
         "cv/t2_hybrid_reflect.json")

t2 = load("t2_scene.json")
if t2:
    line("  served split: hybrid", f"macro-F1 {t2['hybrid']['macro_f1']:.3f}  acc {t2['hybrid']['accuracy']:.3f}",
         "t2_scene.json")

cvi = load("cv/t3_best_track_rotate_joint.json")
if cvi:
    print("\nT3 INTENSITY vs IMD BEST TRACK (5-fold by storm, %d patches, %d storms)"
          % (cvi["patches"], cvi["storms"]))
    for k in ("predict_the_mean", "cold_cloud", "cnn_raw"):
        v = cvi[k]
        line(f"  {k}", f"RMSE {v['rmse']:.2f} [{v['rmse_ci95'][0]:.2f},{v['rmse_ci95'][1]:.2f}] kt  "
             f"MAE {v['mae']:.2f}  bias {v['bias']:+.2f}  category {100*v['imd_category_exact']:.0f}%",
             "cv/t3_best_track_rotate_joint.json")
    b = cvi.get("adt_benchmark")
    if b:
        line("  ADT on the same scenes", f"RMSE {b['adt']['rmse']:.2f} kt  bias {b['adt']['bias']:+.2f}  "
             f"category {100*b['adt']['imd_category_exact']:.0f}%  (n={b['n']})",
             "cv/t3_best_track_rotate_joint.json")
        line("  chakravat minus ADT RMSE", str([round(x, 2) for x in b["cnn_raw_minus_adt_rmse_ci95"]]),
             "cv/t3_best_track_rotate_joint.json")
    line("  bias by band", str({k: (round(v["bias"], 1) if v["bias"] is not None else None)
                                for k, v in cvi["cnn_raw"]["bands"].items()}),
         "cv/t3_best_track_rotate_joint.json")
    alt = load("cv/t3_best_track_rotate.json")
    if alt:
        line("  same model trained on GridSat alone",
             f"RMSE {alt['cnn_raw']['rmse']:.2f} kt  bias {alt['cnn_raw']['bias']:+.2f}  "
             f"(replaced: loses 1.3 kt on INSAT imagery)", "cv/t3_best_track_rotate.json")

t3 = load("t3_intensity_gridsat.json")
if t3:
    line("  served split: raw", f"RMSE {t3['raw']['rmse']:.2f} kt  MAE {t3['raw']['mae']:.2f}  "
         f"bias {t3['raw']['bias']:+.2f}", "t3_intensity_gridsat.json")
    if t3.get("adt_benchmark"):
        line("  served split: ADT", f"RMSE {t3['adt_benchmark']['adt']['rmse']:.2f} kt",
             "t3_intensity_gridsat.json")

adt = load("adt_vs_best_track.json")
if adt:
    print("\nADT WIND SCALE vs IMD BEST TRACK")
    a = adt["adt_vs_best_track"]
    line("  whole archive", f"bias {a['bias']:+.2f} kt  RMSE {a['rmse']:.2f}  (n={a['n']})",
         "adt_vs_best_track.json")
    line("  mean ratio at 34 kt+", f"{adt['mean_ratio_adt_to_best_track_34kt_up']:.3f}",
         "adt_vs_best_track.json")

print("\nSENSOR EXPERIMENTS (INSAT vs GridSat training)")
for task in ("t3", "t2"):
    for mode in ("gridsat", "joint", "insat"):
        r = load(f"cv/sensors_{task}_{mode}.json")
        if not r:
            continue
        for sensor in ("gridsat", "insat"):
            v = r.get(f"on_{sensor}")
            if not v:
                continue
            if task == "t3":
                line(f"  T3 trained {mode}, scored on {sensor}",
                     f"RMSE {v['rmse']:.2f} [{v['rmse_ci95'][0]:.2f},{v['rmse_ci95'][1]:.2f}] kt  "
                     f"bias {v['bias']:+.2f}  (n={v['n']})", f"cv/sensors_{task}_{mode}.json")
            else:
                h = v["hybrid"]
                line(f"  T2 trained {mode}, scored on {sensor}",
                     f"hybrid macro-F1 {h['macro_f1']:.3f} "
                     f"[{h['macro_f1_ci95'][0]:.3f},{h['macro_f1_ci95'][1]:.3f}]  (n={v['n']})",
                     f"cv/sensors_{task}_{mode}.json")

for tag in ("noflip", "joint"):
    r = load(f"candidates/t1_detection_by_sensor_{tag}.json")
    if r:
        for src in ("gridsat", "insat"):
            if src in r:
                v = r[src]
                line(f"  T1 candidate {tag}, on {src}",
                     f"F1 {v['f1']:.3f}  named F1 {v['named_only']['f1']:.3f}  "
                     f"median fix {v['median_km']:.0f} km",
                     f"candidates/t1_detection_by_sensor_{tag}.json")
print()
