# vision endpoints: detection, scene typing, and intensity from imagery

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts"
GRIDSAT = ROOT / "data" / "gridsat"

# imagery sources on the same 0.07 degree grid with the same file naming, so
# every function below works on any of them. "insat" is the INSAT-3D/3DR
# archive regridded for training; "live" is the latest INSAT-3DS scans fetched
# by src/ingest/insat_live.py.
SOURCES = {"gridsat": GRIDSAT,
           "insat": ROOT / "data" / "insat" / "grid",
           "live": ROOT / "data" / "insat" / "live"}


# which detector reads which imagery. the GridSat-trained detector scores
# 0.408 F1 on named storms in INSAT scenes against 0.675 for one trained on both
# sensors, and the live feed is INSAT - so INSAT imagery gets its own. the other
# direction is why there are two: the joint detector loses 0.07 F1 on GridSat,
# and nothing that works today is allowed to get worse.
DETECTORS = {"gridsat": ("detector.pt", "t1_detection.json"),
             "insat": ("detector_insat.pt", "t1_detection_insat.json")}


def detector_family(source: str) -> str:
    return "insat" if source in ("insat", "live") else "gridsat"


def tier_threshold(tier: str, source: str = "gridsat") -> float:
    """The peak threshold for a tier.

    The same checkpoint has two jobs. "warn" is the served operating point,
    tuned for F1, which is what a detection that becomes a forecast needs. "watch"
    is a more sensitive point tuned for F2 on the same validation seasons, for
    asking whether anything is forming at all. It finds more weak systems and
    costs false alarms, and both numbers are in reports/t1_operating_points.json.
    """
    if tier == "warn":
        return tuned_threshold(source)
    if tier != "watch":
        raise ValueError(f"tier must be warn or watch, not {tier!r}")
    try:
        d = json.loads((ROOT / "reports" / "t1_operating_points.json").read_text())
        return float(d["watch"]["threshold"])
    except (OSError, ValueError, KeyError):
        return tuned_threshold(source)


def operating_points() -> dict:
    """What each tier measured on the held-out seasons, for printing beside it."""
    try:
        d = json.loads((ROOT / "reports" / "t1_operating_points.json").read_text())
    except (OSError, ValueError):
        return {}
    keep = ("threshold", "precision", "recall", "f1", "weak_recall",
            "false_alarms_per_scene")
    return {t: {k: d[t][k] for k in keep if k in d[t]} for t in ("warn", "watch") if t in d}


def tuned_threshold(source: str = "gridsat") -> float:
    # the peak threshold train_detect.py picked on validation seasons for
    # whichever detector serves this source. the API used to hard-code 0.30
    # while the reported skill was scored at the tuned value.
    import json
    _, name = DETECTORS[detector_family(source)]
    for report in (ROOT / "reports" / name, ROOT / "reports" / "t1_detection.json"):
        try:
            return float(json.loads(report.read_text())["threshold"])
        except (OSError, ValueError, KeyError):
            continue
    return 0.30


def root_of(source: str) -> Path:
    if source not in SOURCES:
        raise ValueError(f"unknown imagery source {source!r}")
    return SOURCES[source]


class VisionModels:
    # lazy holder for the three imagery checkpoints

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._detectors: dict[str, object] = {}
        self._scene = None
        self._scene_classes: list[str] = []
        self.scene_stats = None          # cold-cloud half of the T2 hybrid
        self.scene_blend = 0.0
        self._intensity = None
        self.intensity_scale = (0.0, 1.0)
        self.intensity_cal = (1.0, 0.0)
        self.intensity_domain = "unknown"
        self._loaded = {"scene": False, "intensity": False}
        # FastAPI runs these endpoints on a thread pool and the dashboard asks
        # for an intensity estimate and its attention map at the same moment.
        # Grad-CAM hangs a hook on the shared model, so one thread's hook fired
        # inside the other's plain forward pass and crashed it. One lock around
        # every model call; each is a single GPU pass, so the wait is tiny.
        self.lock = threading.RLock()

    # - loaders ---------------------------------------------------------

    def detector(self, source: str = "gridsat"):
        # under the lock: the flag is set before loading starts, so without it a
        # second request mid-load saw "loaded" and got None back
        with self.lock:
            return self._detector_unlocked(source)

    def _detector_unlocked(self, source: str = "gridsat"):
        family = detector_family(source)
        if family not in self._detectors:
            self._detectors[family] = None
            name, _ = DETECTORS[family]
            path = ART / name
            if not path.exists():             # fall back to the GridSat one
                path = ART / DETECTORS["gridsat"][0]
            if path.exists():
                from train_detect import HeatmapNet
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                model = HeatmapNet().to(self.device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._detectors[family] = model
        return self._detectors[family]

    def scene(self):
        # under the lock: the flag is set before loading starts, so without it a
        # second request mid-load saw "loaded" and got None back
        with self.lock:
            return self._scene_unlocked()

    def _scene_unlocked(self):
        if not self._loaded["scene"]:
            self._loaded["scene"] = True
            path = ART / "scene_classifier.pt"
            if path.exists():
                import timm
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                self._scene_classes = ckpt["classes"]
                model = timm.create_model(ckpt["backbone"], pretrained=False,
                                          num_classes=len(ckpt["classes"]),
                                          in_chans=3).to(self.device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._scene = model
                # a checkpoint trained as the hybrid carries its logistic model
                self.scene_stats = ckpt.get("stat_model")
                self.scene_blend = float(ckpt.get("blend", 0.0)) if self.scene_stats else 0.0
        return self._scene, self._scene_classes

    def intensity(self):
        # under the lock: the flag is set before loading starts, so without it a
        # second request mid-load saw "loaded" and got None back
        with self.lock:
            return self._intensity_unlocked()

    def _intensity_unlocked(self):
        if not self._loaded["intensity"]:
            self._loaded["intensity"] = True
            # intensity_vision.pt was trained on Digital Typhoon and must not
            # be served against GridSat patches - different sensor, channels
            # and hemisphere. The GridSat-domain model is the deployable one.
            path = (ART / "intensity_gridsat.pt" if (ART / "intensity_gridsat.pt").exists()
                    else ART / "intensity_vision.pt")
            if path.exists():
                import timm
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                model = timm.create_model(ckpt["backbone"], pretrained=False,
                                          num_classes=1, in_chans=3).to(self.device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._intensity = model
                self.intensity_scale = (ckpt.get("y_mean", 0.0),
                                        ckpt.get("y_std", 1.0))
                self.intensity_domain = ckpt.get("domain", "digital_typhoon")
                # variance-restoring recalibration, fitted on validation.
                # without it the model shrinks toward the mean and reads
                # severe cyclones ~17 kt low - see reports/t3_calibration.json.
                cal = ckpt.get("calibration")
                self.intensity_cal = ((cal["slope"], cal["intercept"])
                                      if cal else (1.0, 0.0))
        return self._intensity

    def status(self) -> dict:
        return {
            "device": self.device,
            "detector": (ART / "detector.pt").exists(),
            "detector_insat": (ART / "detector_insat.pt").exists(),
            "scene_classifier": (ART / "scene_classifier.pt").exists(),
            "intensity_vision": ((ART / "intensity_gridsat.pt").exists()
                                 or (ART / "intensity_vision.pt").exists()),
            "intensity_domain": self.intensity_domain,
            "intensity_calibrated": self.intensity_cal != (1.0, 0.0),
            "gridsat_scenes": len(list(GRIDSAT.rglob("nio_*.nc"))),
        }


MODELS = VisionModels()


def nearest_scene(when: pd.Timestamp, source: str = "gridsat") -> pd.Timestamp | None:
    # the 3-hourly slot nearest a requested time, if it is on disk
    from vision.detect import gridsat_path

    slot = pd.Timestamp(when).round("3h")
    for shift in (0, -3, 3, -6, 6):
        cand = slot + pd.Timedelta(hours=shift)
        if gridsat_path(root_of(source), cand).exists():
            return cand
    return None


def detect(when: pd.Timestamp, threshold: float | None = None, source: str = "gridsat") -> dict:
    # run T1 over a whole basin scene
    from vision.detect import BasinScenes, find_peaks, gridsat_path, STRIDE

    model = MODELS.detector(source)
    if model is None:
        return {"available": False,
                "reason": "detector not trained yet (artifacts/detector.pt missing)"}

    if threshold is None:
        threshold = tuned_threshold(source)
    slot = nearest_scene(when, source)
    if slot is None:
        return {"available": False,
                "reason": f"no {source} scene near {when}"}

    index = pd.DataFrame([{"time": slot, "season": slot.year, "n_storms": 0}])
    ds_wrap = BasinScenes(index, pd.DataFrame(columns=["ISO_TIME", "LAT", "LON"]),
                          root_of(source))
    # only the image is wanted here. The dataset also yields the target heatmap,
    # the per-storm loss weight map and a target count, all of which exist for
    # training. Unpacking positionally broke the moment the weight map was added
    # for intensity-weighted targets, so index rather than destructure.
    x = ds_wrap[0][0]
    with MODELS.lock, torch.no_grad():
        heat = torch.sigmoid(
            model(x.unsqueeze(0).to(MODELS.device)))[0, 0].cpu().numpy()

    with xr.open_dataset(gridsat_path(root_of(source), slot)) as g:
        lats, lons = g["lat"].values, g["lon"].values

    found = []
    for y, x_, score in find_peaks(heat, threshold):
        lat, lon = BasinScenes.from_pixel(y * STRIDE, x_ * STRIDE, lats, lons)
        found.append({"lat": lat, "lon": lon, "score": float(score)})
    # the single strongest response even when nothing clears the threshold, so
    # "no storm" on a live scene still shows the detector looked
    iy, ix = np.unravel_index(int(np.argmax(heat)), heat.shape)
    plat, plon = BasinScenes.from_pixel(iy * STRIDE, ix * STRIDE, lats, lons)

    return {"available": True, "scene_time": str(slot), "threshold": threshold,
            "source": source, "detector": detector_family(source),
            "detections": found,
            "strongest": {"lat": plat, "lon": plon, "score": float(heat.max())},
            # verified on held-out seasons; surfaced so the output is never read
            # as more precise than it is.
            "skill_note": "held-out F1 and median centre-fix in /api/skill"}


def _as_tensor(patch: np.ndarray, norm: str = "imagenet") -> torch.Tensor:
    # patch -> model input, with the same normalisation used in training. the
    # mode travels in the checkpoint, so a model trained one way cannot be
    # served the other by accident.
    from vision.scenes import normalise

    a = np.ascontiguousarray((patch.astype(np.float32) / 255.0).transpose(2, 0, 1))
    return torch.from_numpy(normalise(a, norm)).unsqueeze(0)


def _patch_at(when: pd.Timestamp, lat: float, lon: float, source: str = "gridsat"):
    from vision.scenes import extract_patch, gridsat_path

    slot = nearest_scene(when, source)
    if slot is None:
        return None, None
    with xr.open_dataset(gridsat_path(root_of(source), slot)) as ds:
        patch = extract_patch(ds, lat, lon)
    return patch, slot


_SCENE_RELIABILITY: dict | None = None


def scene_reliability() -> dict:
    """Per-class F1 on each sensor, measured on 505 exactly paired scenes.

    Same storm, same time, same ADT label, read through both sensors, so the
    only difference is the instrument. It matters because the live feed is
    INSAT and two of the five classes do not survive the change: an eye is a
    geometric feature and transfers, while "is this central overcast embedded
    or irregular" is a temperature-texture judgement and does not.
    """
    global _SCENE_RELIABILITY
    if _SCENE_RELIABILITY is None:
        path = ROOT / "reports" / "t2_sensor_paired.json"
        try:
            d = json.loads(path.read_text())
            _SCENE_RELIABILITY = {
                "gridsat": d["gridsat"]["per_class_f1"],
                "insat": d["insat"]["per_class_f1"],
                "paired_scenes": d["paired_scenes"],
                "support": d["support"],
            }
        except Exception:
            _SCENE_RELIABILITY = {}
    return _SCENE_RELIABILITY


def classify_scene(when: pd.Timestamp, lat: float, lon: float,
                   source: str = "gridsat") -> dict:
    # run T2 on a storm-centred patch
    model, classes = MODELS.scene()
    if model is None:
        return {"available": False,
                "reason": "scene classifier not trained yet"}

    patch, slot = _patch_at(when, lat, lon, source)
    if patch is None:
        return {"available": False, "reason": f"no usable {source} patch near {when}"}

    x = _as_tensor(patch)
    with MODELS.lock, torch.no_grad():
        cnn = torch.softmax(model(x.to(MODELS.device)), dim=1)[0].cpu().numpy()

    probs, method = cnn, "cnn"
    if MODELS.scene_stats is not None:
        # the hybrid: CNN probabilities averaged with a logistic model on the
        # cold-cloud statistics ADT's own scene rules are built from. under
        # storm-grouped cross-validation either alone scored about 0.64-0.65
        # macro-F1 and the average 0.70.
        from train_scene import stats_features
        stats = MODELS.scene_stats.predict_proba(stats_features(patch[None], np.array([0])))[0]
        w = MODELS.scene_blend
        probs, method = (1 - w) * cnn + w * stats, "hybrid"

    order = np.argsort(-probs)
    top = classes[int(order[0])]
    out = {
        "available": True, "scene_time": str(slot),
        "scene": top,
        "confidence": float(probs[order[0]]),
        "probabilities": {classes[i]: float(probs[i]) for i in order},
        "method": method,
        "labels_note": "trained on CIMSS ADT scene types (algorithm output, "
                       "not analyst labels)",
    }

    # how well this class is actually read on the sensor it was read from. the
    # classifier is GridSat-trained and the live feed is INSAT, so saying "EMBC"
    # off an INSAT scene is a weaker statement than saying it off a GridSat one,
    # and the number that says how much weaker is measured.
    rel = scene_reliability()
    sensor = "insat" if source in ("insat", "live") else "gridsat"
    if rel and top in rel.get(sensor, {}):
        here = rel[sensor][top]
        out["reliability"] = {
            "sensor": sensor,
            "class_f1_here": round(here, 2),
            "class_f1_on_gridsat": round(rel["gridsat"][top], 2),
            "measured_on": rel["paired_scenes"],
            "note": (f"{top} is read at F1 {here:.2f} on {sensor.upper()} imagery against "
                     f"{rel['gridsat'][top]:.2f} on GridSat, measured on "
                     f"{rel['paired_scenes']} scenes carrying both."
                     if sensor == "insat" else
                     f"{top} is read at F1 {here:.2f} on GridSat imagery."),
            "weak_on_this_sensor": bool(here < 0.55),
        }
    return out


_INTERVALS: dict | None = None


def intensity_intervals() -> dict:
    """Offsets to add to a T3 estimate for a stated confidence level.

    Fitted on out-of-fold residuals - every patch predicted by a model that never
    saw its storm - and binned by what the model said rather than by the truth,
    because the prediction is the only thing available at inference. The coverage
    each level actually achieved is in the same file.
    """
    global _INTERVALS
    if _INTERVALS is None:
        try:
            _INTERVALS = json.loads((ROOT / "reports" / "t3_intervals.json").read_text())
        except (OSError, ValueError):
            _INTERVALS = {}
    return _INTERVALS


def _interval_for(kt: float, level: float) -> dict | None:
    d = intensity_intervals()
    if not d:
        return None
    key = f"{level:.2f}"
    if key not in d.get("levels", {}):
        key = "0.67"
        if key not in d.get("levels", {}):
            return None
    spec = d["levels"][key]
    edges, names = d["edges_kt"], d["bands"]
    band = names[-1]
    for i in range(1, len(edges)):
        hi = edges[i]
        if hi == "inf" or kt < float(hi):
            band = names[i - 1]
            break
    t = spec["by_band"].get(band)
    if not t:
        return None
    return {
        "lower_kt": max(kt + t["lower_offset_kt"], 0.0),
        "upper_kt": kt + t["upper_offset_kt"],
        "level": float(key),
        "band": band,
        "measured_coverage": spec["coverage"],
        "fitted_on": t["n"],
        "note": (f"{100 * float(key):.0f}% interval from out-of-fold residuals; "
                 f"this level contained {100 * spec['coverage']:.0f}% of held-out "
                 f"truths when each fold was scored against the other four."),
    }


def estimate_intensity(when: pd.Timestamp, lat: float, lon: float,
                       source: str = "gridsat", level: float = 0.67) -> dict:
    # run T3 on a storm-centred patch
    model = MODELS.intensity()
    if model is None:
        return {"available": False,
                "reason": "intensity-from-imagery model not trained yet"}

    patch, slot = _patch_at(when, lat, lon, source)
    if patch is None:
        return {"available": False, "reason": f"no usable {source} patch near {when}"}

    x = _as_tensor(patch)
    with MODELS.lock, torch.no_grad():
        raw = float(model(x.to(MODELS.device)).squeeze().cpu())
    # the GridSat model regresses a standardised target, then the calibration
    # undoes its shrinkage toward the mean.
    kt = raw * MODELS.intensity_scale[1] + MODELS.intensity_scale[0]
    slope, intercept = MODELS.intensity_cal
    kt = (kt - intercept) / max(slope, 1e-6)

    from ingest.ibtracs import imd_category
    kt = max(kt, 0.0)
    out = {"available": True, "scene_time": str(slot),
           "vmax_kt": kt, "category": imd_category(kt)}
    # one number is not an answer when the error is 13 kt wide. the interval is
    # conditioned on what the model said, which is all that is known here.
    band = _interval_for(kt, level)
    if band:
        out["interval"] = band
        out["category_upper"] = imd_category(band["upper_kt"])
        out["category_lower"] = imd_category(band["lower_kt"])
    return out


def explain(when: pd.Timestamp, lat: float, lon: float, task: str = "scene",
            source: str = "gridsat") -> tuple[bytes, dict] | None:
    # Grad-CAM for T2 (the CNN half) or T3 on a storm-centred patch
    from api import explain as xp

    patch, slot = _patch_at(when, lat, lon, source)
    if patch is None:
        return None
    x = _as_tensor(patch).to(MODELS.device)
    if task == "scene":
        model, classes = MODELS.scene()
        if model is None:
            return None
        with MODELS.lock:
            with torch.no_grad():
                k = int(model(x).argmax(1)[0])
            cam, _ = xp.grad_cam(model, x, lambda out: out[0, k])
        answer = {"task": "scene", "cnn_scene": classes[k]}
    elif task == "intensity":
        model = MODELS.intensity()
        if model is None:
            return None
        with MODELS.lock:
            cam, _ = xp.grad_cam(model, x, lambda out: out[0, 0])
        answer = {"task": "intensity"}
    else:
        raise ValueError("task must be scene or intensity")
    meta = {**answer, "scene_time": str(slot), "source": source, **xp.core_share(cam)}
    return xp.render(patch, cam), meta


def scene_bounds(when: pd.Timestamp, source: str = "gridsat") -> dict | None:
    # geographic extent of a scene, without rendering it
    from vision.detect import gridsat_path

    slot = nearest_scene(when, source)
    if slot is None:
        return None
    with xr.open_dataset(gridsat_path(root_of(source), slot)) as ds:
        lats, lons = ds["lat"].values, ds["lon"].values
    return {"scene_time": str(slot),
            "bounds": {"west": float(lons.min()), "east": float(lons.max()),
                       "south": float(lats.min()), "north": float(lats.max())}}


def scene_image(when: pd.Timestamp, enhance: bool = True,
                source: str = "gridsat") -> tuple[bytes, dict] | None:
    # render a GridSat scene as a PNG for map overlay, with its bounds
    import io as _io

    from PIL import Image
    from vision.detect import gridsat_path

    slot = nearest_scene(when, source)
    if slot is None:
        return None

    with xr.open_dataset(gridsat_path(root_of(source), slot)) as ds:
        ir = ds["irwin_cdr"].isel(time=0).values.astype(np.float32)
        lats, lons = ds["lat"].values, ds["lon"].values

    ir = np.nan_to_num(ir, nan=300.0)
    # 180-300 K spans everything from overshooting tops to warm ocean.
    v = np.clip((300.0 - ir) / (300.0 - 180.0), 0.0, 1.0)
    grey = (v * 255).astype(np.uint8)
    rgb = np.dstack([grey, grey, grey])

    if enhance:
        cold = ir < 203.0                      # about -70 C
        very = ir < 193.0                      # about -80 C
        rgb[cold] = np.dstack([
            np.full(cold.sum(), 255), (v[cold] * 90 + 90).astype(np.uint8),
            np.full(cold.sum(), 40)])[0]
        rgb[very] = np.array([255, 60, 60], dtype=np.uint8)

    # netCDF rows run south to north; PNG rows run top-down.
    if lats[0] < lats[-1]:
        rgb = rgb[::-1]

    buf = _io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", optimize=True)
    bounds = {"west": float(lons.min()), "east": float(lons.max()),
              "south": float(lats.min()), "north": float(lats.max())}
    return buf.getvalue(), {"scene_time": str(slot), "bounds": bounds}
