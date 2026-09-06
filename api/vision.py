# vision endpoints: detection, scene typing, and intensity from imagery

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ART = ROOT / "artifacts"
GRIDSAT = ROOT / "data" / "gridsat"


class VisionModels:
    # lazy holder for the three imagery checkpoints

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._detector = None
        self._scene = None
        self._scene_classes: list[str] = []
        self._intensity = None
        self.intensity_scale = (0.0, 1.0)
        self.intensity_cal = (1.0, 0.0)
        self.intensity_domain = "unknown"
        self._loaded = {"detector": False, "scene": False, "intensity": False}

    # - loaders ---------------------------------------------------------

    def detector(self):
        if not self._loaded["detector"]:
            self._loaded["detector"] = True
            path = ART / "detector.pt"
            if path.exists():
                from train_detect import HeatmapNet
                ckpt = torch.load(path, map_location=self.device, weights_only=False)
                model = HeatmapNet().to(self.device)
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                self._detector = model
        return self._detector

    def scene(self):
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
        return self._scene, self._scene_classes

    def intensity(self):
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
            "scene_classifier": (ART / "scene_classifier.pt").exists(),
            "intensity_vision": ((ART / "intensity_gridsat.pt").exists()
                                 or (ART / "intensity_vision.pt").exists()),
            "intensity_domain": self.intensity_domain,
            "intensity_calibrated": self.intensity_cal != (1.0, 0.0),
            "gridsat_scenes": len(list(GRIDSAT.rglob("nio_*.nc"))),
        }


MODELS = VisionModels()


def nearest_scene(when: pd.Timestamp) -> pd.Timestamp | None:
    # the GridSat slot nearest a requested time, if it is on disk
    from vision.detect import gridsat_path

    slot = pd.Timestamp(when).round("3h")
    for shift in (0, -3, 3, -6, 6):
        cand = slot + pd.Timedelta(hours=shift)
        if gridsat_path(GRIDSAT, cand).exists():
            return cand
    return None


def detect(when: pd.Timestamp, threshold: float = 0.30) -> dict:
    # run T1 over a whole basin scene
    from vision.detect import BasinScenes, find_peaks, gridsat_path, STRIDE

    model = MODELS.detector()
    if model is None:
        return {"available": False,
                "reason": "detector not trained yet (artifacts/detector.pt missing)"}

    slot = nearest_scene(when)
    if slot is None:
        return {"available": False,
                "reason": f"no GridSat scene near {when}"}

    index = pd.DataFrame([{"time": slot, "season": slot.year, "n_storms": 0}])
    ds_wrap = BasinScenes(index, pd.DataFrame(columns=["ISO_TIME", "LAT", "LON"]),
                          GRIDSAT)
    # only the image is wanted here. The dataset also yields the target heatmap,
    # the per-storm loss weight map and a target count, all of which exist for
    # training. Unpacking positionally broke the moment the weight map was added
    # for intensity-weighted targets, so index rather than destructure.
    x = ds_wrap[0][0]
    with torch.no_grad():
        heat = torch.sigmoid(
            model(x.unsqueeze(0).to(MODELS.device)))[0, 0].cpu().numpy()

    with xr.open_dataset(gridsat_path(GRIDSAT, slot)) as g:
        lats, lons = g["lat"].values, g["lon"].values

    found = []
    for y, x_, score in find_peaks(heat, threshold):
        lat, lon = BasinScenes.from_pixel(y * STRIDE, x_ * STRIDE, lats, lons)
        found.append({"lat": lat, "lon": lon, "score": float(score)})

    return {"available": True, "scene_time": str(slot), "threshold": threshold,
            "detections": found,
            # verified on held-out seasons; surfaced so the output is never read
            # as more precise than it is.
            "skill_note": "held-out F1 and median centre-fix in /api/skill"}


def _as_tensor(patch: np.ndarray) -> torch.Tensor:
    # patch -> model input, with the same normalisation used in training
    from vision.scenes import IMAGENET_MEAN, IMAGENET_STD

    a = (patch.astype(np.float32) / 255.0).transpose(2, 0, 1)
    a = (a - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    return torch.from_numpy(np.ascontiguousarray(a)).unsqueeze(0)


def _patch_at(when: pd.Timestamp, lat: float, lon: float):
    from vision.scenes import extract_patch, gridsat_path

    slot = nearest_scene(when)
    if slot is None:
        return None, None
    with xr.open_dataset(gridsat_path(GRIDSAT, slot)) as ds:
        patch = extract_patch(ds, lat, lon)
    return patch, slot


def classify_scene(when: pd.Timestamp, lat: float, lon: float) -> dict:
    # run T2 on a storm-centred patch
    model, classes = MODELS.scene()
    if model is None:
        return {"available": False,
                "reason": "scene classifier not trained yet"}

    patch, slot = _patch_at(when, lat, lon)
    if patch is None:
        return {"available": False, "reason": f"no usable GridSat patch near {when}"}

    x = _as_tensor(patch)
    with torch.no_grad():
        probs = torch.softmax(model(x.to(MODELS.device)), dim=1)[0].cpu().numpy()

    order = np.argsort(-probs)
    return {
        "available": True, "scene_time": str(slot),
        "scene": classes[int(order[0])],
        "confidence": float(probs[order[0]]),
        "probabilities": {classes[i]: float(probs[i]) for i in order},
        "labels_note": "trained on CIMSS ADT scene types (algorithm output, "
                       "not analyst labels)",
    }


def estimate_intensity(when: pd.Timestamp, lat: float, lon: float) -> dict:
    # run T3 on a storm-centred patch
    model = MODELS.intensity()
    if model is None:
        return {"available": False,
                "reason": "intensity-from-imagery model not trained yet"}

    patch, slot = _patch_at(when, lat, lon)
    if patch is None:
        return {"available": False, "reason": f"no usable GridSat patch near {when}"}

    x = _as_tensor(patch)
    with torch.no_grad():
        raw = float(model(x.to(MODELS.device)).squeeze().cpu())
    # the GridSat model regresses a standardised target, then the calibration
    # undoes its shrinkage toward the mean.
    kt = raw * MODELS.intensity_scale[1] + MODELS.intensity_scale[0]
    slope, intercept = MODELS.intensity_cal
    kt = (kt - intercept) / max(slope, 1e-6)

    from ingest.ibtracs import imd_category
    kt = max(kt, 0.0)
    return {"available": True, "scene_time": str(slot),
            "vmax_kt": kt, "category": imd_category(kt)}


def scene_bounds(when: pd.Timestamp) -> dict | None:
    # geographic extent of a scene, without rendering it
    from vision.detect import gridsat_path

    slot = nearest_scene(when)
    if slot is None:
        return None
    with xr.open_dataset(gridsat_path(GRIDSAT, slot)) as ds:
        lats, lons = ds["lat"].values, ds["lon"].values
    return {"scene_time": str(slot),
            "bounds": {"west": float(lons.min()), "east": float(lons.max()),
                       "south": float(lats.min()), "north": float(lats.max())}}


def scene_image(when: pd.Timestamp, enhance: bool = True) -> tuple[bytes, dict] | None:
    # render a GridSat scene as a PNG for map overlay, with its bounds
    import io as _io

    from PIL import Image
    from vision.detect import gridsat_path

    slot = nearest_scene(when)
    if slot is None:
        return None

    with xr.open_dataset(gridsat_path(GRIDSAT, slot)) as ds:
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
