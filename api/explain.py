# where the imagery models look: Grad-CAM on the last ConvNeXt stage
#
# a forecaster can't trust a pattern call they can't check. Grad-CAM weights the
# last stage's feature maps by how much each one moves the answer, giving a
# coarse map (7x7 cells at 224 px, one cell about 55 km) of the regions that
# drove it. coarse is enough for the question that matters: is the model
# looking at the storm, or at something else in the frame.
#
# for T2 the map explains the CNN half of the hybrid; the cold-cloud half is a
# set of temperature statistics with no spatial map to draw.

from __future__ import annotations

import io

import numpy as np
import torch

# 150 km around the centre, in patch pixels at 0.07 degrees (~7.8 km)
CORE_KM = 150.0
KM_PER_PX = 0.07 * 111.2


def grad_cam(model, x: torch.Tensor, score_fn) -> tuple[np.ndarray, torch.Tensor]:
    # normalised Grad-CAM over the model's last stage, upsampled to the input
    store = {}

    def hook(_module, _inp, out):
        # only the explain pass itself; anything else running through the
        # model at the same moment has no gradient to keep
        if out.requires_grad:
            out.retain_grad()
            store["a"] = out

    handle = model.stages[-1].register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = model(x)
            score = score_fn(out)
            model.zero_grad(set_to_none=True)
            score.backward()
        a, g = store["a"], store["a"].grad
        weights = g.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * a).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, size=x.shape[-2:], mode="bilinear",
                                              align_corners=False)[0, 0]
        cam = cam / (cam.max() + 1e-8)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    return cam.detach().cpu().numpy(), out.detach()


def core_share(cam: np.ndarray) -> dict:
    # how much of the attention sits within CORE_KM of the patch centre
    n = cam.shape[0]
    yy, xx = np.mgrid[0:n, 0:n]
    r_px = CORE_KM / KM_PER_PX
    disc = np.hypot(yy - n / 2, xx - n / 2) <= r_px
    total = float(cam.sum()) or 1.0
    return {"core_km": CORE_KM, "share_in_core": float(cam[disc].sum() / total),
            "core_area_share": float(disc.mean())}


def render(patch: np.ndarray, cam: np.ndarray) -> bytes:
    # two panels: the infrared patch, and the same patch under the heat map
    from matplotlib import colormaps
    from PIL import Image, ImageDraw

    # infrared channel is scaled cold = 0; show cold cloud white, north up
    ir = 255 - patch[:, :, 0].astype(np.float32)
    ir = ir[::-1]
    heat = cam[::-1]
    grey = np.dstack([ir, ir, ir]) / 255.0
    rgba = colormaps["inferno"](heat)[..., :3]
    alpha = (0.15 + 0.7 * heat)[..., None]
    blend = grey * (1 - alpha) + rgba * alpha

    left = Image.fromarray((grey * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR)
    right = Image.fromarray((blend * 255).astype(np.uint8)).resize((448, 448), Image.BILINEAR)
    canvas = Image.new("RGB", (448 * 2 + 6, 448), (255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (454, 0))
    draw = ImageDraw.Draw(canvas)
    r = CORE_KM / KM_PER_PX * 2          # the core ring, at display scale
    for x0 in (0, 454):
        cx, cy = x0 + 224, 224
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(80, 200, 255), width=2)
        draw.line([cx - 8, cy, cx + 8, cy], fill=(80, 200, 255), width=2)
        draw.line([cx, cy - 8, cx, cy + 8], fill=(80, 200, 255), width=2)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
