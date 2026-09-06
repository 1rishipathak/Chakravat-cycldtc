# capture every API response to disk as JSON fixtures

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fixtures"

# storms chosen so the fixtures cover the shapes a frontend has to branch on,
# not just the happy path: one that makes landfall and one that does not.
AMPHAN = "2020136N10088"        # 130 kt, full imagery coverage, no landfall left
MANDOUS = "2022338N05100"       # lands near Chennai, exercises the landfall shape


def fetch(base: str, path: str, text: bool = False):
    url = base + path
    try:
        with urllib.request.urlopen(url, timeout=900) as r:
            raw = r.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return {"_error": f"HTTP {exc.code}", "_url": path}
    except Exception as exc:                                  # noqa: BLE001
        return {"_error": type(exc).__name__, "_url": path}
    return raw if text else json.loads(raw)


def write(name: str, payload, text: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / name
    if text:
        dest.write_text(payload, encoding="utf-8")
    else:
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    size = dest.stat().st_size
    flag = "" if not isinstance(payload, dict) or "_error" not in payload \
        else f"   <- {payload['_error']}"
    print(f"  {name:<34} {size/1024:7.1f} KB{flag}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="skip the slow imagery-chain call")
    args = ap.parse_args()
    b = args.base

    print(f"capturing from {b} into {OUT.relative_to(ROOT)}/\n")

    write("seasons.json", fetch(b, "/api/seasons"))
    write("storms_2020.json", fetch(b, "/api/storms?season=2020"))
    write("storm_amphan.json", fetch(b, f"/api/storm/{AMPHAN}"))

    for lvl in ("0.5", "0.67", "0.9"):
        write(f"forecast_amphan_{lvl.replace('.', '')}.json",
              fetch(b, f"/api/storm/{AMPHAN}/forecast?level={lvl}"))

    # both landfall shapes. The frontend must branch on hours_to_landfall being
    # null, so it needs an example of each.
    write("landfall_lands.json", fetch(b, f"/api/storm/{MANDOUS}/landfall"))
    write("landfall_none.json", fetch(b, f"/api/storm/{AMPHAN}/landfall"))

    write("bulletin_amphan.txt", fetch(b, f"/api/storm/{AMPHAN}/bulletin", text=True),
          text=True)

    write("scene_amphan.json", fetch(b, f"/api/storm/{AMPHAN}/scene"))
    write("intensity_amphan.json",
          fetch(b, f"/api/storm/{AMPHAN}/intensity_from_image"))
    write("vision_status.json", fetch(b, "/api/vision/status"))
    write("scene_meta.json",
          fetch(b, "/api/vision/scene_meta?time="
                   + urllib.parse.quote("2020-05-18 15:00")))
    write("skill.json", fetch(b, "/api/skill"))

    if args.no_pipeline:
        print("  (pipeline skipped)")
    else:
        print("  running the imagery chain, ~30 s ...")
        when = urllib.parse.quote("2020-05-19 12:00")
        write("pipeline.json",
              fetch(b, f"/api/vision/pipeline?time={when}&level=0.67"))

    print(f"\ndone. Point the frontend at {OUT.name}/ and build with no backend.")


if __name__ == "__main__":
    main()
