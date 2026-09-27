# check our CAP documents against the CAP 1.2 structure
#
# API.md claimed the alerts were "validated against the OASIS schema", and when
# it came time to add <references> to the alert block nothing in the repo could
# check that claim - there is no XSD here and no validator ever ran. a claim
# you cannot re-run is not a check, so this is the check.
#
# it is a structural test, not the OASIS XSD: required elements, the order the
# spec fixes them in, and the closed value lists. that is enough to catch the
# thing that actually breaks - an element added in the wrong place, which is
# exactly the mistake <references> invites, because it goes after <note> and
# not where you would guess.
#
#   python src/check_cap.py                 # every stored alert
#   python src/check_cap.py path/to.xml     # one file

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "alerts"
NS = "urn:oasis:names:tc:emergency:cap:1.2"

# CAP 1.2 fixes the order of the alert block. anything out of sequence is
# invalid even when every element is present.
ALERT_ORDER = ["identifier", "sender", "sent", "status", "msgType", "source",
               "scope", "restriction", "addresses", "code", "note", "references",
               "incidents", "info"]
ALERT_REQUIRED = ["identifier", "sender", "sent", "status", "msgType", "scope"]

INFO_ORDER = ["language", "category", "event", "responseType", "urgency", "severity",
              "certainty", "audience", "eventCode", "effective", "onset", "expires",
              "senderName", "headline", "description", "instruction", "web", "contact",
              "parameter", "resource", "area"]
INFO_REQUIRED = ["category", "event", "urgency", "severity", "certainty"]

# closed value lists, kept apart by the block they belong to. they were one
# dict at first and every info-level value silently went unchecked, because
# findall on <alert> never sees inside <info> - a deliberately wrong
# <severity> passed clean until a negative test caught it.
ALERT_VALUES = {
    "status": {"Actual", "Exercise", "System", "Test", "Draft"},
    "msgType": {"Alert", "Update", "Cancel", "Ack", "Error"},
    "scope": {"Public", "Restricted", "Private"},
}
INFO_VALUES = {
    "urgency": {"Immediate", "Expected", "Future", "Past", "Unknown"},
    "severity": {"Extreme", "Severe", "Moderate", "Minor", "Unknown"},
    "certainty": {"Observed", "Likely", "Possible", "Unlikely", "Unknown"},
    "category": {"Geo", "Met", "Safety", "Security", "Rescue", "Fire", "Health",
                 "Env", "Transport", "Infra", "CBRNE", "Other"},
}


def _tags(parent) -> list[str]:
    return [c.tag.split("}")[-1] for c in parent]


def _ordered(tags: list[str], order: list[str], where: str) -> list[str]:
    seen = [t for t in tags if t in order]
    ranks = [order.index(t) for t in seen]
    if ranks != sorted(ranks):
        out = [t for t, _ in sorted(zip(seen, ranks), key=lambda p: p[1])]
        return [f"{where}: elements out of CAP order, {seen} should be {out}"]
    return [f"{where}: unknown element <{t}>" for t in tags if t not in order]


def check(path: Path) -> list[str]:
    bad: list[str] = []
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError as exc:
        return [f"not well-formed XML: {exc}"]

    if root.tag != f"{{{NS}}}alert":
        bad.append(f"root is {root.tag}, expected a CAP 1.2 <alert>")
        return bad

    tags = _tags(root)
    bad += [f"alert: missing required <{t}>" for t in ALERT_REQUIRED if t not in tags]
    bad += _ordered(tags, ALERT_ORDER, "alert")

    for name, allowed in ALERT_VALUES.items():
        for el in root.findall(f"{{{NS}}}{name}"):
            if el.text not in allowed:
                bad.append(f"alert: <{name}> is {el.text!r}, not one of {sorted(allowed)}")

    # a prototype must never be able to emit something that reads as a live
    # public warning, so this is a hard check and not a style note
    status = root.findtext(f"{{{NS}}}status")
    if status != "Exercise":
        bad.append(f"alert: status is {status!r} - this prototype must only emit Exercise")

    if root.findtext(f"{{{NS}}}msgType") in {"Update", "Cancel"}:
        refs = root.findtext(f"{{{NS}}}references")
        if not refs:
            bad.append("alert: an Update or Cancel must carry <references>")
        elif len(refs.split(",")) != 3:
            bad.append(f"alert: <references> is {refs!r}, expected sender,identifier,sent")

    infos = root.findall(f"{{{NS}}}info")
    if not infos:
        bad.append("alert: no <info> block")
    for i, info in enumerate(infos):
        lang = info.findtext(f"{{{NS}}}language") or f"#{i}"
        itags = _tags(info)
        bad += [f"info[{lang}]: missing required <{t}>"
                for t in INFO_REQUIRED if t not in itags]
        bad += _ordered(itags, INFO_ORDER, f"info[{lang}]")
        for name, allowed in INFO_VALUES.items():
            for el in info.findall(f"{{{NS}}}{name}"):
                if el.text not in allowed:
                    bad.append(f"info[{lang}]: <{name}> is {el.text!r}, "
                               f"not one of {sorted(allowed)}")
        for area in info.findall(f"{{{NS}}}area"):
            poly = area.findtext(f"{{{NS}}}polygon")
            if poly:
                pts = poly.split()
                if len(pts) < 4:
                    bad.append(f"info[{lang}]: polygon has {len(pts)} points, needs 4")
                elif pts[0] != pts[-1]:
                    bad.append(f"info[{lang}]: polygon is not closed")
    return bad


def main() -> None:
    args = sys.argv[1:]
    files = [Path(a) for a in args] if args else sorted(STORE.glob("*.xml"))
    if not files:
        raise SystemExit("no CAP files - publish some with src/replay_alerts.py")

    bad_files = 0
    for path in files:
        problems = check(path)
        if problems:
            bad_files += 1
            print(f"FAIL {path.name}")
            for p in problems:
                print(f"       {p}")
    print(f"\n{len(files) - bad_files}/{len(files)} CAP documents well formed "
          f"against the CAP 1.2 structure")
    raise SystemExit(1 if bad_files else 0)


if __name__ == "__main__":
    main()
