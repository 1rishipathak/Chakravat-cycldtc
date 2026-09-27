# the subscription point: an Atom feed of CAP alerts
#
# this is how CAP actually travels. a consumer - NDMA's SACHET, a state
# emergency operations centre, Google Public Alerts, anyone - is given one URL
# and polls it. each entry points at a CAP document that never moves. it is a
# boring, twenty-year-old pattern, and that is exactly why it is the right one:
# every alerting authority already knows how to read it, so nothing has to be
# built on their side to consume us.
#
# we publish the feed. we do not push to the public: in India that is IMD's
# job, through NDMA's channels. see limitations.md.

from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd

ATOM_NS = "http://www.w3.org/2005/Atom"
CAP_MIME = "application/cap+xml"
TITLE = "Chakravat cyclone alerts (prototype, Exercise only)"
SUBTITLE = ("Decision-support alerts from Chakravat, SIH 2026 PS 26070. Every alert "
            "carries CAP status Exercise. These are NOT IMD warnings: official "
            "cyclone warnings for the North Indian Ocean are issued by RSMC New Delhi.")


def _sub(parent, tag, text=None, **attrib):
    el = ET.SubElement(parent, f"{{{ATOM_NS}}}{tag}", attrib)
    if text is not None:
        el.text = str(text)
    return el


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def atom(entries: list[dict], base: str) -> str:
    # newest first, which is what a poller wants to read
    base = base.rstrip("/")
    ET.register_namespace("", ATOM_NS)
    ordered = sorted(entries, key=lambda e: e["sent"], reverse=True)

    feed = ET.Element(f"{{{ATOM_NS}}}feed")
    _sub(feed, "id", f"{base}/api/alerts/feed.atom")
    _sub(feed, "title", TITLE)
    _sub(feed, "subtitle", SUBTITLE)
    _sub(feed, "updated", _iso(ordered[0]["sent"]) if ordered else _iso(pd.Timestamp.utcnow()))
    _sub(feed, "link", rel="self", href=f"{base}/api/alerts/feed.atom",
         type="application/atom+xml")
    author = _sub(feed, "author")
    _sub(author, "name", "Chakravat prototype (not IMD)")

    for e in ordered:
        url = f"{base}/api/alerts/{e['identifier']}.xml"
        entry = _sub(feed, "entry")
        _sub(entry, "id", f"urn:oid:{e['identifier']}")
        _sub(entry, "title", f"[{e['msgType']}] {e['headline']}")
        _sub(entry, "updated", _iso(e["sent"]))
        # the CAP document itself, at a URL that will not move
        _sub(entry, "link", rel="alternate", href=url, type=CAP_MIME)
        _sub(entry, "summary", f"{e['severity']} / {e['urgency']} / {e['certainty']}. "
                               f"Issued because: {e['reason']}.")
        cat = _sub(entry, "category")
        cat.set("term", e["sid"])
        cat.set("label", "storm")

    body = ET.tostring(feed, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body
