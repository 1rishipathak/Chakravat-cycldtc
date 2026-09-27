# alert formats for disaster management: CAP 1.2 XML and a Hindi bulletin
#
# CAP (the OASIS Common Alerting Protocol) is the format behind NDMA's SACHET
# integrated alert system, which IMD already feeds. producing it means a
# forecast from here could travel the same pipe as every other alert in the
# country. every alert is marked status "Exercise" and says in plain words that
# it is not an IMD warning - a prototype must never be mistaken for the real
# thing on a channel built to reach phones.

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import timedelta

import pandas as pd

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
SENDER = "chakravat.prototype"
KMH_PER_KT = 1.852

# Hindi category names. अवदाब, गहन अवदाब, चक्रवाती तूफान and प्रचण्ड
# चक्रवाती तूफान are as IMD's own Hindi national bulletins print them. we could
# not find the three strongest in an IMD document, so they follow the same
# pattern and are unverified - which is why every Hindi category is printed
# with the English name beside it.
HINDI_CATEGORY = {
    "Low Pressure Area": "निम्न दबाव का क्षेत्र",
    "Depression": "अवदाब",
    "Deep Depression": "गहन अवदाब",
    "Cyclonic Storm": "चक्रवाती तूफान",
    "Severe Cyclonic Storm": "प्रचण्ड चक्रवाती तूफान",
    "Very Severe Cyclonic Storm": "अति प्रचण्ड चक्रवाती तूफान",
    "Extremely Severe Cyclonic Storm": "अत्यंत प्रचण्ड चक्रवाती तूफान",
    "Super Cyclonic Storm": "सुपर चक्रवाती तूफान",
}
CATEGORY_ORDER = list(HINDI_CATEGORY)

# CAP severity from the IMD category of the strongest point in the alert window
SEVERITY = {
    "Low Pressure Area": "Minor", "Depression": "Moderate", "Deep Depression": "Moderate",
    "Cyclonic Storm": "Severe", "Severe Cyclonic Storm": "Severe",
    "Very Severe Cyclonic Storm": "Extreme", "Extremely Severe Cyclonic Storm": "Extreme",
    "Super Cyclonic Storm": "Extreme",
}

NOT_IMD = ("Prototype decision support from Chakravat (SIH 2026, PS 26070). "
           "NOT an IMD warning. Official warnings are issued by RSMC New Delhi.")
NOT_IMD_HI = ("यह भारत मौसम विज्ञान विभाग (IMD) की चेतावनी नहीं है। यह केवल "
              "निर्णय सहायता हेतु एक प्रोटोटाइप है; आधिकारिक चेतावनियाँ RSMC "
              "नई दिल्ली द्वारा जारी की जाती हैं।")


def hindi_category(cat: str) -> str:
    return f"{HINDI_CATEGORY.get(cat, cat)} ({cat.upper()})"


def _peak(fc: dict) -> dict:
    # strongest of the present intensity and every forecast lead time
    cands = [fc["current"]] + fc["forecast"]
    return max(cands, key=lambda p: p["vmax_kt"])


def _cap_time(ts) -> str:
    # CAP wants an explicit offset, never "Z". this stays +00:00: it is a machine
    # format and the offset says so unambiguously either way.
    return pd.Timestamp(ts).tz_localize("UTC").strftime("%Y-%m-%dT%H:%M:%S+00:00")


IST = pd.Timedelta(hours=5, minutes=30)


def ist(ts, label: str = " IST") -> str:
    """UTC timestamp -> the local time an Indian bulletin is read in.

    Everything upstream is UTC, because best track sits on the synoptic hours
    and the satellite slots are UTC. IMD publishes in IST, so that is what a
    bulletin prints, and it always prints the label with it.
    """
    return (pd.Timestamp(ts) + IST).strftime("%Y-%m-%d %H:%M") + label


def _urgency(lf: dict) -> str:
    if lf.get("over_land"):
        return "Immediate"
    h = lf.get("hours_to_landfall")
    if h is None:
        return "Future"
    return "Immediate" if h <= 6 else ("Expected" if h <= 24 else "Future")


def _certainty(lf: dict) -> str:
    if lf.get("over_land"):
        return "Observed"
    p = lf.get("probability_within_72h", 0.0)
    return "Likely" if p > 0.5 else ("Possible" if p > 0.1 else "Unlikely")


def _outlook(lf: dict) -> str:
    if lf.get("over_land"):
        return "centre over land after landfall"
    if lf.get("hours_to_landfall") is not None:
        return f"landfall expected in about {lf['hours_to_landfall']:.0f} h"
    return "no landfall indicated within 72 h"


def _sub(parent, tag, text=None):
    el = ET.SubElement(parent, f"{{{CAP_NS}}}{tag}")
    if text is not None:
        el.text = str(text)
    return el


def _cap_polygon(poly) -> str:
    # CAP polygon: space-separated "lat,lon" pairs, first equal to last
    coords = list(poly.exterior.coords)
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return " ".join(f"{lat:.3f},{lon:.3f}" for lon, lat in coords)


def _english_text(fc, lf, places):
    cur, name = fc["current"], fc["name"] or "UNNAMED"
    peak = _peak(fc)
    lines = [f"{cur['category']} {name} at {cur['lat']:.1f}N {cur['lon']:.1f}E, "
             f"{cur['vmax_kt']:.0f} kt ({cur['vmax_kt'] * KMH_PER_KT:.0f} km/h, 3-min mean)."]
    lines.append(f"Peak forecast intensity {peak['vmax_kt']:.0f} kt ({peak['category']}) "
                 f"within 72 h.")
    if lf.get("over_land"):
        lines.append("The centre is over land: landfall has already occurred.")
    elif lf.get("hours_to_landfall") is not None:
        lines.append(f"Landfall probability {100 * lf['probability_within_72h']:.0f}% within 72 h; "
                     f"expected in about {lf['hours_to_landfall']:.0f} h "
                     f"(typical error {lf['typical_error']['timing_h']:.0f} h) near "
                     f"{lf['lat']:.1f}N {lf['lon']:.1f}E (typical error "
                     f"{lf['typical_error']['position_km']:.0f} km) at {lf['vmax_kt']:.0f} kt.")
    else:
        lines.append(f"No landfall indicated within 72 h "
                     f"(probability {100 * lf['probability_within_72h']:.0f}%).")
    if places:
        top = ", ".join(f"{p['name']} ({p['first_inside_h']:.0f} h)" for p in places[:6])
        lines.append(f"Places in the threat zone ({100 * fc['cone_level']:.0f}% cone widened by "
                     f"the typical gale radius): {top}.")
    return " ".join(lines)


def _hindi_text(fc, lf, places):
    cur, name = fc["current"], fc["name"] or "UNNAMED"
    peak = _peak(fc)
    lines = [f"{hindi_category(cur['category'])} {name}: {cur['lat']:.1f} उ. "
             f"{cur['lon']:.1f} पू. पर, {cur['vmax_kt']:.0f} नॉट "
             f"({cur['vmax_kt'] * KMH_PER_KT:.0f} किमी/घंटा)।"]
    lines.append(f"72 घंटे में अधिकतम पूर्वानुमानित तीव्रता {peak['vmax_kt']:.0f} नॉट "
                 f"({hindi_category(peak['category'])})।")
    if lf.get("over_land"):
        lines.append("केंद्र स्थल पर है: तट पार कर चुका है।")
    elif lf.get("hours_to_landfall") is not None:
        lines.append(f"72 घंटे में तट पार करने (लैंडफॉल) की संभावना "
                     f"{100 * lf['probability_within_72h']:.0f}%; लगभग "
                     f"{lf['hours_to_landfall']:.0f} घंटे में {lf['lat']:.1f} उ. "
                     f"{lf['lon']:.1f} पू. के निकट।")
    else:
        lines.append("72 घंटे में तट पार करने का संकेत नहीं।")
    if places:
        top = ", ".join(f"{p['name']} ({p['first_inside_h']:.0f} घंटे)" for p in places[:6])
        lines.append(f"खतरे के क्षेत्र में आने वाले स्थान: {top}।")
    return " ".join(lines)


def identifier_for(sid: str, issued) -> str:
    # stable and idempotent: the same storm at the same issue time is the same
    # alert, so replaying a forecast cannot mint a second copy of it
    return f"chakravat.{sid}.{pd.Timestamp(issued):%Y%m%d%H%M}"


def summary(fc: dict, lf: dict) -> dict:
    # the header fields a feed and an index need, without reaching into privates
    peak = _peak(fc)
    name = fc["name"] or "UNNAMED"
    return {
        "headline": f"{peak['category']} {name}: {_outlook(lf)} "
                    f"(prototype, not an IMD warning)",
        "severity": SEVERITY.get(peak["category"], "Unknown"),
        "urgency": _urgency(lf),
        "certainty": _certainty(lf),
    }


def cap_alert(sid: str, fc: dict, lf: dict, places: list[dict], cone,
              msg_type: str = "Alert", references: str | None = None,
              expires_h: float = 6.0) -> str:
    # one CAP 1.2 alert, English and Hindi info blocks, cone as the area.
    # msg_type and references carry the lifecycle: an Update names the alert it
    # supersedes, so a consumer sees one storm's story and not fifty strangers.
    ET.register_namespace("", CAP_NS)
    issued = pd.Timestamp(fc["issued_at"])
    peak = _peak(fc)
    name = fc["name"] or "UNNAMED"

    alert = ET.Element(f"{{{CAP_NS}}}alert")
    _sub(alert, "identifier", identifier_for(sid, issued))
    _sub(alert, "sender", SENDER)
    _sub(alert, "sent", _cap_time(issued))
    _sub(alert, "status", "Exercise")
    _sub(alert, "msgType", msg_type)
    _sub(alert, "scope", "Public")
    _sub(alert, "note", NOT_IMD)
    if references:
        # CAP 1.2 orders the alert block: ... note, references, incidents, info
        _sub(alert, "references", references)

    for lang, headline_fn, text_fn in (
            ("en-IN", lambda: f"{peak['category']} {name}: {_outlook(lf)}"
                              f" (prototype, not an IMD warning)",
             lambda: _english_text(fc, lf, places)),
            ("hi-IN", lambda: f"{hindi_category(peak['category'])} {name} "
                              f"(प्रोटोटाइप, IMD चेतावनी नहीं)",
             lambda: _hindi_text(fc, lf, places))):
        info = _sub(alert, "info")
        _sub(info, "language", lang)
        _sub(info, "category", "Met")
        _sub(info, "event", "Tropical Cyclone" if lang == "en-IN" else "उष्णकटिबंधीय चक्रवात")
        _sub(info, "urgency", _urgency(lf))
        _sub(info, "severity", SEVERITY.get(peak["category"], "Unknown"))
        _sub(info, "certainty", _certainty(lf))
        _sub(info, "effective", _cap_time(issued))
        _sub(info, "expires", _cap_time(issued + timedelta(hours=expires_h)))
        _sub(info, "senderName", "Chakravat prototype (not IMD)")
        _sub(info, "headline", headline_fn())
        _sub(info, "description", text_fn())
        _sub(info, "instruction",
             "Follow official warnings from IMD / RSMC New Delhi and your State and "
             "District Disaster Management Authority." if lang == "en-IN" else
             "IMD / RSMC नई दिल्ली तथा राज्य एवं ज़िला आपदा प्रबंधन प्राधिकरण की "
             "आधिकारिक चेतावनियों का पालन करें।")
        for key, value in (("vmax_kt_current", f"{fc['current']['vmax_kt']:.0f}"),
                           ("vmax_kt_peak_72h", f"{peak['vmax_kt']:.0f}"),
                           ("landfall_probability_72h", f"{lf['probability_within_72h']:.2f}"),
                           ("cone_confidence", f"{fc['cone_level']:.2f}")):
            param = _sub(info, "parameter")
            _sub(param, "valueName", key)
            _sub(param, "value", value)
        area = _sub(info, "area")
        _sub(area, "areaDesc", f"{100 * fc['cone_level']:.0f}% forecast cone, 0-72 h"
             if lang == "en-IN" else f"{100 * fc['cone_level']:.0f}% पूर्वानुमान शंकु, 0-72 घंटे")
        _sub(area, "polygon", _cap_polygon(cone))

    body = ET.tostring(alert, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body


def bulletin_hindi(sid: str, fc: dict, lf: dict, places: list[dict]) -> str:
    cur, name = fc["current"], fc["name"] or "UNNAMED"
    lines = [
        "=" * 66,
        f"चक्रवात निर्णय सहायता - {name} ({sid})",
        f"जारी: {ist(fc['issued_at'])}",
        "=" * 66,
        "",
        f"वर्तमान स्थिति  : {cur['lat']:.1f} उ.  {cur['lon']:.1f} पू.",
        f"वर्तमान तीव्रता : {cur['vmax_kt']:.0f} नॉट ({cur['vmax_kt'] * KMH_PER_KT:.0f} किमी/घंटा), 3-मिनट औसत",
        f"वर्गीकरण       : {hindi_category(cur['category'])}",
        "",
        "पूर्वानुमान मार्ग और तीव्रता",
        f"  {'समय':<8} {'स्थिति':<16} {'तीव्रता (नॉट)':<18} शंकु",
    ]
    for p in fc["forecast"]:
        lines.append(f"  {p['horizon_h']:>3} घंटे  {p['lat']:.1f}उ {p['lon']:.1f}पू     "
                     f"{p['vmax_kt']:.0f} ({p['vmax_lower']:.0f}-{p['vmax_upper']:.0f})"
                     f"{'':<8} ±{p['radius_km']:.0f} किमी")
    ri = fc["rapid_intensification"]
    lines += ["", "तीव्र गहनीकरण (24 घंटे में +30 नॉट)",
              f"  संभावना: {100 * ri['probability']:.0f}%"
              f"{'   *** निगरानी ***' if ri['flagged'] else ''}", "",
              "तट पार करना (लैंडफॉल)"]
    if lf.get("over_land"):
        lines.append("  केंद्र स्थल पर है - तट पार कर चुका है।")
    elif lf.get("hours_to_landfall") is not None:
        lines += [
            f"  72 घंटे में संभावना : {100 * lf['probability_within_72h']:.0f}%",
            f"  अपेक्षित समय       : {lf['hours_to_landfall']:.0f} घंटे में "
            f"(± {lf['typical_error']['timing_h']:.0f} घंटे)",
            f"  अपेक्षित स्थान      : {lf['lat']:.1f} उ. {lf['lon']:.1f} पू. "
            f"(± {lf['typical_error']['position_km']:.0f} किमी)",
            f"  तट पर तीव्रता       : {lf['vmax_kt']:.0f} नॉट, "
            f"{hindi_category(lf['category_at_landfall'])}",
        ]
    else:
        lines.append(f"  72 घंटे में तट पार करने का संकेत नहीं "
                     f"(संभावना {100 * lf['probability_within_72h']:.0f}%)")
    if places:
        lines += ["", "खतरे के क्षेत्र में आने वाले प्रमुख स्थान"]
        for p in places[:8]:
            lines.append(f"  {p['name']:<18} {p['first_inside_h']:>4.0f} घंटे में")
    lines += ["", "-" * 66,
              "खतरे का क्षेत्र = पूर्वानुमान शंकु + तीव्रता के अनुसार आँधी (34 नॉट) की सामान्य त्रिज्या।",
              f"शंकु और तीव्रता सीमाएँ {100 * fc['cone_level']:.0f}% विश्वास स्तर पर "
              "अंशांकित हैं, 2020-2025 के अनदेखे मौसमों पर सत्यापित।", "",
              NOT_IMD_HI, "=" * 66]
    return "\n".join(lines)


# --------------------------------------------------------------------- SMS
# CAP reaches people through SACHET, which sends SMS. an SMS is not a short
# bulletin: it is a hard character budget that depends on the script. the GSM
# 03.38 alphabet packs 160 characters into one message, but Devanagari is not
# in that alphabet, so a Hindi message is sent as UCS-2 and gets 70. that is
# the real constraint on the last mile, and it is why the Hindi text below is
# not a translation of the English one.
GSM7 = set("@\u00a3$\u00a5\u00e8\u00e9\u00f9\u00ec\u00f2\u00c7\n\u00d8\u00f8\r\u00c5\u00e5"
           "\u0394_\u03a6\u0393\u039b\u03a9\u03a0\u03a8\u03a3\u0398\u039e\u00c6\u00e6"
           "\u00df\u00c9 !\"#\u00a4%&'()*+,-./0123456789:;<=>?\u00a1"
           "ABCDEFGHIJKLMNOPQRSTUVWXYZ\u00c4\u00d6\u00d1\u00dc\u00a7\u00bf"
           "abcdefghijklmnopqrstuvwxyz\u00e4\u00f6\u00f1\u00fc\u00e0")
GSM7_EXT = set("^{}\\[~]|\u20ac")          # these cost two characters each


def sms_encoding(text: str) -> dict:
    """How a network would actually send this string."""
    if set(text) <= (GSM7 | GSM7_EXT):
        units = sum(2 if c in GSM7_EXT else 1 for c in text)
        single, multi, enc = 160, 153, "GSM-7"
    else:
        units, single, multi, enc = len(text), 70, 67, "UCS-2"
    segments = 1 if units <= single else -(-units // multi)
    return {"encoding": enc, "units": units, "segments": segments,
            "budget": single if units <= single else multi * segments}


def _assemble(parts, probe, max_segments):
    """Join fragments, dropping optional ones from the end until it fits.

    Fragments are (text, essential). Essential ones are never dropped and never
    cut: a half-written instruction to evacuate is worse than a longer message.
    If the essential parts alone do not fit, the message runs to more segments
    and the caller is told so.
    """
    keep = list(parts)
    while True:
        text = " ".join(t for t, _ in keep if t)
        if probe(text)["segments"] <= max_segments:
            return text
        droppable = [i for i, (_, essential) in enumerate(keep) if not essential]
        if not droppable:
            return text
        del keep[droppable[-1]]


def sms(fc: dict, lf: dict, lang: str = "en", max_segments: int | None = None) -> dict:
    """One alert, sized for SMS.

    English fits a single 160-character GSM-7 message. Hindi is sent as UCS-2 at
    70 characters a segment, so it is written shorter and is still allowed two
    segments by default; the result says which it used.
    """
    name = (fc["name"] or "UNNAMED").title()
    cur = fc["current"]
    hours = lf.get("hours_to_landfall")
    coast_kt = lf.get("vmax_kt")          # intensity where the track crosses the coast
    if max_segments is None:
        max_segments = 2 if lang == "hi" else 1

    if lang == "hi":
        parts = [(f"चक्रवात {name}:", True)]
        if lf.get("over_land"):
            parts += [("तट पार कर चुका।", True), ("तेज़ हवा-बारिश जारी।", False),
                      ("घर के अंदर रहें।", True)]
        elif hours is not None:
            kt = f" {coast_kt:.0f} नॉट" if coast_kt else ""
            parts += [(f"{hours:.0f} घंटे में तट पर{kt}।", True),
                      ("सुरक्षित जगह जाएं।", True)]
        else:
            parts += [(f"{cur['vmax_kt']:.0f} नॉट, समुद्र में।", True),
                      ("मछुआरे समुद्र में न जाएं।", True)]
        parts.append(("IMD चेतावनी नहीं।", True))
    else:
        heads = [f"CYCLONE {name.upper()} ({cur['category']}):", f"CYCLONE {name.upper()}:"]
        parts = [(heads[0], True)]
        if lf.get("over_land"):
            parts += [("centre inland after landfall.", True),
                      ("Damaging wind and rain continue.", False),
                      ("Stay indoors, follow local orders.", True)]
        elif hours is not None:
            kt = f" at {coast_kt:.0f}kt" if coast_kt else ""
            parts += [(f"landfall in about {hours:.0f}h{kt}.", True),
                      ("Move to safety now, follow local orders.", True)]
        else:
            parts += [(f"{cur['vmax_kt']:.0f}kt at {cur['lat']:.1f}N {cur['lon']:.1f}E.", True),
                      ("No landfall indicated in 72h.", True),
                      ("Mariners stay in port.", False)]
        parts.append(("Prototype, not an IMD warning.", True))
        # drop the category from the head before dropping anything that matters
        if sms_encoding(" ".join(t for t, _ in parts))["segments"] > max_segments:
            parts[0] = (heads[1], True)

    text = _assemble(parts, sms_encoding, max_segments)
    out = sms_encoding(text)
    out.update({"text": text, "lang": lang, "characters": len(text),
                "fits_requested_segments": out["segments"] <= max_segments})
    return out
