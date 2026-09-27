"""Authenticated access to NASA Earthdata, without the credentials leaving here.

Earthdata Login is an OAuth redirect chain: a request to a GES DISC file is
bounced to urs.earthdata.nasa.gov, authenticated there, and bounced back with a
cookie. `requests` deliberately drops the Authorization header on a cross-host
redirect, which is the right default and the reason the obvious three-line
version of this returns 401 forever. The session below re-attaches it for the
login host only.

Credentials are read from a netrc-format file and are never logged, printed or
written anywhere else. Pass a different path with EARTHDATA_NETRC if needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

HOME = Path(os.path.expanduser("~"))
CANDIDATES = [
    Path(os.environ["EARTHDATA_NETRC"]) if os.environ.get("EARTHDATA_NETRC") else None,
    HOME / "projects-artifact" / "nasa_earthdata_creds.txt",
    HOME / ".netrc",
    HOME / "_netrc",
]
LOGIN_HOST = "urs.earthdata.nasa.gov"


def _parse(path: Path) -> tuple[str, str] | None:
    # a netrc line: machine <host> login <user> password <secret>
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for block in text.replace("\n", " ").split("machine"):
        words = block.split()
        if not words or LOGIN_HOST not in words[0]:
            continue
        got = {}
        for key in ("login", "password"):
            if key in words:
                got[key] = words[words.index(key) + 1]
        if "login" in got and "password" in got:
            return got["login"], got["password"]
    return None


def credentials() -> tuple[str, str]:
    for path in CANDIDATES:
        if path and path.exists():
            found = _parse(path)
            if found:
                return found
    raise SystemExit(
        "no Earthdata credentials found. expected a netrc-format line\n"
        "  machine urs.earthdata.nasa.gov login <user> password <secret>\n"
        "in ~/projects-artifact/nasa_earthdata_creds.txt, ~/.netrc, or the file "
        "named by EARTHDATA_NETRC")


def session():
    """A requests session that survives the Earthdata redirect chain."""
    import requests

    user, secret = credentials()

    class EarthdataSession(requests.Session):
        def rebuild_auth(self, prepared, response):
            # the default drops auth across hosts; put it back for the login
            # host only, and never for anywhere else
            headers = prepared.headers
            if "Authorization" in headers:
                original = urlparse(response.request.url).hostname
                target = urlparse(prepared.url).hostname
                if target != original and target != LOGIN_HOST \
                        and original != LOGIN_HOST:
                    del headers["Authorization"]

    s = EarthdataSession()
    s.auth = (user, secret)
    s.headers.update({"User-Agent": "chakravat-research/1.0"})
    del user, secret
    return s


def check() -> bool:
    """One small authenticated request, so a wrong password fails loudly here."""
    s = session()
    url = ("https://data.gesdisc.earthdata.nasa.gov/data/GPM_L1C/"
           "GPM_1CGPMGMI.07/2020/139/")
    r = s.get(url, timeout=90, allow_redirects=True)
    if r.status_code == 200:
        print("Earthdata login OK")
        return True
    if r.status_code in (401, 403):
        print(f"Earthdata rejected the login ({r.status_code}).")
        print("if the password is right, the usual cause is the archive not")
        print("being authorised: urs.earthdata.nasa.gov/profile -> Applications")
        print("-> Authorized Apps -> approve NASA GESDISC DATA ARCHIVE")
        return False
    print(f"unexpected status {r.status_code} from {url}")
    return False


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
