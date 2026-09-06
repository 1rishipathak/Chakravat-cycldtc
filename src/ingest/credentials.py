# read credentials from a file outside the repository

from __future__ import annotations

import os
from pathlib import Path

# only ever read from here. Override with CHAKRAVAT_MOSDAC_CREDS if the file
# lives somewhere else.
DEFAULT_MOSDAC_PATH = Path(os.environ.get("CHAKRAVAT_MOSDAC_CREDS",
                                          r"D:\mosdac_creds.txt"))

_QUOTES = ("'", '"')


def load(path: Path | str) -> dict[str, str]:
    # parse a key=value credential file
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create it with one key=value per line; see "
            f"the module docstring for the format.")

    creds: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for sep in ("=", ":"):
            if sep in line:
                key, value = line.split(sep, 1)
                break
        else:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTES:
            value = value[1:-1]
        creds[key.strip().lower()] = value

    if not creds:
        raise ValueError(
            f"{path} is empty or has no key=value lines. Expected something "
            f"like:\n    username=...\n    password=...")
    return creds


def require(creds: dict[str, str], *keys: str) -> tuple[str, ...]:
    # fetch required keys, naming the missing ones without echoing values
    missing = [k for k in keys if not creds.get(k)]
    if missing:
        raise KeyError(f"missing or blank in credential file: "
                       f"{', '.join(missing)} (found: {sorted(creds)})")
    return tuple(creds[k] for k in keys)


def describe(path: Path | str = DEFAULT_MOSDAC_PATH) -> str:
    # human-readable confirmation that never reveals a value
    path = Path(path)
    if not path.exists():
        return f"{path}: NOT FOUND"
    try:
        creds = load(path)
    except ValueError as exc:
        return f"{path}: {exc}"
    parts = ", ".join(f"{k}=<{len(v)} chars>" for k, v in sorted(creds.items()))
    return f"{path}: OK - {parts}"


if __name__ == "__main__":
    print(describe())
