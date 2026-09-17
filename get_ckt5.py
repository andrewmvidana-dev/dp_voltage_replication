"""Download the pinned public EPRI Ckt5 OpenDSS model into ``feeders/ckt5``.

The feeder directory is ignored by Git, matching the existing IEEE 123 download
workflow. The model is fetched from a pinned public revision so a run can be
reproduced without silently changing feeder data.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path


REVISION = "5005c668a72d20775f4c2d060feebb2866ba1d38"
BASE = (
    "https://raw.githubusercontent.com/tshort/OpenDSS/"
    f"{REVISION}/Distrib/EPRITestCircuits/ckt5"
)
FILES = (
    "Buscoords_ckt5.dss",
    "Capacitors_ckt5.dss",
    "Generators_ckt5.dss",
    "LineCodes_ckt5.dss",
    "LineGeometry_ckt5.dss",
    "Lines_ckt5.dss",
    "Loads_ckt5.dss",
    "Loadshapes_ckt5.dss",
    "Master_ckt5.dss",
    "Regulators_ckt5.dss",
    "Transformers_ckt5.dss",
    "WireData_ckt5.dss",
    "XFR_Loads_ckt5.dss",
)


def main() -> None:
    """Download every file, refusing to replace an existing model."""
    destination = Path("feeders") / "ckt5"
    destination.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = destination / name
        if target.exists():
            print(f"exists {target}")
            continue
        url = f"{BASE}/{name}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                target.write_bytes(response.read())
        except (urllib.error.URLError, OSError) as exc:
            if target.exists():
                target.unlink()
            raise RuntimeError(f"failed to download {url}: {exc}") from exc
        print(f"downloaded {target} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
