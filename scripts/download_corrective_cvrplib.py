"""Download six declared public CVRPLIB instances and their reference solutions."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path("data/raw/cvrplib/corrective_20260930")
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, number in [
        ("A-n32-k5", 4),
        ("A-n53-k7", 20),
        ("B-n51-k7", 43),
        ("E-n51-k5", 60),
        ("E-n101-k8", 65),
        ("P-n21-k2", 78),
    ]:
        for kind, extension in [("instance", "vrp"), ("bks", "sol")]:
            url = f"https://galgos.inf.puc-rio.br/cvrplib/index.php/en/download/{kind}/{number}"
            destination = root / f"{name}.{extension}"
            if destination.with_suffix(destination.suffix + ".unavailable.txt").exists():
                continue
            if not destination.exists():
                request = urllib.request.Request(
                    url, headers={"User-Agent": "VRP research reproducibility"}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                if b"<html" in payload.lower():
                    raise ValueError("server returned HTML instead of benchmark data")
                if kind == "instance" and (b"NAME" not in payload or b"EOF" not in payload):
                    raise ValueError("incomplete instance data")
                if kind == "bks" and b"Cost" not in payload:
                    raise ValueError("reference solution is unavailable")
                destination.write_bytes(payload)
            entries.append(
                {
                    "file": destination.name,
                    "url": url,
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                }
            )
        print(name, flush=True)
    (root / "download_manifest.json").write_text(json.dumps(entries, indent=2) + "\n")


if __name__ == "__main__":
    main()
