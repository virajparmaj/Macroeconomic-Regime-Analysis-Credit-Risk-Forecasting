"""Explicit public-data acquisition; never silently substitute a different index."""

import json
import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .data import PROTOCOL, ROOT, digest


def fetch_public():
    out = ROOT / "data/research_external"
    out.mkdir(parents=True, exist_ok=True)
    report = {"retrieved_at": datetime.now(timezone.utc).isoformat(), "sources": {}, "vintage": {}}
    for sid in PROTOCOL["source_ids"]:
        if sid == "EA19LORSGPORGYSAM":
            continue
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?" + urlencode({"id": sid})
        try:
            request = Request(url, headers={"User-Agent": "CreditSpreadResearch/1.0"})
            with urlopen(request, timeout=30) as response:
                text = response.read().decode()
            frame = pd.read_csv(StringIO(text))
            if sid not in frame or len(frame.columns) != 2:
                raise ValueError("Unexpected public export schema")
            frame.columns = ["observation_date", sid]
            frame.observation_date = pd.to_datetime(frame.observation_date)
            frame = frame.sort_values("observation_date")
            path = out / f"{sid}.csv"
            frame.to_csv(path, index=False)
            report["sources"][sid] = {
                "status": "downloaded",
                "first": str(frame.observation_date.min().date()),
                "last": str(frame.observation_date.max().date()),
                "rows": len(frame),
                "sha256": digest(path),
                "source": url,
            }
        except Exception as exc:
            report["sources"][sid] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {str(exc)[:180]}",
            }
        print(sid, report["sources"][sid]["status"], flush=True)
    # Test vintage access without exposing keys. Missing credentials are not fabricated vintages.
    for sid in ["CPIAUCSL", "INDPRO", "UNRATE"]:
        params = {
            "series_id": sid,
            "file_type": "json",
            "realtime_start": "2008-12-31",
            "realtime_end": "2008-12-31",
        }
        key = os.environ.get("FRED_API_KEY")
        if key:
            params["api_key"] = key
        try:
            with urlopen(
                "https://api.stlouisfed.org/fred/series/observations?" + urlencode(params),
                timeout=20,
            ) as response:
                payload = json.load(response)
            observations = payload.get("observations", [])
            if not observations:
                raise ValueError("No vintage observations returned")
            pd.DataFrame(observations).to_csv(out / f"vintage_{sid}_2008-12-31.csv", index=False)
            report["vintage"][sid] = {
                "status": "snapshot_downloaded",
                "rows": len(observations),
                "limitation": "One snapshot is not a complete historical vintage panel",
            }
        except Exception as exc:
            # Exception URL text could include a supplied API key; retain only the exception class.
            report["vintage"][sid] = {
                "status": "unavailable",
                "reason": type(exc).__name__,
                "credential_configured": bool(key),
            }
    (out / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    fetch_public()
