"""Fetch ALL data from Truflation API (763 endpoints).

Pulls every available stream and saves to a single Parquet file
plus per-category CSVs for inspection.

Usage:
    python scripts/fetch_all_api_data.py

Requires API key in environment or hardcoded below.
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np

API_BASE = "https://api.truflation.com/api/v1/feed/truflation"
import os
API_KEY = os.environ.get("TRUFLATION_API_KEY", "")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "truflation" / "api"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ENDPOINTS_FILE = Path(__file__).parent.parent / "data" / "api_endpoints.json"


def fetch_endpoint(endpoint: str) -> dict | None:
    """Fetch a single endpoint. Returns {dates, values, name} or None on failure."""
    url = f"{API_BASE}/{endpoint}"
    headers = {"Authorization": API_KEY}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()

        if "index" not in data:
            return None

        dates = data["index"]
        # Find the value column (not 'index' or 'start_date')
        value_key = None
        for k in data:
            if k not in ("index", "start_date"):
                value_key = k
                break

        if value_key is None:
            return None

        return {
            "dates": dates,
            "values": data[value_key],
            "name": value_key,
            "endpoint": endpoint,
            "num_points": len(dates),
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
        }
    except Exception as e:
        print(f"  ERROR {endpoint}: {e}")
        return None


def main():
    # Load endpoint list
    with open(ENDPOINTS_FILE) as f:
        endpoints = json.load(f)

    print(f"Fetching {len(endpoints)} endpoints...")
    print(f"Output: {OUTPUT_DIR}")

    all_series = {}
    catalog = []
    failed = []
    categories = {}

    for i, ep in enumerate(endpoints):
        # Extract category and stream name from path
        # /api/v1/feed/truflation/labor-data-us/bls_unemployment_rate
        parts = ep.strip("/").split("/")
        if len(parts) >= 6:
            category = parts[5]
            stream = parts[6] if len(parts) > 6 else parts[5]
        else:
            category = "unknown"
            stream = ep

        result = fetch_endpoint(ep.lstrip("/").replace("api/v1/feed/truflation/", ""))

        if result is not None:
            series_name = f"{category}/{result['name']}"
            all_series[series_name] = result
            catalog.append({
                "category": category,
                "stream": result["name"],
                "endpoint": ep,
                "num_points": result["num_points"],
                "start_date": result["start"],
                "end_date": result["end"],
            })
            categories.setdefault(category, []).append(result["name"])
        else:
            failed.append(ep)

        # Progress
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(endpoints)}] fetched {len(all_series)} streams, {len(failed)} failed")

        # Rate limit — be nice to the API
        time.sleep(0.1)

    print(f"\nDone: {len(all_series)} streams fetched, {len(failed)} failed")

    # Save catalog
    catalog_df = pd.DataFrame(catalog)
    catalog_path = OUTPUT_DIR / "catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    print(f"Catalog: {catalog_path} ({len(catalog_df)} streams)")

    # Save per-category summary
    print(f"\nCategory summary:")
    for cat, streams in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:30s} {len(streams):4d} streams")

    # Build unified DataFrame — all series aligned to the same date index
    print("\nBuilding unified dataset...")
    all_dates = set()
    for s in all_series.values():
        all_dates.update(s["dates"])
    all_dates = sorted(all_dates)

    unified = pd.DataFrame({"date": all_dates})
    unified["date"] = pd.to_datetime(unified["date"], errors="coerce")
    unified = unified.dropna(subset=["date"])

    for name, s in all_series.items():
        series_df = pd.DataFrame({
            "date": pd.to_datetime(s["dates"], errors="coerce"),
            name: pd.to_numeric(s["values"], errors="coerce"),
        })
        series_df = series_df.dropna(subset=["date"])
        unified = unified.merge(series_df, on="date", how="left")

    unified = unified.sort_values("date").reset_index(drop=True)

    # Save unified dataset
    unified_path = OUTPUT_DIR / "all_streams.parquet"
    unified.to_parquet(unified_path, index=False)
    print(f"Unified: {unified_path} ({unified.shape[0]} rows x {unified.shape[1]} cols)")

    # Also save as CSV for inspection
    csv_path = OUTPUT_DIR / "all_streams.csv"
    unified.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")

    # Save failed endpoints
    if failed:
        failed_path = OUTPUT_DIR / "failed_endpoints.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"Failed: {failed_path} ({len(failed)} endpoints)")

    # Summary stats
    print(f"\n{'='*60}")
    print(f"  DATA COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Streams:    {len(all_series)}")
    print(f"  Date range: {unified['date'].min()} to {unified['date'].max()}")
    print(f"  Total data points: {unified.iloc[:, 1:].notna().sum().sum():,}")
    print(f"  Categories: {len(categories)}")
    print(f"  Failed:     {len(failed)}")

    # Save metadata
    meta = {
        "fetch_date": datetime.now().isoformat(),
        "total_endpoints": len(endpoints),
        "successful": len(all_series),
        "failed": len(failed),
        "date_range": [str(unified["date"].min()), str(unified["date"].max())],
        "total_data_points": int(unified.iloc[:, 1:].notna().sum().sum()),
        "categories": {k: len(v) for k, v in categories.items()},
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
