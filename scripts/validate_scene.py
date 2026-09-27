from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.services.aoi_service import load_aoi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate AOI and metadata structure for a local scene archive.")
    parser.add_argument("--aoi", default=str(settings.aoi_path), help="GeoJSON AOI file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_aoi(args.aoi)
    feature = payload["features"][0]
    geometry = feature["geometry"]
    if geometry["type"] != "Polygon":
        raise ValueError("AOI geometry must be a Polygon")
    if not feature["properties"].get("source_method"):
        raise ValueError("AOI metadata is missing source method")

    print(json.dumps({"status": "valid", "aoi": feature["properties"].get("name"), "source": feature["properties"].get("source")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
