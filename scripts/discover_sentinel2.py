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
from backend.app.services.sentinel_service import discover_sentinel2_scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover candidate Copernicus Sentinel-2 scenes for the Gurugram AOI.")
    parser.add_argument("--aoi", default=str(settings.aoi_path), help="GeoJSON AOI file")
    parser.add_argument("--start-year", type=int, default=settings.default_start_year)
    parser.add_argument("--end-year", type=int, default=settings.default_end_year)
    parser.add_argument("--max-cloud", type=float, default=settings.default_cloud_max)
    parser.add_argument("--max-scenes", type=int, default=settings.max_default_scenes)
    parser.add_argument("--dry-run", action="store_true", help="Show discovery candidates without downloading assets")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    aoi = load_aoi(args.aoi)
    geom = aoi["features"][0]["geometry"]
    poly = geom["coordinates"][0]
    west = min(coord[0] for coord in poly)
    south = min(coord[1] for coord in poly)
    east = max(coord[0] for coord in poly)
    north = max(coord[1] for coord in poly)

    scenes = discover_sentinel2_scenes([west, south, east, north], start_year=args.start_year, end_year=args.end_year, max_cloud_cover=args.max_cloud, max_scenes=args.max_scenes)
    print(json.dumps({"discovery_mode": True, "dry_run": args.dry_run, "scene_count": len(scenes), "scenes": scenes}, indent=2))
    output_path = Path(settings.metadata_root) / "scenes_discovery.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"scenes": scenes}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
