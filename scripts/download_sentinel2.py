from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.services.aoi_service import load_aoi
from backend.app.services.download_service import download_and_validate_scene
from backend.app.services.sentinel_service import discover_sentinel2_scenes, select_best_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and validate one CDSE Sentinel-2 L2A scene for Gurugram.")
    parser.add_argument("--aoi", default=str(settings.aoi_path), help="GeoJSON AOI file")
    parser.add_argument("--scene", nargs="?", const="best", default="best", metavar="SCENE_ID", help="Download the best eligible scene, or select an exact STAC scene ID")
    parser.add_argument("--max-scenes", type=int, default=1, help="Kept for compatibility; this workflow is limited to exactly one scene")
    parser.add_argument("--dry-run", action="store_true", help="Select and report a scene without authenticating or downloading")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_scenes != 1:
        raise SystemExit("This first-download workflow accepts exactly one scene; use --max-scenes 1.")
    aoi = load_aoi(args.aoi)
    geometry = aoi["features"][0]["geometry"]
    coordinates = geometry["coordinates"][0]
    west = min(point[0] for point in coordinates)
    south = min(point[1] for point in coordinates)
    east = max(point[0] for point in coordinates)
    north = max(point[1] for point in coordinates)
    scenes = discover_sentinel2_scenes(
        [west, south, east, north],
        start_year=settings.default_start_year,
        end_year=settings.default_end_year,
        max_cloud_cover=settings.default_cloud_max,
        max_scenes=settings.max_default_scenes,
    )
    if not scenes:
        print("No suitable Sentinel-2 scenes were found for the current AOI and cloud filters.")
        return 1
    candidates = scenes
    if args.scene != "best":
        candidates = [scene for scene in scenes if scene["scene_id"] == args.scene]
        if not candidates:
            print(f"Scene ID was not returned by current STAC discovery: {args.scene}")
            return 1
    try:
        scene = select_best_scene(candidates, aoi)
    except ValueError as error:
        print(f"Scene selection failed: {error}")
        return 1

    print(f"Scene: {scene['scene_id']}")
    print(f"Date: {scene['acquisition_date']}")
    print(f"Tile: {scene['tile']}")
    print(f"Cloud: {scene['cloud_cover']}%")
    print(f"Product ID: {scene['product_id']}")
    print(f"Product: {scene['product_identifier']}")
    if args.dry_run:
        print("Download: SKIPPED (--dry-run)")
        return 0

    try:
        result = download_and_validate_scene(scene, aoi)
    except Exception as error:
        print(f"Download: FAILED ({type(error).__name__})")
        print(str(error))
        return 2

    validation = result["validation"]
    print("Download: SUCCESS")
    print(f"Validation: {validation['status']}")
    print(f"Bands: {', '.join(validation['bands'])}")
    first_band = next(iter(validation["bands"].values()))
    print(f"CRS: {first_band['crs']}")
    print(f"Size: {result['provenance']['file_size_bytes']} bytes")
    print(f"Local: {result['provenance']['local_product_path']}")
    print(f"Provenance: {result['provenance_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
