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
from backend.app.services.sentinel_service import discover_sentinel2_scenes


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 scenes for the GeoIntel temporal benchmark."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover and display selected scenes without downloading.",
    )

    args = parser.parse_args()

    aoi = load_aoi(settings.aoi_path)

    geometry = aoi["features"][0]["geometry"]
    coordinates = geometry["coordinates"][0]

    west = min(point[0] for point in coordinates)
    south = min(point[1] for point in coordinates)
    east = max(point[0] for point in coordinates)
    north = max(point[1] for point in coordinates)

    print("Discovering temporal benchmark scenes...")

    scenes = discover_sentinel2_scenes(
        [west, south, east, north],
        start_year=2022,
        end_year=2026,
        max_cloud_cover=30.0,
        max_scenes=5,
    )

    if not scenes:
        print("No suitable scenes found.")
        return 1

    print()
    print(f"Selected scenes: {len(scenes)}")
    print("-" * 100)

    for scene in scenes:
        print(
            f"{scene['acquisition_date'][:10]} | "
            f"cloud={scene['cloud_cover']:.2f}% | "
            f"tile={scene.get('tile_id')} | "
            f"{scene['scene_id']}"
        )

    print("-" * 100)

    if args.dry_run:
        print("Dry run complete. No files were downloaded.")
        return 0

    print()

    successful = 0
    failed = 0

    for index, scene in enumerate(scenes, start=1):
        print("=" * 100)
        print(f"[{index}/{len(scenes)}] Downloading scene")
        print(f"Date:   {scene['acquisition_date']}")
        print(f"Tile:   {scene.get('tile_id')}")
        print(f"Cloud:  {scene['cloud_cover']}%")
        print(f"Scene:  {scene['scene_id']}")
        print("=" * 100)

        try:
            result = download_and_validate_scene(scene, aoi)

            validation = result["validation"]

            print("Download: SUCCESS")
            print(f"Validation: {validation['status']}")
            print(
                f"Local: {result['provenance']['local_product_path']}"
            )
            print(
                f"Provenance: {result['provenance_path']}"
            )

            successful += 1

        except Exception as error:
            print(
                f"Download: FAILED "
                f"({type(error).__name__})"
            )
            print(str(error))
            failed += 1

    print()
    print("=" * 100)
    print("TEMPORAL BENCHMARK DOWNLOAD SUMMARY")
    print("=" * 100)
    print(f"Selected:   {len(scenes)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print("=" * 100)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())