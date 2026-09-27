from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.services.quality_asset_service import QUALITY_ASSET_KEYS, fetch_quality_assets

DEFAULT_SCENE = "S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch only STAC-listed SCL/cloud/snow QA rasters for an already-downloaded Sentinel-2 L2A scene.")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="Existing scene ID; this command never discovers or downloads another scene")
    parser.add_argument("--asset", choices=QUALITY_ASSET_KEYS, action="append", help="Optional QA asset key; repeat to select assets (default: SCL_20m, CLD_20m, SNW_20m)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = fetch_quality_assets(args.scene, asset_keys=tuple(args.asset or QUALITY_ASSET_KEYS))
    except Exception as error:
        print(f"Quality asset fetch: FAILED ({type(error).__name__})")
        print(str(error))
        return 1

    print(f"Scene: {result['scene_id']}")
    print(f"Product ID: {result['product_id']}")
    print(f"Product: {result['product_identifier']}")
    for asset in result["downloaded"]:
        print(f"Downloaded: {asset['key']} ({asset['file_size_bytes']} bytes)")
        print(f"  Source: {asset['href']}")
        print(f"  Local: {asset['local_path_from_repo']}")
    for key in result["skipped_existing_verified"]:
        print(f"Already present and checksum-verified: {key}")
    print(f"Bytes downloaded this run: {result['total_bytes_downloaded_this_run']}")
    print(f"Provenance: {Path(result['provenance_path'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
