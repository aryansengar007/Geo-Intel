from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings
from backend.app.services.aoi_service import load_aoi
from backend.app.services.scene_processing import inspect_scene, preprocess_scene, write_quality_report

DEFAULT_SCENE = "S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one local Sentinel-2 SAFE product and optionally preprocess it for analysis.")
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="Scene ID (must already exist locally; no data is downloaded)")
    parser.add_argument("--scene-dir", type=Path, help="Explicit SAFE directory; overrides the standard data path")
    parser.add_argument("--aoi", type=Path, default=settings.aoi_path, help="Existing AOI GeoJSON")
    parser.add_argument("--preprocess", action="store_true", help="Create AOI-clipped aligned reflectance bands, indices, and previews")
    parser.add_argument("--no-previews", action="store_true", help="Skip development PNG previews")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matches = list((settings.data_root / "raw" / "sentinel2").glob(f"*/{args.scene}.SAFE"))
    if args.scene_dir:
        scene_dir = args.scene_dir
    elif len(matches) == 1:
        scene_dir = matches[0]
    else:
        print(f"Expected one local SAFE directory for {args.scene}; found {len(matches)}.")
        return 2
    if not scene_dir.is_dir():
        print(f"Local SAFE directory not found: {scene_dir}")
        return 2
    provenance_path = settings.metadata_root / "downloads" / f"{args.scene}.provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.is_file() else None
    aoi = load_aoi(args.aoi)
    try:
        report = inspect_scene(scene_dir, aoi, provenance)
        if args.preprocess:
            report = preprocess_scene(
                scene_dir,
                aoi,
                report,
                output_root=settings.data_root / "processed",
                make_previews=not args.no_previews,
            )
        else:
            existing_report_path = settings.metadata_root / "quality" / f"{args.scene}.quality.json"
            if existing_report_path.is_file():
                existing_report = json.loads(existing_report_path.read_text(encoding="utf-8"))
                prior_processing = existing_report.get("preprocessing")
                if prior_processing:
                    output_paths = [*prior_processing.get("aligned_band_paths", {}).values(), *prior_processing.get("index_paths", {}).values()]
                    if all(Path(path).is_file() for path in output_paths):
                        report["preprocessing"] = prior_processing
                        report["preprocessing_status"] = existing_report.get("preprocessing_status", "complete")
    except Exception as error:
        print(f"Quality gate: FAIL ({type(error).__name__})")
        print(str(error))
        failure_report = {
            "status": "FAIL",
            "scene_id": args.scene,
            "product_path": str(scene_dir),
            "preprocessing_status": "not_run",
            "errors": [f"{type(error).__name__}: {error}"],
            "warnings": [],
        }
        json_path, markdown_path = write_quality_report(failure_report, settings.metadata_root / "quality")
        print(f"Quality JSON: {json_path}")
        print(f"Quality report: {markdown_path}")
        return 1

    report_dir = settings.metadata_root / "quality"
    json_path, markdown_path = write_quality_report(report, report_dir)
    print(f"Scene: {report['scene_id']}")
    print(f"Acquisition: {report['acquisition_datetime']}")
    print(f"Tile: {report['tile']}")
    print(f"Scene cloud metadata: {report['scene_cloud_cover_percent']}%")
    print(f"Quality gate: {report['status']}")
    print(f"Bands inspected: {', '.join(report['available_bands'])}")
    print(f"SCL available: {report['scl']['available']}")
    if args.preprocess:
        print(f"Preprocessing: {report['preprocessing_status']}")
        print(f"Target grid: {report['preprocessing']['width']} x {report['preprocessing']['height']} at 10 m, {report['preprocessing']['target_crs']}")
        print(f"Indices: {', '.join(report['preprocessing']['index_paths'])}")
        print(f"All-band valid AOI pixels: {report['preprocessing']['valid_pixels_all_required_bands_percent']:.4f}%")
    print(f"Quality JSON: {json_path}")
    print(f"Quality report: {markdown_path}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    return 0 if report["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
