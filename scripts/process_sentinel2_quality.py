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
from backend.app.services.scene_processing import (
    apply_quality_mask_to_aligned,
    inspect_scene,
    preprocess_scene,
    write_baseline_manifest,
    write_quality_report,
)

DEFAULT_SCENE = "S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519"


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply same-scene SCL/cloud/snow quality masks to existing aligned data; does not redownload or rebuild base bands.")
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    args = parser.parse_args()

    safe_matches = list(
        (Path("D:/GIData") / "sentinel2").glob(f"*/{args.scene}*")
    )

    if len(safe_matches) != 1:
        print(
            f"Expected one existing scene directory for "
            f"{args.scene}; found {len(safe_matches)}."
        )
        return 2

    scene_dir = safe_matches[0]

    if not (scene_dir / "GRANULE").is_dir():
        nested_safe = scene_dir / f"{args.scene}.SAFE"

        if nested_safe.is_dir():
            scene_dir = nested_safe
        
    provenance_path = (
        settings.metadata_root
        / "downloads"
        / f"{args.scene}.provenance.json"
    )

    if not provenance_path.is_file():
        print("Existing scene provenance is missing.")
        return 2

    try:
        provenance = json.loads(
            provenance_path.read_text(encoding="utf-8")
        )

        aoi = load_aoi(settings.aoi_path)

        report = inspect_scene(
            scene_dir,
            aoi,
            provenance,
        )

        report = preprocess_scene(
            scene_dir,
            aoi,
            report,
            output_root=settings.data_root / "processed",
            make_previews=True,
        )

        report = apply_quality_mask_to_aligned(
            scene_dir,
            aoi,
            report,
            output_root=settings.data_root / "processed",
        )

        json_path, markdown_path = write_quality_report(
            report,
            settings.metadata_root / "quality",
        )

        baseline_path = write_baseline_manifest(
            report,
            provenance,
            aoi_path=settings.aoi_path,
            output_dir=settings.metadata_root / "baselines",
        )

    except Exception as error:
        print(
            f"Quality masking: FAILED "
            f"({type(error).__name__})"
        )
        print(str(error))
        return 1

    stats = report["pixel_cloud_statistics"]
    print(f"Scene: {report['scene_id']}")
    print(f"Quality status: {report['status']}")
    print(f"SCL: {report['scl']['path']}")
    print(f"SCL classes observed in AOI: {', '.join(map(str, report['scl']['raster']['class_values_observed_in_aoi']))}")
    print(f"AOI pixels: {stats['total_aoi_pixels']}")
    print(f"Cloud: {stats['cloud_pixels']} ({stats['cloud_percent']:.4f}%)")
    print(f"Cloud shadow: {stats['cloud_shadow_pixels']} ({stats['cloud_shadow_percent']:.4f}%)")
    print(f"Cirrus: {stats['cirrus_pixels']} ({stats['cirrus_percent']:.4f}%)")
    print(f"Snow/ice: {stats['snow_ice_pixels']} ({stats['snow_ice_percent']:.4f}%)")
    print(f"Other masked: {stats['other_masked_pixels']} ({stats['other_masked_percent']:.4f}%)")
    print(f"Usable: {stats['usable_analysis_pixels']} ({stats['usable_analysis_percent']:.4f}%)")
    print("Cloud-screened indices:")
    for name, path in report["preprocessing"]["quality_screening"]["cloud_screened_index_paths"].items():
        print(f"  {name}: {path}")
    print(f"Quality JSON: {json_path}")
    print(f"Quality report: {markdown_path}")
    print(f"Baseline manifest: {baseline_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
