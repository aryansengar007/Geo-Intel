from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import Window
from shapely.geometry import box, shape

from backend.app.services.sentinel_service import SELECTED_ASSET_KEYS


def _xml_values(path: Path) -> dict[str, list[str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Required Sentinel-2 metadata is missing or empty: {path.name}")
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"Sentinel-2 metadata XML is invalid: {path.name}") from error
    values: dict[str, list[str]] = {}
    values["__ROOT__"] = [root.tag.rsplit("}", 1)[-1].upper()]
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1].upper()
        if element.text and element.text.strip():
            values.setdefault(name, []).append(element.text.strip())
    return values


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_sample_stats(dataset: rasterio.io.DatasetReader) -> dict[str, float | None]:
    width = min(512, dataset.width)
    height = min(512, dataset.height)
    window = Window((dataset.width - width) // 2, (dataset.height - height) // 2, width, height)
    sample = dataset.read(1, window=window, masked=True)
    values = np.asarray(sample.compressed())
    if values.size == 0:
        return {"sample_min": None, "sample_max": None}
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"sample_min": None, "sample_max": None}
    return {"sample_min": float(values.min()), "sample_max": float(values.max())}


def validate_scene_assets(
    scene: dict[str, Any],
    downloaded_assets: dict[str, Path],
    aoi_geojson: dict[str, Any],
) -> dict[str, Any]:
    required = set(SELECTED_ASSET_KEYS) | {"safe_manifest", "product_metadata", "granule_metadata"}
    missing = sorted(required - downloaded_assets.keys())
    if missing:
        raise ValueError(f"Downloaded scene is missing required assets: {', '.join(missing)}")
    for key in required:
        path = downloaded_assets[key]
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"Downloaded asset is missing or empty: {key}")

    manifest_values = _xml_values(downloaded_assets["safe_manifest"])
    product_values = _xml_values(downloaded_assets["product_metadata"])
    granule_values = _xml_values(downloaded_assets["granule_metadata"])
    if "XFDU" not in manifest_values["__ROOT__"]:
        raise ValueError("Downloaded manifest is not a Sentinel SAFE manifest.")
    if not product_values.get("PRODUCT_URI"):
        raise ValueError("Product metadata is not a Sentinel-2 L2A product metadata document.")

    product_uris = product_values.get("PRODUCT_URI", [])
    product_name = scene.get("product_name") or scene.get("scene_id") or ""
    if not product_uris or product_name.casefold() not in product_uris[0].casefold():
        raise ValueError("Sentinel-2 product metadata does not match the selected STAC product.")
    metadata_times = product_values.get("PRODUCT_START_TIME") or product_values.get("DATATAKE_SENSING_START")
    if not metadata_times:
        raise ValueError("Sentinel-2 product metadata has no acquisition start time.")
    stac_time = _parse_datetime(scene["acquisition_date"])
    metadata_time = _parse_datetime(metadata_times[0])
    if abs((metadata_time - stac_time).total_seconds()) > 2:
        raise ValueError("Sentinel-2 product acquisition time does not match the STAC item.")
    tile_ids = granule_values.get("TILE_ID", [])
    if scene.get("tile") and not any(scene["tile"] in tile_id for tile_id in tile_ids):
        raise ValueError("Granule metadata tile does not match the selected STAC item.")

    aoi = shape(aoi_geojson["features"][0]["geometry"])
    raster_reports: dict[str, dict[str, Any]] = {}
    for asset_key, (band, expected_resolution) in SELECTED_ASSET_KEYS.items():
        path = downloaded_assets[asset_key]
        try:
            with rasterio.open(path) as raster:
                if raster.width <= 0 or raster.height <= 0:
                    raise ValueError(f"Band {band} has invalid raster dimensions.")
                if raster.crs is None:
                    raise ValueError(f"Band {band} has no CRS.")
                resolution = [abs(float(raster.transform.a)), abs(float(raster.transform.e))]
                if any(not math.isclose(value, expected_resolution, rel_tol=0.01, abs_tol=0.05) for value in resolution):
                    raise ValueError(f"Band {band} resolution does not match the expected {expected_resolution} m.")
                west, south, east, north = transform_bounds(raster.crs, "EPSG:4326", *raster.bounds, densify_pts=21)
                if not box(west, south, east, north).intersects(aoi):
                    raise ValueError(f"Band {band} raster does not overlap the Gurugram AOI.")
                raster_reports[band] = {
                    "path": str(path),
                    "width": raster.width,
                    "height": raster.height,
                    "crs": raster.crs.to_string(),
                    "transform": list(raster.transform)[:6],
                    "resolution_m": resolution,
                    "dtype": raster.dtypes[0],
                    "nodata": raster.nodata,
                    "bounds": [float(value) for value in raster.bounds],
                    "aoi_overlap": True,
                    **_safe_sample_stats(raster),
                }
        except rasterio.errors.RasterioError as error:
            raise ValueError(f"Band {band} is not a readable raster: {path.name}") from error

    return {
        "status": "PASS",
        "collection": scene["collection"],
        "processing_level": scene["processing_level"],
        "product_identifier": scene["product_identifier"],
        "acquisition_datetime": stac_time.isoformat(),
        "metadata_acquisition_datetime": metadata_time.isoformat(),
        "aoi_overlap": True,
        "safe_manifest_readable": True,
        "product_metadata_readable": True,
        "granule_metadata_readable": True,
        "bands": raster_reports,
    }
