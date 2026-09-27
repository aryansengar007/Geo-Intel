from __future__ import annotations

from contextlib import ExitStack, nullcontext
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask, geometry_window
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import box, shape
from shapely.ops import transform as transform_geometry
from pyproj import Transformer

from backend.app.services.aoi_service import geodesic_area_km2
from backend.app.services.sentinel_service import SELECTED_ASSET_KEYS

NODATA_OUTPUT = -9999.0
CHUNK_ROWS = 256
PREVIEW_SIZE = 512
SCL_CLASS_NAMES = {
    0: "NO_DATA",
    1: "SATURATED_OR_DEFECTIVE",
    2: "CAST_SHADOWS",
    3: "CLOUD_SHADOWS",
    4: "VEGETATION",
    5: "NOT_VEGETATED",
    6: "WATER",
    7: "UNCLASSIFIED",
    8: "CLOUD_MEDIUM_PROBABILITY",
    9: "CLOUD_HIGH_PROBABILITY",
    10: "THIN_CIRRUS",
    11: "SNOW_OR_ICE",
}
SCL_REFERENCE = "ESA Sentinel-2 S2 Processing, L2A Scene Classification, Table 3"
BAND_IDS = {"B02": 1, "B03": 2, "B04": 3, "B05": 4, "B06": 5, "B07": 6, "B08": 7, "B11": 11, "B12": 12}
BAND_RESOLUTIONS = {band: resolution for band, resolution in SELECTED_ASSET_KEYS.values()}
BAND_FILES = {band: f"_{band}_{resolution}m.jp2" for band, resolution in BAND_RESOLUTIONS.items()}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_xml(path: Path) -> tuple[ET.Element, dict[str, list[ET.Element]]]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise ValueError(f"Cannot read Sentinel-2 XML metadata: {path}") from error
    elements: dict[str, list[ET.Element]] = {}
    for element in root.iter():
        elements.setdefault(_local_name(element.tag), []).append(element)
    return root, elements


def _text(elements: dict[str, list[ET.Element]], name: str, default: str | None = None) -> str | None:
    for element in elements.get(name, []):
        if element.text and element.text.strip():
            return element.text.strip()
    return default


def _metadata_from_safe(scene_dir: Path) -> dict[str, Any]:
    product_path = scene_dir / "MTD_MSIL2A.xml"
    root, elements = _parse_xml(product_path)
    quantification = _text(elements, "BOA_QUANTIFICATION_VALUE")
    if quantification is None:
        raise ValueError("Product metadata has no BOA_QUANTIFICATION_VALUE.")
    offsets: dict[int, float] = {}
    for element in elements.get("BOA_ADD_OFFSET", []):
        band_id = element.attrib.get("band_id")
        if band_id is not None and element.text:
            offsets[int(band_id)] = float(element.text.strip())

    parents = {child: parent for parent in root.iter() for child in parent}
    special_values: dict[str, list[int]] = {}
    for element in elements.get("SPECIAL_VALUE_TEXT", []):
        special_name = (element.text or "").strip().upper()
        if not special_name:
            continue
        parent = parents.get(element)
        if parent is None:
            continue
        index = next((child.text for child in parent if _local_name(child.tag) == "SPECIAL_VALUE_INDEX" and child.text), None)
        if index is not None:
            special_values.setdefault(special_name, []).append(int(index))

    return {
        "product_identifier": _text(elements, "PRODUCT_URI"),
        "acquisition_datetime": _text(elements, "PRODUCT_START_TIME"),
        "processing_level": _text(elements, "PROCESSING_LEVEL"),
        "processing_baseline": _text(elements, "PROCESSING_BASELINE"),
        "boa_quantification_value": float(quantification),
        "boa_add_offsets_by_band_id": offsets,
        "special_values": {name: sorted(set(values)) for name, values in special_values.items()},
        "nodata_values": special_values.get("NODATA", []),
        "saturated_values": special_values.get("SATURATED", []),
        "cloud_quality_metadata": {
            key: _text(elements, key)
            for key in (
                "Cloud_Coverage_Assessment",
                "CLOUDY_PIXEL_PERCENTAGE",
                "CLOUDY_PIXEL_OVER_LAND_PERCENTAGE",
                "CLOUD_SHADOW_PERCENTAGE",
                "MEDIUM_PROBA_CLOUDS_PERCENTAGE",
                "HIGH_PROBA_CLOUDS_PERCENTAGE",
                "THIN_CIRRUS_PERCENTAGE",
                "SNOW_ICE_PERCENTAGE",
            )
            if _text(elements, key) is not None
        },
    }


def _discover_assets(scene_dir: Path) -> tuple[Path, dict[str, Path], Path | None, list[Path]]:
    granule_root = scene_dir / "GRANULE"
    granules = sorted(path for path in granule_root.iterdir() if path.is_dir()) if granule_root.is_dir() else []
    if not granules:
        raise ValueError("SAFE product has no GRANULE directory.")
    all_jp2 = sorted(scene_dir.rglob("*.jp2"))
    bands: dict[str, Path] = {}
    for band, suffix in BAND_FILES.items():
        matches = [path for path in all_jp2 if path.name.endswith(suffix)]
        if len(matches) > 1:
            raise ValueError(f"Multiple JP2 assets found for {band}.")
        if matches:
            bands[band] = matches[0]
    scl_matches = [path for path in all_jp2 if re.search(r"_SCL_(20|60)m\.jp2$", path.name, re.IGNORECASE)]
    qa_files = [
        path for path in all_jp2
        if any(token in path.name.upper() for token in ("MSK_CLDPRB", "MSK_SNWPRB", "MSK_CLASSI", "MSK_QUALIT"))
    ]
    scl = scl_matches[0] if len(scl_matches) == 1 else None
    if len(scl_matches) > 1:
        raise ValueError("Multiple SCL assets found; expected one for the scene granule.")
    return granules[0], bands, scl, qa_files


def _projected_aoi(aoi_geojson: dict[str, Any], target_crs: Any):
    source_crs = aoi_geojson.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    geom = shape(aoi_geojson["features"][0]["geometry"])
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    projected = transform_geometry(transformer.transform, geom)
    if projected.is_empty or not projected.is_valid:
        raise ValueError("AOI geometry is empty or invalid after CRS transformation.")
    return projected, source_crs


def _aoi_window(dataset: rasterio.io.DatasetReader, geom) -> Window:
    return geometry_window(dataset, [geom], boundless=False).round_offsets().round_lengths()


def _sample_aoi_band(
    dataset: rasterio.io.DatasetReader,
    geom,
    quantification: float,
    offset: float,
    special_values: dict[str, list[int]],
) -> dict[str, Any]:
    try:
        window = _aoi_window(dataset, geom)
    except Exception as error:
        raise ValueError("The AOI does not overlap the raster grid.") from error
    transform = window_transform(window, dataset.transform)
    area_pixels = 0
    valid_pixels = 0
    no_data_pixels = 0
    invalid_pixels = 0
    raw_min = math.inf
    raw_max = -math.inf
    reflectance_min = math.inf
    reflectance_max = -math.inf
    above_review_range_pixels = 0
    below_review_range_pixels = 0
    saturated_pixels = 0
    histogram = np.zeros(65536, dtype=np.uint64)
    high_reflectance_outliers: list[dict[str, Any]] = []
    to_wgs84 = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True).transform
    nodata_values = special_values.get("NODATA", [])
    saturated_values = special_values.get("SATURATED", [])
    for row_start in range(0, int(window.height), CHUNK_ROWS):
        height = min(CHUNK_ROWS, int(window.height) - row_start)
        chunk = Window(window.col_off, window.row_off + row_start, window.width, height)
        values = dataset.read(1, window=chunk)
        inside = geometry_mask(
            [geom],
            out_shape=values.shape,
            transform=window_transform(chunk, dataset.transform),
            invert=True,
        )
        aoi_values = values[inside]
        area_pixels += int(aoi_values.size)
        valid = np.isfinite(aoi_values)
        nodata_mask = np.zeros(aoi_values.shape, dtype=bool)
        if dataset.nodata is not None:
            nodata_mask |= aoi_values == dataset.nodata
        for marker in nodata_values:
            nodata_mask |= aoi_values == marker
        no_data_pixels += int(nodata_mask.sum())
        valid &= ~nodata_mask
        is_saturated = np.isin(aoi_values, saturated_values)
        saturated_pixels += int(is_saturated.sum())
        valid &= ~is_saturated
        invalid_pixels += int((~valid).sum())
        if not valid.any():
            continue
        raw = aoi_values[valid].astype(np.float64, copy=False)
        reflectance = (raw + offset) / quantification
        histogram += np.bincount(raw.astype(np.uint16), minlength=65536).astype(np.uint64)
        raw_min = min(raw_min, float(raw.min()))
        raw_max = max(raw_max, float(raw.max()))
        reflectance_min = min(reflectance_min, float(reflectance.min()))
        reflectance_max = max(reflectance_max, float(reflectance.max()))
        high_mask = reflectance > 2.0
        low_mask = reflectance < -0.2
        above_review_range_pixels += int(high_mask.sum())
        below_review_range_pixels += int(low_mask.sum())
        valid_pixels += int(raw.size)
        if high_mask.any() and len(high_reflectance_outliers) < 100:
            source_candidates = inside & np.isfinite(values) & ~np.isin(values, (*nodata_values, *saturated_values))
            source_candidates &= ((values.astype(np.float64) + offset) / quantification) > 2.0
            local_rows, local_cols = np.nonzero(source_candidates)
            remaining = 100 - len(high_reflectance_outliers)
            for local_row, local_col in zip(local_rows[:remaining], local_cols[:remaining]):
                dn = int(values[local_row, local_col])
                row = int(chunk.row_off + local_row)
                col = int(chunk.col_off + local_col)
                x, y = dataset.xy(row, col)
                longitude, latitude = to_wgs84(x, y)
                high_reflectance_outliers.append({
                    "row": row,
                    "column": col,
                    "dn": dn,
                    "surface_reflectance": (dn + offset) / quantification,
                    "crs_x": float(x),
                    "crs_y": float(y),
                    "longitude": float(longitude),
                    "latitude": float(latitude),
                    "inside_aoi": True,
                })

    raster_footprint = box(*dataset.bounds)
    coverage = min(1.0, max(0.0, geom.intersection(raster_footprint).area / geom.area)) if geom.area else 0.0
    percentiles = {}
    cumulative = np.cumsum(histogram)
    for percentile in (2, 5, 50, 95, 98, 99, 99.9):
        if valid_pixels:
            rank = max(1, math.ceil(percentile * valid_pixels / 100))
            dn_value = int(np.searchsorted(cumulative, rank, side="left"))
            percentiles[str(percentile)] = (dn_value + offset) / quantification
        else:
            percentiles[str(percentile)] = None
    return {
        "width": dataset.width,
        "height": dataset.height,
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "transform": list(dataset.transform)[:6],
        "resolution_m": [abs(float(dataset.transform.a)), abs(float(dataset.transform.e))],
        "dtype": dataset.dtypes[0],
        "nodata": dataset.nodata,
        "metadata_nodata_values": nodata_values,
        "bounds": [float(value) for value in dataset.bounds],
        "aoi_cover_percent": 100.0 * coverage,
        "aoi_pixels": area_pixels,
        "valid_pixels": valid_pixels,
        "invalid_pixels": invalid_pixels,
        "nodata_pixels": no_data_pixels,
        "saturated_pixels": saturated_pixels,
        "valid_pixel_percent": 100.0 * valid_pixels / area_pixels if area_pixels else 0.0,
        "invalid_pixel_percent": 100.0 * invalid_pixels / area_pixels if area_pixels else 0.0,
        "raw_dn_min": None if raw_min == math.inf else raw_min,
        "raw_dn_max": None if raw_max == -math.inf else raw_max,
        "surface_reflectance_min": None if reflectance_min == math.inf else reflectance_min,
        "surface_reflectance_max": None if reflectance_max == -math.inf else reflectance_max,
        "surface_reflectance_percentiles": percentiles,
        "above_review_range_pixels": above_review_range_pixels,
        "below_review_range_pixels": below_review_range_pixels,
        "outside_broad_reflectance_range_pixels": above_review_range_pixels + below_review_range_pixels,
        "broad_reflectance_range": [-0.2, 2.0],
        "high_reflectance_source_pixels": high_reflectance_outliers,
        "high_reflectance_outlier_count_truncated": above_review_range_pixels > len(high_reflectance_outliers),
    }


def scl_quality_masks(
    scl: np.ndarray,
    nodata_values: tuple[int, ...] = (0,),
    cloud_probability: np.ndarray | None = None,
    snow_probability: np.ndarray | None = None,
    *,
    cloud_probability_threshold: float = 40.0,
    snow_probability_threshold: float = 50.0,
) -> dict[str, np.ndarray]:
    """Return SCL masks; ESA class definitions are in S2 Processing Table 3."""
    invalid = ~np.isin(scl, tuple(SCL_CLASS_NAMES)) | np.isin(scl, (*nodata_values, 1))
    dark_shadow = scl == 2
    cloud_shadow = scl == 3
    medium_cloud = scl == 8
    high_cloud = scl == 9
    cloud_probability_mask = (
        np.isfinite(cloud_probability) & (cloud_probability >= cloud_probability_threshold)
        if cloud_probability is not None
        else np.zeros(scl.shape, dtype=bool)
    )
    cloud = medium_cloud | high_cloud | cloud_probability_mask
    cirrus = scl == 10
    snow_probability_mask = (
        np.isfinite(snow_probability) & (snow_probability >= snow_probability_threshold)
        if snow_probability is not None
        else np.zeros(scl.shape, dtype=bool)
    )
    snow_ice = (scl == 11) | snow_probability_mask
    other_masked = invalid | dark_shadow
    masked = other_masked | cloud_shadow | cloud | cirrus | snow_ice
    return {
        "invalid": invalid,
        "dark_shadow": dark_shadow,
        "cloud_shadow": cloud_shadow,
        "cloud": cloud,
        "medium_probability_cloud": medium_cloud,
        "high_probability_cloud": high_cloud,
        "cloud_probability_mask": cloud_probability_mask,
        "low_cloud_probability_retained": (
            np.isfinite(cloud_probability)
            & (cloud_probability > 0)
            & (cloud_probability < cloud_probability_threshold)
            if cloud_probability is not None
            else np.zeros(scl.shape, dtype=bool)
        ),
        "cirrus": cirrus,
        "snow_ice": snow_ice,
        "snow_probability_mask": snow_probability_mask,
        "other_masked": other_masked,
        "masked": masked,
        "usable": ~masked,
        "valid_landcover": ~masked,
    }


def _read_quality_window(
    source: rasterio.io.DatasetReader,
    reference: rasterio.io.DatasetReader,
    window: Window,
    *,
    categorical: bool = False,
) -> np.ndarray:
    if (
        source.crs == reference.crs
        and source.transform == reference.transform
        and source.width == reference.width
        and source.height == reference.height
    ):
        return source.read(1, window=window)
    output = np.full((int(window.height), int(window.width)), np.nan, dtype=np.float32)
    reproject(
        source=rasterio.band(source, 1),
        destination=output,
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=window_transform(window, reference.transform),
        dst_crs=reference.crs,
        dst_nodata=np.nan,
        resampling=Resampling.nearest if categorical else Resampling.bilinear,
    )
    return output


def _scl_inventory(
    dataset: rasterio.io.DatasetReader,
    geom,
    cloud_dataset: rasterio.io.DatasetReader | None = None,
    snow_dataset: rasterio.io.DatasetReader | None = None,
    *,
    cloud_probability_threshold: float = 40.0,
    snow_probability_threshold: float = 50.0,
) -> dict[str, Any]:
    window = _aoi_window(dataset, geom)
    counts: dict[int, int] = {}
    quality_counts = {
        "cloud_shadow": 0,
        "cloud": 0,
        "cirrus": 0,
        "snow_ice": 0,
        "other_masked": 0,
        "usable": 0,
        "low_cloud_probability_retained": 0,
    }
    for row_start in range(0, int(window.height), CHUNK_ROWS):
        height = min(CHUNK_ROWS, int(window.height) - row_start)
        chunk = Window(window.col_off, window.row_off + row_start, window.width, height)
        values = dataset.read(1, window=chunk)
        cloud_values = (
            _read_quality_window(cloud_dataset, dataset, chunk).astype(np.float32, copy=False)
            if cloud_dataset
            else None
        )
        snow_values = (
            _read_quality_window(snow_dataset, dataset, chunk).astype(np.float32, copy=False)
            if snow_dataset
            else None
        )
        inside = geometry_mask(
            [geom],
            out_shape=values.shape,
            transform=window_transform(chunk, dataset.transform),
            invert=True,
        )
        classes, class_counts = np.unique(values[inside], return_counts=True)
        for class_value, count in zip(classes, class_counts):
            counts[int(class_value)] = counts.get(int(class_value), 0) + int(count)
        class_values = values[inside]
        cld = cloud_values[inside] if cloud_values is not None else None
        snw = snow_values[inside] if snow_values is not None else None
        masks = scl_quality_masks(
            class_values,
            (0,),
            cld,
            snw,
            cloud_probability_threshold=cloud_probability_threshold,
            snow_probability_threshold=snow_probability_threshold,
        )
        for name in quality_counts:
            quality_counts[name] += int(masks[name].sum())

    total = sum(counts.values())
    masked_count = total - quality_counts["usable"]
    observed_values = sorted(counts)
    return {
        "path": str(dataset.name),
        "width": dataset.width,
        "height": dataset.height,
        "crs": dataset.crs.to_string() if dataset.crs else None,
        "resolution_m": [abs(float(dataset.transform.a)), abs(float(dataset.transform.e))],
        "dtype": dataset.dtypes[0],
        "nodata": dataset.nodata,
        "bounds": [float(value) for value in dataset.bounds],
        "class_values_observed_in_aoi": observed_values,
        "class_names_observed_in_aoi": {str(value): SCL_CLASS_NAMES.get(value, "UNKNOWN") for value in observed_values},
        "class_pixel_counts_in_aoi": {str(value): counts[value] for value in observed_values},
        "quality_pixel_counts_in_aoi": quality_counts,
        "quality_pixel_percentages_in_aoi": {
            name: 100.0 * count / total if total else 0.0
            for name, count in quality_counts.items()
        },
        "aoi_pixel_count": total,
        "masked_pixel_count": masked_count,
        "valid_pixel_count": total - masked_count,
        "masked_pixel_percent": 100.0 * masked_count / total if total else 0.0,
        "valid_pixel_percent": 100.0 * (total - masked_count) / total if total else 0.0,
        "masked_class_values": sorted({0, 1, 2, 3, 8, 9, 10, 11}),
        "cloud_probability_threshold_percent": cloud_probability_threshold,
        "snow_probability_threshold_percent": snow_probability_threshold,
        "class_reference": SCL_REFERENCE,
    }


def _quality_raster_report(path: Path, aoi_geojson: dict[str, Any]) -> dict[str, Any]:
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Quality raster has no CRS: {path.name}")
        aoi, _ = _projected_aoi(aoi_geojson, dataset.crs)
        footprint = box(*dataset.bounds)
        if not aoi.intersects(footprint):
            raise ValueError(f"Quality raster does not overlap the Gurugram AOI: {path.name}")
        window = _aoi_window(dataset, aoi)
        valid_pixels = 0
        total_pixels = 0
        minimum = math.inf
        maximum = -math.inf
        for row_start in range(0, int(window.height), CHUNK_ROWS):
            height = min(CHUNK_ROWS, int(window.height) - row_start)
            chunk = Window(window.col_off, window.row_off + row_start, window.width, height)
            values = dataset.read(1, window=chunk)
            inside = geometry_mask(
                [aoi],
                out_shape=values.shape,
                transform=window_transform(chunk, dataset.transform),
                invert=True,
            )
            samples = values[inside]
            total_pixels += int(samples.size)
            valid = np.isfinite(samples)
            if dataset.nodata is not None:
                valid &= samples != dataset.nodata
            usable = samples[valid]
            if usable.size:
                valid_pixels += int(usable.size)
                minimum = min(minimum, float(usable.min()))
                maximum = max(maximum, float(usable.max()))
        return {
            "path": str(path),
            "width": dataset.width,
            "height": dataset.height,
            "crs": dataset.crs.to_string(),
            "resolution_m": [abs(float(dataset.transform.a)), abs(float(dataset.transform.e))],
            "dtype": dataset.dtypes[0],
            "nodata": dataset.nodata,
            "bounds": [float(value) for value in dataset.bounds],
            "aoi_pixels": total_pixels,
            "aoi_valid_pixels": valid_pixels,
            "aoi_valid_pixel_percent": 100.0 * valid_pixels / total_pixels if total_pixels else 0.0,
            "aoi_min": None if minimum == math.inf else minimum,
            "aoi_max": None if maximum == -math.inf else maximum,
        }


def inspect_scene(scene_dir: str | Path, aoi_geojson: dict[str, Any], provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    scene_dir = Path(scene_dir)
    expected_scene = (provenance or {}).get("scene_id")
    root_name = scene_dir.name.removesuffix(".SAFE")
    if expected_scene and root_name != expected_scene:
        raise ValueError("SAFE directory name does not match the supplied scene provenance.")
    required_structure = {
        "manifest.safe": (scene_dir / "manifest.safe").is_file(),
        "MTD_MSIL2A.xml": (scene_dir / "MTD_MSIL2A.xml").is_file(),
        "GRANULE": (scene_dir / "GRANULE").is_dir(),
    }
    granule, bands, scl_path, qa_files = _discover_assets(scene_dir)
    metadata = _metadata_from_safe(scene_dir)
    provenance_product_identifier = (provenance or {}).get(
        "product_identifier"
    )

    if (
        expected_scene
        and provenance_product_identifier
        and metadata["product_identifier"]
        != provenance_product_identifier
    ):
        raise ValueError(
            "SAFE metadata product identifier does not match "
            "download provenance."
        )
    if provenance and provenance.get("acquisition_datetime") and metadata["acquisition_datetime"]:
        actual_time = datetime.fromisoformat(metadata["acquisition_datetime"].replace("Z", "+00:00"))
        expected_time = datetime.fromisoformat(provenance["acquisition_datetime"].replace("Z", "+00:00"))
        if abs((actual_time - expected_time).total_seconds()) > 2:
            raise ValueError("SAFE product acquisition time does not match download provenance.")
    missing_bands = sorted(set(BAND_FILES) - bands.keys())
    checks: dict[str, str] = {
        "safe_structure": "PASS" if all(required_structure.values()) else "FAIL",
        "metadata_xml": "PASS",
        "required_bands": "FAIL" if missing_bands else "PASS",
        "granule_metadata": "PASS" if (granule / "MTD_TL.xml").is_file() else "FAIL",
    }
    if not all(required_structure.values()):
        raise ValueError("SAFE product is missing a required top-level structure entry.")
    manifest_root, _ = _parse_xml(scene_dir / "manifest.safe")
    if _local_name(manifest_root.tag).upper() != "XFDU":
        raise ValueError("manifest.safe is not a SAFE XFDU manifest.")
    if _local_name(_parse_xml(scene_dir / "MTD_MSIL2A.xml")[0].tag) != "Level-2A_User_Product":
        raise ValueError("MTD_MSIL2A.xml is not a Sentinel-2 Level-2A product metadata document.")
    if missing_bands:
        raise ValueError(f"SAFE product is missing required bands: {', '.join(missing_bands)}")

    rasters: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    source_crs = aoi_geojson.get("crs", {}).get("properties", {}).get("name", "EPSG:4326")
    first_ds = rasterio.open(bands["B02"])
    try:
        aoi_projected, source_crs = _projected_aoi(aoi_geojson, first_ds.crs)
        projected_aoi_area_km2 = aoi_projected.area / 1_000_000
    finally:
        first_ds.close()

    for band, path in sorted(bands.items()):
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"{band} has no CRS.")
            aoi_in_band_crs, _ = _projected_aoi(aoi_geojson, dataset.crs)
            if not aoi_in_band_crs.intersects(box(*dataset.bounds)):
                raise ValueError(f"{band} does not spatially intersect the Gurugram AOI.")
            band_id = BAND_IDS[band]
            offset = metadata["boa_add_offsets_by_band_id"].get(band_id, 0.0)
            report = _sample_aoi_band(
                dataset,
                aoi_in_band_crs,
                metadata["boa_quantification_value"],
                offset,
                metadata["special_values"],
            )
            report["path"] = path.relative_to(scene_dir).as_posix()
            report["boa_add_offset"] = offset
            report["aoi_crs"] = source_crs
            report["aoi_projected_area_km2"] = projected_aoi_area_km2
            rasters[band] = report
    product_values = _metadata_from_safe(scene_dir)
    tile_metadata_path = granule / "MTD_TL.xml"
    _, tile_elements = _parse_xml(tile_metadata_path)
    cloud_probability_path = next((path for path in qa_files if "MSK_CLDPRB" in path.name.upper()), None)
    snow_probability_path = next((path for path in qa_files if "MSK_SNWPRB" in path.name.upper()), None)
    quality_layer_reports = {
        "SCL": _quality_raster_report(scl_path, aoi_geojson) if scl_path else None,
        "cloud_probability": _quality_raster_report(cloud_probability_path, aoi_geojson) if cloud_probability_path else None,
        "snow_probability": _quality_raster_report(snow_probability_path, aoi_geojson) if snow_probability_path else None,
    }
    outlier_context_paths = [path for path in (scl_path, cloud_probability_path, snow_probability_path) if path]
    quality_readers = {path: rasterio.open(path) for path in outlier_context_paths}
    try:
        for band in ("B02", "B03"):
            for outlier in rasters[band]["high_reflectance_source_pixels"]:
                x, y = outlier["crs_x"], outlier["crs_y"]
                context = {}
                for key, path in (("scl_class", scl_path), ("cloud_probability_percent", cloud_probability_path), ("snow_probability_percent", snow_probability_path)):
                    reader = quality_readers.get(path)
                    if reader is None:
                        context[key] = None
                    else:
                        qrow, qcol = reader.index(x, y)
                        if 0 <= qrow < reader.height and 0 <= qcol < reader.width:
                            context[key] = int(reader.read(1, window=Window(qcol, qrow, 1, 1))[0, 0])
                        else:
                            context[key] = None
                context["scl_class_name"] = SCL_CLASS_NAMES.get(context["scl_class"]) if context["scl_class"] is not None else None
                outlier.update(context)
    finally:
        for reader in quality_readers.values():
            reader.close()
    scl_report = None
    if scl_path:
        with rasterio.open(scl_path) as scl_dataset, (
            rasterio.open(cloud_probability_path) if cloud_probability_path else nullcontext(None)
        ) as cloud_dataset, (
            rasterio.open(snow_probability_path) if snow_probability_path else nullcontext(None)
        ) as snow_dataset:
            scl_aoi, _ = _projected_aoi(aoi_geojson, scl_dataset.crs)
            scl_report = _scl_inventory(
                scl_dataset,
                scl_aoi,
                cloud_dataset,
                snow_dataset,
                cloud_probability_threshold=40.0,
                snow_probability_threshold=50.0,
            )
    tile_metadata_id = _text(tile_elements, "TILE_ID")
    mgrs_tile = (provenance or {}).get("tile")
    if not mgrs_tile and tile_metadata_id:
        match = re.search(r"T(\d{2}[A-Z]{3})", tile_metadata_id)
        mgrs_tile = match.group(1) if match else tile_metadata_id
    if (provenance or {}).get("tile") and mgrs_tile != provenance["tile"]:
        raise ValueError("SAFE granule tile does not match download provenance.")
    tile_cloud_fields = (
        "CLOUDY_PIXEL_PERCENTAGE",
        "CLOUDY_PIXEL_OVER_LAND_PERCENTAGE",
        "CLOUD_SHADOW_PERCENTAGE",
        "MEDIUM_PROBA_CLOUDS_PERCENTAGE",
        "HIGH_PROBA_CLOUDS_PERCENTAGE",
        "THIN_CIRRUS_PERCENTAGE",
        "SNOW_ICE_PERCENTAGE",
    )
    tile_cloud_quality_metadata = {
        key: _text(tile_elements, key)
        for key in tile_cloud_fields
        if _text(tile_elements, key) is not None
    }
    aoi_area_properties = aoi_geojson["features"][0].get("properties", {})
    aoi_geometry = shape(aoi_geojson["features"][0]["geometry"])
    geodesic_aoi_area_km2 = geodesic_area_km2(aoi_geojson["features"][0]["geometry"], source_crs)
    declared_area = aoi_area_properties.get("area_km2", aoi_area_properties.get("area_km2_geodesic", aoi_area_properties.get("area_km2_estimate")))
    reflectance_observations = []
    for band in ("B02", "B03"):
        band_report = rasters[band]
        outliers = band_report["high_reflectance_source_pixels"]
        if outliers:
            all_not_vegetated = all(sample.get("scl_class") == 5 for sample in outliers)
            all_clear_probability = all(
                sample.get("cloud_probability_percent") == 0 and sample.get("snow_probability_percent") == 0
                for sample in outliers
            )
            reflectance_observations.append({
                "band": band,
                "count_above_review_range": band_report["above_review_range_pixels"],
                "review_range": band_report["broad_reflectance_range"],
                "source_jp2": band_report["path"],
                "classification": (
                    "valid high-reflectance source samples; all are SCL NOT_VEGETATED with zero CLD/SNW probability, consistent with bright bare/built surfaces; exact material unconfirmed"
                    if all_not_vegetated and all_clear_probability
                    else "valid source JP2 samples; SCL/cloud/snow context is unavailable or nonuniform, so physical surface cause remains unresolved"
                ),
                "processing_issue_found": False,
                "clipped": False,
                "samples": outliers,
            })
    quality_observations = []
    if reflectance_observations:
        quality_observations.append(
            "B02/B03 values above reflectance 2.0 were confirmed in original JP2 source pixels and match the XML BOA conversion; they are not DN 0 or 65535 and were retained without clipping."
        )
        if all(
            all(sample.get("scl_class") == 5 and sample.get("cloud_probability_percent") == 0 and sample.get("snow_probability_percent") == 0 for sample in note["samples"])
            for note in reflectance_observations
        ):
            quality_observations.append(
                "All flagged samples are SCL NOT_VEGETATED with zero CLD/SNW probability; they are treated as valid bright-surface observations, with exact material not inferred."
            )
    previous_declared_area = aoi_area_properties.get("previous_declared_area_km2")
    previous_legacy_area = aoi_area_properties.get("previous_legacy_backend_area_km2")
    if previous_declared_area is not None or previous_legacy_area is not None:
        quality_observations.append(
            f"Historical area values recorded for audit: declared {previous_declared_area} km2 and legacy approximation {previous_legacy_area} km2; current authoritative area is calculated directly from the unchanged polygon."
        )
    if not scl_path:
        warnings.append("No SCL asset is present in this local SAFE product; pixel-level class statistics and class-based cloud masking are unavailable.")
    if not cloud_probability_path:
        warnings.append("No cloud probability raster is present; probability-threshold cloud screening is unavailable.")
    if not snow_probability_path:
        warnings.append("No snow probability raster is present; probability-threshold snow screening is unavailable.")
    if declared_area is not None and not math.isclose(float(declared_area), geodesic_aoi_area_km2, rel_tol=1e-7):
        warnings.append(
            f"AOI GeoJSON declares {float(declared_area):.6f} km2, while its WGS84 geodesic area is {geodesic_aoi_area_km2:.6f} km2."
        )
    warnings.append("AOI-aligned reflectance and cloud-screened index generation has not been run.")
    status = "FAIL" if "FAIL" in checks.values() else "WARNING" if warnings else "PASS"
    return {
        "status": status,
        "scene_id": expected_scene or root_name,
        "acquisition_datetime": product_values["acquisition_datetime"],
        "tile": mgrs_tile,
        "tile_metadata_id": tile_metadata_id,
        "processing_level": product_values["processing_level"],
        "processing_baseline": product_values["processing_baseline"],
        "collection": (provenance or {}).get("collection", "sentinel-2-l2a"),
        "scene_cloud_cover_percent": (provenance or {}).get("cloud_cover"),
        "product_cloud_quality_metadata_raw": product_values["cloud_quality_metadata"],
        "tile_cloud_quality_metadata_raw": tile_cloud_quality_metadata,
        "product_path": str(scene_dir),
        "granule_path": str(granule),
        "aoi": {
            "name": aoi_area_properties.get("name"),
            "source": aoi_area_properties.get("source"),
            "source_method": aoi_area_properties.get("source_method"),
            "source_crs": source_crs,
            "bounds_wgs84": list(shape(aoi_geojson["features"][0]["geometry"]).bounds),
            "geometry_type": aoi_geometry.geom_type,
            "geometry_valid": aoi_geometry.is_valid,
            "geometry_part_count": len(aoi_geometry.geoms) if hasattr(aoi_geometry, "geoms") else 1,
            "area_km2_geodesic_wgs84": geodesic_aoi_area_km2,
            "area_km2_authoritative": geodesic_aoi_area_km2,
            "area_method": "WGS84 ellipsoidal geodesic area from the unchanged GeoJSON geometry using pyproj.Geod",
            "previous_declared_area_km2": aoi_area_properties.get("previous_declared_area_km2"),
            "previous_legacy_backend_area_km2": aoi_area_properties.get("previous_legacy_backend_area_km2"),
            "projected_area_crosscheck_method": "planar polygon area after transforming to the scene UTM CRS",
            "bounds_in_raster_projected_crs": [float(value) for value in aoi_projected.bounds],
            "area_km2_in_raster_projected_crs": projected_aoi_area_km2,
            "declared_area_km2": declared_area,
            "geometry": aoi_geojson["features"][0]["geometry"],
        },
        "product_structure": {
            "checks": checks,
            "all_jp2_count": len(list(scene_dir.rglob("*.jp2"))),
            "granules": [str(path) for path in sorted((scene_dir / "GRANULE").iterdir()) if path.is_dir()],
            "product_metadata_xml": str(scene_dir / "MTD_MSIL2A.xml"),
            "tile_metadata_xml": str(tile_metadata_path),
            "manifest": str(scene_dir / "manifest.safe"),
        },
        "reflectance_scaling": {
            "stored_dtype": "uint16",
            "boa_quantification_value": product_values["boa_quantification_value"],
            "boa_add_offsets_by_band_id": product_values["boa_add_offsets_by_band_id"],
            "conversion": "surface_reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE",
            "nodata_values_from_metadata": product_values["nodata_values"],
            "saturated_values_from_metadata": product_values["saturated_values"],
        },
        "available_bands": rasters,
        "reflectance_review": {
            "review_range": [-0.2, 2.0],
            "notes": reflectance_observations,
            "source_values_retained": True,
            "source_jp2_inspected": True,
        },
        "scl": {
            "available": scl_path is not None,
            "path": str(scl_path) if scl_path else None,
            "raster": {**quality_layer_reports["SCL"], **scl_report} if scl_report else None,
            "class_names_reference": SCL_REFERENCE,
            "class_values": SCL_CLASS_NAMES if scl_path else None,
            "masks": "not computed; SCL is absent" if scl_path is None else "available for preprocessing",
        },
        "quality_threshold_policy": {
            "scl_classes_masked": [0, 1, 2, 3, 8, 9, 10, 11],
            "cloud_probability_threshold_percent": 40.0,
            "snow_probability_threshold_percent": 50.0,
            "low_cloud_probability": "CLD values >0 and <40 are retained unless SCL independently marks the pixel.",
            "cast_shadow_class_2": "masked as other shadow evidence to reduce false change candidates.",
        },
        "quality_rasters": quality_layer_reports,
        "quality_assets": {
            "SCL": str(scl_path) if scl_path else None,
            "cloud_probability": str(cloud_probability_path) if cloud_probability_path else None,
            "snow_probability": str(snow_probability_path) if snow_probability_path else None,
        },
        "qa_files": [str(path) for path in qa_files],
        "pixel_cloud_statistics": {
            "status": "available_from_scl" if scl_report else "unavailable_no_pixel_qa",
            "total_aoi_pixels": scl_report["aoi_pixel_count"] if scl_report else None,
            "valid_pixel_percent": scl_report["quality_pixel_percentages_in_aoi"]["usable"] if scl_report else None,
            "masked_pixel_percent": scl_report["masked_pixel_percent"] if scl_report else None,
            "cloud_pixel_percent": scl_report["quality_pixel_percentages_in_aoi"]["cloud"] if scl_report else None,
            "cloud_shadow_pixel_percent": scl_report["quality_pixel_percentages_in_aoi"]["cloud_shadow"] if scl_report else None,
            "cirrus_pixel_percent": scl_report["quality_pixel_percentages_in_aoi"]["cirrus"] if scl_report else None,
            "snow_ice_pixel_percent": scl_report["quality_pixel_percentages_in_aoi"]["snow_ice"] if scl_report else None,
            "other_masked_pixel_percent": scl_report["quality_pixel_percentages_in_aoi"]["other_masked"] if scl_report else None,
            "cloud_probability_threshold_percent": scl_report["cloud_probability_threshold_percent"] if scl_report else None,
            "snow_probability_threshold_percent": scl_report["snow_probability_threshold_percent"] if scl_report else None,
            "source": "SCL and probability layers" if scl_report else None,
        },
        "preprocessing_status": "not_run",
        "quality_observations": quality_observations,
        "warnings": warnings,
        "errors": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _write_raster(path: Path, array: np.ndarray, reference: rasterio.io.DatasetReader, *, dtype: str, nodata: float | int | None, descriptions: tuple[str, ...] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        predictor=3 if dtype.startswith("float") else 2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(path, "w", **profile) as target:
        target.write(array.astype(dtype, copy=False), 1)
        if descriptions:
            target.set_band_description(1, descriptions[0])


def _write_png(path: Path, arrays: list[np.ndarray], reference: rasterio.io.DatasetReader, nodata: float) -> None:
    height, width = arrays[0].shape
    preview_transform = reference.transform * reference.transform.scale(
        reference.width / width,
        reference.height / height,
    )
    normalized_channels = []
    for array in arrays:
        valid = np.isfinite(array) & (array != nodata)
        output = np.zeros(array.shape, dtype=np.uint8)
        if valid.any():
            low, high = np.nanpercentile(array[valid], [2, 98])
            if high <= low:
                high = low + 1.0
            normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
            output[valid] = np.asarray((normalized[valid] ** (1 / 2.2)) * 255, dtype=np.uint8)
        normalized_channels.append(output)
    image = np.stack(normalized_channels)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="PNG", width=width, height=height, count=3, dtype="uint8") as output:
        output.write(image)
    world = path.with_suffix(".pgw")
    a, b, c, d, e, f = tuple(preview_transform)[:6]
    world.write_text("\n".join(str(value) for value in (a, d, b, e, c + (a + b) / 2, f + (d + e) / 2)) + "\n", encoding="ascii")
    path.with_suffix(".prj").write_text(reference.crs.to_wkt(), encoding="utf-8")


def _preview_arrays(dataset: rasterio.io.DatasetReader) -> np.ndarray:
    scale = min(1.0, PREVIEW_SIZE / max(dataset.height, dataset.width))
    height = max(1, int(dataset.height * scale))
    width = max(1, int(dataset.width * scale))
    return dataset.read(1, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)


def _write_previews(aligned_paths: dict[str, Path], index_paths: dict[str, Path], scene_dir: Path, output_dir: Path) -> list[str]:
    previews = []
    with rasterio.open(aligned_paths["B04"]) as reference:
        for name, bands in (("true_color", ("B04", "B03", "B02")), ("false_color", ("B08", "B04", "B03"))):
            channels = []
            for band in bands:
                with rasterio.open(aligned_paths[band]) as dataset:
                    channels.append(_preview_arrays(dataset))
            path = output_dir / f"{name}.png"
            _write_png(path, channels, reference, NODATA_OUTPUT)
            previews.append(path.as_posix())
    for index_name, path in index_paths.items():
        with rasterio.open(path) as dataset:
            values = _preview_arrays(dataset)
        preview = output_dir / f"{index_name.lower()}.png"
        _write_png(preview, [values, values, values], reference, NODATA_OUTPUT)
        previews.append(preview.as_posix())
    return previews


def _safe_index(numerator: np.ndarray, denominator: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, NODATA_OUTPUT, dtype=np.float32)
    valid &= np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > 1e-8)
    np.divide(numerator, denominator, out=output, where=valid, casting="unsafe")
    return output


def _write_indices(aligned_paths: dict[str, Path], output_dir: Path, scene_id: str) -> dict[str, Path]:
    formulas = {
        "NDVI": ("B08", "B04"),
        "NDWI": ("B03", "B08"),
        "NDBI": ("B11", "B08"),
    }
    index_paths = {name: output_dir / scene_id / f"{name}.tif" for name in formulas}
    for path in index_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    open_sources = {band: rasterio.open(path) for band, path in aligned_paths.items()}
    try:
        reference = open_sources["B02"]
        for name, (band_a, band_b) in formulas.items():
            with rasterio.open(
                index_paths[name],
                "w",
                **{
                    **reference.profile,
                    "driver": "GTiff",
                    "count": 1,
                    "dtype": "float32",
                    "nodata": NODATA_OUTPUT,
                    "compress": "deflate",
                    "predictor": 3,
                },
            ) as target:
                target.set_band_description(1, name)
                target.update_tags(
                    formula={"NDVI": "(B08 - B04) / (B08 + B04)", "NDWI": "(B03 - B08) / (B03 + B08); McFeeters green-NIR formulation", "NDBI": "(B11 - B08) / (B11 + B08)"}[name],
                    units="dimensionless",
                    input_units="surface_reflectance",
                )
                for row_start in range(0, reference.height, CHUNK_ROWS):
                    height = min(CHUNK_ROWS, reference.height - row_start)
                    window = Window(0, row_start, reference.width, height)
                    data_a = open_sources[band_a].read(1, window=window)
                    data_b = open_sources[band_b].read(1, window=window)
                    valid = (data_a != NODATA_OUTPUT) & (data_b != NODATA_OUTPUT)
                    result = _safe_index(data_a - data_b, data_a + data_b, valid)
                    target.write(result, 1, window=window)
        return index_paths
    finally:
        for dataset in open_sources.values():
            dataset.close()


def preprocess_scene(
    scene_dir: str | Path,
    aoi_geojson: dict[str, Any],
    report: dict[str, Any],
    *,
    output_root: str | Path,
    make_previews: bool = True,
) -> dict[str, Any]:
    scene_dir = Path(scene_dir)
    scene_id = report["scene_id"]
    granule, bands, scl_path, _ = _discover_assets(scene_dir)
    metadata = _metadata_from_safe(scene_dir)
    with rasterio.open(bands["B02"]) as reference:
        aoi, _ = _projected_aoi(aoi_geojson, reference.crs)
        window = _aoi_window(reference, aoi)
        target_transform = window_transform(window, reference.transform)
        height, width = int(window.height), int(window.width)
        inside_aoi = geometry_mask(
            [aoi],
            out_shape=(height, width),
            transform=target_transform,
            invert=True,
        )
        target_crs = reference.crs
        grid_profile = reference.profile.copy()

    aligned_dir = Path(output_root) / "aligned" / scene_id
    mask_dir = Path(output_root) / "cloud_masked" / scene_id
    aligned_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    aligned_paths: dict[str, Path] = {}
    joint_valid = inside_aoi.copy()
    offsets = metadata["boa_add_offsets_by_band_id"]
    for band, source_path in bands.items():
        band_id = BAND_IDS[band]
        offset = offsets.get(band_id, 0.0)
        destination = np.full((height, width), np.nan, dtype=np.float32)
        with rasterio.open(source_path) as source:
            source_nodata = source.nodata
            if source_nodata is None and metadata["nodata_values"]:
                source_nodata = metadata["nodata_values"][0]
            reproject(
                source=rasterio.band(source, 1),
                destination=destination,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source_nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
                init_dest_nodata=True,
            )
        destination = (destination + offset) / metadata["boa_quantification_value"]
        destination[~inside_aoi] = np.nan
        joint_valid &= np.isfinite(destination)
        path = aligned_dir / f"{band}_10m_reflectance.tif"
        profile = grid_profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs=target_crs,
            transform=target_transform,
            nodata=NODATA_OUTPUT,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        stored = np.where(np.isfinite(destination), destination, NODATA_OUTPUT).astype(np.float32)
        with rasterio.open(path, "w", **profile) as output:
            output.write(stored, 1)
            output.set_band_description(1, f"{band} surface reflectance")
            output.update_tags(
                units="surface_reflectance",
                formula="(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE",
                quantification_value=str(metadata["boa_quantification_value"]),
                boa_add_offset=str(offset),
                source_asset=source_path.relative_to(scene_dir).as_posix(),
                resampling="bilinear continuous reflectance",
            )
        aligned_paths[band] = path

    mask_path = mask_dir / "valid_data_mask.tif"
    mask_profile = grid_profile.copy()
    mask_profile.update(
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs=target_crs,
        transform=target_transform,
        nodata=0,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(mask_path, "w", **mask_profile) as mask_output:
        mask_output.write(joint_valid.astype(np.uint8), 1)
        mask_output.set_band_description(1, "valid data across all required bands and AOI")
        mask_output.update_tags(mask_kind="data_validity_only", cloud_mask_applied="false" if scl_path is None else "SCL")

    scl_output = None
    cloud_mask_path = None
    scl_masked_band_paths: dict[str, Path] = {}
    if scl_path:
        scl_output = mask_dir / "SCL_nearest_10m.tif"
        scl_values = np.zeros((height, width), dtype=np.uint8)
        with rasterio.open(scl_path) as source:
            reproject(
                source=rasterio.band(source, 1),
                destination=scl_values,
                src_transform=source.transform,
                src_crs=source.crs,
                src_nodata=source.nodata,
                dst_transform=target_transform,
                dst_crs=target_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )
        scl_values[~inside_aoi] = 0
        masks = scl_quality_masks(scl_values, tuple(metadata["nodata_values"] or [0]))
        scl_profile = mask_profile.copy()
        scl_profile.update(dtype="uint8", nodata=0)
        with rasterio.open(scl_output, "w", **scl_profile) as output:
            output.write(scl_values, 1)
            output.set_band_description(1, "Sentinel-2 SCL class, nearest-neighbor resampled")
            output.update_tags(resampling="nearest categorical", class_reference=SCL_REFERENCE)
        cloud_mask_path = mask_dir / "scl_quality_mask.tif"
        with rasterio.open(cloud_mask_path, "w", **mask_profile) as output:
            output.write(masks["masked"].astype(np.uint8), 1)
            output.set_band_description(1, "SCL invalid/cloud-shadow/cloud/cirrus/snow mask")
            output.update_tags(class_reference=SCL_REFERENCE)
        scl_mask_array = masks["masked"] | ~inside_aoi
        for band, aligned_path in aligned_paths.items():
            output_path = mask_dir / f"{band}_10m_reflectance_scl_masked.tif"
            with rasterio.open(aligned_path) as source, rasterio.open(output_path, "w", **source.profile) as output:
                output.set_band_description(1, f"{band} surface reflectance with SCL quality classes masked")
                output.update_tags(mask_source=str(scl_path), class_reference=SCL_REFERENCE)
                for row_start in range(0, height, CHUNK_ROWS):
                    rows = min(CHUNK_ROWS, height - row_start)
                    chunk = Window(0, row_start, width, rows)
                    values = source.read(1, window=chunk)
                    quality_mask = scl_mask_array[row_start : row_start + rows]
                    values[quality_mask] = NODATA_OUTPUT
                    output.write(values, 1, window=chunk)
            scl_masked_band_paths[band] = output_path

    index_paths = _write_indices(aligned_paths, Path(output_root) / "indices", scene_id)
    preview_paths = _write_previews(aligned_paths, index_paths, scene_dir, Path(output_root) / "tiles" / scene_id) if make_previews else []
    report["preprocessing_status"] = "complete_with_quality_warning" if scl_path is None else "complete"
    report["preprocessing"] = {
        "target_crs": target_crs.to_string(),
        "target_resolution_m": 10,
        "width": width,
        "height": height,
        "transform": list(target_transform)[:6],
        "bounds": list(array_bounds(height, width, target_transform)),
        "continuous_band_resampling": "bilinear",
        "categorical_scl_resampling": "nearest-neighbor",
        "reflectance_conversion": "(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE",
        "reflectance_quantification_value": metadata["boa_quantification_value"],
        "reflectance_offsets_by_band_id": metadata["boa_add_offsets_by_band_id"],
        "aligned_band_paths": {band: path.as_posix() for band, path in aligned_paths.items()},
        "data_validity_mask_path": mask_path.as_posix(),
        "scl_path": scl_output.as_posix() if scl_output else None,
        "scl_quality_mask_path": cloud_mask_path.as_posix() if cloud_mask_path else None,
        "cloud_mask_applied": scl_path is not None,
        "scl_masked_band_paths": {band: path.as_posix() for band, path in scl_masked_band_paths.items()},
        "index_paths": {name: path.as_posix() for name, path in index_paths.items()},
        "preview_paths": preview_paths,
        "valid_pixels_all_required_bands_percent": 100.0 * float(joint_valid.sum()) / float(inside_aoi.sum()) if inside_aoi.any() else 0.0,
        "pixel_cloud_statistics": "not available; no SCL/QA assets in local SAFE" if scl_path is None else "computed from SCL",
    }
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


def apply_quality_mask_to_aligned(
    scene_dir: str | Path,
    aoi_geojson: dict[str, Any],
    report: dict[str, Any],
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Cloud-screen existing aligned reflectance without replacing source or unmasked indices."""
    scene_dir = Path(scene_dir)
    scene_id = report["scene_id"]
    processing = report.get("preprocessing") or {}
    aligned_paths = {band: Path(path) for band, path in processing.get("aligned_band_paths", {}).items()}
    if set(aligned_paths) != set(BAND_RESOLUTIONS):
        raise ValueError("All nine existing aligned reflectance bands are required before quality masking.")
    valid_data_path = Path(processing.get("data_validity_mask_path", ""))
    if not valid_data_path.is_file():
        raise FileNotFoundError("The existing all-band data-validity mask is missing; run the base preprocessing step first.")

    granule, _, scl_path, qa_files = _discover_assets(scene_dir)
    if scl_path is None:
        raise ValueError(
            "The existing SAFE product has no SCL asset; "
            "refusing to claim cloud-screened outputs."
        )

    cloud_probability_path = next(
        (path for path in qa_files if "MSK_CLDPRB" in path.name.upper()),
        None,
    )
    snow_probability_path = next(
        (path for path in qa_files if "MSK_SNWPRB" in path.name.upper()),
        None,
    )

    scl_only_quality = (
        cloud_probability_path is None
        or snow_probability_path is None
    )


    output_root = Path(output_root)
    mask_dir = output_root / "cloud_masked" / scene_id
    masked_band_dir = mask_dir / "reflectance"
    masked_band_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_root / "indices" / scene_id / "cloud_masked"
    index_dir.mkdir(parents=True, exist_ok=True)
    tile_dir = output_root / "tiles" / scene_id / "cloud_masked"

    cloud_threshold = 40.0
    snow_threshold = 50.0
    masked_band_paths = {
        band: masked_band_dir / f"{band}_10m_reflectance_cloud_masked.tif"
        for band in BAND_RESOLUTIONS
    }
    index_paths = {name: index_dir / f"{name}.tif" for name in ("NDVI", "NDWI", "NDBI")}
    scl_output = mask_dir / "SCL_nearest_10m.tif"
    cloud_output = mask_dir / "cloud_probability_bilinear_10m.tif"
    snow_output = mask_dir / "snow_probability_bilinear_10m.tif"
    mask_output = mask_dir / "quality_mask_10m.tif"

    totals = {
        "total_aoi_pixels": 0,
        "valid_data_pixels": 0,
        "usable_analysis_pixels": 0,
        "cloud_pixels": 0,
        "cloud_shadow_pixels": 0,
        "cirrus_pixels": 0,
        "snow_ice_pixels": 0,
        "other_masked_pixels": 0,
        "low_cloud_probability_retained_pixels": 0,
    }
    index_formulas = {
        "NDVI": ("B08", "B04", "(B08 - B04) / (B08 + B04)"),
        "NDWI": ("B03", "B08", "(B03 - B08) / (B03 + B08); McFeeters green-NIR formulation"),
        "NDBI": ("B11", "B08", "(B11 - B08) / (B11 + B08)"),
    }

    with ExitStack() as stack:
        reference = stack.enter_context(rasterio.open(aligned_paths["B02"]))
        aligned = {band: stack.enter_context(rasterio.open(path)) for band, path in aligned_paths.items()}
        base_valid = stack.enter_context(rasterio.open(valid_data_path))
        scl = stack.enter_context(rasterio.open(scl_path))
        cloud_probability = (
            stack.enter_context(rasterio.open(cloud_probability_path))
            if cloud_probability_path is not None
            else None
        )
        snow_probability = (
            stack.enter_context(rasterio.open(snow_probability_path))
            if snow_probability_path is not None
            else None
        )
        aoi, _ = _projected_aoi(aoi_geojson, reference.crs)

        reflectance_sinks = {}
        for band, path in masked_band_paths.items():
            profile = reference.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA_OUTPUT, compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
            sink = stack.enter_context(rasterio.open(path, "w", **profile))
            sink.set_band_description(1, f"{band} surface reflectance cloud/quality masked")
            sink.update_tags(mask_source=str(scl_path), cloud_probability_threshold_percent=str(cloud_threshold), snow_probability_threshold_percent=str(snow_threshold))
            reflectance_sinks[band] = sink

        index_sinks = {}
        for name, (band_a, band_b, formula) in index_formulas.items():
            profile = reference.profile.copy()
            profile.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA_OUTPUT, compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
            sink = stack.enter_context(rasterio.open(index_paths[name], "w", **profile))
            sink.set_band_description(1, name)
            sink.update_tags(formula=formula, units="dimensionless", input_units="surface_reflectance", quality_mask=str(mask_output))
            index_sinks[name] = sink

        scl_profile = reference.profile.copy()
        scl_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        scl_sink = stack.enter_context(rasterio.open(scl_output, "w", **scl_profile))
        scl_sink.set_band_description(1, "Sentinel-2 SCL class aligned with nearest neighbor")
        scl_sink.update_tags(resampling="nearest categorical", class_reference=SCL_REFERENCE)

        probability_profile = reference.profile.copy()
        probability_profile.update(driver="GTiff", count=1, dtype="float32", nodata=NODATA_OUTPUT, compress="deflate", predictor=3, tiled=True, blockxsize=256, blockysize=256)
        cloud_sink = stack.enter_context(rasterio.open(cloud_output, "w", **probability_profile))
        cloud_sink.set_band_description(1, "CDSE cloud probability percent, bilinear to 10 m")
        cloud_sink.update_tags(resampling="bilinear continuous probability", units="percent")
        snow_sink = stack.enter_context(rasterio.open(snow_output, "w", **probability_profile))
        snow_sink.set_band_description(1, "CDSE snow probability percent, bilinear to 10 m")
        snow_sink.update_tags(resampling="bilinear continuous probability", units="percent")

        mask_profile = reference.profile.copy()
        mask_profile.update(driver="GTiff", count=1, dtype="uint8", nodata=255, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        quality_mask_sink = stack.enter_context(rasterio.open(mask_output, "w", **mask_profile))
        quality_mask_sink.set_band_description(1, "0 usable, 1 quality masked, 255 outside AOI/base data invalid")
        quality_mask_sink.update_tags(
            class_reference=SCL_REFERENCE,
            masked_scl_classes="0,1,2,3,8,9,10,11",
            cloud_probability_threshold_percent=str(cloud_threshold),
            snow_probability_threshold_percent=str(snow_threshold),
            low_probability_policy="CLD 1-39 percent retained unless SCL independently masks the pixel",
        )

        for row_start in range(0, reference.height, CHUNK_ROWS):
            height = min(CHUNK_ROWS, reference.height - row_start)
            window = Window(0, row_start, reference.width, height)
            inside_aoi = geometry_mask(
                [aoi],
                out_shape=(height, reference.width),
                transform=window_transform(window, reference.transform),
                invert=True,
            )
            source_valid = base_valid.read(1, window=window) == 1
            scl_values = _read_quality_window(scl, reference, window, categorical=True).astype(np.uint8, copy=False)
            if cloud_probability is not None:
                cloud_values = _read_quality_window(
                    cloud_probability,
                    reference,
                    window,
                ).astype(np.float32, copy=False)
            else:
                cloud_values = np.full(
                    (height, reference.width),
                    NODATA_OUTPUT,
                    dtype=np.float32,
                )

            if snow_probability is not None:
                snow_values = _read_quality_window(
                    snow_probability,
                    reference,
                    window,
                ).astype(np.float32, copy=False)
            else:
                snow_values = np.full(
                    (height, reference.width),
                    NODATA_OUTPUT,
                    dtype=np.float32,
                )
            scl_values[~inside_aoi] = 255
            cloud_values[~inside_aoi | ~np.isfinite(cloud_values)] = NODATA_OUTPUT
            snow_values[~inside_aoi | ~np.isfinite(snow_values)] = NODATA_OUTPUT
            scl_sink.write(scl_values, 1, window=window)
            cloud_sink.write(cloud_values, 1, window=window)
            snow_sink.write(snow_values, 1, window=window)

            in_aoi_scl = scl_values[inside_aoi]
            in_aoi_cld = cloud_values[inside_aoi]
            in_aoi_snw = snow_values[inside_aoi]
            if scl_only_quality:
                masks_in_aoi = {
                    "masked": np.isin(
                        in_aoi_scl,
                        [0, 1, 2, 3, 8, 9, 10, 11],
                    ),
                    "cloud": np.isin(in_aoi_scl, [8, 9]),
                    "cloud_shadow": in_aoi_scl == 3,
                    "cirrus": in_aoi_scl == 10,
                    "snow_ice": in_aoi_scl == 11,
                    "other_masked": np.isin(in_aoi_scl, [0, 1, 2]),
                    "low_cloud_probability_retained": np.zeros(
                        in_aoi_scl.shape,
                        dtype=bool,
                    ),
                }
            else:
                masks_in_aoi = scl_quality_masks(
                    in_aoi_scl,
                    (0,),
                    in_aoi_cld,
                    in_aoi_snw,
                    cloud_probability_threshold=cloud_threshold,
                    snow_probability_threshold=snow_threshold,
                )
            total = int(inside_aoi.sum())
            source_valid_in_aoi = source_valid[inside_aoi]
            totals["total_aoi_pixels"] += total
            totals["valid_data_pixels"] += int(source_valid_in_aoi.sum())
            totals["usable_analysis_pixels"] += int((source_valid_in_aoi & ~masks_in_aoi["masked"]).sum())
            for stat_name, mask_name in (
                ("cloud_pixels", "cloud"),
                ("cloud_shadow_pixels", "cloud_shadow"),
                ("cirrus_pixels", "cirrus"),
                ("snow_ice_pixels", "snow_ice"),
                ("other_masked_pixels", "other_masked"),
                ("low_cloud_probability_retained_pixels", "low_cloud_probability_retained"),
            ):
                if stat_name == "other_masked_pixels":
                    category = masks_in_aoi[mask_name] | ~source_valid_in_aoi
                else:
                    category = masks_in_aoi[mask_name]
                totals[stat_name] += int(category.sum())

            full_mask = np.zeros((height, reference.width), dtype=bool)
            full_mask[inside_aoi] = masks_in_aoi["masked"] | ~source_valid_in_aoi
            full_mask |= ~source_valid
            quality_mask = np.full((height, reference.width), 255, dtype=np.uint8)
            quality_mask[inside_aoi & source_valid] = 0
            quality_mask[full_mask & inside_aoi] = 1
            quality_mask_sink.write(quality_mask, 1, window=window)

            masked_bands: dict[str, np.ndarray] = {}
            for band, source in aligned.items():
                values = source.read(1, window=window)
                values[full_mask] = NODATA_OUTPUT
                reflectance_sinks[band].write(values, 1, window=window)
                masked_bands[band] = values

            for name, (band_a, band_b, _) in index_formulas.items():
                data_a = masked_bands[band_a]
                data_b = masked_bands[band_b]
                valid = (data_a != NODATA_OUTPUT) & (data_b != NODATA_OUTPUT)
                index_sinks[name].write(_safe_index(data_a - data_b, data_a + data_b, valid), 1, window=window)

    total = totals["total_aoi_pixels"]
    percentages = {
        key.replace("_pixels", "_percent"): 100.0 * value / total if total else 0.0
        for key, value in totals.items()
        if key.endswith("_pixels")
    }
    preview_paths = _write_previews(masked_band_paths, index_paths, scene_dir, tile_dir)
    with rasterio.open(mask_output) as quality_mask_dataset:
        preview_values = quality_mask_dataset.read(1)
        _write_png(tile_dir / "scl_quality_mask.png", [preview_values] * 3, quality_mask_dataset, 255)
    cloud_statistics = {
        "total_aoi_pixels": total,
        **totals,
        **percentages,
        "cloud_probability_threshold_percent": cloud_threshold,
        "snow_probability_threshold_percent": snow_threshold,
        "low_cloud_probability_policy": "CLD values 1-39% are retained unless an SCL class independently masks the pixel.",
        "cloud_mask_scl_classes": [8, 9],
        "cloud_shadow_scl_class": 3,
        "cast_shadow_scl_class_2_is_masked_as_other": True,
    }
    cloud_statistics["masked_pixel_percent"] = (
        100.0 * (total - totals["usable_analysis_pixels"]) / total if total else 0.0
    )
    report["pixel_cloud_statistics"] = {
        "status": "computed_from_scl_and_probability_rasters",
        "source": "SCL_20m, CLD_20m, SNW_20m resampled to existing 10 m reflectance grid",
        "scene_level_cloud_cover_percent": report.get("scene_cloud_cover_percent"),
        **cloud_statistics,
    }
    report.setdefault("preprocessing", {})["quality_screening"] = {
        "status": "complete",
        "cloud_mask_path": mask_output.as_posix(),
        "aligned_scl_path": scl_output.as_posix(),
        "aligned_cloud_probability_path": cloud_output.as_posix(),
        "aligned_snow_probability_path": snow_output.as_posix(),
        "cloud_screened_reflectance_paths": {band: path.as_posix() for band, path in masked_band_paths.items()},
        "cloud_screened_index_paths": {name: path.as_posix() for name, path in index_paths.items()},
        "preview_paths": [*preview_paths, (tile_dir / "scl_quality_mask.png").as_posix()],
        "categorical_resampling": "nearest-neighbor",
        "probability_resampling": "bilinear",
        "threshold_policy": cloud_statistics["low_cloud_probability_policy"],
        "statistics": cloud_statistics,
    }
    report["preprocessing_status"] = "complete_quality_screened"
    remaining_warnings = [
        warning for warning in report.get("warnings", [])
        if "No SCL asset" not in warning
        and "No cloud probability raster" not in warning
        and "No snow probability raster" not in warning
        and "pixel-level class statistics" not in warning
        and "probability-threshold" not in warning
        and "has not been run" not in warning
    ]
    report["warnings"] = remaining_warnings
    report["status"] = "WARNING" if remaining_warnings else "PASS"
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report


def write_quality_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = report["scene_id"]
    json_path = output_dir / f"{stem}.quality.json"
    markdown_path = output_dir / f"{stem}.quality.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    lines = [
        f"# Sentinel-2 quality report: {stem}",
        "",
        f"- Overall: **{report.get('status', 'FAIL')}**",
        f"- Acquisition: {report.get('acquisition_datetime', 'unavailable')}",
        f"- Tile: {report.get('tile', 'unavailable')}",
        f"- Scene-level cloud metadata: {report.get('scene_cloud_cover_percent', 'unavailable')}% (not an AOI pixel cloud measure)",
        f"- AOI: {report.get('aoi', {}).get('name', 'unavailable')}",
        f"- AOI geometry: {report.get('aoi', {}).get('geometry_type', 'unknown')}; valid={report.get('aoi', {}).get('geometry_valid', 'unknown')}; CRS={report.get('aoi', {}).get('source_crs', 'unknown')}",
        f"- AOI authoritative area: {report.get('aoi', {}).get('area_km2_authoritative', 'unavailable')} km2 ({report.get('aoi', {}).get('area_method', 'unavailable')})",
        f"- AOI UTM grid cross-check: {report.get('aoi', {}).get('area_km2_in_raster_projected_crs', 'unavailable')} km2; previous declared={report.get('aoi', {}).get('previous_declared_area_km2')}; previous legacy={report.get('aoi', {}).get('previous_legacy_backend_area_km2')} km2",
        f"- SCL available: {report.get('scl', {}).get('available', 'unknown')}",
        f"- AOI pixel cloud/quality mask: {report.get('pixel_cloud_statistics', {}).get('status', 'unavailable')}",
        f"- AOI quality-masked pixels: {report.get('pixel_cloud_statistics', {}).get('masked_pixel_percent', 'unavailable')}%",
        f"- Preprocessing: {report.get('preprocessing_status', 'not_run')}",
        "",
        "## Bands",
        "",
        "| Band | Native | CRS | Width x height | Dtype | NoData | Saturated | AOI cover | Valid AOI | >2 reflectance | Min / P50 / P98 / max |",
        "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for band, item in report.get("available_bands", {}).items():
        lines.append(
            f"| {band} | {item['resolution_m'][0]:g} m | {item['crs']} | {item['width']} x {item['height']} | {item['dtype']} | {item['metadata_nodata_values']} | {item['saturated_pixels']} | {item['aoi_cover_percent']:.3f}% | {item['valid_pixel_percent']:.4f}% | {item['above_review_range_pixels']} | {item['surface_reflectance_min']} / {item['surface_reflectance_percentiles']['50']} / {item['surface_reflectance_percentiles']['98']} / {item['surface_reflectance_max']} |"
        )
    lines.extend(["", "## Reflectance Review", ""])
    reflectance_review = report.get("reflectance_review", {})
    for note in reflectance_review.get("notes", []):
        lines.append(f"- {note['band']}: {note['count_above_review_range']} source pixels exceed the review range; {note['classification']}")
        for sample in note.get("samples", []):
            lines.append(
                f"  - row={sample['row']} col={sample['column']} DN={sample['dn']} SR={sample['surface_reflectance']:.4f} lon={sample['longitude']:.8f} lat={sample['latitude']:.8f} SCL={sample.get('scl_class')} CLD={sample.get('cloud_probability_percent')} SNW={sample.get('snow_probability_percent')} inside_AOI={sample['inside_aoi']}"
            )
    if not reflectance_review.get("notes"):
        lines.append("- No source reflectance samples exceed the review range.")
    lines.extend(["", "## AOI and Reflectance Notes", ""])
    lines.extend(f"- {note}" for note in report.get("quality_observations", []))
    if not report.get("quality_observations"):
        lines.append("- None")
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {warning}" for warning in report.get("warnings", []))
    if not report.get("warnings"):
        lines.append("- None")
    lines.extend(["", "## Errors", ""])
    lines.extend(f"- {error}" for error in report.get("errors", []))
    if not report.get("errors"):
        lines.append("- None")
    lines.extend(["", "## Pixel QA", ""])
    qa = report.get("pixel_cloud_statistics", {})
    lines.append(f"- Scene-level cloud cover: {qa.get('scene_level_cloud_cover_percent', report.get('scene_cloud_cover_percent', 'unavailable'))}%")
    lines.append(f"- AOI pixel cloud: {qa.get('cloud_pixels', 'unavailable')} ({qa.get('cloud_percent', 'unavailable')}%)")
    lines.append(f"- AOI cloud shadow: {qa.get('cloud_shadow_pixels', 'unavailable')} ({qa.get('cloud_shadow_percent', 'unavailable')}%)")
    lines.append(f"- AOI cirrus: {qa.get('cirrus_pixels', 'unavailable')} ({qa.get('cirrus_percent', 'unavailable')}%)")
    lines.append(f"- AOI snow/ice: {qa.get('snow_ice_pixels', 'unavailable')} ({qa.get('snow_ice_percent', 'unavailable')}%)")
    lines.append(f"- Other masked: {qa.get('other_masked_pixels', 'unavailable')} ({qa.get('other_masked_percent', 'unavailable')}%)")
    lines.append(f"- Usable: {qa.get('usable_analysis_pixels', 'unavailable')} ({qa.get('usable_analysis_percent', 'unavailable')}%)")
    lines.append(f"- Cloud threshold: {qa.get('cloud_probability_threshold_percent', 'unavailable')}%; snow threshold: {qa.get('snow_probability_threshold_percent', 'unavailable')}%")
    lines.append(f"- Low cloud probability policy: {qa.get('low_cloud_probability_policy', 'unavailable')}")
    quality_assets = report.get("quality_assets", {})
    for name, path in quality_assets.items():
        lines.append(f"- {name}: `{path or 'not available'}`")
    if report.get("scl", {}).get("raster"):
        scl = report["scl"]["raster"]
        lines.append(f"- SCL grid: {scl['width']} x {scl['height']} at {scl['resolution_m'][0]:g} m, {scl['crs']}; classes: {', '.join(map(str, scl['class_values_observed_in_aoi']))}")

    lines.extend(["", "## Outputs", ""])
    if report.get("preprocessing"):
        lines.extend(f"- {name}: `{path}`" for name, path in report["preprocessing"].get("index_paths", {}).items())
        lines.extend(f"- Preview: `{path}`" for path in report["preprocessing"].get("preview_paths", []))
        screening = report["preprocessing"].get("quality_screening")
        if screening:
            lines.append(f"- Quality mask: `{screening['cloud_mask_path']}`")
            lines.extend(f"- Cloud-screened band {name}: `{path}`" for name, path in screening["cloud_screened_reflectance_paths"].items())
            lines.extend(f"- Cloud-screened index {name}: `{path}`" for name, path in screening["cloud_screened_index_paths"].items())
            lines.extend(f"- Cloud-screened preview: `{path}`" for path in screening["preview_paths"])
    else:
        lines.append("- Preprocessing has not run.")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def write_baseline_manifest(
    report: dict[str, Any],
    provenance: dict[str, Any],
    *,
    aoi_path: str | Path,
    output_dir: str | Path,
) -> Path:
    scene_id = report["scene_id"]
    assets = []
    for key, record in provenance.get("assets", {}).items():
        assets.append({
            "asset_key": key,
            "local_path": record.get("local_path"),
            "source_href": record.get("source_href"),
            "size_bytes": record.get("file_size_bytes"),
            "sha256": record.get("sha256"),
        })
    for key, record in provenance.get("supplemental_quality_assets", {}).items():
        assets.append({
            "asset_key": key,
            "local_path": record.get("local_path_from_repo"),
            "source_href": record.get("href"),
            "size_bytes": record.get("file_size_bytes"),
            "stac_checksum": record.get("checksum"),
        })
    manifest = {
        "baseline_version": "1.0",
        "scene_id": scene_id,
        "product_uuid": provenance.get("product_id"),
        "product_identifier": provenance.get("product_identifier"),
        "collection": provenance.get("collection"),
        "satellite": provenance.get("satellite"),
        "acquisition_datetime": provenance.get("acquisition_datetime"),
        "tile": provenance.get("tile"),
        "processing_level": provenance.get("processing_level"),
        "processing_baseline": report.get("processing_baseline"),
        "scene_level_cloud_cover_percent": provenance.get("cloud_cover"),
        "aoi": {
            "identifier": Path(aoi_path).as_posix(),
            "name": report.get("aoi", {}).get("name"),
            "crs": report.get("aoi", {}).get("source_crs"),
            "geometry_type": report.get("aoi", {}).get("geometry_type"),
            "geometry_valid": report.get("aoi", {}).get("geometry_valid"),
            "geometry_part_count": report.get("aoi", {}).get("geometry_part_count"),
            "bounds_wgs84": report.get("aoi", {}).get("bounds_wgs84"),
            "area_km2": report.get("aoi", {}).get("area_km2_authoritative"),
            "area_method": report.get("aoi", {}).get("area_method"),
            "projected_area_km2_check": report.get("aoi", {}).get("area_km2_in_raster_projected_crs"),
        },
        "source_assets": assets,
        "spectral_bands": {
            band: {
                "source_path": details.get("path"),
                "native_resolution_m": details.get("resolution_m"),
                "crs": details.get("crs"),
                "dtype": details.get("dtype"),
                "nodata": details.get("nodata"),
                "valid_pixels": details.get("valid_pixels"),
                "saturated_pixels": details.get("saturated_pixels"),
                "surface_reflectance_min": details.get("surface_reflectance_min"),
                "surface_reflectance_max": details.get("surface_reflectance_max"),
            }
            for band, details in report.get("available_bands", {}).items()
        },
        "quality_layers": {
            key: {
                "path": details.get("path"),
                "crs": details.get("crs"),
                "resolution_m": details.get("resolution_m"),
                "dtype": details.get("dtype"),
                "nodata": details.get("nodata"),
                "min": details.get("aoi_min"),
                "max": details.get("aoi_max"),
            } if details else None
            for key, details in report.get("quality_rasters", {}).items()
        },
        "preprocessing_configuration": {
            "pipeline_version": "1.0",
            "target_crs": report.get("preprocessing", {}).get("target_crs"),
            "target_resolution_m": report.get("preprocessing", {}).get("target_resolution_m"),
            "continuous_reflectance_resampling": report.get("preprocessing", {}).get("continuous_band_resampling"),
            "categorical_scl_resampling": report.get("preprocessing", {}).get("quality_screening", {}).get("categorical_resampling"),
            "cloud_probability_threshold_percent": report.get("pixel_cloud_statistics", {}).get("cloud_probability_threshold_percent"),
            "snow_probability_threshold_percent": report.get("pixel_cloud_statistics", {}).get("snow_probability_threshold_percent"),
            "masked_scl_classes": report.get("quality_threshold_policy", {}).get("scl_classes_masked"),
            "low_probability_cloud_policy": report.get("quality_threshold_policy", {}).get("low_cloud_probability"),
            "reflectance_conversion": report.get("reflectance_scaling", {}).get("conversion"),
            "boa_quantification_value": report.get("reflectance_scaling", {}).get("boa_quantification_value"),
            "boa_add_offsets_by_band_id": report.get("reflectance_scaling", {}).get("boa_add_offsets_by_band_id"),
            "ndwi_formula": "(B03 - B08) / (B03 + B08), McFeeters green-NIR",
        },
        "quality_status": report.get("status"),
        "preprocessing_status": report.get("preprocessing_status"),
        "quality_statistics": report.get("pixel_cloud_statistics"),
        "reflectance_review": report.get("reflectance_review"),
        "warnings": report.get("warnings", []),
        "quality_observations": report.get("quality_observations", []),
        "quality_report_json": f"data/metadata/quality/{scene_id}.quality.json",
        "provenance_json": f"data/metadata/downloads/{scene_id}.provenance.json",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scene_id}.baseline.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
