from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping
from shapely.ops import transform as transform_geometry

from backend.app.services.scene_processing import (
    BAND_FILES,
    BAND_RESOLUTIONS,
    NODATA_OUTPUT,
    SCL_CLASS_NAMES,
    _safe_index,
    apply_quality_mask_to_aligned,
    inspect_scene,
    preprocess_scene,
    scl_quality_masks,
    write_baseline_manifest,
)


SCENE_ID = "S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519"
PRODUCT_NAME = f"{SCENE_ID}.SAFE"


def _xml_fixture(scene_dir: Path, *, include_scl: bool = False) -> None:
    (scene_dir / "manifest.safe").write_text("<XFDU><informationPackageMap/></XFDU>", encoding="utf-8")
    (scene_dir / "MTD_MSIL2A.xml").write_text(
        f"""<Level-2A_User_Product>
          <PRODUCT_URI>{PRODUCT_NAME}</PRODUCT_URI>
          <PRODUCT_START_TIME>2026-09-10T05:26:41.025Z</PRODUCT_START_TIME>
          <PROCESSING_LEVEL>Level-2A</PROCESSING_LEVEL>
          <PROCESSING_BASELINE>05.12</PROCESSING_BASELINE>
          <BOA_QUANTIFICATION_VALUE>10000</BOA_QUANTIFICATION_VALUE>
          <BOA_ADD_OFFSET band_id="0">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="1">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="3">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="4">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="5">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="6">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="7">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="8">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="9">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="10">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="11">-1000</BOA_ADD_OFFSET>
          <BOA_ADD_OFFSET band_id="12">-1000</BOA_ADD_OFFSET>
                    <SPECIAL_VALUES>
                        <SPECIAL_VALUE><SPECIAL_VALUE_TEXT>NODATA</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>0</SPECIAL_VALUE_INDEX></SPECIAL_VALUE>
                        <SPECIAL_VALUE><SPECIAL_VALUE_TEXT>SATURATED</SPECIAL_VALUE_TEXT><SPECIAL_VALUE_INDEX>65535</SPECIAL_VALUE_INDEX></SPECIAL_VALUE>
                    </SPECIAL_VALUES>
          <Cloud_Coverage_Assessment>0.0</Cloud_Coverage_Assessment>
        </Level-2A_User_Product>""",
        encoding="utf-8",
    )
    granule = scene_dir / "GRANULE" / "L2A_T43RFM_A010514_20260910T053601"
    (granule / "IMG_DATA" / "R10m").mkdir(parents=True)
    (granule / "IMG_DATA" / "R20m").mkdir(parents=True)
    (granule / "MTD_TL.xml").write_text(
        "<Level-2A_Tile_ID><TILE_ID>S2C_OPER_T43RFM</TILE_ID><CLOUDY_PIXEL_PERCENTAGE>0</CLOUDY_PIXEL_PERCENTAGE></Level-2A_Tile_ID>",
        encoding="utf-8",
    )
    for band, resolution in BAND_RESOLUTIONS.items():
        width = height = 20 if resolution == 10 else 10
        path = granule / "IMG_DATA" / f"R{resolution}m" / f"T43RFM_20260910T052641{BAND_FILES[band]}"
        values = np.full((height, width), 11000, dtype=np.uint16)
        if band == "B04":
            values[:, :] = 4000
        elif band == "B08":
            values[:, :] = 8000
        elif band == "B03":
            values[:, :] = 5000
        elif band in {"B11", "B12"}:
            values[:, :] = 6000
        if band == "B02":
            values[5, 5] = 0
            values[5, 6] = 31000
        if band == "B03":
            values[5, 6] = 25000
            values[5, 7] = 65535
        with rasterio.open(
            path,
            "w",
            driver="JP2OpenJPEG",
            width=width,
            height=height,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(500000, 3200000, resolution, resolution),
            REVERSIBLE="YES",
        ) as dataset:
            dataset.write(values, 1)
    if include_scl:
        scl_path = granule / "IMG_DATA" / "R20m" / "T43RFM_20260910T052641_SCL_20m.jp2"
        scl_values = np.full((10, 10), 4, dtype=np.uint8)
        scl_values[3:5, 3:5] = 8
        scl_values[5:7, 5:7] = 3
        scl_values[1, 1] = 7
        with rasterio.open(
            scl_path,
            "w",
            driver="JP2OpenJPEG",
            width=10,
            height=10,
            count=1,
            dtype="uint8",
            crs="EPSG:32643",
            transform=from_origin(500000, 3200000, 20, 20),
            REVERSIBLE="YES",
        ) as dataset:
            dataset.write(scl_values, 1)
        for name, filename, probability in (
            ("cloud", "MSK_CLDPRB_20m.jp2", np.zeros((10, 10), dtype=np.uint8)),
            ("snow", "MSK_SNWPRB_20m.jp2", np.zeros((10, 10), dtype=np.uint8)),
        ):
            probability[1, 1] = 45 if name == "cloud" else 0
            probability[2, 2] = 0 if name == "cloud" else 55
            path = granule / "QI_DATA" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(
                path,
                "w",
                driver="JP2OpenJPEG",
                width=10,
                height=10,
                count=1,
                dtype="uint8",
                crs="EPSG:32643",
                transform=from_origin(500000, 3200000, 20, 20),
                REVERSIBLE="YES",
            ) as dataset:
                dataset.write(probability, 1)


def _aoi() -> dict:
    polygon = Polygon([(500010, 3199810), (500190, 3199810), (500190, 3199990), (500010, 3199990), (500010, 3199810)])
    to_wgs84 = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True).transform
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
        "features": [{"type": "Feature", "properties": {"name": "test AOI"}, "geometry": mapping(transform_geometry(to_wgs84, polygon))}],
    }


def test_safe_discovery_band_metadata_and_absent_scl(tmp_path):
    scene_dir = tmp_path / PRODUCT_NAME
    scene_dir.mkdir()
    _xml_fixture(scene_dir)
    report = inspect_scene(scene_dir, _aoi())
    assert report["status"] == "WARNING"
    assert report["product_structure"]["checks"]["safe_structure"] == "PASS"
    assert report["product_structure"]["checks"]["required_bands"] == "PASS"
    assert set(report["available_bands"]) == set(BAND_RESOLUTIONS)
    assert report["reflectance_scaling"]["boa_quantification_value"] == 10000
    assert report["reflectance_scaling"]["boa_add_offsets_by_band_id"][2] == -1000
    assert report["scl"]["available"] is False
    assert report["pixel_cloud_statistics"]["valid_pixel_percent"] is None
    assert report["available_bands"]["B02"]["invalid_pixels"] > 0
    assert report["available_bands"]["B03"]["saturated_pixels"] == 1
    assert report["available_bands"]["B02"]["surface_reflectance_percentiles"]["50"] is not None
    assert report["aoi"]["area_km2_in_raster_projected_crs"] == pytest.approx(0.0324, rel=0.02)


def test_preprocessing_aligns_bands_clips_applies_scaling_and_writes_indices(tmp_path):
    scene_dir = tmp_path / PRODUCT_NAME
    scene_dir.mkdir()
    _xml_fixture(scene_dir)
    report = inspect_scene(scene_dir, _aoi())
    updated = preprocess_scene(scene_dir, _aoi(), report, output_root=tmp_path / "processed", make_previews=False)
    preprocessing = updated["preprocessing"]
    assert preprocessing["target_resolution_m"] == 10
    assert preprocessing["target_crs"] == "EPSG:32643"
    assert preprocessing["cloud_mask_applied"] is False
    assert preprocessing["scl_path"] is None
    with rasterio.open(preprocessing["aligned_band_paths"]["B04"]) as red:
        assert red.res == (10.0, 10.0)
        assert red.crs.to_string() == "EPSG:32643"
        assert red.nodata == NODATA_OUTPUT
        assert red.read(1)[10, 10] == pytest.approx(0.3, abs=1e-4)
    with rasterio.open(preprocessing["aligned_band_paths"]["B05"]) as red_edge:
        assert red_edge.res == (10.0, 10.0)
        assert red_edge.width == preprocessing["width"]
        assert red_edge.height == preprocessing["height"]
    with rasterio.open(preprocessing["index_paths"]["NDVI"]) as ndvi:
        value = ndvi.read(1)[10, 10]
        assert value == pytest.approx((0.7 - 0.3) / (0.7 + 0.3), abs=1e-5)
        assert ndvi.tags()["formula"] == "(B08 - B04) / (B08 + B04)"
    with rasterio.open(preprocessing["index_paths"]["NDWI"]) as ndwi:
        assert ndwi.read(1)[10, 10] == pytest.approx((0.4 - 0.7) / (0.4 + 0.7), abs=0.002)
        assert "McFeeters" in ndwi.tags()["formula"]
    with rasterio.open(preprocessing["index_paths"]["NDBI"]) as ndbi:
        assert ndbi.read(1)[10, 10] == pytest.approx((0.5 - 0.7) / (0.5 + 0.7), abs=0.002)


def test_scl_quality_mask_retains_useful_landcover_classes():
    values = np.array([[0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype=np.uint8)
    masks = scl_quality_masks(values)
    assert masks["masked"].tolist() == [[True, True, True, False, False, False, False, True, True, True, True]]
    assert SCL_CLASS_NAMES[4] == "VEGETATION"
    assert SCL_CLASS_NAMES[6] == "WATER"
    assert SCL_CLASS_NAMES[10] == "THIN_CIRRUS"


def test_cloud_and_snow_probability_threshold_policy_is_explicit():
    scl = np.array([[7, 7, 4, 4]], dtype=np.uint8)
    cloud_probability = np.array([[39.0, 40.0, 0.0, 0.0]], dtype=np.float32)
    snow_probability = np.array([[0.0, 0.0, 49.0, 50.0]], dtype=np.float32)
    masks = scl_quality_masks(scl, cloud_probability=cloud_probability, snow_probability=snow_probability)
    assert masks["low_cloud_probability_retained"].tolist() == [[True, False, False, False]]
    assert masks["cloud"].tolist() == [[False, True, False, False]]
    assert masks["snow_ice"].tolist() == [[False, False, False, True]]
    assert masks["masked"].tolist() == [[False, True, False, True]]


def test_scl_is_discovered_and_resampled_with_nearest_neighbor(tmp_path):
    scene_dir = tmp_path / PRODUCT_NAME
    scene_dir.mkdir()
    _xml_fixture(scene_dir, include_scl=True)
    report = inspect_scene(scene_dir, _aoi())
    assert report["scl"]["available"] is True
    assert report["scl"]["path"].endswith("SCL_20m.jp2")
    assert report["pixel_cloud_statistics"]["status"] == "available_from_scl"
    assert report["scl"]["raster"]["masked_pixel_percent"] > 0
    assert report["scl"]["raster"]["class_names_observed_in_aoi"]["4"] == "VEGETATION"
    assert report["pixel_cloud_statistics"]["cloud_probability_threshold_percent"] == 40.0
    assert report["pixel_cloud_statistics"]["cloud_pixel_percent"] > 0
    updated = preprocess_scene(scene_dir, _aoi(), report, output_root=tmp_path / "processed", make_previews=False)
    scl_path = updated["preprocessing"]["scl_path"]
    mask_path = updated["preprocessing"]["scl_quality_mask_path"]
    with rasterio.open(scl_path) as dataset:
        classes = set(np.unique(dataset.read(1)).tolist())
        assert classes <= set(SCL_CLASS_NAMES)
        assert dataset.tags()["resampling"] == "nearest categorical"
    with rasterio.open(mask_path) as dataset:
        assert set(np.unique(dataset.read(1)).tolist()) <= {0, 1}
        assert dataset.read(1).max() == 1
    masked_band = updated["preprocessing"]["scl_masked_band_paths"]["B04"]
    with rasterio.open(masked_band) as dataset:
        assert dataset.nodata == NODATA_OUTPUT
        assert NODATA_OUTPUT in np.unique(dataset.read(1))


def test_safe_index_masks_zero_denominator_and_invalid_pixels():
    numerator = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    denominator = np.array([[2.0, 0.0, 2.0]], dtype=np.float32)
    valid = np.array([[True, True, False]])
    result = _safe_index(numerator, denominator, valid)
    assert result[0, 0] == pytest.approx(0.5)
    assert result[0, 1] == NODATA_OUTPUT
    assert result[0, 2] == NODATA_OUTPUT


def test_reflectance_review_traces_outliers_to_unmodified_source_dn(tmp_path):
    scene_dir = tmp_path / PRODUCT_NAME
    scene_dir.mkdir()
    _xml_fixture(scene_dir)
    report = inspect_scene(scene_dir, _aoi())
    band = report["available_bands"]["B02"]
    assert band["high_reflectance_source_pixels"][0]["dn"] == 31000
    assert band["high_reflectance_source_pixels"][0]["surface_reflectance"] == pytest.approx(3.0)
    assert band["high_reflectance_source_pixels"][0]["inside_aoi"] is True
    assert band["surface_reflectance_percentiles"]["50"] is not None
    assert report["reflectance_scaling"]["saturated_values_from_metadata"] == [65535]
    assert report["reflectance_review"]["source_values_retained"] is True
    assert report["reflectance_review"]["notes"]


def test_baseline_manifest_references_scene_assets_and_quality_without_rasters(tmp_path):
    report = {
        "scene_id": SCENE_ID,
        "product_uuid": "unused",
        "processing_baseline": "05.12",
        "status": "PASS",
        "preprocessing_status": "complete_quality_screened",
        "aoi": {
            "name": "Gurugram study area",
            "source_crs": "EPSG:4326",
            "geometry_type": "Polygon",
            "geometry_valid": True,
            "geometry_part_count": 1,
            "bounds_wgs84": [77.005, 28.33, 77.11, 28.52],
            "area_km2_authoritative": 216.6173568,
            "area_method": "WGS84 geodesic",
            "area_km2_in_raster_projected_crs": 216.6610663,
        },
        "available_bands": {"B02": {"path": "source.jp2", "resolution_m": [10, 10], "crs": "EPSG:32643", "dtype": "uint16", "nodata": None, "valid_pixels": 100, "saturated_pixels": 0, "surface_reflectance_min": 0.1, "surface_reflectance_max": 0.9}},
        "quality_rasters": {"SCL": {"path": "scl.jp2", "crs": "EPSG:32643", "resolution_m": [20, 20], "dtype": "uint8", "nodata": None, "aoi_min": 2, "aoi_max": 7}},
        "preprocessing": {"target_crs": "EPSG:32643", "target_resolution_m": 10, "continuous_band_resampling": "bilinear", "quality_screening": {"categorical_resampling": "nearest-neighbor"}},
        "pixel_cloud_statistics": {"usable_analysis_percent": 99.9},
        "reflectance_scaling": {"conversion": "formula", "boa_quantification_value": 10000, "boa_add_offsets_by_band_id": {"1": -1000}},
        "quality_threshold_policy": {"scl_classes_masked": [0, 1, 2], "low_cloud_probability": "retain"},
        "reflectance_review": {"source_values_retained": True},
        "warnings": [],
        "quality_observations": [],
    }
    provenance = {
        "scene_id": SCENE_ID,
        "product_id": "product-uuid",
        "product_identifier": PRODUCT_NAME,
        "collection": "sentinel-2-l2a",
        "satellite": "sentinel-2c",
        "acquisition_datetime": "2026-09-10T05:26:41.025Z",
        "tile": "43RFM",
        "processing_level": "L2",
        "cloud_cover": 0.0,
        "assets": {"B02_10m": {"local_path": "data/raw/B02.jp2", "source_href": "https://example.invalid/b02", "file_size_bytes": 4, "sha256": "abc"}},
        "supplemental_quality_assets": {"SCL_20m": {"local_path_from_repo": "data/raw/SCL.jp2", "href": "https://example.invalid/scl", "file_size_bytes": 5, "checksum": "multihash"}},
    }
    path = write_baseline_manifest(report, provenance, aoi_path="data/metadata/aoi_gurugram.geojson", output_dir=tmp_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["quality_status"] == "PASS"
    assert manifest["product_uuid"] == "product-uuid"
    assert manifest["aoi"]["area_km2"] == pytest.approx(216.6173568)
    assert {asset["asset_key"] for asset in manifest["source_assets"]} == {"B02_10m", "SCL_20m"}
    assert manifest["source_assets"][0]["sha256"] == "abc"


def test_quality_mask_writes_cloud_screened_indices_without_replacing_base_outputs(tmp_path):
    scene_dir = tmp_path / PRODUCT_NAME
    scene_dir.mkdir()
    _xml_fixture(scene_dir, include_scl=True)
    aoi = _aoi()
    base_report = inspect_scene(scene_dir, aoi)
    base_report = preprocess_scene(scene_dir, aoi, base_report, output_root=tmp_path / "processed", make_previews=False)
    original_index = Path(base_report["preprocessing"]["index_paths"]["NDVI"])
    original_bytes = original_index.read_bytes()

    result = apply_quality_mask_to_aligned(scene_dir, aoi, base_report, output_root=tmp_path / "processed")
    screening = result["preprocessing"]["quality_screening"]
    assert result["preprocessing_status"] == "complete_quality_screened"
    assert result["pixel_cloud_statistics"]["cloud_percent"] > 0
    assert result["pixel_cloud_statistics"]["usable_analysis_percent"] < 100
    assert result["pixel_cloud_statistics"]["masked_pixel_percent"] == pytest.approx(
        100 - result["pixel_cloud_statistics"]["usable_analysis_percent"]
    )
    assert original_index.read_bytes() == original_bytes
    assert all(Path(path).is_file() for path in screening["cloud_screened_index_paths"].values())
    with rasterio.open(screening["cloud_mask_path"]) as dataset:
        values = dataset.read(1)
        assert 1 in np.unique(values)
        assert 255 in np.unique(values)
    with rasterio.open(screening["cloud_screened_index_paths"]["NDVI"]) as dataset:
        assert np.count_nonzero(dataset.read(1) == NODATA_OUTPUT) > 0
    assert (tmp_path / "processed" / "tiles" / SCENE_ID / "cloud_masked" / "scl_quality_mask.png").is_file()
