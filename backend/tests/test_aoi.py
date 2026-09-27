import pytest
from shapely.geometry import box, mapping

from backend.app.services.aoi_service import aoi_summary, build_study_area_polygon, geodesic_area_km2, load_aoi


def test_build_study_area_polygon_has_valid_geojson():
    payload = build_study_area_polygon()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 1
    assert payload["features"][0]["geometry"]["type"] == "Polygon"


def test_aoi_summary_includes_expected_fields():
    payload = build_study_area_polygon()
    summary = aoi_summary({
        "type": "FeatureCollection",
        "features": payload["features"],
    })
    assert summary["name"] == "Gurugram study area"
    assert summary["bbox"][0] <= summary["bbox"][2]
    assert summary["area_km2"] > 0
    assert summary["area_method"] == "WGS84 ellipsoidal geodesic area using pyproj.Geod"


def test_actual_gurugram_aoi_has_authoritative_geodesic_area():
    payload = load_aoi("data/metadata/aoi_gurugram.geojson")
    geometry = payload["features"][0]["geometry"]
    summary = aoi_summary(payload)
    assert summary["area_km2"] == pytest.approx(216.62, abs=0.01)
    assert geodesic_area_km2(geometry, "EPSG:4326") == pytest.approx(216.6173568, abs=0.001)
    assert geometry["type"] == "Polygon"


def test_geodesic_area_transforms_projected_geometry_before_measuring():
    polygon_utm = box(500000, 3100000, 501000, 3101000)
    area = geodesic_area_km2(mapping(polygon_utm), "EPSG:32643")
    assert area == pytest.approx(1.0, rel=0.002)


def test_load_aoi_rejects_invalid_polygon_geometry():
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0, 0], [1, 1], [1, 0], [0, 1], [0, 0]]],
            },
        }],
    }
    with pytest.raises(ValueError, match="invalid"):
        load_aoi(payload)
