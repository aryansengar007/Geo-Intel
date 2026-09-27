from __future__ import annotations

import json
from pathlib import Path

from pyproj import CRS, Geod, Transformer
from shapely.geometry import shape
from shapely.ops import transform as transform_geometry

from backend.app.config import settings


def load_aoi(path: str | Path | dict | None = None) -> dict:
    if isinstance(path, dict):
        payload = path
    else:
        target = Path(path) if path else settings.aoi_path
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError("AOI file must contain a GeoJSON object")

    if payload.get("type") != "FeatureCollection":
        raise ValueError("AOI file must be a GeoJSON FeatureCollection")

    features = payload.get("features") or []
    if not features:
        raise ValueError("AOI file has no features")

    geom = features[0].get("geometry")
    if not geom:
        raise ValueError("AOI feature is missing geometry")

    shapely_geom = shape(geom)
    if shapely_geom.is_empty:
        raise ValueError("AOI geometry is empty")
    if not shapely_geom.is_valid:
        raise ValueError("AOI geometry is invalid")

    return payload


def geodesic_area_km2(geometry: dict, crs_name: str = "EPSG:4326") -> float:
    crs = CRS.from_user_input(crs_name)
    geom = shape(geometry)
    if not crs.is_geographic:
        geodetic_crs = crs.geodetic_crs
        transformer = Transformer.from_crs(crs, geodetic_crs, always_xy=True)
        geom = transform_geometry(transformer.transform, geom)
        crs = geodetic_crs
    ellipsoid = crs.ellipsoid
    geod = Geod(a=ellipsoid.semi_major_metre, rf=ellipsoid.inverse_flattening)
    area, _ = geod.geometry_area_perimeter(geom)
    return abs(float(area)) / 1_000_000


def build_study_area_polygon() -> dict:
    """Create a documented Gurugram study-area polygon when an exact official boundary is unavailable."""
    return {
        "type": "FeatureCollection",
        "name": "gurugram_study_area",
        "crs": {
            "type": "name",
            "properties": {"name": "EPSG:4326"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Gurugram study area",
                    "description": "Approximate urban study area covering Gurugram, Haryana, India; created for project development because a precise official administrative boundary was not fetched at build time.",
                    "source": "study_area_polygon",
                    "source_method": "documented operational polygon for development AOI; not an official administrative boundary",
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [77.0050, 28.3300],
                            [77.1100, 28.3300],
                            [77.1100, 28.5200],
                            [77.0050, 28.5200],
                            [77.0050, 28.3300],
                        ]
                    ],
                },
            }
        ],
    }


def aoi_summary(path: str | Path | dict | None = None) -> dict:
    payload = load_aoi(path)
    feature = payload["features"][0]
    geometry = feature["geometry"]
    geom = shape(geometry)
    bbox = list(geom.bounds)
    crs = payload.get("crs", {"type": "name", "properties": {"name": "EPSG:4326"}})
    crs_name = crs.get("properties", {}).get("name", "EPSG:4326")
    area_km2 = geodesic_area_km2(geometry, crs_name)

    return {
        "name": feature["properties"].get("name", "Gurugram AOI"),
        "description": feature["properties"].get("description", "Geospatial intelligence AOI for Gurugram"),
        "crs": crs,
        "source": feature["properties"].get("source", "study_area_polygon"),
        "source_method": feature["properties"].get("source_method", "documented study-area polygon"),
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "area_km2": round(float(area_km2), 2),
            "area_km2_geodesic": float(area_km2),
            "area_method": "WGS84 ellipsoidal geodesic area using pyproj.Geod",
            "geometry_type": geom.geom_type,
            "geometry_valid": geom.is_valid,
            "geometry_part_count": len(geom.geoms) if hasattr(geom, "geoms") else 1,
        "geometry": geometry,
    }
