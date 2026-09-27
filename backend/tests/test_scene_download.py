from __future__ import annotations

import hashlib
import logging

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform

from backend.app.services.scene_validation import validate_scene_assets
from backend.app.services.sentinel_service import (
    CDSEAuthenticationError,
    CDSETokenProvider,
    METADATA_ASSET_KEYS,
    SELECTED_ASSET_KEYS,
    download_asset_authenticated,
    scene_from_stac_item,
    select_best_scene,
)


class FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200, content_type: str = "application/octet-stream"):
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int):
        yield self.content

    def close(self):
        pass

    def json(self):
        return {"access_token": "test-access-token", "refresh_token": "test-refresh-token", "expires_in": 3600}


class FakeSession:
    def __init__(self, response: FakeResponse | None = None):
        self.response = response or FakeResponse(b"asset-bytes")
        self.post_calls = []
        self.get_calls = []

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return FakeResponse(b"{}")

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.response


class ReadyToken:
    def get_token(self, force_refresh: bool = False):
        return "test-token"


def _stac_item(scene_id: str, cloud_cover: float = 5.0) -> dict:
    assets = {}
    product_name = f"{scene_id}.SAFE"
    for key in (*SELECTED_ASSET_KEYS, *METADATA_ASSET_KEYS):
        assets[key] = {
            "href": f"https://download.dataspace.copernicus.eu/odata/v1/asset/{key}",
            "file:local_path": f"{product_name}/GRANULE/test/IMG_DATA/{key}.jp2",
            "file:size": 5,
            "type": "image/jp2",
            "gsd": SELECTED_ASSET_KEYS.get(key, (None, None))[1],
            "proj:code": "EPSG:32643",
        }
    return {
        "id": scene_id,
        "collection": "sentinel-2-l2a",
        "links": [{"rel": "self", "href": f"https://stac.dataspace.copernicus.eu/v1/items/{scene_id}"}],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[77.0, 28.3], [77.2, 28.3], [77.2, 28.6], [77.0, 28.6], [77.0, 28.3]]],
        },
        "properties": {
            "datetime": "2026-09-10T05:26:41.025Z",
            "eo:cloud_cover": cloud_cover,
            "platform": "sentinel-2c",
            "grid:code": "MGRS-43RFM",
            "processing:level": "L2",
            "product:type": "S2MSI2A",
            "instruments": ["msi"],
            "_private": {"product_uuid": "unit-product-id", "product_name": f"{scene_id}.SAFE"},
        },
        "assets": assets,
    }


def _aoi() -> dict:
    return {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"name": "Gurugram study area"}, "geometry": {
            "type": "Polygon",
            "coordinates": [[[77.005, 28.33], [77.11, 28.33], [77.11, 28.52], [77.005, 28.52], [77.005, 28.33]]],
        }}],
    }


def test_missing_cdse_credentials_has_clear_error():
    provider = CDSETokenProvider("", "", session=FakeSession())
    with pytest.raises(CDSEAuthenticationError, match="CDSE_USERNAME and CDSE_PASSWORD"):
        provider.get_token()


def test_authentication_does_not_log_credentials_or_tokens(caplog):
    session = FakeSession()
    provider = CDSETokenProvider("private-user", "private-password", session=session)
    with caplog.at_level(logging.DEBUG):
        assert provider.get_token() == "test-access-token"
    assert session.post_calls[0][1]["data"] == {
        "client_id": "cdse-public",
        "grant_type": "password",
        "username": "private-user",
        "password": "private-password",
    }
    assert "private-password" not in caplog.text
    assert "test-access-token" not in caplog.text
    assert "test-refresh-token" not in caplog.text


def test_expired_access_token_uses_refresh_token(monkeypatch):
    session = FakeSession()
    provider = CDSETokenProvider("private-user", "private-password", session=session)
    ticks = iter([100.0, 200.0, 500.0])
    monkeypatch.setattr("backend.app.services.sentinel_service.time.monotonic", lambda: next(ticks))
    monkeypatch.setattr(
        FakeResponse,
        "json",
        lambda self: {"access_token": "test-access-token", "refresh_token": "test-refresh-token", "expires_in": 120},
    )
    assert provider.get_token() == "test-access-token"
    assert provider.get_token() == "test-access-token"
    assert session.post_calls[1][1]["data"]["grant_type"] == "refresh_token"
    assert "password" not in session.post_calls[1][1]["data"]


def test_stac_serialization_and_selection_require_real_l2a_metadata():
    aoi = _aoi()
    low_cloud = scene_from_stac_item(_stac_item("low-cloud", 0.0))
    higher_cloud = scene_from_stac_item(_stac_item("higher-cloud", 10.0))
    selected = select_best_scene([higher_cloud, low_cloud], aoi)
    assert selected["scene_id"] == "low-cloud"
    assert selected["collection"] == "sentinel-2-l2a"
    assert selected["tile"] == "43RFM"
    assert selected["product_id"] == "unit-product-id"
    assert selected["product_identifier"] == "low-cloud.SAFE"
    assert "B02_10m" in selected["assets"]

    wrong_collection = dict(low_cloud, collection="sentinel-2-l1c")
    with pytest.raises(ValueError, match="No complete Sentinel-2 L2A"):
        select_best_scene([wrong_collection], aoi)


def test_successful_asset_response_is_streamed_and_checked(tmp_path):
    body = b"real-product-asset"
    session = FakeSession(FakeResponse(body))
    destination = tmp_path / "band.jp2"
    result = download_asset_authenticated(
        "https://download.dataspace.copernicus.eu/odata/v1/asset/B02",
        destination,
        ReadyToken(),
        expected_size=len(body),
        expected_checksum=hashlib.sha256(body).hexdigest(),
        session=session,
    )
    assert result.read_bytes() == body
    assert session.get_calls[0][1]["headers"]["Authorization"] == "Bearer test-token"


def test_invalid_html_download_is_rejected(tmp_path):
    session = FakeSession(FakeResponse(b"<html>login</html>", content_type="text/html"))
    destination = tmp_path / "band.jp2"
    with pytest.raises(ValueError, match="HTML page"):
        download_asset_authenticated(
            "https://download.dataspace.copernicus.eu/odata/v1/asset/B02",
            destination,
            ReadyToken(),
            session=session,
        )
    assert not destination.exists()


def test_product_validation_checks_bands_projection_dimensions_and_aoi_overlap(tmp_path):
    scene_id = "S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519"
    product_name = f"{scene_id}.SAFE"
    center_x, center_y = transform("EPSG:4326", "EPSG:32643", [77.05], [28.42])
    downloaded = {}
    for key, (band, resolution) in SELECTED_ASSET_KEYS.items():
        path = tmp_path / f"{band}.jp2"
        width = height = 64
        origin_x = center_x[0] - width * resolution / 2
        origin_y = center_y[0] + height * resolution / 2
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="uint16",
            crs="EPSG:32643",
            transform=from_origin(origin_x, origin_y, resolution, resolution),
            nodata=0,
        ) as raster:
            raster.write(np.ones((height, width), dtype=np.uint16) * 42, 1)
        downloaded[key] = path

    metadata_files = {
        "safe_manifest": "<XFDU><informationPackageMap>SAFE</informationPackageMap></XFDU>",
        "product_metadata": (
            f"<Product><PRODUCT_URI>{product_name}</PRODUCT_URI>"
            "<PRODUCT_START_TIME>2026-09-10T05:26:41.025Z</PRODUCT_START_TIME></Product>"
        ),
        "granule_metadata": "<Tile><TILE_ID>43RFM</TILE_ID></Tile>",
    }
    for key, content in metadata_files.items():
        path = tmp_path / f"{key}.xml"
        path.write_text(content, encoding="utf-8")
        downloaded[key] = path

    scene = {
        "scene_id": scene_id,
        "collection": "sentinel-2-l2a",
        "product_name": product_name,
        "product_identifier": product_name,
        "processing_level": "L2",
        "acquisition_date": "2026-09-10T05:26:41.025Z",
    }
    report = validate_scene_assets(scene, downloaded, _aoi())
    assert report["status"] == "PASS"
    assert set(report["bands"]) == {details[0] for details in SELECTED_ASSET_KEYS.values()}
    assert report["bands"]["B02"]["width"] == 64
    assert report["bands"]["B11"]["resolution_m"] == [20.0, 20.0]
    assert report["bands"]["B02"]["aoi_overlap"] is True

    del downloaded["B02_10m"]
    with pytest.raises(ValueError, match="missing required assets"):
        validate_scene_assets(scene, downloaded, _aoi())

    downloaded["B02_10m"] = tmp_path / "invalid.jp2"
    downloaded["B02_10m"].write_text("<html>not a JP2</html>", encoding="utf-8")
    with pytest.raises(ValueError, match="not a readable raster"):
        validate_scene_assets(scene, downloaded, _aoi())
