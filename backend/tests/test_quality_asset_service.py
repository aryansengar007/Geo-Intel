from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.services.quality_asset_service import QUALITY_ASSET_KEYS, fetch_quality_assets


SCENE_ID = "S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519"
PRODUCT_NAME = f"{SCENE_ID}.SAFE"


class FakeResponse:
    def __init__(self, *, payload=None, body=b"", status_code=200):
        self.payload = payload
        self.body = body
        self.status_code = status_code
        self.headers = {"Content-Type": "application/octet-stream"}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        if self.payload is not None:
            return self.payload
        return {"access_token": "fake-access-token", "refresh_token": "fake-refresh-token", "expires_in": 3600}

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


class FakeSession:
    def __init__(self, item, bodies):
        self.item = item
        self.bodies = bodies
        self.get_urls = []
        self.post_calls = []

    def get(self, url, **kwargs):
        self.get_urls.append(url)
        if "/collections/sentinel-2-l2a/items/" in url:
            return FakeResponse(payload=self.item)
        key = next(key for key, asset in self.item["assets"].items() if asset["alternate"]["https"]["href"] == url)
        return FakeResponse(body=self.bodies[key])

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return FakeResponse(payload={"access_token": "fake-access-token", "refresh_token": "fake-refresh-token", "expires_in": 3600})


def _item_and_bodies():
    bodies = {
        "SCL_20m": b"scl-jp2-fixture",
        "CLD_20m": b"cloud-jp2-fixture",
        "SNW_20m": b"snow-jp2-fixture",
    }
    item = {
        "id": SCENE_ID,
        "collection": "sentinel-2-l2a",
        "properties": {
            "datetime": "2026-09-10T05:26:41.025Z",
            "_private": {"product_uuid": "product-uuid", "product_name": PRODUCT_NAME},
        },
        "assets": {},
    }
    sizes = {"SCL_20m": 20, "CLD_20m": 20, "SNW_20m": 20}
    for key in QUALITY_ASSET_KEYS:
        body = bodies[key]
        digest = hashlib.sha3_256(body).hexdigest()
        filename = {"SCL_20m": "T43RFM_SCL_20m.jp2", "CLD_20m": "MSK_CLDPRB_20m.jp2", "SNW_20m": "MSK_SNWPRB_20m.jp2"}[key]
        folder = "IMG_DATA/R20m" if key == "SCL_20m" else "QI_DATA"
        item["assets"][key] = {
            "alternate": {"https": {"href": f"https://download.dataspace.copernicus.eu/odata/v1/Products(product-uuid)/Nodes({PRODUCT_NAME})/Nodes(GRANULE)/Nodes({folder})/Nodes({filename})/$value"}},
            "file:local_path": f"{PRODUCT_NAME}/GRANULE/L2A_TEST/{folder}/{filename}",
            "file:size": len(body),
            "file:checksum": "1620" + digest,
            "type": "image/jp2",
            "gsd": 20,
            "nodata": 0 if key == "SCL_20m" else None,
            "proj:code": "EPSG:32643",
            "proj:shape": [5490, 5490],
            "title": key,
            "roles": ["data"],
        }
    return item, bodies


def _settings(tmp_path: Path, username="local-user", password="local-password") -> Settings:
    data_root = tmp_path / "data"
    metadata_root = data_root / "metadata"
    safe_dir = data_root / "raw" / "sentinel2" / "2026" / PRODUCT_NAME
    safe_dir.mkdir(parents=True)
    downloads = metadata_root / "downloads"
    downloads.mkdir(parents=True)
    provenance = {
        "scene_id": SCENE_ID,
        "product_id": "product-uuid",
        "product_identifier": PRODUCT_NAME,
        "collection": "sentinel-2-l2a",
        "assets": {"B02_10m": {"file_size_bytes": 100}},
        "file_size_bytes": 100,
    }
    (downloads / f"{SCENE_ID}.provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return Settings(
        _env_file=None,
        data_root=data_root,
        metadata_root=metadata_root,
        aoi_path=metadata_root / "aoi.geojson",
        stac_url="https://stac.dataspace.copernicus.eu/v1",
        cdse_username=username,
        cdse_password=password,
    )


def test_quality_asset_download_reuses_auth_and_records_only_small_assets(tmp_path):
    config = _settings(tmp_path)
    item, bodies = _item_and_bodies()
    session = FakeSession(item, bodies)
    result = fetch_quality_assets(SCENE_ID, config=config, session=session)

    assert {record["key"] for record in result["downloaded"]} == set(QUALITY_ASSET_KEYS)
    assert result["total_bytes_downloaded_this_run"] == sum(map(len, bodies.values()))
    assert all(
        (config.data_root / Path(*Path(record["local_path_from_repo"]).parts[1:])).is_file()
        for record in result["downloaded"]
    )
    assert not any("/Products(product-uuid)/$value" in url for url in session.get_urls)
    assert len(session.post_calls) == 1

    provenance_path = result["provenance_path"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert set(provenance["supplemental_quality_assets"]) == set(QUALITY_ASSET_KEYS)
    assert provenance["assets"]["B02_10m"]["file_size_bytes"] == 100
    assert provenance["file_size_bytes"] == 100 + sum(map(len, bodies.values()))
    serialized = json.dumps(provenance)
    assert "fake-access-token" not in serialized
    assert "local-password" not in serialized


def test_quality_asset_download_refuses_to_replace_mismatched_existing_asset(tmp_path):
    config = _settings(tmp_path)
    item, bodies = _item_and_bodies()
    target = config.data_root / "raw" / "sentinel2" / "2026" / PRODUCT_NAME / "GRANULE" / "L2A_TEST" / "IMG_DATA" / "R20m" / "T43RFM_SCL_20m.jp2"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"user-existing-file")
    with pytest.raises(FileExistsError, match="refusing to replace"):
        fetch_quality_assets(SCENE_ID, asset_keys=("SCL_20m",), config=config, session=FakeSession(item, bodies))
    assert target.read_bytes() == b"user-existing-file"


def test_quality_asset_selection_is_allowlisted(tmp_path):
    config = _settings(tmp_path)
    with pytest.raises(ValueError, match="Unsupported supplemental quality assets"):
        fetch_quality_assets(SCENE_ID, asset_keys=("Product",), config=config, session=FakeSession({}, {}))
