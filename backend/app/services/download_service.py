from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable

import requests

from backend.app.config import PROJECT_ROOT, Settings, settings
from backend.app.services.scene_validation import validate_scene_assets
from backend.app.services.sentinel_service import (
    CDSEAuthenticationError,
    CDSETokenProvider,
    METADATA_ASSET_KEYS,
    SELECTED_ASSET_KEYS,
    download_asset_authenticated,
)


def _safe_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("STAC scene or product identifier contains unsafe path characters.")
    return value


def _asset_path(product_dir: Path, scene: dict[str, Any], asset: dict[str, Any]) -> Path:
    raw_path = asset.get("local_path")
    if not raw_path:
        raise ValueError("STAC asset has no product-relative path.")
    relative = PurePosixPath(str(raw_path).replace("\\", "/"))
    product_name = scene.get("product_name") or scene["scene_id"]
    if relative.parts and relative.parts[0] == product_name:
        relative = PurePosixPath(*relative.parts[1:])
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("STAC asset path is not a safe product-relative path.")
    target = (product_dir / Path(*relative.parts)).resolve()
    if not target.is_relative_to(product_dir.resolve()):
        raise ValueError("STAC asset path escapes the selected product directory.")
    return target


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_download_state(scene: dict[str, Any], state: str, message: str, config: Settings = settings) -> Path:
    scene_id = _safe_component(str(scene["scene_id"]))
    path = config.metadata_root / "downloads" / f"{scene_id}.status.json"
    _write_json(path, {
        "scene_id": scene_id,
        "status": state,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_validate_scene(
    scene: dict[str, Any],
    aoi_geojson: dict[str, Any],
    *,
    config: Settings = settings,
    session: Any = requests,
    on_status: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    def set_status(state: str, message: str) -> None:
        write_download_state(scene, state, message, config)
        if on_status:
            on_status(state, message)

    scene_id = _safe_component(str(scene["scene_id"]))
    product_name = _safe_component(
        str(scene.get("product_name") or scene["scene_id"])
    )
    acquisition = datetime.fromisoformat(
        scene["acquisition_date"].replace("Z", "+00:00")
    )
    
    year = acquisition.year
    product_dir = (
        Path("D:/GIData")
        / "sentinel2"
        / str(year)
        / product_name
    ).resolve()
    
    raw_root = (
        Path("D:/GIData")
        / "sentinel2"
    ).resolve()

    if not product_dir.is_relative_to(raw_root):
        
        raise ValueError("Selected product path is outside the configured Sentinel-2 data directory.")

    try:
        set_status("authenticating", "Requesting a CDSE access token.")
        token_provider = CDSETokenProvider(config.cdse_username, config.cdse_password, session=session)
        token_provider.get_token()
    except CDSEAuthenticationError as error:
        status = "credentials_missing" if not config.cdse_username.strip() or not config.cdse_password else "failed"
        set_status(status, str(error))
        raise

    try:
        required_assets = (*SELECTED_ASSET_KEYS, *METADATA_ASSET_KEYS)
        missing = [key for key in required_assets if key not in scene.get("assets", {})]
        if missing:
            raise ValueError(f"Selected STAC item is missing required assets: {', '.join(missing)}")
        set_status("downloading", f"Downloading {len(required_assets)} selected bands and metadata assets.")
        downloaded: dict[str, Path] = {}
        for asset_key in required_assets:
            asset = scene["assets"][asset_key]
            destination = _asset_path(product_dir, scene, asset)
            downloaded[asset_key] = download_asset_authenticated(
                asset["href"],
                destination,
                token_provider,
                expected_size=asset.get("size"),
                expected_checksum=asset.get("checksum"),
                session=session,
            )
        set_status("downloaded", "All selected JP2 bands and SAFE metadata assets were downloaded.")
        set_status("validating", "Reading product metadata and raster headers; checking AOI overlap.")
        validation = validate_scene_assets(scene, downloaded, aoi_geojson)

        downloaded_at = datetime.now(timezone.utc).isoformat()
        asset_records = {}
        for key, path in downloaded.items():
            try:
                local_path = path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
            except ValueError:
                local_path = os.path.relpath(path.resolve(), PROJECT_ROOT.resolve()).replace(os.sep, "/")
            asset_records[key] = {
                "local_path": local_path,
                "file_size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "source_href": scene["assets"][key]["href"],
            }
        total_size = sum(asset["file_size_bytes"] for asset in asset_records.values())
        provenance = {
            "scene_id": scene_id,
            "product_id": scene["product_id"],
            "product_identifier": scene["product_identifier"],
            "collection": scene["collection"],
            "satellite": scene.get("satellite"),
            "tile": scene.get("tile"),
            "acquisition_datetime": scene["acquisition_date"],
            "cloud_cover": scene.get("cloud_cover"),
            "processing_level": scene.get("processing_level"),
            "aoi": {
                "name": aoi_geojson["features"][0].get("properties", {}).get("name"),
                "source": aoi_geojson["features"][0].get("properties", {}).get("source"),
                "geometry": aoi_geojson["features"][0]["geometry"],
            },
            "stac_item_url": scene["source_url"],
            "download_method": "Authenticated CDSE OData asset URLs resolved from STAC; selected JP2 bands and SAFE metadata only.",
            "source_api": "https://download.dataspace.copernicus.eu/odata/v1/",
            "download_timestamp": downloaded_at,
            "local_product_path": str(product_dir.resolve()),
            "original_product_name": scene["product_name"],
            "file_size_bytes": total_size,
            "selected_bands": list(SELECTED_ASSET_KEYS),
            "assets": asset_records,
            "validation_status": "validated",
            "validation": validation,
        }
        provenance_path = config.metadata_root / "downloads" / f"{scene_id}.provenance.json"
        _write_json(provenance_path, provenance)
        set_status("validated", "Selected bands and Sentinel-2 metadata passed validation.")
        return {"provenance": provenance, "provenance_path": provenance_path, "validation": validation}
    except Exception as error:
        set_status("failed", f"{type(error).__name__}: {error}")
        raise


def local_dataset_status(config: Settings = settings) -> dict[str, Any]:
    downloads_dir = config.metadata_root / "downloads"
    provenance_files = list(downloads_dir.glob("*.provenance.json")) if downloads_dir.is_dir() else []
    validated_scenes = []
    downloaded_scenes = []
    for path in provenance_files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        assets = record.get("assets", {})
        files_exist = bool(assets) and all((PROJECT_ROOT / asset.get("local_path", "")).is_file() for asset in assets.values())
        if files_exist:
            downloaded_scenes.append(record.get("scene_id"))
            if record.get("validation_status") == "validated":
                validated_scenes.append(record.get("scene_id"))

    state_files = list(downloads_dir.glob("*.status.json")) if downloads_dir.is_dir() else []
    latest_state = None
    if state_files:
        try:
            latest_state = json.loads(max(state_files, key=lambda path: path.stat().st_mtime).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            latest_state = None

    discovered_count = 0
    discovery_path = config.metadata_root / "scenes_discovery.json"
    if discovery_path.is_file():
        try:
            discovered_count = len(json.loads(discovery_path.read_text(encoding="utf-8")).get("scenes", []))
        except (OSError, json.JSONDecodeError):
            pass

    if validated_scenes:
        status = "validated"
        message = f"{len(validated_scenes)} locally stored Sentinel-2 scene(s) passed validation."
    elif latest_state and latest_state.get("status") in {"authenticating", "downloading", "downloaded", "validating", "failed"}:
        status = latest_state["status"]
        message = latest_state.get("message", "A scene download is in progress.")
    elif discovered_count and (not config.cdse_username.strip() or not config.cdse_password):
        status = "credentials_missing"
        message = "Scenes are discoverable, but CDSE_USERNAME and CDSE_PASSWORD are not configured in the local .env file."
    elif downloaded_scenes:
        status = "downloaded"
        message = "Scene files exist locally but have not passed product validation."
    elif discovered_count:
        status = "discovered"
        message = f"{discovered_count} scene(s) are recorded as discovered; none are downloaded locally."
    else:
        status = "empty"
        message = "No Sentinel-2 discovery metadata or downloaded products are present locally."

    return {
        "status": status,
        "discovered_count": discovered_count,
        "downloaded_count": len(downloaded_scenes),
        "validated_count": len(validated_scenes),
        "failed_count": int(bool(latest_state and latest_state.get("status") == "failed")),
        "real_data_count": len(validated_scenes),
        "current_scene": latest_state.get("scene_id") if latest_state else None,
        "message": message,
    }
