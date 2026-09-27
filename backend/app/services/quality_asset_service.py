from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
from urllib.parse import urlparse

import requests

from backend.app.config import PROJECT_ROOT, Settings, settings
from backend.app.services.sentinel_service import (
    CDSE_DOWNLOAD_HOSTS,
    CDSETokenProvider,
    download_asset_authenticated,
)

QUALITY_ASSET_KEYS = ("SCL_20m", "CLD_20m", "SNW_20m")
STAC_URL = "https://stac.dataspace.copernicus.eu/v1"


def _safe_scene_component(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("Scene identifier contains unsafe path characters.")
    return value


def _read_stac_item(scene_id: str, stac_url: str, session: Any = requests) -> dict[str, Any]:
    url = f"{stac_url.rstrip('/')}/collections/sentinel-2-l2a/items/{scene_id}"
    response = session.get(url, timeout=(10, 45))
    response.raise_for_status()
    item = response.json()
    if item.get("id") != scene_id or item.get("collection") != "sentinel-2-l2a":
        raise ValueError("CDSE STAC response does not match the requested Sentinel-2 L2A item.")
    return item


def _quality_asset_record(item: dict[str, Any], asset_key: str) -> dict[str, Any]:
    asset = item.get("assets", {}).get(asset_key)
    if not asset:
        raise ValueError(f"The STAC item has no {asset_key} asset.")
    href = asset.get("alternate", {}).get("https", {}).get("href")
    if not href:
        raise ValueError(f"The STAC {asset_key} asset has no direct HTTPS download reference.")
    parsed = urlparse(href)
    if parsed.scheme != "https" or parsed.hostname not in CDSE_DOWNLOAD_HOSTS:
        raise ValueError(f"The STAC {asset_key} asset does not use the official CDSE download host.")
    relative = PurePosixPath(str(asset.get("file:local_path", "")).replace("\\", "/"))
    product_name = item.get("properties", {}).get("_private", {}).get("product_name")
    if product_name and relative.parts and relative.parts[0] == product_name:
        relative = PurePosixPath(*relative.parts[1:])
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"The STAC {asset_key} asset has no safe SAFE-relative path.")
    return {
        "key": asset_key,
        "href": href,
        "local_path": relative.as_posix(),
        "size": int(asset["file:size"]) if asset.get("file:size") is not None else None,
        "checksum": asset.get("file:checksum"),
        "type": asset.get("type"),
        "gsd": asset.get("gsd"),
        "nodata": asset.get("nodata"),
        "crs": asset.get("proj:code"),
        "shape": asset.get("proj:shape"),
        "title": asset.get("title"),
        "roles": asset.get("roles", []),
    }


def _checksum_matches(path: Path, stac_checksum: str | None) -> bool:
    if not stac_checksum:
        return False
    encoded = stac_checksum.lower()
    if encoded.startswith("1620") and len(encoded) == 68:
        digest = hashlib.sha3_256()
        expected = encoded[4:]
    elif encoded.startswith("1220") and len(encoded) == 68:
        digest = hashlib.sha256()
        expected = encoded[4:]
    elif len(encoded) == 64:
        digest = hashlib.sha256()
        expected = encoded
    else:
        return False
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fetch_quality_assets(
    scene_id: str,
    *,
    asset_keys: tuple[str, ...] = QUALITY_ASSET_KEYS,
    config: Settings = settings,
    session: Any = requests,
) -> dict[str, Any]:
    scene_id = _safe_scene_component(scene_id)
    unexpected = set(asset_keys) - set(QUALITY_ASSET_KEYS)
    if unexpected:
        raise ValueError(f"Unsupported supplemental quality assets: {', '.join(sorted(unexpected))}")

    item = _read_stac_item(scene_id, config.stac_url, session=session)
    properties = item.get("properties", {})
    private = properties.get("_private", {})

    product_uuid = private.get("product_uuid")
    product_name = private.get("product_name") or scene_id

    provenance_path = (
        config.metadata_root
        / "downloads"
        / f"{scene_id}.provenance.json"
    )

    if not provenance_path.is_file():
        raise FileNotFoundError(
            f"Existing scene provenance was not found: {provenance_path}"
        )

    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )

    provenance_scene_id = provenance.get("scene_id")

    if provenance_scene_id != scene_id:
        raise ValueError(
            "Existing scene provenance does not match the requested scene."
        )

    provenance_product_name = (
        provenance.get("product_name")
        or provenance.get("product_identifier")
        or scene_id
    )

    if provenance_product_name != product_name:
        raise ValueError(
            "STAC SAFE product name does not match the existing "
            "scene provenance."
        )

    local_product_path = provenance.get("local_product_path")
    if not local_product_path:
        raise ValueError(
            "Existing scene provenance has no local product path."
        )

    product_dir = Path(local_product_path).resolve()

    if not product_dir.is_dir():
        raise FileNotFoundError(
            f"The existing SAFE product directory was not found: "
            f"{product_dir}"
        )

    selected_assets = [_quality_asset_record(item, key) for key in asset_keys]
    if not product_dir.is_dir():
        raise FileNotFoundError(f"The existing SAFE product directory was not found: {product_dir}")

    missing_assets = []
    skipped_assets = []
    for asset in selected_assets:
        destination = (product_dir / Path(*PurePosixPath(asset["local_path"]).parts)).resolve()
        if not destination.is_relative_to(product_dir.resolve()):
            raise ValueError(f"STAC {asset['key']} path escapes the existing SAFE product.")
        asset["destination"] = destination
        if destination.exists():
            if destination.stat().st_size != asset["size"] or not _checksum_matches(destination, asset["checksum"]):
                raise FileExistsError(f"Existing {asset['key']} file does not match STAC size/checksum; refusing to replace it: {destination}")
            skipped_assets.append(asset)
        else:
            missing_assets.append(asset)

    if missing_assets and (not config.cdse_username.strip() or not config.cdse_password):
        raise RuntimeError("CDSE_USERNAME and CDSE_PASSWORD must be configured in the existing local .env to fetch missing quality assets.")
    token_provider = CDSETokenProvider(config.cdse_username, config.cdse_password, session=session) if missing_assets else None
    if token_provider:
        token_provider.get_token()

    now = datetime.now(timezone.utc).isoformat()
    downloaded_assets: list[dict[str, Any]] = []
    for asset in missing_assets:
        path = download_asset_authenticated(
            asset["href"],
            asset["destination"],
            token_provider,
            expected_size=asset["size"],
            expected_checksum=asset["checksum"],
            session=session,
        )
        downloaded_assets.append({
            **{key: asset[key] for key in ("key", "href", "local_path", "size", "checksum", "type", "gsd", "nodata", "crs", "shape", "title", "roles")},
            "downloaded_at": now,
            "file_size_bytes": path.stat().st_size,
            "local_path_from_repo": str(path.resolve()),
        })

    existing = provenance.setdefault("supplemental_quality_assets", {})
    for asset in skipped_assets:
        existing[asset["key"]] = {
            **{key: asset[key] for key in ("key", "href", "local_path", "size", "checksum", "type", "gsd", "nodata", "crs", "shape", "title", "roles")},
            "downloaded_at": existing.get(asset["key"], {}).get("downloaded_at", now),
            "file_size_bytes": asset["destination"].stat().st_size,
            "local_path_from_repo": str(asset["destination"].resolve()),
        }
    for asset in downloaded_assets:
        existing[asset["key"]] = asset
    provenance["supplemental_quality_assets_updated_at"] = now
    provenance["file_size_bytes"] = sum(
        int(record.get("file_size_bytes", 0))
        for record in provenance.get("assets", {}).values()
    ) + sum(int(record.get("file_size_bytes", 0)) for record in existing.values())
    _write_json(provenance_path, provenance)

    return {
        "scene_id": scene_id,
        "product_id": product_uuid,
        "product_identifier": product_name,
        "downloaded": downloaded_assets,
        "skipped_existing_verified": [asset["key"] for asset in skipped_assets],
        "total_bytes_downloaded_this_run": sum(asset["file_size_bytes"] for asset in downloaded_assets),
        "provenance_path": provenance_path,
    }
