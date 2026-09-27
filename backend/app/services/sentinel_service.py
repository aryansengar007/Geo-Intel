from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import urlparse

import requests
from pystac_client import Client
from shapely.geometry import shape

from backend.app.config import settings


CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_DOWNLOAD_HOSTS = {
    "download.dataspace.copernicus.eu",
    "zipper.dataspace.copernicus.eu",
}
SELECTED_ASSET_KEYS = {
    "B02_10m": ("B02", 10),
    "B03_10m": ("B03", 10),
    "B04_10m": ("B04", 10),
    "B08_10m": ("B08", 10),
    "B05_20m": ("B05", 20),
    "B06_20m": ("B06", 20),
    "B07_20m": ("B07", 20),
    "B11_20m": ("B11", 20),
    "B12_20m": ("B12", 20),
}
METADATA_ASSET_KEYS = ("safe_manifest", "product_metadata", "granule_metadata")


class CDSEAuthenticationError(RuntimeError):
    """Authentication errors that never include credentials or token values."""


class CDSETokenProvider:
    def __init__(self, username: str, password: str, session: Any = requests):
        self._username = username.strip()
        self._password = password
        self._session = session
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at = 0.0

    def get_token(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._access_token and time.monotonic() < self._expires_at - 60:
            return self._access_token

        if self._refresh_token:
            try:
                return self._request_token({
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                })
            except CDSEAuthenticationError:
                self._refresh_token = None

        if not self._username or not self._password:
            raise CDSEAuthenticationError(
                "CDSE credentials are missing. Set CDSE_USERNAME and CDSE_PASSWORD in the local .env file."
            )
        return self._request_token({
            "grant_type": "password",
            "username": self._username,
            "password": self._password,
        })

    def _request_token(self, credentials: dict[str, str]) -> str:
        try:
            response = self._session.post(
                settings.cdse_token_url or CDSE_TOKEN_URL,
                data={"client_id": "cdse-public", **credentials},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=(10, 30),
            )
            if response.status_code != 200:
                raise CDSEAuthenticationError(
                    f"CDSE authentication failed (HTTP {response.status_code}); check the local credentials and account access."
                )
            payload = response.json()
            access_token = payload.get("access_token")
            if not isinstance(access_token, str) or not access_token:
                raise CDSEAuthenticationError("CDSE authentication response did not contain an access token.")
        except CDSEAuthenticationError:
            raise
        except (requests.RequestException, ValueError, TypeError) as error:
            raise CDSEAuthenticationError("Could not obtain a CDSE access token; check network access and account credentials.") from error

        self._access_token = access_token
        self._refresh_token = payload.get("refresh_token")
        self._expires_at = time.monotonic() + max(0, int(payload.get("expires_in", 300)))
        return access_token


def scene_from_stac_item(item: dict[str, Any]) -> dict[str, Any]:
    properties = item.get("properties") or {}
    private = properties.get("_private") or {}
    product_name = private.get("product_name") or properties.get("product:name")
    assets: dict[str, dict[str, Any]] = {}
    for key in (*SELECTED_ASSET_KEYS, *METADATA_ASSET_KEYS):
        asset = (item.get("assets") or {}).get(key)
        if not asset:
            continue
        href = (asset.get("alternate") or {}).get("https", {}).get("href") or asset.get("href")
        if not href:
            continue
        assets[key] = {
            "href": href,
            "local_path": asset.get("file:local_path"),
            "size": asset.get("file:size"),
            "checksum": asset.get("file:checksum"),
            "type": asset.get("type"),
            "gsd": asset.get("gsd"),
            "crs": asset.get("proj:code"),
        }

    links = item.get("links") or []
    self_link = next((link["href"] for link in links if link.get("rel") == "self"), None)
    tile = properties.get("grid:code")
    if isinstance(tile, str) and tile.startswith("MGRS-"):
        tile = tile.removeprefix("MGRS-")
    return {
        "scene_id": item.get("id"),
        "collection": item.get("collection"),
        "product_id": private.get("product_uuid") or properties.get("s2:product_id"),
        "product_identifier": product_name,
        "product_name": product_name,
        "platform": properties.get("platform"),
        "satellite": properties.get("platform"),
        "sensor": (properties.get("instruments") or [None])[0],
        "tile": tile,
        "acquisition_date": properties.get("datetime"),
        "processing_level": properties.get("processing:level"),
        "product_type": properties.get("product:type"),
        "cloud_cover": properties.get("eo:cloud_cover"),
        "bbox": item.get("bbox"),
        "geometry": item.get("geometry"),
        "source": "Copernicus Data Space STAC",
        "source_url": self_link,
        "assets": assets,
        "download_status": "discovered",
    }


def select_best_scene(scenes: list[dict[str, Any]], aoi_geojson: dict[str, Any]) -> dict[str, Any]:
    aoi = shape(aoi_geojson["features"][0]["geometry"])
    candidates = []
    for scene in scenes:
        if scene.get("collection") != "sentinel-2-l2a":
            continue
        if str(scene.get("processing_level", "")).upper() not in {"L2", "L2A"}:
            continue
        if not scene.get("product_id") or not scene.get("product_identifier"):
            continue
        if not all(key in scene.get("assets", {}) for key in (*SELECTED_ASSET_KEYS, *METADATA_ASSET_KEYS)):
            continue
        geometry = scene.get("geometry")
        if not geometry or not shape(geometry).intersects(aoi):
            continue
        cloud = scene.get("cloud_cover")
        acquisition = scene.get("acquisition_date") or ""
        candidates.append((float(cloud) if cloud is not None else 100.0, acquisition, scene))
    if not candidates:
        raise ValueError("No complete Sentinel-2 L2A product with required assets overlaps the AOI.")
    return min(candidates, key=lambda candidate: (candidate[0], -_datetime_sort_value(candidate[1])))[2]


def _datetime_sort_value(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError, OverflowError):
        return 0.0


def download_asset_authenticated(
    url: str,
    output_path: str | Path,
    token_provider: CDSETokenProvider,
    *,
    expected_size: int | None = None,
    expected_checksum: str | None = None,
    session: Any = requests,
) -> Path:
    parsed = urlparse(url)

    if parsed.scheme != "https" or parsed.hostname not in CDSE_DOWNLOAD_HOSTS:
        raise ValueError(
            "Refusing download URL outside the official CDSE download host."
        )

    target = Path(output_path)

    for attempt in range(3):
        temporary_path: Path | None = None
        response = None

        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = session.get(
                url,
                headers={
                    "Authorization": (
                        f"Bearer "
                        f"{token_provider.get_token(force_refresh=attempt > 0)}"
                    )
                },
                stream=True,
                allow_redirects=True,
                timeout=(20, 120),
            )

            if response.status_code in (401, 403):
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue

                raise CDSEAuthenticationError(
                    "CDSE rejected the access token while downloading an asset."
                )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "text/html" in content_type:
                raise ValueError(
                    "CDSE returned an HTML page instead of the requested "
                    "product asset."
                )

            hasher, expected_digest = _checksum_hasher(
                expected_checksum
            )

            bytes_written = 0
            prefix = bytearray()
            
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".part",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if not chunk:
                        continue

                    if len(prefix) < 32:
                        prefix.extend(
                            chunk[: 32 - len(prefix)]
                        )

                    if (
                        bytes_written == 0
                        and _looks_like_error_document(prefix)
                    ):
                        raise ValueError(
                            "CDSE returned an error document instead "
                            "of a product asset."
                        )

                    handle.write(chunk)
                    hasher.update(chunk)
                    bytes_written += len(chunk)

            if bytes_written == 0:
                raise ValueError(
                    "CDSE returned an empty product asset."
                )

            if (
                expected_size is not None
                and bytes_written != int(expected_size)
            ):
                raise ValueError(
                    "Downloaded asset size mismatch: "
                    f"expected {expected_size} bytes, "
                    f"received {bytes_written}."
                )

            if (
                expected_digest
                and hasher.hexdigest().lower() != expected_digest
            ):
                raise ValueError(
                    "Downloaded asset checksum does not match "
                    "the STAC checksum."
                )

            os.replace(temporary_path, target)
            temporary_path = None

            return target

        except requests.RequestException:
            if attempt == 2:
                raise

            time.sleep(2 * (attempt + 1))

        finally:
            if response is not None:
                response.close()

            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
                
def _looks_like_error_document(prefix: bytes) -> bool:
    sample = prefix.lstrip().lower()
    return sample.startswith((b"<!doctype html", b"<html", b"{\"error", b"{\"message")) or sample.startswith(
        (b"<error", b"<exceptionreport")
    )


def _checksum_hasher(checksum: str | None) -> tuple[Any, str | None]:
    value = (checksum or "").removeprefix("sha256:").strip().lower()
    if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
        return hashlib.sha256(), value
    if len(value) == 68 and value.startswith("1620"):
        digest = value[4:]
        if all(character in "0123456789abcdef" for character in digest):
            return hashlib.sha3_256(), digest
    if len(value) == 68 and value.startswith("1220"):
        digest = value[4:]
        if all(character in "0123456789abcdef" for character in digest):
            return hashlib.sha256(), digest
    return hashlib.sha256(), None


def build_search_query(aoi_bbox: list[float], start_year: int, end_year: int, cloud_max: float = 30.0):
    start = datetime(start_year, 1, 1, tzinfo=None)
    end = datetime(end_year, 12, 31, 23, 59, 59)
    return {
        "bbox": aoi_bbox,
        "collections": ["sentinel-2-l2a"],
        "datetime": f"{start.isoformat()}Z/{end.isoformat()}Z",
        "query": {"eo:cloud_cover": {"lt": cloud_max}},
        "max_items": 20,
    }


def discover_sentinel2_scenes(
    aoi_bbox: list[float],
    start_year: int | None = None,
    end_year: int | None = None,
    max_cloud_cover: float | None = None,
    max_scenes: int | None = None,
) -> list[dict]:
    start_year = start_year or settings.default_start_year
    end_year = end_year or settings.default_end_year
    max_cloud_cover = (
        max_cloud_cover
        if max_cloud_cover is not None
        else settings.default_cloud_max
    )
    max_scenes = (
        max_scenes
        if max_scenes is not None
        else settings.max_default_scenes
    )

    client = Client.open(settings.stac_url)

    candidates: list[dict] = []

    for year in range(start_year, end_year + 1):
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=aoi_bbox,
            datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z",
            max_items=max_scenes * 10,
        )

        for item in search.items():
            cloud_cover = item.properties.get("eo:cloud_cover")

            if cloud_cover is None or float(cloud_cover) > max_cloud_cover:
                continue

            scene = scene_from_stac_item(item.to_dict())
            scene.update(
                {
                    "scene_id": item.id,
                    "tile_id": scene.get("tile"),
                    "acquisition_date": (
                        item.datetime.isoformat()
                        if item.datetime
                        else None
                    ),
                    "cloud_cover": float(cloud_cover),
                    "bbox": item.bbox,
                    "geometry": item.geometry,
                    "crs": item.properties.get("proj:epsg", "EPSG:4326"),
                    "resolution": {
                        "B02": 10,
                        "B03": 10,
                        "B04": 10,
                        "B08": 10,
                        "B05": 20,
                        "B06": 20,
                        "B07": 20,
                        "B11": 20,
                        "B12": 20,
                    },
                    "available_bands": [
                        "B02",
                        "B03",
                        "B04",
                        "B05",
                        "B06",
                        "B07",
                        "B08",
                        "B11",
                        "B12",
                    ],
                    "processing_version": item.properties.get(
                        "processing:version"
                    ),
                }
            )

            candidates.append(scene)

            unique_candidates = {
        scene["scene_id"]: scene
        for scene in candidates
    }

    candidates = list(unique_candidates.values())

    years = list(range(start_year, end_year + 1))

    # Find the tile with the widest temporal coverage.
    tile_years: dict[str, set[int]] = {}

    for scene in candidates:
        tile = scene.get("tile_id")
        acquisition_date = scene.get("acquisition_date")

        if not tile or not acquisition_date:
            continue

        year = int(acquisition_date[:4])
        tile_years.setdefault(tile, set()).add(year)

    if tile_years:
        preferred_tile = max(
            tile_years,
            key=lambda tile: (
                len(tile_years[tile]),
                tile,
            ),
        )
    else:
        preferred_tile = None

    # Prefer the common tile for the temporal benchmark.
    if preferred_tile:
        preferred_candidates = [
            scene
            for scene in candidates
            if scene.get("tile_id") == preferred_tile
        ]
    else:
        preferred_candidates = candidates

    selected: list[dict] = []

    # Select the clearest scene from the preferred tile for each year.
    for year in years:
        year_scenes = [
            scene
            for scene in preferred_candidates
            if (scene.get("acquisition_date") or "")[:4] == str(year)
        ]

        if year_scenes:
            best_scene = min(
                year_scenes,
                key=lambda scene: scene["cloud_cover"],
            )
            selected.append(best_scene)

    # If some years are unavailable on the preferred tile,
    # fill remaining slots with the clearest available scenes.
    if len(selected) < max_scenes:
        selected_ids = {scene["scene_id"] for scene in selected}

        remaining = [
            scene
            for scene in candidates
            if scene["scene_id"] not in selected_ids
        ]

        remaining.sort(
            key=lambda scene: (
                scene["cloud_cover"],
                scene.get("acquisition_date") or "",
            )
        )

        selected.extend(
            remaining[: max_scenes - len(selected)]
        )

    selected.sort(
        key=lambda scene: scene.get("acquisition_date") or ""
    )

    return selected[:max_scenes]


def detect_download_auth_requirement() -> str:
    return "Copernicus Data Space access may require a valid user account or token; if asset URLs require authentication, log in via the official Copernicus Data Space portal and configure credentials in .env before downloading assets."


def download_scene_asset(url: str, output_path: str | Path, token: str | None = None) -> Path:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(url, headers=headers, timeout=60, stream=True)
    if response.status_code in (401, 403):
        raise PermissionError(detect_download_auth_requirement())
    response.raise_for_status()

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    return target
