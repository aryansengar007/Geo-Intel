from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "GeoIntel"
    api_v1_str: str = "/api"
    data_root: Path = Path(__file__).resolve().parents[2] / "data"
    metadata_root: Path = data_root / "metadata"
    aoi_path: Path = metadata_root / "aoi_gurugram.geojson"
    stac_url: str = "https://stac.dataspace.copernicus.eu/v1"
    cdse_username: str = ""
    cdse_password: str = ""
    cdse_token_url: str = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    default_cloud_max: float = 30.0
    default_start_year: int = 2022
    default_end_year: int = 2026
    max_default_scenes: int = 5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
PROJECT_ROOT = Path(__file__).resolve().parents[2]
