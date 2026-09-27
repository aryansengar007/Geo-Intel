from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AoiInfo(BaseModel):
    name: str
    description: str
    crs: str
    source: str
    source_method: str
    bbox: list[float]
    area_km2: float
    geometry: dict[str, Any]


class SceneMetadata(BaseModel):
    scene_id: str
    product_id: str | None = None
    platform: str | None = None
    sensor: str | None = None
    acquisition_date: str | None = None
    processing_level: str | None = None
    cloud_cover: float | None = None
    bbox: list[float] | None = None
    geometry: dict[str, Any] | None = None
    crs: str | None = None
    resolution: dict[str, float] | None = None
    available_bands: list[str] = Field(default_factory=list)
    source: str | None = None
    source_url: str | None = None
    local_path: str | None = None
    download_status: str = "not_started"
    processing_version: str | None = None


class DatasetStatus(BaseModel):
    status: str
    real_data_count: int = 0
    loading_count: int = 0
    empty_count: int = 0
    error_count: int = 0
    demo_count: int = 0
    message: str


class TemporalCoverage(BaseModel):
    start_year: int
    end_year: int
    years: list[int]
    coverage: list[str]


class DerivedLayer(BaseModel):
    name: str
    description: str
    formula: str
    source_bands: list[str]
    resolution: str
    resampling: str | None = None
    interpretation: str
    limitations: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str = "0.1.0"
    dataset_status: str = "empty"
    
    
class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=50)


class SemanticSearchResult(BaseModel):
    rank: int
    score: float
    scene_id: str
    image_path: str
    cloud_masked: bool
    acquisition_datetime: str
    tile_id: str
    mission: str
    relative_orbit: str


class SemanticSearchResponse(BaseModel):
    query: str
    model: str
    device: str
    total_indexed_vectors: int
    results: list[SemanticSearchResult]
    

class ChangeAnalysisRequest(BaseModel):
    before_scene: str = Field(..., min_length=1, max_length=300)
    after_scene: str = Field(..., min_length=1, max_length=300)
    sensitivity: float = Field(default=0.20, gt=0.0, le=1.0)


class ChangeClassSummary(BaseModel):
    change_class: str
    pixel_count: int
    percentage_of_valid_pixels: float


class ChangeAnalysisResponse(BaseModel):
    before_scene: str
    after_scene: str
    before_date: str
    after_date: str
    valid_pixel_count: int
    changed_pixel_count: int
    changed_percentage: float
    mean_spectral_change: float
    mean_delta_ndvi: float
    mean_delta_ndwi: float
    mean_delta_ndbi: float
    sensitivity: float
    change_raster_path: str
    change_class_raster_path: str
    change_classes: list[ChangeClassSummary]
