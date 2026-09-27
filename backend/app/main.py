from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.models.schemas import (
    ChangeAnalysisRequest,
    ChangeAnalysisResponse,
    HealthResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from backend.app.services.remoteclip_service import remoteclip_service
from backend.app.services.change_analysis import analyze_change

from backend.app.config import PROJECT_ROOT, settings
from backend.app.models.schemas import HealthResponse
from backend.app.services.aoi_service import aoi_summary, load_aoi
from backend.app.services.download_service import local_dataset_status
from backend.app.services.sentinel_service import detect_download_auth_requirement, discover_sentinel2_scenes
from backend.app.services.spectral_indices import DERIVED_LAYERS

app = FastAPI(title="GeoIntel API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    dataset_status = local_dataset_status()
    return HealthResponse(
        status="ok",
        service="GeoIntel",
        dataset_status=dataset_status["status"],
    )


@app.get("/api/aoi")
def get_aoi() -> dict:
    if not settings.aoi_path.exists():
        raise FileNotFoundError("AOI file not found")
    return aoi_summary(settings.aoi_path)


@app.get("/api/dataset-status")
def dataset_status() -> dict:
    return local_dataset_status()


@app.get("/api/scenes")
def get_scenes() -> list[dict]:
    if not settings.aoi_path.exists():
        return []
    aoi = load_aoi(settings.aoi_path)
    polygon_coords = aoi["features"][0]["geometry"]["coordinates"][0]
    west = min(coord[0] for coord in polygon_coords)
    south = min(coord[1] for coord in polygon_coords)
    east = max(coord[0] for coord in polygon_coords)
    north = max(coord[1] for coord in polygon_coords)
    return discover_sentinel2_scenes([west, south, east, north], start_year=2022, end_year=2026, max_cloud_cover=30.0, max_scenes=5)


@app.get("/api/temporal-coverage")
def get_temporal_coverage() -> dict:
    years = list(range(2022, 2027))
    return {
        "start_year": 2022,
        "end_year": 2026,
        "years": years,
        "coverage": [f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z" for year in years],
    }


@app.get("/api/processing-status")
def get_processing_status() -> dict:
    dataset = local_dataset_status()
    return {
        "status": dataset["status"],
        "message": dataset["message"],
        "stages": [
            "STAC Discovery",
            "Scene Selection",
            "Metadata Storage",
            "Download",
            "Validation",
            "AOI Clipping",
            "Cloud / Quality Masking",
            "Alignment",
            "Derived Indices",
            "Tile Generation",
            "Feature / Embedding Extraction",
            "Vector Index",
        ],
    }


@app.get("/api/derived-layers")
def get_derived_layers() -> list[dict]:
    return DERIVED_LAYERS


@app.get("/api/auth-status")
def auth_status() -> dict:
    credentials_configured = bool(settings.cdse_username.strip() and settings.cdse_password)
    return {
        "copernicus_download_auth_required": True,
        "credentials_configured": credentials_configured,
        "message": (
            "CDSE credentials are configured locally."
            if credentials_configured
            else "Set CDSE_USERNAME and CDSE_PASSWORD in the local .env file to download products."
        ),
    }

@app.post(
    "/api/search",
    response_model=SemanticSearchResponse,
)
def semantic_search(
    request: SemanticSearchRequest,
) -> SemanticSearchResponse:
    result = remoteclip_service.search(
        query=request.query,
        top_k=request.top_k,
    )

    return SemanticSearchResponse(**result)

@app.post(
    "/api/change-analysis",
    response_model=ChangeAnalysisResponse,
)
def change_analysis(
    request: ChangeAnalysisRequest,
) -> ChangeAnalysisResponse:
    result = analyze_change(
        before_scene=request.before_scene,
        after_scene=request.after_scene,
        sensitivity=request.sensitivity,
    )

    return ChangeAnalysisResponse(**result)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
