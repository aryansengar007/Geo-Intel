# GeoIntel

GeoIntel is a geospatial intelligence platform for semantic retrieval and multi-temporal analysis of satellite imagery. The project is designed around the Gurugram, Haryana AOI and a real Copernicus Sentinel-2 Level-2A data pipeline.

## Project overview

The first working milestone focuses on a real, explainable, and testable foundation:

- real AOI definition for Gurugram
- real STAC discovery against the Copernicus Data Space catalogue
- scene selection and metadata persistence
- controlled, safe data downloading
- API-backed dataset status and geospatial services
- GIS-style frontend shell for analyst workflows
- modular architecture for eventual semantic retrieval and change analysis

## Problem statement

The system must eventually support semantic retrieval, temporal change analysis, multi-scene evidence review, and analyst provenance. The architecture is intentionally structured to support those capabilities without hard-coding a fake AI pipeline.

## Gurugram AOI

The primary AOI is Gurugram, Haryana, India. The project stores the polygon in `data/metadata/aoi_gurugram.geojson`.

Important note: this polygon is a documented study-area polygon used for operational development, not an official legal administrative boundary. The file explicitly defines the source and method in metadata.

## Data source

Primary data source:

- Copernicus Data Space STAC catalogue: https://stac.dataspace.copernicus.eu/v1
- Collection: sentinel-2-l2a

The project does not use fabricated imagery or mock satellite metadata as actual analysis data.

## Architecture

The repository is organized around a layered architecture:

- `backend/` for the FastAPI service and geospatial logic
- `scripts/` for STAC discovery, download controls, and validation utilities
- `data/` for AOI metadata and local data archives
- `docs/` for architecture, dataset, and evaluation notes
- `frontend/` for the GIS workstation UI

## Setup

Python environment:

```powershell
Set-Location "D:\My Projects\GeoIntel"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e .
```

Frontend:

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Backend:

```powershell
Set-Location "D:\My Projects\GeoIntel"
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Data acquisition

Discovery uses the current Copernicus Data Space STAC API. The one-scene download
workflow selects an overlapping Sentinel-2 L2A item, obtains a short-lived CDSE
token, and downloads the nine requested JP2 bands plus SAFE/product/granule XML
metadata from the asset HTTPS links supplied by STAC. It does not download the
complete SAFE archive.

```powershell
Set-Location "D:\My Projects\GeoIntel"
.\.venv\Scripts\Activate.ps1
python scripts/download_sentinel2.py --scene --dry-run
python scripts/download_sentinel2.py --scene
```

Before the real download, create a local `.env` from `.env.example` and enter the
CDSE account credentials there. Keep `.env` local; never put credentials in source
files or share them in chat. The downloader uses CDSE's documented OIDC password
grant and keeps access/refresh tokens in memory only. The selected band files are
validated with Rasterio before provenance is recorded under
`data/metadata/downloads/`.

## Data directories

```text
data/
  metadata/
    aoi_gurugram.geojson
    scenes_discovery.json
    downloads/
  raw/
    sentinel2/<year>/<product>.SAFE/  # selected band/metadata assets only
  processed/
  samples/
```

Large raster and model files are intentionally not committed to Git.

## Processing foundations

The pipeline is structured as:

1. STAC Discovery
2. Scene Selection
3. Metadata Storage
4. Download
5. Validation
6. AOI Clipping
7. Cloud / Quality Masking
8. Alignment
9. Derived Indices
10. Tile Generation
11. Feature / Embedding Extraction
12. Vector Index

## Derived spectral products

Implemented formulas include:

- NDVI = (B08 - B04) / (B08 + B04)
- NDWI = (B03 - B08) / (B03 + B08)
- NDBI = (B11 - B08) / (B11 + B08)

These are considered evidence layers, not AI-proof change detections.

## API

The backend exposes a minimal working API with health, AOI, temporal coverage, and dataset status endpoints.

## Frontend

The frontend is a dark, map-centric analytical dashboard shell for future GIS workflows.

## Testing

Run:

```powershell
Set-Location "D:\My Projects\GeoIntel"
.\.venv\Scripts\Activate.ps1
pytest
```

## Offline operation

The design supports offline analysis after data, processing states, and vector indexes are prepared locally. The project avoids requiring a live internet connection for analysis after preparation.

## Limitations

- The current project is a working foundation rather than a complete fully-trained AI system.
- Real Copernicus downloads may require credentials.
- No claims are made about final model quality, retrieval accuracy, or change-detection performance without evaluation.
- The advanced semantic retrieval and multi-temporal analysis layers are architected but not yet fully evaluated on real data.

## Next steps

1. Create a small real scene archive for Gurugram.
2. Add local validation utilities for raster metadata and bands.
3. Add vector index and embedding interface.
4. Extend the GIS frontend with map layers and analyst evidence views.
5. Build retrieval and change-analysis services over validated data.
