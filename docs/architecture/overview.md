# Architecture Overview

GeoIntel is built around a layered geospatial intelligence pipeline:

1. AOI definition and metadata validation.
2. STAC-based Copernicus discovery for Sentinel-2 Level-2A scenes.
3. Controlled selection and metadata persistence.
4. Download and validation for approved scenes.
5. AOI clipping, masking, alignment, and index generation.
6. Local vector embedding and retrieval architecture.
7. FastAPI-backed analyst services and React GIS UI.

The project intentionally emphasizes a real data foundation before advanced AI features. The first milestone is a working Gurugram Sentinel-2 archive with preprocessing, API access, and a local GIS workstation.
