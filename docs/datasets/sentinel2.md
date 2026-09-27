# Sentinel-2 Dataset Strategy

The project uses Copernicus Sentinel-2 Level-2A Surface Reflectance data as the primary Earth observation dataset for Gurugram. The discovery endpoint is the official STAC catalog at https://stac.dataspace.copernicus.eu/v1 and the collection is sentinel-2-l2a.

Important rules:

- Discovery is implemented without downloading the full archive.
- Cloud filtering is applied before scene selection.
- Scene selection is conservative and intentionally small.
- Downloading actual assets may require a Copernicus account or token.
- The project distinguishes between discovered, downloaded, validated, and demo data states at all times.
