# Sentinel-2 Quality and Preprocessing

## Scope

The local first-scene SAFE directory began as a partial product. The exact scene's live STAC item exposed independently downloadable `SCL_20m`, `CLD_20m`, and `SNW_20m` assets. These were retrieved through their authenticated OData `$value` asset links and saved at their STAC-declared paths under the existing SAFE: `IMG_DATA/R20m/*_SCL_20m.jp2` and `QI_DATA/MSK_CLDPRB_20m.jp2` / `MSK_SNWPRB_20m.jp2`. No scene or full SAFE archive was redownloaded. Provenance records the source hrefs, sizes, checksums, and download timestamps.

Run a read-only inspection:

```powershell
python scripts/inspect_sentinel2.py
```

Run inspection, AOI preprocessing, indices, and visual previews:

```powershell
python scripts/inspect_sentinel2.py --preprocess
```

Use `--no-previews` to skip PNG preview generation. These commands use only the already-downloaded product and do not request additional scenes.

Fetch the small quality assets for this same local scene (the command is restricted to STAC-listed SCL/CLD/SNW keys):

```powershell
python scripts/download_sentinel2_quality.py --scene S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519
```

Apply quality masks to the existing aligned outputs and write separate masked bands, indices, and previews:

```powershell
python scripts/process_sentinel2_quality.py --scene S2C_MSIL2A_20260910T052641_N0512_R105_T43RFM_20260910T102519
```

## Surface Reflectance

The product's `MTD_MSIL2A.xml` is authoritative for this acquisition. It declares `BOA_QUANTIFICATION_VALUE=10000` and `BOA_ADD_OFFSET=-1000` for each band ID. Conversion is performed as:

`surface reflectance = (stored DN + band BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE`

For the bands used here, Sentinel-2 band IDs are B02=1, B03=2, B04=3, B05=4, B06=5, B07=6, B08=7, B11=11, and B12=12. Stored JP2 samples are `uint16`; Rasterio reports no embedded nodata value for the current JP2 files. Product `SPECIAL_VALUES` identify DN 0 as no-data and DN 65535 as saturated. Converted negative or unusually high reflectance values are retained and reported, not clamped.

## Grid and Clipping

The common analysis grid is the B02 10 m EPSG:32643 grid clipped to the projected AOI bounding window, with pixels outside the AOI polygon set to output nodata. Continuous reflectance bands are resampled with bilinear interpolation. Derived bands are kept as float32 surface reflectance. Outputs use GeoTIFF, DEFLATE compression, and tiled storage. Processing reads/reprojects one source band at a time; index calculations run in 256-row windows.

SCL is aligned with nearest-neighbor resampling. SCL is categorical; continuous interpolation is not appropriate. Cloud/snow probability rasters are continuous and are bilinearly aligned.

## SCL Classes

Class meanings follow ESA's Sentinel-2 processing documentation, _S2 Processing_, L2A Scene Classification, Table 3:

| Value | Class                     |
| ----: | ------------------------- |
|     0 | No data                   |
|     1 | Saturated or defective    |
|     2 | Cast shadows              |
|     3 | Cloud shadows             |
|     4 | Vegetation                |
|     5 | Not vegetated             |
|     6 | Water                     |
|     7 | Unclassified              |
|     8 | Cloud, medium probability |
|     9 | Cloud, high probability   |
|    10 | Thin cirrus               |
|    11 | Snow or ice               |

The quality-screened analysis mask flags classes 0, 1, 2, 3, 8, 9, 10, and 11. Class 2 is treated as cast/dark-shadow evidence and masked conservatively for change analysis; classes 4, 5, 6, and 7 remain usable. Cloud probability is additionally masked at >=40%; values from 1% through 39% are retained unless SCL independently marks them. Snow probability is masked at >=50%. These are explicit project policy thresholds, not claims about the official SCL classification thresholds. The mask does not classify land cover.

Official source: ESA Copernicus Sentinel-2 _S2 Processing_, https://sentiwiki.copernicus.eu/web/s2-processing

## Indices

- NDVI: `(B08 - B04) / (B08 + B04)`
- NDWI: McFeeters green-NIR water index, `(B03 - B08) / (B03 + B08)`
- NDBI: `(B11 - B08) / (B11 + B08)`

Zero/non-finite denominators and source nodata produce output nodata. The index TIFFs carry formula and input-unit tags.

## AOI

The existing valid single Polygon is retained unchanged, with bounds 77.005-77.110 E and 28.330-28.520 N. Its authoritative area is **216.6173568 km2**, calculated geodesically on the WGS84 ellipsoid with `pyproj.Geod` from the actual EPSG:4326 coordinates. EPSG:32643 is appropriate (the AOI is within UTM zone 43N); its planar area is 216.6610663 km2, a 0.0202% difference used as a raster-grid cross-check. The prior GeoJSON estimate of 247 km2 was unsupported; the backend's former 89.0004 km2 came from `geometry.area * 111.32^2 * 0.36`, an invalid fixed-factor conversion of square degrees. The metadata property now records the geodesic value and retains the old values for audit.

## Quality Interpretation

The quality report records scene-level STAC cloud cover separately from pixel-level SCL/CLD/SNW AOI counts. On this scene, the SCL grid is 20 m, `uint8`, EPSG:32643, 5490 x 5490. After nearest-neighbor resampling to the 10 m aligned analysis grid, the AOI has 2,166,609 pixels: 0 cloud (0.0000%), 0 cloud shadow (0.0000%), 0 cirrus (0.0000%), 0 snow/ice (0.0000%), 312 other masked cast-shadow/no-data pixels (0.0144%), and 2,166,297 usable pixels (99.9856%). CLD values >=40% and SNW values >=50% are included in those masks; low positive CLD values are retained unless SCL masks them.

The B02/B03 review outliers were rechecked in the original JP2s. Six B02 samples (DN 21,840-23,891; SR 2.0840-2.2891) and five B03 samples (DN 21,039-21,566; SR 2.0039-2.0566) are inside the AOI. They are not DN 0 or 65,535, occur in SCL class 5 with CLD/SNW probability zero, and their aligned 10 m values match the exact `(DN - 1000) / 10000` result. The surrounding spectra are consistent with very bright bare/built surfaces; exact materials are not inferred. These source-valid values were not clipped. Per-band DN/reflectance percentiles, source row/column/map coordinates, and outlier context are recorded in the quality JSON.

The catalogue's scene-level cloud cover is 0.0%; pixel-level AOI statistics are independently calculated from SCL/CLD/SNW. A quality gate may report PASS after source scaling, masks, AOI geometry/area, processing outputs, and tests validate; the outlier observations remain documented.

Cloud-screened reflectance, quality mask, aligned QA layers, NDVI/NDWI/NDBI, and previews are written under `data/processed/cloud_masked/`, `data/processed/indices/<scene>/cloud_masked/`, and `data/processed/tiles/<scene>/cloud_masked/`. Previously generated unmasked reflectance and index products are preserved. The machine-readable baseline is stored at `data/metadata/baselines/<scene>.baseline.json` and references rather than copies large rasters.
