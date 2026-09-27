from __future__ import annotations


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def compute_ndvi(red: float, nir: float) -> float:
    return safe_divide(nir - red, nir + red)


def compute_ndwi(green: float, nir: float) -> float:
    return safe_divide(green - nir, green + nir)


def compute_ndbi(swir: float, nir: float) -> float:
    return safe_divide(swir - nir, swir + nir)


DERIVED_LAYERS = [
    {
        "name": "NDVI",
        "description": "Normalized Difference Vegetation Index",
        "formula": "(B08 - B04) / (B08 + B04)",
        "source_bands": ["B08", "B04"],
        "resolution": "10 m",
        "resampling": "none",
        "interpretation": "Positive values generally indicate photosynthetic vegetation; values near zero indicate bare soil or built structures.",
        "limitations": "Sensitive to soil brightness, atmospheric conditions, and seasonal phenology; not a stand-alone change detector.",
    },
    {
        "name": "NDWI",
        "description": "Normalized Difference Water Index",
        "formula": "(B03 - B08) / (B03 + B08)",
        "source_bands": ["B03", "B08"],
        "resolution": "10 m",
        "resampling": "none",
        "interpretation": "Higher values suggest water presence or moisture; useful for monitoring open water and soil moisture.",
        "limitations": "Can be affected by shadows, built surfaces, and mixed pixels; should be combined with contextual evidence.",
    },
    {
        "name": "NDBI",
        "description": "Normalized Difference Built-up Index",
        "formula": "(B11 - B08) / (B11 + B08)",
        "source_bands": ["B11", "B08"],
        "resolution": "20 m (resample to 10 m only if needed)",
        "resampling": "bilinear if harmonized to 10 m",
        "interpretation": "Higher values indicate built-up or impervious urban surfaces compared with vegetation or soil.",
        "limitations": "Urban indices are sensitive to spectral mixing, shadows, and bare-soil response; requires contextual classification.",
    },
]
