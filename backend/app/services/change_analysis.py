from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLOUD_MASKED_ROOT = PROJECT_ROOT / "data" / "processed" / "cloud_masked"
CHANGE_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "outputs" / "change_analysis"
)


BANDS = [
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B11",
    "B12",
]


def _find_scene_directory(scene_id: str) -> Path:
    matches = list(CLOUD_MASKED_ROOT.glob(f"{scene_id}"))

    if not matches:
        matches = list(CLOUD_MASKED_ROOT.glob(f"*{scene_id}*"))

    if not matches:
        raise FileNotFoundError(
            f"Processed cloud-masked scene not found: {scene_id}"
        )

    return matches[0]


def _band_path(scene_dir: Path, band: str) -> Path:
    path = scene_dir / f"{band}_10m_reflectance_scl_masked.tif"

    if not path.exists():
        raise FileNotFoundError(
            f"Required band not found: {path}"
        )

    return path


def _read_band(scene_dir: Path, band: str) -> tuple[np.ndarray, np.ndarray]:
    path = _band_path(scene_dir, band)

    with rasterio.open(path) as dataset:
        values = dataset.read(1, masked=True)
        valid = ~np.ma.getmaskarray(values)
        data = values.filled(np.nan).astype(np.float32)

    return data, valid


def _read_scene(scene_id: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    scene_dir = _find_scene_directory(scene_id)

    bands: dict[str, np.ndarray] = {}
    joint_valid: np.ndarray | None = None

    reference_shape: tuple[int, int] | None = None

    for band in BANDS:
        data, valid = _read_band(scene_dir, band)

        if reference_shape is None:
            reference_shape = data.shape
        elif data.shape != reference_shape:
            raise ValueError(
                f"Band shape mismatch in {scene_id}: "
                f"{band}={data.shape}, expected={reference_shape}"
            )

        bands[band] = data

        if joint_valid is None:
            joint_valid = valid.copy()
        else:
            joint_valid &= valid

    if joint_valid is None:
        raise ValueError(f"No bands available for scene: {scene_id}")

    return bands, joint_valid


def _normalized_difference(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    denominator = first + second

    result = np.full(first.shape, np.nan, dtype=np.float32)

    valid = (
        np.isfinite(first)
        & np.isfinite(second)
        & (np.abs(denominator) > 1e-6)
    )

    result[valid] = (
        (first[valid] - second[valid])
        / denominator[valid]
    )

    return result


def _compute_indices(
    bands: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        "ndvi": _normalized_difference(
            bands["B08"],
            bands["B04"],
        ),
        "ndwi": _normalized_difference(
            bands["B03"],
            bands["B08"],
        ),
        "ndbi": _normalized_difference(
            bands["B11"],
            bands["B08"],
        ),
    }


def _mean_absolute_spectral_change(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
    valid: np.ndarray,
) -> np.ndarray:
    differences = []

    for band in BANDS:
        difference = np.abs(after[band] - before[band])
        differences.append(difference)

    stacked = np.stack(differences, axis=0)

    result = np.full(
        valid.shape,
        np.nan,
        dtype=np.float32,
    )

    valid_pixels = valid & np.any(
        np.isfinite(stacked),
        axis=0,
    )

    result[valid_pixels] = np.nanmean(
        stacked[:, valid_pixels],
        axis=0,
    ).astype(np.float32)

    return result


def _classify_changes(
    spectral_change: np.ndarray,
    delta_ndvi: np.ndarray,
    delta_ndwi: np.ndarray,
    delta_ndbi: np.ndarray,
    valid: np.ndarray,
    sensitivity: float,
) -> dict[str, np.ndarray]:
    spectral_threshold = sensitivity
    index_threshold = sensitivity * 0.75

    changed = (
        valid
        & np.isfinite(spectral_change)
        & (spectral_change >= spectral_threshold)
    )

    remaining = changed.copy()

    classes: dict[str, np.ndarray] = {}

    classes["vegetation_loss"] = (
        remaining
        & (delta_ndvi <= -index_threshold)
    )
    remaining &= ~classes["vegetation_loss"]

    classes["vegetation_growth"] = (
        remaining
        & (delta_ndvi >= index_threshold)
    )
    remaining &= ~classes["vegetation_growth"]

    classes["water_expansion"] = (
        remaining
        & (delta_ndwi >= index_threshold)
        & (delta_ndvi <= index_threshold)
    )
    remaining &= ~classes["water_expansion"]

    classes["water_contraction"] = (
        remaining
        & (delta_ndwi <= -index_threshold)
        & (delta_ndvi >= -index_threshold)
    )
    remaining &= ~classes["water_contraction"]

    classes["built_up_construction"] = (
        remaining
        & (delta_ndbi >= index_threshold)
        & (delta_ndvi <= 0.0)
    )
    remaining &= ~classes["built_up_construction"]

    classes["other_surface_change"] = remaining

    return classes

def _write_change_raster(
    output_path: Path,
    reference_scene: str,
    change_raster: np.ndarray,
) -> None:
    reference_dir = _find_scene_directory(reference_scene)
    reference_path = _band_path(reference_dir, "B04")

    with rasterio.open(reference_path) as source:
        profile = source.profile.copy()

    profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = np.where(
        np.isfinite(change_raster),
        change_raster,
        -9999.0,
    ).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(output, 1)


def _write_class_raster(
    output_path: Path,
    reference_scene: str,
    change_classes: dict[str, np.ndarray],
) -> None:
    reference_dir = _find_scene_directory(reference_scene)
    reference_path = _band_path(reference_dir, "B04")

    with rasterio.open(reference_path) as source:
        profile = source.profile.copy()

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="deflate",
    )

    class_raster = np.zeros(
        source.shape,
        dtype=np.uint8,
    )

    class_codes = {
        "vegetation_loss": 1,
        "vegetation_growth": 2,
        "water_expansion": 3,
        "water_contraction": 4,
        "built_up_construction": 5,
        "other_surface_change": 6,
    }

    for class_name, code in class_codes.items():
        class_raster[change_classes[class_name]] = code

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(output_path, "w", **profile) as destination:
        destination.write(class_raster, 1)


def analyze_change(
    before_scene: str,
    after_scene: str,
    sensitivity: float = 0.20,
) -> dict:
    if not 0.0 < sensitivity <= 1.0:
        raise ValueError(
            "Sensitivity must be greater than 0 and at most 1."
        )

    before, before_valid = _read_scene(before_scene)
    after, after_valid = _read_scene(after_scene)

    if before["B02"].shape != after["B02"].shape:
        raise ValueError(
            "Before and after scenes do not have matching raster dimensions."
        )

    valid = before_valid & after_valid

    if not np.any(valid):
        raise ValueError(
            "No jointly valid pixels exist between the two scenes."
        )

    before_indices = _compute_indices(before)
    after_indices = _compute_indices(after)

    spectral_change = _mean_absolute_spectral_change(
        before,
        after,
        valid,
    )

    delta_ndvi = (
        after_indices["ndvi"]
        - before_indices["ndvi"]
    )

    delta_ndwi = (
        after_indices["ndwi"]
        - before_indices["ndwi"]
    )

    delta_ndbi = (
        after_indices["ndbi"]
        - before_indices["ndbi"]
    )

    change_classes = _classify_changes(
        spectral_change,
        delta_ndvi,
        delta_ndwi,
        delta_ndbi,
        valid,
        sensitivity,
    )
    output_name = (
        f"{before_scene[:15]}_to_{after_scene[:15]}"
        f"_change_score.tif"
    )

    change_raster_path = (
        CHANGE_OUTPUT_ROOT / output_name
    )

    _write_change_raster(
        change_raster_path,
        before_scene,
        spectral_change,
    )

    class_raster_name = (
        f"{before_scene[:15]}_to_{after_scene[:15]}"
        f"_change_classes.tif"
    )

    class_raster_path = (
        CHANGE_OUTPUT_ROOT / class_raster_name
    )

    _write_class_raster(
        class_raster_path,
        before_scene,
        change_classes,
    )

    changed = (
        valid
        & np.isfinite(spectral_change)
        & (spectral_change >= sensitivity)
    )

    valid_count = int(np.count_nonzero(valid))
    changed_count = int(np.count_nonzero(changed))

    def valid_mean(values: np.ndarray) -> float:
        subset = values[valid]
        finite = subset[np.isfinite(subset)]

        if finite.size == 0:
            return 0.0

        return float(np.mean(finite))

    class_summaries = []

    for name, mask in change_classes.items():
        count = int(np.count_nonzero(mask))

        class_summaries.append(
            {
                "change_class": name,
                "pixel_count": count,
                "percentage_of_valid_pixels": (
                    (count / valid_count) * 100.0
                    if valid_count
                    else 0.0
                ),
            }
        )

    return {
        "before_scene": before_scene,
        "after_scene": after_scene,
        "before_date": (
            before_scene.split("_")[2][:8]
            if len(before_scene.split("_")) > 2
            else before_scene
        ),
        "after_date": (
            after_scene.split("_")[2][:8]
            if len(after_scene.split("_")) > 2
            else after_scene
        ),
        "valid_pixel_count": valid_count,
        "changed_pixel_count": changed_count,
        "changed_percentage": (
            (changed_count / valid_count) * 100.0
            if valid_count
            else 0.0
        ),
        "mean_spectral_change": valid_mean(spectral_change),
        "mean_delta_ndvi": valid_mean(delta_ndvi),
        "mean_delta_ndwi": valid_mean(delta_ndwi),
        "mean_delta_ndbi": valid_mean(delta_ndbi),
        "sensitivity": sensitivity,
        "change_raster_path": str(change_raster_path),
        "change_class_raster_path": str(class_raster_path),
        "change_classes": class_summaries,
    }