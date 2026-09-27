from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from backend.app.config import PROJECT_ROOT


CHANGE_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "outputs" / "change_analysis"
)

OVERLAY_OUTPUT_ROOT = (
    PROJECT_ROOT / "data" / "processed" / "tiles" / "change_overlays"
)

CLASS_COLORS = {
    1: (255, 80, 80, 150),      # vegetation loss
    2: (80, 220, 120, 150),     # vegetation growth
    3: (50, 170, 255, 160),     # water expansion
    4: (0, 100, 255, 160),      # water contraction
    5: (255, 180, 40, 175),     # built-up / construction
    6: (190, 100, 255, 150),    # other surface change
}


def _scene_prefix(scene_id: str) -> str:
    """
    Convert a full Sentinel-2 scene ID into the prefix used by
    the existing change-analysis output filenames.

    Example:
        S2B_MSIL2A_20221205T053209_N0510_R105_T43RGM_20240807T114821
        ->
        S2B_MSIL2A_2022
    """
    parts = scene_id.split("_")

    if len(parts) < 3:
        raise ValueError(f"Invalid Sentinel-2 scene ID: {scene_id}")

    date_token = parts[2]

    if len(date_token) < 4:
        raise ValueError(f"Invalid acquisition date in scene ID: {scene_id}")

    return f"{parts[0]}_{parts[1]}_{date_token[:4]}"


def _find_class_raster(
    before_scene: str,
    after_scene: str,
) -> Path:
    before_prefix = _scene_prefix(before_scene)
    after_prefix = _scene_prefix(after_scene)

    raster_path = (
        CHANGE_OUTPUT_ROOT
        / f"{before_prefix}_to_{after_prefix}_change_classes.tif"
    )

    if not raster_path.is_file():
        raise FileNotFoundError(
            f"Change-class raster not found: {raster_path}"
        )

    return raster_path


def render_change_overlay(
    before_scene: str,
    after_scene: str,
) -> Path:
    """
    Render the existing change-class raster as a transparent PNG.

    Class 0 remains fully transparent.
    Classes 1-6 receive semi-transparent colors.

    The PNG uses the exact raster dimensions, preserving spatial
    alignment with the Sentinel-2 preview.
    """

    class_raster_path = _find_class_raster(
        before_scene,
        after_scene,
    )

    output_name = (
        f"{_scene_prefix(before_scene)}"
        f"_to_{_scene_prefix(after_scene)}"
        "_change_overlay.png"
    )

    output_path = OVERLAY_OUTPUT_ROOT / output_name

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with rasterio.open(class_raster_path) as dataset:
        classes = dataset.read(1)

    height, width = classes.shape

    rgba = np.zeros(
        (height, width, 4),
        dtype=np.uint8,
    )

    for class_code, color in CLASS_COLORS.items():
        mask = classes == class_code
        rgba[mask] = color

    image = Image.fromarray(
        rgba,
        mode="RGBA",
    )

    # Match the existing Sentinel preview size exactly.
    preview_width = 166
    preview_height = 512

    image = image.resize(
        (preview_width, preview_height),
        Image.Resampling.NEAREST,
    )

    image.save(
        output_path,
        format="PNG",
    )

    return output_path