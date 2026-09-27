from pathlib import Path
import json
import re

import faiss
import numpy as np
import torch
import open_clip
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TILES_ROOT = PROJECT_ROOT / "data" / "processed" / "tiles"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "remoteclip"

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "models--chendelong--RemoteCLIP"
    / "snapshots"
    / "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    / "RemoteCLIP-ViT-B-32.pt"
)

IMAGE_NAME = "true_color.png"


def extract_scene_id(image_path: Path) -> str:
    """
    Extract the Sentinel-2 scene ID from the directory structure.
    """
    for parent in image_path.parents:
        if parent.name.startswith("S2") and "_MSIL2A_" in parent.name:
            return parent.name

    raise ValueError(f"Could not determine scene ID: {image_path}")


def extract_scene_metadata(scene_id: str) -> dict:
    """
    Extract basic metadata directly from the Sentinel-2 scene ID.
    """
    match = re.match(
        r"(?P<mission>S2[ABCE])_MSIL2A_"
        r"(?P<date>\d{8}T\d{6})_"
        r"N(?P<processing>\d+)_"
        r"R(?P<relative_orbit>\d+)_"
        r"T(?P<tile>\w+)_"
        r"(?P<generation>\d{8}T\d{6})",
        scene_id,
    )

    if not match:
        raise ValueError(f"Unexpected Sentinel-2 scene ID: {scene_id}")

    data = match.groupdict()

    return {
        "mission": data["mission"],
        "acquisition_datetime": data["date"],
        "processing_baseline": data["processing"],
        "relative_orbit": data["relative_orbit"],
        "tile_id": data["tile"],
        "product_generation_datetime": data["generation"],
    }


def main():
    print("=" * 70)
    print("GeoIntel - RemoteCLIP FAISS Index Builder")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RemoteCLIP weights not found:\n{MODEL_PATH}"
        )

    print("\nLoading RemoteCLIP...")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=str(MODEL_PATH),
    )

    model = model.to(device)
    model.eval()

    print("RemoteCLIP loaded.")

    image_paths = sorted(
        TILES_ROOT.rglob(IMAGE_NAME)
    )

    # Avoid indexing the same Sentinel-2 scene twice.
    # Prefer the cloud-masked version when available.
    selected = {}

    for image_path in image_paths:
        scene_id = extract_scene_id(image_path)

        is_cloud_masked = "cloud_masked" in image_path.parts

        if scene_id not in selected:
            selected[scene_id] = (image_path, is_cloud_masked)
        elif is_cloud_masked:
            selected[scene_id] = (image_path, True)

    records = []

    print(f"\nDiscovered {len(image_paths)} true-color files.")
    print(f"Unique scenes: {len(selected)}")

    for scene_id, (image_path, is_cloud_masked) in sorted(selected.items()):
        print(f"\nProcessing:")
        print(f"  Scene: {scene_id}")
        print(f"  Image: {image_path}")
        print(f"  Cloud masked: {is_cloud_masked}")

        image = Image.open(image_path).convert("RGB")

        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        vector = embedding.cpu().numpy().astype("float32")[0]

        metadata = extract_scene_metadata(scene_id)

        records.append(
            {
                "scene_id": scene_id,
                "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                "cloud_masked": is_cloud_masked,
                "image_width": image.width,
                "image_height": image.height,
                **metadata,
            }
        )

        # Temporarily attach vector for index construction.
        records[-1]["_embedding"] = vector

    if not records:
        raise RuntimeError(
            f"No {IMAGE_NAME} files found under {TILES_ROOT}"
        )

    embeddings = np.vstack(
        [record.pop("_embedding") for record in records]
    ).astype("float32")

    dimension = embeddings.shape[1]

    print(f"\nEmbedding dimension: {dimension}")
    print(f"Vectors: {len(embeddings)}")

    # Inner product on L2-normalized vectors = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index_path = INDEX_DIR / "remoteclip.index"
    metadata_path = INDEX_DIR / "metadata.json"

    faiss.write_index(index, str(index_path))

    metadata = {
        "model": "RemoteCLIP ViT-B/32",
        "model_weights": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "embedding_dimension": dimension,
        "metric": "cosine_similarity",
        "index_type": "IndexFlatIP",
        "vector_count": len(records),
        "records": records,
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("INDEX CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"FAISS index : {index_path}")
    print(f"Metadata    : {metadata_path}")
    print(f"Vectors     : {len(records)}")
    print(f"Dimension   : {dimension}")
    print(f"Metric      : cosine similarity")


if __name__ == "__main__":
    main()