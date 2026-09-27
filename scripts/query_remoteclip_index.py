from pathlib import Path
import json

import faiss
import numpy as np
import torch
import open_clip


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "remoteclip"

INDEX_PATH = INDEX_DIR / "remoteclip.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

MODEL_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "models--chendelong--RemoteCLIP"
    / "snapshots"
    / "bf1d8a3ccf2ddbf7c875705e46373bfe542bce38"
    / "RemoteCLIP-ViT-B-32.pt"
)

QUERIES = [
    "dense vegetation and trees",
    "urban area with buildings",
    "new construction",
    "roads and transportation infrastructure",
    "bare soil or cleared land",
    "river or water body",
]


def main():
    print("=" * 70)
    print("GeoIntel - RemoteCLIP Semantic Retrieval")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    index = faiss.read_index(str(INDEX_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Index vectors: {index.ntotal}")
    print(f"Embedding dimension: {index.d}")

    print("\nLoading RemoteCLIP...")

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=str(MODEL_PATH),
    )

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model = model.to(device)
    model.eval()

    print("RemoteCLIP loaded.")

    tokens = tokenizer(QUERIES).to(device)

    with torch.no_grad():
        text_embeddings = model.encode_text(tokens)

    text_embeddings = text_embeddings / text_embeddings.norm(
        dim=-1,
        keepdim=True,
    )

    query_vectors = (
        text_embeddings
        .cpu()
        .numpy()
        .astype("float32")
    )

    scores, indices = index.search(
        query_vectors,
        min(5, index.ntotal),
    )

    print("\n" + "=" * 70)
    print("SEARCH RESULTS")
    print("=" * 70)

    records = metadata["records"]

    for query, query_scores, query_indices in zip(
        QUERIES,
        scores,
        indices,
    ):
        print(f"\nQuery: {query}")

        for rank, (score, index_id) in enumerate(
            zip(query_scores, query_indices),
            start=1,
        ):
            record = records[index_id]

            print(
                f"  {rank}. "
                f"{record['scene_id']} "
                f"| score={score:.4f} "
                f"| {record['image_path']}"
            )


if __name__ == "__main__":
    main()