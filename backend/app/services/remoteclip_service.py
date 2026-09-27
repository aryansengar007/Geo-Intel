from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import open_clip
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


class RemoteCLIPService:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = None
        self.tokenizer = None
        self.index = None
        self.metadata: dict[str, Any] | None = None

    def _load_model(self) -> None:
        if self.model is not None:
            return

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"RemoteCLIP weights not found: {MODEL_PATH}"
            )

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained=str(MODEL_PATH),
        )

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

    def _load_index(self) -> None:
        if self.index is not None and self.metadata is not None:
            return

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {INDEX_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"FAISS metadata not found: {METADATA_PATH}"
            )

        self.index = faiss.read_index(str(INDEX_PATH))

        with METADATA_PATH.open("r", encoding="utf-8") as file:
            self.metadata = json.load(file)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        query = query.strip()

        if not query:
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self._load_model()
        self._load_index()

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.index is not None
        assert self.metadata is not None

        tokens = self.tokenizer([query]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens)

        embedding = embedding / embedding.norm(
            dim=-1,
            keepdim=True,
        )

        vector = (
            embedding
            .cpu()
            .numpy()
            .astype("float32")
        )

        actual_top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(
            vector,
            actual_top_k,
        )

        records = self.metadata["records"]

        results = []

        for rank, (score, index_id) in enumerate(
            zip(scores[0], indices[0]),
            start=1,
        ):
            if index_id < 0:
                continue

            record = records[int(index_id)]

            results.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "scene_id": record["scene_id"],
                    "image_path": record["image_path"],
                    "cloud_masked": record["cloud_masked"],
                    "acquisition_datetime": record[
                        "acquisition_datetime"
                    ],
                    "tile_id": record["tile_id"],
                    "mission": record["mission"],
                    "relative_orbit": record[
                        "relative_orbit"
                    ],
                }
            )

        return {
            "query": query,
            "model": "RemoteCLIP ViT-B/32",
            "device": self.device,
            "total_indexed_vectors": self.index.ntotal,
            "results": results,
        }


remoteclip_service = RemoteCLIPService()