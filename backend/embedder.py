import os
import json
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


ASSESSMENTS_PATH = "data/assessments.json"
INDEX_DIR = "data/faiss_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")


class Embedder:
    """
    Handles:
      - Creating rich text representations for assessments
      - Building a FAISS index
      - Loading the index + metadata
      - Searching for similar assessments
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata: List[Dict[str, Any]] = []

    def create_embedding_text(self, assessment: Dict[str, Any]) -> str:
        """
        Create a rich text field from an assessment to feed into the embedding model.
        """
        name = assessment.get("name", "") or ""
        description = assessment.get("description", "") or ""
        test_types = assessment.get("test_type", []) or []
        adaptive_support = assessment.get("adaptive_support", "No") or "No"
        remote_support = assessment.get("remote_support", "No") or "No"
        duration = assessment.get("duration", 0) or 0

        parts = [
            f"Assessment Name: {name}",
            f"Description: {description}",
        ]

        if test_types:
            parts.append(f"Test Types: {', '.join(test_types)}")

        if adaptive_support == "Yes":
            parts.append("Adaptive Test: Yes")

        if remote_support == "Yes":
            parts.append("Remote / Online: Yes")

        if duration > 0:
            parts.append(f"Duration: {duration} minutes")

        return ". ".join(parts)

    def build_index(self, assessments: List[Dict[str, Any]]) -> None:
        """
        Build a FAISS index from assessments and save it + metadata.
        """
        print(f"Building FAISS index for {len(assessments)} assessments...")

        texts = [self.create_embedding_text(a) for a in assessments]

        print("Generating embeddings...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        print(f"Index built with {index.ntotal} vectors")

        self.index = index
        self.metadata = assessments

        self.save_index(INDEX_DIR)

    def save_index(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"Index saved to {INDEX_PATH}")
        print(f"Metadata saved to {METADATA_PATH}")

    def load_index(self, index_dir: str = INDEX_DIR) -> None:
        if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
            raise FileNotFoundError(
                f"Index or metadata not found in {index_dir}. "
                f"Run embedder.py to build the index first."
            )

        print(f"Loading FAISS index from {INDEX_PATH}")
        self.index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def search(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search the index for the most similar assessments to the given query text.
        Returns a list of assessments with an added 'score' field.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load_index() first.")

        query_emb = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, indices = self.index.search(query_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx < 0:
                continue
            metadata = self.metadata[idx]
            item = dict(metadata)
            item["score"] = float(score)
            item["rank"] = rank
            results.append(item)

        return results


if __name__ == "__main__":
    if not os.path.exists(ASSESSMENTS_PATH):
        raise FileNotFoundError(
            f"{ASSESSMENTS_PATH} not found. Run crawler.py first to create it."
        )

    with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
        assessments_data = json.load(f)

    embedder = Embedder()
    embedder.build_index(assessments_data)