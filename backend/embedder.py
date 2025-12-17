import os
import json
from typing import List, Dict, Any, Optional

import faiss
import numpy as np

# Use absolute paths relative to this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSESSMENTS_PATH = os.path.join(DATA_DIR, "assessments.json")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

# Try importing SentenceTransformer; if it fails, we are in production (Render)
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    HAS_ST = False

class Embedder:
    """
    Handles:
      - Creating rich text representations for assessments (local only)
      - Building a FAISS index (local only)
      - Loading the index + metadata (local and Render)
      - Searching for similar assessments
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

        if HAS_ST:
            # Local/dev: full functionality, load model
            print(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(self.model_name)
        else:
            # Production on Render: no model, only precomputed index
            print("SentenceTransformer not available. Running in load-only mode.")
            self.model = None

        self._load_index_and_metadata()

    def _load_index_and_metadata(self) -> None:
        if not (os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)):
            raise RuntimeError(
                f"FAISS index or metadata not found. "
                f"Expected {INDEX_PATH} and {METADATA_PATH}. "
                f"Generate them locally with embedder.py and commit to the repo."
            )

        # Load FAISS index
        print(f"Loading FAISS index from {INDEX_PATH}")
        self.index = faiss.read_index(INDEX_PATH)

        # Load metadata
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loaded FAISS index with {self.index.ntotal} vectors")

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
        Only used locally to build index. Requires sentence-transformers & torch.
        """
        if not HAS_ST or self.model is None:
            raise RuntimeError(
                "SentenceTransformer not available. "
                "This method is for local use only. Do not call on Render."
            )

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

        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"Index saved to {INDEX_PATH}")
        print(f"Metadata saved to {METADATA_PATH}")

    def embed_query(self, query_text: str) -> np.ndarray:
        """
        Embeds a query text. Only available if SentenceTransformer is loaded.
        """
        if not HAS_ST or self.model is None:
            raise RuntimeError(
                "Query embedding requested but SentenceTransformer is not available. "
                "This method is for local use or environments with ST installed."
            )
        emb = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Search the index for the most similar assessments to the given query vector.
        Returns a list of assessments with an added 'score' field.
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Call load_index() first.")

        scores, indices = self.index.search(query_vector, top_k)
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
    # This block is for local index building only
    if not os.path.exists(ASSESSMENTS_PATH):
        raise FileNotFoundError(
            f"{ASSESSMENTS_PATH} not found. Run crawler.py first to create it."
        )

    with open(ASSESSMENTS_PATH, "r", encoding="utf-8") as f:
        assessments_data = json.load(f)

    embedder = Embedder()
    embedder.build_index(assessments_data)