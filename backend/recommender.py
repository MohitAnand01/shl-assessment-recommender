from typing import List, Dict, Any

from embedder import Embedder
from query_processor import QueryProcessor


class Recommender:
    """
    Uses:
      - QueryProcessor to understand the query
      - Embedder to get initial candidates
      - A re-ranking stage to:
          * boost skill matches
          * boost test type matches
          * respect duration constraints
    """

    def __init__(
        self,
        embedder: Embedder,
        query_processor: QueryProcessor,
        candidate_pool_size: int = 50,
    ) -> None:
        self.embedder = embedder
        self.query_processor = query_processor
        self.candidate_pool_size = candidate_pool_size

    def recommend(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        signals = self.query_processor.process_query(query)
        enhanced_query = signals["enhanced_query"]
        extracted_skills: List[str] = signals["extracted_skills"]
        extracted_test_types: List[str] = signals["extracted_test_types"]
        max_duration_minutes = signals["max_duration_minutes"]

        # 1) Initial retrieval from FAISS with an oversampled pool
        candidates = self.embedder.search(
            enhanced_query, top_k=self.candidate_pool_size
        )

        # 2) Re-ranking
        reranked: List[Dict[str, Any]] = []
        for cand in candidates:
            score = cand.get("score", 0.0)
            name = (cand.get("name") or "").lower()
            desc = (cand.get("description") or "").lower()
            test_types = [t.lower() for t in cand.get("test_type", [])]
            duration = cand.get("duration", 0) or 0

            # --- Boost: skills present in name/description ---
            if extracted_skills:
                for skill in extracted_skills:
                    if skill.lower() in name or skill.lower() in desc:
                        score *= 1.2  # small boost per matching skill

            # --- Boost: overlapping test types ---
            if extracted_test_types and test_types:
                if any(tt in test_types for tt in extracted_test_types):
                    score *= 1.5  # bigger boost for test type match

            # --- Duration penalty: if over max_duration_minutes ---
            if max_duration_minutes is not None and duration > 0:
                if duration > max_duration_minutes:
                    # penalize tests that exceed requested duration
                    score *= 0.5

            cand_copy = {
                "url": cand.get("url"),
                "name": cand.get("name"),
                "adaptive_support": cand.get("adaptive_support", "No"),
                "description": cand.get("description", ""),
                "duration": duration,
                "remote_support": cand.get("remote_support", "No"),
                "test_type": cand.get("test_type", []),
                # Keep scores internally if you want them for debugging
                "rerank_score": score,
            }
            reranked.append(cand_copy)

        # 3) Sort by rerank_score
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 4) Take top_k and strip internal scoring fields
        final: List[Dict[str, Any]] = []
        for item in reranked[:top_k]:
            item.pop("rerank_score", None)
            final.append(item)

        return final