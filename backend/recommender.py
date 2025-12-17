from typing import List, Dict, Any

from embedder import Embedder, HAS_ST
from query_processor import QueryProcessor


class Recommender:
    """
    Uses:
      - QueryProcessor to understand the query
      - Embedder to get initial candidates (FAISS if available, keyword fallback otherwise)
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

        # 1) Initial retrieval
        if HAS_ST:
            # Local environment: use semantic FAISS search
            query_vector = self.embedder.embed_query(enhanced_query)
            candidates = self.embedder.search(
                query_vector, top_k=self.candidate_pool_size
            )
        else:
            # Render environment (no ST): use keyword-based fallback
            candidates = self._keyword_search(
                enhanced_query, 
                extracted_skills, 
                extracted_test_types,
                top_k=self.candidate_pool_size
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
            # Optionally keep score for debugging
            item["score"] = item.pop("rerank_score", 0.0)
            final.append(item)

        return final

    def _keyword_search(
        self, 
        query: str, 
        skills: List[str], 
        test_types: List[str],
        top_k: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Simple keyword-based scoring fallback for environments without SentenceTransformer.
        Scores assessments based on keyword matches in name and description.
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_assessments = []
        
        for assessment in self.embedder.metadata:
            name = (assessment.get("name") or "").lower()
            desc = (assessment.get("description") or "").lower()
            test_type_list = [t.lower() for t in assessment.get("test_type", [])]
            
            # Base score: count matching words
            name_words = set(name.split())
            desc_words = set(desc.split())
            
            word_matches = len(query_words & (name_words | desc_words))
            score = float(word_matches)
            
            # Boost for skill matches
            for skill in skills:
                if skill.lower() in name or skill.lower() in desc:
                    score += 2.0
            
            # Boost for test type matches
            for tt in test_types:
                if any(tt.lower() in test_t for test_t in test_type_list):
                    score += 3.0
            
            if score > 0:
                assessment_copy = dict(assessment)
                assessment_copy["score"] = score
                scored_assessments.append(assessment_copy)
        
        # Sort by score and return top_k
        scored_assessments.sort(key=lambda x: x["score"], reverse=True)
        return scored_assessments[:top_k]