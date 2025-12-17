import re
from typing import Dict, List, Optional


class QueryProcessor:
    """
    Extracts structured signals from the raw query:
      - skills / technologies
      - test types
      - duration constraints
    And returns an enhanced query string plus a signals dict.
    """

    def __init__(self) -> None:
        # Simple keyword lists – you can expand these over time
        self.skill_keywords = [
            "sql",
            "python",
            "excel",
            "java",
            "javascript",
            "js",
            "html",
            "css",
            "selenium",
            "qa",
            "quality assurance",
            "marketing",
            "sales",
            "analyst",
            "data analyst",
            "consultant",
            "manager",
            "coo",
            "graduate",
            "admin",
        ]

        # Based on SHL's test types
        self.test_type_keywords = [
            "ability & aptitude",
            "biodata & situational judgement",
            "competencies",
            "development & 360",
            "assessment exercises",
            "knowledge & skills",
            "personality & behavior",
            "personality & behaviour",
            "simulations",
        ]

    # ---------- Main entry point ----------

    def process_query(self, query: str) -> Dict:
        """
        Given a raw query string, return:
          - enhanced_query: string to embed
          - extracted_skills: [str]
          - extracted_test_types: [str]
          - max_duration_minutes: Optional[int]
        """
        q_lower = query.lower()

        # 1. Extract skills
        extracted_skills = self._extract_skills(q_lower)

        # 2. Extract test types
        extracted_test_types = self._extract_test_types(q_lower)

        # 3. Extract duration constraints
        max_duration_minutes = self._extract_duration(q_lower)

        # 4. Build enhanced query
        enhanced_parts: List[str] = [query]

        if extracted_skills:
            enhanced_parts.append("Required skills: " + ", ".join(extracted_skills))

        if extracted_test_types:
            enhanced_parts.append("Desired test types: " + ", ".join(extracted_test_types))

        if max_duration_minutes is not None:
            enhanced_parts.append(f"Maximum duration: {max_duration_minutes} minutes")

        enhanced_query = ". ".join(enhanced_parts)

        return {
            "enhanced_query": enhanced_query,
            "extracted_skills": extracted_skills,
            "extracted_test_types": extracted_test_types,
            "max_duration_minutes": max_duration_minutes,
        }

    # ---------- Helpers ----------

    def _extract_skills(self, q_lower: str) -> List[str]:
        skills: List[str] = []
        for skill in self.skill_keywords:
            if skill in q_lower:
                skills.append(skill)
        return skills

    def _extract_test_types(self, q_lower: str) -> List[str]:
        tts: List[str] = []
        for tt in self.test_type_keywords:
            if tt in q_lower:
                tts.append(tt)
        return tts

    def _extract_duration(self, q_lower: str) -> Optional[int]:
        """
        Extract something like:
          - "40 minutes"
          - "1 hour"
          - "1-2 hour"
          - "at most 90 mins"
        and return a max duration in minutes, if found.
        """
        # Explicit "at most X min"
        m = re.search(r"at\s+most\s+(\d+)\s*(min|mins|minute|minutes)", q_lower)
        if m:
            return int(m.group(1))

        # Ranges like "1-2 hour"
        m = re.search(r"(\d+)\s*-\s*(\d+)\s*hour", q_lower)
        if m:
            # e.g., 1-2 hours -> use upper bound
            return int(m.group(2)) * 60

        # Simple "X hour(s)"
        m = re.search(r"(\d+)\s*(hour|hours)", q_lower)
        if m:
            return int(m.group(1)) * 60

        # Simple "X min/mins/minute/minutes"
        m = re.search(r"(\d+)\s*(min|mins|minute|minutes)", q_lower)
        if m:
            return int(m.group(1))

        return None