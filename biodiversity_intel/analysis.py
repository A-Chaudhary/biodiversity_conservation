"""
Analysis and Evaluation Modules

This module contains:
- Threat detection logic
- Population trend analysis
- Cross-source consistency checking
- Confidence scoring
- Evaluation metrics
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel


class ThreatAssessment(BaseModel):
    """Model for threat assessment results."""
    threats: List[str]
    population_trend: str
    confidence_score: float
    early_warning: bool
    sources: List[str]


class ThreatDetector:
    """Identifies threats from IUCN descriptions and other sources."""

    def __init__(self):
        # TODO: Load IUCN threat taxonomy
        self.threat_taxonomy = {}

    def detect_threats(self, iucn_data: Dict[str, Any], llm_analysis: str) -> List[str]:
        """
        Extract and categorize threats.

        Args:
            iucn_data: IUCN Red List data
            llm_analysis: LLM-generated threat analysis

        Returns:
            List of identified threats
        """
        # TODO: Implement threat detection logic
        pass


class TrendAnalyzer:
    """Analyzes population trends from GBIF occurrence data."""

    def analyze_trend(self, gbif_data: Dict[str, Any]) -> str:
        """
        Analyze temporal trend in occurrence data.

        Args:
            gbif_data: GBIF occurrence data with temporal distribution

        Returns:
            Trend direction: "increasing", "stable", "decreasing", or "unknown"
        """
        # TODO: Implement trend analysis logic
        pass


class ConsistencyChecker:
    """Cross-validates data from different sources."""

    def check_consistency(
        self,
        iucn_trend: str,
        gbif_trend: str
    ) -> Dict[str, Any]:
        """
        Check if IUCN and GBIF trends align.

        Args:
            iucn_trend: IUCN population trend assessment
            gbif_trend: GBIF-derived trend

        Returns:
            Dictionary with consistency flag and details
        """
        # TODO: Implement consistency checking logic
        pass


class ConfidenceScorer:
    """Calculates confidence scores for assessments."""

    def calculate_score(
        self,
        data_quality: Dict[str, float],
        source_agreement: float
    ) -> float:
        """
        Calculate normalized confidence score (0-1).

        Args:
            data_quality: Quality scores for each data source
            source_agreement: Level of agreement between sources

        Returns:
            Confidence score between 0 and 1
        """
        # TODO: Implement confidence scoring logic
        pass


class EvaluationMetrics:
    """Evaluation metrics for the system."""

    def threat_detection_recall(
        self,
        detected_threats: List[str],
        iucn_threats: List[str]
    ) -> float:
        """
        Calculate recall: ratio of IUCN threats identified.

        Args:
            detected_threats: Threats identified by the system
            iucn_threats: Official IUCN threat list

        Returns:
            Recall score between 0 and 1
        """
        if not iucn_threats:
            return 1.0

        matches = len(set(detected_threats) & set(iucn_threats))
        return matches / len(iucn_threats)

    def confidence_alignment_score(
        self,
        iucn_trend: str,
        gbif_trend: str
    ) -> float:
        """
        Measure consistency between IUCN and GBIF trends.

        Args:
            iucn_trend: IUCN population trend (e.g., "Increasing", "Decreasing", "Stable", "Unknown")
            gbif_trend: GBIF-derived trend (e.g., "increasing", "decreasing", "stable", "unknown")

        Returns:
            Alignment score between 0 and 1:
            - 1.0: Perfect match (both increasing, both decreasing, both stable)
            - 0.5: Partial alignment (one unknown, one known)
            - 0.0: Direct contradiction (one increasing, other decreasing)
        """
        if not iucn_trend or not gbif_trend:
            return 0.0
        
        # Normalize trends to lowercase for comparison
        iucn_normalized = iucn_trend.lower().strip()
        gbif_normalized = gbif_trend.lower().strip()
        
        # Handle exact matches
        if iucn_normalized == gbif_normalized:
            # Both unknown is ambiguous - return 0.5 (neutral)
            if iucn_normalized in ["unknown", "uncertain", "not assessed"]:
                return 0.5
            return 1.0
        
        # Handle cases where one is unknown
        if iucn_normalized in ["unknown", "uncertain", "not assessed"]:
            # If IUCN is unknown but GBIF has data, partial confidence
            if gbif_normalized not in ["unknown", "uncertain", "not assessed"]:
                return 0.5
            return 0.5  # Both unknown
        
        if gbif_normalized in ["unknown", "uncertain", "not assessed"]:
            # If GBIF is unknown but IUCN has assessment, partial confidence
            return 0.5
        
        # Define trend categories
        increasing_keywords = ["increasing", "growing", "expanding", "rising"]
        decreasing_keywords = ["decreasing", "declining", "shrinking", "falling", "reducing"]
        stable_keywords = ["stable", "constant", "unchanged", "steady"]
        
        # Categorize trends
        iucn_category = None
        gbif_category = None
        
        if any(kw in iucn_normalized for kw in increasing_keywords):
            iucn_category = "increasing"
        elif any(kw in iucn_normalized for kw in decreasing_keywords):
            iucn_category = "decreasing"
        elif any(kw in iucn_normalized for kw in stable_keywords):
            iucn_category = "stable"
        
        if any(kw in gbif_normalized for kw in increasing_keywords):
            gbif_category = "increasing"
        elif any(kw in gbif_normalized for kw in decreasing_keywords):
            gbif_category = "decreasing"
        elif any(kw in gbif_normalized for kw in stable_keywords):
            gbif_category = "stable"
        
        # If we couldn't categorize either, return neutral score
        if iucn_category is None or gbif_category is None:
            return 0.5
        
        # Perfect match
        if iucn_category == gbif_category:
            return 1.0
        
        # Direct contradiction (increasing vs decreasing)
        if (iucn_category == "increasing" and gbif_category == "decreasing") or \
           (iucn_category == "decreasing" and gbif_category == "increasing"):
            return 0.0
        
        # Partial alignment (stable with increasing/decreasing)
        # Stable trend is somewhat consistent with either direction
        if iucn_category == "stable" or gbif_category == "stable":
            return 0.7  # Moderate alignment
        
        # Default: no clear relationship
        return 0.5
