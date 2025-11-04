"""
Multi-Agent System Components

This module contains the three main agents:
- DataAgent: Retrieves data from IUCN, GBIF, and news sources
- AnalysisAgent: Performs LLM-based reasoning and threat detection
- ReportAgent: Generates structured threat summaries
"""

from typing import Dict, Any, List
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("biodiversity_intel.agents")


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
        logger.info(f"Initialized {self.name}")

    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task."""
        pass


class DataAgent(BaseAgent):
    """
    Agent responsible for retrieving data from multiple sources.

    Data sources:
    - IUCN Red List API
    - GBIF Occurrence API
    - Conservation news sources
    """

    def __init__(self):
        super().__init__("Data Agent")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve data for a given species."""
        from biodiversity_intel.data_sources import IUCNClient, GBIFClient, MongabayClient

        species_name = state.get("species_name", "Unknown")
        logger.info(f"DataAgent: Starting data retrieval for species '{species_name}'")

        try:
            # Initialize API clients
            logger.debug(f"DataAgent: Initializing API clients")
            iucn_client = IUCNClient()
            gbif_client = GBIFClient()
            news_client = MongabayClient()

            # Fetch IUCN data
            logger.debug(f"DataAgent: Fetching IUCN data for {species_name}")
            iucn_data = iucn_client.get_species_data(species_name)
            if iucn_data:
                state["iucn_data"] = iucn_data.model_dump()
                logger.info(f"DataAgent: IUCN data retrieved - Status: {iucn_data.conservation_status}, Threats: {len(iucn_data.threats)}")
            else:
                state["iucn_data"] = {}
                logger.warning(f"DataAgent: No IUCN data found for {species_name}")

            # Fetch GBIF data
            logger.debug(f"DataAgent: Fetching GBIF data for {species_name}")
            gbif_data = gbif_client.get_occurrences(species_name)
            if gbif_data:
                state["gbif_data"] = gbif_data.model_dump()
                logger.info(f"DataAgent: GBIF data retrieved - Occurrences: {gbif_data.occurrence_count}")
            else:
                state["gbif_data"] = {}
                logger.warning(f"DataAgent: No GBIF data found for {species_name}")

            # Fetch news data
            logger.debug(f"DataAgent: Fetching news data for {species_name}")
            news_articles = news_client.search_species_news(species_name, max_articles=5)
            if news_articles:
                state["news_data"] = news_articles
                logger.info(f"DataAgent: News data retrieved - Articles: {len(news_articles)}")
            else:
                state["news_data"] = []
                logger.warning(f"DataAgent: No news articles found for {species_name}")

            logger.info(f"DataAgent: Successfully completed data retrieval for '{species_name}'")
            return state

        except Exception as e:
            logger.error(f"DataAgent: Error retrieving data for '{species_name}': {e}", exc_info=True)
            raise


class AnalysisAgent(BaseAgent):
    """
    Agent responsible for LLM-based reasoning and analysis.

    Tasks:
    - Cross-check population trends
    - Extract potential threats
    - Validate consistency across sources
    """

    def __init__(self):
        super().__init__("Analysis Agent")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collected data using LLM."""
        from biodiversity_intel.llm import LLMClient, ANALYSIS_PROMPT
        import json

        species_name = state.get("species_name", "Unknown")
        logger.info(f"AnalysisAgent: Starting analysis for species '{species_name}'")

        try:
            # Get collected data
            iucn_data = state.get("iucn_data", {})
            gbif_data = state.get("gbif_data", {})
            news_data = state.get("news_data", [])

            # Format data for LLM analysis
            logger.debug(f"AnalysisAgent: Formatting data for LLM analysis")
            iucn_summary = json.dumps(iucn_data, indent=2) if iucn_data else "No IUCN data available"
            gbif_summary = json.dumps(gbif_data, indent=2) if gbif_data else "No GBIF data available"

            # Format news data
            if news_data:
                news_summary = "\n".join([
                    f"- {article['title']}\n  URL: {article['url']}\n  Summary: {article['summary']}"
                    for article in news_data
                ])
                logger.debug(f"AnalysisAgent: Formatted {len(news_data)} news articles")
            else:
                news_summary = "No recent conservation news articles available"

            # Initialize LLM client
            logger.debug(f"AnalysisAgent: Initializing LLM client")
            llm_client = LLMClient()

            # Create analysis prompt
            logger.debug(f"AnalysisAgent: Creating analysis prompt")
            prompt = ANALYSIS_PROMPT.format(
                species_name=species_name,
                iucn_data=iucn_summary,
                gbif_data=gbif_summary,
                news_data=news_summary
            )

            # Generate analysis
            logger.info(f"AnalysisAgent: Requesting LLM analysis for '{species_name}'")
            analysis_result = await llm_client.generate(prompt)
            state["analysis"] = analysis_result

            # Extract key information from IUCN data for state
            if iucn_data:
                state["conservation_status"] = iucn_data.get("conservation_status", "Unknown")
                state["population_trend"] = iucn_data.get("population_trend", "Unknown")
                state["threats"] = iucn_data.get("threats", [])
                state["assessment_date"] = iucn_data.get("assessment_date")
                logger.info(f"AnalysisAgent: Extracted {len(state['threats'])} threats from IUCN data")
            else:
                state["conservation_status"] = "Unknown"
                state["population_trend"] = "Unknown"
                state["threats"] = []
                state["assessment_date"] = None
                logger.warning(f"AnalysisAgent: No IUCN data to extract conservation status")

            # Calculate confidence score based on data availability
            confidence_factors = []
            if iucn_data:
                confidence_factors.append(0.5)  # IUCN data present
            if gbif_data and gbif_data.get("occurrence_count", 0) > 0:
                confidence_factors.append(0.3)  # GBIF data present
            if state.get("news_data"):
                confidence_factors.append(0.2)  # News data present

            state["confidence_score"] = sum(confidence_factors)
            logger.info(f"AnalysisAgent: Confidence score: {state['confidence_score']:.2f}")

            # Detect early warning signals
            early_warnings = []
            if state.get("population_trend", "").lower() == "decreasing":
                early_warnings.append("Population declining")
            if state.get("conservation_status") in ["CR", "EN"]:
                early_warnings.append(f"Critical conservation status: {state.get('conservation_status')}")
            if len(state.get("threats", [])) > 10:
                early_warnings.append(f"High threat count: {len(state.get('threats', []))}")

            state["early_warnings"] = early_warnings
            if early_warnings:
                logger.warning(f"AnalysisAgent: Early warnings detected: {', '.join(early_warnings)}")

            logger.info(f"AnalysisAgent: Successfully completed analysis for '{species_name}'")
            return state

        except Exception as e:
            logger.error(f"AnalysisAgent: Error during analysis for '{species_name}': {e}", exc_info=True)
            raise


class ReportAgent(BaseAgent):
    """
    Agent responsible for generating structured reports.

    Output includes:
    - Detected threats
    - Population directionality
    - Confidence scores
    - Early warning signals
    """

    def __init__(self):
        super().__init__("Report Agent")

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate threat summary report."""
        from biodiversity_intel.llm import LLMClient, REPORT_PROMPT

        species_name = state.get("species_name", "Unknown")
        logger.info(f"ReportAgent: Starting report generation for species '{species_name}'")

        try:
            # Get analysis results
            analysis = state.get("analysis", "No analysis available")

            # Initialize LLM client
            logger.debug(f"ReportAgent: Initializing LLM client")
            llm_client = LLMClient()

            # Create report prompt
            logger.debug(f"ReportAgent: Creating report prompt")
            prompt = REPORT_PROMPT.format(
                species_name=species_name,
                analysis_results=analysis
            )

            # Generate structured report
            logger.info(f"ReportAgent: Requesting LLM report generation for '{species_name}'")
            report = await llm_client.generate(prompt)
            state["report"] = report

            # Create a summary for quick reference
            logger.debug(f"ReportAgent: Creating summary")
            summary_parts = []
            summary_parts.append(f"Species: {species_name}")
            summary_parts.append(f"Conservation Status: {state.get('conservation_status', 'Unknown')}")
            summary_parts.append(f"Population Trend: {state.get('population_trend', 'Unknown')}")
            summary_parts.append(f"Threats Identified: {len(state.get('threats', []))}")
            summary_parts.append(f"Confidence Score: {state.get('confidence_score', 0.0):.2f}")

            if state.get("early_warnings"):
                summary_parts.append(f"Early Warnings: {', '.join(state.get('early_warnings', []))}")

            state["summary"] = "\n".join(summary_parts)
            logger.info(f"ReportAgent: Summary created")

            logger.info(f"ReportAgent: Successfully generated report for '{species_name}'")
            logger.debug(f"ReportAgent: Report length: {len(report)} chars")

            return state

        except Exception as e:
            logger.error(f"ReportAgent: Error generating report for '{species_name}': {e}", exc_info=True)
            raise
