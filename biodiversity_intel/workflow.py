"""
LangGraph Workflow Orchestration

This module defines the multi-agent workflow using LangGraph.
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import logging
import json

logger = logging.getLogger("biodiversity_intel.workflow")


class ConservationState(TypedDict):
    """State object passed between agents."""
    species_name: str
    iucn_data: Dict[str, Any]
    gbif_data: Dict[str, Any]
    news_data: list
    analysis: str
    threats: list
    population_trend: str
    confidence_score: float
    early_warning: bool
    report: str


def build_conservation_workflow() -> StateGraph:
    """
    Build the LangGraph workflow for conservation intelligence.

    Returns:
        Configured StateGraph
    """
    logger.info("Building conservation workflow")
    from biodiversity_intel.agents import DataAgent, AnalysisAgent, ReportAgent

    # Initialize agents
    logger.debug("Initializing workflow agents")
    data_agent = DataAgent()
    analysis_agent = AnalysisAgent()
    report_agent = ReportAgent()

    # Create state graph
    logger.debug("Creating StateGraph with ConservationState")
    workflow = StateGraph(ConservationState)

    # Add nodes
    logger.debug("Adding workflow nodes: data_collection, analysis, report_generation")
    workflow.add_node("data_collection", data_agent.execute)
    workflow.add_node("analysis", analysis_agent.execute)
    workflow.add_node("report_generation", report_agent.execute)

    # Define edges (workflow flow)
    logger.debug("Defining workflow edges")
    workflow.set_entry_point("data_collection")
    workflow.add_edge("data_collection", "analysis")
    workflow.add_edge("analysis", "report_generation")
    workflow.add_edge("report_generation", END)

    logger.info("Conservation workflow built successfully")
    return workflow.compile()


async def run_conservation_analysis(species_name: str) -> Dict[str, Any]:
    """
    Run complete conservation analysis workflow.

    Args:
        species_name: Scientific name of the species to analyze

    Returns:
        Complete analysis results with report
    """
    logger.info(f"Starting conservation analysis workflow for species '{species_name}'")

    try:
        workflow = build_conservation_workflow()

        logger.debug(f"Initializing workflow state for '{species_name}'")
        initial_state = ConservationState(
            species_name=species_name,
            iucn_data={},
            gbif_data={},
            news_data=[],
            analysis="",
            threats=[],
            population_trend="unknown",
            confidence_score=0.0,
            early_warning=False,
            report=""
        )

        logger.info(f"Invoking workflow for '{species_name}'")
        result = await workflow.ainvoke(initial_state)

        with open('trash_workflow_results.json', 'w') as f:
            json.dump(result, f)

        logger.info(f"Workflow completed successfully for '{species_name}'")
        logger.debug(f"Final state: threats={result.get('threats', [])}, trend={result.get('population_trend', 'unknown')}, confidence={result.get('confidence_score', 0.0)}")

        return result
    except Exception as e:
        logger.error(f"Workflow failed for species '{species_name}': {e}", exc_info=True)
        raise
