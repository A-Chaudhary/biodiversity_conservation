"""
Test script for agent implementations
"""

import asyncio
import logging
from biodiversity_intel.config import config, setup_logging
from biodiversity_intel.agents import DataAgent, AnalysisAgent, ReportAgent

# Setup logging
logger = setup_logging("INFO")

async def test_agents():
    """Test all three agents with a sample species."""

    print("\n" + "=" * 80)
    print("  Testing Agent Implementations")
    print("=" * 80 + "\n")

    # Test species
    species_name = "Panthera tigris"
    print(f"Testing with species: {species_name}\n")

    # Initialize state
    state = {"species_name": species_name}

    try:
        # Test DataAgent
        print("Step 1: DataAgent - Retrieving data from APIs...")
        print("-" * 80)
        data_agent = DataAgent()
        state = await data_agent.execute(state)
        print(f"[PASS] DataAgent completed")
        print(f"   IUCN data: {'[OK]' if state.get('iucn_data') else '[X]'}")
        print(f"   GBIF data: {'[OK]' if state.get('gbif_data') else '[X]'}")
        print(f"   News data: {'[OK]' if state.get('news_data') else '[X]'}")
        print()

        # Test AnalysisAgent
        print("Step 2: AnalysisAgent - Analyzing data with LLM...")
        print("-" * 80)
        analysis_agent = AnalysisAgent()
        state = await analysis_agent.execute(state)
        print(f"[PASS] AnalysisAgent completed")
        print(f"   Conservation Status: {state.get('conservation_status', 'N/A')}")
        print(f"   Population Trend: {state.get('population_trend', 'N/A')}")
        print(f"   Threats Count: {len(state.get('threats', []))}")
        print(f"   Confidence Score: {state.get('confidence_score', 0.0):.2f}")
        if state.get('early_warnings'):
            print(f"   Early Warnings: {', '.join(state.get('early_warnings', []))}")
        print()

        # Test ReportAgent
        print("Step 3: ReportAgent - Generating structured report...")
        print("-" * 80)
        report_agent = ReportAgent()
        state = await report_agent.execute(state)
        print(f"[PASS] ReportAgent completed")
        print(f"   Report generated: {len(state.get('report', ''))} chars")
        print()

        # Display results
        print("=" * 80)
        print("  FINAL RESULTS")
        print("=" * 80 + "\n")

        print("Summary:")
        print("-" * 80)
        print(state.get('summary', 'No summary available'))
        print()

        print("Full Report:")
        print("-" * 80)
        print(state.get('report', 'No report available'))
        print()

        print("=" * 80)
        print("  [PASS] All agents executed successfully!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n[FAIL] Error during agent execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Fix Windows console encoding
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    asyncio.run(test_agents())
