"""
Test script to verify MCP server functionality without actually running it as an MCP server.

This tests all the handler functions directly to ensure they work correctly.
"""

import asyncio
import json


async def test_mcp_handlers():
    """Test all MCP server handlers."""
    print("=" * 80)
    print("Testing MCP Server Handlers")
    print("=" * 80)

    # Import the handlers
    from mcp_server import (
        handle_full_analysis,
        handle_iucn_data,
        handle_gbif_data,
        handle_news_data,
        handle_threat_classifications
    )

    # Test 1: Threat Classifications (no species required)
    print("\n[Test 1] Testing get_threat_classifications...")
    try:
        result = await handle_threat_classifications({})
        data = json.loads(result[0].text)
        print(f"[OK] Retrieved {data['total_classifications']} threat classifications")
        print(f"     Categories: {len(data['categories'])}")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 2: IUCN Data
    print("\n[Test 2] Testing get_iucn_data for Panthera tigris...")
    try:
        result = await handle_iucn_data({"species_name": "Panthera tigris"})
        data = json.loads(result[0].text)
        print(f"[OK] IUCN Data Retrieved:")
        print(f"     Scientific Name: {data['scientific_name']}")
        print(f"     Conservation Status: {data['conservation_status']}")
        print(f"     Population Trend: {data['population_trend']}")
        print(f"     Threats: {data['threats']['total_count']}")
        if data['threats']['total_count'] > 0:
            print(f"     First Threat: {data['threats']['details'][0]['mapped_name']}")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 3: GBIF Data
    print("\n[Test 3] Testing get_gbif_occurrences for Panthera tigris...")
    try:
        result = await handle_gbif_data({"species_name": "Panthera tigris", "limit": 1000})
        data = json.loads(result[0].text)
        print(f"[OK] GBIF Data Retrieved:")
        print(f"     Scientific Name: {data['scientific_name']}")
        print(f"     Occurrence Count: {data['occurrence_count']}")
        print(f"     Years Covered: {data['temporal_summary']['years_covered']}")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 4: Conservation News
    print("\n[Test 4] Testing get_conservation_news for tiger...")
    try:
        result = await handle_news_data({"species_name": "tiger", "max_articles": 5})
        data = json.loads(result[0].text)
        print(f"[OK] News Data Retrieved:")
        print(f"     Species: {data['species_name']}")
        print(f"     Articles Found: {data['article_count']}")
        if data['article_count'] > 0:
            print(f"     First Article: {data['articles'][0]['title'][:60]}...")
    except Exception as e:
        print(f"[ERROR] {e}")

    # Test 5: Full Analysis (this takes longer)
    print("\n[Test 5] Testing analyze_species_conservation for Panthera tigris...")
    print("     (This may take 30-60 seconds with LLM calls...)")
    try:
        result = await handle_full_analysis({"species_name": "Panthera tigris"})
        data = json.loads(result[0].text)
        print(f"[OK] Full Analysis Completed:")
        print(f"     Species: {data['species_name']}")
        print(f"     Conservation Status: {data['conservation_status']}")
        print(f"     Population Trend: {data['population_trend']}")
        print(f"     Threats: {data['threats']['total_count']}")
        print(f"     Confidence Score: {data['confidence_score']:.2%}")
        print(f"     Early Warnings: {len(data.get('early_warnings', []))}")
        print(f"     Analysis Length: {len(data.get('analysis', ''))} chars")
        print(f"     Report Length: {len(data.get('report', ''))} chars")
    except Exception as e:
        print(f"[ERROR] {e}")

    print("\n" + "=" * 80)
    print("MCP Server Handler Tests Complete!")
    print("=" * 80)
    print("\nIf all tests passed, your MCP server is ready to use with Claude Desktop!")
    print("\nNext steps:")
    print("1. Copy claude_desktop_config.json content to your Claude Desktop config")
    print("2. Update the 'cwd' path to match your system")
    print("3. Restart Claude Desktop")


if __name__ == "__main__":
    asyncio.run(test_mcp_handlers())
