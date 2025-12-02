"""
MCP Server for Biodiversity Conservation Intelligence System

This MCP server exposes the biodiversity intelligence agents as tools
that can be used by Claude Desktop or other MCP clients.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

# MCP SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Biodiversity Intelligence imports
from biodiversity_intel.workflow import run_conservation_analysis
from biodiversity_intel.data_sources import IUCNClient, GBIFClient, MongabayClient
from biodiversity_intel.config import setup_logging, config

# Setup logging
logger = setup_logging(config.log_level)
mcp_logger = logging.getLogger("biodiversity_intel.mcp_server")

# Initialize MCP server
app = Server("biodiversity-conservation-intel")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools for the MCP server."""
    return [
        Tool(
            name="analyze_species_conservation",
            description=(
                "Run a complete conservation analysis for a species. This performs multi-agent workflow "
                "that retrieves IUCN Red List data, GBIF occurrence records, conservation news, and "
                "generates an AI-powered threat assessment with both threat codes and human-readable names. "
                "Returns comprehensive analysis including conservation status, population trends, threats, "
                "confidence scores, and detailed reports."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "species_name": {
                        "type": "string",
                        "description": "Scientific name of the species (e.g., 'Panthera tigris', 'Ailuropoda melanoleuca')"
                    }
                },
                "required": ["species_name"]
            }
        ),
        Tool(
            name="get_iucn_data",
            description=(
                "Retrieve IUCN Red List data for a species including conservation status, population trend, "
                "threats with both codes and mapped names, assessment history, and assessment date. "
                "This is faster than the full analysis and focuses only on IUCN data."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "species_name": {
                        "type": "string",
                        "description": "Scientific name of the species"
                    }
                },
                "required": ["species_name"]
            }
        ),
        Tool(
            name="get_gbif_occurrences",
            description=(
                "Retrieve GBIF occurrence data for a species including occurrence count, "
                "temporal distribution (by year), and spatial distribution. Useful for understanding "
                "species distribution patterns and population data over time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "species_name": {
                        "type": "string",
                        "description": "Scientific name of the species"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of occurrences to retrieve (default: 10000)",
                        "default": 10000
                    }
                },
                "required": ["species_name"]
            }
        ),
        Tool(
            name="get_conservation_news",
            description=(
                "Search for recent conservation news articles about a species from Mongabay. "
                "Returns article titles, URLs, summaries, and publication dates. Useful for "
                "understanding current conservation efforts and threats."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "species_name": {
                        "type": "string",
                        "description": "Scientific or common name of the species"
                    },
                    "max_articles": {
                        "type": "integer",
                        "description": "Maximum number of articles to retrieve (default: 20)",
                        "default": 20
                    }
                },
                "required": ["species_name"]
            }
        ),
        Tool(
            name="get_threat_classifications",
            description=(
                "Retrieve the complete IUCN threat classification mapping. Returns a dictionary "
                "mapping threat codes (e.g., '5_1_3') to their human-readable descriptions "
                "(e.g., 'Persecution/control'). Useful for understanding threat categories."
            ),
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls from the MCP client."""
    try:
        mcp_logger.info(f"MCP: Tool called: {name} with arguments: {arguments}")

        if name == "analyze_species_conservation":
            return await handle_full_analysis(arguments)

        elif name == "get_iucn_data":
            return await handle_iucn_data(arguments)

        elif name == "get_gbif_occurrences":
            return await handle_gbif_data(arguments)

        elif name == "get_conservation_news":
            return await handle_news_data(arguments)

        elif name == "get_threat_classifications":
            return await handle_threat_classifications(arguments)

        else:
            error_msg = f"Unknown tool: {name}"
            mcp_logger.error(f"MCP: {error_msg}")
            return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

    except Exception as e:
        error_msg = f"Error executing tool {name}: {str(e)}"
        mcp_logger.error(f"MCP: {error_msg}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({
            "error": error_msg,
            "type": type(e).__name__
        }))]


async def handle_full_analysis(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle the analyze_species_conservation tool."""
    species_name = arguments.get("species_name")

    if not species_name:
        return [TextContent(type="text", text=json.dumps({
            "error": "species_name is required"
        }))]

    mcp_logger.info(f"MCP: Running full conservation analysis for '{species_name}'")

    # Run the conservation analysis workflow
    result = await run_conservation_analysis(species_name)

    # Format the response for better readability
    formatted_result = {
        "species_name": result.get("species_name"),
        "conservation_status": result.get("conservation_status", "Unknown"),
        "population_trend": result.get("population_trend", "Unknown"),
        "assessment_date": result.get("assessment_date"),
        "threats": {
            "total_count": len(result.get("threat_details", [])),
            "details": result.get("threat_details", [])
        },
        "confidence_score": result.get("confidence_score", 0.0),
        "early_warnings": result.get("early_warnings", []),
        "data_sources": {
            "iucn_available": bool(result.get("iucn_data")),
            "gbif_occurrences": result.get("gbif_data", {}).get("occurrence_count", 0),
            "news_articles": len(result.get("news_data", []))
        },
        "analysis": result.get("analysis", ""),
        "report": result.get("report", ""),
        "summary": result.get("summary", "")
    }

    mcp_logger.info(f"MCP: Analysis completed for '{species_name}'")
    return [TextContent(type="text", text=json.dumps(formatted_result, indent=2))]


async def handle_iucn_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle the get_iucn_data tool."""
    species_name = arguments.get("species_name")

    if not species_name:
        return [TextContent(type="text", text=json.dumps({
            "error": "species_name is required"
        }))]

    mcp_logger.info(f"MCP: Fetching IUCN data for '{species_name}'")

    # Initialize IUCN client
    iucn_client = IUCNClient()

    # Fetch data in executor to avoid blocking
    loop = asyncio.get_event_loop()
    iucn_data = await loop.run_in_executor(None, iucn_client.get_species_data, species_name)

    if not iucn_data:
        return [TextContent(type="text", text=json.dumps({
            "error": f"No IUCN data found for species '{species_name}'"
        }))]

    # Get threat classifications
    threats_mapping = iucn_client.get_threats_mapping()

    # Enrich threat details with mapped names
    enriched_threats = []
    for threat in iucn_data.threat_details:
        code = threat.get("code", "")
        name = threat.get("name", "")
        mapped_name = threats_mapping.get(code, name)

        enriched_threats.append({
            "code": code,
            "name": name,
            "mapped_name": mapped_name
        })

    result = {
        "scientific_name": iucn_data.scientific_name,
        "conservation_status": iucn_data.conservation_status,
        "population_trend": iucn_data.population_trend,
        "assessment_date": iucn_data.assessment_date,
        "threats": {
            "total_count": len(enriched_threats),
            "details": enriched_threats
        },
        "assessment_history": iucn_data.assessment_history[:10]  # Limit to 10 most recent
    }

    mcp_logger.info(f"MCP: IUCN data retrieved for '{species_name}'")
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_gbif_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle the get_gbif_occurrences tool."""
    species_name = arguments.get("species_name")
    limit = arguments.get("limit", 10000)

    if not species_name:
        return [TextContent(type="text", text=json.dumps({
            "error": "species_name is required"
        }))]

    mcp_logger.info(f"MCP: Fetching GBIF occurrences for '{species_name}' (limit: {limit})")

    # Initialize GBIF client
    gbif_client = GBIFClient()

    # Fetch data in executor to avoid blocking
    loop = asyncio.get_event_loop()
    gbif_data = await loop.run_in_executor(None, gbif_client.get_occurrences, species_name, limit)

    if not gbif_data:
        return [TextContent(type="text", text=json.dumps({
            "error": f"No GBIF data found for species '{species_name}'"
        }))]

    result = {
        "scientific_name": gbif_data.scientific_name,
        "occurrence_count": gbif_data.occurrence_count,
        "temporal_distribution": gbif_data.temporal_distribution,
        "temporal_summary": {
            "years_covered": len(gbif_data.temporal_distribution),
            "earliest_year": min(gbif_data.temporal_distribution.keys()) if gbif_data.temporal_distribution else None,
            "latest_year": max(gbif_data.temporal_distribution.keys()) if gbif_data.temporal_distribution else None
        }
    }

    mcp_logger.info(f"MCP: GBIF data retrieved for '{species_name}' - {gbif_data.occurrence_count} occurrences")
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_news_data(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle the get_conservation_news tool."""
    species_name = arguments.get("species_name")
    max_articles = arguments.get("max_articles", 20)

    if not species_name:
        return [TextContent(type="text", text=json.dumps({
            "error": "species_name is required"
        }))]

    mcp_logger.info(f"MCP: Fetching conservation news for '{species_name}' (max: {max_articles})")

    # Initialize Mongabay client
    news_client = MongabayClient()

    # Fetch data in executor to avoid blocking
    loop = asyncio.get_event_loop()
    articles = await loop.run_in_executor(
        None,
        news_client.search_species_news,
        species_name,
        max_articles
    )

    if not articles:
        return [TextContent(type="text", text=json.dumps({
            "message": f"No conservation news found for species '{species_name}'",
            "articles": []
        }))]

    result = {
        "species_name": species_name,
        "article_count": len(articles),
        "articles": articles
    }

    mcp_logger.info(f"MCP: Found {len(articles)} news articles for '{species_name}'")
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def handle_threat_classifications(arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle the get_threat_classifications tool."""
    mcp_logger.info("MCP: Fetching IUCN threat classifications")

    # Initialize IUCN client
    iucn_client = IUCNClient()

    # Get threat classifications
    threats_mapping = iucn_client.get_threats_mapping()

    # Organize by category
    categories = {}
    for code, description in threats_mapping.items():
        # Top-level category (e.g., "1", "2", "3")
        category_code = code.split('_')[0]
        if category_code not in categories:
            categories[category_code] = {
                "code": category_code,
                "name": threats_mapping.get(category_code, f"Category {category_code}"),
                "subcategories": []
            }

        # Only add non-top-level threats as subcategories
        if '_' in code:
            categories[category_code]["subcategories"].append({
                "code": code,
                "description": description
            })

    result = {
        "total_classifications": len(threats_mapping),
        "categories": list(categories.values()),
        "full_mapping": threats_mapping
    }

    mcp_logger.info(f"MCP: Retrieved {len(threats_mapping)} threat classifications")
    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Main entry point for the MCP server."""
    mcp_logger.info("Starting Biodiversity Conservation Intelligence MCP Server")

    async with stdio_server() as (read_stream, write_stream):
        mcp_logger.info("MCP Server: Connected via stdio")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
