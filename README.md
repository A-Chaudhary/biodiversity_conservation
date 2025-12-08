# Biodiversity Conservation Intelligence System

An agentic threat intelligence system leveraging large language models (LLMs) to synthesize information from multiple open biodiversity sources into coherent, explainable threat summaries.

## Table of Contents

- [Biodiversity Conservation Intelligence System](#biodiversity-conservation-intelligence-system)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
  - [System Architecture](#system-architecture)
  - [Technology Stack](#technology-stack)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup](#setup)
  - [Tutorial](#tutorial)
    - [Quick Start: Analyzing a Species](#quick-start-analyzing-a-species)
    - [Using the MCP Server with Claude Desktop](#using-the-mcp-server-with-claude-desktop)
    - [Running System Evaluation](#running-system-evaluation)
    - [Managing Cache](#managing-cache)
    - [Development and Testing](#development-and-testing)
  - [Project Structure](#project-structure)
  - [Development Timeline](#development-timeline)
  - [Evaluation Metrics](#evaluation-metrics)
    - [Quantitative Metrics](#quantitative-metrics)
    - [Performance Metrics](#performance-metrics)
    - [Evaluation Outputs](#evaluation-outputs)
  - [Data Sources](#data-sources)
  - [Performance Optimizations](#performance-optimizations)
    - [Parallel API Calls](#parallel-api-calls)
    - [Parallel GBIF Pagination](#parallel-gbif-pagination)
    - [File-Based Caching](#file-based-caching)
    - [GPU-Accelerated Anomaly Detection](#gpu-accelerated-anomaly-detection)
    - [Combined Impact](#combined-impact)
  - [Contributing](#contributing)
  - [Authors](#authors)
  - [Acknowledgments](#acknowledgments)
  - [License](#license)
  - [References](#references)
  - [Contact](#contact)

## Overview

Global biodiversity loss is accelerating, yet conservation intelligence systems remain fragmented and reactive. This project proposes a multi-agent LLM-based system that automates data retrieval, reasoning, and reporting for any queried species, enabling transparent, evidence-based ecological assessment at scale.

## Features

- **Multi-Agent Architecture**: Three specialized agents (Data, Analysis, Report) working collaboratively
- **Multi-Source Integration**: Combines IUCN Red List, GBIF occurrence data, and conservation news
- **Time-Series Anomaly Detection**: Chronos-based forecasting model identifies population anomalies, declines, and surges in occurrence data
- **Automated Threat Detection**: Identifies threats and inconsistencies across data sources
- **Explainable AI**: Generates transparent, evidence-based conservation insights with anomaly correlation
- **Streamlit Interface**: Interactive web app for species queries and visualization
- **Comprehensive Evaluation**: Quantitative metrics including threat detection recall, precision, F1-score, and confidence alignment scoring
- **Automated Evaluation Script**: Run systematic evaluations on test species sets with detailed metrics and statistics
- **Performance Optimizations**:
  - Parallel API calls for multi-source data retrieval (40-50% faster)
  - Parallel pagination for GBIF occurrence data (5x faster for large datasets)
  - File-based caching system (96% faster on repeated queries)
  - GPU-accelerated anomaly detection (when available)

## System Architecture

```
User Query (Species Name)
        │
        ▼
   Streamlit App ◄─────────────────────────────────────────┐
     (app.py)                                              │
        │                                                  │
        ▼                                                  │
  LangGraph Workflow                                       │
   (workflow.py)                                           │
        │                                                  │
        ├─► Data Agent ──────► IUCN API ──────────┐        │
        │   (agents.py)    ┌─► GBIF API ──────────┤        │
        │                  └─► News Sources ──────┤        │
        │                                         │        │
        ▼                                         ▼        │
   Analysis Agent ──────► Anomaly Detection ─────────────► │
   (agents.py)      │     (anomaly_detection.py)           │
        │           │     Chronos Model                    │
        │           │            │                         │
        │           └──────► LLM Client ───────────────────┤
        │                   (llm.py)                       │
        ▼                                                  │
   Report Agent                                            │
   (agents.py)                                             │
        │                                                  │
        ├──────────────► LLM Client ───────────────────────┤
        │                (llm.py)                          │
        ▼                                                  │
   Storage & Output ───────────────────────────────────────┘
    (storage.py)
```

**Key Components:**
- [`app.py`](app.py) - Streamlit web interface
- [`workflow.py`](biodiversity_intel/workflow.py) - LangGraph orchestration
- [`agents.py`](biodiversity_intel/agents.py) - Multi-agent system
- [`anomaly_detection.py`](biodiversity_intel/anomaly_detection.py) - Chronos-based forecasting
- [`llm.py`](biodiversity_intel/llm.py) - LLM client integration
- [`storage.py`](biodiversity_intel/storage.py) - Caching system

## Technology Stack

- **Python**: >=3.10
- **Package Management**: uv
- **Agent Framework**: LangGraph
- **LLM**: OpenAI GPT-4o-mini
- **Time-Series Forecasting**: Chronos (Amazon, pretrained transformer model)
- **Deep Learning**: PyTorch (with automatic GPU/CPU detection)
- **Frontend**: Streamlit
- **Data Storage**: JSON file-based caching
- **Testing**: pytest

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/A-Chaudhary/biodiversity_conservation.git
   cd biodiversity_conservation
   ```

2. **Install uv (if not already installed)**
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create virtual environment and install dependencies**
   ```bash
   uv venv

   # Activate virtual environment
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate

   # Install dependencies
   uv pip install -e ".[dev]"
   ```

4. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your OpenAI API key
   # - OPENAI_API_KEY (required)
   # - IUCN_API_TOKEN (required)
   ```

5. **Optional: Configure MCP Server**

   The Model Context Protocol (MCP) server ([`mcp_server.py`](mcp_server.py)) exposes biodiversity data and analysis functions to AI assistants like Claude Desktop.

   To use the MCP server, add this configuration to your Claude Desktop config file:

   **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

   **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

   ```json
   {
     "mcpServers": {
       "biodiversity-conservation-intel": {
         "command": "/path/to/biodiversity_conservation/.venv/Scripts/python",
         "args": [
           "/path/to/biodiversity_conservation/mcp_server.py"
         ]
       }
     }
   }
   ```

   Replace `/path/to/biodiversity_conservation` with the actual path to your project directory. On Windows, use forward slashes or escaped backslashes in the paths.

   After configuring, restart Claude Desktop to load the MCP server.

## Tutorial

### Quick Start: Analyzing a Species

1. **Launch the Streamlit Web Application**

   ```bash
   streamlit run app.py
   ```

   The application will be available at `http://localhost:8501`

2. **Enter a species name** (e.g., "Panthera tigris" for Tiger)

3. **View the analysis results:**
   - Conservation status and population trends
   - Identified threats from multiple sources
   - GBIF occurrence data with temporal anomaly detection
   - Recent conservation news articles

### Using the MCP Server with Claude Desktop

Once configured (see Installation step 5), you can query biodiversity data directly in Claude Desktop:

**Example queries:**
- "Get IUCN data for the African Elephant"
- "What are the threats to Panthera tigris?"
- "Show me GBIF occurrence data for Giant Panda"
- "Run a full conservation analysis on the Mountain Gorilla"

**Available MCP tools:**
- `get_iucn_data` - Retrieve IUCN Red List data and threats
- `get_gbif_data` - Get species occurrence records with temporal analysis
- `get_news_data` - Search conservation news articles
- `get_threat_classifications` - View all IUCN threat classification codes
- `run_full_analysis` - Execute complete multi-agent workflow

### Running System Evaluation

Evaluate the system's performance on a test set of species using [`evaluate_system.py`](evaluate_system.py):

```bash
python evaluate_system.py
```

This will:
- Run analysis on test species (default: Tiger, Giant Panda, Mountain Gorilla)
- Calculate quantitative metrics (threat detection, confidence alignment, etc.)
- Generate summary statistics and detailed results
- Save results to evaluation output files (CSV, JSON formats)

### Managing Cache

Clear cached data to force fresh API requests using [`clear_cache.py`](clear_cache.py):

```bash
# Clear specific cache
python clear_cache.py iucn      # Clear IUCN cache only
python clear_cache.py gbif      # Clear GBIF cache only
python clear_cache.py anomaly   # Clear anomaly detection cache
python clear_cache.py news      # Clear news cache only

# Clear all caches
python clear_cache.py all
```

### Development and Testing

```bash
# Test individual components
python test_agents.py
python test_data_sources.py
python test_api_live.py
python test_performance.py
```

## Project Structure

```
biodiversity_conservation/
├── biodiversity_intel/         # Main package
│   ├── __init__.py
│   ├── agents.py              # All three agents (Data, Analysis, Report)
│   ├── data_sources.py        # API clients (IUCN, GBIF, News)
│   ├── anomaly_detection.py   # Chronos-based time-series anomaly detection (run to build standalone visual)
│   ├── llm.py                 # LLM client and prompts
│   ├── workflow.py            # LangGraph orchestration
│   ├── analysis.py            # Threat detection and evaluation
│   ├── storage.py             # Caching and database
│   └── config.py              # Configuration management
├── app.py                     # Streamlit web application
├── mcp_server.py              # Model Context Protocol server (exposes tools to Claude Desktop)
├── evaluate_system.py         # System evaluation script
├── clear_cache.py             # Utility to clear specific cache directories (anomaly, gbif, iucn, news)
├── test_agents.py             # Agent testing script
├── test_api_live.py           # API Testing Script
├── test_data_sources.py       # Data source testing script
├── test_performance.py        # Performance and caching tests
├── test_mcp_server.py         # MCP Server handler tests
├── config/
│   ├── settings.yaml          # Application settings
│   └── prompts/               # Prompt templates
├── data/                      # Local data storage
│   └── cache/                 # API response cache
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks for exploration
├── EVALUATION_GUIDE.md        # Comprehensive evaluation guide
├── .env.example               # Environment variable template
├── pyproject.toml             # uv project configuration
└── README.md

```

## Development Timeline

- **Week 1**: Data pipeline and API integration
- **Week 2**: Agent scaffolding and orchestration
- **Weeks 3-4**: LLM integration and prompt engineering
- **Week 5**: Analysis modules and confidence scoring
- **Week 6**: Streamlit interface
- **Week 7**: Evaluation and final report

## Evaluation Metrics

The system includes comprehensive evaluation metrics implemented in [`biodiversity_intel/analysis.py`](biodiversity_intel/analysis.py):

### Quantitative Metrics

1. **Threat Detection Recall**: Ratio of IUCN-documented threats identified by the system
   - Formula: `(Detected Threats ∩ IUCN Threats) / IUCN Threats`
   - Range: 0.0 to 1.0 (higher is better)

2. **Threat Detection Precision**: Accuracy of detected threats (no false positives)
   - Formula: `True Positives / (True Positives + False Positives)`
   - Range: 0.0 to 1.0 (higher is better)

3. **Threat Detection F1-Score**: Harmonic mean of precision and recall
   - Formula: `2 × (Precision × Recall) / (Precision + Recall)`
   - Range: 0.0 to 1.0 (higher is better)

4. **Confidence Alignment Score**: Measures consistency between IUCN and GBIF population trends
   - Compares IUCN official assessments with GBIF occurrence-derived trends
   - Returns: 1.0 (perfect match), 0.7 (partial alignment), 0.5 (neutral/unknown), 0.0 (contradiction)
   - Handles variations in terminology and unknown values gracefully

5. **Temporal Anomaly Metrics**: Statistical analysis of GBIF occurrence patterns
   - Detects anomalous years using Chronos forecasting model (z-score threshold: 2.0)
   - Classifies episodes as population declines or surges
   - Reports episode severity, duration, and correlation with known threats

6. **Early Warning Signal**: Binary flag indicating data contradictions or high-risk conditions
   - Triggers on: declining populations, critical conservation status, high threat counts, anomalous patterns

### Performance Metrics

- **Execution Time**: End-to-end workflow performance (~77.6s average, <2s with caching)
- **Data Coverage**: Percentage of species with complete data from all sources (100% for all test species)
- **API Success Rates**: Reliability of external data source connections
- **Caching Performance**: 96% faster on repeated queries (77.6s → <2s)
- **Parallel Processing**: 40-50% faster with parallel API calls and parallel GBIF pagination

### Evaluation Outputs

Running [`evaluate_system.py`](evaluate_system.py) generates:
- **evaluation_results.csv**: Detailed metrics per species
- **evaluation_summary.json**: Aggregated statistics and averages
- **evaluation_full_results.json**: Complete evaluation data including system outputs

For detailed evaluation methodology and best practices, see [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md).

## Data Sources

- **[IUCN Red List](https://www.iucnredlist.org)**: Official conservation status and threat assessments
- **[GBIF](https://www.gbif.org)**: Species occurrence data (spatial and temporal)
  - Optimized with parallel pagination (5x faster for large datasets)
  - Supports up to 10,000 occurrence records per species
- **Conservation News**: Curated news sources (e.g., Mongabay)

## Performance Optimizations

### Parallel API Calls
- **DataAgent** fetches IUCN, GBIF, and News data concurrently using `asyncio.gather()`
- **Performance**: 40-50% faster than sequential calls (77s → 40-45s)
- **Implementation**: [`biodiversity_intel/agents.py`](biodiversity_intel/agents.py) (DataAgent)

### Parallel GBIF Pagination
- **GBIF Client** uses `ThreadPoolExecutor` for concurrent batch requests
- **Performance**: 5x faster for large datasets (68s → 14s for 10,000 records)
- **Rate Limiting**: Max 5 concurrent requests with 0.1s delays to respect API limits
- **Implementation**: [`biodiversity_intel/data_sources.py`](biodiversity_intel/data_sources.py) (GBIFClient.get_occurrences)

### File-Based Caching
- **All API clients** (IUCN, GBIF, News) and **anomaly detection** support persistent file-based caching
- **Performance**: 96% faster on repeated queries (77.6s → <2s)
- **Storage**: JSON files in `data/cache/{source}/` directories
- **Implementation**: [`biodiversity_intel/storage.py`](biodiversity_intel/storage.py) (FileCache)

### GPU-Accelerated Anomaly Detection
- **Chronos model** automatically detects and uses GPU when available (CUDA)
- **Fallback**: CPU inference when GPU unavailable
- **Performance**: Faster time-series forecasting on compatible hardware
- **Implementation**: [`biodiversity_intel/anomaly_detection.py`](biodiversity_intel/anomaly_detection.py) (GBIFAnomalyDetector)

### Combined Impact
- **First query**: ~77.6 seconds (with parallel optimizations)
- **Cached queries**: <2 seconds (96% improvement)
- **Total system improvement**: ~50% faster execution with all optimizations enabled

## Contributing

This is an academic project for COMS 6998-013: LLM-based GenAI at Columbia University.

## Authors

- **Abhishek Chaudhary** - Department of Computer Science, Columbia University
- **Bharath Vishal Ganesamoorthy** - Department of Biomedical Engineering, Columbia University

## Acknowledgments

Special thanks to Professors Parijat Dube and Chen Wang for their guidance in the COMS 6998-013 LLM-based GenAI course at Columbia University.

## License

MIT License - See LICENSE file for details

## References

1. IUCN Red List of Threatened Species. https://www.iucnredlist.org
2. Global Biodiversity Information Facility (GBIF). https://www.gbif.org
3. LangGraph: Building Agentic Workflows with Large Language Models, 2024
4. Park et al., "Generative Agents: Interactive Simulacra of Human Behavior," 2023
5. Ansari et al., "Chronos: Learning the Language of Time Series," arXiv:2403.07815, 2024

## Contact

For questions or feedback, please contact:
- ac5003@columbia.edu
- bg2879@columbia.edu
