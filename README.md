# Biodiversity Conservation Intelligence System

An agentic threat intelligence system leveraging large language models (LLMs) to synthesize information from multiple open biodiversity sources into coherent, explainable threat summaries.

## Overview

Global biodiversity loss is accelerating, yet conservation intelligence systems remain fragmented and reactive. This project proposes a multi-agent LLM-based system that automates data retrieval, reasoning, and reporting for any queried species, enabling transparent, evidence-based ecological assessment at scale.

## Features

- **Multi-Agent Architecture**: Three specialized agents (Data, Analysis, Report) working collaboratively
- **Multi-Source Integration**: Combines IUCN Red List, GBIF occurrence data, and conservation news
- **Automated Threat Detection**: Identifies threats and inconsistencies across data sources
- **Explainable AI**: Generates transparent, evidence-based conservation insights
- **Streamlit Interface**: Interactive web app for species queries and visualization
- **Evaluation Metrics**: Quantitative assessment of threat detection and confidence alignment

## System Architecture

```
User Query (Species Name)
        │
        ▼
   Streamlit App
     (app.py)
        │
        ▼
  LangGraph Workflow
   (workflow.py)
        │
        ├─► Data Agent ──────► IUCN API
        │   (agents.py)        GBIF API
        │                      News Sources
        │
        ▼
   Analysis Agent ──────► LLM Client
   (agents.py)           (llm.py)
        │
        ▼
   Report Agent
   (agents.py)
        │
        ▼
   Storage & Output
    (storage.py)
```

## Technology Stack

- **Python**: >=3.10
- **Package Management**: uv
- **Agent Framework**: LangGraph
- **LLM**: OpenAI GPT-4o-mini
- **Frontend**: Streamlit
- **Data Storage**: SQLite / JSON
- **Testing**: pytest

## Installation

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/biodiversity_conservation.git
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
   # - IUCN_API_TOKEN (optional)
   ```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=biodiversity_intel --cov-report=html

# Format code
black biodiversity_intel/ app.py tests/

# Lint code
ruff check biodiversity_intel/ app.py tests/

# Type checking
mypy biodiversity_intel/
```

## Project Structure

```
biodiversity_conservation/
├── biodiversity_intel/         # Main package
│   ├── __init__.py
│   ├── agents.py              # All three agents (Data, Analysis, Report)
│   ├── data_sources.py        # API clients (IUCN, GBIF, News)
│   ├── llm.py                 # LLM client and prompts
│   ├── workflow.py            # LangGraph orchestration
│   ├── analysis.py            # Threat detection and evaluation
│   ├── storage.py             # Caching and database
│   └── config.py              # Configuration management
├── app.py                     # Streamlit web application
├── config/
│   ├── settings.yaml          # Application settings
│   └── prompts/               # Prompt templates
├── data/                      # Local data storage
│   ├── cache/                 # API response cache
│   ├── outputs/               # Generated reports
│   └── sample/                # Sample test data
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks for exploration
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

1. **Threat Detection Recall**: Ratio of IUCN-documented threats identified
2. **Confidence Alignment Score**: Consistency between IUCN and GBIF trends
3. **Early Warning Signal**: Binary flag for data contradictions

## Data Sources

- **[IUCN Red List](https://www.iucnredlist.org)**: Official conservation status and threat assessments
- **[GBIF](https://www.gbif.org)**: Species occurrence data (spatial and temporal)
- **Conservation News**: Curated news sources (e.g., Mongabay)

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

## Contact

For questions or feedback, please contact:
- ac5003@columbia.edu
- bg2879@columbia.edu
