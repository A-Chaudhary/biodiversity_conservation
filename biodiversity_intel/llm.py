"""
LLM Integration Module

This module provides:
- OpenAI LLM client
- Prompt templates and management
- Output parsing and validation
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("biodiversity_intel.llm")


class LLMClient:
    """OpenAI LLM client for conservation analysis."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4000
    ):
        """
        Initialize OpenAI LLM client.

        Args:
            api_key: OpenAI API key (defaults to environment variable)
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        from biodiversity_intel.config import config

        self.api_key = api_key or config.openai_api_key
        self.model = model or config.openai_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"Initializing LLM client with model: {self.model}, temperature: {self.temperature}")
        self._init_openai()

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            logger.debug("Initializing OpenAI client")
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")
        except ImportError as e:
            logger.error("OpenAI package not installed", exc_info=True)
            raise ImportError("OpenAI package not installed. Run: uv pip install openai")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}", exc_info=True)
            raise

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using OpenAI.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)

        Returns:
            Generated text
        """
        logger.info(f"Generating LLM response with model: {self.model}")
        logger.debug(f"Prompt length: {len(prompt)} chars, Has system prompt: {system_prompt is not None}")

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            logger.debug(f"Calling OpenAI API with {len(messages)} messages")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            result = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else 'unknown'
            logger.info(f"LLM response generated successfully (tokens: {tokens_used})")
            logger.debug(f"Response length: {len(result)} chars")

            return result
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}", exc_info=True)
            raise


class PromptTemplate:
    """Template for LLM prompts."""

    def __init__(self, template: str):
        """
        Initialize prompt template.

        Args:
            template: Template string with {variable} placeholders
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Format template with variables.

        Args:
            **kwargs: Variable values

        Returns:
            Formatted prompt string
        """
        return self.template.format(**kwargs)


# Prompt templates
ANALYSIS_PROMPT = PromptTemplate("""
You are a conservation biologist analyzing biodiversity data for {species_name}.

IUCN Red List Data:
{iucn_data}

GBIF Occurrence Data:
{gbif_data}

Conservation News Articles:
{news_data}

Anomaly Detection Results (Time-Series Analysis):
{anomaly_data}

Task: Analyze the data and identify:
1. Key threats to this species
2. Population trend (increasing, stable, decreasing, or unknown)
3. Any inconsistencies between IUCN assessment and GBIF occurrence patterns
4. Relevant insights from recent conservation news
5. Early warning signals that may require urgent attention
6. Temporal trends in conservation status from IUCN assessment history
7. **Anomalous patterns detected in occurrence data** - Pay special attention to:
   - Anomaly episodes (declines/surges) from the time-series analysis
   - Years with significant deviations from expected occurrence counts
   - Whether these anomalies correlate with known threats or conservation events
   - How anomaly detection complements or contradicts other data sources

Provide a detailed but concise analysis focusing on evidence-based conclusions.
Consider how news articles complement or contradict the scientific data.
Pay attention to how the conservation status has changed over time based on assessment history.
**When anomaly detection data is available, integrate these findings into your analysis** - explain what the anomalies reveal about population dynamics, whether they align with reported threats, and if they suggest emerging conservation concerns.
""")

REPORT_PROMPT = PromptTemplate("""
Generate a structured conservation threat report for {species_name}.

Analysis Results:
{analysis_results}

Create a report with the following sections:
1. Executive Summary
2. Conservation Status
3. Identified Threats
4. Population Trend Assessment
5. Data Consistency Analysis
6. Recommendations
7. Confidence Assessment

Use clear, professional language suitable for conservation practitioners and policymakers.
""")