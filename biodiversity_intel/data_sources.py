"""
Data Source API Clients

This module provides clients for interacting with biodiversity data APIs:
- IUCN Red List API
"""

import os
import requests
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("biodiversity_intel.data_sources")


class IUCNData(BaseModel):
    """Model for IUCN Red List data."""
    scientific_name: str
    conservation_status: str
    population_trend: Optional[str] = None
    threats: List[str] = []
    assessment_date: Optional[str] = None


class IUCNClient:
    """Client for IUCN Red List API."""

    def __init__(self, api_url: Optional[str] = None, api_token: Optional[str] = None):
        self.api_url = api_url or os.getenv("IUCN_API_URL", "https://api.iucnredlist.org/api/v4")
        self.api_token = api_token or os.getenv("IUCN_API_TOKEN")
        self.session = requests.Session()
        if self.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})
        logger.info(f"Initialized IUCN client with URL: {self.api_url}")

    def get_species_data(self, species_name: str) -> Optional[IUCNData]:
        """
        Retrieve IUCN data for a species.

        Args:
            species_name: Scientific name of the species

        Returns:
            IUCNData object or None if not found
        """
        logger.info(f"IUCN: Fetching data for species '{species_name}'")

        if not self.api_token:
            logger.warning("IUCN: No API token provided, request may be limited")

        try:
            # Split scientific name into genus and species
            # v4 API uses query parameters: genus_name and species_name
            name_parts = species_name.split()
            if len(name_parts) < 2:
                logger.error(f"IUCN: Invalid scientific name format '{species_name}' - need genus and species")
                return None

            genus_name = name_parts[0]
            species_part = name_parts[1]

            # Get species assessment IDs using v4 taxa endpoint
            endpoint = f"{self.api_url}/taxa/scientific_name"
            params = {
                "genus_name": genus_name,
                "species_name": species_part
            }

            # Add subspecies if provided (trinomial name)
            if len(name_parts) > 2:
                params["infra_name"] = name_parts[2]

            logger.debug(f"IUCN: Making API request to {endpoint} with params: {params}")

            response = self.session.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            logger.debug(f"IUCN: Received response: {data}")

            # The taxa/scientific_name endpoint returns assessment IDs
            # We need to get the latest assessment_id and fetch full details
            if not data or 'assessments' not in data or len(data['assessments']) == 0:
                logger.warning(f"IUCN: No assessment data found for '{species_name}'")
                return None

            # Get the most recent assessment ID
            assessments = data['assessments']
            latest_assessment = assessments[0]  # Assuming first is most recent
            assessment_id = latest_assessment.get('assessment_id')

            if not assessment_id:
                logger.warning(f"IUCN: No assessment_id found for '{species_name}'")
                return None

            # Fetch full assessment data
            logger.debug(f"IUCN: Fetching full assessment data for assessment_id: {assessment_id}")
            assessment_endpoint = f"{self.api_url}/assessment/{assessment_id}"
            assessment_response = self.session.get(assessment_endpoint, timeout=30)
            assessment_response.raise_for_status()

            assessment = assessment_response.json()
            logger.debug(f"IUCN: Received assessment data")

            # Extract basic information from v4 API structure
            # Conservation status is nested: {code: "EN", description: {en: "Endangered"}}
            red_list_category = assessment.get('red_list_category', {})
            conservation_status = red_list_category.get('code', 'Unknown')

            # Population trend is nested: {code: "1", description: {en: "Decreasing"}}
            pop_trend_obj = assessment.get('population_trend', {})
            population_trend_desc = pop_trend_obj.get('description', {}).get('en', 'Unknown')

            # Assessment date
            assessment_date = assessment.get('assessment_date')

            # Scientific name is in taxon object
            taxon = assessment.get('taxon', {})
            scientific_name = taxon.get('scientific_name', species_name)

            logger.debug(f"IUCN: Status={conservation_status}, Trend={population_trend_desc}")

            # Get threats data if available
            threats_list = []
            threats_data = assessment.get('threats', [])

            if isinstance(threats_data, list):
                for threat in threats_data:
                    if isinstance(threat, dict):
                        # Try multiple fields for threat name
                        threat_name = (
                            threat.get('title') or
                            threat.get('name') or
                            threat.get('code', 'Unknown threat')
                        )
                        threats_list.append(threat_name)
                    elif isinstance(threat, str):
                        threats_list.append(threat)

            logger.info(f"IUCN: Successfully retrieved data for '{species_name}' - {conservation_status}, {len(threats_list)} threats")

            return IUCNData(
                scientific_name=scientific_name,
                conservation_status=conservation_status,
                population_trend=population_trend_desc,
                threats=threats_list,
                assessment_date=assessment_date
            )

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"IUCN: Species '{species_name}' not found (404)")
                return None
            elif e.response.status_code == 403:
                logger.error(f"IUCN: Access forbidden (403) - check API token or rate limits")
                return None
            else:
                logger.error(f"IUCN: HTTP error fetching data for '{species_name}': {e}", exc_info=True)
                raise
        except requests.exceptions.Timeout:
            logger.error(f"IUCN: Request timeout for '{species_name}'")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"IUCN: Request error for '{species_name}': {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"IUCN: Unexpected error fetching data for '{species_name}': {e}", exc_info=True)
            return None
