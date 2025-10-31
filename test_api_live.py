"""
Live API Test Script

This script makes actual API calls to IUCN and GBIF to verify connectivity.
It provides a quick way to test if your API credentials are working.

Usage:
    python test_api_live.py
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def print_header(title: str):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_iucn_api():
    """Test IUCN Red List API with real request."""
    print_header("[LION] Testing IUCN Red List API v4")

    api_url = os.getenv("IUCN_API_URL", "https://api.iucnredlist.org/api/v4")
    api_token = os.getenv("IUCN_API_TOKEN")

    if not api_token:
        print("\n[WARN]  IUCN_API_TOKEN not set in .env file")
        print("   API v4 requires authentication")
        print("   Get a free token at: https://api.iucnredlist.org/users/sign_up")

    # Test species: Tiger
    species = "Panthera tigris"
    print(f"\n[CLIPBOARD] Querying IUCN for: {species}")

    try:
        # Split scientific name into genus and species
        name_parts = species.split()
        genus_name = name_parts[0]
        species_part = name_parts[1] if len(name_parts) > 1 else ""

        # Use the taxa/scientific_name endpoint with query parameters
        endpoint = f"{api_url}/taxa/scientific_name"
        params = {
            "genus_name": genus_name,
            "species_name": species_part
        }

        # Set up headers with token
        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        print(f"   URL: {endpoint}")
        print(f"   Params: {params}")
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)

        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"   Response data: {data}")

            if data and "assessments" in data and len(data["assessments"]) > 0:
                # Get the first assessment ID
                assessment_id = data["assessments"][0].get("assessment_id")
                print(f"\n   Found assessment_id: {assessment_id}")

                # Fetch full assessment data
                assessment_endpoint = f"{api_url}/assessment/{assessment_id}"
                print(f"   Fetching full assessment from: {assessment_endpoint}")
                assessment_response = requests.get(assessment_endpoint, headers=headers, timeout=10)

                if assessment_response.status_code == 200:
                    assessment = assessment_response.json()

                    # Extract v4 API response structure
                    red_list_category = assessment.get('red_list_category', {})
                    status_code = red_list_category.get('code', 'N/A')

                    pop_trend = assessment.get('population_trend', {})
                    trend_desc = pop_trend.get('description', {}).get('en', 'N/A')

                    taxon = assessment.get('taxon', {})
                    sci_name = taxon.get('scientific_name', 'N/A')

                    print(f"\n[PASS] IUCN API v4 is working!")
                    print(f"   Scientific Name: {sci_name}")
                    print(f"   Conservation Status: {status_code}")
                    print(f"   Population Trend: {trend_desc}")
                    print(f"   Assessment Date: {assessment.get('assessment_date', 'N/A')}")
                    threats_count = len(assessment.get('threats', []))
                    print(f"   Threats Identified: {threats_count}")
                    return True
                else:
                    print(f"\n[WARN]  Failed to fetch assessment details (Status {assessment_response.status_code})")
            else:
                print("\n[WARN]  No assessment data found for species")
        elif response.status_code == 401:
            print("\n[FAIL] Authentication failed - check IUCN_API_TOKEN")
            print("   Make sure to use 'Token' authorization (not 'Bearer')")
        elif response.status_code == 403:
            print("\n[WARN]  Access forbidden (403)")
            print("   Your API token may be invalid or expired")
            print("   Get a new token at: https://api.iucnredlist.org/users/sign_up")
        elif response.status_code == 404:
            print("\n[WARN]  Species not found (404)")
        else:
            print(f"\n[FAIL] Request failed (Status {response.status_code})")

    except requests.exceptions.Timeout:
        print("\n[FAIL] Request timed out - check your internet connection")
    except requests.exceptions.RequestException as e:
        print(f"\n[FAIL] Request error: {e}")
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")

    return False


def test_gbif_api():
    """Test GBIF API with real request."""
    print_header("[GLOBE] Testing GBIF Occurrence API")

    api_url = os.getenv("GBIF_API_URL", "https://api.gbif.org/v1")

    # Test species: Tiger
    species = "Panthera tigris"
    print(f"\n[CLIPBOARD] Querying GBIF for: {species}")

    try:
        # Search for species key first
        endpoint = f"{api_url}/species/match"
        params = {"name": species}

        print(f"   URL: {endpoint}")
        response = requests.get(endpoint, params=params, timeout=10)

        print(f"   Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if "usageKey" in data:
                species_key = data["usageKey"]
                print(f"\n[PASS] Species matched! Key: {species_key}")
                print(f"   Scientific Name: {data.get('scientificName', 'N/A')}")
                print(f"   Rank: {data.get('rank', 'N/A')}")
                print(f"   Status: {data.get('status', 'N/A')}")

                # Now get occurrence count
                occurrence_endpoint = f"{api_url}/occurrence/search"
                occ_params = {"taxonKey": species_key, "limit": 0}

                print(f"\n[CHART] Fetching occurrence statistics...")
                occ_response = requests.get(occurrence_endpoint, params=occ_params, timeout=10)

                if occ_response.status_code == 200:
                    occ_data = occ_response.json()
                    count = occ_data.get("count", 0)
                    print(f"   Total occurrences: {count:,}")
                    print("\n[PASS] GBIF API is working!")
                    return True
            else:
                print("\n[WARN]  No match found for species")
        else:
            print(f"\n[FAIL] Request failed: {response.text[:200]}")

    except requests.exceptions.Timeout:
        print("\n[FAIL] Request timed out - check your internet connection")
    except requests.exceptions.RequestException as e:
        print(f"\n[FAIL] Request error: {e}")
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")

    return False


def test_openai_api():
    """Test OpenAI API connectivity."""
    print_header("[BOT] Testing OpenAI API")

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print("\n[FAIL] OPENAI_API_KEY not set in .env file")
        print("   Get your API key at: https://platform.openai.com/api-keys")
        return False

    try:
        from openai import OpenAI

        print("\n[CLIPBOARD] Testing OpenAI connection...")
        client = OpenAI(api_key=api_key)

        # Simple test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API test successful' if you receive this."}],
            max_tokens=10
        )

        result = response.choices[0].message.content
        print(f"   Response: {result}")
        print(f"\n[PASS] OpenAI API is working!")
        return True

    except ImportError:
        print("\n[FAIL] OpenAI package not installed")
        print("   Run: uv pip install openai")
    except Exception as e:
        print(f"\n[FAIL] OpenAI API error: {e}")

    return False


def main():
    """Run all live API tests."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                      Live API Connectivity Test Suite                        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    print("\n[SEARCH] Checking environment variables...")
    print(f"   IUCN_API_TOKEN: {'[PASS] Set' if os.getenv('IUCN_API_TOKEN') else '[WARN]  Not set'}")
    print(f"   OPENAI_API_KEY: {'[PASS] Set' if os.getenv('OPENAI_API_KEY') else '[FAIL] Not set (required)'}")

    results = {
        "IUCN": test_iucn_api(),
        "GBIF": test_gbif_api(),
        "OpenAI": test_openai_api()
    }

    # Summary
    print_header("[CHART] Test Results Summary")

    for api_name, passed in results.items():
        status = "[PASS] WORKING" if passed else "[FAIL] FAILED"
        print(f"{api_name:20s} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\n{'─' * 80}")
    print(f"Total: {total_passed}/{total_tests} APIs working")

    if total_passed == total_tests:
        print("\n[CELEBRATE] All APIs are accessible and working!")
    else:
        print("\n[WARN]  Some APIs need configuration or are not accessible")
        print("   Check your .env file and API credentials")

    print(f"{'─' * 80}\n")


if __name__ == "__main__":
    # Fix Windows console encoding
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    main()
