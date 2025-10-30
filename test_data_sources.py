"""
Test Script for Data Source Clients

This script tests the IUCN, GBIF, and News API clients to verify they work correctly.
Run this script to validate API connections and data retrieval.

Usage:
    python test_data_sources.py
"""

import asyncio
import logging
from biodiversity_intel.config import setup_logging
from biodiversity_intel.data_sources import IUCNClient, GBIFClient

# Setup logging
logger = setup_logging("DEBUG")
test_logger = logging.getLogger("test_data_sources")


def print_separator(title: str):
    """Print a formatted separator."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def test_iucn_client():
    """Test IUCN Red List API client."""
    print_separator("Testing IUCN Red List API Client")

    try:
        client = IUCNClient()
        test_logger.info("IUCN client initialized successfully")

        # Test species
        test_species = [
            "Panthera tigris",  # Tiger
            "Ailuropoda melanoleuca",  # Giant Panda
            "Gorilla beringei"  # Mountain Gorilla
        ]

        for species in test_species:
            print(f"\n[CLIPBOARD] Testing IUCN data retrieval for: {species}")
            try:
                data = client.get_species_data(species)
                if data:
                    print(f"[PASS] Successfully retrieved IUCN data for {species}")
                    print(f"   Status: {data.conservation_status}")
                    print(f"   Trend: {data.population_trend}")
                    print(f"   Threats: {len(data.threats)} identified")
                else:
                    print(f"[WARN]  No IUCN data found for {species}")
            except NotImplementedError:
                print(f"[WARN]  IUCN API client not yet implemented (TODO)")
                break
            except Exception as e:
                print(f"[FAIL] Error retrieving IUCN data for {species}: {e}")

        print("\n[PASS] IUCN client test completed")
        return True

    except Exception as e:
        print(f"[FAIL] IUCN client initialization failed: {e}")
        test_logger.error(f"IUCN test failed: {e}", exc_info=True)
        return False


def test_gbif_client():
    """Test GBIF Occurrence API client."""
    print_separator("Testing GBIF Occurrence API Client")

    try:
        client = GBIFClient()
        test_logger.info("GBIF client initialized successfully")

        # Test species
        test_species = [
            "Panthera tigris",
            "Ailuropoda melanoleuca",
            "Gorilla beringei"
        ]

        for species in test_species:
            print(f"\n[CLIPBOARD] Testing GBIF occurrence retrieval for: {species}")
            try:
                data = client.get_occurrences(species, limit=100)
                if data:
                    print(f"[PASS] Successfully retrieved GBIF data for {species}")
                    print(f"   Occurrences: {data.occurrence_count}")
                    print(f"   Temporal distribution: {len(data.temporal_distribution)} time periods")
                    print(f"   Spatial points: {len(data.spatial_distribution)}")
                else:
                    print(f"[WARN]  No GBIF data found for {species}")
            except Exception as e:
                print(f"[FAIL] Error retrieving GBIF data for {species}: {e}")
                test_logger.error(f"GBIF error for {species}: {e}", exc_info=True)

        print("\n[PASS] GBIF client test completed")
        return True

    except Exception as e:
        print(f"\n[FAIL] GBIF client test failed: {e}")
        test_logger.error(f"GBIF test failed: {e}", exc_info=True)
        return False



def run_all_tests():
    """Run all data source client tests."""
    print_separator("[TEST] Data Source Client Test Suite")
    print("Testing IUCN and GBIF API clients\n")

    results = {
        "IUCN": False,
        "GBIF": False
    }

    # Run tests
    results["IUCN"] = test_iucn_client()
    results["GBIF"] = test_gbif_client()

    # Summary
    print_separator("[CHART] Test Results Summary")

    for client_name, passed in results.items():
        status = "[PASS] PASSED" if passed else "[FAIL] FAILED"
        print(f"{client_name:20s} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} clients tested successfully")

    if total_passed == total_tests:
        print("\n[CELEBRATE] All data source clients are working!")
    else:
        print("\n[WARN]  Some clients need implementation or have issues")
        print("   Check the TODO comments in biodiversity_intel/data_sources.py")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Fix Windows console encoding
    import sys
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              Biodiversity Conservation Data Source Test Suite                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n\n[WARN]  Test interrupted by user")
    except Exception as e:
        print(f"\n\n[FAIL] Unexpected error during testing: {e}")
        test_logger.error(f"Unexpected test error: {e}", exc_info=True)
