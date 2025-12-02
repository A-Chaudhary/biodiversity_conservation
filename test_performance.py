"""
Performance Testing Script

Tests caching and performance improvements:
- Parallel API calls vs sequential
- Caching performance (first vs cached query)
- GBIF parallel pagination
- Overall system performance
"""

import asyncio
import time
from biodiversity_intel.workflow import run_conservation_analysis
from biodiversity_intel.data_sources import IUCNClient, GBIFClient, MongabayClient
from biodiversity_intel.config import setup_logging

# Setup logging (reduce verbosity for cleaner output)
logger = setup_logging("WARNING")

def test_caching_performance():
    """Test caching performance - first call vs cached call."""
    print("\n" + "=" * 80)
    print("TEST 1: Caching Performance")
    print("=" * 80)
    
    species_name = "Panthera tigris"
    gbif_client = GBIFClient()
    
    # First call (cache miss)
    print(f"\nFirst call (cache miss) for {species_name}...")
    start_time = time.time()
    result1 = gbif_client.get_occurrences(species_name, limit=1000)
    first_call_time = time.time() - start_time
    
    if result1:
        print(f"First call completed: {first_call_time:.2f} seconds")
        print(f"   Records retrieved: {result1.occurrence_count}")
    else:
        print("First call failed")
        return
    
    # Second call (cache hit)
    print(f"\nSecond call (cache hit) for {species_name}...")
    start_time = time.time()
    result2 = gbif_client.get_occurrences(species_name, limit=1000)
    cached_call_time = time.time() - start_time
    
    if result2:
        print(f"Cached call completed: {cached_call_time:.2f} seconds")
        print(f"   Records retrieved: {result2.occurrence_count}")
        
        # Calculate improvement
        if first_call_time > 0:
            improvement = ((first_call_time - cached_call_time) / first_call_time) * 100
            speedup = first_call_time / cached_call_time if cached_call_time > 0 else 0
            print(f"\nPerformance Improvement:")
            print(f"   Speedup: {speedup:.1f}x faster")
            print(f"   Improvement: {improvement:.1f}%")
    else:
        print("Cached call failed")


def test_parallel_api_calls():
    """Test parallel API calls performance."""
    print("\n" + "=" * 80)
    print("TEST 2: Parallel API Calls Performance")
    print("=" * 80)
    
    species_name = "Panthera tigris"
    
    # Test sequential calls
    print(f"\nSequential API calls for {species_name}...")
    start_time = time.time()
    
    iucn_client = IUCNClient()
    gbif_client = GBIFClient()
    news_client = MongabayClient()
    
    iucn_data = iucn_client.get_species_data(species_name)
    gbif_data = gbif_client.get_occurrences(species_name, limit=1000)
    news_data = news_client.search_species_news(species_name, 5)
    
    sequential_time = time.time() - start_time
    print(f"Sequential calls completed: {sequential_time:.2f} seconds")
    
    # Test parallel calls (simulated - actual parallel happens in DataAgent)
    print(f"\nParallel API calls (via DataAgent) for {species_name}...")
    start_time = time.time()
    
    async def test_parallel():
        from biodiversity_intel.agents import DataAgent
        agent = DataAgent()
        state = {"species_name": species_name}
        result = await agent.execute(state)
        return result
    
    parallel_result = asyncio.run(test_parallel())
    parallel_time = time.time() - start_time
    
    print(f"Parallel calls completed: {parallel_time:.2f} seconds")
    
    # Calculate improvement
    if sequential_time > 0:
        improvement = ((sequential_time - parallel_time) / sequential_time) * 100
        speedup = sequential_time / parallel_time if parallel_time > 0 else 0
        print(f"\nPerformance Improvement:")
        print(f"   Speedup: {speedup:.1f}x faster")
        print(f"   Improvement: {improvement:.1f}%")


def test_gbif_pagination():
    """Test GBIF parallel pagination performance."""
    print("\n" + "=" * 80)
    print("TEST 3: GBIF Parallel Pagination Performance")
    print("=" * 80)
    
    species_name = "Panthera tigris"
    gbif_client = GBIFClient(enable_cache=False)  # Disable cache for fair test
    
    print(f"\nTesting GBIF pagination for {species_name}...")
    print("   (Parallel pagination is enabled by default)")
    
    start_time = time.time()
    result = gbif_client.get_occurrences(species_name, limit=5000)
    pagination_time = time.time() - start_time
    
    if result:
        print(f"Pagination completed: {pagination_time:.2f} seconds")
        print(f"   Records retrieved: {result.occurrence_count}")
        print(f"   Temporal distribution: {len(result.temporal_distribution)} years")
        
        # Estimate sequential time (approximate)
        estimated_sequential = pagination_time * 5  # Rough estimate
        print(f"\nEstimated sequential time: ~{estimated_sequential:.2f} seconds")
        print(f"   (Parallel pagination uses 5 concurrent requests)")
    else:
        print("Pagination failed")


def test_full_workflow_performance():
    """Test full workflow performance with and without cache."""
    print("\n" + "=" * 80)
    print("TEST 4: Full Workflow Performance")
    print("=" * 80)
    
    species_name = "Panthera tigris"
    
    # First run (cache miss)
    print(f"\nFirst workflow run (cache miss) for {species_name}...")
    start_time = time.time()
    result1 = asyncio.run(run_conservation_analysis(species_name))
    first_run_time = time.time() - start_time
    
    if result1:
        print(f"First run completed: {first_run_time:.2f} seconds")
        print(f"   Status: {result1.get('conservation_status', 'Unknown')}")
        print(f"   Threats: {len(result1.get('threats', []))}")
    else:
        print("First run failed")
        return
    
    # Second run (cache hit)
    print(f"\nSecond workflow run (cache hit) for {species_name}...")
    start_time = time.time()
    result2 = asyncio.run(run_conservation_analysis(species_name))
    cached_run_time = time.time() - start_time
    
    if result2:
        print(f"Cached run completed: {cached_run_time:.2f} seconds")
        print(f"   Status: {result2.get('conservation_status', 'Unknown')}")
        print(f"   Threats: {len(result2.get('threats', []))}")
        
        # Calculate improvement
        if first_run_time > 0:
            improvement = ((first_run_time - cached_run_time) / first_run_time) * 100
            speedup = first_run_time / cached_run_time if cached_run_time > 0 else 0
            print(f"\n Performance Improvement:")
            print(f"   Speedup: {speedup:.1f}x faster")
            print(f"   Improvement: {improvement:.1f}%")
            print(f"   Time saved: {first_run_time - cached_run_time:.2f} seconds")
    else:
        print("Cached run failed")


def test_cache_status():
    """Check cache status and files."""
    print("\n" + "=" * 80)
    print("TEST 5: Cache Status Check")
    print("=" * 80)
    
    from pathlib import Path
    
    cache_dirs = {
        "IUCN": Path("data/cache/iucn"),
        "GBIF": Path("data/cache/gbif"),
        "News": Path("data/cache/news")
    }
    
    print("\nCache Directory Status:")
    for source, cache_dir in cache_dirs.items():
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.json"))
            print(f"   {source}: {len(cache_files)} cached files")
            if cache_files:
                # Show most recent file
                latest = max(cache_files, key=lambda p: p.stat().st_mtime)
                size_kb = latest.stat().st_size / 1024
                print(f"      Latest: {latest.name} ({size_kb:.1f} KB)")
        else:
            print(f"   {source}: No cache directory (will be created on first use)")


def main():
    """Run all performance tests."""
    print("=" * 80)
    print("  BIODIVERSITY CONSERVATION SYSTEM - PERFORMANCE TESTS")
    print("=" * 80)
    print("\nThis script tests:")
    print("  1. Caching performance (first vs cached call)")
    print("  2. Parallel API calls performance")
    print("  3. GBIF parallel pagination")
    print("  4. Full workflow performance")
    print("  5. Cache status check")
    
    try:
        # Test 1: Caching
        test_caching_performance()
        
        # Test 2: Parallel API calls
        test_parallel_api_calls()
        
        # Test 3: GBIF pagination
        test_gbif_pagination()
        
        # Test 4: Full workflow
        test_full_workflow_performance()
        
        # Test 5: Cache status
        test_cache_status()
        
        print("\n" + "=" * 80)
        print("  ALL TESTS COMPLETED")
        print("=" * 80)
        print("\nPerformance tests finished!")
        print("   Check the results above to verify improvements.")
        
    except KeyboardInterrupt:
        print("\n\n Tests interrupted by user")
    except Exception as e:
        print(f"\n\n Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

