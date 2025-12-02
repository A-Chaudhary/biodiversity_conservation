"""
Quick Performance Test

Quick test to verify caching and performance improvements are working.
"""

import asyncio
import time
from biodiversity_intel.workflow import run_conservation_analysis

def quick_test():
    """Quick test of caching performance."""
    print("=" * 60)
    print("  QUICK PERFORMANCE TEST")
    print("=" * 60)
    
    species = "Panthera tigris"
    
    print(f"\nTesting species: {species}")
    print("\n1️⃣  First run (cache miss - will be slower)...")
    start = time.time()
    result1 = asyncio.run(run_conservation_analysis(species))
    time1 = time.time() - start
    
    print(f"   ✅ Completed in {time1:.2f} seconds")
    
    print("\n2️⃣  Second run (cache hit - should be much faster)...")
    start = time.time()
    result2 = asyncio.run(run_conservation_analysis(species))
    time2 = time.time() - start
    
    print(f"   ✅ Completed in {time2:.2f} seconds")
    
    if time1 > 0 and time2 > 0:
        improvement = ((time1 - time2) / time1) * 100
        speedup = time1 / time2
        print(f"\n🚀 Results:")
        print(f"   First run:  {time1:.2f}s")
        print(f"   Cached run: {time2:.2f}s")
        print(f"   Speedup:    {speedup:.1f}x faster")
        print(f"   Improvement: {improvement:.1f}%")
        
        if speedup > 2:
            print("\n✅ Caching is working! Significant speedup detected.")
        else:
            print("\n⚠️  Caching may not be working as expected.")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    quick_test()

