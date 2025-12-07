"""Clear cache for specific data sources or all caches."""
import os
import glob
import argparse
import sys


CACHE_SOURCES = {
    "anomaly": "data/cache/anomaly",
    "gbif": "data/cache/gbif",
    "iucn": "data/cache/iucn",
    "news": "data/cache/news"
}


def clear_cache(source: str) -> None:
    """
    Clear cache for a specific source.

    Args:
        source: Cache source name (anomaly, gbif, iucn, news, or all)
    """
    if source == "all":
        sources_to_clear = CACHE_SOURCES.items()
        print("Clearing all caches...")
    elif source.lower() in CACHE_SOURCES:
        sources_to_clear = [(source.lower(), CACHE_SOURCES[source.lower()])]
        print(f"Clearing {source.upper()} cache...")
    else:
        print(f"Error: Unknown cache source '{source}'")
        print(f"Available sources: {', '.join(CACHE_SOURCES.keys())}, all")
        sys.exit(1)

    total_deleted = 0
    for source_name, cache_dir in sources_to_clear:
        if os.path.exists(cache_dir):
            cache_files = glob.glob(os.path.join(cache_dir, "*.json"))
            if cache_files:
                print(f"\n{source_name.upper()}: Found {len(cache_files)} cache files")
                for file in cache_files:
                    print(f"  Deleting: {os.path.basename(file)}")
                    os.remove(file)
                total_deleted += len(cache_files)
            else:
                print(f"\n{source_name.upper()}: No cache files found")
        else:
            print(f"\n{source_name.upper()}: Cache directory not found ({cache_dir})")

    if total_deleted > 0:
        print(f"\nCache cleared successfully! Deleted {total_deleted} file(s)")
    else:
        print("\nNo cache files to delete")


def main():
    """Main function to parse arguments and clear cache."""
    parser = argparse.ArgumentParser(
        description="Clear cache for specific data sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python clear_cache.py iucn           # Clear IUCN cache only
  python clear_cache.py gbif           # Clear GBIF cache only
  python clear_cache.py anomaly        # Clear anomaly detection cache
  python clear_cache.py news           # Clear news cache only
  python clear_cache.py all            # Clear all caches
        """
    )

    parser.add_argument(
        "source",
        type=str,
        choices=list(CACHE_SOURCES.keys()) + ["all"],
        help="Cache source to clear (anomaly, gbif, iucn, news, or all)"
    )

    args = parser.parse_args()
    clear_cache(args.source)


if __name__ == "__main__":
    main()
