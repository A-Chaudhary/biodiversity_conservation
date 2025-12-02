"""
Storage and Caching Utilities

This module provides:
- API response caching (in-memory, file-based)
- SQLite database interface
- JSON file storage for reports
"""

import os
import json
import sqlite3
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path


class Cache:
    """Simple in-memory cache with TTL support."""

    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize cache.

        Args:
            ttl_seconds: Time-to-live in seconds (default: 24 hours)
        """
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Store value in cache with timestamp."""
        self.cache[key] = (value, datetime.now())

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()


class FileCache:
    """File-based cache for persistent storage with TTL support."""

    def __init__(self, cache_dir: str = "data/cache", ttl_days: int = 7):
        """
        Initialize file cache.

        Args:
            cache_dir: Directory for cache files
            ttl_days: Time-to-live in days (default: 7 days = 1 week)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_days * 24 * 60 * 60  # Convert days to seconds

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached data from file if not expired.
        
        Args:
            key: Cache key (species name)
            
        Returns:
            Cached data if valid and not expired, None otherwise
        """
        cache_file = self.cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache has timestamp (new format) or is old format
            if isinstance(cache_data, dict) and '_cached_at' in cache_data:
                # New format with timestamp
                cached_at = datetime.fromisoformat(cache_data['_cached_at'])
                age = datetime.now() - cached_at
                
                if age.total_seconds() > self.ttl_seconds:
                    # Cache expired, delete file and return None
                    cache_file.unlink()
                    return None
                
                # Return the actual data (without metadata)
                return cache_data.get('data')
            else:
                # Old format without timestamp - check file modification time
                file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                age = datetime.now() - file_mtime
                
                if age.total_seconds() > self.ttl_seconds:
                    # Cache expired, delete file and return None
                    cache_file.unlink()
                    return None
                
                # Return old format data (backward compatibility)
                return cache_data
        except (json.JSONDecodeError, OSError, ValueError) as e:
            # If file is corrupted or can't be read, delete it
            if cache_file.exists():
                cache_file.unlink()
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Store data to cache file with timestamp.
        
        Args:
            key: Cache key (species name)
            value: Data to cache
        """
        cache_file = self.cache_dir / f"{key}.json"
        
        # Wrap data with timestamp metadata
        cache_data = {
            '_cached_at': datetime.now().isoformat(),
            'data': value
        }
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except OSError as e:
            # If we can't write, log but don't fail
            import logging
            logging.warning(f"Failed to cache data for key '{key}': {e}")


class Database:
    """SQLite database interface for structured data storage."""

    def __init__(self, db_path: str = "data/biodiversity.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create species assessments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species_name TEXT NOT NULL,
                assessment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                threats TEXT,
                population_trend TEXT,
                confidence_score REAL,
                early_warning BOOLEAN,
                iucn_data TEXT,
                gbif_data TEXT,
                report TEXT
            )
        """)

        conn.commit()
        conn.close()

    def save_assessment(self, assessment_data: Dict[str, Any]) -> int:
        """
        Save a species assessment to the database.

        Args:
            assessment_data: Dictionary containing assessment information

        Returns:
            ID of the saved assessment
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO assessments
            (species_name, threats, population_trend, confidence_score,
             early_warning, iucn_data, gbif_data, report)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            assessment_data.get('species_name'),
            json.dumps(assessment_data.get('threats', [])),
            assessment_data.get('population_trend'),
            assessment_data.get('confidence_score'),
            assessment_data.get('early_warning', False),
            json.dumps(assessment_data.get('iucn_data', {})),
            json.dumps(assessment_data.get('gbif_data', {})),
            assessment_data.get('report', '')
        ))

        assessment_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return assessment_id


class ReportStorage:
    """Storage for generated reports in JSON and Markdown formats."""

    def __init__(self, output_dir: str = "data/outputs"):
        """
        Initialize report storage.

        Args:
            output_dir: Directory for storing reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_report(
        self,
        species_name: str,
        report_data: Dict[str, Any],
        format: str = "both"
    ) -> None:
        """
        Save report to file.

        Args:
            species_name: Name of the species
            report_data: Report data dictionary
            format: "json", "markdown", or "both"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = species_name.replace(" ", "_").lower()

        if format in ["json", "both"]:
            json_path = self.output_dir / f"{safe_name}_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2)

        if format in ["markdown", "both"]:
            md_path = self.output_dir / f"{safe_name}_{timestamp}.md"
            markdown_content = self._generate_markdown(report_data)
            with open(md_path, 'w') as f:
                f.write(markdown_content)

    def _generate_markdown(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown format from report data."""
        # TODO: Implement markdown generation
        return "# Species Threat Assessment\n\n" + json.dumps(report_data, indent=2)
