"""
Configuration Management

This module handles loading and managing configuration from:
- Environment variables (.env)
- YAML configuration files
- Default settings
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("biodiversity_intel")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


class Config:
    """Central configuration manager."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file (default: config/settings.yaml)
        """
        # Load environment variables
        load_dotenv()

        # Load YAML configuration
        self.config_path = config_path or "config/settings.yaml"
        self.settings = self._load_yaml_config()

        # Initialize configuration values
        self._init_config()

        # Setup logging
        self.logger = setup_logging(self.log_level)
        self.logger.info("Configuration initialized successfully")

    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _init_config(self) -> None:
        """Initialize configuration from environment and YAML."""
        # OpenAI Configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        self.llm_max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4000"))

        # API Configuration
        self.iucn_api_url = os.getenv("IUCN_API_URL", "https://api.iucnredlist.org/api/v4")
        self.iucn_api_token = os.getenv("IUCN_API_TOKEN")
        self.gbif_api_url = os.getenv("GBIF_API_URL", "https://api.gbif.org/v1")

        # Cache Configuration
        self.enable_cache = os.getenv("ENABLE_CACHE", "true").lower() == "true"
        self.cache_type = os.getenv("CACHE_TYPE", "memory")
        self.cache_ttl = int(os.getenv("CACHE_TTL", "86400"))

        # Database Configuration
        self.database_type = os.getenv("DATABASE_TYPE", "sqlite")
        self.sqlite_db_path = os.getenv("SQLITE_DB_PATH", "data/biodiversity.db")

        # Application Settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation for nested keys)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        # Try environment variable first
        env_key = key.upper().replace(".", "_")
        env_value = os.getenv(env_key)
        if env_value is not None:
            return env_value

        # Try YAML config with dot notation
        keys = key.split(".")
        value = self.settings
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value


# Global configuration instance
config = Config()
