"""Environment configuration for the MCP ClickHouse server.

This module handles all environment variable configuration with sensible defaults
and type conversion.
"""

from dataclasses import dataclass
import os

required_vars = ["REMOTE_DB", "DATA_DIR", "OPENAI_URL"]


@dataclass
class AppConfig:
    """
    This class handles all environment variable configuration with sensible
    defaults and type conversion. It provides typed methods for accessing
    each configuration value.

    Required environment variables:
        REMOTE_DB: HOST:PORT of remote database

    """

    def __init__(self):
        """Initialize the configuration from environment variables."""
        self._validate_required_vars()

    def _validate_required_vars(self) -> None:
        """Validate that all required environment variables are set.

        Raises:
            ValueError: If any required environment variable is missing.
        """
        missing_vars = []
        for var in required_vars:
            if var not in os.environ:
                missing_vars.append(var)

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {
                             ', '.join(missing_vars)}")

    @property
    def remote_db(self) -> str:
        """host:port of VectorDB"""
        return os.environ.get("REMOTE_DB") or ""

    @property
    def openai_url(self) -> str:
        """proto://host:port for openai server"""
        return os.environ.get("OPENAI_URL")

    @property
    def collection_name(self) -> str:
        """Name of collection to create or APPEND to"""
        return os.environ.get("COLLECTION_NAME") or "default"

    @property
    def embedding_model(self) -> str:
        """Name of collection to create or APPEND to"""
        return os.environ.get("EMBEDDING_MODEL") or "nomic_embed_text"

    @property
    def batch_size(self) -> str:
        """Size of items to add to collection per batch"""
        if "BATCH_SIZE" in os.environ:
            return int(os.environ.get("BATCH_SIZE"))
        return 100

    @property
    def append(self) -> str:
        """Whether or not to append to collection or recreate it"""
        if "APPEND" in os.environ:
            return os.environ.get("APPEND") == "true"
        return False

    @property
    def chunk_overlap(self) -> str:
        """"""
        if "CHUNK_OVERLAY" in os.environ:
            return os.environ.get("CHUNK_OVERLAY")
        return 200

    @property
    def chunk_size(self) -> str:
        """"""
        if "CHUNK_SIZE" in os.environ:
            return os.environ.get("CHUNK_SIZE")
        return 1000

    @property
    def data_dir(self) -> str:
        """Directory containing data to be turned into embeddings"""
        return os.getenv("DATA_DIR")


# Global instance placeholders for the singleton pattern
_CONFIG_INSTANCE = None
_CHDB_CONFIG_INSTANCE = None


def get_config():
    """
    Gets the singleton instance of AppConfig.
    Instantiates it on the first call.
    """
    global _CONFIG_INSTANCE
    if _CONFIG_INSTANCE is None:
        # Instantiate the config object here, ensuring load_dotenv() has likely run
        _CONFIG_INSTANCE = AppConfig()
    return _CONFIG_INSTANCE
