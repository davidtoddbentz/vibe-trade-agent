"""LangGraph middleware to extract user ID from request headers."""

import logging

from langchain_core.runnables import RunnableConfig

from .auth import extract_user_id_from_token

logger = logging.getLogger(__name__)


def extract_user_id_from_config(config: RunnableConfig | None) -> str | None:
    """Extract user_id from LangGraph config/metadata.

    LangGraph Server may pass request metadata through config.
    Check for Authorization header or user_id in metadata.

    Note: LangGraph Server's exact metadata structure may vary.
    This function checks multiple possible locations.

    Args:
        config: LangGraph RunnableConfig (may contain request metadata)

    Returns:
        User ID (Firebase UID) if found, None otherwise
    """
    if not config:
        return None

    # Check if user_id is already in config metadata
    metadata = config.get("metadata", {})
    if "user_id" in metadata:
        return metadata["user_id"]

    # Try to extract from Authorization header in metadata
    # LangGraph Server may pass HTTP headers through metadata
    headers = metadata.get("headers", {})
    if isinstance(headers, dict):
        auth_header = headers.get("authorization") or headers.get("Authorization")
    else:
        # Headers might be a list of tuples
        auth_header = None
        if isinstance(headers, list):
            for key, value in headers:
                if key.lower() == "authorization":
                    auth_header = value
                    break

    if auth_header:
        user_id = extract_user_id_from_token(auth_header)
        if user_id:
            return user_id

    # Also check if LangGraph passes it directly in config
    # Some LangGraph Server implementations pass user context here
    if "user_id" in config:
        return config["user_id"]

    # Check configurable for user_id (passed explicitly from UI)
    configurable = config.get("configurable", {})
    if configurable and "user_id" in configurable:
        user_id = configurable.get("user_id")
        if user_id:
            logger.info(f"Extracted user_id from configurable: {user_id}")
            return user_id

    return None
