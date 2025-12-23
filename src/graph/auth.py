"""Firebase authentication utilities for extracting user ID from tokens."""

import logging

try:
    import firebase_admin
    from firebase_admin import auth, credentials
except ImportError:
    firebase_admin = None
    auth = None
    credentials = None

logger = logging.getLogger(__name__)

# Initialize Firebase Admin SDK (uses Application Default Credentials on GCP)
_firebase_initialized = False


def _ensure_firebase_initialized():
    """Initialize Firebase Admin SDK if not already initialized."""
    global _firebase_initialized
    if _firebase_initialized or firebase_admin is None:
        return

    try:
        if not firebase_admin._apps:
            # Use Application Default Credentials (works on GCP Cloud Run)
            firebase_admin.initialize_app()
        _firebase_initialized = True
    except Exception as e:
        logger.warning(f"Failed to initialize Firebase Admin SDK: {e}")


def extract_user_id_from_token(token: str | None) -> str | None:
    """Extract user ID (Firebase UID) from Firebase ID token.

    Args:
        token: Firebase ID token from Authorization header (Bearer <token>)

    Returns:
        Firebase UID if token is valid, None otherwise
    """
    if not token or not firebase_admin:
        return None

    try:
        _ensure_firebase_initialized()
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]

        # Verify token and extract UID
        decoded_token = auth.verify_id_token(token)
        user_id = decoded_token.get("uid")
        logger.info(f"Extracted user_id from Firebase token: {user_id}")
        return user_id
    except Exception as e:
        logger.warning(f"Failed to verify Firebase token: {e}")
        return None

