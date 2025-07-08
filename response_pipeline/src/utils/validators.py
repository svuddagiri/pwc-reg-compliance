"""
Input validation utilities
"""
import re
from typing import Optional
from uuid import UUID


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format (UUID)
    
    Args:
        session_id: Session ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        UUID(session_id)
        return True
    except (ValueError, AttributeError):
        return False


def validate_query_length(
    query: str,
    min_length: int = 1,
    max_length: int = 1000
) -> tuple[bool, Optional[str]]:
    """
    Validate query length
    
    Args:
        query: Query string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not query or not query.strip():
        return False, "Query cannot be empty"
    
    query_length = len(query.strip())
    
    if query_length < min_length:
        return False, f"Query must be at least {min_length} characters"
    
    if query_length > max_length:
        return False, f"Query cannot exceed {max_length} characters"
    
    return True, None


def validate_regulation_name(regulation: str) -> bool:
    """
    Validate regulation name format
    
    Args:
        regulation: Regulation name to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Allow alphanumeric, spaces, hyphens, and underscores
    pattern = r"^[A-Za-z0-9\s\-_]+$"
    return bool(re.match(pattern, regulation))


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks
    
    Args:
        text: Text to sanitize
        
    Returns:
        Sanitized text
    """
    # Remove any potential script tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Escape special characters
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    text = text.replace("'", '&#x27;')
    
    return text.strip()


def validate_search_filters(filters: dict) -> tuple[bool, Optional[str]]:
    """
    Validate search filters
    
    Args:
        filters: Dictionary of search filters
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    allowed_keys = {
        'regulation', 'category', 'subcategory', 'document_type',
        'date_from', 'date_to', 'section', 'article'
    }
    
    # Check for unknown keys
    unknown_keys = set(filters.keys()) - allowed_keys
    if unknown_keys:
        return False, f"Unknown filter keys: {', '.join(unknown_keys)}"
    
    # Validate specific filter values
    if 'regulation' in filters:
        if isinstance(filters['regulation'], list):
            for reg in filters['regulation']:
                if not validate_regulation_name(reg):
                    return False, f"Invalid regulation name: {reg}"
        elif isinstance(filters['regulation'], str):
            if not validate_regulation_name(filters['regulation']):
                return False, f"Invalid regulation name: {filters['regulation']}"
    
    return True, None