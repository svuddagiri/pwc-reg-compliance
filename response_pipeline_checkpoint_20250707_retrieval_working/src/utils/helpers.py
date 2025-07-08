"""
General helper utilities
"""
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from uuid import uuid4
from src.models.search import Citation


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid4())


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\/\'\"]', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def format_citation(citation: Citation) -> str:
    """
    Format a citation for display
    
    Args:
        citation: Citation object
        
    Returns:
        Formatted citation string
    """
    parts = []
    
    # Add source
    parts.append(f"[{citation.source}")
    
    # Add page if available
    if citation.page:
        parts.append(f", p. {citation.page}")
    
    # Add section if available
    if citation.section:
        parts.append(f", {citation.section}")
    
    parts.append("]")
    
    return "".join(parts)


def calculate_hash(text: str) -> str:
    """
    Calculate SHA256 hash of text
    
    Args:
        text: Text to hash
        
    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode()).hexdigest()


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        overlap: Number of characters to overlap
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation)
    
    Args:
        text: Text to extract keywords from
        max_keywords: Maximum number of keywords
        
    Returns:
        List of keywords
    """
    # Simple keyword extraction based on word frequency
    # In production, use more sophisticated NLP methods
    
    # Remove common words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'cannot'
    }
    
    # Split into words and count frequency
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:max_keywords]]


def format_time_ago(timestamp: datetime) -> str:
    """
    Format timestamp as "time ago" string
    
    Args:
        timestamp: Datetime to format
        
    Returns:
        Formatted string (e.g., "2 hours ago")
    """
    now = datetime.utcnow()
    delta = now - timestamp
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"


def merge_citations(citations: List[Citation]) -> List[Citation]:
    """
    Merge duplicate citations
    
    Args:
        citations: List of citations
        
    Returns:
        List of unique citations
    """
    seen = set()
    unique_citations = []
    
    for citation in citations:
        # Create a unique key for the citation
        key = (citation.source, citation.page, citation.section)
        
        if key not in seen:
            seen.add(key)
            unique_citations.append(citation)
    
    return unique_citations