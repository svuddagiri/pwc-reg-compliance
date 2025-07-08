"""Base data models for the regulatory chatbot."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class DocumentType(Enum):
    """Types of regulatory documents."""
    REGULATION = "regulation"
    DIRECTIVE = "directive"
    GUIDELINE = "guideline"
    POLICY = "policy"
    STANDARD = "standard"
    ACT = "act"
    LAW = "law"
    OTHER = "other"


class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class BaseDocument:
    """Base document model."""
    document_id: str
    name: str
    content: str
    document_type: DocumentType
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseChunk:
    """Base chunk model."""
    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaseMetadata:
    """Base metadata model."""
    document_id: str
    title: str
    created_at: datetime
    extracted_at: datetime = field(default_factory=datetime.utcnow)
    attributes: Dict[str, Any] = field(default_factory=dict)