"""Abstract base classes for document processing."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProcessingConfig:
    """Base configuration for processors."""
    pass


class BaseProcessor(ABC):
    """Abstract base class for all document processors."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process the input data."""
        pass
    
    @abstractmethod
    def validate_input(self, input_data: Any) -> bool:
        """Validate the input data."""
        pass


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""
    
    @abstractmethod
    def chunk_document(self, content: str, **kwargs) -> List[Any]:
        """Chunk the document content."""
        pass


class BaseExtractor(ABC):
    """Abstract base class for metadata extractors."""
    
    @abstractmethod
    def extract(self, content: str) -> Dict[str, Any]:
        """Extract metadata from content."""
        pass


class BaseStorage(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def store(self, data: Any) -> str:
        """Store data and return identifier."""
        pass
    
    @abstractmethod
    async def retrieve(self, identifier: str) -> Any:
        """Retrieve data by identifier."""
        pass
    
    @abstractmethod
    async def delete(self, identifier: str) -> bool:
        """Delete data by identifier."""
        pass
    
    @abstractmethod
    async def search(self, query: Dict[str, Any]) -> List[Any]:
        """Search for data based on query."""
        pass