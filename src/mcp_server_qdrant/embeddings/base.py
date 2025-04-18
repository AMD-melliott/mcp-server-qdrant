import os
from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.
    """
    
    @abstractmethod
    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        pass

    @abstractmethod
    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        pass

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Can be overridden with an environment variable.
        """
        force_name = os.environ.get("FORCE_VECTOR_NAME")
        if force_name:
            return force_name
        # If no forced name, use the provider's implementation
        return self._get_vector_name()
    
    @abstractmethod
    def _get_vector_name(self) -> str:
        """Get the default vector name for this provider."""
        pass

    @abstractmethod
    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        pass
