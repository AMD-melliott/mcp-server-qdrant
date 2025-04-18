import os
from abc import ABC, abstractmethod
from typing import List, Optional


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

    async def search(
        self, query: str, *, collection_name: Optional[str] = None, limit: int = 10
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        try:
            # First try with the vector name (for multi-vector collections)
            search_results = await self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                using=vector_name,
                limit=limit,
            )
        except Exception as e:
            # If that fails, try without specifying the vector name (for single unnamed vector collections)
            logger.warning(f"Query with vector name '{vector_name}' failed: {e}. Trying without vector name.")
            try:
                search_results = await self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                )
            except Exception as fallback_error:
                # If both approaches fail, log the error and return empty results
                logger.error(f"Query without vector name also failed: {fallback_error}")
                return []

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
            )
            for result in search_results.points
        ]
