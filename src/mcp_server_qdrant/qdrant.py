import logging
import uuid
from typing import Any, Dict, Optional, Set, Union, List

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)

Metadata = Dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """
    content: str
    metadata: Optional[Metadata] = None
    vector: Optional[List[float]] = None
    score: Optional[float]] = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: Optional[str],
        qdrant_api_key: Optional[str],
        collection_name: Optional[str],
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: Optional[str] = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        # Cache to track collections with known vector name mismatches
        self._collections_with_unnamed_vectors: Set[str] = set()

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: Optional[str] = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        
        # Check if we already know this collection has unnamed vectors
        if collection_name in self._collections_with_unnamed_vectors:
            await self._store_with_unnamed_vector(entry, collection_name)
            return
            
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([entry.content])
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, "metadata": entry.metadata}
        
        try:
            # First try with named vector format (for multi-vector collections)
            await self._client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=uuid.uuid4().hex,
                        vector={vector_name: embeddings[0]},
                        payload=payload,
                    )
                ],
            )
            logger.debug(f"Successfully stored entry using named vector: {vector_name}")
        except Exception as e:
            # If that fails, try with unnamed vector format (for single unnamed vector collections)
            error_message = str(e).lower()
            logger.warning(f"Upsert with vector name '{vector_name}' failed: {e}. Trying without vector name.")
            
            try:
                await self._client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=uuid.uuid4().hex,
                            vector=embeddings[0],  # Direct vector without name
                            payload=payload,
                        )
                    ],
                )
                # Remember this collection uses unnamed vectors
                self._collections_with_unnamed_vectors.add(collection_name)
                logger.info(f"Collection {collection_name} uses unnamed vectors, storing this preference")
            except Exception as fallback_error:
                # If both approaches fail, log the error and re-raise
                logger.error(f"Upsert without vector name also failed: {fallback_error}")
                raise

    async def _store_with_unnamed_vector(self, entry: Entry, collection_name: str):
        """Helper method to store an entry with unnamed vector format"""
        await self._ensure_collection_exists(collection_name)
        
        # Embed the document
        embeddings = await self._embedding_provider.embed_documents([entry.content])
        payload = {"document": entry.content, "metadata": entry.metadata}
        
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=embeddings[0],  # Direct vector without name
                    payload=payload,
                )
            ],
        )
        logger.debug(f"Stored entry using unnamed vector in collection {collection_name}")

    async def search(
        self, query: str, *, collection_name: Optional[str] = None, limit: int = 10,
        offset: int = 0, score_threshold: Optional[float] = None, with_vectors: bool = False,
        with_payload: Union[bool, List[str]] = True
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param offset: The number of entries to skip (for pagination).
        :param score_threshold: Minimum similarity score threshold. Only entries with scores above 
                                this threshold will be returned.
        :param with_vectors: Whether to include vector data in the results.
        :param with_payload: Whether to include payload data in the results. Can be a boolean or 
                             a list of payload field names to include.
        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        query_vector = await self._embedding_provider.embed_query(query)
        
        # If we already know this collection uses unnamed vectors, skip the named vector attempt
        if collection_name in self._collections_with_unnamed_vectors:
            try:
                search_results = await self._client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit,
                )
                logger.debug(f"Searched collection {collection_name} with unnamed vector format (from cache)")
            except Exception as e:
                logger.error(f"Search with unnamed vector failed: {e}")
                return []
        else:
            # Try with named vector first
            vector_name = self._embedding_provider.get_vector_name()
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
                    # Remember this collection uses unnamed vectors for future queries
                    self._collections_with_unnamed_vectors.add(collection_name)
                    logger.info(f"Collection {collection_name} uses unnamed vectors, storing preference for future queries")
                except Exception as fallback_error:
                    # If both approaches fail, log the error and return empty results
                    logger.error(f"Query without vector name also failed: {fallback_error}")
                    return []

        # Process the results, handling potential missing keys in the payload
        results = []
        for result in search_results.points:
            payload = result.payload or {}
            
            # Replace excessive debug prints with a single log message at debug level
            logger.debug(f"Processing search result with score: {result.score}")
            
            # Handle missing document field or alternative field names
            content = payload.get("document")
            
            # If document field is missing, try other commonly used field names
            if content is None:
                # Try common alternative field names for the content
                for field in ["text", "content", "page_content", "chunk"]:
                    if field in payload:
                        content = payload[field]
                        logger.debug(f"Found content in field: {field}")
                        break
            
            # If still no content found, use a readable representation of the payload
            if content is None:
                content = f"[No document content found. Available payload keys: {', '.join(payload.keys())}]"
                logger.debug("No content field found in payload")
            
            # After extracting content, convert to string if needed
            if content is not None and not isinstance(content, str):
                logger.debug(f"Converting content from type {type(content)} to string")
                content = str(content)
            
            # Create entry with vector data and score if requested
            entry = Entry(
                content=content,
                metadata=payload.get("metadata"),
            )
            
            # Add vector if available and requested
            if with_vectors and hasattr(result, "vector"):
                entry.vector = result.vector
            
            # Always include score when available
            if hasattr(result, "score"):
                entry.score = result.score
            
            results.append(entry)

        return results

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()
            vector_name = self._embedding_provider.get_vector_name()
            
            # For backward compatibility with existing unnamed vector collections
            if not vector_name:  # If vector_name is empty, create unnamed vector collection
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    ),
                )
            else:  # Otherwise create named vector collection
                await self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config={
                        vector_name: models.VectorParams(
                            size=vector_size,
                            distance=models.Distance.COSINE,
                        )
                    },
                )
