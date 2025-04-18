import asyncio
from typing import List

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

class BAAIEmbeddingProvider(EmbeddingProvider):
    """
    BAAI implementation of the embedding provider.
    :param model_name: The name of the BAAI model to use.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        # Import BAAI embeddings library here to avoid importing it globally
        try:
            from FlagEmbedding import FlagModel
            self.embedding_model = FlagModel(model_name, use_fp16=True)
        except ImportError:
            raise ImportError("Please install the FlagEmbedding package: pip install FlagEmbedding")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since BAAI embeddings might be synchronous
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(documents)
        )
        return embeddings.tolist() if hasattr(embeddings, 'tolist') else [e.tolist() for e in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since BAAI embeddings might be synchronous
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.embedding_model.encode(query)
        )
        return embedding.tolist() if hasattr(embedding, 'tolist') else embedding[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        This should match the vector name used when creating the collection.
        """
        # Use a consistent naming scheme that includes the model name
        model_short_name = self.model_name.split("/")[-1].lower()
        return f"baai-{model_short_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        # You might need to adjust this based on your specific BAAI model
        # For example, BAAI embeddings might be 768 or 1024 dimensions
        # If you can't determine this programmatically, you might need to hardcode it
        # based on the model you're using
        
        # Sample vector to determine size
        sample_embedding = self.embedding_model.encode(["test"])
        if hasattr(sample_embedding, 'shape'):
            return sample_embedding.shape[1]
        else:
            return len(sample_embedding[0])

    def _get_vector_name(self) -> str:
        """Get the default vector name for this provider."""
        model_short_name = self.model_name.split("/")[-1].lower()
        return f"baai-{model_short_name}"