import json
import logging
from typing import Any, List

from mcp.server.fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: str,
            collection_name: str,
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def store_with_default_collection(
            ctx: Context,
            information: str,
            metadata: Metadata = None,  # type: ignore
        ) -> str:
            assert self.qdrant_settings.collection_name is not None
            return await store(
                ctx, information, self.qdrant_settings.collection_name, metadata
            )

        async def find(
            ctx: Context,
            query: str,
            collection_name: str,
            limit: int = None,
            score_threshold: float = None,
            with_vectors: bool = None,
        ) -> List[str]:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param limit: Maximum number of results to return. If not provided, uses server default.
            :param score_threshold: Minimum similarity score (0-1) for results. If not provided, uses server default.
            :param with_vectors: Whether to include vector data in results. Default is False.
            :return: A list of entries found.
            """
            await ctx.debug(f"Finding results for query {query}")
            if collection_name:
                await ctx.debug(f"Overriding the collection name with {collection_name}")
            
            # Apply defaults from server settings if not provided by client
            actual_limit = limit if limit is not None else self.qdrant_settings.search_limit
            actual_score_threshold = score_threshold if score_threshold is not None else self.qdrant_settings.search_score_threshold
            actual_with_vectors = with_vectors if with_vectors is not None else self.qdrant_settings.search_with_vectors
            
            # Enforce maximum limit to prevent abuse
            if actual_limit > self.qdrant_settings.max_context_limit:
                await ctx.debug(f"Limiting results to max_context_limit: {self.qdrant_settings.max_context_limit}")
                actual_limit = self.qdrant_settings.max_context_limit
            
            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=actual_limit,
                score_threshold=actual_score_threshold,
                with_vectors=actual_with_vectors,
            )
            
            if not entries:
                return [f"No information found for the query '{query}'"]
            
            content = [
                f"Results for the query '{query}'",
            ]
            
            for entry in entries:
                # Include score in the formatted output if available
                score_info = f" (score: {entry.score:.4f})" if entry.score is not None else ""
                content.append(f"{self.format_entry(entry)}{score_info}")
            
            return content

        async def find_with_default_collection(
            ctx: Context,
            query: str,
        ) -> List[str]:
            assert self.qdrant_settings.collection_name is not None
            return await find(ctx, query, self.qdrant_settings.collection_name)

        # Register the tools depending on the configuration

        if self.qdrant_settings.collection_name:
            self.add_tool(
                find_with_default_collection,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )
        else:
            self.add_tool(
                find,
                name="qdrant-find",
                description=self.tool_settings.tool_find_description,
            )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database

            if self.qdrant_settings.collection_name:
                self.add_tool(
                    store_with_default_collection,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
            else:
                self.add_tool(
                    store,
                    name="qdrant-store",
                    description=self.tool_settings.tool_store_description,
                )
