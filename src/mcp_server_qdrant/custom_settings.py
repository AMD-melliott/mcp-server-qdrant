import os
from pydantic import BaseSettings, Field
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

class CustomEmbeddingProviderSettings(BaseSettings):
    """
    Custom settings class that handles string conversion for provider_type.
    """
    _provider_type_raw: str = Field(
        default="fastembed",
        alias="EMBEDDING_PROVIDER"
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL"
    )
    
    @property
    def provider_type(self) -> EmbeddingProviderType:
        """Convert string to enum value, with fallback to default."""
        raw_value = self._provider_type_raw.lower()
        for enum_val in EmbeddingProviderType:
            if enum_val.value == raw_value:
                return enum_val
        # Default if not found
        print(f"Warning: Unknown provider type '{raw_value}', using default 'fastembed'")
        return EmbeddingProviderType.FASTEMBED