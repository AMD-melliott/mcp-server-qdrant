#Find your vector name using Qdrant client:
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
collection_info = client.get_collection(collection_name="your_collection_name")
print("Vector names:", list(collection_info.config.params.vectors.keys()))