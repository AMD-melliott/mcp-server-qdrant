# Save this as get_vectors.py
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333")
collection_info = client.get_collection(collection_name="mnm_storage")  # Replace with your collection name

# The structure of collection_info varies depending on Qdrant version
# This handles multiple possible structures
try:
    # Newer Qdrant client versions
    if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
        if hasattr(collection_info.config.params, 'vectors'):
            # For newer multi-vector configurations
            if hasattr(collection_info.config.params.vectors, 'keys'):
                vector_names = list(collection_info.config.params.vectors.keys())
                print("Vector names:", vector_names)
            # For single vector configurations or different structure
            elif hasattr(collection_info.config.params, 'vector_name'):
                print("Vector name:", collection_info.config.params.vector_name)
            else:
                print("Single unnamed vector configuration detected")
                print("Full vector config:", collection_info.config.params.vectors)
                # Try accessing vectors directly if it's an object
                print("Vectors dir:", dir(collection_info.config.params.vectors))
        else:
            print("No vectors attribute found in params")
            print("Available attributes:", dir(collection_info.config.params))
    # Older Qdrant client versions
    elif hasattr(collection_info, 'vectors_config'):
        print("Vector names:", list(collection_info.vectors_config.keys()))
    else:
        print("Unknown collection info structure")
        print("Available attributes:", dir(collection_info))
        
    # Print out the entire collection info to help debug
    print("\nFull collection info structure:")
    print(collection_info)
except Exception as e:
    print(f"Error exploring collection info: {e}")
    print("Available attributes on collection_info:", dir(collection_info))