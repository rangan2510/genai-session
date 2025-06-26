# %%
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding
from regex import search  # miniCOIL lives here

# %% Define embedding models
dense_embedder = TextEmbedding(
    model_name="snowflake/snowflake-arctic-embed-s", cache_dir="models"
)
sparse_embedder = SparseTextEmbedding(
    model_name="Qdrant/minicoil-v1", cache_dir="models"
)

qdrant_client = QdrantClient(
    url="http://localhost:6333",
    prefer_grpc=True, timeout=300
)

# %%
async def search_local_corpus(
    query_text : str,
    count: int = 50,
):
    """
    Search the local corpus that contains chapters from Oncology handbook.
    """

    client = QdrantClient(url="http://localhost:6333", prefer_grpc=True, timeout=300)
    
    dense_vector = list(dense_embedder.embed(query_text))[0]
    sparse_vector = list(sparse_embedder.embed(query_text))[0]
    sparse_vector = models.SparseVector(
        indices=sparse_vector.indices,
        values=sparse_vector.values,
    )

    fast_dense_prefetcher = models.Prefetch(
        query=dense_vector,
        using="snowflake_s",
        limit=250,
    )

    prefetcher = models.Prefetch(
        prefetch=fast_dense_prefetcher, query=sparse_vector, using="minicoil", limit=50
    )

    response = client.query_points(
        collection_name="corpus",
        prefetch=prefetcher,
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
        ),
        limit= count,
    )

    # return response
    result = [i.payload for i in response.points]
    return result

#%%
# Example usage
# search_local_corpus("breast cancer", count=5)
# %%
