# %%
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding, TextEmbedding  # miniCOIL lives here
from tqdm import tqdm
from uuid import uuid4
load_dotenv()

# %% Define embedding models
dense_embedder = TextEmbedding(
    model_name="snowflake/snowflake-arctic-embed-s", cache_dir="models"
)
sparse_embedder = SparseTextEmbedding(
    model_name="Qdrant/minicoil-v1", cache_dir="models"
)


# %% Initialize Qdrant client
qdrant_client = QdrantClient(
    url="http://localhost:6333"
)

print("Found collections in Qdrant:")
print("\n".join(i.name for i in qdrant_client.get_collections().collections))

# %%
# Read the corpus into a single string, filtering out blank lines and single special chars
file_path = os.path.join("data", "corpus", "Oxford-Handbook-of-Oncology-4th-Ed.txt")
filtered_lines = []
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        stripped = line.strip()
        # Skip blank lines
        if not stripped:
            continue
        # Skip lines that are a single non-alphanumeric character
        if len(stripped) == 1 and not stripped.isalnum():
            continue
        filtered_lines.append(stripped)

# Join filtered lines into one text blob
text = " ".join(filtered_lines)

# Tokenize into words
words = text.split()

# Set sliding window parameters
chunk_size = 500        # number of words per chunk
overlap = 50              # number of words to overlap between chunks

# Build the list of chunks
chunks = []
for start in range(0, len(words), chunk_size - overlap):
    end = start + chunk_size
    chunk = " ".join(words[start:end])
    if chunk:
        chunks.append(chunk)

print(f"Created {len(chunks)} chunks for embedding")

# %% Embed the chunks
qdrant_client.delete_collection("corpus")
qdrant_client.create_collection(
    collection_name="corpus",
    vectors_config={
        "snowflake_s": models.VectorParams(size=384, distance=models.Distance.COSINE)
    },
    sparse_vectors_config={
        "minicoil": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        )
    },
)

# %% Define embedding models
points = []
for chunk in tqdm(chunks, desc="Embedding chunks"):
    # Get dense and sparse embeddings for the chunk
    dense_vector = list(dense_embedder.embed(chunk))[0]
    sparse_vector = list(sparse_embedder.embed(chunk))[0]

    # Build the PointStruct for Qdrant
    point = models.PointStruct(
        id=str(uuid4()),
        vector={
            "snowflake_s": dense_vector,
            "minicoil": models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
        },
        payload={"text": chunk},
    )
    points.append(point)
    
#%% Upsert points to Qdrant
qdrant_client.upsert(
        collection_name="corpus",
        points=points,
        wait=True,
    )
# %%
