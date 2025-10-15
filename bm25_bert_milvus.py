from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, Function, FunctionType
from sentence_transformers import SentenceTransformer

# 1. Connect to Milvus
print("Connecting to milvus")
connections.connect("default", host="127.0.0.1", port="19530")

# Drop collection if exists (for clean start)
if utility.has_collection("hybrid_demo_native"):
    utility.drop_collection("hybrid_demo_native")

# 2. Define BM25 Function
bm25_function = Function(
    name="bm25",
    function_type=FunctionType.BM25,
    input_field_names=["text"],
    output_field_names=["sparse_vector"]
)

# 3. Define Schema with BM25 Function
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(
        name="text", 
        dtype=DataType.VARCHAR, 
        max_length=1024,
        enable_analyzer=True  # Required for BM25
    ),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(
        name="sparse_vector",
        dtype=DataType.SPARSE_FLOAT_VECTOR,
        is_function_output=True  # Mark as function output
    )
]

schema = CollectionSchema(
    fields,
    description="Hybrid BERT + BM25 using SparseVectorField",
    functions=[bm25_function]  # Pass Function object, not dict
)

collection_name = "hybrid_demo_native"
collection = Collection(collection_name, schema)

# 4. Prepare Documents
docs = [
    "Milvus is a vector database built for AI applications.",
    "BM25 is a ranking function used by search engines.",
    "BERT is a transformer model for language understanding.",
    "Milvus supports hybrid search combining BM25 and embeddings.",
]

# 5. BERT Embeddings
print("Generating BERT embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(docs).tolist()

# 6. Insert Data (without sparse_vector - Milvus generates it)
print("Inserting data...")
collection.insert([docs, embeddings])
collection.flush()

# 6. Create Indexes
print("Creating indexes...")
# Semantic index (BERT)
collection.create_index(
    "embedding",
    {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 100}
    }
)

# BM25 index (sparse vector)
collection.create_index(
    "sparse_vector",
    {
        "index_type": "SPARSE_INVERTED_INDEX",
        "metric_type": "BM25"
    }
)

# 7. Load Collection
print("Loading collection...")
collection.load()

# 8. Hybrid Search
query_text = "hybrid search with embeddings"
query_embedding = model.encode([query_text]).tolist()

print("\n=== Performing Hybrid Search ===")
# Use AnnSearchRequest for hybrid search
from pymilvus import AnnSearchRequest, RRFRanker

# Dense vector search request
dense_req = AnnSearchRequest(
    data=query_embedding,
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
    limit=10
)

# Sparse vector search request (BM25)
sparse_req = AnnSearchRequest(
    data=[query_text],
    anns_field="sparse_vector",
    param={"metric_type": "BM25"},
    limit=10
)

# Perform hybrid search with RRF (Reciprocal Rank Fusion)
res = collection.hybrid_search(
    reqs=[dense_req, sparse_req],
    rerank=RRFRanker(),
    limit=5,
    output_fields=["text"]
)

print("\n=== Hybrid Search Results ===")
for hit in res[0]:
    print(f"Text: {hit.entity.get('text')}")
    print(f"Score: {hit.score:.4f}")
    print()

# Cleanup
print("\nClosing connection...")
connections.disconnect("default")