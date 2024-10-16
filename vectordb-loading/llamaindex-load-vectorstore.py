#!/usr/bin/env python3

### System imports
import sys

### Google Vertex AI imports
from google.cloud import aiplatform

### Lllama Index imports
from llama_index.core import (
    StorageContext,
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)
from llama_index.llms.vertex import Vertex
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.vector_stores.vertexaivectorsearch import VertexAIVectorStore

#Basic environment settings
PROJECT_ID = "team-croissants-l200"
REGION = "us-central1"
GCS_BUCKET_NAME = "devsite-vector"
GCS_BUCKET_URI = f"gs://{GCS_BUCKET_NAME}"


aiplatform.init(project=PROJECT_ID, location=REGION)

# TODO : replace 1234567890123456789 with your actual index ID
vs_index = aiplatform.MatchingEngineIndex(index_name="cloud_ac_store_index_1729097500630")

# TODO : replace 1234567890123456789 with your actual endpoint ID
vs_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name="8167185565259661312"
)

# load documents from local path
documents = SimpleDirectoryReader("/Users/donmccasland/devsite-crawler/architecture").load_data()
print(f"# of documents = {len(documents)}")
sys.exit(0)

# setup storage
vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    index_id=vs_index.resource_name,
    endpoint_id=vs_endpoint.resource_name,
    gcs_bucket_name=GCS_BUCKET_NAME,
)

# set storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# configure embedding model
embed_model = VertexTextEmbedding(
    model_name="text-embedding-004",
    project=PROJECT_ID,
    location=REGION,
)

vertex_gemini = Vertex(model="gemini-pro", temperature=0, additional_kwargs={})

# setup the index/query process, ie the embedding model (and completion if used)
Settings.llm = vertex_gemini
Settings.embed_model = embed_model

# define index from vector store
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

query_engine = index.as_query_engine()

response = query_engine.query(
    "who are the authors of paper Attention is All you need?"
)

print(f"Response:")
print("-" * 80)
print(response.response)
print("-" * 80)
print(f"Source Documents:")
print("-" * 80)
for source in response.source_nodes:
    print(f"Sample Text: {source.text[:50]}")
    print(f"Relevance score: {source.get_score():.3f}")
    print(f"File Name: {source.metadata.get('file_name')}")
    print(f"Page #: {source.metadata.get('page_label')}")
    print(f"File Path: {source.metadata.get('file_path')}")
    print("-" * 80)

