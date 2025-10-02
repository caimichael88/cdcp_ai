"""
Test Vector DB Service Only
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Testing Vector DB Service...")

from services.vector_db_service import VectorDBService

# Initialize vector database
print("\n1. Initializing ChromaDB...")
vector_db = VectorDBService(
    collection_name="test_collection",
    persist_directory="./test_chroma_db"
)
print(f"✓ ChromaDB initialized")

# Sample data
print("\n2. Adding sample documents...")
documents = [
    "To qualify for the CDCP, you must be a Canadian resident.",
    "The income limit is $90,000 per year.",
    "Dental coverage includes preventive and restorative care."
]

# Simple embeddings (mock data for testing)
import numpy as np
embeddings = np.random.rand(3, 384).tolist()  # 384-dim vectors

metadatas = [
    {"title": "Eligibility", "section": "eligibility"},
    {"title": "Income Requirements", "section": "eligibility"},
    {"title": "Coverage", "section": "coverage"}
]

vector_db.add_documents(
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)
print(f"✓ Added {len(documents)} documents")

# Get stats
print("\n3. Database statistics:")
stats = vector_db.get_stats()
print(f"  Total documents: {stats['total_documents']}")
print(f"  Collection: {stats['collection_name']}")
print(f"  Location: {stats['persist_directory']}")

# Search
print("\n4. Testing search...")
query_embedding = np.random.rand(1, 384).tolist()
results = vector_db.search(
    query_embeddings=query_embedding,
    n_results=2
)

print(f"  Found {len(results['ids'][0])} results")
for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
    print(f"    {i+1}. {meta['title']} - {doc[:50]}...")

print("\n✓ Vector DB test successful!")
