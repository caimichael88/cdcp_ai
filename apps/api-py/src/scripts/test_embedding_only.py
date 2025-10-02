"""
Test embedding service only
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

print("Testing embedding service...")

from services.embedding_service_with_transformer import EmbeddingService

# Initialize service
print("Loading embedding model...")
embedding_service = EmbeddingService()
print("✓ Model loaded")

# Test with sample texts
texts = [
    "To qualify for the CDCP, you must be a Canadian resident.",
    "The income limit is $90,000 per year.",
    "Dental coverage includes preventive and restorative care."
]

print(f"\nEmbedding {len(texts)} sample texts...")
embeddings = embedding_service.embed_documents(texts)

print(f"✓ Generated {len(embeddings)} embeddings")
print(f"  Embedding dimensions: {embeddings.shape}")
print(f"  First embedding sample: [{embeddings[0][0]:.4f}, {embeddings[0][1]:.4f}, ..., {embeddings[0][-1]:.4f}]")

print("\n✓ Embedding service working correctly!")
