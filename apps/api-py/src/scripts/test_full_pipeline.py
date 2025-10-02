"""
Test the complete pipeline: Scrape → Chunk → Embed
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.scraper_service import WebScraperService, DocumentChunker
from services.embedding_service_with_transformer import EmbeddingService

print("=" * 60)
print("CDCP AI - Complete Pipeline Test")
print("=" * 60)

# Step 1: Scrape CDCP pages
print("\n[Step 1] Scraping CDCP pages...")
scraper = WebScraperService(
    base_urls=["https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"],
    max_pages=3,  # Start small
    allowed_paths=['/dental-care-plan/']
)
documents = scraper.crawl()
print(f"✓ Scraped {len(documents)} documents")

# Step 2: Chunk documents
print("\n[Step 2] Chunking documents...")
chunker = DocumentChunker(chunk_size=512, overlap=50)
all_chunks = []
for doc in documents:
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)
    print(f"  - {doc.title[:50]}... → {len(chunks)} chunks")

print(f"✓ Total chunks: {len(all_chunks)}")

# Step 3: Generate embeddings
print("\n[Step 3] Generating embeddings...")
embedding_service = EmbeddingService()

# Get text content from chunks
chunk_texts = [chunk.content for chunk in all_chunks[:5]]  # Test with first 5 chunks
print(f"  Embedding {len(chunk_texts)} chunks...")

embeddings = embedding_service.embed_documents(chunk_texts)
print(f"✓ Generated {len(embeddings)} embeddings")
print(f"  Embedding dimensions: {embeddings.shape[1]}")

# Step 4: Show results
print("\n[Step 4] Sample Results:")
print("=" * 60)
for i, (chunk, embedding) in enumerate(zip(all_chunks[:3], embeddings[:3])):
    print(f"\nChunk {i+1}:")
    print(f"  Source: {chunk.title}")
    print(f"  Length: {len(chunk.content)} chars")
    print(f"  Preview: {chunk.content[:100]}...")
    print(f"  Embedding: [{embedding[0]:.4f}, {embedding[1]:.4f}, ..., {embedding[-1]:.4f}]")
    print(f"  Vector norm: {(embedding ** 2).sum() ** 0.5:.4f}")

print("\n" + "=" * 60)
print("Pipeline Complete! ✓")
print("=" * 60)
print("\nSummary:")
print(f"  Documents scraped: {len(documents)}")
print(f"  Total chunks: {len(all_chunks)}")
print(f"  Embeddings created: {len(embeddings)}")
print(f"  Embedding dimension: {embeddings.shape[1]}")
print(f"\nNext step: Store embeddings in vector database (ChromaDB/FAISS)")
