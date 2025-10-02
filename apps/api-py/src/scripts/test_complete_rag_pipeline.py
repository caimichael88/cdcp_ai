"""
Complete RAG Pipeline Test: Scrape → Chunk → Embed → Store → Search
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.scraper_service import WebScraperService, DocumentChunker
from services.embedding_service_with_transformer import EmbeddingService
from services.vector_db_service import VectorDBService

print("=" * 70)
print("CDCP AI - Complete RAG Pipeline Test")
print("=" * 70)

# ============================================================================
# Step 1: Scrape CDCP Pages
# ============================================================================
print("\n[Step 1/5] Scraping CDCP pages...")
scraper = WebScraperService(
    base_urls=["https://www.canada.ca/en/services/benefits/dental/dental-care-plan/qualify.html"],
    max_pages=3,  # Start with 3 pages
    allowed_paths=['/dental-care-plan/']
)
documents = scraper.crawl()
print(f"✓ Scraped {len(documents)} documents")
for i, doc in enumerate(documents):
    print(f"  {i+1}. {doc.title[:60]}...")

# ============================================================================
# Step 2: Chunk Documents
# ============================================================================
print("\n[Step 2/5] Chunking documents...")
chunker = DocumentChunker(chunk_size=512, overlap=50)
all_chunks = []
for doc in documents:
    chunks = chunker.chunk_document(doc)
    all_chunks.extend(chunks)
    print(f"  {doc.title[:50]}... → {len(chunks)} chunks")

print(f"✓ Total chunks: {len(all_chunks)}")

# ============================================================================
# Step 3: Generate Embeddings
# ============================================================================
print("\n[Step 3/5] Generating embeddings...")
embedding_service = EmbeddingService()

# Get text content from chunks
chunk_texts = [chunk.content for chunk in all_chunks]
print(f"  Processing {len(chunk_texts)} chunks...")

embeddings = embedding_service.embed_documents(chunk_texts)
print(f"✓ Generated {len(embeddings)} embeddings")
print(f"  Embedding dimensions: {embeddings.shape[1]}")

# ============================================================================
# Step 4: Store in Vector Database
# ============================================================================
print("\n[Step 4/5] Storing in ChromaDB...")
vector_db = VectorDBService(
    collection_name="cdcp_documents",
    persist_directory="./chroma_db"
)

# Prepare metadata for each chunk
metadatas = []
chunk_ids = []
for i, chunk in enumerate(all_chunks):
    metadatas.append({
        "title": chunk.title,
        "url": chunk.url,
        "section": chunk.section or "general",
        "doc_id": chunk.doc_id,
        "language": chunk.language,
        "chunk_length": len(chunk.content)
    })
    chunk_ids.append(f"{chunk.doc_id}_chunk_{i}")

# Add to vector database
vector_db.add_documents_batch(
    documents=chunk_texts,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    batch_size=50
)

print(f"✓ Stored {len(chunk_texts)} chunks in ChromaDB")
print(f"  Database location: ./chroma_db")

# ============================================================================
# Step 5: Test Semantic Search
# ============================================================================
print("\n[Step 5/5] Testing semantic search...")

# Test queries
test_queries = [
    "Who is eligible for CDCP?",
    "How do I apply for dental coverage?",
    "What services are covered?"
]

for query in test_queries:
    print(f"\n{'─' * 70}")
    print(f"Query: '{query}'")
    print(f"{'─' * 70}")

    # Generate query embedding
    query_embedding = embedding_service.embed_query(query)

    # Search vector database
    results = vector_db.search(
        query_embeddings=[query_embedding.tolist()],
        n_results=3
    )

    # Display results
    for i, (doc_id, document, metadata, distance) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        print(f"\n  Result {i+1} (similarity: {1 - distance:.4f}):")
        print(f"    Source: {metadata['title'][:50]}...")
        print(f"    Section: {metadata['section']}")
        print(f"    URL: {metadata['url']}")
        print(f"    Content: {document[:150]}...")

# ============================================================================
# Statistics
# ============================================================================
print("\n" + "=" * 70)
print("Pipeline Complete! ✓")
print("=" * 70)

stats = vector_db.get_stats()
scraper_stats = scraper.get_stats()

print("\nFinal Statistics:")
print(f"  Documents scraped: {len(documents)}")
print(f"  Total chunks created: {len(all_chunks)}")
print(f"  Embeddings generated: {len(embeddings)}")
print(f"  Vectors in database: {stats['total_documents']}")
print(f"  Embedding dimensions: {embeddings.shape[1]}")
print(f"  Scraping time: {scraper_stats['total_time']:.2f}s")
print(f"  Success rate: {scraper_stats['success_rate']:.1%}")

print("\n" + "=" * 70)
print("Your RAG system is ready to use!")
print("=" * 70)
print("\nNext steps:")
print("  1. Add more pages by increasing max_pages")
print("  2. Query the system with: vector_db.search_by_text(['your question'])")
print("  3. Build an API endpoint to expose the search functionality")
