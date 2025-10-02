"""
RAG Controller
API endpoints for scraping, embedding, storing, and searching CDCP documents
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.scraper_service import WebScraperService, DocumentChunker
from ..services.embedding_service_with_transformer import EmbeddingService
from ..services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["RAG"])

# Initialize services (singleton pattern)
embedding_service = None
vector_db = None


def get_embedding_service():
    """Lazy load embedding service"""
    global embedding_service
    if embedding_service is None:
        logger.info("Loading embedding service...")
        embedding_service = EmbeddingService()
    return embedding_service


def get_vector_db():
    """Lazy load vector database"""
    global vector_db
    if vector_db is None:
        logger.info("Loading vector database...")
        vector_db = VectorDBService(
            collection_name="cdcp_documents",
            persist_directory="./chroma_db"
        )
    return vector_db


# ============================================================================
# Request/Response Models
# ============================================================================

class IngestionRequest(BaseModel):
    """Request model for scraping and ingesting CDCP pages"""
    base_urls: List[str] = Field(..., description="Starting URLs to scrape")
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum pages to scrape")
    allowed_paths: Optional[List[str]] = Field(
        default=["/dental-care-plan/"],
        description="URL paths to filter"
    )
    chunk_size: int = Field(default=512, ge=100, le=2000, description="Chunk size in tokens")
    chunk_overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class IngestionResponse(BaseModel):
    """Response model for ingestion operation"""
    success: bool
    scraped_documents: int
    chunked_documents: int
    ingested_documents: int
    scraping_time: float
    message: str


class SearchRequest(BaseModel):
    """Request model for searching documents"""
    query: str = Field(..., description="Search query")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    filter_section: Optional[str] = Field(None, description="Filter by section")


class SearchResult(BaseModel):
    """Individual search result"""
    content: str
    title: str
    url: str
    section: str
    similarity_score: float


class SearchResponse(BaseModel):
    """Response model for search operation"""
    query: str
    results: List[SearchResult]
    total_results: int


class StatsResponse(BaseModel):
    """Response model for database statistics"""
    total_documents: int
    collection_name: str


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/ingest", response_model=IngestionResponse)
async def ingest_content(request: IngestionRequest):
    """
    Scrape CDCP pages, chunk, embed, and store in vector database

    Complete RAG pipeline:
    1. Scrapes web pages from provided URLs
    2. Chunks the documents into smaller pieces
    3. Generates embeddings for each chunk
    4. Stores everything in ChromaDB
    """
    try:
        logger.info(f"Starting ingestion for {len(request.base_urls)} URLs")

        # Step 1: Scrape documents
        scraper = WebScraperService(
            base_urls=request.base_urls,
            max_pages=request.max_pages,
            allowed_paths=request.allowed_paths
        )
        documents = scraper.crawl()

        if not documents:
            raise HTTPException(status_code=404, detail="No documents were scraped")

        # Step 2: Chunk documents
        logger.info(f"Chunking {len(documents)} documents")
        chunker = DocumentChunker(
            chunk_size=request.chunk_size,
            overlap=request.chunk_overlap
        )

        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=500, detail="No chunks created")

        # Step 3: Generate embeddings
        logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
        emb_service = get_embedding_service()
        chunk_texts = [chunk.content for chunk in all_chunks]
        embeddings = emb_service.embed_documents(chunk_texts)

        # Step 4: Store in vector database
        logger.info("Storing in vector database")
        db = get_vector_db()

        # Prepare metadata and deterministic IDs
        metadatas = []
        chunk_ids = []

        for i, chunk in enumerate(all_chunks):
            metadatas.append({
                "title": chunk.title,
                "url": chunk.url,
                "section": chunk.section or "general",
                "doc_id": chunk.doc_id,
                "language": chunk.language
            })
            # Create deterministic ID: doc_id is MD5 of URL, append chunk index
            # This ensures same URL always gets same IDs
            chunk_ids.append(f"{chunk.doc_id}_chunk_{i}")

        # Store in batches with deterministic IDs (will upsert if URL re-scraped)
        db.add_documents_batch(
            documents=chunk_texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=chunk_ids,
            batch_size=50
        )

        stats = scraper.get_stats()

        return IngestionResponse(
            success=True,
            scraped_documents=len(documents),
            chunked_documents=len(all_chunks),
            ingested_documents=len(all_chunks),
            scraping_time=stats['total_time'],
            message=f"Successfully ingested {len(documents)} documents ({len(all_chunks)} chunks)"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ingest_content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for documents using semantic similarity

    Performs semantic search across all stored documents.
    """
    try:
        # Generate query embedding
        emb_service = get_embedding_service()
        query_embedding = emb_service.embed_query(request.query)

        # Search vector database
        db = get_vector_db()

        where_filter = None
        if request.filter_section:
            where_filter = {"section": request.filter_section}

        results = db.search(
            query_embeddings=[query_embedding.tolist()],
            n_results=request.n_results,
            where=where_filter
        )

        # Format results
        search_results = []
        for i in range(len(results['ids'][0])):
            search_results.append(SearchResult(
                content=results['documents'][0][i],
                title=results['metadatas'][0][i]['title'],
                url=results['metadatas'][0][i]['url'],
                section=results['metadatas'][0][i]['section'],
                similarity_score=1 - results['distances'][0][i]
            ))

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results)
        )

    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    try:
        db = get_vector_db()
        stats = db.get_stats()

        return StatsResponse(
            total_documents=stats['total_documents'],
            collection_name=stats['collection_name']
        )

    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_database():
    """Clear all documents from the database"""
    try:
        db = get_vector_db()
        db.clear_collection()

        return {"status": "success", "message": "Database cleared"}

    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        emb_service = get_embedding_service()
        db = get_vector_db()

        return {
            "status": "healthy",
            "embedding_service": "loaded",
            "vector_db": "loaded",
            "total_documents": db.count()
        }

    except Exception as e:
        logger.error(f"Error in health_check: {e}")
        raise HTTPException(status_code=503, detail=str(e))
