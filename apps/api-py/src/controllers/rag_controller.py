"""
RAG Controller
API endpoints for scraping, embedding, storing, and searching CDCP documents
"""

import logging

from fastapi import APIRouter, HTTPException

from ..services.rag_service import RAGService
from ..schemas.rag_schemas import (
    IngestionRequest,
    IngestionResponse,
    SearchRequest,
    SearchResult,
    SearchResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rag", tags=["RAG"])

# Service singleton
_rag_service = None


def get_rag_service() -> RAGService:
    """Lazy load RAG service"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService(
            collection_name="cdcp_documents",
            persist_directory="./chroma_db"
        )
    return _rag_service


# API Endpoints
@router.post("/ingest", response_model=IngestionResponse)
async def ingest_content(request: IngestionRequest):
    """Scrape, chunk, embed, and store CDCP documents in vector database"""
    try:
        result = get_rag_service().ingest_content(
            base_urls=request.base_urls,
            max_pages=request.max_pages,
            allowed_paths=request.allowed_paths,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        return IngestionResponse(**result.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search for documents using semantic similarity"""
    try:
        result = get_rag_service().search(
            query=request.query,
            n_results=request.n_results,
            filter_section=request.filter_section
        )
        return SearchResponse(
            query=result.query,
            results=[SearchResult(**r.__dict__) for r in result.results],
            total_results=result.total_results
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    try:
        return StatsResponse(**get_rag_service().get_stats())
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear")
async def clear_database():
    """Clear all documents from the database"""
    try:
        get_rag_service().clear_database()
        return {"status": "success", "message": "Database cleared"}
    except Exception as e:
        logger.error(f"Clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        return get_rag_service().health_check()
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
