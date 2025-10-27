"""
RAG Schemas
Pydantic models for RAG API request/response validation
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    """Request model for document ingestion"""
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
    """Request model for semantic search"""
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
