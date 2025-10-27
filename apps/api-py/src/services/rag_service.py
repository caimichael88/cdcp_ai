"""
RAG Service
Business logic for RAG operations: ingestion, search, and database management
"""

import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .scraper_service import WebScraperService, DocumentChunker
from .embedding_service_with_transformer import EmbeddingService
from .vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Result of an ingestion operation"""
    success: bool
    scraped_documents: int
    chunked_documents: int
    ingested_documents: int
    scraping_time: float
    message: str


@dataclass
class SearchResult:
    """Individual search result"""
    content: str
    title: str
    url: str
    section: str
    similarity_score: float


@dataclass
class SearchResults:
    """Collection of search results"""
    query: str
    results: List[SearchResult]
    total_results: int


class RAGService:
    """Service for RAG operations"""

    def __init__(
        self,
        collection_name: str = "cdcp_documents",
        persist_directory: str = "./chroma_db"
    ):
        """Initialize RAG service with lazy-loaded dependencies"""
        self._embedding_service = None
        self._vector_db = None
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy load embedding service"""
        if self._embedding_service is None:
            logger.info("Loading embedding service...")
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    @property
    def vector_db(self) -> VectorDBService:
        """Lazy load vector database"""
        if self._vector_db is None:
            logger.info("Loading vector database...")
            self._vector_db = VectorDBService(
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
        return self._vector_db

    def ingest_content(
        self,
        base_urls: List[str],
        max_pages: int = 10,
        allowed_paths: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> IngestionResult:
        """
        Scrape, chunk, embed, and store documents in vector database

        Complete RAG pipeline:
        1. Scrapes web pages from provided URLs
        2. Chunks the documents into smaller pieces
        3. Generates embeddings for each chunk
        4. Stores everything in ChromaDB

        Args:
            base_urls: Starting URLs to scrape
            max_pages: Maximum pages to scrape
            allowed_paths: URL paths to filter
            chunk_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks

        Returns:
            IngestionResult with operation statistics

        Raises:
            ValueError: If no documents scraped or no chunks created
            Exception: For any other errors during ingestion
        """
        try:
            logger.info(f"Starting ingestion for {len(base_urls)} URLs")

            # Step 1: Scrape documents
            scraper = WebScraperService(
                base_urls=base_urls,
                max_pages=max_pages,
                allowed_paths=allowed_paths or ["/dental-care-plan/"]
            )
            documents = scraper.crawl()

            if not documents:
                raise ValueError("No documents were scraped")

            # Step 2: Chunk documents
            logger.info(f"Chunking {len(documents)} documents")
            chunker = DocumentChunker(
                chunk_size=chunk_size,
                overlap=chunk_overlap
            )

            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)

            if not all_chunks:
                raise ValueError("No chunks created")

            # Step 3: Generate embeddings
            logger.info(f"Generating embeddings for {len(all_chunks)} chunks")
            chunk_texts = [chunk.content for chunk in all_chunks]
            embeddings = self.embedding_service.embed_documents(chunk_texts)

            # Step 4: Store in vector database
            logger.info("Storing in vector database")
            metadatas, chunk_ids = self._prepare_metadata_and_ids(all_chunks)

            self.vector_db.add_documents_batch(
                documents=chunk_texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=chunk_ids,
                batch_size=50
            )

            stats = scraper.get_stats()

            return IngestionResult(
                success=True,
                scraped_documents=len(documents),
                chunked_documents=len(all_chunks),
                ingested_documents=len(all_chunks),
                scraping_time=stats['total_time'],
                message=f"Successfully ingested {len(documents)} documents ({len(all_chunks)} chunks)"
            )

        except ValueError as e:
            logger.error(f"Validation error in ingest_content: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in ingest_content: {e}")
            raise

    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_section: Optional[str] = None
    ) -> SearchResults:
        """
        Search for documents using semantic similarity

        Args:
            query: Search query
            n_results: Number of results to return
            filter_section: Optional section filter

        Returns:
            SearchResults with matching documents

        Raises:
            Exception: For any errors during search
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)

            # Search vector database
            where_filter = None
            if filter_section:
                where_filter = {"section": filter_section}

            results = self.vector_db.search(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
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

            return SearchResults(
                query=query,
                results=search_results,
                total_results=len(search_results)
            )

        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = self.vector_db.get_stats()
            return {
                "total_documents": stats['total_documents'],
                "collection_name": stats['collection_name']
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            raise

    def clear_database(self) -> None:
        """Clear all documents from the database"""
        try:
            self.vector_db.clear_collection()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """Check health of RAG service components"""
        try:
            # Accessing properties will trigger lazy loading
            _ = self.embedding_service
            _ = self.vector_db

            return {
                "status": "healthy",
                "embedding_service": "loaded",
                "vector_db": "loaded",
                "total_documents": self.vector_db.count()
            }
        except Exception as e:
            logger.error(f"Error in health_check: {e}")
            raise

    def _prepare_metadata_and_ids(self, chunks) -> tuple[List[Dict], List[str]]:
        """
        Prepare metadata and deterministic IDs for chunks

        Args:
            chunks: List of document chunks

        Returns:
            Tuple of (metadatas, chunk_ids)
        """
        metadatas = []
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            metadatas.append({
                "title": chunk.title,
                "url": chunk.url,
                "section": chunk.section or "general",
                "doc_id": chunk.doc_id,
                "language": chunk.language
            })
            # Create deterministic ID: doc_id is MD5 of URL, append chunk index
            # This ensures same URL always gets same IDs (upsert behavior)
            chunk_ids.append(f"{chunk.doc_id}_chunk_{i}")

        return metadatas, chunk_ids


# Global singleton instance
_rag_service_instance = None

def get_rag_service(
    collection_name: str = "cdcp_documents",
    persist_directory: str = "./chroma_db"
) -> RAGService:
    """
    Get or create the global RAG service singleton instance.
    This ensures the embedding model is only loaded once.
    """
    global _rag_service_instance
    if _rag_service_instance is None:
        logger.info("Initializing global RAG service singleton...")
        _rag_service_instance = RAGService(
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    return _rag_service_instance
