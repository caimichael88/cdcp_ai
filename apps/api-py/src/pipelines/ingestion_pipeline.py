"""High-level ingestion pipeline for CDCP RAG system"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from ..services.scraper_service import (
    DocumentChunker,
    ScrapedDocument,
    WebScraperService,
)
from ..services.embedding_service_with_transformer import EmbeddingService
from ..services.vector_db_service import VectorDBService

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the ingestion pipeline."""

    base_urls: List[str]
    allowed_paths: Optional[List[str]] = None
    max_pages: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "cdcp_documents"
    persist_directory: str = "./chroma_db"
    batch_size: int = 100


class CDCPIngestionPipeline:
    """Runs scraping, chunking, embedding, and persistence."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._scraper: Optional[WebScraperService] = None
        self._chunker: Optional[DocumentChunker] = None
        self._embedding_service: Optional[EmbeddingService] = None
        self._vector_db: Optional[VectorDBService] = None

    def _init_services(self) -> None:
        """Lazy-initialize service dependencies."""
        if self._scraper is None:
            self._scraper = WebScraperService(
                base_urls=self.config.base_urls,
                allowed_paths=self.config.allowed_paths,
                max_pages=self.config.max_pages,
            )
        if self._chunker is None:
            self._chunker = DocumentChunker(
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap,
            )
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(
                model_name=self.config.embedding_model,
            )
        if self._vector_db is None:
            self._vector_db = VectorDBService(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
            )

    def run(self) -> Dict[str, object]:
        """Execute the ingestion pipeline and return runtime statistics."""
        self._init_services()
        assert self._scraper and self._chunker and self._embedding_service and self._vector_db

        logger.info("Starting CDCP ingestion pipeline")
        scraped_documents = self._scraper.crawl()
        if not scraped_documents:
            logger.warning("Scraper returned no documents")
            return {
                "success": False,
                "error": "No documents scraped",
                "scraper_stats": self._scraper.get_stats(),
            }

        chunked_documents: List[ScrapedDocument] = []
        for document in scraped_documents:
            chunked_documents.extend(self._chunker.chunk_document(document))

        if not chunked_documents:
            logger.warning("Chunker produced zero documents")
            return {
                "success": False,
                "error": "No chunks produced",
                "scraper_stats": self._scraper.get_stats(),
            }

        chunk_texts = [doc.content for doc in chunked_documents]
        embeddings_array = self._embedding_service.embed_documents(chunk_texts)
        embeddings = embeddings_array.tolist()

        metadatas = [
            {
                "url": doc.url,
                "title": doc.title,
                "section": doc.section,
                "doc_id": doc.doc_id,
                "last_updated": doc.last_updated,
                "language": doc.language,
            }
            for doc in chunked_documents
        ]

        self._vector_db.add_documents_batch(
            documents=chunk_texts,
            embeddings=embeddings,
            metadatas=metadatas,
            batch_size=self.config.batch_size,
        )

        scraper_stats = self._scraper.get_stats()
        vector_stats = self._vector_db.get_stats()

        logger.info(
            "Pipeline complete: %s scraped, %s chunks ingested",
            len(scraped_documents),
            len(chunked_documents),
        )

        return {
            "success": True,
            "scraped_documents": len(scraped_documents),
            "chunked_documents": len(chunked_documents),
            "ingested_documents": len(chunked_documents),
            "scraper_stats": scraper_stats,
            "vector_db_stats": vector_stats,
        }


def run_pipeline(config: PipelineConfig) -> Dict[str, object]:
    """Convenience helper that constructs and runs the pipeline."""
    pipeline = CDCPIngestionPipeline(config)
    return pipeline.run()
