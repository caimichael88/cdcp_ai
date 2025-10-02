"""
Vector Database Service using ChromaDB
Handles storing and retrieving document embeddings with metadata
"""

import logging
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np

logger = logging.getLogger(__name__)


class VectorDBService:
    """
    Service for managing vector embeddings and semantic search using ChromaDB
    Stores documents, embeddings, and metadata together
    """

    def __init__(
        self,
        collection_name: str = "cdcp_documents",
        persist_directory: str = "./chroma_db",
        embedding_function: Optional[Any] = None
    ):
        """
        Initialize Vector Database Service

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
            embedding_function: Optional custom embedding function
                              If None, ChromaDB will use default
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collection
        if embedding_function:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function
            )
        else:
            # Use default sentence transformer
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )

        logger.info(f"VectorDBService initialized with collection: {collection_name}")
        logger.info(f"Database location: {persist_directory}")

    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        upsert: bool = True
    ) -> None:
        """
        Add documents with embeddings and metadata to the database

        Args:
            documents: List of document text content
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of unique IDs (auto-generated if None)
            upsert: If True, update existing documents; if False, add only new ones
        """
        try:
            if ids is None:
                # Auto-generate IDs (sequential)
                count = self.collection.count()
                ids = [f"doc_{count + i}" for i in range(len(documents))]

            # Validate inputs
            if not (len(documents) == len(embeddings) == len(metadatas) == len(ids)):
                raise ValueError(
                    f"Mismatched lengths: docs={len(documents)}, "
                    f"embeddings={len(embeddings)}, metadata={len(metadatas)}, ids={len(ids)}"
                )

            if upsert:
                # Upsert: Update existing documents with same ID, add new ones
                self.collection.upsert(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Upserted {len(documents)} documents to collection")
            else:
                # Add only (will error if IDs already exist)
                self.collection.add(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
                logger.info(f"Added {len(documents)} documents to collection")

        except Exception as e:
            logger.error(f"Error adding/upserting documents: {e}")
            raise

    def add_documents_batch(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        upsert: bool = True
    ) -> None:
        """
        Add documents in batches for better performance with large datasets

        Args:
            documents: List of document text content
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            ids: Optional list of unique IDs
            batch_size: Number of documents per batch
            upsert: If True, update existing documents; if False, add only new ones
        """
        total = len(documents)
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)

            batch_ids = ids[i:batch_end] if ids else None

            self.add_documents(
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=batch_ids,
                upsert=upsert
            )

            logger.info(f"Processed batch {i // batch_size + 1}: {batch_end}/{total} documents")

    def search(
        self,
        query_embeddings: List[List[float]],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar documents using embeddings

        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return per query
            where: Metadata filter (e.g., {"section": "eligibility"})
            where_document: Document content filter

        Returns:
            Dictionary with ids, documents, metadatas, and distances
        """
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )

            logger.debug(f"Search returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Error during search: {e}")
            raise

    def search_by_text(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search using text queries (ChromaDB will embed them automatically)

        Args:
            query_texts: List of query strings
            n_results: Number of results to return per query
            where: Metadata filter

        Returns:
            Dictionary with ids, documents, metadatas, and distances
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where
            )

            logger.debug(f"Text search returned {len(results['ids'][0])} results")
            return results

        except Exception as e:
            logger.error(f"Error during text search: {e}")
            raise

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get documents by their IDs

        Args:
            ids: List of document IDs

        Returns:
            Dictionary with documents, metadatas, and embeddings
        """
        try:
            results = self.collection.get(ids=ids)
            return results

        except Exception as e:
            logger.error(f"Error getting documents by IDs: {e}")
            raise

    def get_by_metadata(
        self,
        where: Dict[str, Any],
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get documents by metadata filter

        Args:
            where: Metadata filter (e.g., {"section": "coverage"})
            limit: Maximum number of results

        Returns:
            Dictionary with documents and metadatas
        """
        try:
            results = self.collection.get(
                where=where,
                limit=limit
            )
            return results

        except Exception as e:
            logger.error(f"Error getting documents by metadata: {e}")
            raise

    def update_metadata(
        self,
        ids: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Update metadata for existing documents

        Args:
            ids: List of document IDs to update
            metadatas: New metadata for each document
        """
        try:
            self.collection.update(
                ids=ids,
                metadatas=metadatas
            )
            logger.info(f"Updated metadata for {len(ids)} documents")

        except Exception as e:
            logger.error(f"Error updating metadata: {e}")
            raise

    def delete_by_ids(self, ids: List[str]) -> None:
        """
        Delete documents by IDs

        Args:
            ids: List of document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")

        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise

    def delete_by_metadata(self, where: Dict[str, Any]) -> None:
        """
        Delete documents by metadata filter

        Args:
            where: Metadata filter (e.g., {"source": "old_scrape"})
        """
        try:
            self.collection.delete(where=where)
            logger.info(f"Deleted documents matching filter: {where}")

        except Exception as e:
            logger.error(f"Error deleting documents by metadata: {e}")
            raise

    def count(self) -> int:
        """
        Get total number of documents in collection

        Returns:
            Total document count
        """
        return self.collection.count()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Dictionary with statistics
        """
        count = self.count()

        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": self.persist_directory
        }

    def clear_collection(self) -> None:
        """
        Clear all documents from the collection (use with caution!)
        """
        try:
            # Delete and recreate collection
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )
            logger.warning(f"Cleared all documents from collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def reset_database(self) -> None:
        """
        Reset entire database (delete all collections)
        """
        try:
            self.client.reset()
            logger.warning("Database reset complete - all collections deleted")

            # Recreate the collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name
            )

        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            raise
