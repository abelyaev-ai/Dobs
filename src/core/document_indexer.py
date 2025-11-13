"""Document indexing module using LlamaIndex and FAISS.

This module provides a comprehensive document indexing system that:
1. Extracts pages from PDFs using PDFPageExtractor
2. Extracts metadata using MetadataExtractor with service catalog
3. Creates LlamaIndex Documents with rich metadata (page-based chunking)
4. Sets up FAISS vector store for similarity search
5. Configures OpenAI embeddings and LLM
6. Provides query interface with metadata filtering
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import faiss
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.faiss import FaissVectorStore

from src.core.metadata_extractor import MetadataExtractor
from src.core.page_chunker import PageChunker
from src.core.page_extractor import PDFPageExtractor
from src.core.service_catalog import ServiceCatalog
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_exception

logger = get_logger(__name__)


class DocumentIndexerError(Exception):
    """Raised when document indexing operations fail."""


class DocumentIndexer:
    """Document indexing system using LlamaIndex and FAISS.

    This class manages the complete lifecycle of document indexing:
    - Extracting tables from PDFs
    - Creating vector embeddings
    - Building and persisting FAISS index
    - Providing query interface with metadata filtering

    Attributes:
        persist_dir: Directory for storing the index
        dimension: Vector dimension for embeddings (1536 for text-embedding-3-small)
        index: LlamaIndex VectorStoreIndex instance
        vector_store: FAISS vector store instance
    """

    # Default embedding dimension for OpenAI text-embedding-3-small
    EMBEDDING_DIMENSION = 1536

    def __init__(self, persist_dir: str = "data/storage") -> None:
        """Initialize the DocumentIndexer.

        Args:
            persist_dir: Directory path for storing the index. Defaults to "data/storage".

        Raises:
            DocumentIndexerError: If initialization fails.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.dimension = self.EMBEDDING_DIMENSION
        self.index: VectorStoreIndex | None = None
        self.vector_store: FaissVectorStore | None = None

        # Initialize page-based extractors
        self.page_extractor = PDFPageExtractor()
        self.page_chunker = PageChunker()
        # metadata_extractor will be created with service catalog in build_index()

        # Configure LlamaIndex settings
        self._configure_settings()

        logger.info(f"DocumentIndexer initialized with persist_dir={self.persist_dir}")

    def _configure_settings(self) -> None:
        """Configure global LlamaIndex settings with OpenAI integrations.

        Sets up:
        - OpenAI LLM (gpt-4-turbo-preview) with temperature=0
        - OpenAI embeddings (text-embedding-3-small)
        - API key from config system

        Raises:
            DocumentIndexerError: If configuration fails.
        """
        try:
            config = get_settings()

            # Set OpenAI API key
            os.environ["OPENAI_API_KEY"] = config.openai_api_key

            # Configure LLM with config model and temperature
            Settings.llm = OpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                api_key=config.openai_api_key,
            )

            # Configure embeddings from config
            Settings.embed_model = OpenAIEmbedding(
                model=config.embedding_model,
                api_key=config.openai_api_key,
            )

            # Configure node parser for chunking
            # Use large chunk size to keep tables intact (prevent splitting mid-table)
            Settings.node_parser = SentenceSplitter(
                chunk_size=8192,  # Large enough for complete tables
                chunk_overlap=0,  # No overlap needed for complete tables
            )

            logger.info(
                f"LlamaIndex settings configured: "
                f"LLM={config.llm_model}, "
                f"Embedding={config.embedding_model}, "
                f"ChunkSize=8192 (tables kept intact)"
            )

        except Exception as e:
            raise DocumentIndexerError(
                f"Failed to configure LlamaIndex settings: {e}"
            ) from e

    def _create_faiss_index(self) -> faiss.IndexFlatL2:
        """Create a FAISS IndexFlatL2 for similarity search.

        Returns:
            FAISS IndexFlatL2 instance configured for the embedding dimension.
        """
        faiss_index = faiss.IndexFlatL2(self.dimension)
        logger.debug(f"Created FAISS IndexFlatL2 with dimension={self.dimension}")
        return faiss_index

    def _extract_documents_from_pdf(
        self,
        pdf_path: str,
        carrier_name: str,
        metadata_extractor: MetadataExtractor,
    ) -> list[Document]:
        """Extract documents from PDF using page-based approach.

        Creates one chunk per page (instead of one per table). This results
        in ~50-60 chunks instead of 271 for the FedEx PDF.

        Args:
            pdf_path: Path to the PDF file.
            carrier_name: Name of the shipping carrier.
            metadata_extractor: MetadataExtractor instance (with service catalog).

        Returns:
            List of LlamaIndex Document objects with rich metadata.

        Raises:
            DocumentIndexerError: If extraction fails.
        """
        try:
            logger.info(
                f"Extracting documents from {pdf_path} for {carrier_name} "
                f"(mode: page-based)"
            )

            # Step 1: Extract raw page data
            pages = self.page_extractor.extract_pages(pdf_path)
            logger.info(f"Extracted {len(pages)} pages with tables")

            # Step 2: Extract metadata for each page using provided metadata_extractor
            metadata_list = [
                metadata_extractor.extract_metadata(page, carrier_name)
                for page in pages
            ]
            logger.info(f"Extracted metadata for {len(metadata_list)} pages")

            # Step 3: Create Document chunks (one per page)
            documents = self.page_chunker.create_chunks(pages, metadata_list)

            logger.info(
                f"Created {len(documents)} page-based chunks from {pdf_path}"
            )

            return documents

        except Exception as e:
            raise DocumentIndexerError(
                f"Failed to extract documents from {pdf_path}: {e}"
            ) from e

    def index_pdf(
        self,
        pdf_path: str,
        carrier_name: str,
        metadata_extractor: MetadataExtractor,
    ) -> int:
        """Index a single PDF file.

        Args:
            pdf_path: Path to the PDF file.
            carrier_name: Name of the shipping carrier.
            metadata_extractor: MetadataExtractor instance with service catalog.

        Returns:
            Number of documents indexed.

        Raises:
            DocumentIndexerError: If indexing fails.
        """
        try:
            logger.info(f"Indexing PDF: {pdf_path} (carrier: {carrier_name})")

            # Extract documents
            documents = self._extract_documents_from_pdf(
                pdf_path, carrier_name, metadata_extractor
            )

            if not documents:
                logger.warning(f"No documents extracted from {pdf_path}")
                return 0

            # Create or update index
            if self.index is None:
                # Create new index
                faiss_index = self._create_faiss_index()
                self.vector_store = FaissVectorStore(faiss_index=faiss_index)
                storage_context = StorageContext.from_defaults(
                    vector_store=self.vector_store
                )

                self.index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context,
                    show_progress=True,
                )

                logger.info(f"Created new index with {len(documents)} documents")
            else:
                # Add to existing index
                for doc in documents:
                    self.index.insert(doc, show_progress=True)

                logger.info(f"Added {len(documents)} documents to existing index")

            return len(documents)

        except Exception as e:
            log_exception(
                logger,
                f"Failed to index PDF {pdf_path}",
                e,
                extra={"pdf_path": pdf_path, "carrier": carrier_name},
            )
            raise DocumentIndexerError(f"Failed to index PDF {pdf_path}: {e}") from e

    def build_index(self, pdf_paths: list[tuple[str, str]]) -> dict[str, int]:
        """Index multiple PDF files.

        This method follows a two-phase approach:
        1. Build service catalog from all PDFs to extract canonical service names
        2. Index documents using the catalog for service name normalization

        Args:
            pdf_paths: List of tuples (pdf_path, carrier_name) to index.

        Returns:
            Dictionary mapping PDF paths to number of documents indexed.

        Raises:
            DocumentIndexerError: If indexing fails for all PDFs.
        """
        logger.info(f"Building index from {len(pdf_paths)} PDF(s)")

        # STEP 1: Build service catalog from all PDFs
        logger.info("Phase 1: Building service catalog from PDFs...")
        service_catalog = ServiceCatalog()

        for pdf_path, carrier_name in pdf_paths:
            try:
                logger.debug(f"Extracting services from {pdf_path}")
                service_catalog.extract_from_pdf(pdf_path)
            except Exception as e:
                logger.warning(f"Failed to extract services from {pdf_path}: {e}")
                # Continue anyway - we can still index without this PDF's services

        logger.info(
            f"Service catalog built with {len(service_catalog.get_all_services())} canonical services: "
            f"{service_catalog.get_all_services()}"
        )

        # STEP 2: Create metadata extractor with service catalog
        metadata_extractor = MetadataExtractor(service_catalog=service_catalog)

        # STEP 3: Index documents using the catalog
        logger.info("Phase 2: Indexing documents with canonical service names...")
        results: dict[str, int] = {}
        errors: list[str] = []

        for pdf_path, carrier_name in pdf_paths:
            try:
                count = self.index_pdf(pdf_path, carrier_name, metadata_extractor)
                results[pdf_path] = count
            except DocumentIndexerError as e:
                logger.error(f"Failed to index {pdf_path}: {e}")
                errors.append(pdf_path)
                results[pdf_path] = 0

        # Check if all indexing failed
        if all(count == 0 for count in results.values()):
            raise DocumentIndexerError(
                f"Failed to index any documents. Errors: {errors}"
            )

        total_docs = sum(results.values())
        logger.info(
            f"Index built successfully: {total_docs} total documents "
            f"from {len(results)} PDFs"
        )

        return results

    def add_documents(self, documents: list[Document]) -> None:
        """Add new documents to the existing index.

        Args:
            documents: List of LlamaIndex Document objects to add.

        Raises:
            DocumentIndexerError: If no index exists or adding fails.
        """
        if self.index is None:
            raise DocumentIndexerError(
                "No index exists. Use build_index() or index_pdf() first."
            )

        try:
            logger.info(f"Adding {len(documents)} documents to index")

            for doc in documents:
                self.index.insert(doc, show_progress=True)

            logger.info(f"Successfully added {len(documents)} documents")

        except Exception as e:
            raise DocumentIndexerError(f"Failed to add documents to index: {e}") from e

    def save_index(self) -> None:
        """Persist the index to disk.

        Raises:
            DocumentIndexerError: If no index exists or save fails.
        """
        if self.index is None:
            raise DocumentIndexerError("No index to save. Build an index first.")

        try:
            logger.info(f"Saving index to {self.persist_dir}")

            # Persist the index
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))

            logger.info(f"Index saved successfully to {self.persist_dir}")

        except Exception as e:
            raise DocumentIndexerError(
                f"Failed to save index to {self.persist_dir}: {e}"
            ) from e

    def load_index(self) -> None:
        """Load a persisted index from disk.

        Raises:
            DocumentIndexerError: If loading fails or no index found.
        """
        try:
            logger.info(f"Loading index from {self.persist_dir}")

            # Check if index exists
            if not (self.persist_dir / "docstore.json").exists():
                raise DocumentIndexerError(
                    f"No index found at {self.persist_dir}. "
                    "Build and save an index first."
                )

            # Load FAISS vector store from persist directory
            self.vector_store = FaissVectorStore.from_persist_dir(
                persist_dir=str(self.persist_dir)
            )

            # Load storage context with the loaded vector store
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store,
                persist_dir=str(self.persist_dir),
            )

            # Load index
            self.index = load_index_from_storage(
                storage_context=storage_context,
            )

            logger.info(f"Index loaded successfully from {self.persist_dir}")

        except DocumentIndexerError:
            raise
        except Exception as e:
            raise DocumentIndexerError(
                f"Failed to load index from {self.persist_dir}: {e}"
            ) from e

    def get_query_engine(
        self,
        similarity_top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        """Get a query engine for searching the index.

        Args:
            similarity_top_k: Number of top similar results to retrieve.
            filters: Optional metadata filters as dict.
                    Example: {"carrier": "FedEx", "service_type": "Ground"}

        Returns:
            Query engine instance for executing searches.

        Raises:
            DocumentIndexerError: If no index exists.
        """
        if self.index is None:
            raise DocumentIndexerError(
                "No index loaded. Use load_index() or build_index() first."
            )

        try:
            logger.info(
                f"Creating query engine with similarity_top_k={similarity_top_k}"
            )

            # Build metadata filters if provided
            metadata_filters = None
            if filters:
                filter_list = []
                for key, value in filters.items():
                    filter_list.append(
                        MetadataFilter(
                            key=key,
                            value=value,
                            operator=FilterOperator.EQ,
                        )
                    )
                metadata_filters = MetadataFilters(filters=filter_list)
                logger.debug(f"Applied metadata filters: {filters}")

            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                filters=metadata_filters,
            )

            return query_engine

        except Exception as e:
            raise DocumentIndexerError(f"Failed to create query engine: {e}") from e

    def query(
        self,
        query_text: str,
        similarity_top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a query against the index.

        Args:
            query_text: Natural language query string.
            similarity_top_k: Number of top results to retrieve.
            filters: Optional metadata filters.
                    Example: {"carrier": "FedEx"} for FedEx-only results.

        Returns:
            Query response object with results and source nodes.

        Raises:
            DocumentIndexerError: If query fails.
        """
        try:
            logger.info(f"Executing query: {query_text[:100]}...")

            query_engine = self.get_query_engine(
                similarity_top_k=similarity_top_k,
                filters=filters,
            )

            response = query_engine.query(query_text)

            logger.info("Query completed successfully")

            return response

        except Exception as e:
            log_exception(
                logger,
                f"Query failed: {query_text[:100]}",
                e,
                extra={"query": query_text, "filters": filters},
            )
            raise DocumentIndexerError(f"Query failed: {e}") from e
