"""FastAPI application main entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.endpoints import router
from src.core.document_indexer import DocumentIndexer, DocumentIndexerError
from src.utils.config import load_settings
from src.utils.logger import get_logger, log_exception, setup_logger

# Initialize logger
setup_logger("shipping_price_search")
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager.

    Handles startup and shutdown events for the FastAPI application.
    Initializes the DocumentIndexer on startup and makes it available
    via app.state for all requests.
    """
    # Startup
    logger.info("Starting Shipping Price Search API")

    try:
        # Load configuration
        settings = load_settings()
        logger.info("Configuration loaded successfully")
        logger.info(f"PDF storage: {settings.pdf_storage_path}")
        logger.info(f"Vector store: {settings.vector_store_path}")

        # Initialize DocumentIndexer
        logger.info("Initializing DocumentIndexer...")
        indexer = DocumentIndexer(persist_dir=str(settings.vector_store_path))

        # Try to load existing index
        try:
            indexer.load_index()
            logger.info("Existing index loaded successfully")
        except DocumentIndexerError as e:
            logger.warning(f"No existing index found: {e}")
            logger.info("Index will be created when first PDF is uploaded")

        # Store indexer in app state for access in endpoints
        app.state.indexer = indexer
        logger.info("DocumentIndexer initialized and ready")

    except Exception as e:
        log_exception(logger, "Failed to initialize application", e)
        raise

    logger.info("Shipping Price Search API started successfully")

    yield

    # Shutdown
    logger.info("Shutting down Shipping Price Search API")


# Create FastAPI application
app = FastAPI(
    title="Shipping Price Search API",
    description=(
        "Search shipping prices from PDF price lists using semantic search. "
        "Upload PDF rate cards, index their tables, and query prices using natural language."
    ),
    version=__version__,
    lifespan=lifespan,
)

# Configure CORS for localhost development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status and version information.

    Example:
        ```bash
        curl http://localhost:8000/health
        ```

        Response:
        ```json
        {
            "status": "healthy",
            "version": "0.1.0"
        }
        ```
    """
    return {
        "status": "healthy",
        "version": __version__,
    }


# Include API routers
app.include_router(router, tags=["Shipping Price Search"])
