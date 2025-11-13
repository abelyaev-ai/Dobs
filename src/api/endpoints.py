"""API endpoint handlers for shipping price search."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from src.api.models import (
    DocumentInfo,
    DocumentListResponse,
    PriceRequest,
    PriceResponse,
    UploadResponse,
)
from src.core.document_indexer import DocumentIndexer, DocumentIndexerError
from src.core.price_retriever import PriceRetriever, PriceRetrieverError
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_exception

logger = get_logger(__name__)

# Create router
router = APIRouter()


def get_indexer(request: Request) -> DocumentIndexer:
    """Get the DocumentIndexer instance from app state.

    Args:
        request: FastAPI request object.

    Returns:
        DocumentIndexer instance.

    Raises:
        HTTPException: If indexer is not initialized.
    """
    if not hasattr(request.app.state, "indexer"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document indexer not initialized",
        )
    return request.app.state.indexer


@router.post("/price", response_model=PriceResponse, status_code=status.HTTP_200_OK)
async def get_price(request: Request, price_request: PriceRequest) -> PriceResponse:
    """Get shipping price for a query.

    This endpoint parses the shipping query and retrieves the price from
    indexed PDF documents using semantic search and LLM-based extraction.

    Args:
        request: FastAPI request object.
        price_request: PriceRequest with the query string.

    Returns:
        PriceResponse with price, confidence, source, and details.

    Raises:
        HTTPException: If price retrieval fails or query is invalid.

    Example:
        Request:
        ```json
        {
            "query": "FedEx 2Day, Zone 5, 3 lb"
        }
        ```

        Response:
        ```json
        {
            "price": 48.05,
            "confidence": "high",
            "source": "fedex_rates.pdf, Page 2, Table 1",
            "details": {
                "raw_response": "48.05",
                "search_query": "FedEx 2Day Zone 5 3 lbs...",
                "source_nodes": 10
            },
            "query_breakdown": {
                "carrier": "FedEx",
                "service": "2Day",
                "zone": "5",
                "weight": 3.0,
                "packaging": null
            }
        }
        ```
    """
    try:
        logger.info(f"Price request received: {price_request.query}")

        # Get indexer from app state
        indexer = get_indexer(request)

        # Create PriceRetriever with the shared indexer
        retriever = PriceRetriever(indexer=indexer)

        # Retrieve price
        result = retriever.get_price(price_request.query)

        # Build query breakdown
        query_breakdown: dict[str, Any] = {}
        if result.query:
            query_breakdown = {
                "carrier": result.query.carrier,
                "service": result.query.service,
                "zone": result.query.zone,
                "weight": result.query.weight,
                "packaging": result.query.packaging,
            }

        # Build response
        response = PriceResponse(
            price=result.price,
            confidence=result.confidence,
            source=result.source,
            details=result.details,
            query_breakdown=query_breakdown,
        )

        logger.info(
            f"Price retrieved successfully: ${result.price:.2f} "
            f"(confidence: {result.confidence})"
        )

        return response

    except PriceRetrieverError as e:
        logger.error(f"Price retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Price not found: {str(e)}",
        ) from e
    except Exception as e:
        log_exception(logger, "Unexpected error during price retrieval", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}",
        ) from e


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file to upload and index"),
    carrier_name: str = Form(..., description="Shipping carrier name (e.g., FedEx, UPS)"),
) -> UploadResponse:
    """Upload and index a new PDF file.

    This endpoint:
    1. Saves the uploaded PDF to the configured storage directory
    2. Extracts tables from the PDF
    3. Indexes the tables using the DocumentIndexer
    4. Persists the updated index

    Args:
        request: FastAPI request object.
        file: Uploaded PDF file.
        carrier_name: Name of the shipping carrier.

    Returns:
        UploadResponse with success status, filename, and number of tables indexed.

    Raises:
        HTTPException: If upload or indexing fails.

    Example:
        Request (multipart/form-data):
        ```
        file: fedex_rates.pdf
        carrier_name: FedEx
        ```

        Response:
        ```json
        {
            "success": true,
            "filename": "fedex_rates.pdf",
            "tables_indexed": 42,
            "message": "Successfully indexed 42 tables from fedex_rates.pdf"
        }
        ```
    """
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed",
            )

        logger.info(f"Upload request received: {file.filename} (carrier: {carrier_name})")

        # Get settings and indexer
        settings = get_settings()
        indexer = get_indexer(request)

        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.pdf_storage_path)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = upload_dir / file.filename

        # Read file content
        content = await file.read()

        # Write to disk
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File saved to {file_path}")

        # Index the PDF
        try:
            tables_indexed = indexer.index_pdf(str(file_path), carrier_name)

            # Save the updated index
            indexer.save_index()

            logger.info(
                f"Successfully indexed {tables_indexed} tables from {file.filename}"
            )

            return UploadResponse(
                success=True,
                filename=file.filename,
                tables_indexed=tables_indexed,
                message=f"Successfully indexed {tables_indexed} tables from {file.filename}",
            )

        except DocumentIndexerError as e:
            # Clean up the uploaded file if indexing fails
            if file_path.exists():
                file_path.unlink()

            logger.error(f"Indexing failed for {file.filename}: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Failed to index PDF: {str(e)}",
            ) from e

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, f"Upload failed for {file.filename}", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}",
        ) from e


@router.get("/documents", response_model=DocumentListResponse, status_code=status.HTTP_200_OK)
async def list_documents(request: Request) -> DocumentListResponse:
    """List all indexed documents.

    This endpoint returns information about all PDFs that have been indexed,
    including the number of tables extracted from each document.

    Args:
        request: FastAPI request object.

    Returns:
        DocumentListResponse with list of documents and totals.

    Example:
        Response:
        ```json
        {
            "documents": [
                {
                    "filename": "fedex_rates.pdf",
                    "carrier": "FedEx",
                    "tables": 42,
                    "indexed_at": "2025-11-11T00:00:00"
                },
                {
                    "filename": "ups_rates.pdf",
                    "carrier": "UPS",
                    "tables": 38,
                    "indexed_at": "2025-11-11T01:00:00"
                }
            ],
            "total_documents": 2,
            "total_tables": 80
        }
        ```
    """
    try:
        logger.info("Listing indexed documents")

        settings = get_settings()
        indexer = get_indexer(request)

        # Check if index exists
        if indexer.index is None:
            logger.info("No index loaded yet")
            return DocumentListResponse(
                documents=[],
                total_documents=0,
                total_tables=0,
            )

        # Get all documents from the PDF storage directory
        pdf_dir = Path(settings.pdf_storage_path)

        if not pdf_dir.exists():
            return DocumentListResponse(
                documents=[],
                total_documents=0,
                total_tables=0,
            )

        # Collect document information
        documents: list[DocumentInfo] = []
        total_tables = 0

        # Get all PDF files
        pdf_files = list(pdf_dir.glob("*.pdf"))

        # Track tables per document by analyzing the index
        # We'll need to query the index metadata to get this info
        doc_table_counts: dict[str, dict[str, Any]] = {}

        # If we have an index, try to get metadata from it
        if indexer.index and hasattr(indexer.index, "docstore"):
            docstore = indexer.index.docstore

            # Iterate through all documents in the docstore
            for doc_id, doc in docstore.docs.items():
                if hasattr(doc, "metadata") and doc.metadata:
                    source = doc.metadata.get("source", "")
                    carrier = doc.metadata.get("carrier", "Unknown")

                    if source:
                        # Extract filename from source path
                        source_filename = os.path.basename(source)

                        if source_filename not in doc_table_counts:
                            doc_table_counts[source_filename] = {
                                "carrier": carrier,
                                "tables": 0,
                            }

                        doc_table_counts[source_filename]["tables"] += 1

        # Build response for each PDF file
        for pdf_file in sorted(pdf_files):
            filename = pdf_file.name

            # Get info from docstore if available
            if filename in doc_table_counts:
                carrier = doc_table_counts[filename]["carrier"]
                tables = doc_table_counts[filename]["tables"]
            else:
                # Fallback if not in index
                carrier = "Unknown"
                tables = 0

            # Get file modification time as indexed_at
            indexed_at = datetime.fromtimestamp(pdf_file.stat().st_mtime).isoformat()

            documents.append(
                DocumentInfo(
                    filename=filename,
                    carrier=carrier,
                    tables=tables,
                    indexed_at=indexed_at,
                )
            )

            total_tables += tables

        logger.info(
            f"Found {len(documents)} documents with {total_tables} total tables"
        )

        return DocumentListResponse(
            documents=documents,
            total_documents=len(documents),
            total_tables=total_tables,
        )

    except Exception as e:
        log_exception(logger, "Failed to list documents", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}",
        ) from e
