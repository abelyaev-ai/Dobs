"""Pydantic models for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="API health status")
    version: str = Field(description="Application version")


class PriceRequest(BaseModel):
    """Request model for price retrieval queries."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Natural language query for shipping prices",
        examples=["FedEx 2Day, Zone 5, 3 lb"],
    )


class PriceResponse(BaseModel):
    """Response model for price retrieval."""

    price: float = Field(description="Shipping price in USD")
    confidence: str = Field(description="Confidence level (high, medium, low)")
    source: str = Field(description="Source document/table identifier")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional information including raw response",
    )
    query_breakdown: dict[str, Any] = Field(
        default_factory=dict,
        description="Parsed query components (carrier, service, zone, weight)",
    )


class UploadResponse(BaseModel):
    """Response model for PDF upload."""

    success: bool = Field(description="Whether the upload was successful")
    filename: str = Field(description="Name of the uploaded file")
    tables_indexed: int = Field(ge=0, description="Number of tables indexed from the PDF")
    message: str = Field(description="Status message")


class DocumentInfo(BaseModel):
    """Information about an indexed document."""

    filename: str = Field(description="Name of the PDF file")
    carrier: str = Field(description="Shipping carrier name")
    tables: int = Field(ge=0, description="Number of tables indexed from this document")
    indexed_at: str = Field(description="Timestamp when document was indexed")


class DocumentListResponse(BaseModel):
    """Response model for listing indexed documents."""

    documents: list[DocumentInfo] = Field(
        default_factory=list,
        description="List of indexed documents with metadata",
    )
    total_documents: int = Field(ge=0, description="Total number of indexed documents")
    total_tables: int = Field(ge=0, description="Total number of indexed tables")
