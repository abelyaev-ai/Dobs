"""Extract raw page data from PDFs without metadata interpretation.

This module provides low-level PDF page extraction functionality, separating
the concerns of data extraction from metadata interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pdfplumber

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PageExtractionError(Exception):
    """Raised when page extraction fails."""


@dataclass
class PageData:
    """Raw data extracted from a PDF page.

    Attributes:
        page_number: Page number (0-indexed).
        page_text: Full text content of the page.
        page_header: First 800 characters of page for context.
        tables: List of DataFrames containing table data from the page.
        source_document: Name of the source PDF file.
    """

    page_number: int
    page_text: str
    page_header: str
    tables: list[pd.DataFrame]
    source_document: str


class PDFPageExtractor:
    """Extract raw page data from PDFs.

    This class focuses solely on extracting raw data (text and tables) from PDF pages.
    It does not interpret metadata like service types or zones - that is handled by
    MetadataExtractor.

    Only pages containing tables are extracted, as pages without tables are not
    relevant for shipping rate analysis.
    """

    # pdfplumber table extraction settings optimized for shipping rate tables
    TABLE_SETTINGS: dict[str, any] = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 3,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 1,
        "min_words_horizontal": 1,
    }

    def __init__(
        self,
        table_settings: dict[str, any] | None = None,
        min_table_rows: int = 2,
        min_table_cols: int = 2,
    ) -> None:
        """Initialize the PDF page extractor.

        Args:
            table_settings: Custom pdfplumber table extraction settings.
            min_table_rows: Minimum rows to consider a valid table.
            min_table_cols: Minimum columns to consider a valid table.
        """
        self.table_settings = table_settings or self.TABLE_SETTINGS
        self.min_table_rows = min_table_rows
        self.min_table_cols = min_table_cols

    def extract_pages(self, pdf_path: str | Path) -> list[PageData]:
        """Extract all pages with tables from PDF.

        Returns one PageData per page containing:
        - Full page text
        - Page header (first 800 chars for context)
        - All tables on the page as DataFrames

        Only pages containing at least one valid table are included.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of PageData objects, one per page with tables.

        Raises:
            PageExtractionError: If PDF cannot be opened or extraction fails.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise PageExtractionError(f"PDF file not found: {pdf_path}")

        if not pdf_path.is_file():
            raise PageExtractionError(f"Path is not a file: {pdf_path}")

        try:
            logger.info(f"Extracting pages from {pdf_path.name}")
            pages_data: list[PageData] = []

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.debug(f"Processing {total_pages} pages from {pdf_path.name}")

                for page_num, page in enumerate(pdf.pages):
                    # Extract full page text
                    page_text = page.extract_text() or ""

                    # Extract page header (first 800 chars for context)
                    page_header = page_text[:800] if page_text else ""

                    # Extract all tables from this page
                    raw_tables = page.extract_tables(table_settings=self.table_settings)

                    # Convert raw tables to DataFrames
                    tables = []
                    for raw_table in raw_tables or []:
                        if self._is_valid_table(raw_table):
                            df = self._create_dataframe(raw_table)
                            if not df.empty:
                                tables.append(df)

                    # Only include pages with at least one valid table
                    if tables:
                        page_data = PageData(
                            page_number=page_num,
                            page_text=page_text,
                            page_header=page_header,
                            tables=tables,
                            source_document=pdf_path.name,
                        )
                        pages_data.append(page_data)
                        logger.debug(
                            f"Extracted page {page_num + 1} with {len(tables)} table(s)"
                        )
                    else:
                        logger.debug(
                            f"Skipping page {page_num + 1} - no valid tables found"
                        )

            logger.info(
                f"Extracted {len(pages_data)} pages with tables from {pdf_path.name}"
            )
            return pages_data

        except pdfplumber.PDFSyntaxError as e:
            raise PageExtractionError(f"Invalid PDF file {pdf_path.name}: {e}") from e
        except Exception as e:
            raise PageExtractionError(
                f"Failed to extract pages from {pdf_path.name}: {e}"
            ) from e

    def _is_valid_table(self, raw_table: list[list[any]] | None) -> bool:
        """Check if a raw table meets minimum validity criteria.

        Args:
            raw_table: Raw table data from pdfplumber.

        Returns:
            True if table is valid, False otherwise.
        """
        if not raw_table:
            return False

        if len(raw_table) < self.min_table_rows:
            return False

        # Check if table has enough columns
        if raw_table and len(raw_table[0]) < self.min_table_cols:
            return False

        # Check if table is not completely empty
        non_empty_cells = sum(
            1 for row in raw_table for cell in row if cell and str(cell).strip()
        )

        if non_empty_cells < 2:
            return False

        return True

    def _create_dataframe(self, raw_table: list[list[any]]) -> pd.DataFrame:
        """Create a pandas DataFrame from raw table data.

        Handles:
        - Empty cells (replaces with None)
        - Header extraction (uses first row as header if suitable)
        - Data cleaning (strips whitespace)

        Args:
            raw_table: Raw table data from pdfplumber.

        Returns:
            Cleaned pandas DataFrame.
        """
        if not raw_table or len(raw_table) < 2:
            return pd.DataFrame()

        # Clean all cells: strip whitespace and convert empty strings to None
        cleaned_table = [
            [
                cell.strip() if isinstance(cell, str) and cell.strip() else None
                for cell in row
            ]
            for row in raw_table
        ]

        # Use first row as header if it looks like a header
        first_row = cleaned_table[0]
        has_header = any(cell and str(cell).strip() for cell in first_row)

        if has_header and len(cleaned_table) > 1:
            # Extract header and data rows
            header = [
                str(cell) if cell else f"Col_{i}" for i, cell in enumerate(first_row)
            ]
            data_rows = cleaned_table[1:]

            # Ensure all rows have the same number of columns as header
            data_rows = [
                (
                    row + [None] * (len(header) - len(row))
                    if len(row) < len(header)
                    else row[: len(header)]
                )
                for row in data_rows
            ]

            df = pd.DataFrame(data_rows, columns=header)
        else:
            # No clear header, use generic column names
            df = pd.DataFrame(cleaned_table)
            df.columns = [f"Col_{i}" for i in range(len(df.columns))]

        # Remove completely empty rows
        df = df.dropna(how="all")

        # Remove completely empty columns
        df = df.dropna(axis=1, how="all")

        return df
