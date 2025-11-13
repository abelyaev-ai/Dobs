"""Create document chunks from pages.

This module converts page data and metadata into LlamaIndex Document chunks,
with one chunk per page containing all tables and context.
"""

from __future__ import annotations

from llama_index.core import Document

from src.core.metadata_extractor import PageMetadata
from src.core.page_extractor import PageData
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PageChunker:
    """Create one Document chunk per page.

    Each chunk contains:
    - Metadata header with carrier, services, zones, weight range
    - All tables from the page in markdown format
    - Full page context preserved
    """

    def create_chunks(
        self, pages: list[PageData], metadata_list: list[PageMetadata]
    ) -> list[Document]:
        """Create Document chunks (one per page).

        Each chunk contains:
        - Page metadata header
        - All tables from page in markdown
        - Full page context

        Args:
            pages: List of PageData objects.
            metadata_list: List of PageMetadata objects (must match pages length).

        Returns:
            List of LlamaIndex Document objects, one per page.

        Raises:
            ValueError: If pages and metadata_list have different lengths.
        """
        if len(pages) != len(metadata_list):
            raise ValueError(
                f"Mismatch: {len(pages)} pages but {len(metadata_list)} metadata objects"
            )

        logger.info(f"Creating {len(pages)} page-based document chunks")

        documents: list[Document] = []

        for page_data, metadata in zip(pages, metadata_list, strict=False):
            # Build markdown content for this page
            content = self._build_page_content(page_data, metadata)

            # Create Document with rich metadata
            doc = Document(
                text=content,
                metadata={
                    "carrier": metadata.carrier_name,
                    "service_types": metadata.service_types,  # List of services
                    "zones": metadata.zones,  # List of zones
                    "page": metadata.page_number,
                    "source": metadata.source_document,
                    # FIXED: Store weight_range as tuple for filtering, not string
                    "weight_range": metadata.weight_range,  # tuple or None
                    "table_count": metadata.table_count,
                    # For backward compatibility with existing queries
                    "service_type": (
                        metadata.service_types[0]
                        if metadata.service_types
                        else "Unknown"
                    ),
                },
            )

            documents.append(doc)

            logger.debug(
                f"Created chunk for page {metadata.page_number + 1}: "
                f"{metadata.table_count} table(s), "
                f"{len(metadata.service_types)} service(s), "
                f"{len(metadata.zones)} zone(s)"
            )

        logger.info(f"Successfully created {len(documents)} page-based chunks")

        return documents

    def _build_page_content(self, page_data: PageData, metadata: PageMetadata) -> str:
        """Build markdown content for a page.

        Creates a structured markdown document with:
        1. Page metadata header (carrier, services, zones, weight range, page info)
        2. All tables from the page in markdown format
        3. Table separators for clarity

        Args:
            page_data: Raw page data with tables.
            metadata: Extracted metadata for the page.

        Returns:
            Markdown formatted string.
        """
        lines: list[str] = []

        # Add page-level metadata header
        lines.append("# Shipping Rate Page")
        lines.append("")
        lines.append("## Page Metadata")
        lines.append(f"- **Carrier**: {metadata.carrier_name}")

        if metadata.service_types:
            lines.append(f"- **Service Types**: {', '.join(metadata.service_types)}")
        else:
            lines.append("- **Service Types**: Unknown")

        if metadata.zones:
            lines.append(f"- **Zones**: {', '.join(metadata.zones)}")
        else:
            lines.append("- **Zones**: Unknown")

        if metadata.weight_range:
            min_wt, max_wt = metadata.weight_range
            lines.append(f"- **Weight Range**: {min_wt} - {max_wt} lbs")
        else:
            lines.append("- **Weight Range**: Unknown")

        lines.append(
            f"- **Source**: {metadata.source_document} (Page {metadata.page_number + 1})"
        )
        lines.append(f"- **Tables on Page**: {metadata.table_count}")
        lines.append("")

        # Add context from page header (first 400 chars, cleaned)
        if page_data.page_header:
            # Clean up the header text - remove excessive whitespace
            header_preview = " ".join(page_data.page_header[:400].split())
            lines.append("## Page Context")
            lines.append(f"{header_preview}...")
            lines.append("")

        # Add all tables from the page
        for table_idx, df in enumerate(page_data.tables):
            lines.append(f"## Table {table_idx + 1} of {len(page_data.tables)}")
            lines.append("")

            if not df.empty:
                # Convert DataFrame to markdown
                table_md = df.to_markdown(index=False)
                lines.append(table_md)
            else:
                lines.append("*Empty table*")

            lines.append("")

            # Add separator between tables (except after last table)
            if table_idx < len(page_data.tables) - 1:
                lines.append("---")
                lines.append("")

        return "\n".join(lines)
