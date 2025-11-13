"""Extract metadata from page data using priority-based logic.

This module handles metadata extraction from raw page data, using a priority-based
approach that checks page headers first, then table headers, then table data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.core.page_extractor import PageData
from src.core.service_catalog import ServiceCatalog
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PageMetadata:
    """Metadata for a page-level chunk.

    Attributes:
        carrier_name: Name of the shipping carrier.
        service_types: List of service types found on the page.
        zones: List of zone identifiers found on the page.
        page_number: Page number (0-indexed).
        source_document: Name of the source PDF file.
        weight_range: Tuple of (min_weight, max_weight) in pounds, or None.
        table_count: Number of tables on the page.
    """

    carrier_name: str
    service_types: list[str]
    zones: list[str]
    page_number: int
    source_document: str
    weight_range: tuple[float, float] | None
    table_count: int


class MetadataExtractor:
    """Extract metadata from pages with priority-based detection.

    Priority order for all metadata:
    1. Page header text (first 800 chars) - HIGHEST priority
    2. Table headers
    3. Table row data

    Attributes:
        service_catalog: ServiceCatalog for normalizing service names to canonical forms.
    """

    def __init__(self, service_catalog: ServiceCatalog) -> None:
        """Initialize MetadataExtractor.

        Args:
            service_catalog: ServiceCatalog for normalizing service names
                           to canonical forms.
        """
        self.service_catalog = service_catalog
        logger.debug(
            f"MetadataExtractor initialized with service catalog "
            f"({len(service_catalog.get_all_services())} services)"
        )

    # Service type patterns
    SERVICE_TYPE_PATTERNS: tuple[str, ...] = (
        r"2\s*Day",
        r"Standard\s+Overnight",
        r"Priority\s+Overnight",
        r"First\s+Overnight",
        r"Express\s+Saver",
        r"Ground",
        r"Home\s+Delivery",
        r"SmartPost",
        r"Freight",
        r"International\s+Priority",
        r"International\s+Economy",
    )

    # Zone patterns
    # FIXED: Exclude lowercase "zone f" false positives
    # Valid zones: numeric (2-8, 17, 22, etc.) or uppercase letters for international (A, B, AA, etc.)
    ZONE_PATTERNS: tuple[str, ...] = (
        r"Zone\s+\d+",  # Zone 2, Zone 5, Zone 17, etc.
        r"Z\s*\d+",     # Z2, Z5, Z 17, etc.
        r"Zone\s+[A-Z]{1,2}\b",  # Zone A, Zone B, Zone AA (international), but NOT "zone f"
    )

    # Weight patterns
    WEIGHT_PATTERNS: tuple[str, ...] = (
        r"lb[s]?",
        r"#",
        r"weight",
        r"wt\.?",
        r"pounds?",
    )

    def extract_metadata(self, page_data: PageData, carrier_name: str) -> PageMetadata:
        """Extract comprehensive metadata from page.

        Priority order:
        1. Page header text (first 800 chars)
        2. Table headers
        3. Table row data

        Args:
            page_data: Raw page data from PDFPageExtractor.
            carrier_name: Name of the shipping carrier.

        Returns:
            PageMetadata with comprehensive information about the page.
        """
        logger.debug(
            f"Extracting metadata for page {page_data.page_number + 1} "
            f"of {page_data.source_document}"
        )

        # Extract all services on page using generic ®-based extraction
        services = self._extract_all_services(page_data)

        # Extract all zones on page
        zones = self._extract_all_zones(page_data)

        # Extract weight range from all tables
        weight_range = self._extract_weight_range(page_data.tables)

        metadata = PageMetadata(
            carrier_name=carrier_name,
            service_types=list(set(services)),  # Unique services
            zones=list(set(zones)),  # Unique zones
            page_number=page_data.page_number,
            source_document=page_data.source_document,
            weight_range=weight_range,
            table_count=len(page_data.tables),
        )

        logger.debug(
            f"Page {page_data.page_number + 1} metadata: "
            f"{len(metadata.service_types)} service(s), "
            f"{len(metadata.zones)} zone(s), "
            f"weight_range={metadata.weight_range}, "
            f"{metadata.table_count} table(s)"
        )

        return metadata

    def _extract_all_services(self, page_data: PageData) -> list[str]:
        """Extract all service types from page using ® symbol pattern.

        Uses the registered trademark symbol (®) to identify FedEx service names,
        which is a reliable indicator across all PDF versions.

        If a service catalog is provided, normalizes extracted names to canonical forms.

        Args:
            page_data: Raw page data.

        Returns:
            List of unique service type strings (canonical if catalog provided).
        """
        services: list[str] = []

        # Collect individual text chunks (preserving boundaries)
        text_chunks: list[str] = []

        # 1. Page header
        text_chunks.append(page_data.page_header)

        # 2. Table headers (each column separately to preserve boundaries)
        for df in page_data.tables:
            for col in df.columns:
                text_chunks.append(str(col))

        # 3. First few rows of each table (each cell separately)
        for df in page_data.tables:
            if not df.empty:
                for row_idx in range(min(3, len(df))):
                    for col_idx in range(len(df.columns)):
                        cell_value = df.iloc[row_idx, col_idx]
                        if pd.notna(cell_value):
                            text_chunks.append(str(cell_value))

        # Extract services using ® pattern FROM EACH CHUNK INDIVIDUALLY
        # This preserves cell boundaries and prevents cross-cell matching
        pattern = r'(?:FedEx\s+)?([A-Z][A-Za-z0-9\s\-\.]+?)®'
        matches: list[str] = []

        for chunk in text_chunks:
            chunk_matches = re.findall(pattern, chunk, re.MULTILINE | re.DOTALL)
            matches.extend(chunk_matches)

        # Store potential services for catalog matching
        potential_services: list[str] = []

        for match in matches:
            service_name = match.strip()

            # Normalize whitespace (collapse newlines, multiple spaces into single spaces)
            service_name = re.sub(r'\s+', ' ', service_name)

            # Filter out very short matches
            if len(service_name) < 3:
                continue

            # Skip if longer than 50 characters (likely OCR garbage)
            if len(service_name) > 50:
                continue

            # Skip standalone "FedEx" (not a service name)
            if service_name == "FedEx":
                continue

            # Skip if contains multiple service names concatenated
            # (e.g., "FedEx FedEx FedEx FedEx FedEx First Priority Standard 2Day")
            if service_name.count("FedEx") > 2:
                continue

            # Filter out garbage patterns more aggressively
            skip_terms = [
                "Office",
                "Delivery Manager",
                "Ship Manager",
                "Account",
                "Next day by",
                "day by",
                "commitment",
                "Delivery commitment",
                "ZIP code",
                "RateToolsMain",
                "Clearance",
                "One-pound rate",
                "rate applies",
                "up to 8 oz",
                "lbs.",
                "oz.",
                "Express FedEx",  # Reversed extraction from page header
            ]
            if any(term in service_name for term in skip_terms):
                continue

            # Skip if contains "Zones" followed by numbers (e.g., "Zones 2 3 4")
            if re.search(r'Zones\s+\d', service_name):
                continue

            # Skip if contains country names with numbers (price tables)
            # e.g., "France FedEx", "Qatar 5 527.12"
            if re.search(r'\d+\.?\d*\s+\d+\.?\d*', service_name):
                continue

            # Skip if starts with "Weight" (table column headers)
            if service_name.startswith("Weight "):
                continue

            # Remove "FedEx" prefix if present for now
            if service_name.startswith("FedEx "):
                service_name = service_name[6:].strip()

            # Skip if empty after removing FedEx prefix
            if not service_name:
                continue

            # Add to potential services for catalog matching
            if service_name and service_name not in potential_services:
                potential_services.append(service_name)

        # Match potential services against canonical names from catalog
        canonical_services: list[str] = []
        catalog_services = self.service_catalog.get_all_services()

        for potential in potential_services:
            # Try to find matching canonical service
            matched = False
            for canonical in catalog_services:
                if self._services_match(potential, canonical):
                    if canonical not in canonical_services:
                        canonical_services.append(canonical)
                        logger.debug(
                            f"Matched '{potential}' -> '{canonical}' "
                            f"(page {page_data.page_number + 1})"
                        )
                    matched = True
                    break

            # If no match found, keep the extracted name
            if not matched:
                logger.debug(
                    f"No canonical match for '{potential}', keeping as-is "
                    f"(page {page_data.page_number + 1})"
                )
                if potential not in canonical_services:
                    canonical_services.append(potential)

        services = canonical_services

        # Add service aliases
        # Some OCR variations don't match canonical names exactly
        for service_name in canonical_services:
            # "Express Service Overnight" → also add "FedEx Standard Overnight"
            if "Express Service Overnight" in service_name or "Service Overnight" in service_name:
                if "FedEx Standard Overnight" not in services:
                    services.append("FedEx Standard Overnight")

        if not services:
            logger.debug(
                f"No services found on page {page_data.page_number + 1} - "
                "may be a non-rate page"
            )

        return services

    def _extract_all_zones(self, page_data: PageData) -> list[str]:
        """Extract all zone identifiers from page.

        Priority order:
        1. Page header text (first 800 chars)
        2. Table headers
        3. Table row data (first 5 rows)

        Args:
            page_data: Raw page data.

        Returns:
            List of zone identifier strings (e.g., ["Zone 2", "Zone 3"]).
        """
        zones: list[str] = []

        # PRIORITY 1: Check page header text
        for pattern in self.ZONE_PATTERNS:
            matches = re.finditer(pattern, page_data.page_header, re.IGNORECASE)
            for match in matches:
                zone = match.group(0).strip()
                if zone not in zones:
                    zones.append(zone)
                    logger.debug(
                        f"Found zone '{zone}' in page header "
                        f"(page {page_data.page_number + 1})"
                    )

        # Also check for simple zone number patterns in page header
        if page_data.page_header:
            # Look for "Zone X" patterns
            zone_with_text = re.findall(
                r"Zone\s+(\d+)", page_data.page_header, re.IGNORECASE
            )
            for zone_num in zone_with_text:
                zone_label = f"Zone {zone_num}"
                if zone_label not in zones:
                    zones.append(zone_label)
                    logger.debug(
                        f"Found zone '{zone_label}' in page header "
                        f"(page {page_data.page_number + 1})"
                    )

        # PRIORITY 2: Check table headers
        for table_idx, df in enumerate(page_data.tables):
            header_text = " ".join(str(col) for col in df.columns)

            for pattern in self.ZONE_PATTERNS:
                matches = re.finditer(pattern, header_text, re.IGNORECASE)
                for match in matches:
                    zone = match.group(0).strip()
                    if zone not in zones:
                        zones.append(zone)
                        logger.debug(
                            f"Found zone '{zone}' in table {table_idx} headers "
                            f"(page {page_data.page_number + 1})"
                        )

        # PRIORITY 3: Check first few rows for zone information
        if not zones:
            for table_idx, df in enumerate(page_data.tables):
                if df.empty:
                    continue

                for i in range(min(5, len(df))):
                    row_text = " ".join(str(val) for val in df.iloc[i] if pd.notna(val))

                    # Check for pattern like "2 3 4 5 6 7 8" which indicates zones
                    simple_zone_pattern = r"\b([2-9])\s+([2-9])\s+([2-9])"
                    if re.search(simple_zone_pattern, row_text):
                        # Extract individual zone numbers
                        zone_nums = re.findall(r"\b([2-9])\b", row_text)
                        for zone_num in zone_nums[:8]:  # Max 8 zones typically
                            zone_label = f"Zone {zone_num}"
                            if zone_label not in zones:
                                zones.append(zone_label)
                                logger.debug(
                                    f"Found zone '{zone_label}' in table {table_idx} row {i} "
                                    f"(page {page_data.page_number + 1})"
                                )
                        break

                    # Regular zone pattern matching in rows
                    for pattern in self.ZONE_PATTERNS:
                        matches = re.finditer(pattern, row_text, re.IGNORECASE)
                        for match in matches:
                            zone = match.group(0).strip()
                            if zone not in zones:
                                zones.append(zone)
                                logger.debug(
                                    f"Found zone '{zone}' in table {table_idx} row {i} "
                                    f"(page {page_data.page_number + 1})"
                                )

        if not zones:
            logger.debug(
                f"No zones found on page {page_data.page_number + 1} - "
                "may be a non-rate page"
            )

        return zones

    def _extract_weight_range(
        self, tables: list[pd.DataFrame]
    ) -> tuple[float, float] | None:
        """Extract weight range from all tables on page.

        Finds the minimum and maximum weight values across all tables.

        Args:
            tables: List of DataFrames from the page.

        Returns:
            Tuple of (min_weight, max_weight) in pounds, or None if no weights found.
        """
        all_weights: list[float] = []

        for df in tables:
            if df.empty:
                continue

            # Find weight column
            weight_col_idx = self._find_weight_column(df)
            if weight_col_idx is None:
                continue

            weight_col_name = df.columns[weight_col_idx]
            weight_series = df[weight_col_name]

            # Extract numeric weights
            for weight_val in weight_series:
                if pd.isna(weight_val):
                    continue

                weight_str = str(weight_val).strip()

                # Try to extract numeric value (handles "150 lbs", "150", "150#", etc.)
                match = re.search(r"(\d+(?:\.\d+)?)", weight_str)
                if match:
                    try:
                        all_weights.append(float(match.group(1)))
                    except ValueError:
                        continue

        if not all_weights:
            return None

        return (min(all_weights), max(all_weights))

    def _find_weight_column(self, df: pd.DataFrame) -> int | None:
        """Find the index of the weight column in the DataFrame.

        Args:
            df: DataFrame to search.

        Returns:
            Column index of weight column, or None if not found.
        """
        for col_idx, col_name in enumerate(df.columns):
            col_text = str(col_name).lower()

            for pattern in self.WEIGHT_PATTERNS:
                if re.search(pattern, col_text, re.IGNORECASE):
                    return col_idx

        return None

    def _services_match(self, extracted: str, canonical: str) -> bool:
        """Check if extracted service matches canonical name.

        Uses fuzzy matching with normalized comparisons and substring matching.

        Args:
            extracted: Service name extracted from page.
            canonical: Canonical service name from catalog.

        Returns:
            True if services match, False otherwise.
        """
        # Normalize both strings
        ext_norm = extracted.lower().strip()
        can_norm = canonical.lower().strip()

        # Remove "FedEx" prefix for comparison
        ext_norm = ext_norm.replace("fedex ", "").strip()
        can_norm = can_norm.replace("fedex ", "").strip()

        # Exact match after normalization
        if ext_norm == can_norm:
            return True

        # Substring match (one contains the other)
        if ext_norm in can_norm or can_norm in ext_norm:
            return True

        return False

    def _normalize_service_name(self, raw_service_name: str) -> str:
        """Normalize a service name for consistency.

        Args:
            raw_service_name: Raw service name extracted from header.

        Returns:
            Normalized service name with consistent spacing and capitalization.
        """
        # Remove newlines and extra whitespace first
        normalized = raw_service_name.replace("\n", " ").replace("\r", " ")
        normalized = " ".join(normalized.split())

        # Standardize common abbreviations and variations
        replacements = {
            r"2\s*Day": "2Day",
            r"Express\s+Saver": "Express Saver",
            r"Standard\s+Overnight": "Standard Overnight",
            r"Priority\s+Overnight": "Priority Overnight",
            r"First\s+Overnight": "First Overnight",
            r"Home\s+Delivery": "Home Delivery",
            r"International\s+Priority": "International Priority",
            r"International\s+Economy": "International Economy",
        }

        for pattern, replacement in replacements.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # Capitalize first letter of each word for consistency
        return normalized.title()
