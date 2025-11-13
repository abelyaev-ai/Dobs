"""Service catalog extraction and management.

Extracts canonical service names from PDFs using ® symbol pattern.
Provides service name lookup and fuzzy matching for user queries.
"""

from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ServiceCatalog:
    """Manages canonical service names extracted from PDFs."""

    def __init__(self) -> None:
        """Initialize empty service catalog."""
        self.services: set[str] = set()

    def extract_from_pdf(self, pdf_path: str | Path) -> list[str]:
        """Extract all canonical service names from a PDF.

        Scans entire PDF for patterns like "FedEx Service®" and extracts
        unique service names.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of canonical service names found.
        """
        services_found: set[str] = set()

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""

                # Look for all ® symbols and extract the word/phrase before it
                # Pattern captures up to 6 words before ®
                pattern = r"(?:FedEx\s+)?([A-Z][A-Za-z0-9]+(?:\s+[A-Z]?[A-Za-z0-9\.]+){0,5}?)\s*®"
                matches = re.findall(pattern, text)

                # Special handling for "2Day" patterns which sometimes appear without full prefix
                # e.g., "A.M. 2Day®" → extract "2Day A.M."
                # Pattern 1: "A.M. 2Day®" (reversed order in text)
                twoday_am_pattern = r"A\.M\.\s+(2Day)\s*®"
                for match in re.findall(twoday_am_pattern, text):
                    matches.append("2Day A.M.")

                # Pattern 2: Just "2Day®"
                twoday_pattern = r"\b(2Day)\s*®"
                twoday_matches = re.findall(twoday_pattern, text)
                matches.extend(twoday_matches)

                for match in matches:
                    # Clean the service name
                    service_name = match.strip()

                    # Normalize whitespace (collapse multiple spaces and newlines)
                    service_name = re.sub(r"\s+", " ", service_name)

                    # Skip if too short or too long
                    if len(service_name) < 3 or len(service_name) > 50:
                        continue

                    # Skip if contains "FedEx" within the match (not prefix)
                    if "FedEx" in service_name:
                        continue

                    # Skip if starts with lowercase
                    if service_name[0].islower():
                        continue

                    # Skip garbage patterns
                    skip_terms = [
                        "Office",
                        "Delivery Manager",
                        "Ship Manager",
                        "commitment",
                        "ZIP code",
                        "Next day",
                        "RateToolsMain",
                        "Clearance",
                        "One-pound",
                        "rate applies",
                        "Account",
                        "Service Guide",
                        "Freight rate",
                        "rates and",
                        "and FedEx",
                        "kg Box",
                    ]
                    if any(term in service_name for term in skip_terms):
                        continue

                    # Skip country-specific patterns (e.g., "Austria Envelope First")
                    # These appear in international rate tables but aren't service names
                    country_indicators = [
                        "Andorra",
                        "Anguilla",
                        "Antigua",
                        "Argentina",
                        "Australia",
                        "Austria",
                        "Bahrain",
                        "Bangladesh",
                        "Belgium",
                        "Belize",
                        "Bhutan",
                        "Brunei",
                        "Bulgaria",
                        "Canada",
                        "China",
                        "Czech",
                        "Guam",
                        "Japan",
                        "Mexico",
                        "Morocco",
                        "Philippines",
                        "South ",  # "South Korea", "South Africa", etc.
                    ]
                    if any(country in service_name for country in country_indicators):
                        continue

                    # Skip generic fragments that aren't complete service names
                    invalid_services = [
                        "Day",
                        "Economy",
                        "Priority",
                        "Saver",
                        "Overnight",
                        "Weight First",
                        "Envelope First",
                        "Weight Overnight",
                        "Service Overnight",
                        "Express Priority",
                        "First Priority 2Day",  # Incorrect extraction
                        "First Priority Standard 2Day",  # Incorrect extraction
                        "Express Service Overnight",  # Should be "Standard Overnight"
                        "Express Weight Overnight",  # Should be "Priority/First Overnight"
                        "Ex Ground Multiweight",  # Partial match
                        "International First",  # Incomplete
                        "International International Connect Envelope First",  # Duplicate
                        "International International Connect Weight First",  # Duplicate
                        "International International Priority",  # Duplicate "International"
                    ]
                    if service_name in invalid_services:
                        continue

                    # Skip if contains multiple numbers (likely table data or page numbers)
                    if re.search(r"\d+\s+\d+", service_name):
                        continue

                    # Skip if contains "Zones" followed by numbers
                    if re.search(r"Zones\s+\d", service_name):
                        continue

                    # Skip if ends with standalone numbers (page numbers)
                    if re.search(r"\d+$", service_name):
                        continue

                    # Add with "FedEx" prefix for canonical form
                    canonical = f"FedEx {service_name}"
                    services_found.add(canonical)

                    logger.debug(
                        f"Found service '{canonical}' on page {page_num} of {pdf_path}"
                    )

        # Store in catalog
        self.services.update(services_found)

        # Return sorted list
        result = sorted(services_found)
        logger.info(f"Extracted {len(result)} canonical service names from {pdf_path}")

        return result

    def get_all_services(self) -> list[str]:
        """Get all services in catalog.

        Returns:
            Sorted list of all canonical service names.
        """
        return sorted(self.services)

    def normalize_service_name(self, service: str) -> str:
        """Normalize a service name to canonical form.

        Args:
            service: Service name (may or may not have "FedEx" prefix).

        Returns:
            Canonical service name with "FedEx" prefix.
        """
        # Remove extra whitespace
        service = re.sub(r"\s+", " ", service.strip())

        # Ensure FedEx prefix
        if not service.startswith("FedEx "):
            service = f"FedEx {service}"

        return service

    def match_service(self, query: str) -> str | None:
        """Match a query string to a canonical service name.

        Uses exact matching and common variations.

        Args:
            query: User query (e.g., "2Day", "Priority Overnight").

        Returns:
            Canonical service name if match found, None otherwise.
        """
        # Normalize query
        query_normalized = query.strip()

        # Try exact match with FedEx prefix
        canonical = self.normalize_service_name(query_normalized)
        if canonical in self.services:
            return canonical

        # Try partial match (query is substring of service)
        query_lower = query_normalized.lower()
        for service in self.services:
            if query_lower in service.lower():
                return service

        # No match found
        return None
