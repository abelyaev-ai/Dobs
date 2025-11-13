"""Query parsing module for extracting shipping parameters from natural language.

This module provides robust parsing capabilities for free-form shipping queries,
extracting carrier, service type, zone, weight, and packaging information using
regex patterns and normalization rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryParserError(Exception):
    """Raised when query parsing fails."""


@dataclass
class ShippingQuery:
    """Structured representation of a shipping query.

    Attributes:
        carrier: Shipping carrier name (e.g., "FedEx", "UPS").
        service: Service type (e.g., "2Day", "Ground", "Standard Overnight").
        zone: Shipping zone (e.g., "2", "5", "8").
        weight: Package weight in pounds.
        packaging: Packaging type (e.g., "other", "envelope").
        raw_query: Original query string for reference.
    """

    carrier: Optional[str] = None
    service: Optional[str] = None
    zone: Optional[str] = None
    weight: Optional[float] = None
    packaging: Optional[str] = None
    raw_query: str = ""

    def __str__(self) -> str:
        """Return human-readable representation of the query."""
        parts = []
        if self.carrier:
            parts.append(f"Carrier: {self.carrier}")
        if self.service:
            parts.append(f"Service: {self.service}")
        if self.zone:
            parts.append(f"Zone: {self.zone}")
        if self.weight:
            parts.append(f"Weight: {self.weight} lb")
        if self.packaging:
            parts.append(f"Packaging: {self.packaging}")
        return ", ".join(parts) if parts else "Empty query"


class QueryParser:
    """Parser for extracting shipping parameters from natural language queries.

    This parser uses regex patterns to extract carrier, service type, zone,
    weight, and packaging information from free-form text queries.

    Example:
        >>> parser = QueryParser()
        >>> query = parser.parse("FedEx 2Day, Zone 5, 3 lb")
        >>> print(query.carrier, query.service, query.zone, query.weight)
        FedEx 2Day 5 3.0
    """

    # Carrier patterns
    CARRIER_PATTERNS = {
        "FedEx": r"\bFedEx\b",
        "UPS": r"\bUPS\b",
        "USPS": r"\bUSPS\b",
    }

    # Service type patterns (order matters - match longer patterns first)
    SERVICE_PATTERNS = {
        "Standard Overnight": r"\bStandard\s+Overnight\b",
        "Priority Overnight": r"\bPriority\s+Overnight\b",
        "First Overnight": r"\bFirst\s+Overnight\b",
        "2Day AM": r"\b2Day\s+AM\b",
        "2Day": r"\b2Day\b",
        "Express Saver": r"\bExpress\s+Saver\b",
        "Ground": r"\bGround\b",
        "Home Delivery": r"\bHome\s+Delivery\b",
        "International Priority": r"\bInternational\s+Priority\b",
        "International Economy": r"\bInternational\s+Economy\b",
    }

    # Service name aliases and variants for normalization
    SERVICE_ALIASES = {
        # Standard Overnight variants
        "Standard Overnight": [
            "standard overnight",
            "std overnight",
            "standard ovn",
            "std ovn",
            "so",
        ],
        # 2Day variants
        "2Day": [
            "2day",
            "2 day",
            "two day",
            "fedex 2day",
            "2-day",
        ],
        # Express Saver variants
        "Express Saver": [
            "express saver",
            "exp saver",
            "express svr",
            "exp svr",
            "es",
        ],
        # Ground variants
        "Ground": [
            "ground",
            "gnd",
            "fedex ground",
        ],
        # Home Delivery variants
        "Home Delivery": [
            "home delivery",
            "home del",
            "hd",
            "residential",
        ],
        # Priority Overnight variants
        "Priority Overnight": [
            "priority overnight",
            "pri overnight",
            "priority ovn",
            "po",
        ],
        # First Overnight variants
        "First Overnight": [
            "first overnight",
            "first ovn",
            "fo",
        ],
    }

    # Zone patterns - match "Zone 5", "Z5", "zone5", etc.
    ZONE_PATTERN = r"(?:zone|z)\s*(\d+)"

    # Weight patterns - match "3 lb", "10 lbs", "5.5 pounds", "12#"
    WEIGHT_PATTERN = r"(\d+(?:\.\d+)?)\s*(?:lb|lbs|pound|pounds|#)"

    # Packaging patterns
    PACKAGING_PATTERNS = {
        "envelope": r"\benvelope\b",
        "pak": r"\bpak\b",
        "box": r"\bbox\b",
        "tube": r"\btube\b",
        "other": r"\bother\s+packaging\b",
    }

    def __init__(self) -> None:
        """Initialize the QueryParser."""
        logger.debug("QueryParser initialized")

    def parse(self, query: str) -> ShippingQuery:
        """Parse a shipping query string into structured components.

        Args:
            query: Natural language query string.

        Returns:
            ShippingQuery object with extracted parameters.

        Raises:
            QueryParserError: If query is empty or parsing fails critically.

        Example:
            >>> parser = QueryParser()
            >>> result = parser.parse("Express Saver Z8 1 lb")
            >>> print(result.service, result.zone, result.weight)
            Express Saver 8 1.0
        """
        if not query or not query.strip():
            raise QueryParserError("Query string cannot be empty")

        query_lower = query.lower()
        logger.info(f"Parsing query: {query}")

        result = ShippingQuery(raw_query=query)

        # Extract carrier
        result.carrier = self._extract_carrier(query)

        # Extract service type
        result.service = self._extract_service(query)

        # Extract zone
        result.zone = self._extract_zone(query_lower)

        # Extract weight
        result.weight = self._extract_weight(query_lower)

        # Extract packaging
        result.packaging = self._extract_packaging(query_lower)

        logger.info(f"Parsed query: {result}")

        return result

    def _extract_carrier(self, query: str) -> Optional[str]:
        """Extract carrier name from query.

        Args:
            query: Query string (case-sensitive for carrier names).

        Returns:
            Carrier name or None if not found.
        """
        for carrier, pattern in self.CARRIER_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(f"Extracted carrier: {carrier}")
                return carrier
        return None

    def _normalize_service(self, service_raw: str) -> Optional[str]:
        """Normalize service name using aliases.

        Args:
            service_raw: Raw service name from query.

        Returns:
            Normalized service name or None if not found.
        """
        service_lower = service_raw.lower().strip()

        # Check each canonical service name and its aliases
        for canonical_name, aliases in self.SERVICE_ALIASES.items():
            if service_lower in aliases:
                logger.debug(f"Normalized '{service_raw}' to '{canonical_name}'")
                return canonical_name

        # If no alias match, return original (might match via regex)
        return None

    def _extract_service(self, query: str) -> Optional[str]:
        """Extract service type from query.

        Args:
            query: Query string (case-insensitive).

        Returns:
            Service type or None if not found.
        """
        # First try regex patterns (exact matches)
        for service, pattern in self.SERVICE_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(f"Extracted service via regex: {service}")
                return service

        # If regex fails, try normalization via aliases
        # Extract potential service words from query
        query_lower = query.lower()
        for canonical_name, aliases in self.SERVICE_ALIASES.items():
            for alias in aliases:
                if alias in query_lower:
                    logger.debug(f"Extracted service via alias: {canonical_name} (matched '{alias}')")
                    return canonical_name

        return None

    def _extract_zone(self, query: str) -> Optional[str]:
        """Extract zone from query.

        Args:
            query: Query string (lowercase).

        Returns:
            Zone number as string or None if not found.
        """
        match = re.search(self.ZONE_PATTERN, query, re.IGNORECASE)
        if match:
            zone = match.group(1)
            logger.debug(f"Extracted zone: {zone}")
            return zone
        return None

    def _extract_weight(self, query: str) -> Optional[float]:
        """Extract weight from query.

        Args:
            query: Query string (lowercase).

        Returns:
            Weight in pounds as float or None if not found.

        Raises:
            QueryParserError: If weight value is invalid.
        """
        match = re.search(self.WEIGHT_PATTERN, query, re.IGNORECASE)
        if match:
            try:
                weight = float(match.group(1))
                logger.debug(f"Extracted weight: {weight} lb")
                return weight
            except ValueError as e:
                raise QueryParserError(
                    f"Invalid weight value: {match.group(1)}"
                ) from e
        return None

    def _extract_packaging(self, query: str) -> Optional[str]:
        """Extract packaging type from query.

        Args:
            query: Query string (lowercase).

        Returns:
            Packaging type or None if not found.
        """
        for packaging, pattern in self.PACKAGING_PATTERNS.items():
            if re.search(pattern, query, re.IGNORECASE):
                logger.debug(f"Extracted packaging: {packaging}")
                return packaging
        return None

    def validate_query(self, query: ShippingQuery) -> tuple[bool, list[str]]:
        """Validate that a query has minimum required fields.

        Args:
            query: Parsed ShippingQuery object.

        Returns:
            Tuple of (is_valid, list of missing fields).

        Example:
            >>> parser = QueryParser()
            >>> query = parser.parse("Zone 5")
            >>> is_valid, missing = parser.validate_query(query)
            >>> print(missing)
            ['service', 'weight']
        """
        missing_fields = []

        # Service type is critical for identifying the right table
        if not query.service:
            missing_fields.append("service")

        # Weight is required for price lookup
        if query.weight is None:
            missing_fields.append("weight")

        # Zone is required for most services
        if not query.zone:
            missing_fields.append("zone")

        is_valid = len(missing_fields) == 0

        if not is_valid:
            logger.warning(
                f"Query validation failed. Missing fields: {missing_fields}"
            )

        return is_valid, missing_fields
