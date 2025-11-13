"""LLM-based query parser using structured output for intelligent query understanding.

This module provides an elegant alternative to regex/alias-based parsing by using
LLM reasoning to understand shipping queries naturally, including:
- Abbreviations and typos
- Natural language variations
- Different word orders
- Missing or ambiguous information

The parser uses OpenAI's structured output with function calling to ensure
type-safe, validated results with built-in caching for performance.
"""

from __future__ import annotations

import json
from typing import Optional
from functools import lru_cache

from pydantic import BaseModel, Field
from llama_index.core import Settings

from src.core.query_parser import QueryParser, ShippingQuery
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMParsedQuery(BaseModel):
    """Structured output schema for LLM query parsing.

    This model defines the expected JSON structure from the LLM,
    ensuring type safety and validation.
    """

    carrier: Optional[str] = Field(
        None,
        description="Shipping carrier (FedEx, UPS, USPS). Null if not specified."
    )
    service: Optional[str] = Field(
        None,
        description=(
            "CANONICAL service name from the provided catalog. "
            "MUST match exactly (e.g., 'FedEx Express Saver', 'FedEx Priority Overnight', 'FedEx 2Day'). "
            "Map user abbreviations to the full canonical form with 'FedEx' prefix."
        )
    )
    zone: Optional[str] = Field(
        None,
        description="Shipping zone as a string (e.g., '2', '5', '8'). Null if not specified."
    )
    weight: Optional[float] = Field(
        None,
        description="Package weight in pounds. Null if not specified."
    )
    packaging: Optional[str] = Field(
        None,
        description="Package type (e.g., 'envelope', 'pak', 'box'). Null if not specified."
    )
    reasoning: str = Field(
        ...,
        description=(
            "Brief explanation of how you parsed this query, "
            "including any assumptions or mappings made."
        )
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'"
    )


class LLMQueryParser:
    """Intelligent query parser using LLM reasoning with structured output.

    This parser uses OpenAI's GPT models with function calling to parse shipping
    queries intelligently, handling abbreviations, typos, and natural language
    variations that would be difficult with regex-based approaches.

    Features:
    - Structured output using Pydantic models
    - Built-in result caching for performance
    - Fallback to regex parser on errors
    - Detailed reasoning and confidence scores

    Example:
        >>> parser = LLMQueryParser()
        >>> result = parser.parse("Expr saver z8 2lb")
        >>> print(result.service)
        "Express Saver"
    """

    def __init__(
        self,
        service_catalog: Optional[list[str]] = None,
        fallback_parser: Optional[QueryParser] = None,
        cache_size: int = 1000
    ):
        """Initialize the LLM query parser.

        Args:
            service_catalog: List of canonical service names from the PDF catalog.
                           If None, uses default hardcoded service list.
            fallback_parser: Optional QueryParser to use if LLM parsing fails.
                           If None, creates a new QueryParser instance.
            cache_size: Maximum number of queries to cache (default: 1000).
        """
        self.llm = Settings.llm
        self.fallback = fallback_parser or QueryParser()
        self.cache_size = cache_size
        self.service_catalog = service_catalog or self._get_default_services()

        # Set up the parsing function with caching
        self._parse_with_cache = lru_cache(maxsize=cache_size)(self._parse_with_llm)

        logger.info(
            f"LLMQueryParser initialized with cache_size={cache_size}, "
            f"service_catalog={len(self.service_catalog)} services"
        )

    @staticmethod
    def _get_default_services() -> list[str]:
        """Get default list of canonical service names.

        This fallback list is used when no service catalog is provided.
        Ideally, the service catalog should come from PDF extraction.

        Returns:
            List of canonical FedEx service names
        """
        return [
            "FedEx First Overnight",
            "FedEx Priority Overnight",
            "FedEx Standard Overnight",
            "FedEx 2Day A.M.",
            "FedEx 2Day",
            "FedEx Express Saver",
            "FedEx Ground",
            "FedEx Home Delivery",
            "FedEx International Priority",
            "FedEx International Economy",
        ]

    def _build_system_prompt(self) -> str:
        """Build the system prompt dynamically using the service catalog.

        Returns:
            System prompt with canonical service names from the catalog
        """
        # Format service list with common abbreviations/mappings
        service_list = "\n".join(f"- {service}" for service in self.service_catalog)

        return f"""You are an expert shipping rate query parser. Your job is to extract structured information from natural language shipping queries and map service names to their CANONICAL forms.

**SERVICE MAPPING (CRITICAL):**
The user may refer to services using short names, abbreviations, or partial names. You MUST map them to the exact canonical service names from this catalog:

{service_list}

**Mapping Examples:**
- "Standard Overnight" → "FedEx Standard Overnight"
- "Priority Overnight" → "FedEx Priority Overnight"
- "First Overnight" → "FedEx First Overnight"
- "2Day AM" → "FedEx 2Day A.M."
- "2Day" → "FedEx 2Day"
- "Express Saver" → "FedEx Express Saver"
- "Ground" → "FedEx Ground"
- "Home Delivery" → "FedEx Home Delivery"
- "overnight" (ambiguous) → "FedEx Standard Overnight" (most common)
- "express" → "FedEx Express Saver"
- "2 day" → "FedEx 2Day"

**Common Abbreviations to Handle:**
- "std overnight", "std ovn", "so" → "FedEx Standard Overnight"
- "pri overnight", "priority ovn", "po" → "FedEx Priority Overnight"
- "first ovn", "fo" → "FedEx First Overnight"
- "2day am", "2d am" → "FedEx 2Day A.M."
- "2 day", "two day", "2-day", "2d" → "FedEx 2Day"
- "expr saver", "exp saver", "es" → "FedEx Express Saver"
- "gnd", "grnd" → "FedEx Ground"
- "home del", "hd", "residential" → "FedEx Home Delivery"

**Carriers:**
- FedEx (variations: fedex, fed ex, fedx, fx)
- UPS
- USPS

**Parsing Rules:**
1. Be flexible with abbreviations and typos
2. Handle natural language - "three pounds" = 3.0 lbs
3. Support different word orders - "z5 2lb overnight" = "overnight 2lb z5"
4. Zone format: "zone 5", "z5", "z 5" all mean zone "5"
5. Weight format: "3lb", "3 lbs", "3 pounds" all mean 3.0
6. Always return the **FULL CANONICAL service name** from the catalog above
7. If the user's service is ambiguous, choose the most common/likely match
8. Return null for fields that cannot be determined
9. Provide reasoning for your mapping decisions
10. Be confident with clear queries, less confident with ambiguous ones

**IMPORTANT:** The service name you return MUST exactly match one of the canonical names listed above. This ensures proper lookup in the pricing index."""

    def parse(self, query_text: str) -> ShippingQuery:
        """Parse a shipping query using LLM reasoning.

        This method:
        1. Normalizes the query for cache lookup
        2. Calls LLM with structured output
        3. Validates the result
        4. Falls back to regex parser on errors

        Args:
            query_text: Natural language shipping query

        Returns:
            ShippingQuery object with extracted fields

        Example:
            >>> parser.parse("fedx 2day zone 5 three pounds")
            ShippingQuery(carrier='FedEx', service='2Day', zone='5', weight=3.0, ...)
        """
        if not query_text or not query_text.strip():
            logger.warning("Empty query provided to LLM parser")
            return ShippingQuery(raw_query=query_text)

        # Normalize for caching (lowercase, strip whitespace)
        normalized_query = query_text.lower().strip()

        try:
            # Try cached LLM parsing
            llm_result = self._parse_with_cache(normalized_query)

            # Log the mapping, emphasizing canonical service name
            if llm_result.service:
                logger.info(
                    f"LLM parsed '{query_text}' → "
                    f"carrier={llm_result.carrier}, service='{llm_result.service}' (CANONICAL), "
                    f"zone={llm_result.zone}, weight={llm_result.weight} "
                    f"(confidence: {llm_result.confidence})"
                )
                logger.debug(f"Service mapping reasoning: {llm_result.reasoning}")
            else:
                logger.info(
                    f"LLM parsed '{query_text}' → "
                    f"carrier={llm_result.carrier}, service=None, "
                    f"zone={llm_result.zone}, weight={llm_result.weight} "
                    f"(confidence: {llm_result.confidence})"
                )
                logger.warning(f"No service extracted from query: {llm_result.reasoning}")

            # Convert to ShippingQuery
            return ShippingQuery(
                raw_query=query_text,
                carrier=llm_result.carrier,
                service=llm_result.service,
                zone=llm_result.zone,
                weight=llm_result.weight,
                packaging=llm_result.packaging,
            )

        except Exception as e:
            logger.warning(
                f"LLM parsing failed for '{query_text}': {e}. "
                f"Falling back to regex parser."
            )

            # Fallback to regex parser
            return self.fallback.parse(query_text)

    def _parse_with_llm(self, normalized_query: str) -> LLMParsedQuery:
        """Internal method to parse query with LLM (cached by lru_cache).

        Args:
            normalized_query: Normalized query string for cache key

        Returns:
            LLMParsedQuery with structured output from LLM

        Raises:
            Exception: If LLM call fails or returns invalid format
        """
        # Construct the user message
        user_message = f"Parse this shipping query: \"{normalized_query}\""

        # Define the JSON schema for structured output
        # This tells OpenAI to return data matching our Pydantic model
        # Note: OpenAI strict mode requires all properties in "required" array
        json_schema = {
            "type": "object",
            "properties": {
                "carrier": {
                    "type": ["string", "null"],
                    "description": "Shipping carrier (FedEx, UPS, USPS)"
                },
                "service": {
                    "type": ["string", "null"],
                    "description": "Full canonical service name (e.g., 'Express Saver')"
                },
                "zone": {
                    "type": ["string", "null"],
                    "description": "Shipping zone as string (e.g., '5', '8')"
                },
                "weight": {
                    "type": ["number", "null"],
                    "description": "Package weight in pounds"
                },
                "packaging": {
                    "type": ["string", "null"],
                    "description": "Package type (envelope, pak, box)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of parsing decisions"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level"
                }
            },
            "required": ["carrier", "service", "zone", "weight", "packaging", "reasoning", "confidence"],
            "additionalProperties": False
        }

        # Make LLM call with structured output
        # Using OpenAI's chat completion with response_format
        try:
            from openai import OpenAI
            client = OpenAI()  # Uses OPENAI_API_KEY from environment

            # Build dynamic system prompt with service catalog
            system_prompt = self._build_system_prompt()

            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Fast and cheap model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "shipping_query",
                        "schema": json_schema,
                        "strict": True
                    }
                },
                temperature=0.0,  # Deterministic output
            )

            # Parse JSON response
            content = response.choices[0].message.content
            result_dict = json.loads(content)

            # Validate with Pydantic
            return LLMParsedQuery(**result_dict)

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def clear_cache(self):
        """Clear the query parsing cache.

        Useful for testing or if the cache grows too large.
        """
        self._parse_with_cache.cache_clear()
        logger.info("LLM query parser cache cleared")

    def get_cache_info(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache hits, misses, size, and maxsize
        """
        info = self._parse_with_cache.cache_info()
        return {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
            "maxsize": info.maxsize,
            "hit_rate": info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }

    def validate_query(self, query: ShippingQuery) -> tuple[bool, list[str]]:
        """Validate a parsed shipping query.

        Checks if required fields (service, zone, weight) are present.

        Args:
            query: ShippingQuery object to validate

        Returns:
            Tuple of (is_valid, missing_fields) where:
            - is_valid: True if all required fields present
            - missing_fields: List of missing field names (empty if valid)
        """
        required_fields = {
            "service": query.service,
            "zone": query.zone,
            "weight": query.weight,
        }

        missing = [field for field, value in required_fields.items() if not value]

        return (len(missing) == 0, missing)
