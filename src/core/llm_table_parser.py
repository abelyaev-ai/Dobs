"""LLM-based table parser for extracting shipping prices.

This module provides an elegant, simple alternative to programmatic table parsing
by using LLM intelligence to extract prices from any table format. It handles:
- Multi-service tables (Express Saver, 2Day, Priority Overnight)
- Single-service tables (Ground, Home Delivery)
- Any future table format variations

The parser uses OpenAI's structured output with built-in caching for performance.
"""

from __future__ import annotations

import json
import hashlib
from functools import lru_cache
from typing import Tuple, Optional
from collections import OrderedDict

from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level cache (shared across all LLMTableParser instances)
# This ensures caching works even when creating new parser instances
_GLOBAL_PRICE_CACHE: OrderedDict[str, "PriceExtractionResult"] = OrderedDict()
_MAX_CACHE_SIZE = 10000


class PriceExtractionResult(BaseModel):
    """Structured output schema for LLM price extraction."""

    price: Optional[float] = Field(
        None,
        description="The extracted price in dollars, or null if not found"
    )
    confidence: str = Field(
        ...,
        description="Confidence level: 'high', 'medium', or 'low'"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of how the price was found"
    )


class LLMTableParserError(Exception):
    """Raised when LLM table parsing fails."""
    pass


class LLMTableParser:
    """Simple, elegant table parser using LLM intelligence.

    This parser replaces complex programmatic logic with LLM reasoning,
    allowing it to handle any table format without hardcoded rules.

    Features:
    - Handles any table format (multi-service, single-service, etc.)
    - Built-in LRU caching for performance (<1ms for cached queries)
    - OpenAI structured output for reliability
    - ~50 lines of code vs 400+ for programmatic approach

    Example:
        >>> parser = LLMTableParser()
        >>> price, confidence = parser.parse_price(table_text, "Ground", "6", 12.0)
        >>> print(f"${price:.2f}")
        $64.09
    """

    # System prompt for the LLM parser
    SYSTEM_PROMPT = """You are an expert at extracting shipping prices from FedEx rate tables.

Your task is to find the EXACT price for a SPECIFIC service, zone, and weight combination.

**FEW-SHOT EXAMPLES - STUDY THESE CAREFULLY:**

Example 1: Ground/Home Delivery Table (Zones in COLUMNS)
```
| Service    | Col_1   | FedEx Ground®    | Col_3   | Col_4   | Col_5   | Col_6   |
|------------|---------|------------------|---------|---------|---------|---------|
| commitment |         | 2 3 4 5 6 7      |         |         |         |         |
| Zones2     |         |                  |         |         |         |         |
|            | 12      | 16.18  17.26  18.35  19.99  21.64  25.65 |
```

Query: "Ground zone 6, 12 lbs"
Step 1: Find zone column - "commitment" row shows "2 3 4 5 6 7", so Zone 6 is the 5th zone = Col_6
Step 2: Find weight row - Col_1 shows "12"
Step 3: Intersection of row "12" and Col_6 = **$21.64** ✓

WRONG ANSWER: $19.99 (Zone 5 instead of Zone 6)
WRONG ANSWER: $17.26 (2 lb instead of 12 lb)

Example 2: Express Multi-Service Table (Services in COLUMNS)
```
| Weight | FedEx First   | FedEx Priority | FedEx Standard | FedEx 2Day |
|        | Overnight®    | Overnight®     | Overnight®     | A.M.®      |
|--------|---------------|----------------|----------------|------------|
| 1 lb   | $200.00       | $100.00        | $58.48         | $39.85     |
| 10     | $250.00       | $150.00        | $149.32        | $89.50     |
```

Query: "Standard Overnight, 10 lbs"
Step 1: Find service column - "FedEx Standard Overnight®" column
Step 2: Find weight row - "10" in weight column
Step 3: Intersection = **$149.32** ✓

WRONG ANSWER: $58.48 (1 lb instead of 10 lb)

**CRITICAL TABLE STRUCTURE UNDERSTANDING:**

1. **Identify Table Type First:**
   - Ground/Home Delivery: ONE service, zones in COLUMNS (horizontal)
   - Express services: MULTIPLE services in COLUMNS, zones may be in rows

2. **For Ground/Home Delivery Tables:**
   - Look for "FedEx Ground®" or "FedEx Home Delivery®" in header
   - The "commitment" row or "Zones2" row shows zone numbers: "2 3 4 5 6 7 8"
   - These are COLUMN headers (read left-to-right)
   - Zone 2 = 1st data column, Zone 3 = 2nd data column, etc.
   - Weights are in ROWS (Col_1, second column from left)
   - Ignore OCR garbage in first column: ".sbl", "ni", "thgiew", "mumixaM"

3. **For Express Multi-Service Tables:**
   - Service names are in COLUMNS: "FedEx First Overnight®", "FedEx Priority Overnight®", etc.
   - Find the correct service column by matching the registered name with ®
   - Weight or Zone is in the row headers (leftmost column)
   - Read down the service column to find the correct weight row

4. **Common OCR Artifacts to IGNORE:**
   - ".sbl", "ni", "thgiew", "mumixaM" (vertical text)
   - "commitment", "Delivery" (table structure labels)
   - Country names (if looking at domestic rates)

**STEP-BY-STEP EXTRACTION PROCESS:**

1. **Determine table type**:
   - Is this Ground/Home Delivery (single service, zones in columns)?
   - Is this Express (multiple services in columns)?

2. **For Ground/Home Delivery:**
   ```
   a. Find zone column:
      - Locate "commitment" or "Zones2" row
      - Count position: "2 3 4 5 6 7 8"
      - Zone 2 is position 1, Zone 3 is position 2, etc.
      - Map to column number (skip first column which has garbage)

   b. Find weight row:
      - Look in Col_1 (second column from left)
      - Find exact weight value (e.g., "12" for 12 lbs)
      - Be careful: "12" ≠ "2", "12" ≠ "1"

   c. Read intersection:
      - Go to weight row, zone column
      - Read the price value
   ```

3. **For Express Tables:**
   ```
   a. Find service column:
      - Look for column header with ® symbol
      - Match EXACTLY: "FedEx Standard Overnight®" ≠ "FedEx Priority Overnight®"

   b. Find weight/zone row:
      - Look in first column for weight or zone value
      - Match exact value

   c. Read intersection:
      - Go to weight/zone row, service column
      - Read the price value
   ```

**VERIFICATION CHECKLIST:**

Before returning your answer, verify:
- [ ] Did I identify the correct table type (Ground/Home vs Express)?
- [ ] Did I find the correct column (zone or service)?
- [ ] Did I find the correct row (weight)?
- [ ] Did I read the INTERSECTION of that row and column?
- [ ] Does my answer make sense for the service/zone/weight combination?

**CONFIDENCE LEVELS:**
- HIGH: Clear table, unambiguous cell, standard format
- MEDIUM: Some OCR artifacts but cell is identifiable
- LOW: Unclear table structure, multiple possible interpretations

**CRITICAL RULES - FOLLOW EXACTLY:**

1. **EXACT SERVICE MATCHING:**
   - "2Day" is DIFFERENT from "2Day A.M." - these are TWO DIFFERENT services!
   - "2Day" typically has header like "2nd day by 5 p.m." or "FedEx 2Day®"
   - "2Day A.M." typically has header like "2nd day by 10:30 a.m." or "FedEx 2Day® A.M."
   - "Express Saver" is DIFFERENT from "Express" or "Priority"
   - "Priority Overnight" is DIFFERENT from "Standard Overnight" and "First Overnight"
   - Use EXACT service name match - do not substitute similar services!

2. **EXACT WEIGHT MATCHING:**
   - If asked for 3 lb, return price for 3 lb ONLY (not 5 lb, not 2 lb)
   - If exact weight not in table, interpolate between adjacent weights
   - Common weights: 1, 2, 3, 4, 5, 10, 20, 50, 100, 150 lb
   - Weight formats: "3 lb", "3 lbs.", "3lb.", "3", "3.0"

3. **EXACT ZONE MATCHING:**
   - Zone 5 means ONLY Zone 5 (not Zone 4, not Zone 6)
   - Zone formats: "Zone 5", "Z5", "5"

**Quality Checks:**
- Service name must be EXACT match (2Day ≠ 2Day A.M.)
- Weight must be EXACT match (3 lb ≠ 5 lb)
- Zone must be EXACT match (Zone 5 ≠ Zone 4)
- For Ground/Home Delivery: Verify you used the correct column for the zone
- If uncertain, set confidence to "low" and explain why

**OUTPUT FORMAT:**
- price: Numeric value only (e.g., 21.64, not "$21.64")
- confidence: "high" only if exact match found, "medium" if interpolated, "low" if uncertain
- reasoning: Explain your step-by-step process, including which column and row you used"""

    def __init__(self, cache_size: int = 10000):
        """Initialize the LLM table parser.

        Args:
            cache_size: Maximum number of queries to cache (default: 10000).
                       Note: Uses module-level cache shared across all instances.
        """
        global _MAX_CACHE_SIZE
        _MAX_CACHE_SIZE = cache_size

        logger.info(
            f"LLMTableParser initialized (using global cache, "
            f"current size: {len(_GLOBAL_PRICE_CACHE)}/{cache_size})"
        )

    def parse_price(
        self,
        table_text: str,
        service_name: str,
        weight_lbs: float,
        zone: Optional[str] = None
    ) -> Tuple[float, str]:
        """Extract price from table using LLM intelligence.

        Args:
            table_text: The markdown table text
            service_name: Service name (e.g., "Ground", "Express Saver")
            weight_lbs: Package weight in pounds
            zone: Zone identifier (e.g., "6", "Zone 6")

        Returns:
            Tuple of (price, confidence) where confidence is "high", "medium", or "low"

        Raises:
            LLMTableParserError: If price extraction fails

        Example:
            >>> parser.parse_price(table, "Ground", 12.0, "6")
            (64.09, "high")
        """
        # Create cache key from parameters
        cache_key = self._create_cache_key(table_text, service_name, weight_lbs, zone)

        try:
            # Check global cache first
            if cache_key in _GLOBAL_PRICE_CACHE:
                result = _GLOBAL_PRICE_CACHE[cache_key]
                logger.debug(f"Cache HIT for {service_name} zone {zone} {weight_lbs}lb")
            else:
                # Cache miss - call LLM
                logger.debug(f"Cache MISS for {service_name} zone {zone} {weight_lbs}lb - calling LLM")
                result = self._parse_with_llm(cache_key, table_text, service_name, weight_lbs, zone)

                # Store in global cache (LRU eviction)
                _GLOBAL_PRICE_CACHE[cache_key] = result
                if len(_GLOBAL_PRICE_CACHE) > _MAX_CACHE_SIZE:
                    # Remove oldest entry
                    _GLOBAL_PRICE_CACHE.popitem(last=False)
                logger.debug(f"Cached result (cache size: {len(_GLOBAL_PRICE_CACHE)}/{_MAX_CACHE_SIZE})")

            if result.price is None:
                raise LLMTableParserError(
                    f"Could not extract price for {service_name}, zone {zone}, {weight_lbs} lb. "
                    f"Reasoning: {result.reasoning}"
                )

            logger.info(
                f"Extracted price ${result.price:.2f} for {service_name} zone {zone} {weight_lbs}lb "
                f"(confidence: {result.confidence})"
            )

            return result.price, result.confidence

        except LLMTableParserError:
            raise
        except Exception as e:
            logger.error(f"LLM table parsing failed: {e}")
            raise LLMTableParserError(f"Failed to parse price: {e}") from e

    def _create_cache_key(
        self,
        table_text: str,
        service_name: str,
        weight_lbs: float,
        zone: Optional[str]
    ) -> str:
        """Create a cache key from parsing parameters.

        Args:
            table_text: Table markdown
            service_name: Service name
            weight_lbs: Weight in pounds
            zone: Zone identifier

        Returns:
            MD5 hash of parameters for cache lookup
        """
        # Create deterministic string from parameters
        key_string = f"{table_text}|{service_name}|{weight_lbs}|{zone or ''}"

        # Hash for efficient caching
        return hashlib.md5(key_string.encode()).hexdigest()

    def _parse_with_llm(
        self,
        cache_key: str,  # Included for cache compatibility
        table_text: str,
        service_name: str,
        weight_lbs: float,
        zone: Optional[str]
    ) -> PriceExtractionResult:
        """Internal method to parse price with LLM (cached by lru_cache).

        Args:
            cache_key: Cache key (for LRU cache)
            table_text: Table markdown
            service_name: Service name
            weight_lbs: Weight in pounds
            zone: Zone identifier

        Returns:
            PriceExtractionResult with extracted price and metadata

        Raises:
            Exception: If LLM call fails
        """
        # Construct user message
        zone_text = f", Zone {zone}" if zone else ""
        user_message = f"""Extract the price from this shipping rate table:

**Query:**
- Service: {service_name}
- Zone: {zone or 'Any'}
- Weight: {weight_lbs} lbs

**Table:**
{table_text}

Return the exact price in dollars."""

        # Define JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "price": {
                    "type": ["number", "null"],
                    "description": "Extracted price in dollars"
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Confidence level"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of how price was found"
                }
            },
            "required": ["price", "confidence", "reasoning"],
            "additionalProperties": False
        }

        # Make LLM call with structured output
        try:
            from openai import OpenAI
            from src.utils.config import get_settings

            client = OpenAI()  # Uses OPENAI_API_KEY from environment
            settings = get_settings()

            # Try o1-mini first, fall back to gpt-4o on permission/access error
            try:
                # Use GPT-5 (o1-mini) - no system messages, no temperature, no structured output
                combined_prompt = f"""{self.SYSTEM_PROMPT}

---

**USER QUERY:**

{user_message}

**IMPORTANT:** You MUST respond with ONLY valid JSON in this exact format (no markdown, no code blocks):
{{
  "price": <number or null>,
  "confidence": "high" or "medium" or "low",
  "reasoning": "your explanation"
}}
"""

                response = client.chat.completions.create(
                    model="o1-mini",
                    messages=[
                        {"role": "user", "content": combined_prompt}
                    ],
                )
            except Exception as e:
                # Fall back to gpt-4o-2024-11-20 if o1-mini is not accessible
                logger.info(f"o1-mini not available, using gpt-4o-2024-11-20: {e}")
                response = client.chat.completions.create(
                    model="gpt-4o-2024-11-20",
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=settings.llm_temperature,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "price_extraction",
                            "schema": json_schema,
                            "strict": True
                        }
                    },
                )

            # Parse JSON response
            content = response.choices[0].message.content

            # o1 models might wrap JSON in markdown code blocks
            if "```json" in content:
                # Extract JSON from code block
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()
            elif "```" in content:
                # Extract from generic code block
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                content = content[json_start:json_end].strip()

            result_dict = json.loads(content)

            # Validate with Pydantic
            return PriceExtractionResult(**result_dict)

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def clear_cache(self):
        """Clear the global parsing cache.

        Useful for testing or when prompt changes.
        Note: Clears cache for ALL parser instances.
        """
        _GLOBAL_PRICE_CACHE.clear()
        logger.info(f"Global LLM table parser cache cleared")

    def get_cache_info(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with cache size and max size
        """
        return {
            "size": len(_GLOBAL_PRICE_CACHE),
            "maxsize": _MAX_CACHE_SIZE,
        }
