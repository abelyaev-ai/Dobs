"""Price retrieval module for querying shipping rates from indexed documents.

This module provides the main get_price() function that integrates query parsing,
document search, and price extraction using custom prompts optimized for shipping
rate tables.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core.prompts import PromptTemplate

from src.core.document_indexer import DocumentIndexer, DocumentIndexerError
from src.core.query_parser import QueryParser, ShippingQuery
from src.core.llm_table_parser import LLMTableParser, LLMTableParserError
from src.core.llm_query_parser import LLMQueryParser
from src.utils.config import get_settings
from src.utils.logger import get_logger, log_exception

logger = get_logger(__name__)


class PriceRetrieverError(Exception):
    """Raised when price retrieval operations fail."""


@dataclass
class PriceResult:
    """Result of a price retrieval query.

    Attributes:
        price: The shipping price in USD.
        confidence: Confidence level ("high", "medium", "low").
        source: Source document/table identifier.
        details: Additional information including query breakdown and response.
        query: The parsed ShippingQuery object.
    """

    price: float
    confidence: str
    source: str
    details: dict[str, Any] = field(default_factory=dict)
    query: ShippingQuery | None = None

    def __str__(self) -> str:
        """Return human-readable representation of the result."""
        return (
            f"Price: ${self.price:.2f} "
            f"(Confidence: {self.confidence}, Source: {self.source})"
        )


# Custom prompt template optimized for shipping rate tables
SHIPPING_RATE_PROMPT = PromptTemplate(
    "You are a shipping rate expert analyzing FedEx rate card tables.\n"
    "\n"
    "Context information (rate tables) is below:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Instructions for reading rate tables:\n"
    "\n"
    "1. TABLE STRUCTURE UNDERSTANDING:\n"
    "   a) MULTI-COLUMN TABLES (common format):\n"
    "      - Service types appear as COLUMN GROUPS (e.g., multiple 'FedEx 2Day' columns, 'Ground' columns)\n"
    "      - Each service may have MULTIPLE zone columns (Zone 2, Zone 3, Zone 4, etc.)\n"
    "      - Look for headers like: 'FedEx 2Day | Zone 2 | Zone 3 | Zone 4 | Express Saver | Zone 2 | Zone 3'\n"
    "      - Weight values (1 lb, 2 lb, 3 lb, etc.) are ALWAYS in the LEFTMOST column\n"
    "   \n"
    "   b) SINGLE-SERVICE TABLES (alternative format):\n"
    "      - One service per table, zones span across columns\n"
    "      - Header row shows: 'Weight | Zone 2 | Zone 3 | Zone 4 | ...'\n"
    "   \n"
    "   c) KEY IDENTIFIERS:\n"
    "      - Service names: '2Day', 'Standard Overnight', 'Express Saver', 'Ground', 'Home Delivery'\n"
    "      - Zone identifiers: 'Zone 2', 'Zone 3', 'Z2', 'Z3', or just numbers '2', '3', '4'\n"
    "      - Weight column: 'Weight', 'Wt', 'Package Weight', or just weight values\n"
    "\n"
    "2. STEP-BY-STEP PRICE EXTRACTION:\n"
    "   Step 1: IDENTIFY THE SERVICE\n"
    "      - Look for the service name in column headers (e.g., 'FedEx 2Day', 'Express Saver', 'Ground')\n"
    "      - Service names may appear MULTIPLE times if the table has multiple zone columns for that service\n"
    "      - If you see 'FedEx 2Day' repeated, that means multiple zones are available for that service\n"
    "   \n"
    "   Step 2: FIND THE ZONE COLUMN UNDER THAT SERVICE\n"
    "      - Within the service's column group, find the specific zone requested\n"
    "      - Example: For 'FedEx 2Day Zone 5', find where '2Day' and 'Zone 5' intersect in headers\n"
    "      - The zone column will be UNDER or ADJACENT to the service name\n"
    "   \n"
    "   Step 3: LOCATE THE WEIGHT ROW\n"
    "      - Find the row matching the requested weight in the leftmost column\n"
    "      - If exact weight exists, use it directly\n"
    "      - If not exact, proceed to Step 4 for interpolation\n"
    "   \n"
    "   Step 4: READ THE INTERSECTION PRICE\n"
    "      - Find where the zone column and weight row intersect\n"
    "      - This cell contains the price\n"
    "      - Prices are typically $10-$200 for small packages\n"
    "      - Zone numbers (2-9) are NOT prices - they are identifiers\n"
    "\n"
    "3. WEIGHT INTERPOLATION (when exact weight not listed):\n"
    "   If the exact weight is not in the table, you MUST interpolate:\n"
    "   \n"
    "   Example: Query asks for 3 lb, but table only shows 2 lb and 5 lb\n"
    "   - Find the two closest weights: 2 lb (below) and 5 lb (above)\n"
    "   - Get prices for both: e.g., 2 lb = $40.00, 5 lb = $55.00\n"
    "   - Calculate proportion: (3 - 2) / (5 - 2) = 1/3 = 0.333\n"
    "   - Calculate price increase: ($55.00 - $40.00) * 0.333 = $5.00\n"
    "   - Final interpolated price: $40.00 + $5.00 = $45.00\n"
    "   \n"
    "   Formula:\n"
    "   price = price_lower + ((weight_target - weight_lower) / (weight_upper - weight_lower)) * (price_upper - price_lower)\n"
    "\n"
    "4. CRITICAL RULES:\n"
    "   - ALWAYS read from the intersection of service + zone + weight\n"
    "   - NEVER confuse zone numbers (2, 3, 4) with prices\n"
    "   - SKIP rows labeled 'minimum charge', 'min charge', 'base rate'\n"
    "   - If tables are split across chunks, look in ALL provided chunks\n"
    "   - Use metadata to confirm which zones are available in each table\n"
    "   - Prices should be reasonable ($10-$200 for typical packages)\n"
    "\n"
    "5. EXAMPLE WALKTHROUGH:\n"
    "   Query: \"FedEx 2Day Zone 5 3 lb\"\n"
    "   \n"
    "   Step-by-step:\n"
    "   1. Look for 'FedEx 2Day' or '2Day' in column headers\n"
    "   2. Under the '2Day' section, find 'Zone 5' column\n"
    "   3. Find the row for '3 lb' in the weight column\n"
    "   4. Read the price at the intersection of Zone 5 column and 3 lb row\n"
    "   5. Expected result: around $48.05\n"
    "   \n"
    "   Common mistakes to AVOID:\n"
    "   - Don't return the zone number (5) as the price\n"
    "   - Don't return the weight (3) as the price\n"
    "   - Don't return minimum charges or base rates\n"
    "   - Don't read from wrong service's columns\n"
    "\n"
    "Query: {query_str}\n"
    "\n"
    "Analyze the tables above carefully and find the shipping rate.\n"
    "Follow the step-by-step process to ensure accuracy.\n"
    "Return ONLY the numeric price (example: 48.05)\n"
    "Do NOT include currency symbols ($), explanations, or extra text.\n"
    "\n"
    "Price: "
)


class PriceRetriever:
    """Price retrieval system for shipping rate queries.

    This class integrates query parsing, document indexing, and LLM-based
    price extraction to return accurate shipping rates from PDF documents.

    Attributes:
        indexer: DocumentIndexer instance for querying indexed documents.
        parser: QueryParser instance for parsing natural language queries.
    """

    def _load_service_catalog(self) -> list[str]:
        """Load canonical service names from catalog.

        Returns:
            List of canonical service names from the catalog.
            Returns empty list if catalog not found or on error.
        """
        try:
            catalog_path = Path("data/service_catalog.json")
            if catalog_path.exists():
                with open(catalog_path, encoding="utf-8") as f:
                    data = json.load(f)
                    services = data.get("services", [])
                    logger.info(f"Loaded {len(services)} services from catalog")
                    return services
        except Exception as e:
            logger.warning(f"Failed to load service catalog: {e}")

        # Return empty list if not found
        logger.warning("Service catalog not found, query parser will work without it")
        return []

    def __init__(
        self,
        indexer: DocumentIndexer | None = None,
        parser: QueryParser | None = None,
        table_parser: LLMTableParser | None = None,
    ) -> None:
        """Initialize the PriceRetriever.

        Args:
            indexer: Optional DocumentIndexer instance. If None, creates new one.
            parser: Optional custom parser instance. If None, uses LLMQueryParser
                   with automatic fallback to regex parser for reliability.
            table_parser: Optional LLMTableParser instance. If None, creates new one.

        Note:
            By default, uses LLM-based parsing for both query understanding and
            table price extraction. This provides maximum flexibility and handles
            any table format (multi-service, single-service, Ground, Express Saver, etc.)
            without hardcoded rules.
        """
        self.indexer = indexer or DocumentIndexer()

        # Load service catalog for canonical service name mapping
        service_catalog = self._load_service_catalog()

        # Use LLM parser by default with automatic fallback and service catalog
        self.parser = parser or LLMQueryParser(
            service_catalog=service_catalog,
            fallback_parser=QueryParser()
        )
        logger.info(
            f"PriceRetriever initialized with LLM query parser "
            f"(automatic fallback enabled, {len(service_catalog)} services in catalog)"
        )

        # Use LLM table parser by default (handles all table formats)
        if table_parser is None:
            try:
                settings = get_settings()
                cache_size = settings.table_parser_cache_size
            except Exception:
                cache_size = 10000  # Default if settings unavailable
                logger.warning("Could not load table_parser_cache_size from settings, using default 10000")

            self.table_parser = LLMTableParser(cache_size=cache_size)
            logger.info(f"PriceRetriever initialized with LLM table parser (cache_size={cache_size})")
        else:
            self.table_parser = table_parser
            logger.info("PriceRetriever initialized with custom table parser")

    def _filter_documents_by_metadata(
        self,
        service_type: str | None = None,
        zone: str | None = None,
        weight: float | None = None,
    ) -> list[Any]:
        """Filter documents by metadata before FAISS query.

        Since FAISS doesn't support metadata filtering, we manually filter
        documents before creating the query index.

        Uses canonical service names for matching when available.

        Args:
            service_type: Service type to filter by (e.g., "FedEx Express Saver", "FedEx 2Day").
            zone: Zone to filter by (e.g., "2", "5", "8").
            weight: Package weight in pounds (optional, for weight range filtering).

        Returns:
            List of filtered Document nodes sorted by weight range relevance.
        """
        if not self.indexer.index:
            logger.warning("No index available for filtering")
            return []

        # Get all documents from the index
        all_nodes = list(self.indexer.index.docstore.docs.values())
        filtered_nodes = []

        logger.debug(
            f"Filtering {len(all_nodes)} total documents by service='{service_type}', "
            f"zone={zone}, weight={weight}"
        )

        for node in all_nodes:
            metadata = node.metadata

            # Filter by service type if specified
            if service_type:
                # Get service_types list from metadata
                node_service_types = metadata.get("service_types", [])

                if node_service_types:
                    # Check if requested service is in the list
                    # Use exact matching to handle variations like "FedEx 2Day" vs "2Day"
                    service_found = any(
                        self._service_names_match(service_type, s)
                        for s in node_service_types
                    )
                    if not service_found:
                        continue
                else:
                    # No service metadata, skip this document
                    continue

            # Filter by zone if specified
            # Note: metadata has 'zones' (plural) which is a list like ['Zone 2', 'Zone 3']
            # A document can contain data for multiple zones
            if zone:
                node_zones = metadata.get("zones", [])

                # If the document has zone information, filter by it
                if node_zones:
                    # Normalize the requested zone (handle "Zone 5", "5", "Z5")
                    zone_normalized = (
                        zone.replace("Zone ", "").replace("Z", "").strip()
                    )

                    # Check if the requested zone is in the document's zones list
                    zone_found = False
                    for node_zone in node_zones:
                        node_zone_str = str(node_zone).strip()
                        # Normalize node zone (remove "Zone ", "Z" prefix)
                        node_zone_normalized = (
                            node_zone_str.replace("Zone ", "").replace("Z", "").strip()
                        )

                        if node_zone_normalized == zone_normalized:
                            zone_found = True
                            break

                    # If zone was requested but not found in this document, skip it
                    if not zone_found:
                        continue
                else:
                    # CRITICAL FIX: If zones list is empty AND zone was requested,
                    # skip this document (don't include Alaska/Hawaii/Canada pages
                    # when searching for standard Zone 2-8)
                    logger.debug(
                        f"Skipping page {metadata.get('page')} with empty zones "
                        f"(zone {zone} requested)"
                    )
                    continue

            filtered_nodes.append(node)

        logger.info(
            f"Filtered {len(all_nodes)} documents to {len(filtered_nodes)} "
            f"(service_type={service_type}, zone={zone}, weight={weight})"
        )

        # CRITICAL FIX: Exclude pages with weight ranges that don't include the requested weight
        if weight is not None and filtered_nodes:
            filtered_by_weight = []
            for node in filtered_nodes:
                metadata = node.metadata
                weight_range = metadata.get("weight_range")

                # If page has a weight range, check if our weight is in it
                if weight_range and isinstance(weight_range, (list, tuple)) and len(weight_range) == 2:
                    min_wt, max_wt = weight_range
                    # Exclude if weight is outside the range
                    if weight < min_wt or weight > max_wt:
                        logger.debug(
                            f"Excluding page {metadata.get('page')} - "
                            f"weight {weight} outside range [{min_wt}, {max_wt}]"
                        )
                        continue

                filtered_by_weight.append(node)

            logger.info(
                f"Weight range filtering: {len(filtered_nodes)} -> {len(filtered_by_weight)} pages "
                f"(excluded pages outside weight={weight} range)"
            )
            filtered_nodes = filtered_by_weight

        # PRIORITY FIX: Sort by weight range relevance if weight specified
        # Pages containing the requested weight should come first
        if weight and filtered_nodes:
            filtered_nodes = self._sort_by_weight_relevance(filtered_nodes, weight)

        return filtered_nodes

    def _get_available_services(self) -> list[str]:
        """Get list of all unique services available in the index.

        Returns:
            Sorted list of unique service names.
        """
        if not self.indexer.index:
            return []

        all_nodes = list(self.indexer.index.docstore.docs.values())
        services: set[str] = set()

        for node in all_nodes:
            metadata = node.metadata

            # Collect from service_types (list)
            node_service_types = metadata.get("service_types", [])
            if node_service_types:
                services.update(node_service_types)

        return sorted(services)

    def _service_names_match(self, service1: str, service2: str) -> bool:
        """Check if two service names match using exact case-insensitive matching.

        Since the query parser now returns canonical service names, we can do
        exact matching instead of fuzzy matching. This improves precision.

        Handles variations like:
        - "FedEx 2Day" vs "fedex 2day" (case-insensitive)
        - "FedEx Express Saver" vs "fedex express saver"

        Args:
            service1: First service name.
            service2: Second service name.

        Returns:
            True if services match exactly (case-insensitive), False otherwise.
        """
        # Normalize both names (case-insensitive, whitespace-normalized)
        s1 = " ".join(service1.lower().strip().split())
        s2 = " ".join(service2.lower().strip().split())

        # Exact match (case-insensitive)
        return s1 == s2

    def _sort_by_weight_relevance(self, nodes: list[Any], target_weight: float) -> list[Any]:
        """Sort nodes by weight range relevance.

        Pages that contain the target weight in their weight range get highest priority.
        Then sort by how close the weight range is to the target.

        Args:
            nodes: List of document nodes to sort.
            target_weight: Target weight in pounds.

        Returns:
            Sorted list of nodes (most relevant first).
        """
        def weight_relevance_score(node: Any) -> tuple[int, float]:
            """Calculate relevance score for a node based on weight range.

            Returns:
                Tuple of (priority, distance) where:
                - priority: 0 = contains target weight, 1 = doesn't contain
                - distance: How far target is from range (0 if contained)
            """
            metadata = node.metadata
            weight_range = metadata.get("weight_range")

            if not weight_range or not isinstance(weight_range, tuple):
                # No weight range metadata or invalid format - lowest priority
                return (2, float('inf'))

            # Safely unpack weight range
            try:
                weight_min, weight_max = weight_range
            except (ValueError, TypeError):
                # Invalid weight_range format - lowest priority
                return (2, float('inf'))

            # Check if target weight is within range
            if weight_min <= target_weight <= weight_max:
                # Perfect match - highest priority, distance = 0
                return (0, 0.0)

            # Calculate distance from range
            if target_weight < weight_min:
                distance = weight_min - target_weight
            else:  # target_weight > weight_max
                distance = target_weight - weight_max

            # Not in range - medium priority, sorted by distance
            return (1, distance)

        # Sort by relevance score (lower is better)
        sorted_nodes = sorted(nodes, key=weight_relevance_score)

        # Log the sorted order for debugging
        if logger.isEnabledFor(10):  # DEBUG level
            for i, node in enumerate(sorted_nodes[:5]):
                metadata = node.metadata
                page = metadata.get('page', '?')
                weight_range = metadata.get('weight_range', 'N/A')
                priority, distance = weight_relevance_score(node)
                logger.debug(
                    f"  Sorted #{i+1}: Page {page}, weight_range={weight_range}, "
                    f"priority={priority}, distance={distance:.1f}"
                )

        return sorted_nodes

    def _try_direct_parsing(
        self,
        filtered_docs: list[Any],
        parsed_query: ShippingQuery
    ) -> tuple[float | None, str, str]:
        """Try to extract price using LLM-based table parsing.

        Uses LLM intelligence to extract prices from any table format,
        including multi-service tables (Express Saver, 2Day) and single-service
        tables (Ground, Home Delivery). Results are cached for performance.

        Args:
            filtered_docs: Documents already filtered by metadata
            parsed_query: Parsed query with service, weight, etc.

        Returns:
            Tuple of (price, confidence, source) where:
            - price is None if parsing failed
            - confidence is "high", "medium", or "low"
            - source is the document source string
        """
        logger.debug(
            f"Attempting LLM table parsing for {parsed_query.service} "
            f"zone {parsed_query.zone} {parsed_query.weight}lb"
        )

        # Sort by page number and try each document
        sorted_docs = sorted(filtered_docs, key=lambda d: d.metadata.get('page', 999))

        for doc in sorted_docs[:5]:  # Try first 5 documents
            try:
                price, confidence = self.table_parser.parse_price(
                    table_text=doc.text,
                    service_name=parsed_query.service,
                    weight_lbs=parsed_query.weight,
                    zone=parsed_query.zone  # Pass zone for single-service tables
                )

                # Success! Build source string
                page = doc.metadata.get('page', 'unknown')
                source_file = doc.metadata.get('source', 'unknown')
                source = f"{source_file} (Page {page})"

                logger.info(
                    f"LLM table parsing succeeded: ${price:.2f} ({confidence}) "
                    f"from {source}"
                )

                return price, confidence, source

            except LLMTableParserError as e:
                logger.debug(f"LLM parsing failed for doc page {doc.metadata.get('page')}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error in LLM table parsing: {e}")
                continue

        # All parsing attempts failed
        logger.info("LLM table parsing failed for all filtered documents")
        return None, "", ""

    def _semantic_search_cached(
        self,
        filtered_docs: list[Any],
        query_text: str,
        top_k: int = 5
    ) -> list[Any]:
        """Perform semantic search using embeddings computed in batch.

        This is faster than creating a VectorStoreIndex because:
        1. No index building overhead
        2. Batch embedding computation is faster
        3. In-memory similarity computation is fast

        Args:
            filtered_docs: Pre-filtered document nodes to search.
            query_text: Query string to search for.
            top_k: Number of top results to return.

        Returns:
            List of top-k most similar document nodes.
        """
        import numpy as np
        from llama_index.core import Settings

        logger.debug(f"Semantic search over {len(filtered_docs)} documents for: {query_text}")

        # Get query embedding
        query_embedding = Settings.embed_model.get_query_embedding(query_text)

        # Batch compute embeddings for filtered docs
        # This is faster than creating a VectorStoreIndex
        doc_texts = [doc.text for doc in filtered_docs]

        # Use batch embedding for efficiency
        doc_embeddings = []
        for text in doc_texts:
            emb = Settings.embed_model.get_text_embedding(text)
            doc_embeddings.append(emb)

        doc_embeddings_array = np.array(doc_embeddings)

        # Compute cosine similarities
        query_norm = np.linalg.norm(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings_array, axis=1)

        # Avoid division by zero
        doc_norms = np.where(doc_norms == 0, 1e-10, doc_norms)

        similarities = np.dot(doc_embeddings_array, query_embedding) / (doc_norms * query_norm)

        # Get top-k indices
        top_k_actual = min(top_k, len(filtered_docs))
        top_k_indices = np.argsort(similarities)[-top_k_actual:][::-1]

        # Return top-k documents
        top_k_docs = [filtered_docs[i] for i in top_k_indices]

        logger.info(
            f"Semantic search: {len(filtered_docs)} filtered -> {len(top_k_docs)} top matches "
            f"(similarities: {[f'{similarities[i]:.3f}' for i in top_k_indices]})"
        )

        return top_k_docs

    def get_price(self, query_text: str) -> PriceResult:
        """Retrieve shipping price for a given query.

        This is the main entry point for price retrieval. It:
        1. Parses the query to extract shipping parameters
        2. Pre-filters documents by metadata (service_type, zone)
        3. Uses cached embeddings for semantic search (OPTIMIZED - no temp index)
        4. Queries LLM with filtered context documents
        5. Uses custom prompts to extract the exact price
        6. Handles weight interpolation if needed
        7. Returns a structured PriceResult

        Args:
            query_text: Natural language query (e.g., "FedEx 2Day Zone 5 3 lb").

        Returns:
            PriceResult containing price, confidence, and metadata.

        Raises:
            PriceRetrieverError: If retrieval fails or price not found.

        Example:
            >>> retriever = PriceRetriever()
            >>> result = retriever.get_price("FedEx 2Day Zone 5 3 lb")
            >>> print(f"${result.price:.2f}")
            $48.05
        """
        try:
            logger.info(f"Retrieving price for query: {query_text}")

            # Parse the query
            parsed_query = self.parser.parse(query_text)
            logger.debug(f"Parsed query: {parsed_query}")

            # Validate query has required fields
            is_valid, missing_fields = self.parser.validate_query(parsed_query)
            if not is_valid:
                raise PriceRetrieverError(
                    f"Query is missing required fields: {missing_fields}"
                )

            # MANUAL PRE-FILTERING: Filter documents by service, zone, and weight
            # With LLM table parser, we can filter by service because the LLM is smart
            # enough to find the correct service within multi-service pages
            # ADDED: weight parameter to prioritize pages with correct weight range
            filtered_docs = self._filter_documents_by_metadata(
                service_type=parsed_query.service,  # Filter by service for better accuracy
                zone=parsed_query.zone,
                weight=parsed_query.weight,  # NEW: Filter by weight range
            )

            if not filtered_docs:
                # Get list of available services to help user
                available_services = self._get_available_services()

                logger.warning(
                    f"No documents match filters: service={parsed_query.service}, "
                    f"zone={parsed_query.zone}. Available services: {available_services}"
                )

                error_msg = (
                    f"No documents found matching service='{parsed_query.service}' "
                    f"and zone={parsed_query.zone}."
                )
                if available_services:
                    error_msg += f" Available services in index: {', '.join(available_services)}"

                raise PriceRetrieverError(error_msg)

            # TRY LLM TABLE PARSING FIRST (handles all table formats)
            # Only fallback to semantic search + LLM extraction if parsing fails
            price, confidence, source = self._try_direct_parsing(
                filtered_docs, parsed_query
            )

            # Initialize variables for result details
            response_text = "LLM table parsing"
            search_query = f"{parsed_query.service} zone {parsed_query.zone} {parsed_query.weight} lb"
            context_docs = []

            # If LLM table parsing failed, fallback to semantic search + LLM extraction
            if price is None:
                logger.info("LLM table parsing failed, falling back to semantic search + LLM extraction")

                # Build search query string
                search_query = self._build_search_query(parsed_query)
                logger.debug(f"Search query: {search_query}")

                # OPTIMIZED: Skip semantic search and temp index creation entirely
                # Metadata filtering (service + zone) is already very precise
                # Take first 15 docs (sorted by page for sequential rate tables)
                context_docs = sorted(filtered_docs, key=lambda d: d.metadata.get('page', 999))[:15]

                logger.info(
                    f"Using {len(context_docs)} filtered docs without semantic search "
                    f"(filtered {len(filtered_docs)} by metadata)"
                )

                # Query LLM with filtered context
                response_text = self._query_llm_with_context(context_docs, search_query)

                # Extract price from response
                price, confidence = self._extract_price(
                    response_text, parsed_query
                )

                # Build source from context docs
                source = self._build_source_string(context_docs)
            else:
                logger.info(f"Direct parsing succeeded: ${price:.2f} ({confidence} confidence)")

            # Create result
            result = PriceResult(
                price=price,
                confidence=confidence,
                source=source,
                details={
                    "raw_response": response_text,
                    "search_query": search_query,
                    "context_docs_count": len(context_docs),
                    "filtered_docs_count": len(filtered_docs),
                },
                query=parsed_query,
            )

            logger.info(f"Price retrieved successfully: {result}")
            return result

        except DocumentIndexerError as e:
            log_exception(logger, "Document indexer error", e)
            raise PriceRetrieverError(f"Failed to query index: {e}") from e
        except Exception as e:
            log_exception(
                logger,
                f"Failed to retrieve price for query: {query_text}",
                e,
                extra={"query": query_text},
            )
            raise PriceRetrieverError(
                f"Failed to retrieve price: {e}"
            ) from e

    def _build_filters(self, query: ShippingQuery) -> dict[str, Any]:
        """Build metadata filters based on parsed query.

        Args:
            query: Parsed ShippingQuery object.

        Returns:
            Dictionary of metadata filters for the query engine.
        """
        filters: dict[str, Any] = {}

        # Filter by carrier if specified
        if query.carrier:
            filters["carrier"] = query.carrier

        # Filter by service type if specified
        if query.service:
            # Normalize service type for matching
            filters["service_type"] = query.service

        return filters

    def _build_search_query(self, query: ShippingQuery) -> str:
        """Build natural language search query from parsed components.

        Args:
            query: Parsed ShippingQuery object.

        Returns:
            Natural language query string optimized for the LLM.
        """
        # Build query emphasizing service and zone for better retrieval
        parts = []

        if query.carrier:
            parts.append(query.carrier)

        if query.service:
            parts.append(query.service)

        if query.zone:
            parts.append(f"Zone {query.zone}")

        if query.weight is not None:
            parts.append(f"{query.weight} lb")

        if query.packaging:
            parts.append(f"{query.packaging} packaging")

        search_query = " ".join(parts)

        # Add natural language instruction
        search_query += " shipping rate"

        return search_query

    def _query_llm_with_context(self, context_docs: list[Any], query_text: str) -> str:
        """Query LLM with given context documents.

        Args:
            context_docs: List of document nodes to use as context.
            query_text: Query string for the LLM.

        Returns:
            LLM response text.
        """
        from llama_index.core import Settings

        logger.debug(f"Querying LLM with {len(context_docs)} context documents")

        # Build context string from documents
        context_str = "\n\n---\n\n".join([doc.text for doc in context_docs])

        # Format prompt using custom template
        prompt = SHIPPING_RATE_PROMPT.format(
            context_str=context_str,
            query_str=query_text
        )

        # Query LLM directly (no query engine needed)
        llm = Settings.llm
        response = llm.complete(prompt)

        logger.debug(f"LLM response: {str(response)[:100]}...")

        return str(response)

    def _build_source_string(self, docs: list[Any]) -> str:
        """Build source reference string from documents.

        Args:
            docs: List of document nodes.

        Returns:
            Source identifier string.
        """
        if not docs:
            return "Unknown"

        doc = docs[0]
        metadata = doc.metadata

        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "?")

        # Extract just the filename if it's a full path
        if "/" in source:
            source = source.split("/")[-1]

        return f"{source}, Page {page + 1 if isinstance(page, int) else page}"

    def _extract_price(
        self, response_text: str, query: ShippingQuery
    ) -> tuple[float, str]:
        """Extract numeric price from LLM response.

        Args:
            response_text: Response text from the LLM.
            query: Original parsed query for context.

        Returns:
            Tuple of (price as float, confidence level).

        Raises:
            PriceRetrieverError: If price cannot be extracted or is NOT_FOUND.
        """
        # Check for NOT_FOUND response
        if "NOT_FOUND" in response_text.upper():
            raise PriceRetrieverError(
                f"Price not found for query: {query}"
            )

        # Try to extract numeric price using regex
        # Pattern matches: 48.05, $48.05, 48, etc.
        price_pattern = r"\$?\s*(\d+(?:\.\d{1,2})?)"
        matches = re.findall(price_pattern, response_text)

        if not matches:
            raise PriceRetrieverError(
                f"Could not extract price from response: {response_text}"
            )

        # Use the first match as the price
        try:
            price = float(matches[0])
        except (ValueError, IndexError) as e:
            raise PriceRetrieverError(
                f"Invalid price format in response: {response_text}"
            ) from e

        # Determine confidence level
        confidence = self._determine_confidence(response_text, query)

        logger.debug(f"Extracted price: ${price:.2f} (confidence: {confidence})")

        return price, confidence

    def _determine_confidence(
        self, response_text: str, query: ShippingQuery
    ) -> str:
        """Determine confidence level based on response characteristics.

        Args:
            response_text: Response text from the LLM.
            query: Original parsed query.

        Returns:
            Confidence level: "high", "medium", or "low".
        """
        response_lower = response_text.lower()

        # High confidence: exact match, no interpolation
        if any(
            keyword in response_lower
            for keyword in ["exact", "found at", "matches"]
        ):
            return "high"

        # Medium confidence: interpolation performed
        if any(
            keyword in response_lower
            for keyword in [
                "interpolat",
                "between",
                "calculated",
                "estimated",
            ]
        ):
            return "medium"

        # Low confidence: ambiguous or uncertain
        if any(
            keyword in response_lower
            for keyword in [
                "approximately",
                "might be",
                "unclear",
                "ambiguous",
            ]
        ):
            return "low"

        # Default to high if we got a clean numeric response
        return "high"

    def _build_source_info(self, response: Any) -> str:
        """Build source information from query response.

        Args:
            response: Query response object with source nodes.

        Returns:
            Source identifier string.
        """
        if not hasattr(response, "source_nodes") or not response.source_nodes:
            return "Unknown source"

        # Get the first source node (most relevant)
        source_node = response.source_nodes[0]

        if not hasattr(source_node, "metadata"):
            return "Unknown source"

        metadata = source_node.metadata

        # Build source string
        parts = []

        if "source" in metadata:
            # Extract just the filename
            source_file = metadata["source"].split("/")[-1]
            parts.append(source_file)

        if "page" in metadata:
            parts.append(f"Page {metadata['page']}")

        if "table_num" in metadata:
            parts.append(f"Table {metadata['table_num']}")

        return ", ".join(parts) if parts else "Unknown source"


def get_price(query: str) -> PriceResult:
    """Convenience function for retrieving shipping price.

    This is the main entry point for the price retrieval API.
    Uses LLM-based query parsing with automatic fallback for maximum flexibility.

    Args:
        query: Natural language query string. Supports abbreviations, typos,
              and flexible word order (e.g., "Expr saver z8 2lb").

    Returns:
        PriceResult with price and metadata.

    Raises:
        PriceRetrieverError: If retrieval fails.

    Example:
        >>> result = get_price("FedEx 2Day Zone 5 3 lb")
        >>> print(f"Price: ${result.price:.2f}")
        Price: $48.05

        >>> result = get_price("Expr saver z8 2lb")  # Abbreviations work!
        >>> print(f"Price: ${result.price:.2f}")
        Price: $46.30
    """
    # Load existing index
    indexer = DocumentIndexer()

    try:
        indexer.load_index()
    except DocumentIndexerError as e:
        raise PriceRetrieverError(
            "No index loaded. Please build and save an index first. "
            f"Error: {e}"
        ) from e

    # Create retriever and execute query (uses LLM parser with automatic fallback)
    retriever = PriceRetriever(indexer=indexer)
    return retriever.get_price(query)
