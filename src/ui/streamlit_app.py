"""Streamlit UI for shipping price search tool.

This module provides a comprehensive web interface for the Shipping Price Search Tool.
Features include natural language query search, document management, and result visualization.
"""

from __future__ import annotations

import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from src import __version__
from src.core.document_indexer import DocumentIndexer, DocumentIndexerError
from src.core.price_retriever import PriceResult, PriceRetrieverError, get_price
from src.utils.config import ConfigError, get_settings, load_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Shipping Price Search Tool",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .big-price {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        margin: 20px 0;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .source-info {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .query-breakdown {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .example-query {
        cursor: pointer;
        padding: 8px 12px;
        margin: 5px;
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        display: inline-block;
    }
    .example-query:hover {
        background-color: #e9ecef;
        border-color: #adb5bd;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Example queries from Tasks.md
EXAMPLE_QUERIES = [
    ("FedEx 2Day, Zone 5, 3 lb", "39.85"),
    ("Standard Overnight, z2, 10 lbs, other packaging", "58.48"),
    ("Express Saver Z8 1 lb", "39.86"),
    ("Ground Z6 12 lb", "21.64"),
    ("Home Delivery zone 3 5 lb", "14.76"),
]


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "indexer" not in st.session_state:
        st.session_state.indexer = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "config_loaded" not in st.session_state:
        st.session_state.config_loaded = False
    if "index_loaded" not in st.session_state:
        st.session_state.index_loaded = False


def load_config() -> bool:
    """Load application configuration.

    Returns:
        True if config loaded successfully, False otherwise.
    """
    try:
        load_settings()
        st.session_state.config_loaded = True
        return True
    except ConfigError as e:
        st.error(
            f"Configuration Error: {e}\n\n"
            "Please ensure you have a .env file with OPENAI_API_KEY set."
        )
        return False


def init_indexer() -> bool:
    """Initialize the DocumentIndexer and load existing index.

    Returns:
        True if indexer initialized successfully, False otherwise.
    """
    try:
        if st.session_state.indexer is None:
            config = get_settings()
            st.session_state.indexer = DocumentIndexer(
                persist_dir=str(config.vector_store_path)
            )

        # Try to load existing index
        if not st.session_state.index_loaded:
            st.session_state.indexer.load_index()
            st.session_state.index_loaded = True
            logger.info("Index loaded successfully")

        return True
    except DocumentIndexerError as e:
        logger.warning(f"No existing index found: {e}")
        st.session_state.index_loaded = False
        return False
    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        st.error(f"Error initializing indexer: {e}")
        return False


def get_confidence_color(confidence: str) -> str:
    """Get CSS class for confidence level."""
    confidence_map = {
        "high": "confidence-high",
        "medium": "confidence-medium",
        "low": "confidence-low",
    }
    return confidence_map.get(confidence.lower(), "confidence-medium")


def display_price_result(result: PriceResult) -> None:
    """Display price result with formatted output.

    Args:
        result: PriceResult object to display.
    """
    # Main price display
    st.markdown(
        f'<div class="big-price">${result.price:.2f}</div>',
        unsafe_allow_html=True,
    )

    # Create columns for additional info
    col1, col2 = st.columns(2)

    with col1:
        # Source information
        st.markdown("#### Source Information")
        st.markdown(
            f'<div class="source-info">{result.source}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        # Query breakdown
        if result.query:
            st.markdown("#### Query Breakdown")
            breakdown_text = f"""
            - **Carrier**: {result.query.carrier or 'N/A'}
            - **Service**: {result.query.service or 'N/A'}
            - **Zone**: {result.query.zone or 'N/A'}
            - **Weight**: {result.query.weight} lbs
            """
            if result.query.packaging:
                breakdown_text += f"\n- **Packaging**: {result.query.packaging}"

            st.markdown(
                f'<div class="query-breakdown">{breakdown_text}</div>',
                unsafe_allow_html=True,
            )

    # Expandable section for full details
    with st.expander("View Full LLM Response"):
        st.text(result.details.get("raw_response", "No response available"))
        st.json(
            {
                "search_query": result.details.get("search_query", ""),
                "source_nodes": result.details.get("source_nodes", 0),
            }
        )


def query_section() -> None:
    """Render the main query section."""
    st.header("Search Shipping Prices")

    # Check if index is loaded
    if not st.session_state.index_loaded:
        st.warning(
            "No index loaded. Please upload and index PDF documents first "
            "(see Document Management section below)."
        )
        return

    st.markdown(
        "Enter a natural language query to search for shipping prices. "
        "You can specify carrier, service type, zone, and weight."
    )

    # Query input
    query_text = st.text_input(
        "Enter your shipping query:",
        placeholder="e.g., FedEx 2Day Zone 5 3 lb",
        key="query_input",
    )

    # Search button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        search_button = st.button(
            "Search Price",
            type="primary",
            use_container_width=True,
        )

    # Example queries section
    st.markdown("#### Or click an example to search:")

    # Display example queries in a grid
    cols = st.columns(3)
    example_clicked = None
    for idx, (example_query, expected_price) in enumerate(EXAMPLE_QUERIES):
        col = cols[idx % 3]
        with col:
            if st.button(
                f"{example_query}\n(Expected: ${expected_price})",
                key=f"example_{idx}",
                use_container_width=True,
            ):
                example_clicked = example_query

    # Execute search - either from search button or example click
    query_to_search = None
    if search_button and query_text:
        query_to_search = query_text
    elif example_clicked:
        query_to_search = example_clicked
        st.info(f"🔍 Searching with example: **{example_clicked}**")

    if query_to_search:
        with st.spinner("Searching for price..."):
            try:
                result = get_price(query_to_search)

                # Store in session state
                st.session_state.last_result = result
                st.session_state.query_history.append(
                    {
                        "timestamp": datetime.now(),
                        "query": query_to_search,
                        "price": result.price,
                        "confidence": result.confidence,
                    }
                )

                # Display result
                st.success("Price found successfully!")
                display_price_result(result)

            except PriceRetrieverError as e:
                st.error(f"Failed to retrieve price: {e}")
                logger.error(f"Price retrieval error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")

    elif search_button and not query_text:
        st.warning("Please enter a query.")

    # Display last result if available (only if no new search was performed)
    if st.session_state.last_result and not (search_button or example_clicked):
        st.divider()
        st.subheader("Last Query Result")
        display_price_result(st.session_state.last_result)


def document_management_section() -> None:
    """Render the document management section."""
    st.header("Document Management")

    # Upload section
    st.subheader("Upload New Document")

    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF file containing shipping rates:",
            type=["pdf"],
            help="Upload rate cards or price sheets in PDF format",
        )

    with col2:
        carrier_name = st.text_input(
            "Carrier Name:",
            placeholder="e.g., FedEx",
            help="Name of the shipping carrier",
        )

    if uploaded_file and carrier_name:
        if st.button("Upload and Index", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # Save uploaded file
                    config = get_settings()
                    pdf_path = config.pdf_storage_path / uploaded_file.name
                    pdf_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # Initialize indexer if needed
                    if st.session_state.indexer is None:
                        st.session_state.indexer = DocumentIndexer(
                            persist_dir=str(config.vector_store_path)
                        )

                    # Index the document
                    num_docs = st.session_state.indexer.index_pdf(
                        str(pdf_path), carrier_name
                    )

                    # Save the index
                    st.session_state.indexer.save_index()
                    st.session_state.index_loaded = True

                    st.success(
                        f"Successfully indexed {num_docs} tables from {uploaded_file.name}!"
                    )
                    logger.info(
                        f"Uploaded and indexed {uploaded_file.name} ({num_docs} tables)"
                    )

                except Exception as e:
                    st.error(f"Failed to process document: {e}")
                    logger.error(f"Document processing error: {e}\n{traceback.format_exc()}")

    # List indexed documents
    st.divider()
    st.subheader("Indexed Documents")

    try:
        config = get_settings()
        pdf_dir = config.pdf_storage_path

        if pdf_dir.exists():
            pdf_files = list(pdf_dir.glob("*.pdf"))

            if pdf_files:
                st.write(f"Found {len(pdf_files)} PDF(s) in storage:")

                for pdf_file in pdf_files:
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        st.write(f"📄 {pdf_file.name}")

                    with col2:
                        file_size = pdf_file.stat().st_size / 1024
                        st.write(f"{file_size:.1f} KB")

                    with col3:
                        if st.button("Delete", key=f"delete_{pdf_file.name}"):
                            try:
                                pdf_file.unlink()
                                st.success(f"Deleted {pdf_file.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
            else:
                st.info("No PDF documents in storage yet.")
        else:
            st.info("Storage directory does not exist yet.")

    except Exception as e:
        st.error(f"Error listing documents: {e}")

    # Re-index all documents
    st.divider()

    if st.button("Re-index All Documents"):
        with st.spinner("Re-indexing all documents..."):
            try:
                config = get_settings()
                pdf_dir = config.pdf_storage_path
                pdf_files = list(pdf_dir.glob("*.pdf"))

                if not pdf_files:
                    st.warning("No PDF files to index.")
                    return

                # Create new indexer
                st.session_state.indexer = DocumentIndexer(
                    persist_dir=str(config.vector_store_path)
                )

                # Build list of PDFs to index
                pdf_list = []
                for pdf_file in pdf_files:
                    # Try to infer carrier name from filename
                    carrier = pdf_file.stem.split("_")[0] if "_" in pdf_file.stem else "Unknown"
                    pdf_list.append((str(pdf_file), carrier))

                # Index all PDFs
                results = st.session_state.indexer.build_index(pdf_list)

                # Save index
                st.session_state.indexer.save_index()
                st.session_state.index_loaded = True

                total_docs = sum(results.values())
                st.success(f"Re-indexed {len(pdf_list)} PDFs ({total_docs} total tables)!")

            except Exception as e:
                st.error(f"Failed to re-index: {e}")
                logger.error(f"Re-indexing error: {e}\n{traceback.format_exc()}")


def sidebar_content() -> None:
    """Render sidebar content."""
    with st.sidebar:
        st.header("System Status")

        # Configuration status
        config_status = "✅ Loaded" if st.session_state.config_loaded else "❌ Not Loaded"
        st.write(f"**Configuration**: {config_status}")

        # Index status
        index_status = "✅ Loaded" if st.session_state.index_loaded else "❌ Not Loaded"
        st.write(f"**Index**: {index_status}")

        # Document count
        try:
            config = get_settings()
            pdf_dir = config.pdf_storage_path
            if pdf_dir.exists():
                num_pdfs = len(list(pdf_dir.glob("*.pdf")))
                st.write(f"**Documents**: {num_pdfs}")
            else:
                st.write("**Documents**: 0")
        except Exception:
            st.write("**Documents**: Unknown")

        st.divider()

        # Query history
        st.header("Recent Queries")

        if st.session_state.query_history:
            # Show last 5 queries
            for item in reversed(st.session_state.query_history[-5:]):
                with st.expander(
                    f"${item['price']:.2f} - {item['query'][:30]}...",
                    expanded=False,
                ):
                    st.write(f"**Query**: {item['query']}")
                    st.write(f"**Price**: ${item['price']:.2f}")
                    st.write(f"**Confidence**: {item['confidence']}")
                    st.write(f"**Time**: {item['timestamp'].strftime('%H:%M:%S')}")
        else:
            st.info("No queries yet.")

        st.divider()

        # Settings
        st.header("Settings")

        try:
            config = get_settings()
            st.write(f"**LLM Model**: {config.llm_model}")
            st.write(f"**Embedding Model**: {config.embedding_model}")
            st.write(f"**Top K Results**: {config.retrieval_top_k}")
        except Exception:
            st.write("Settings not available")

        st.divider()

        # Links
        st.header("Documentation")
        st.markdown(
            """
            - [API Guide](../../API_GUIDE.md)
            - [Quick Start](../../QUICK_START.md)
            - [GitHub Issues](https://github.com)
            """
        )

        st.divider()
        st.caption(f"Version {__version__}")


def main() -> None:
    """Main application entry point."""
    # Initialize session state
    init_session_state()

    # Load configuration
    if not st.session_state.config_loaded:
        if not load_config():
            st.stop()

    # Try to initialize indexer
    if st.session_state.indexer is None:
        init_indexer()

    # Render sidebar
    sidebar_content()

    # Main header
    st.title("📦 Shipping Price Search Tool")
    st.markdown(
        """
        Welcome to the Shipping Price Search Tool! This application allows you to:
        - Search for shipping prices using natural language queries
        - Upload and index PDF rate cards from different carriers
        - Manage your document collection
        - View query history and results
        """
    )

    # Check API key
    try:
        config = get_settings()
        if not config.openai_api_key:
            st.error(
                "⚠️ OpenAI API key not configured. "
                "Please set OPENAI_API_KEY in your .env file."
            )
            st.stop()
    except ConfigError:
        st.error(
            "⚠️ Configuration error. Please ensure you have a .env file with required settings."
        )
        st.stop()

    # Main sections
    query_section()

    st.divider()

    document_management_section()

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit, LlamaIndex, and OpenAI. "
        f"Version {__version__}"
    )


if __name__ == "__main__":
    main()
