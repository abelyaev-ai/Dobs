#!/usr/bin/env python3
"""Rebuild the document index with corrected metadata extraction."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.document_indexer import DocumentIndexer
from src.utils.config import load_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Rebuild the index with corrected metadata extraction."""
    print("="*80)
    print("REBUILDING INDEX WITH CORRECTED METADATA EXTRACTION")
    print("="*80)
    print()
    print("Fixes applied:")
    print("  1. Zone extraction regex (no more 'zone f' false positives)")
    print("  2. Multiweight page detection (separate Express Multiweight service)")
    print("  3. Weight range extraction (for priority filtering)")
    print()

    # Load settings
    try:
        load_settings()
        print("✓ Settings loaded")
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        return 1

    # Initialize indexer
    try:
        indexer = DocumentIndexer(persist_dir="data/storage")
        print("✓ DocumentIndexer initialized")
    except Exception as e:
        print(f"✗ Failed to initialize indexer: {e}")
        return 1

    # Define PDFs to index
    pdf_paths = [
        ("data/pdfs/FedEx_Standard_List_Rates_2025.pdf", "FedEx")
    ]

    print()
    print(f"Indexing {len(pdf_paths)} PDF(s)...")
    print()

    # Build index
    try:
        chunk_counts = indexer.build_index(pdf_paths)
        print()
        print("✓ Index built successfully")
        print()
        print("Chunk counts:")
        for pdf_path, count in chunk_counts.items():
            print(f"  {pdf_path}: {count} chunks")
    except Exception as e:
        print(f"✗ Failed to build index: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save index
    try:
        indexer.save_index()
        print()
        print("✓ Index saved to disk")
    except Exception as e:
        print(f"✗ Failed to save index: {e}")
        return 1

    print()
    print("="*80)
    print("INDEX REBUILD COMPLETE")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run test_fixes.py to verify all queries work")
    print("  2. Check metadata with inspect_metadata.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
