#!/usr/bin/env python3
"""Test all 5 user-reported queries to verify fixes."""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core.price_retriever import get_price
from src.utils.config import load_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Test all 5 user-reported queries."""
    print("="*80)
    print("TESTING ALL 5 USER-REPORTED QUERIES")
    print("="*80)
    print()

    # Load settings
    try:
        load_settings()
        print("✓ Settings loaded")
    except Exception as e:
        print(f"✗ Failed to load settings: {e}")
        return 1

    # Test queries with EXPECTED values from user
    test_cases = [
        ("FedEx 2Day Zone 5 3 lb", 39.85, "2Day"),
        ("Standard Overnight z2 10lbs", 58.48, "Standard Overnight"),
        ("Express Saver Z8 1lb", 39.86, "Express Saver"),
        ("Ground Z6 12 lb", 21.64, "Ground"),
        ("Home Delivery zone 3 5 lb", 14.76, "Home Delivery"),
    ]

    print()
    print("Running test queries:")
    print("-" * 80)
    print()

    results = []

    for idx, (query, expected, service) in enumerate(test_cases, 1):
        print(f"Test {idx}/{len(test_cases)}: {query}")
        print(f"  Expected: ${expected:.2f} ({service})")

        try:
            result = get_price(query)

            print(f"  Got: ${result.price:.2f}")
            print(f"  Confidence: {result.confidence}")
            print(f"  Source: {result.source}")
            print(f"  Filtered docs: {result.details.get('filtered_docs_count', '?')}")

            # Check if result matches expectations
            price_diff = abs(result.price - expected)

            if price_diff < 0.10:  # Within 10 cents
                status = "✓ PASS"
                success = True
            else:
                status = f"✗ FAIL - Expected ${expected:.2f}, got ${result.price:.2f} (diff: ${price_diff:.2f})"
                success = False

            print(f"  {status}")
            results.append((query, success, result.price, expected, price_diff))

        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            results.append((query, False, None, expected, None))

        print()

    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()

    passed = sum(1 for _, success, _, _, _ in results if success)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")
    print()

    # Detailed results
    print("Detailed Results:")
    print("-" * 80)
    for query, success, got, expected, diff in results:
        symbol = "✓" if success else "✗"
        got_str = f"${got:.2f}" if got else "ERROR"
        diff_str = f"(diff: ${diff:.2f})" if diff is not None else ""
        print(f"{symbol} {query:45}")
        print(f"   Expected: ${expected:.2f}, Got: {got_str} {diff_str}")

    print()
    print("-" * 80)

    if passed == total:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
