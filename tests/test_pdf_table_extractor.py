"""Unit tests for PDF table extractor module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.pdf_table_extractor import (
    ExtractedTable,
    PDFTableExtractor,
    TableExtractionError,
    TableMetadata,
)


class TestPDFTableExtractor:
    """Tests for PDFTableExtractor class."""

    def test_initialization_default(self):
        """Test extractor initializes with default settings."""
        extractor = PDFTableExtractor()

        assert extractor.min_table_rows == 2
        assert extractor.min_table_cols == 2
        assert extractor.try_multiple_strategies is True
        assert extractor.table_settings is not None
        assert len(extractor.alternative_strategies) == 3

    def test_initialization_custom(self):
        """Test extractor initializes with custom settings."""
        custom_settings = {"vertical_strategy": "text"}
        extractor = PDFTableExtractor(
            table_settings=custom_settings,
            min_table_rows=3,
            min_table_cols=3,
            try_multiple_strategies=False,
        )

        assert extractor.min_table_rows == 3
        assert extractor.min_table_cols == 3
        assert extractor.try_multiple_strategies is False
        assert extractor.table_settings == custom_settings

    def test_extract_nonexistent_file(self):
        """Test extraction fails gracefully for nonexistent file."""
        extractor = PDFTableExtractor()

        with pytest.raises(TableExtractionError, match="not found"):
            extractor.extract_tables_from_pdf(
                "/nonexistent/path.pdf", carrier_name="Test"
            )

    def test_extract_directory_instead_of_file(self, tmp_path):
        """Test extraction fails when path is a directory."""
        extractor = PDFTableExtractor()

        with pytest.raises(TableExtractionError, match="not a file"):
            extractor.extract_tables_from_pdf(tmp_path, carrier_name="Test")

    @pytest.mark.slow
    def test_extract_fedex_pdf(self):
        """Test extraction from FedEx PDF (integration test)."""
        pdf_path = Path(
            "/storage/projects/Work/Dobs/data/pdfs/FedEx_Standard_List_Rates_2025.pdf"
        )

        if not pdf_path.exists():
            pytest.skip("FedEx PDF not available")

        extractor = PDFTableExtractor()
        tables = extractor.extract_tables_from_pdf(pdf_path, carrier_name="FedEx")

        # Basic assertions
        assert len(tables) > 0, "Should extract at least one table"

        # Check that tables are ExtractedTable instances
        assert all(isinstance(t, ExtractedTable) for t in tables)

        # Check metadata
        assert all(t.metadata.carrier_name == "FedEx" for t in tables)
        assert all(t.metadata.source_document == pdf_path.name for t in tables)

        # Check that some tables have service types
        tables_with_service = [
            t for t in tables if t.metadata.service_type is not None
        ]
        assert len(tables_with_service) > 0, "Should detect at least one service type"

        # Check that some tables have zones
        tables_with_zones = [t for t in tables if len(t.metadata.zones) > 0]
        assert len(tables_with_zones) > 0, "Should detect zones in at least one table"

    def test_is_valid_table_empty(self):
        """Test validation rejects empty tables."""
        extractor = PDFTableExtractor()

        assert not extractor._is_valid_table(None)
        assert not extractor._is_valid_table([])
        assert not extractor._is_valid_table([[]])

    def test_is_valid_table_too_small(self):
        """Test validation rejects tables below minimum size."""
        extractor = PDFTableExtractor(min_table_rows=2, min_table_cols=2)

        # Only 1 row
        assert not extractor._is_valid_table([["a", "b"]])

        # Only 1 column
        assert not extractor._is_valid_table([["a"], ["b"]])

    def test_is_valid_table_all_empty_cells(self):
        """Test validation rejects tables with all empty cells."""
        extractor = PDFTableExtractor()

        table = [[None, None], [None, None], ["", ""]]
        assert not extractor._is_valid_table(table)

    def test_is_valid_table_valid(self):
        """Test validation accepts valid tables."""
        extractor = PDFTableExtractor(min_table_rows=2, min_table_cols=2)

        table = [["Header 1", "Header 2"], ["Value 1", "Value 2"]]
        assert extractor._is_valid_table(table)

    def test_create_dataframe_simple(self):
        """Test DataFrame creation from simple table."""
        extractor = PDFTableExtractor()

        raw_table = [
            ["Zone", "Price"],
            ["Zone 2", "$10.00"],
            ["Zone 3", "$12.00"],
        ]

        df = extractor._create_dataframe(raw_table)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["Zone", "Price"]
        assert len(df) == 2
        assert df.iloc[0, 0] == "Zone 2"
        assert df.iloc[1, 1] == "$12.00"

    def test_create_dataframe_with_empty_cells(self):
        """Test DataFrame creation handles empty cells."""
        extractor = PDFTableExtractor()

        raw_table = [
            ["Weight", "Zone 2", "Zone 3"],
            ["1 lb", "$10.00", ""],
            ["2 lb", None, "$12.00"],
        ]

        df = extractor._create_dataframe(raw_table)

        assert pd.isna(df.iloc[0, 2])  # Empty string becomes None
        assert pd.isna(df.iloc[1, 1])  # None stays None
        assert df.iloc[1, 2] == "$12.00"

    def test_create_dataframe_strips_whitespace(self):
        """Test DataFrame creation strips whitespace."""
        extractor = PDFTableExtractor()

        raw_table = [
            ["  Header  ", " Value  "],
            [" Data 1 ", "  Data 2  "],
        ]

        df = extractor._create_dataframe(raw_table)

        assert df.columns[0] == "Header"
        assert df.iloc[0, 0] == "Data 1"

    def test_extract_service_type_from_header(self):
        """Test service type extraction from column headers."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Weight", "FedEx 2Day Rate", "FedEx Ground Rate"])

        service_type = extractor._extract_service_type(df, "")

        assert service_type is not None
        assert "2Day" in service_type or "2 Day" in service_type.replace(" ", "")

    def test_extract_service_type_from_page_text(self):
        """Test service type extraction from page text."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Col1", "Col2"])
        page_text = "FedEx Express Saver Rates for 2025"

        service_type = extractor._extract_service_type(df, page_text)

        assert service_type is not None
        assert "Express Saver" in service_type

    def test_extract_service_type_not_found(self):
        """Test service type extraction returns None when not found."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Col1", "Col2"])
        page_text = "Some random text"

        service_type = extractor._extract_service_type(df, page_text)

        assert service_type is None

    def test_extract_zones_from_headers(self):
        """Test zone extraction from column headers."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Weight", "Zone 2", "Zone 3", "Zone 5"])

        zones = extractor._extract_zones(df)

        assert len(zones) == 3
        assert "Zone 2" in zones
        assert "Zone 3" in zones
        assert "Zone 5" in zones

    def test_extract_zones_from_row_pattern(self):
        """Test zone extraction from row pattern like '2 3 4 5 6 7 8'."""
        extractor = PDFTableExtractor()

        # Create DataFrame with zone numbers in a row
        df = pd.DataFrame(
            {
                "Col1": ["Header", "2 3 4 5 6 7 8", "Data"],
                "Col2": [None, None, "$10.00"],
            }
        )

        zones = extractor._extract_zones(df)

        assert len(zones) > 0
        # Should extract Zone 2 through Zone 8
        expected_zones = [f"Zone {i}" for i in range(2, 9)]
        for expected in expected_zones:
            assert expected in zones

    def test_extract_zones_not_found(self):
        """Test zone extraction returns empty list when not found."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Weight", "Price"])

        zones = extractor._extract_zones(df)

        assert zones == []

    def test_has_weight_column_true(self):
        """Test weight column detection returns True when present."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Weight (lbs)", "Zone 2", "Zone 3"])

        assert extractor._has_weight_column(df) is True

    def test_has_weight_column_various_formats(self):
        """Test weight column detection handles various formats."""
        extractor = PDFTableExtractor()

        formats = [
            ["Weight", "Zone 2"],
            ["Wt.", "Zone 2"],
            ["lb", "Zone 2"],
            ["lbs", "Zone 2"],
            ["#", "Zone 2"],
            ["Pounds", "Zone 2"],
        ]

        for columns in formats:
            df = pd.DataFrame(columns=columns)
            assert extractor._has_weight_column(df) is True, f"Failed for {columns}"

    def test_has_weight_column_false(self):
        """Test weight column detection returns False when absent."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(columns=["Zone 2", "Zone 3", "Zone 4"])

        assert extractor._has_weight_column(df) is False

    def test_to_markdown_format(self):
        """Test markdown conversion includes metadata and table."""
        extractor = PDFTableExtractor()

        df = pd.DataFrame(
            {"Weight": ["1 lb", "2 lb"], "Zone 2": ["$10.00", "$12.00"]}
        )

        metadata = TableMetadata(
            carrier_name="FedEx",
            service_type="Ground",
            zones=["Zone 2"],
            page_number=0,
            table_number=0,
            source_document="test.pdf",
            has_weight_column=True,
            row_count=2,
            column_count=2,
        )

        markdown = extractor._to_markdown(df, metadata)

        # Check metadata section
        assert "## Table Metadata" in markdown
        assert "FedEx" in markdown
        assert "Ground" in markdown
        assert "Zone 2" in markdown

        # Check table is present
        assert "Weight" in markdown
        assert "$10.00" in markdown

    def test_extract_tables_as_documents(self):
        """Test document format conversion."""
        # Create a minimal valid PDF path for testing structure
        # (actual extraction tested in integration test)
        extractor = PDFTableExtractor()

        # This would be tested with a real PDF in integration tests
        # Here we just verify the method exists and has correct signature
        assert hasattr(extractor, "extract_tables_as_documents")


class TestTableMetadata:
    """Tests for TableMetadata dataclass."""

    def test_creation(self):
        """Test metadata can be created with all fields."""
        metadata = TableMetadata(
            carrier_name="FedEx",
            service_type="2Day",
            zones=["Zone 2", "Zone 3"],
            page_number=5,
            table_number=2,
            source_document="test.pdf",
            has_weight_column=True,
            row_count=25,
            column_count=8,
        )

        assert metadata.carrier_name == "FedEx"
        assert metadata.service_type == "2Day"
        assert len(metadata.zones) == 2
        assert metadata.page_number == 5
        assert metadata.has_weight_column is True


class TestExtractedTable:
    """Tests for ExtractedTable dataclass."""

    def test_creation(self):
        """Test extracted table can be created with all fields."""
        df = pd.DataFrame({"Col1": [1, 2], "Col2": [3, 4]})

        metadata = TableMetadata(
            carrier_name="Test",
            service_type=None,
            zones=[],
            page_number=0,
            table_number=0,
            source_document="test.pdf",
            has_weight_column=False,
            row_count=2,
            column_count=2,
        )

        markdown = "# Test\n| Col1 | Col2 |\n|------|------|\n| 1 | 3 |"

        table = ExtractedTable(dataframe=df, metadata=metadata, markdown=markdown)

        assert isinstance(table.dataframe, pd.DataFrame)
        assert isinstance(table.metadata, TableMetadata)
        assert isinstance(table.markdown, str)
        assert len(table.dataframe) == 2
