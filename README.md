# Shipping Price Search Tool

A comprehensive semantic search system for extracting shipping prices from PDF rate cards using AI-powered table extraction, vector embeddings, and natural language queries.

**Version**: 0.1.0
**Python**: 3.10-3.12
**License**: MIT

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [API Documentation](#api-documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Shipping Price Search Tool allows you to:
- Upload PDF shipping rate cards (FedEx, UPS, USPS, etc.)
- Index tables using AI-powered extraction and vector embeddings
- Query prices using natural language (e.g., "FedEx 2Day Zone 5 3 lb")
- Get instant, accurate pricing with source citations

**Problem Solved**: Traditional shipping rate lookups require manual PDF searching through complex multi-page rate cards. This tool automates the process using semantic search and LLM-based extraction.

**One input → One price.**

---

## Key Features

- **Natural Language Queries**: "FedEx 2Day, Zone 5, 3 lb" → Price with confidence score
- **Intelligent LLM-Based Parsing**:
  - LLM query parser handles abbreviations, typos, and natural language variations
  - Automatic service name normalization using extracted service catalogs
  - Fallback to regex parsing for reliability
- **Smart Page-Based Extraction**: Extracts complete pages from PDFs instead of individual tables
- **Dual Price Extraction Strategy**:
  - LLM table parser for direct price extraction from any table format
  - Fallback to semantic search + LLM extraction
- **Semantic Search**: FAISS vector store with metadata filtering for fast, accurate retrieval
- **Service Catalog**: Automatically builds canonical service name catalog from PDFs
- **Multiple Interfaces**:
  - RESTful API (FastAPI)
  - Web UI (Streamlit)
  - Python SDK
- **Rich Metadata**: Tracks carrier, service types, zones, weight ranges, and sources
- **Performance Optimizations**: LRU caching for query and table parsing, batch embeddings
- **Persistent Index**: Save once, query many times

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interfaces                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Streamlit UI │  │  FastAPI     │  │  Python SDK  │         │
│  │   (Web)      │  │  (REST API)  │  │  (Direct)    │         │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘         │
└─────────┼──────────────────┼──────────────────┼────────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
          ┌─────────────────────────────────────────────┐
          │         Price Retriever (Core Logic)        │
          │  • LLM query parser with service catalog    │
          │  • Metadata pre-filtering (service+zone)    │
          │  • Try LLM table parser first               │
          │  • Fallback to semantic search + extraction │
          └──────────────────┬──────────────────────────┘
                             │
          ┌──────────────────┴─────────────────────┐
          │                                        │
          ▼                                        ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  LLM Query Parser   │              │   Document Indexer      │
│  • Service catalog  │              │  • Service catalog      │
│  • Abbreviations    │              │  • PDF page extraction  │
│  • Typo handling    │              │  • Metadata extraction  │
│  • LRU caching      │              │  • Page chunking        │
│  • Regex fallback   │              │  • Vector embeddings    │
└─────────┬───────────┘              │  • FAISS indexing       │
          │                          │  • Persistence          │
          │                          └──────────┬──────────────┘
          │                                     │
          ▼                                     ▼
┌─────────────────────┐              ┌─────────────────────────┐
│ LLM Table Parser    │              │  PDF Components         │
│  • Direct price     │              │  ┌──────────────────┐   │
│    extraction       │              │  │ Page Extractor   │   │
│  • All table formats│              │  │  • pdfplumber    │   │
│  • LRU caching      │              │  └──────────────────┘   │
│  • Interpolation    │              │  ┌──────────────────┐   │
└─────────────────────┘              │  │ Service Catalog  │   │
                                     │  │  • Extract names │   │
          ┌──────────────────────────┤  └──────────────────┘   │
          │                          │  ┌──────────────────┐   │
          │                          │  │ Metadata Extract │   │
          │                          │  │  • Services      │   │
          │                          │  │  • Zones         │   │
          │                          │  │  • Weight ranges │   │
          │                          │  └──────────────────┘   │
          │                          └─────────────────────────┘
          │
          ▼                                     ▼
┌─────────────────────┐              ┌─────────────────────────┐
│  FAISS Vector Store │              │   OpenAI API            │
│  • 1536-dim vectors │              │  • gpt-4o (LLM)         │
│  • L2 distance      │              │  • gpt-4o-mini (parser) │
│  • Metadata filter  │              │  • text-embed-3-small   │
│  • Fast retrieval   │              │  • Temperature=0        │
└─────────────────────┘              └─────────────────────────┘
```

### Data Flow

1. **Indexing** (One-time per PDF):
   ```
   PDF → Build Service Catalog
       → Extract Pages (with pdfplumber)
       → Extract Metadata (services, zones, weight ranges)
       → Create Page Chunks
       → Generate Embeddings
       → Store in FAISS Index
   ```

2. **Querying** (Per search):
   ```
   Query → LLM Parse (with service catalog) → Normalize to canonical service
       → Metadata Filter (service + zone + weight range)
       → Try LLM Table Parser
       → If fails: Semantic Search + LLM Extraction
       → Return Price + Confidence + Source
   ```

---

## Quick Start

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **OpenAI API Key**: Required for embeddings and LLM
- **Operating System**: Linux, macOS, or Windows
- **Memory**: 4GB+ recommended
- **Disk Space**: 500MB for dependencies + storage

### Installation

#### Option 1: Using Poetry (Recommended)

```bash
# Clone or navigate to project directory
cd /storage/projects/Dobs

# Install dependencies with Poetry
poetry install

# Activate virtual environment
poetry shell
```

#### Option 2: Using pip

```bash
# Install from pyproject.toml
pip install -e .

# Or install dependencies manually
pip install pdfplumber llama-index llama-index-llms-openai \
    llama-index-embeddings-openai llama-index-vector-stores-faiss \
    faiss-cpu fastapi streamlit python-dotenv pandas pydantic \
    pydantic-settings python-multipart uvicorn tabulate
```

### Configuration

1. **Create `.env` file** in the project root:
   ```bash
   touch .env
   ```

2. **Add your OpenAI API key** to `.env`:
   ```bash
   # Required
   OPENAI_API_KEY=sk-your-actual-api-key-here

   # Optional (defaults provided in src/utils/config.py)
   EMBEDDING_MODEL=text-embedding-3-small
   LLM_MODEL=gpt-4o-2024-11-20
   LLM_TEMPERATURE=0.0
   RETRIEVAL_TOP_K=5
   QUERY_PARSER_CACHE_SIZE=1000
   TABLE_PARSER_CACHE_SIZE=10000
   CHUNK_SIZE=2048
   CHUNK_OVERLAP=200
   ```

   **Note**: See `src/utils/config.py` for all available configuration options.

### Initial Indexing

Before querying, you need to index your PDF rate cards:

```bash
# Using Python script
python -c "
from src.core.document_indexer import DocumentIndexer

# Initialize indexer
indexer = DocumentIndexer(persist_dir='data/storage')

# Index PDFs (builds service catalog first, then indexes)
results = indexer.build_index([
    ('data/pdfs/FedEx_Standard_List_Rates_2025.pdf', 'FedEx'),
])

print(f'Indexed {sum(results.values())} page-based chunks')

# Save index
indexer.save_index()
print('Index saved successfully!')
"
```

**Expected output**:
```
Phase 1: Building service catalog from PDFs...
Service catalog built with X canonical services
Phase 2: Indexing documents with canonical service names...
Indexed 60 page-based chunks
Index saved successfully!
```

**What Happens During Indexing**:
1. **Service Catalog Building**: Extracts all unique service names from PDFs
2. **Page Extraction**: Uses pdfplumber to extract complete pages (not individual tables)
3. **Metadata Extraction**: Extracts services, zones, and weight ranges per page
4. **Page Chunking**: Creates one chunk per page (~60 chunks instead of 277 table-based chunks)
5. **Vector Embedding**: Generates embeddings for semantic search
6. **FAISS Indexing**: Stores in FAISS vector store with metadata

**Note**: Indexing takes 2-5 minutes per PDF on first run. The index is saved and loaded instantly on subsequent runs.

### Running the Application

#### Option 1: Streamlit Web UI (Easiest)

```bash
# Start Streamlit app
./start_streamlit.sh

# Or manually:
streamlit run src/ui/streamlit_app.py --server.port 8501
```

Access at: **http://localhost:8501**

#### Option 2: FastAPI REST API

```bash
# Start FastAPI server
./start_api.sh

# Or manually:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs at: **http://localhost:8000/docs**

#### Option 3: Python SDK

```python
from src.core.price_retriever import get_price

# Query directly
result = get_price("FedEx 2Day Zone 5 3 lb")
print(f"Price: ${result.price:.2f}")
print(f"Confidence: {result.confidence}")
print(f"Source: {result.source}")
```

---

## Usage Examples

### Example 1: Basic Query (Streamlit UI)

1. Open http://localhost:8501
2. Enter query: `FedEx 2Day, Zone 5, 3 lb`
3. Click "Search Price"
4. View result with price, confidence, and source

### Example 2: API Query (cURL)

```bash
curl -X POST "http://localhost:8000/price" \
  -H "Content-Type: application/json" \
  -d '{"query": "Ground Z6 12 lb"}'
```

**Response**:
```json
{
  "price": 21.64,
  "confidence": "high",
  "source": "FedEx_Standard_List_Rates_2025.pdf, Page 2, Table 0",
  "query_breakdown": {
    "service": "Ground",
    "zone": "6",
    "weight": 12.0
  }
}
```

### Example 3: Python SDK

```python
from src.core.price_retriever import PriceRetriever
from src.core.document_indexer import DocumentIndexer

# Initialize
indexer = DocumentIndexer()
indexer.load_index()
retriever = PriceRetriever(indexer=indexer)

# Query
result = retriever.get_price("Express Saver Z8 1 lb")
print(f"Price: ${result.price:.2f}")
print(f"Confidence: {result.confidence}")
```

### Example 4: Uploading New PDFs (API)

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/new_rates.pdf" \
  -F "carrier_name=DHL"
```

### Example Queries (From Requirements)

```python
# Test cases from Taks.md
queries = [
    "FedEx 2Day, Zone 5, 3 lb",           # Expected: $48.05
    "Standard Overnight, z2, 10 lbs",     # Expected: $58.48
    "Express Saver Z8 1 lb",              # Expected: $39.86
    "Ground Z6 12 lb",                    # Expected: $21.64
    "Home Delivery zone 3 5 lb",          # Expected: $14.76
]

for query in queries:
    result = get_price(query)
    print(f"{query} → ${result.price:.2f}")
```

---

## Project Structure

```
/storage/projects/Dobs/
├── README.md                    # This file
├── LICENSE.md                   # MIT License
├── pyproject.toml               # Dependencies and Poetry configuration
├── .env                         # Environment variables (create this)
│
├── src/                         # Source code
│   ├── api/                     # FastAPI REST API
│   │   ├── main.py             # FastAPI app with lifespan management
│   │   ├── endpoints.py        # Route handlers (/price, /upload, /documents)
│   │   └── models.py           # Pydantic request/response models
│   ├── core/                    # Core business logic
│   │   ├── price_retriever.py        # Main PriceRetriever class
│   │   ├── document_indexer.py       # FAISS indexing with service catalog
│   │   ├── llm_query_parser.py       # LLM-based query parsing
│   │   ├── llm_table_parser.py       # LLM-based table price extraction
│   │   ├── query_parser.py           # Regex-based query parser (fallback)
│   │   ├── service_catalog.py        # Service name extraction from PDFs
│   │   ├── page_extractor.py         # PDF page extraction (pdfplumber)
│   │   ├── page_chunker.py           # Page-based document chunking
│   │   └── metadata_extractor.py     # Extract metadata from pages
│   ├── ui/                      # Streamlit interface
│   │   └── streamlit_app.py    # Interactive web UI
│   └── utils/                   # Utilities
│       ├── config.py           # Pydantic settings management
│       └── logger.py           # Structured logging
│
├── tests/                       # Test suite
│   ├── test_query_parser.py
│   ├── test_price_retriever.py
│   ├── test_pdf_extractor.py
│   └── test_pdf_table_extractor.py
├── data/                        # Data directory (created automatically)
│   ├── pdfs/                   # PDF rate cards storage
│   ├── storage/                # FAISS vector store persistence
│   └── service_catalog.json    # Extracted service names
├── start_api.sh                 # Launch FastAPI server
└── start_streamlit.sh           # Launch Streamlit UI
```

---

## Core Components

### 1. Price Retriever

**Location**: `src/core/price_retriever.py`

Main orchestrator for price retrieval with dual extraction strategy.

**Features**:
- LLM-based query parsing with service catalog integration
- Metadata pre-filtering (service + zone + weight range)
- **Primary**: LLM table parser for direct extraction
- **Fallback**: Semantic search + LLM extraction
- Custom prompts optimized for shipping rate tables
- Confidence scoring and source tracking

### 2. LLM Query Parser

**Location**: `src/core/llm_query_parser.py`

Intelligent query parsing using GPT-4o-mini with structured output.

**Features**:
- Handles abbreviations and typos ("expr saver" → "FedEx Express Saver")
- Uses service catalog for canonical name mapping
- LRU caching (default: 1000 queries)
- Automatic fallback to regex parser on errors
- Pydantic validation for type safety

**Example**:
```python
"fedx 2day z5 3lb" → {
    carrier: "FedEx",
    service: "FedEx 2Day",  # Canonical form
    zone: "5",
    weight: 3.0
}
```

### 3. LLM Table Parser

**Location**: `src/core/llm_table_parser.py`

Extracts prices directly from table text using LLM intelligence.

**Features**:
- Works with any table format (multi-service, single-service)
- Handles weight interpolation automatically
- LRU caching (default: 10,000 entries)
- Returns price + confidence level
- Fast and accurate

### 4. Document Indexer

**Location**: `src/core/document_indexer.py`

Two-phase indexing system with service catalog building.

**Features**:
- **Phase 1**: Builds service catalog from PDFs
- **Phase 2**: Indexes pages with canonical service names
- Page-based chunking (~60 chunks per PDF)
- FAISS IndexFlatL2 (1536 dimensions)
- OpenAI text-embedding-3-small
- Rich metadata extraction (services, zones, weight ranges)
- Persistent storage (save/load)

### 5. Service Catalog

**Location**: `src/core/service_catalog.py`

Extracts and normalizes service names from PDFs.

**Features**:
- Automatically discovers all service variations
- Normalizes to canonical forms
- Saves to `data/service_catalog.json`
- Used by query parser for accurate mapping

### 6. PDF Components

**Locations**: `src/core/page_extractor.py`, `src/core/page_chunker.py`, `src/core/metadata_extractor.py`

Page-based PDF processing pipeline.

**Features**:
- **Page Extractor**: Uses pdfplumber to extract complete pages
- **Metadata Extractor**: Extracts services, zones, weight ranges per page
- **Page Chunker**: Creates one document chunk per page

### 7. FastAPI API

**Location**: `src/api/`

**Endpoints**:
- `POST /price` - Get shipping price with query breakdown
- `POST /upload` - Upload and index PDF
- `GET /documents` - List indexed documents with metadata
- `GET /health` - Health check

**Interactive Docs**: http://localhost:8000/docs

### 8. Streamlit UI

**Location**: `src/ui/streamlit_app.py`

Interactive web interface for price search and document management.

**Features**:
- Natural language price search
- Query history
- PDF upload and indexing
- Document management
- Example queries for testing

**Access**: http://localhost:8501

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### Get Price
```http
POST /price
Content-Type: application/json

{
  "query": "FedEx 2Day, Zone 5, 3 lb"
}
```

#### Upload PDF
```http
POST /upload
Content-Type: multipart/form-data

file: <PDF file>
carrier_name: "FedEx"
```

#### List Documents
```http
GET /documents
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## Development

### Running Tests

```bash
pytest                               # Run all tests
pytest --cov=src                     # With coverage
pytest tests/test_query_parser.py    # Specific file
```

### Code Quality

```bash
black src tests           # Format code
ruff check src tests      # Lint code
```

### Configuration

Edit `.env` to customize:

```bash
LLM_MODEL=gpt-4o-2024-11-20        # Main LLM model
RETRIEVAL_TOP_K=5                  # Top results to retrieve
CHUNK_SIZE=2048                    # Characters per chunk
LLM_TEMPERATURE=0.0                # Deterministic output
QUERY_PARSER_CACHE_SIZE=1000       # LLM query parser cache
TABLE_PARSER_CACHE_SIZE=10000      # LLM table parser cache
```

See `src/utils/config.py` for all available options.

### Performance

- **Indexing**: 2-5 minutes per PDF (first-time)
  - Phase 1: Service catalog building (~30 seconds)
  - Phase 2: Page extraction and indexing (~2-4 minutes)
- **Loading**: <5 seconds
- **Query**:
  - LLM table parser (cached): <100ms
  - LLM table parser (uncached): ~1-2 seconds
  - Fallback search: ~2-3 seconds
- **Cost**:
  - Query parsing: ~$0.0001 per query (cached after first use)
  - Table parsing: ~$0.001 per query (cached after first use)
  - Total: <$0.002 per unique query

---

## Troubleshooting

### Index Not Found

```bash
# Re-run indexing
python -c "
from src.core.document_indexer import DocumentIndexer
indexer = DocumentIndexer()
indexer.build_index([('data/pdfs/FedEx_Standard_List_Rates_2025.pdf', 'FedEx')])
indexer.save_index()
"
```

### OpenAI API Key Error

1. Copy `.env.example` to `.env`
2. Add API key: `OPENAI_API_KEY=sk-...`

### Port In Use

```bash
kill -9 $(lsof -ti:8000)    # Kill process
# Or use different port
uvicorn src.api.main:app --port 8001
```

### Query Returns Wrong Price

1. Check query parsing
2. Verify retrieved tables
3. Inspect source PDF manually

---

## Author

Alexander Belyaev <belazzbelazz@gmail.com>

## License

MIT License

---

**Version**: 0.1.0
**Last Updated**: 2025-11-14
