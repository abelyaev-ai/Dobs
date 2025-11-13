#!/bin/bash
# Startup script for Streamlit UI

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Shipping Price Search Tool - Streamlit UI${NC}"
echo "=================================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file with your OPENAI_API_KEY"
    echo ""
    echo "Example:"
    echo "OPENAI_API_KEY=your-api-key-here"
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed!${NC}"
    echo "Install it with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if Poetry has dependencies installed
if ! poetry run python -c "import streamlit" &> /dev/null; then
    echo -e "${RED}Error: Dependencies not installed!${NC}"
    echo "Run: poetry install"
    exit 1
fi

# Check if data directories exist
if [ ! -d "data/pdfs" ]; then
    echo -e "${YELLOW}Creating data/pdfs directory...${NC}"
    mkdir -p data/pdfs
fi

if [ ! -d "data/storage" ]; then
    echo -e "${YELLOW}Creating data/storage directory...${NC}"
    mkdir -p data/storage
fi

# Display configuration
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  - PDFs directory: data/pdfs"
echo "  - Index storage: data/storage"
echo "  - Port: 8501 (default)"
echo ""

# Check if there are indexed documents
if [ -f "data/storage/docstore.json" ]; then
    echo -e "${GREEN}Index found - ready to query!${NC}"
else
    echo -e "${YELLOW}No index found - you'll need to upload PDFs first${NC}"
fi

echo ""
echo "Starting Streamlit..."
echo "=================================================="
echo ""

# Run Streamlit with the app via Poetry
# Poetry automatically handles PYTHONPATH and virtual environment
poetry run streamlit run src/ui/streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=false \
    --browser.gatherUsageStats=false

# Note: If you want to run in the background, add:
# --server.headless=true &
