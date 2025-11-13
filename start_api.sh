#!/bin/bash
# Start script for the Shipping Price Search API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Shipping Price Search API${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found${NC}"
    echo -e "${YELLOW}Please create a .env file with your configuration.${NC}"
    echo -e "${YELLOW}See .env.example for reference.${NC}"
    exit 1
fi

# Check if OPENAI_API_KEY is set
if ! grep -q "OPENAI_API_KEY=sk-" .env; then
    echo -e "${YELLOW}Warning: OPENAI_API_KEY may not be set in .env${NC}"
    echo -e "${YELLOW}The API requires a valid OpenAI API key to function.${NC}"
fi

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry not found${NC}"
    echo -e "${YELLOW}Please install Poetry: https://python-poetry.org/docs/#installation${NC}"
    exit 1
fi

# Install dependencies if needed
echo -e "${BLUE}Checking dependencies...${NC}"
poetry install --quiet

echo -e "${GREEN}Starting API server...${NC}"
echo ""
echo -e "${YELLOW}API will be available at:${NC}"
echo -e "  - API: http://localhost:8000"
echo -e "  - Interactive Docs: http://localhost:8000/docs"
echo -e "  - ReDoc: http://localhost:8000/redoc"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
