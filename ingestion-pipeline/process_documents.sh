#!/bin/bash

# Process Documents Script for Regulatory Chat Bot
# This script processes regulatory documents through the ingestion pipeline

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Regulatory Document Processing Pipeline ===${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file with your Azure credentials."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # turn off automatic export
fi

# Parse command line arguments
BATCH_SIZE=5
SKIP_PROCESSED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip-processed)
            SKIP_PROCESSED=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --batch-size N      Process N documents at a time (default: 5)"
            echo "  --skip-processed    Skip already processed documents"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Run the document processing script
echo -e "${GREEN}Starting document processing...${NC}"
echo "Batch size: $BATCH_SIZE"
echo "Skip processed: $SKIP_PROCESSED"
echo ""

if [ "$SKIP_PROCESSED" = true ]; then
    ./pipeline process --preset full --batch-size $BATCH_SIZE
else
    ./pipeline process --preset full --batch-size $BATCH_SIZE
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Document processing completed successfully!${NC}"
else
    echo -e "${RED}Document processing failed!${NC}"
    exit 1
fi