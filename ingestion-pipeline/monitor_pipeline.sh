#!/bin/bash

# Monitor Pipeline Script for Regulatory Chat Bot
# This script monitors the document processing pipeline status

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=== Regulatory Pipeline Monitor ===${NC}"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${RED}Error: .env file not found!${NC}"
    echo "Please create a .env file with your database credentials."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    set -a  # automatically export all variables
    source .env
    set +a  # turn off automatic export
fi

# Parse command line arguments
WATCH_MODE=false
INTERVAL=30

while [[ $# -gt 0 ]]; do
    case $1 in
        --watch)
            WATCH_MODE=true
            shift
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --watch             Continuously monitor the pipeline"
            echo "  --interval N        Refresh interval in seconds (default: 30)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Function to run monitoring
run_monitor() {
    clear
    echo -e "${CYAN}=== Regulatory Pipeline Monitor ===${NC}"
    echo -e "Last updated: $(date)"
    echo ""
    ./pipeline monitor --once
}

# Run monitoring
if [ "$WATCH_MODE" = true ]; then
    echo -e "${GREEN}Starting continuous monitoring (Ctrl+C to stop)...${NC}"
    echo "Refresh interval: ${INTERVAL}s"
    sleep 2
    
    while true; do
        run_monitor
        sleep $INTERVAL
    done
else
    run_monitor
fi