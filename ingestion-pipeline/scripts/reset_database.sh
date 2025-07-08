#!/bin/bash

# Reset Database Script for Regulatory Chat Bot
# This script resets the SQL Server database for the regulatory chatbot

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Regulatory Chat Bot Database Reset ===${NC}"
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

# Confirm reset
echo -e "${YELLOW}Warning: This will reset the entire database!${NC}"
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Database reset cancelled."
    exit 0
fi

# Run the reset script
echo -e "${GREEN}Starting database reset...${NC}"
./pipeline reset --group all --force

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Database reset completed successfully!${NC}"
else
    echo -e "${RED}Database reset failed!${NC}"
    exit 1
fi