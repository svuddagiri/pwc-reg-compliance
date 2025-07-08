#!/bin/bash
# Load environment variables from .env file
# This script properly handles comments and special characters

if [ -f .env ]; then
    # Read .env file line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
            continue
        fi
        
        # Export valid environment variables
        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            export "$line"
        fi
    done < .env
else
    echo "Warning: .env file not found"
    return 1
fi