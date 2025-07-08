#!/bin/bash
# Linux/Mac script for running the regulatory document pipeline

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if dependencies are installed
python -c "import rich" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Run the pipeline CLI with any provided arguments
python pipeline_cli.py "$@"