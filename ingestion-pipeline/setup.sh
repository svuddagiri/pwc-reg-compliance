#!/bin/bash
# Linux/Mac setup script for the regulatory document pipeline

echo "========================================"
echo "Regulatory Document Pipeline Setup"
echo "========================================"
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org"
    exit 1
fi

echo "Python found:"
python3 --version
echo

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo
    echo "Creating .env file from template..."
    cp .env.example .env
    echo
    echo "IMPORTANT: Edit the .env file with your Azure credentials before running the pipeline!"
    echo
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "Next steps:"
echo "1. Edit the .env file with your Azure credentials"
echo "2. Run the pipeline using: ./run_pipeline.sh"
echo "   Or directly: python pipeline_cli.py --help"
echo