#!/usr/bin/env python3
"""
Run the FastAPI server with proper environment loading
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure we're in the project root
project_root = Path(__file__).parent
os.chdir(project_root)

# Load environment variables explicitly
env_path = project_root / ".env"
if env_path.exists():
    print(f"Loading environment from: {env_path}")
    load_dotenv(env_path)
    
    # Verify key variables
    if os.getenv("AZURE_OPENAI_API_KEY"):
        print("✓ Azure OpenAI API key loaded")
    else:
        print("✗ Azure OpenAI API key NOT found")
else:
    print(f"Warning: .env file not found at {env_path}")

# Now import and run uvicorn
import uvicorn
from main import app
from src.config import settings

if __name__ == "__main__":
    print(f"\nStarting server on {settings.app_host}:{settings.app_port}")
    print(f"Environment: {settings.app_env}")
    print(f"Docs available at: http://localhost:{settings.app_port}/docs")
    
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_development,
        log_level=settings.log_level.lower()
    )