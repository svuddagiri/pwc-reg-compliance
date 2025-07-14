#!/usr/bin/env python3
"""
Project Setup Script
Ensures all dependencies are installed and the environment is properly configured.
"""
import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_requirements():
    """Install all required packages"""
    print("📦 Installing requirements...")
    try:
        # Try to install normally first
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            # If normal install fails, try with --break-system-packages
            print("⚠️ Normal install failed, trying with --break-system-packages...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--break-system-packages"], 
                                  capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All requirements installed successfully")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_env_file():
    """Check if .env file exists"""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
        return True
    else:
        print("⚠️ .env file not found - you'll need to create one with your Azure credentials")
        print("   See .env.example for the required variables")
        return False

def test_critical_imports():
    """Test that critical modules can be imported"""
    print("🧪 Testing critical imports...")
    
    # Test basic imports
    try:
        import fastapi
        import azure.search.documents
        import openai
        import redis
        import sqlalchemy
        print("✅ Core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import core dependency: {e}")
        return False
    
    # Test context manager (with graceful fallback)
    try:
        from src.services.context_manager import ContextManager
        print("✅ Context manager can be imported")
        
        # Try to initialize it
        cm = ContextManager()
        print("✅ Context manager initialized successfully")
    except Exception as e:
        print(f"⚠️ Context manager import/init failed: {e}")
        print("   Context-aware features will be disabled")
    
    return True

def verify_database_connectivity():
    """Test database connectivity (optional)"""
    print("🔌 Testing database connectivity...")
    try:
        from src.clients.sql_manager import get_sql_client
        # This will test if the connection can be established
        sql_client = get_sql_client()
        print("✅ Database client initialized")
        return True
    except Exception as e:
        print(f"⚠️ Database connectivity test failed: {e}")
        print("   Make sure your .env file has correct SQL connection details")
        return False

def main():
    """Main setup process"""
    print("🚀 PWC Regulatory Compliance - Project Setup")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    print(f"📁 Working in: {project_dir}")
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        success = False
    
    # Step 2: Install requirements
    if not install_requirements():
        success = False
    
    # Step 3: Check .env file
    check_env_file()  # Warning only, not critical
    
    # Step 4: Test imports
    if not test_critical_imports():
        success = False
    
    # Step 5: Test database (optional)
    verify_database_connectivity()  # Warning only
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Setup completed successfully!")
        print("🎯 You can now run the chatbot with:")
        print("   python pipeline/reg_conversational_interface.py")
        print("\n💡 To test context-aware features, try:")
        print("   1. Ask: 'Describe three common errors organizations make about consent'")
        print("   2. Follow up: 'Give me couple more'")
    else:
        print("❌ Setup encountered issues")
        print("🔧 Please resolve the errors above before proceeding")
    
    return success

if __name__ == "__main__":
    exit(0 if main() else 1)