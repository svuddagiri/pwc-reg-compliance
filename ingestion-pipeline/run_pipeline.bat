@echo off
REM Windows batch file for running the regulatory document pipeline

REM Check if virtual environment exists
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if dependencies are installed
python -c "import rich" 2>nul
IF ERRORLEVEL 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
)

REM Run the pipeline CLI with any provided arguments
python pipeline_cli.py %*

REM Deactivate virtual environment
deactivate