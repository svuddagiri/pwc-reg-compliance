@echo off
REM Process Documents Script for Windows

echo === Regulatory Document Processing Pipeline ===
echo.

REM Check if .env file exists
if not exist .env (
    echo Error: .env file not found!
    echo Please create a .env file with your Azure credentials.
    exit /b 1
)

REM Load environment variables from .env
for /f "tokens=*" %%a in ('type .env ^| findstr /v "^#"') do (
    set %%a
)

REM Parse command line arguments
set BATCH_SIZE=5
set PRESET=full

:parse_args
if "%~1"=="" goto :run_pipeline
if /i "%~1"=="--batch-size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--preset" (
    set PRESET=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--help" (
    echo Usage: %0 [options]
    echo.
    echo Options:
    echo   --batch-size N      Process N documents at a time (default: 5)
    echo   --preset PRESET     Use configuration preset (default: full)
    echo   --help              Show this help message
    exit /b 0
)
shift
goto :parse_args

:run_pipeline
echo Starting document processing...
echo Batch size: %BATCH_SIZE%
echo Preset: %PRESET%
echo.

python pipeline process --preset %PRESET% --batch-size %BATCH_SIZE%

if %errorlevel% equ 0 (
    echo.
    echo Document processing completed successfully!
) else (
    echo.
    echo Document processing failed!
    exit /b 1
)