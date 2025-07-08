"""
Logging configuration for the pipeline
"""
import logging
import os
import sys
from pathlib import Path
import structlog

def configure_logging(verbose=False, log_file=None):
    """
    Configure logging for the pipeline with different verbosity levels
    
    Args:
        verbose: If True, show all logs. If False, show only warnings and errors
        log_file: Optional log file path to write logs to
    """
    # Determine log level based on environment or parameter
    if verbose or os.getenv('PIPELINE_DEBUG', '').lower() == 'true':
        log_level = logging.DEBUG
    else:
        # Default to WARNING to suppress info logs
        log_level = logging.WARNING
    
    # Basic configuration
    log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    handlers = []
    
    # Console handler with appropriate level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,  # Capture everything, filter at handler level
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Suppress noisy third-party loggers
    noisy_loggers = [
        'azure.core.pipeline.policies.http_logging_policy',
        'azure.storage.blob',
        'azure.ai.formrecognizer',
        'azure.search.documents',
        'openai',
        'httpx',
        'httpcore',
        'urllib3',
        'requests',
        'aiohttp',
        'asyncio',
        'concurrent.futures',
        'PIL',
        'matplotlib',
        'chardet'
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING if not verbose else logging.INFO)
    
    # Specific adjustments for our modules
    if not verbose:
        # Suppress info logs from our modules when not in verbose mode
        logging.getLogger('src.ingestion.enhanced_document_processor').setLevel(logging.WARNING)
        logging.getLogger('src.ingestion.metadata_extractor').setLevel(logging.WARNING)
        logging.getLogger('src.ingestion.advanced_clause_chunker').setLevel(logging.WARNING)
        logging.getLogger('src.ingestion.enhanced_pipeline_orchestrator').setLevel(logging.WARNING)
        logging.getLogger('src.storage.azure_search').setLevel(logging.WARNING)
        logging.getLogger('__main__').setLevel(logging.WARNING)
    
    # Configure structlog to respect Python logging levels
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name):
    """Get a structlog logger with the given name"""
    return structlog.get_logger(name)