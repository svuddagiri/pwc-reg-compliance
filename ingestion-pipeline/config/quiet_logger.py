"""
Quiet logger configuration that overrides structlog
"""
import os
import logging
import structlog
import sys

class QuietLoggerAdapter:
    """A logger adapter that only shows warnings and errors"""
    def __init__(self, name):
        self.name = name
        self._verbose = os.getenv('PIPELINE_VERBOSE', 'false').lower() == 'true'
    
    def _should_log(self, level):
        """Determine if we should log at this level"""
        if self._verbose:
            return True
        return level in ['error', 'critical', 'warning']
    
    def debug(self, msg, **kwargs):
        if self._should_log('debug'):
            print(f"[DEBUG] {msg}")
    
    def info(self, msg, **kwargs):
        if self._should_log('info'):
            print(f"[INFO] {msg}")
    
    def warning(self, msg, **kwargs):
        if self._should_log('warning'):
            print(f"[WARNING] {msg}")
    
    def error(self, msg, **kwargs):
        if self._should_log('error'):
            print(f"[ERROR] {msg}: {kwargs}")
    
    def critical(self, msg, **kwargs):
        if self._should_log('critical'):
            print(f"[CRITICAL] {msg}: {kwargs}")

def get_quiet_logger(name=None):
    """Get a quiet logger that respects verbosity settings"""
    return QuietLoggerAdapter(name or __name__)

def configure_quiet_mode(verbose=False):
    """Configure the entire application for quiet mode"""
    # Set environment variable
    os.environ['PIPELINE_VERBOSE'] = 'true' if verbose else 'false'
    
    # Override structlog.get_logger
    original_get_logger = structlog.get_logger
    
    def quiet_get_logger(*args, **kwargs):
        name = args[0] if args else kwargs.get('name', 'root')
        return QuietLoggerAdapter(name)
    
    # Monkey patch structlog
    structlog.get_logger = quiet_get_logger
    
    # Also configure standard logging
    if not verbose:
        logging.basicConfig(level=logging.WARNING, force=True)
        logging.getLogger().setLevel(logging.WARNING)
        
        # Suppress all existing loggers
        for name in logging.root.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.WARNING)