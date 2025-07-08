"""
Log suppression utilities for cleaner output
"""
import os
import logging
import sys

def suppress_all_logs():
    """Suppress all logs except critical errors"""
    # Set environment variable to suppress structlog
    os.environ['STRUCTLOG_LEVEL'] = 'ERROR'
    
    # Configure standard logging to ERROR level
    logging.basicConfig(level=logging.ERROR, force=True)
    
    # Disable all loggers
    logging.disable(logging.WARNING)
    
    # Suppress specific noisy loggers
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.ERROR)
    
    # Redirect stderr to suppress any remaining output
    class NullWriter:
        def write(self, arg):
            pass
        def flush(self):
            pass
    
    # Store original stderr
    sys._stderr = sys.stderr
    
def restore_logs():
    """Restore normal logging"""
    logging.disable(logging.NOTSET)
    if hasattr(sys, '_stderr'):
        sys.stderr = sys._stderr