"""
Setup quiet logging before any other imports
This should be imported at the very beginning of scripts
"""
import os
import sys
import logging

# Set environment variables before any imports
os.environ['STRUCTLOG_SILENT'] = 'true'
os.environ['PYTHONWARNINGS'] = 'ignore'

# Configure Python logging to WARNING level
logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger().setLevel(logging.WARNING)

# Monkey patch structlog before it's imported anywhere
import builtins
_original_import = builtins.__import__

def _quiet_import(name, *args, **kwargs):
    """Import hook that configures structlog to be quiet"""
    module = _original_import(name, *args, **kwargs)
    
    if name == 'structlog':
        # Configure structlog for quiet mode
        if hasattr(module, 'get_logger'):
            class QuietLogger:
                def __init__(self, name=None):
                    self.name = name
                    self._verbose = os.getenv('PIPELINE_VERBOSE', 'false').lower() == 'true'
                
                def _log(self, level, msg, **kwargs):
                    if self._verbose or level in ['error', 'critical', 'warning']:
                        print(f"[{level.upper()}] {msg}")
                
                def debug(self, msg, **kwargs): 
                    self._log('debug', msg, **kwargs)
                def info(self, msg, **kwargs): 
                    self._log('info', msg, **kwargs)
                def warning(self, msg, **kwargs): 
                    self._log('warning', msg, **kwargs)
                def error(self, msg, **kwargs): 
                    self._log('error', msg, **kwargs)
                def critical(self, msg, **kwargs): 
                    self._log('critical', msg, **kwargs)
                def warn(self, msg, **kwargs): 
                    self._log('warning', msg, **kwargs)
                def exception(self, msg, **kwargs): 
                    self._log('error', msg, **kwargs)
                def log(self, level, msg, **kwargs): 
                    self._log(level, msg, **kwargs)
                    
            # Override get_logger
            def quiet_get_logger(name=None, **kwargs):
                return QuietLogger(name)
            
            module.get_logger = quiet_get_logger
    
    return module

def enable_quiet_mode():
    """Enable quiet mode for all logging"""
    builtins.__import__ = _quiet_import
    os.environ['PIPELINE_VERBOSE'] = 'false'

def enable_verbose_mode():
    """Enable verbose mode for all logging"""
    os.environ['PIPELINE_VERBOSE'] = 'true'
    logging.basicConfig(level=logging.DEBUG, force=True)