"""
Logging configuration and utilities for the macroeconomic regime analysis project.

This module provides centralized logging setup with structured formatting,
different log levels, and file/console output capabilities.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_json: bool = False,
    project_name: str = "macroeconomic-regime-analysis"
) -> None:
    """
    Set up comprehensive logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path. If None, creates logs in project root.
        enable_console: Whether to enable console logging
        enable_json: Whether to use JSON formatting for logs
        project_name: Name of the project for log file naming
    """
    
    # Create logs directory if it doesn't exist
    if log_file is None:
        project_root = Path(__file__).parent.parent
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"{project_name}_{timestamp}.log"
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    
    if enable_json:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=enable_console))
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[]
    )
    
    # Add console handler if enabled
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logging.getLogger().addHandler(console_handler)
    
    # Add file handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
    
    # Log the setup completion
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging configuration completed",
        log_level=log_level,
        log_file=str(log_file) if log_file else None,
        console_enabled=enable_console,
        json_format=enable_json
    )


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name. If None, uses calling module name.
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_function_call(func):
    """
    Decorator to log function calls with arguments and return values.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function with logging
    """
    logger = get_logger(func.__module__)
    
    def wrapper(*args, **kwargs):
        logger.debug(
            "Function called",
            function=func.__name__,
            args=args,
            kwargs=kwargs
        )
        
        try:
            result = func(*args, **kwargs)
            logger.debug(
                "Function completed successfully",
                function=func.__name__,
                result_type=type(result).__name__
            )
            return result
        except Exception as e:
            logger.error(
                "Function failed with exception",
                function=func.__name__,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    return wrapper


def log_data_info(df, operation: str, logger: Optional[structlog.stdlib.BoundLogger] = None) -> None:
    """
    Log DataFrame information for debugging and monitoring.
    
    Args:
        df: DataFrame to log info about
        operation: Description of the operation being performed
        logger: Logger instance. If None, creates a new one.
    """
    if logger is None:
        logger = get_logger(__name__)
    
    logger.info(
        "DataFrame operation",
        operation=operation,
        shape=df.shape,
        columns=list(df.columns),
        dtypes=df.dtypes.to_dict(),
        null_counts=df.isnull().sum().to_dict(),
        memory_usage=df.memory_usage(deep=True).sum()
    )


def log_model_info(model, model_name: str, logger: Optional[structlog.stdlib.BoundLogger] = None) -> None:
    """
    Log model information for debugging and monitoring.
    
    Args:
        model: Trained model instance
        model_name: Name of the model
        logger: Logger instance. If None, creates a new one.
    """
    if logger is None:
        logger = get_logger(__name__)
    
    model_info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "model_module": type(model).__module__,
    }
    
    # Add model-specific information
    if hasattr(model, 'n_features_'):
        model_info["n_features"] = model.n_features_
    if hasattr(model, 'n_clusters'):
        model_info["n_clusters"] = model.n_clusters
    if hasattr(model, 'feature_importances_'):
        model_info["has_feature_importances"] = True
    if hasattr(model, 'coef_'):
        model_info["has_coefficients"] = True
    
    logger.info("Model information", **model_info)


class LogContext:
    """
    Context manager for logging operation timing and context.
    
    Example:
        with LogContext("data_processing", logger=logger):
            # Your code here
            pass
    """
    
    def __init__(self, operation: str, logger: Optional[structlog.stdlib.BoundLogger] = None, **context):
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(
            "Operation started",
            operation=self.operation,
            **self.context
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(
                "Operation completed successfully",
                operation=self.operation,
                duration_seconds=duration,
                **self.context
            )
        else:
            self.logger.error(
                "Operation failed",
                operation=self.operation,
                duration_seconds=duration,
                error_type=exc_type.__name__,
                error_message=str(exc_val),
                **self.context
            )
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Log a message within the context."""
        log_method = getattr(self.logger, level.lower())
        log_method(message, operation=self.operation, **self.context, **kwargs)


# Initialize default logging setup
def init_default_logging():
    """Initialize default logging configuration for the project."""
    setup_logging(
        log_level="INFO",
        enable_console=True,
        enable_json=False,
        project_name="macroeconomic-regime-analysis"
    )