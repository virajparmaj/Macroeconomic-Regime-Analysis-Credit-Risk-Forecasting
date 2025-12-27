#!/usr/bin/env python3
"""
Setup script for Macroeconomic Regime Analysis and Credit Risk Modeling.

This script initializes the project environment, sets up logging,
creates necessary directories, and validates the installation.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def setup_project_environment():
    """Set up the project environment with logging and directories."""
    print("=" * 60)
    print("MACROECONOMIC REGIME ANALYSIS - PROJECT SETUP")
    print("=" * 60)
    
    # Import and initialize logging
    try:
        from utilities.logging_config import init_default_logging, setup_logging, get_logger
        
        # Initialize logging
        init_default_logging()
        logger = get_logger(__name__)
        
        print("✓ Logging system initialized")
        logger.info("Project setup started")
        
    except ImportError as e:
        print(f"✗ Failed to initialize logging: {e}")
        print("  Please install requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"✗ Unexpected error during logging setup: {e}")
        return False
    
    # Create necessary directories
    directories = [
        "data/original",
        "models/checkpoints",
        "models/final", 
        "models/deep",
        "logs",
        "outputs/plots",
        "outputs/reports"
    ]
    
    print("\nCreating project directories...")
    for dir_path in directories:
        full_path = project_root / dir_path
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {dir_path}")
            logger.info("Directory created/verified", directory=str(full_path))
        except Exception as e:
            print(f"✗ Failed to create {dir_path}: {e}")
            logger.error("Directory creation failed", directory=str(full_path), error=str(e))
    
    # Validate configuration
    print("\nValidating configuration...")
    try:
        from config import get_all_config, validate_paths
        
        config = get_all_config()
        paths_valid = validate_paths()
        
        if paths_valid:
            print("✓ Configuration validated successfully")
            logger.info("Configuration validation successful", config_keys=list(config.keys()))
        else:
            print("⚠ Configuration validation had issues")
            logger.warning("Configuration validation had issues")
            
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        logger.error("Configuration validation failed", error=str(e))
        return False
    
    # Test core imports
    print("\nTesting core imports...")
    test_imports = [
        ("Data processing", "utilities.data_processing"),
        ("Core functions", "utilities.functions"),
        ("Model utilities", "utilities.model_utils"),
    ]
    
    all_imports_ok = True
    for module_name, import_path in test_imports:
        try:
            __import__(import_path)
            print(f"✓ {module_name}")
            logger.info("Import test successful", module=import_path)
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            logger.warning("Import test failed", module=import_path, error=str(e))
            all_imports_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_imports_ok:
        print("✓ PROJECT SETUP COMPLETED SUCCESSFULLY")
        print("\nNext steps:")
        print("1. Place your data files in the 'data/original/' directory")
        print("2. Run the analysis notebooks in the 'notebooks/' directory")
        print("3. Check the 'logs/' directory for detailed logging output")
    else:
        print("⚠ PROJECT SETUP COMPLETED WITH WARNINGS")
        print("\nTo resolve issues:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check the logs directory for detailed error information")
    
    logger.info("Project setup completed", all_imports_ok=all_imports_ok)
    print("=" * 60)
    
    return all_imports_ok


if __name__ == "__main__":
    success = setup_project_environment()
    sys.exit(0 if success else 1)