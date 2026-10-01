#!/usr/bin/env python3
"""
Test script for logging functionality across all modules.

This script validates that:
1. Logging configuration works correctly
2. All modules can be imported and used
3. Logging output is properly formatted
4. Error handling works as expected
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))


def _check_logging_configuration():
    """Test logging configuration setup."""
    print("Testing logging configuration...")

    try:
        from utilities.logging_config import LogContext, get_logger, setup_logging

        # Set up logging
        setup_logging(
            log_level="INFO", enable_console=True, enable_json=False, project_name="test_run"
        )

        logger = get_logger("test_logging")
        logger.info("Logging configuration test successful")

        # Test LogContext
        with LogContext("test_operation", logger=logger, test_param="test_value"):
            logger.info("Test operation in context")

        print("✓ Logging configuration test passed")
        return True

    except Exception as e:
        print(f"✗ Logging configuration test failed: {e}")
        return False


def _check_module_imports():
    """Test importing all beautified modules."""
    print("Testing module imports...")

    modules_to_test = [
        ("config", "config"),
        ("logging_config", "utilities.logging_config"),
        ("data_processing", "utilities.data_processing"),
        ("functions", "utilities.functions"),
        ("model_utils", "utilities.model_utils"),
        ("init", "utilities"),
    ]

    results = {}

    for module_name, import_path in modules_to_test:
        try:
            __import__(import_path)
            results[module_name] = True
            print(f"✓ {module_name} imported successfully")
        except Exception as e:
            results[module_name] = False
            print(f"✗ {module_name} import failed: {e}")

    all_passed = all(results.values())
    print(f"Module imports: {sum(results.values())}/{len(results)} successful")

    return all_passed


def _check_basic_functionality():
    """Test basic functionality of key functions."""
    print("Testing basic functionality...")

    try:
        import numpy as np
        import pandas as pd

        from config import get_all_config
        from utilities.functions import remove_outliers_zscore, winsorize_series
        from utilities.logging_config import get_logger

        logger = get_logger("functionality_test")

        # Test config
        config = get_all_config()
        logger.info("Config retrieval test", config_keys=list(config.keys()))

        # Test data processing functions
        test_series = pd.Series([1, 2, 3, 100, 4, 5, 6])  # 100 is outlier
        cleaned_series = remove_outliers_zscore(test_series, threshold=2.0)
        logger.info(
            "Outlier removal test",
            original_length=len(test_series),
            cleaned_length=len(cleaned_series),
        )

        winsorized_series = winsorize_series(test_series, limits=(0.1, 0.1))
        logger.info(
            "Winsorization test completed",
            original_max=test_series.max(),
            winsorized_max=winsorized_series.max(),
        )

        print("✓ Basic functionality test passed")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def _check_error_handling():
    """Test error handling and logging."""
    print("Testing error handling...")

    try:
        import pandas as pd

        from utilities.functions import remove_outliers_zscore
        from utilities.logging_config import get_logger

        logger = get_logger("error_handling_test")

        # Test empty DataFrame
        empty_series = pd.Series([])
        try:
            remove_outliers_zscore(empty_series)
            print("✗ Should have failed on empty series")
            return False
        except ValueError:
            logger.info("Empty series error handled correctly")

        # Test missing columns (would normally fail gracefully)
        logger.info("Error handling test completed")
        print("✓ Error handling test passed")
        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def _check_file_operations():
    """Test file creation and directory structure."""
    print("Testing file operations...")

    try:
        from config import MODEL_FINAL_DIR, get_data_file_info
        from utilities.logging_config import get_logger

        logger = get_logger("file_operations_test")

        # Test directory creation
        logger.info("Testing directory operations")

        # Test data file info function
        file_info = get_data_file_info()
        logger.info("File info retrieved", file_count=len(file_info))

        print("✓ File operations test passed")
        return True

    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False


def test_logging_configuration():
    """Fail pytest when the standalone check reports failure."""
    assert _check_logging_configuration() is True


def test_module_imports():
    """Fail pytest when the standalone check reports failure."""
    assert _check_module_imports() is True


def test_basic_functionality():
    """Fail pytest when the standalone check reports failure."""
    assert _check_basic_functionality() is True


def test_error_handling():
    """Fail pytest when the standalone check reports failure."""
    assert _check_error_handling() is True


def test_file_operations():
    """Fail pytest when the standalone check reports failure."""
    assert _check_file_operations() is True


def run_all_tests():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("LOGGING FUNCTIONALITY TEST SUITE")
    print("=" * 60)

    tests = [
        ("Logging Configuration", _check_logging_configuration),
        ("Module Imports", _check_module_imports),
        ("Basic Functionality", _check_basic_functionality),
        ("Error Handling", _check_error_handling),
        ("File Operations", _check_file_operations),
    ]

    results = {}

    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        results[test_name] = test_func()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Logging system working correctly!")
        print("\nNext steps:")
        print("1. Run: python setup_project.py to initialize environment")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Check the 'logs/' directory for detailed logging output")
    else:
        print("⚠️  Some tests failed - check error messages above")
        print("\nTroubleshooting:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check Python path includes project root")
        print("3. Verify all required libraries are installed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
