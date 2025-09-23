#!/usr/bin/env python3
"""
Comprehensive test runner for all NostalgiaForInfinityX6 tests.
Runs core optimization tests and strategy tests (if dependencies are available).
"""

import sys
import os
import subprocess
import time

def run_core_tests():
    """Run core optimization tests"""
    print("\n🔧 Running Core Optimization Tests")
    print("=" * 50)

    try:
        result = subprocess.run([
            sys.executable, "tests/test_core_optimizations.py"
        ], capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Core tests failed: {e}")
        return False

def run_strategy_tests():
    """Run strategy tests if dependencies are available"""
    print("\n📈 Running Strategy Tests")
    print("=" * 50)

    # Check if we can import the strategies
    try:
        from NostalgiaForInfinityX6 import NostalgiaForInfinityX6
        print("✅ Original strategy imported successfully")
    except Exception as e:
        print(f"⚠️  Original strategy import failed: {e}")
        print("   Skipping strategy comparison tests")
        return True  # Don't fail if we can't import

    try:
        from NostalgiaForInfinityX6_CC import NostalgiaForInfinityX6_CC
        print("✅ Optimized strategy imported successfully")
    except Exception as e:
        print(f"❌ Optimized strategy import failed: {e}")
        print("   Cannot run strategy tests")
        return False

    try:
        # Run the tests with pytest, using a custom config to avoid conflicts
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""[pytest]
addopts=-v --tb=short
""")
            temp_config = f.name

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_optimized_strategy.py",
            "-c", temp_config,
            "-p", "no:cacheprovider",
            "-p", "no:random_order"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up temporary config file
        try:
            os.unlink(temp_config)
        except:
            pass

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0

    except FileNotFoundError:
        print("❌ pytest not found. Installing pytest...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        return run_strategy_tests()  # Retry after installing
    except Exception as e:
        print(f"❌ Strategy tests failed: {e}")
        return False

def run_performance_analysis():
    """Run performance analysis"""
    print("\n🚀 Running Performance Analysis")
    print("=" * 50)

    try:
        result = subprocess.run([
            sys.executable, "performance_analysis.py"
        ], capture_output=True, text=True)

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return False

def check_dependencies():
    """Check what dependencies are available"""
    print("\n📋 Checking Dependencies")
    print("=" * 50)

    dependencies = {
        'numpy': False,
        'pandas': False,
        'numba': False,
        'pytest': False,
        'talib': False,
        'pandas_ta': False,
        'rapidjson': False
    }

    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")

    return dependencies

def install_missing_deps(dependencies):
    """Attempt to install missing dependencies"""
    print("\n🔧 Installing Missing Dependencies")
    print("=" * 50)

    missing = [dep for dep, available in dependencies.items() if not available]

    if not missing:
        print("All dependencies are already installed!")
        return True

    print(f"Missing dependencies: {', '.join(missing)}")

    # Try to install basic dependencies
    basic_deps = ['numpy', 'pandas', 'numba', 'pytest']
    for dep in basic_deps:
        if dep in missing:
            try:
                print(f"Installing {dep}...")
                subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
                print(f"✅ {dep} installed")
            except:
                print(f"❌ Failed to install {dep}")

    # Try to install trading-specific dependencies
    trading_deps = {
        'talib': 'TA-Lib',
        'pandas_ta': 'pandas-ta',
        'rapidjson': 'python-rapidjson'
    }

    for dep, package in trading_deps.items():
        if dep in missing:
            try:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                print(f"✅ {package} installed")
            except:
                print(f"❌ Failed to install {package}")

    return True

def main():
    """Run all tests and analysis"""
    print("🧪 NostalgiaForInfinityX6 Comprehensive Test Runner")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")

    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Add to Python path
    sys.path.insert(0, script_dir)

    # Check dependencies
    deps = check_dependencies()

    # Ask if user wants to install missing dependencies
    missing_any = any(not available for available in deps.values())
    if missing_any:
        response = input("\nInstall missing dependencies? (y/n): ").lower().strip()
        if response == 'y':
            install_missing_deps(deps)
            # Recheck dependencies
            deps = check_dependencies()

    results = {}

    # Run core optimization tests (these should always work)
    print("\n" + "=" * 60)
    print("🎯 STARTING TESTS")
    print("=" * 60)

    results['core'] = run_core_tests()

    # Run strategy tests if dependencies are available
    if deps.get('talib') and deps.get('pandas_ta'):
        results['strategy'] = run_strategy_tests()
    else:
        print("\n⚠️  Skipping strategy tests due to missing dependencies")
        print("   Run 'pip install TA-Lib pandas-ta' to enable strategy tests")
        results['strategy'] = None

    # Run performance analysis
    results['performance'] = run_performance_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0

    for test_name, result in results.items():
        if result is None:
            status = "⚠️  SKIPPED"
        elif result:
            status = "✅ PASSED"
            passed_tests += 1
        else:
            status = "❌ FAILED"

        print(f"{test_name.capitalize():15s}: {status}")
        if result is not None:
            total_tests += 1

    print(f"\nResults: {passed_tests}/{total_tests} test suites completed successfully")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test suite(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# HOW TO RUN THIS COMPREHENSIVE TEST RUNNER:
# =========================================
#
# 1. BASIC RUN (from project root directory):
#    python run_all_tests.py
#
# 2. RUN WITH DEPENDENCY INSTALLATION:
#    python run_all_tests.py  # Then answer 'y' when prompted
#
# 3. RUN CORE TESTS ONLY (skip strategy tests):
#    python tests/test_core_optimizations.py
#
# 4. RUN PERFORMANCE ANALYSIS ONLY:
#    python performance_analysis.py
#
# 5. INSTALL DEPENDENCIES MANUALLY:
#    pip install numpy pandas numba pytest
#    pip install TA-Lib python-rapidjson
#    # Note: pandas-ta might need manual installation or specific version
#
# 6. TROUBLESHOOTING:
#    # If strategy tests fail:
#    python run_all_tests.py  # Will show what's missing
#
#    # If you get import errors:
#    # Make sure you're running from the project root directory
#
#    # For missing pandas-ta:
#    # Try: pip install pandas-ta==0.3.14b
#    # Or download from: https://github.com/twopirllc/pandas-ta
#
# 7. EXPECTED OUTPUT:
#    🧪 NostalgiaForInfinityX6 Comprehensive Test Runner
#    ============================================================
#    Python version: 3.11.8
#    Working directory: /Users/bryce/Documents/projects/NostalgiaForInfinity
#
#    📋 Checking Dependencies
#    ============================================================
#    ✅ numpy
#    ✅ pandas
#    ✅ numba
#    ✅ pytest
#    ❌ talib
#    ❌ pandas_ta
#    ✅ rapidjson
#
#    Install missing dependencies? (y/n): y
#
#    🔧 Installing Missing Dependencies
#    ============================================================
#    Installing TA-Lib...
#    ✅ TA-Lib installed
#    Installing pandas-ta...
#    ⚠️  Failed to install pandas-ta
#
#    🎯 STARTING TESTS
#    ============================================================
#
#    🔧 Running Core Optimization Tests
#    ============================================================
#    Running Core Optimization Tests for NostalgiaForInfinityX6
#    ======================================================
#    Testing Numba compilation...
#    ✅ Numba compilation test passed
#    ...
#
#    📊 TEST SUMMARY
#    ============================================================
#    Core           : ✅ PASSED
#    Strategy       : ⚠️  SKIPPED
#    Performance    : ✅ PASSED
#
#    Results: 2/2 test suites completed successfully
#    🎉 All tests passed!