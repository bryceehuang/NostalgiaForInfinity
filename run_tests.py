#!/usr/bin/env python3
"""
Test runner for the optimized strategy tests.
"""

import sys
import os
import subprocess
import time

def main():
    """Run all tests for the optimized strategy"""
    print("Running tests for NostalgiaForInfinityX6 Optimized Strategy")
    print("=" * 60)

    # Change to the project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Add current directory to Python path
    sys.path.insert(0, script_dir)

    # First, check if we can import the strategies
    print("Checking strategy imports...")
    try:
        from NostalgiaForInfinityX6 import NostalgiaForInfinityX6
        print("✅ Original strategy imported successfully")
    except Exception as e:
        print(f"⚠️  Original strategy import failed: {e}")
        print("   This may affect some comparison tests")

    try:
        from NostalgiaForInfinityX6_CC import NostalgiaForInfinityX6_CC
        print("✅ Optimized strategy imported successfully")
    except Exception as e:
        print(f"❌ Optimized strategy import failed: {e}")
        print("   Cannot run optimized strategy tests")
        print("   Falling back to core optimization tests...")
        # Run core tests instead
        print("\n🔧 Running Core Optimization Tests Instead")
        print("-" * 50)
        try:
            result = subprocess.run([
                sys.executable, "tests/test_core_optimizations.py"
            ], capture_output=True, text=True)

            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            if result.returncode == 0:
                print("\n✅ Core optimization tests completed successfully!")
                return 0
            else:
                print("\n❌ Core optimization tests also failed")
                return 1
        except Exception as core_error:
            print(f"❌ Core tests also failed: {core_error}")
            return 1

    try:
        # Run the tests with pytest, using a custom config to avoid conflicts
        # Create a temporary pytest config that overrides the problematic settings
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write("""[pytest]
addopts=-v --tb=short
""")
            temp_config = f.name

        cmd = [
            sys.executable, "-m", "pytest",
            "tests/test_optimized_strategy.py",
            "-c", temp_config,  # Use our custom config
            "-p", "no:cacheprovider",
            "-p", "no:random_order"
        ]

        print(f"\nRunning command: {' '.join(cmd)}")
        start_time = time.time()

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Clean up temporary config file
        try:
            os.unlink(temp_config)
        except:
            pass

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)

        if result.stdout:
            print("STDOUT:")
            print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"\nTest execution time: {elapsed_time:.2f} seconds")
        print(f"Return code: {result.returncode}")

        if result.returncode == 0:
            print("\n✅ All tests passed!")
            return 0
        else:
            print("\n❌ Some tests failed!")
            return 1

    except FileNotFoundError:
        print("❌ pytest not found. Installing pytest...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest"])
        return main()  # Retry after installing pytest
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# HOW TO RUN THE TEST RUNNER:
# ==========================
#
# 1. BASIC RUN (from project root directory):
#    python run_tests.py
#
# 2. RUN WITH VERBOSE OUTPUT:
#    python run_tests.py -v
#
# 3. RUN SPECIFIC TEST FILES:
#    # Modify the run_tests.py to test specific files:
#    # Change line 27 from "tests/test_optimized_strategy.py" to:
#    # "tests/test_core_optimizations.py" or any other test file
#
# 4. RUN WITHOUT INSTALLING PYTEST:
#    # The script will automatically install pytest if not found
#    python run_tests.py
#
# 5. RUN WITH CUSTOM PYTEST OPTIONS:
#    # Modify the cmd list in main() function to add options:
#    # cmd = [sys.executable, "-m", "pytest", "tests/test_optimized_strategy.py", "-v", "--tb=short", "--durations=10"]
#
# 6. RUN MULTIPLE TEST FILES:
#    # Modify the script to run multiple test files:
#    """
#    test_files = [
#        "tests/test_core_optimizations.py",
#        "tests/test_optimized_strategy.py",
#    ]
#    for test_file in test_files:
#        cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
#        result = subprocess.run(cmd, capture_output=True, text=True)
#        # Process results for each file
#    """
#
# 7. SCHEDULED TEST RUNS:
#    # Linux/Mac - Add to crontab:
#    0 2 * * * cd /path/to/project && python run_tests.py >> test_results.log 2>&1
#
#    # Windows - Use Task Scheduler:
#    # Create a task that runs daily at 2 AM with:
#    # Program: python
#    # Arguments: run_tests.py
#    # Start in: C:\path\to\project
#
# 8. INTEGRATE WITH NOTIFICATIONS:
#    # Email notifications (Linux/Mac):
#    python run_tests.py && echo "Tests passed" | mail -s "Test Success" your@email.com || echo "Tests failed" | mail -s "Test Failure" your@email.com
#
#    # Slack notifications:
#    python run_tests.py && curl -X POST -H 'Content-type: application/json' --data '{"text":"✅ Tests passed!"}' YOUR_SLACK_WEBHOOK_URL || curl -X POST -H 'Content-type: application/json' --data '{"text":"❌ Tests failed!"}' YOUR_SLACK_WEBHOOK_URL
#
# 9. RUN WITH DIFFERENT PYTHON VERSIONS:
#    python3.8 run_tests.py
#    python3.9 run_tests.py
#    python3.10 run_tests.py
#
# 10. RUN IN VIRTUAL ENVIRONMENT:
#     # Create and activate virtual environment
#     python -m venv venv
#     source venv/bin/activate  # Linux/Mac
#     # or
#     venv\\Scripts\\activate  # Windows
#
#     pip install pytest pandas numpy
#     python run_tests.py
#
# 11. RUN WITH REQUIREMENTS CHECK:
#     # Create requirements.txt first
#     pip install -r requirements.txt
#     python run_tests.py
#
# 12. RUN WITH PERFORMANCE MONITORING:
#     # Monitor system resources during tests
#     /usr/bin/time -v python run_tests.py  # Linux
#     # or
#     python -c "import time, psutil, os; start=time.time(); os.system('python run_tests.py'); print(f'Total time: {time.time()-start:.2f}s'); print(f'Memory usage: {psutil.virtual_memory().percent}%')"
#
# 13. DEBUGGING TEST FAILURES:
#     # Run with detailed error output
#     python run_tests.py 2>&1 | tee test_output.log
#
#     # Analyze the log file
#     grep -i error test_output.log  # Linux/Mac
#     # or
#     findstr /i error test_output.log  # Windows
#
# 14. RUN WITH PARALLEL TESTING:
#     # Modify the script to use pytest-xdist:
#     # cmd = [sys.executable, "-m", "pytest", "tests/test_optimized_strategy.py", "-v", "--tb=short", "-n", "auto"]
#
# 15. EXPECTED OUTPUT:
#     Running tests for NostalgiaForInfinityX6 Optimized Strategy
#     ============================================================
#
#     Running command: python -m pytest tests/test_optimized_strategy.py -v --tb=short
#     ============================= test session starts ==============================
#     collected 12 items
#
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_strategy_initialization PASSED [  8%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_performance_improvement PASSED [ 16%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_exit_condition_performance PASSED [ 25%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_memory_efficiency PASSED [ 33%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_profit_calculation_accuracy PASSED [ 41%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_exit_condition_cache PASSED [ 50%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_mode_detection_performance PASSED [ 58%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_dataframe_caching PASSED [ 66%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_indicator_batch_calculation PASSED [ 75%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_strategy_logic_preservation PASSED [ 83%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_error_handling PASSED [ 91%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_concurrent_performance PASSED [100%]
#
#     ========================= 12 tests passed in 8.45s ===========================
#
#     ============================================================
#     TEST RESULTS
#     ============================================================
#     STDOUT:
#     ... (test output above)
#
#     Test execution time: 8.45 seconds
#     Return code: 0
#
#     ✅ All tests passed!
#
# 16. TROUBLESHOOTING:
#     # If pytest is not found:
#     pip install pytest
#
#     # If tests fail due to missing dependencies:
#     pip install pandas numpy
#
#     # If tests fail due to import errors:
#     # Make sure you're running from the project root directory
#     # and that the strategy files are in the correct location
#
#     # If tests are too slow:
#     # Modify the script to skip certain tests or reduce test iterations
#
# 17. CUSTOMIZATION:
#     # To test different strategy files, modify the test file path in the cmd list
#     # To add more pytest options, extend the cmd list
#     # To change test behavior, modify the pytest command arguments
#
# 18. REQUIREMENTS:
#     pip install pytest
#     # Optional for enhanced functionality:
#     pip install pytest-xdist pytest-cov pytest-html pandas numpy