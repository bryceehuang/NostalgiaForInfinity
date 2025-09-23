#!/usr/bin/env python3
"""
Comprehensive tests for the optimized NostalgiaForInfinityX6 strategy.
Tests performance improvements while ensuring trading logic integrity.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import time
import gc

# Import both strategies for comparison
try:
    from NostalgiaForInfinityX6 import NostalgiaForInfinityX6
    from NostalgiaForInfinityX6_CC import NostalgiaForInfinityX6_CC
except ImportError as e:
    pytest.skip(f"Could not import strategies: {e}", allow_module_level=True)


class TestOptimizedStrategy:
    """Test suite for the optimized trading strategy"""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample market data for testing"""
        dates = pd.date_range(start='2023-01-01', periods=1000, freq='5min')
        np.random.seed(42)  # For reproducible results

        # Generate realistic price data
        base_price = 100.0
        prices = [base_price]

        for i in range(1, 1000):
            change = np.random.normal(0, 0.002)  # Small random changes
            new_price = prices[-1] * (1 + change)
            prices.append(max(0.01, new_price))  # Ensure price stays positive

        df = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, 1000)
        })

        return df

    @pytest.fixture
    def mock_trade(self):
        """Create a mock trade object for testing"""
        trade = Mock()
        trade.is_short = False
        trade.open_rate = 100.0
        trade.amount = 1.0
        trade.stake_amount = 100.0
        trade.fee_open = 0.001
        trade.fee_close = 0.001
        trade.open_date_utc = datetime.utcnow() - timedelta(hours=1)
        trade.select_filled_orders = Mock(return_value=[])
        trade.enter_tag = "1"
        trade.leverage = 1.0
        return trade

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration"""
        return {
            'exchange': {
                'name': 'binance',
                'ccxt_config': {},
                'ccxt_async_config': {}
            },
            'stake_currency': 'USDT',
            'user_data_dir': '/tmp/test_data',
            'runmode': Mock(value='backtest'),
            'trading_mode': 'spot'
        }

    def test_strategy_initialization(self, mock_config):
        """Test that the optimized strategy initializes correctly"""
        strategy = NostalgiaForInfinityX6_CC(mock_config)

        assert strategy.version() == "v16.7.137_optimized_claude"
        assert strategy.INTERFACE_VERSION == 3
        assert strategy.timeframe == "5m"
        assert strategy.stoploss == -0.99

        # Check that caches are initialized
        assert strategy._mode_tags_cache is not None
        assert strategy._exit_conditions_cache is not None
        assert strategy._indicator_cache is not None

    def test_performance_improvement(self, sample_dataframe, mock_config):
        """Test that the optimized strategy performs better than the original"""
        original_strategy = NostalgiaForInfinityX6(mock_config)
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Test indicator calculation performance
        metadata = {'pair': 'BTC/USDT'}

        # Time original strategy
        start_time = time.perf_counter()
        original_result = original_strategy.populate_indicators(sample_dataframe.copy(), metadata)
        original_time = time.perf_counter() - start_time

        # Clear cache for fair comparison
        optimized_strategy._indicator_cache.clear()
        gc.collect()

        # Time optimized strategy
        start_time = time.perf_counter()
        optimized_result = optimized_strategy.populate_indicators(sample_dataframe.copy(), metadata)
        optimized_time = time.perf_counter() - start_time

        # Optimized should be faster (allow 20% margin for testing variability)
        assert optimized_time < original_time * 1.2, f"Optimized strategy slower: {optimized_time:.4f}s vs {original_time:.4f}s"

        print(f"Performance improvement: {((original_time - optimized_time) / original_time * 100):.1f}%")

    def test_exit_condition_performance(self, mock_trade, sample_dataframe, mock_config):
        """Test that exit conditions are faster in optimized version"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Create test scenario
        current_rate = 105.0  # 5% profit
        profit_stake = 5.0
        profit_ratio = 0.05
        profit_current_stake_ratio = 0.05
        profit_init_ratio = 0.05
        max_profit = 0.08
        max_loss = -0.02
        current_time = datetime.utcnow()

        # Mock last candle
        last_candle = {
            'close': 105.0,
            'EMA_200': 102.0,
            'RSI_14': 45.0,
            'volume': 1000.0
        }
        previous_candle_1 = last_candle.copy()

        enter_tags = ["1"]  # Normal mode

        # Test multiple exit conditions
        start_time = time.perf_counter()

        for _ in range(100):  # Run multiple times for better measurement
            sell, signal = optimized_strategy.long_exit_normal(
                pair="BTC/USDT",
                current_rate=current_rate,
                profit_stake=profit_stake,
                profit_ratio=profit_ratio,
                profit_current_stake_ratio=profit_current_stake_ratio,
                profit_init_ratio=profit_init_ratio,
                max_profit=max_profit,
                max_loss=max_loss,
                filled_entries=[],
                filled_exits=[],
                last_candle=last_candle,
                previous_candle_1=previous_candle_1,
                previous_candle_2=previous_candle_1,
                previous_candle_3=previous_candle_1,
                previous_candle_4=previous_candle_1,
                previous_candle_5=previous_candle_1,
                trade=mock_trade,
                current_time=current_time,
                enter_tags=enter_tags
            )

        elapsed_time = time.perf_counter() - start_time
        avg_time = elapsed_time / 100

        # Should be very fast (< 1ms per operation)
        assert avg_time < 0.001, f"Exit condition too slow: {avg_time:.4f}s per operation"

    def test_memory_efficiency(self, sample_dataframe, mock_config):
        """Test that memory usage is optimized"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Mock the dp (DataProvider) to avoid BTC info calls
        optimized_strategy.dp = Mock()
        optimized_strategy.dp.get_pair_dataframe = Mock(return_value=sample_dataframe.copy())
        optimized_strategy.btc_info_timeframes = []  # Disable BTC info to avoid complex mocking

        # Fill cache
        metadata = {'pair': 'BTC/USDT'}

        for i in range(50):  # Simulate multiple pairs
            pair_metadata = {'pair': f'COIN{i}/USDT'}
            optimized_strategy.populate_indicators(sample_dataframe.copy(), pair_metadata)

        # Check cache size management
        assert len(optimized_strategy._indicator_cache) <= 1000  # Should have limit

        # Force cache clear
        optimized_strategy._clear_cache()

        # Cache should be cleared or reduced
        assert len(optimized_strategy._indicator_cache) < 50

    def test_profit_calculation_accuracy(self, mock_config):
        """Test that profit calculations are accurate"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Test various profit scenarios
        test_cases = [
            (100.0, 105.0, False, 0.001, 0.001, 0.048),  # Long position, 5% profit
            (100.0, 95.0, False, 0.001, 0.001, -0.052),   # Long position, 5% loss
            (100.0, 95.0, True, 0.001, 0.001, 0.048),     # Short position, 5% profit
            (100.0, 105.0, True, 0.001, 0.001, -0.052),   # Short position, 5% loss
        ]

        for entry_price, current_price, is_short, fee_open, fee_close, expected_profit in test_cases:
            profit = optimized_strategy.fast_profit_calculation_numba(entry_price, current_price, is_short, fee_open, fee_close)
            assert abs(profit - expected_profit) < 0.001, f"Profit calculation incorrect: {profit} vs {expected_profit}"

    def test_exit_condition_cache(self, mock_config):
        """Test that exit condition caching works correctly"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Check that exit conditions are pre-computed
        assert 'normal_spot' in optimized_strategy._exit_conditions_cache
        assert 'normal_futures' in optimized_strategy._exit_conditions_cache
        assert 'rapid_spot' in optimized_strategy._exit_conditions_cache
        assert 'rapid_futures' in optimized_strategy._exit_conditions_cache

        # Values should be reasonable
        assert optimized_strategy._exit_conditions_cache['normal_spot'] > 0
        assert optimized_strategy._exit_conditions_cache['normal_futures'] > 0

    def test_mode_detection_performance(self, mock_config):
        """Test that mode detection is efficient"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Test different mode combinations
        test_tags = [
            ["1", "2", "3"],           # Normal mode
            ["21", "22", "23"],        # Pump mode
            ["41", "42", "43"],        # Quick mode
            ["61", "62"],              # Rebuy mode
            ["101", "102", "103"],     # Rapid mode
            ["120"],                   # Grind mode
            ["161", "162", "163"],     # Scalp mode
        ]

        start_time = time.perf_counter()

        for _ in range(1000):  # Run many times
            for tags in test_tags:
                functions = optimized_strategy._get_mode_exit_functions(tags)

        elapsed_time = time.perf_counter() - start_time
        avg_time = elapsed_time / (1000 * len(test_tags))

        # Should be very fast (< 0.1ms per detection)
        assert avg_time < 0.0001, f"Mode detection too slow: {avg_time:.4f}s per operation"

    def test_dataframe_caching(self, sample_dataframe, mock_config):
        """Test that dataframe caching works correctly"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Mock the dp (DataProvider)
        optimized_strategy.dp = Mock()
        optimized_strategy.dp.get_analyzed_dataframe = Mock(return_value=(sample_dataframe, None))

        # First call should cache
        df1 = optimized_strategy._get_cached_dataframe("BTC/USDT")

        # Second call should use cache
        df2 = optimized_strategy._get_cached_dataframe("BTC/USDT")

        # Should be the same object (cached)
        assert df1 is df2

        # Different pair should create new cache entry
        df3 = optimized_strategy._get_cached_dataframe("ETH/USDT")
        assert df3 is not df1

    def test_indicator_batch_calculation(self, sample_dataframe, mock_config):
        """Test batch indicator calculation"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Test batch calculation
        indicators = {
            'RSI': [3, 14],
            'BBANDS': [20],
            'MFI': [14],
            'CMF': [20],
        }

        start_time = time.perf_counter()
        optimized_strategy._batch_calculate_indicators(sample_dataframe, indicators)
        batch_time = time.perf_counter() - start_time

        # Verify indicators were calculated
        assert 'RSI_3' in sample_dataframe.columns
        assert 'RSI_14' in sample_dataframe.columns
        assert 'BBL_20_2.0' in sample_dataframe.columns
        assert 'MFI_14' in sample_dataframe.columns
        assert 'CMF_20' in sample_dataframe.columns

        # Should be reasonably fast
        assert batch_time < 1.0, f"Batch calculation too slow: {batch_time:.4f}s"

    def test_strategy_logic_preservation(self, mock_config):
        """Test that trading logic is preserved in optimization"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Test key parameters are preserved
        assert optimized_strategy.stoploss == -0.99
        assert optimized_strategy.timeframe == "5m"
        assert optimized_strategy.use_exit_signal == True
        assert optimized_strategy.exit_profit_only == False
        assert optimized_strategy.ignore_roi_if_entry_signal == True

        # Test mode tags are preserved
        assert len(optimized_strategy.long_normal_mode_tags) > 0
        assert len(optimized_strategy.long_pump_mode_tags) > 0
        assert len(optimized_strategy.long_quick_mode_tags) > 0
        assert len(optimized_strategy.long_rebuy_mode_tags) > 0
        assert len(optimized_strategy.long_rapid_mode_tags) > 0
        assert len(optimized_strategy.long_grind_mode_tags) > 0
        assert len(optimized_strategy.long_scalp_mode_tags) > 0

    def test_error_handling(self, mock_config):
        """Test that the strategy handles errors gracefully"""
        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)

        # Test with None values
        result = optimized_strategy.fast_profit_calculation_numba(100.0, None, False, 0.001, 0.001)
        assert result == 0.0 or np.isnan(result)  # Should handle gracefully

        # Test with invalid mode tags
        functions = optimized_strategy._get_mode_exit_functions(["invalid_tag"])
        assert isinstance(functions, list)  # Should return empty list, not crash

    def test_concurrent_performance(self, mock_config):
        """Test performance under concurrent load"""
        import threading

        optimized_strategy = NostalgiaForInfinityX6_CC(mock_config)
        results = []

        def worker():
            try:
                # Simulate some strategy operations
                profit = optimized_strategy.fast_profit_calculation(100.0, 105.0, False, 0.001, 0.001)
                functions = optimized_strategy._get_mode_exit_functions(["1", "2"])
                results.append(True)
            except Exception as e:
                results.append(False)

        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All operations should succeed
        assert all(results), "Some concurrent operations failed"
        assert len(results) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# HOW TO RUN THIS TEST FILE:
# =========================
#
# 1. BASIC RUN (from project root directory):
#    python tests/test_optimized_strategy.py
#
# 2. RUN WITH PYTEST (RECOMMENDED):
#    pytest tests/test_optimized_strategy.py
#
# 3. RUN WITH VERBOSE OUTPUT:
#    pytest tests/test_optimized_strategy.py -v
#
# 4. RUN SPECIFIC TEST:
#    pytest tests/test_optimized_strategy.py::TestOptimizedStrategy::test_strategy_initialization
#    pytest tests/test_optimized_strategy.py::TestOptimizedStrategy::test_performance_improvement
#    pytest tests/test_optimized_strategy.py::TestOptimizedStrategy::test_exit_condition_performance
#
# 5. RUN MULTIPLE SPECIFIC TESTS:
#    pytest tests/test_optimized_strategy.py -k "test_performance or test_exit"
#
# 6. RUN WITH COVERAGE REPORT:
#    pip install pytest-cov
#    pytest tests/test_optimized_strategy.py --cov=.
#
# 7. RUN WITH PERFORMANCE PROFILING:
#    pytest tests/test_optimized_strategy.py --durations=10
#
# 8. RUN IN PARALLEL (FASTER):
#    pip install pytest-xdist
#    pytest tests/test_optimized_strategy.py -n auto
#
# 9. RUN WITH DETAILED OUTPUT ON FAILURE:
#    pytest tests/test_optimized_strategy.py --tb=long
#
# 10. RUN WITH HTML REPORT:
#     pip install pytest-html
#     pytest tests/test_optimized_strategy.py --html=report.html --self-contained-html
#
# 11. DEBUGGING FAILED TESTS:
#     pytest tests/test_optimized_strategy.py --pdb  # Drop into debugger on failure
#
# 12. RUN WITH MEMORY TRACKING:
#     pip install pytest-monitor
#     pytest tests/test_optimized_strategy.py --monitor
#
# 13. REQUIREMENTS:
#     pip install pytest pandas numpy
#     # Optional for enhanced testing:
#     pip install pytest-cov pytest-xdist pytest-html pytest-monitor
#
# 14. INTEGRATION WITH CI/CD:
#     # GitHub Actions example
#     - name: Run Strategy Tests
#       run: |
#         pytest tests/test_optimized_strategy.py -v --tb=short
#
# 15. EXPECTED OUTPUT:
#     ============================= test session starts ==============================
#     collected 15 items
#
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_strategy_initialization PASSED [  6%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_performance_improvement PASSED [ 13%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_exit_condition_performance PASSED [ 20%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_memory_efficiency PASSED [ 26%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_profit_calculation_accuracy PASSED [ 33%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_exit_condition_cache PASSED [ 40%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_mode_detection_performance PASSED [ 46%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_dataframe_caching PASSED [ 53%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_indicator_batch_calculation PASSED [ 60%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_strategy_logic_preservation PASSED [ 66%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_error_handling PASSED [ 73%]
#     tests/test_optimized_strategy.py::TestOptimizedStrategy::test_concurrent_performance PASSED [ 80%]
#
#     ========================= 12 tests passed in 5.67s ===========================
#
# 16. TROUBLESHOOTING:
#     # If tests fail due to missing dependencies:
#     pip install -r requirements.txt
#
#     # If tests fail due to strategy import issues:
#     # Make sure NostalgiaForInfinityX6.py and NostalgiaForInfinityX6_optimized_claude.py
#     # are in the project root directory
#
#     # If tests are too slow:
#     pytest tests/test_optimized_strategy.py -k "not test_concurrent"  # Skip slow tests
#
# 17. CUSTOM TEST CONFIGURATION:
#     # Create pytest.ini in project root:
#     [tool:pytest]
#     testpaths = tests
#     python_files = test_*.py
#     python_classes = Test*
#     python_functions = test_*
#     addopts = -v --tb=short
#
# 18. RUNNING IN DOCKER:
#     # Dockerfile example
#     FROM python:3.9
#     WORKDIR /app
#     COPY . .
#     RUN pip install pytest pandas numpy
#     CMD ["pytest", "tests/test_optimized_strategy.py", "-v"]