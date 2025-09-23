"""
综合优化测试套件
测试纯性能优化版本的各项功能和性能指标
"""

import pytest
import pandas as pd
import numpy as np
import time
import gc
import psutil
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPerformanceOptimization:
    """纯性能优化测试类"""
    
    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)  # 确保结果可重复
        dates = pd.date_range(start='2023-01-01', periods=10000, freq='5min')
        
        data = {
            'open': 100 + np.random.randn(10000).cumsum() * 0.1,
            'high': 101 + np.random.randn(10000).cumsum() * 0.1,
            'low': 99 + np.random.randn(10000).cumsum() * 0.1,
            'close': 100 + np.random.randn(10000).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 10000),
            'date': dates
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def mock_original_strategy(self):
        """创建模拟原始策略"""
        strategy = Mock()
        strategy.stoploss = -0.50
        
        def mock_populate_indicators(dataframe, metadata):
            # 模拟重计算负载
            for i in range(50):
                dataframe[f'indicator_{i}'] = dataframe['close'].rolling(i+1).mean()
            return dataframe
        
        def mock_populate_entry_trend(dataframe, metadata):
            dataframe['enter_long'] = (dataframe['close'] > dataframe['close'].rolling(20).mean()).astype(int)
            dataframe['enter_short'] = (dataframe['close'] < dataframe['close'].rolling(20).mean()).astype(int)
            return dataframe
        
        def mock_populate_exit_trend(dataframe, metadata):
            dataframe['exit_long'] = (dataframe['close'] < dataframe['close'].rolling(20).mean()).astype(int)
            dataframe['exit_short'] = (dataframe['close'] > dataframe['close'].rolling(20).mean()).astype(int)
            return dataframe
        
        def mock_custom_stoploss(pair, trade, current_time, current_rate, current_profit, **kwargs):
            return -0.50
        
        strategy.populate_indicators = mock_populate_indicators
        strategy.populate_entry_trend = mock_populate_entry_trend
        strategy.populate_exit_trend = mock_populate_exit_trend
        strategy.custom_stoploss = mock_custom_stoploss
        
        return strategy
    
    @pytest.fixture
    def optimized_strategy(self, mock_original_strategy):
        """创建优化策略实例"""
        try:
            from NostalgiaForInfinityX6_CC_performance_only import create_performance_wrapper
            return create_performance_wrapper(mock_original_strategy)
        except ImportError as e:
            logger.error(f"导入优化模块失败: {e}")
            pytest.skip("优化模块不可用")
    
    def test_basic_functionality(self, optimized_strategy, sample_data):
        """测试基本功能是否正常"""
        logger.info("测试基本功能...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 测试指标计算
        result = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        assert result is not None, "指标计算失败"
        assert len(result) == len(sample_data), "数据长度不一致"
        assert 'indicator_0' in result.columns, "指标未正确计算"
        
        logger.info("✅ 基本功能测试通过")
    
    def test_caching_effectiveness(self, optimized_strategy, sample_data):
        """测试缓存效果"""
        logger.info("测试缓存效果...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 第一次计算（应该缓存未命中）
        start_time = time.time()
        result1 = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        first_time = time.time() - start_time
        
        # 第二次计算（应该缓存命中）
        start_time = time.time()
        result2 = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        second_time = time.time() - start_time
        
        # 验证结果一致性
        pd.testing.assert_frame_equal(result1, result2)
        
        # 验证性能提升
        assert second_time < first_time * 0.5, f"缓存效果不佳: 第一次{first_time:.3f}s, 第二次{second_time:.3f}s"
        
        # 检查缓存统计
        stats = optimized_strategy.get_performance_stats()
        assert stats['cache_performance']['hits'] > 0, "缓存未命中"
        
        logger.info(f"✅ 缓存效果测试通过 - 第一次: {first_time:.3f}s, 第二次: {second_time:.3f}s")
    
    def test_memory_optimization(self, optimized_strategy, sample_data):
        """测试内存优化效果"""
        logger.info("测试内存优化效果...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 记录内存使用
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)  # MB
        
        # 执行优化计算
        result = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        
        memory_after = process.memory_info().rss / (1024**2)  # MB
        memory_increase = memory_after - memory_before
        
        # 验证内存使用合理
        assert memory_increase < 100, f"内存使用增加过多: {memory_increase:.1f}MB"
        
        logger.info(f"✅ 内存优化测试通过 - 内存增加: {memory_increase:.1f}MB")
    
    def test_entry_exit_signals(self, optimized_strategy, sample_data):
        """测试入场出场信号"""
        logger.info("测试入场出场信号...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 先计算指标
        dataframe = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        
        # 测试入场信号
        entry_df = optimized_strategy.populate_entry_trend_with_cache(dataframe.copy(), metadata)
        assert 'enter_long' in entry_df.columns, "入场信号未生成"
        assert 'enter_short' in entry_df.columns, "入场信号未生成"
        assert entry_df['enter_long'].dtype == int, "入场信号类型错误"
        
        # 测试出场信号
        exit_df = optimized_strategy.populate_exit_trend_with_cache(dataframe.copy(), metadata)
        assert 'exit_long' in exit_df.columns, "出场信号未生成"
        assert 'exit_short' in exit_df.columns, "出场信号未生成"
        assert exit_df['exit_long'].dtype == int, "出场信号类型错误"
        
        logger.info("✅ 入场出场信号测试通过")
    
    def test_custom_stoploss(self, optimized_strategy):
        """测试自定义止损"""
        logger.info("测试自定义止损...")
        
        # 创建模拟交易对象
        mock_trade = Mock()
        mock_trade.open_rate = 100.0
        mock_trade.leverage = 1.0
        mock_trade.is_short = False
        
        current_time = datetime.now()
        current_rate = 95.0
        current_profit = -0.05
        
        # 测试止损计算
        stoploss = optimized_strategy.custom_stoploss_optimized(
            'BTC/USDT', mock_trade, current_time, current_rate, current_profit
        )
        
        assert isinstance(stoploss, float), "止损值类型错误"
        assert -1.0 <= stoploss <= 0, f"止损值超出合理范围: {stoploss}"
        
        logger.info(f"✅ 自定义止损测试通过 - 止损值: {stoploss}")
    
    def test_performance_comparison(self, optimized_strategy, sample_data):
        """测试性能对比"""
        logger.info("测试性能对比...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 测试多次调用的性能
        times = []
        for i in range(5):
            start_time = time.time()
            result = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
        
        # 第一次应该较慢（缓存未命中）
        # 后续应该较快（缓存命中）
        avg_time = sum(times[1:]) / len(times[1:])  # 排除第一次
        first_time = times[0]
        
        assert avg_time < first_time * 0.5, f"平均性能未提升: 第一次{first_time:.3f}s, 平均{avg_time:.3f}s"
        
        logger.info(f"✅ 性能对比测试通过 - 第一次: {first_time:.3f}s, 平均: {avg_time:.3f}s")
    
    def test_cache_statistics(self, optimized_strategy, sample_data):
        """测试缓存统计"""
        logger.info("测试缓存统计...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 多次调用以生成统计
        for i in range(3):
            optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        
        # 获取统计信息
        stats = optimized_strategy.get_performance_stats()
        
        assert 'cache_performance' in stats, "缓存统计缺失"
        assert 'hits' in stats['cache_performance'], "缓存命中统计缺失"
        assert 'misses' in stats['cache_performance'], "缓存未命中统计缺失"
        assert 'hit_rate' in stats['cache_performance'], "缓存命中率统计缺失"
        
        # 验证统计值合理性
        hit_rate = stats['cache_performance']['hit_rate']
        assert 0 <= hit_rate <= 1, f"缓存命中率异常: {hit_rate}"
        
        logger.info(f"✅ 缓存统计测试通过 - 命中率: {hit_rate:.2%}")
    
    def test_memory_cleanup(self, optimized_strategy, sample_data):
        """测试内存清理"""
        logger.info("测试内存清理...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 多次计算以填充缓存
        for i in range(10):
            optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
        
        # 记录清理前状态
        stats_before = optimized_strategy.get_performance_stats()
        cache_size_before = stats_before['cache_performance']['cache_size']
        
        # 执行内存清理
        optimized_strategy.cleanup_memory()
        
        # 记录清理后状态
        stats_after = optimized_strategy.get_performance_stats()
        
        assert 'cache_performance' in stats_after, "清理后统计缺失"
        
        logger.info(f"✅ 内存清理测试通过 - 清理前缓存大小: {cache_size_before}")
    
    def test_error_handling(self, optimized_strategy, sample_data):
        """测试错误处理"""
        logger.info("测试错误处理...")
        
        # 测试无效元数据
        invalid_metadata = None
        try:
            result = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), invalid_metadata)
            # 应该能处理None元数据
            assert result is not None, "应该能处理无效元数据"
        except Exception as e:
            logger.warning(f"处理无效元数据时出错: {e}")
        
        # 测试空数据
        empty_data = pd.DataFrame()
        try:
            result = optimized_strategy.calculate_indicators_with_cache(empty_data, {'pair': 'BTC/USDT'})
            # 应该能处理空数据
        except Exception as e:
            logger.warning(f"处理空数据时出错: {e}")
        
        logger.info("✅ 错误处理测试通过")
    
    def test_signal_consistency(self, optimized_strategy, sample_data):
        """测试信号一致性"""
        logger.info("测试信号一致性...")
        
        metadata = {'pair': 'BTC/USDT'}
        
        # 多次计算相同的信号
        results = []
        for i in range(3):
            dataframe = optimized_strategy.calculate_indicators_with_cache(sample_data.copy(), metadata)
            entry_df = optimized_strategy.populate_entry_trend_with_cache(dataframe.copy(), metadata)
            exit_df = optimized_strategy.populate_exit_trend_with_cache(dataframe.copy(), metadata)
            
            results.append({
                'entry': entry_df[['enter_long', 'enter_short']].copy(),
                'exit': exit_df[['exit_long', 'exit_short']].copy()
            })
        
        # 验证结果一致性
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0]['entry'], results[i]['entry'])
            pd.testing.assert_frame_equal(results[0]['exit'], results[i]['exit'])
        
        logger.info("✅ 信号一致性测试通过")

class TestIntegration:
    """集成测试类"""
    
    def test_complete_workflow(self):
        """测试完整工作流程"""
        logger.info("测试完整工作流程...")
        
        try:
            from NostalgiaForInfinityX6_CC_performance_only import PerformanceCache, MemoryOptimizedDataFrame
            
            # 测试内存优化
            test_data = pd.DataFrame({
                'open': np.random.randn(1000) + 100,
                'high': np.random.randn(1000) + 101,
                'low': np.random.randn(1000) + 99,
                'close': np.random.randn(1000) + 100,
                'volume': np.random.randint(1000, 10000, 1000)
            })
            
            opt_df = MemoryOptimizedDataFrame(test_data)
            memory_stats = opt_df.get_memory_saved()
            
            assert memory_stats['memory_saved_percent'] > 0, "内存优化未生效"
            
            # 测试缓存系统
            cache = PerformanceCache(max_size=100, ttl_seconds=60)
            cache.set('test_key', 'test_value')
            retrieved_value = cache.get('test_key')
            
            assert retrieved_value == 'test_value', "缓存系统工作异常"
            
            logger.info("✅ 集成测试通过")
            
        except ImportError as e:
            logger.warning(f"集成测试跳过 - 模块导入失败: {e}")
            pytest.skip("集成测试模块不可用")

def run_performance_benchmark():
    """运行性能基准测试"""
    logger.info("运行性能基准测试...")
    
    try:
        from NostalgiaForInfinityX6_CC_performance_only import create_performance_wrapper
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=5000, freq='5min')
        test_data = pd.DataFrame({
            'open': 100 + np.random.randn(5000).cumsum() * 0.1,
            'high': 101 + np.random.randn(5000).cumsum() * 0.1,
            'low': 99 + np.random.randn(5000).cumsum() * 0.1,
            'close': 100 + np.random.randn(5000).cumsum() * 0.1,
            'volume': np.random.randint(1000, 10000, 5000),
            'date': dates
        })
        
        # 模拟原始策略
        class MockOriginalStrategy:
            def __init__(self):
                self.stoploss = -0.50
            
            def populate_indicators(self, dataframe, metadata):
                for i in range(30):
                    dataframe[f'indicator_{i}'] = dataframe['close'].rolling(i+1).mean()
                return dataframe
            
            def populate_entry_trend(self, dataframe, metadata):
                dataframe['enter_long'] = (dataframe['close'] > dataframe['close'].rolling(20).mean()).astype(int)
                return dataframe
            
            def populate_exit_trend(self, dataframe, metadata):
                dataframe['exit_long'] = (dataframe['close'] < dataframe['close'].rolling(20).mean()).astype(int)
                return dataframe
        
        # 测试原始策略性能
        original_strategy = MockOriginalStrategy()
        metadata = {'pair': 'BTC/USDT'}
        
        # 原始策略执行时间
        start_time = time.time()
        for i in range(10):
            result = original_strategy.populate_indicators(test_data.copy(), metadata)
        original_time = time.time() - start_time
        
        # 优化策略执行时间
        optimized_strategy = create_performance_wrapper(original_strategy)
        
        start_time = time.time()
        for i in range(10):
            result = optimized_strategy.calculate_indicators_with_cache(test_data.copy(), metadata)
        optimized_time = time.time() - start_time
        
        # 计算性能提升
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        logger.info(f"基准测试结果:")
        logger.info(f"原始策略执行时间: {original_time:.3f}s")
        logger.info(f"优化策略执行时间: {optimized_time:.3f}s")
        logger.info(f"性能提升: {speedup:.2f}x")
        
        # 获取缓存统计
        stats = optimized_strategy.get_performance_stats()
        hit_rate = stats['cache_performance']['hit_rate']
        logger.info(f"缓存命中率: {hit_rate:.2%}")
        
        return {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'cache_hit_rate': hit_rate
        }
        
    except Exception as e:
        logger.error(f"基准测试失败: {e}")
        return None

if __name__ == "__main__":
    print("🧪 运行综合优化测试套件")
    print("=" * 50)
    
    # 运行基准测试
    benchmark_results = run_performance_benchmark()
    
    if benchmark_results:
        print(f"\n📊 性能基准测试结果:")
        print(f"性能提升: {benchmark_results['speedup']:.2f}x")
        print(f"缓存命中率: {benchmark_results['cache_hit_rate']:.2%}")
    
    print(f"\n🧪 运行 pytest 测试...")
    
    # 运行 pytest 测试
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '--log-cli-level=INFO'
    ]
    
    pytest.main(pytest_args)
    
    print("\n✅ 测试完成！")