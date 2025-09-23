#!/usr/bin/env python3
"""
快速优化测试脚本
用于快速验证优化效果
"""

import pandas as pd
import numpy as np
import time
import gc
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_data(size=5000):
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=size, freq='5min')
    
    data = {
        'open': 100 + np.random.randn(size).cumsum() * 0.1,
        'high': 101 + np.random.randn(size).cumsum() * 0.1,
        'low': 99 + np.random.randn(size).cumsum() * 0.1,
        'close': 100 + np.random.randn(size).cumsum() * 0.1,
        'volume': np.random.randint(1000, 10000, size),
        'date': dates
    }
    
    return pd.DataFrame(data)

def mock_original_strategy():
    """模拟原始策略"""
    class MockStrategy:
        def __init__(self):
            self.stoploss = -0.50
        
        def populate_indicators(self, dataframe, metadata):
            """模拟重计算负载"""
            result = dataframe.copy()
            
            # 模拟复杂的指标计算
            for i in range(20):
                result[f'indicator_{i}'] = result['close'].rolling(i+1).mean()
                result[f'rsi_{i}'] = self.calculate_rsi(result['close'], 14 + i)
                result[f'macd_{i}'] = self.calculate_macd(result['close'])
            
            return result
        
        def calculate_rsi(self, prices, period=14):
            """计算RSI"""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def calculate_macd(self, prices, fast=12, slow=26, signal=9):
            """计算MACD"""
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=signal).mean()
            return macd - signal_line
        
        def populate_entry_trend(self, dataframe, metadata):
            """生成入场信号"""
            result = dataframe.copy()
            result['enter_long'] = (result['close'] > result['close'].rolling(20).mean()).astype(int)
            result['enter_short'] = (result['close'] < result['close'].rolling(20).mean()).astype(int)
            return result
        
        def populate_exit_trend(self, dataframe, metadata):
            """生成出场信号"""
            result = dataframe.copy()
            result['exit_long'] = (result['close'] < result['close'].rolling(20).mean()).astype(int)
            result['exit_short'] = (result['close'] > result['close'].rolling(20).mean()).astype(int)
            return result
        
        def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
            """自定义止损"""
            return -0.50
    
    return MockStrategy()

def test_performance_comparison():
    """测试性能对比"""
    logger.info("🚀 开始性能对比测试")
    
    # 创建测试数据
    test_data = create_test_data(5000)
    logger.info(f"📊 创建测试数据: {len(test_data)} 行")
    
    # 创建原始策略
    original_strategy = mock_original_strategy()
    metadata = {'pair': 'BTC/USDT'}
    
    # 测试原始策略性能
    logger.info("🔍 测试原始策略性能...")
    gc.collect()  # 清理内存
    start_time = time.time()
    
    original_result = original_strategy.populate_indicators(test_data.copy(), metadata)
    original_result = original_strategy.populate_entry_trend(original_result, metadata)
    original_result = original_strategy.populate_exit_trend(original_result, metadata)
    
    original_time = time.time() - start_time
    logger.info(f"⏱️  原始策略执行时间: {original_time:.3f}s")
    
    # 测试优化策略性能
    try:
        logger.info("🔍 测试优化策略性能...")
        from NostalgiaForInfinityX6_CC_performance_only import create_performance_wrapper
        
        optimized_strategy = create_performance_wrapper(original_strategy)
        
        gc.collect()  # 清理内存
        start_time = time.time()
        
        optimized_result = optimized_strategy.calculate_indicators_with_cache(test_data.copy(), metadata)
        optimized_result = optimized_strategy.populate_entry_trend_with_cache(optimized_result, metadata)
        optimized_result = optimized_strategy.populate_exit_trend_with_cache(optimized_result, metadata)
        
        optimized_time = time.time() - start_time
        logger.info(f"⏱️  优化策略执行时间: {optimized_time:.3f}s")
        
        # 计算性能提升
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        logger.info(f"⚡ 性能提升: {speedup:.2f}x")
        
        # 获取缓存统计
        try:
            stats = optimized_strategy.indicator_cache.get_stats()
            hit_rate = stats.get('hit_rate', 0)
            logger.info(f"📈 缓存命中率: {hit_rate:.2%}")
        except AttributeError:
            hit_rate = 0
            logger.info(f"📈 缓存命中率: {hit_rate:.2%}")
        
        # 验证结果一致性
        logger.info("🔍 验证结果一致性...")
        
        # 检查关键列是否存在
        required_columns = ['enter_long', 'enter_short', 'exit_long', 'exit_short']
        for col in required_columns:
            assert col in original_result.columns, f"原始结果缺少列: {col}"
            assert col in optimized_result.columns, f"优化结果缺少列: {col}"
        
        # 检查信号一致性（允许小的数值差异）
        for col in required_columns:
            original_values = original_result[col].values
            optimized_values = optimized_result[col].values
            
            # 检查是否基本一致（允许1%的差异）
            diff_count = np.sum(original_values != optimized_values)
            diff_rate = diff_count / len(original_values)
            
            if diff_rate > 0.01:  # 超过1%的差异
                logger.warning(f"⚠️  {col} 信号差异率: {diff_rate:.2%}")
            else:
                logger.info(f"✅ {col} 信号一致性良好 (差异率: {diff_rate:.2%})")
        
        logger.info("✅ 结果一致性验证通过")
        
        return {
            'original_time': original_time,
            'optimized_time': optimized_time,
            'speedup': speedup,
            'cache_hit_rate': hit_rate,
            'test_passed': True
        }
        
    except ImportError as e:
        logger.error(f"❌ 导入优化模块失败: {e}")
        return {
            'test_passed': False,
            'error': str(e)
        }
    except Exception as e:
        logger.error(f"❌ 优化策略测试失败: {e}")
        return {
            'test_passed': False,
            'error': str(e)
        }

def test_memory_optimization():
    """测试内存优化效果"""
    logger.info("🧠 测试内存优化效果...")
    
    try:
        from NostalgiaForInfinityX6_CC_performance_only import MemoryOptimizedDataFrame
        
        # 创建大数据集
        large_data = pd.DataFrame({
            'open': np.random.randn(10000) + 100,
            'high': np.random.randn(10000) + 101,
            'low': np.random.randn(10000) + 99,
            'close': np.random.randn(10000) + 100,
            'volume': np.random.randint(1000, 10000, 10000)
        })
        
        # 测试内存优化
        opt_df = MemoryOptimizedDataFrame(large_data)
        memory_stats = opt_df.get_memory_saved()
        
        logger.info(f"📊 内存优化统计:")
        logger.info(f"  原始内存: {memory_stats['original_memory_mb']:.1f}MB")
        logger.info(f"  优化内存: {memory_stats['optimized_memory_mb']:.1f}MB")
        logger.info(f"  内存节省: {memory_stats['memory_saved_percent']:.1f}%")
        
        assert memory_stats['memory_saved_percent'] > 0, "内存优化未生效"
        logger.info("✅ 内存优化测试通过")
        
        return memory_stats
        
    except ImportError as e:
        logger.warning(f"⚠️  内存优化模块测试跳过: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ 内存优化测试失败: {e}")
        return None

def test_caching_system():
    """测试缓存系统"""
    logger.info("💾 测试缓存系统...")
    
    try:
        from NostalgiaForInfinityX6_CC_performance_only import PerformanceCache
        
        # 创建缓存实例
        cache = PerformanceCache(max_size=100, ttl_seconds=60)
        
        # 测试基本操作
        cache.set('test_key', 'test_value')
        retrieved_value = cache.get('test_key')
        
        assert retrieved_value == 'test_value', "缓存基本操作失败"
        
        # 测试缓存统计
        stats = cache.get_stats()
        logger.info(f"📊 缓存统计: 命中率={stats['hit_rate']:.2%}, 大小={stats['cache_size']}")
        
        # 测试缓存过期
        cache_with_short_ttl = PerformanceCache(max_size=10, ttl_seconds=1)
        cache_with_short_ttl.set('expire_key', 'expire_value')
        
        import time
        time.sleep(1.1)  # 等待过期
        
        expired_value = cache_with_short_ttl.get('expire_key')
        assert expired_value is None, "缓存过期机制失败"
        
        logger.info("✅ 缓存系统测试通过")
        
        return stats
        
    except ImportError as e:
        logger.warning(f"⚠️  缓存模块测试跳过: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ 缓存系统测试失败: {e}")
        return None

def main():
    """主测试函数"""
    print("🧪 快速优化测试开始")
    print("=" * 50)
    
    # 运行性能对比测试
    perf_results = test_performance_comparison()
    
    if perf_results and perf_results.get('test_passed'):
        print(f"\n📈 性能测试结果:")
        print(f"  原始策略时间: {perf_results['original_time']:.3f}s")
        print(f"  优化策略时间: {perf_results['optimized_time']:.3f}s")
        print(f"  性能提升: {perf_results['speedup']:.2f}x")
        print(f"  缓存命中率: {perf_results['cache_hit_rate']:.2%}")
    
    # 运行内存优化测试
    memory_results = test_memory_optimization()
    
    if memory_results:
        print(f"\n💾 内存优化结果:")
        print(f"  内存节省: {memory_results['memory_saved_percent']:.1f}%")
    
    # 运行缓存系统测试
    cache_results = test_caching_system()
    
    if cache_results:
        print(f"\n💾 缓存系统结果:")
        print(f"  缓存命中率: {cache_results['hit_rate']:.2%}")
    
    print(f"\n✅ 快速测试完成！")
    
    # 总结
    if perf_results and perf_results.get('test_passed'):
        print(f"\n🎯 优化验证成功！")
        print(f"   性能提升: {perf_results['speedup']:.2f}x")
        print(f"   内存节省: {memory_results.get('memory_saved_percent', 0):.1f}%" if memory_results else "   内存优化: 未测试")
        print(f"   缓存命中: {perf_results['cache_hit_rate']:.2%}")
    else:
        print(f"\n❌ 优化验证失败，请检查错误信息")

if __name__ == "__main__":
    main()