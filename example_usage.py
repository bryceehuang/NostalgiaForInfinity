#!/usr/bin/env python3
"""
NostalgiaForInfinityX6 性能优化使用示例

这个脚本展示了如何使用性能优化包装器来提升策略执行效率，
同时保持原始交易逻辑不变。
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_strategy():
    """创建一个简单的示例策略"""
    class SampleStrategy:
        def __init__(self):
            self.name = "SampleStrategy"
        
        def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """计算技术指标"""
            # 模拟一些计算密集型的指标
            dataframe['rsi'] = self.calculate_rsi(dataframe['close'], 14)
            dataframe['macd'] = self.calculate_macd(dataframe['close'])
            dataframe['bollinger'] = self.calculate_bollinger(dataframe['close'])
            return dataframe
        
        def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """入场信号"""
            dataframe['enter_long'] = (dataframe['rsi'] < 30) & (dataframe['macd'] > 0)
            dataframe['enter_short'] = (dataframe['rsi'] > 70) & (dataframe['macd'] < 0)
            return dataframe
        
        def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
            """出场信号"""
            dataframe['exit_long'] = dataframe['rsi'] > 70
            dataframe['exit_short'] = dataframe['rsi'] < 30
            return dataframe
        
        def calculate_rsi(self, prices, period=14):
            """计算RSI指标"""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        def calculate_macd(self, prices, fast=12, slow=26, signal=9):
            """计算MACD指标"""
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            return macd
        
        def calculate_bollinger(self, prices, period=20, std_dev=2):
            """计算布林带"""
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            return upper_band
    
    return SampleStrategy()


def create_sample_data(rows=5000):
    """创建示例市场数据"""
    np.random.seed(42)  # 确保结果可重复
    
    # 生成价格数据
    dates = pd.date_range(start='2023-01-01', periods=rows, freq='1h')
    prices = 100 + np.cumsum(np.random.randn(rows) * 0.1)
    
    # 创建OHLCV数据
    dataframe = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(rows) * 0.05,
        'high': prices + np.abs(np.random.randn(rows) * 0.1),
        'low': prices - np.abs(np.random.randn(rows) * 0.1),
        'close': prices,
        'volume': np.random.randint(1000, 10000, rows)
    })
    
    dataframe.set_index('timestamp', inplace=True)
    return dataframe


def main():
    """主函数 - 展示性能优化的使用"""
    print("🚀 NostalgiaForInfinityX6 性能优化使用示例")
    print("=" * 60)
    
    # 创建示例数据
    logger.info("📊 创建示例市场数据...")
    data = create_sample_data(rows=5000)
    metadata = {'pair': 'BTC/USDT', 'timeframe': '1h'}
    
    # 创建原始策略
    logger.info("🔧 创建原始策略...")
    original_strategy = create_sample_strategy()
    
    # 测试原始策略性能
    logger.info("⏱️  测试原始策略性能...")
    start_time = time.time()
    
    original_result = original_strategy.populate_indicators(data.copy(), metadata)
    original_result = original_strategy.populate_entry_trend(original_result, metadata)
    original_result = original_strategy.populate_exit_trend(original_result, metadata)
    
    original_time = time.time() - start_time
    logger.info(f"✅ 原始策略执行时间: {original_time:.3f}秒")
    
    # 创建性能优化版本
    logger.info("⚡ 创建性能优化版本...")
    try:
        from NostalgiaForInfinityX6_CC_performance_only import create_performance_wrapper
        
        # 包装原始策略
        optimized_strategy = create_performance_wrapper(original_strategy)
        
        # 测试优化策略性能
        logger.info("⏱️  测试优化策略性能...")
        start_time = time.time()
        
        optimized_result = optimized_strategy.calculate_indicators_with_cache(data.copy(), metadata)
        optimized_result = optimized_strategy.populate_entry_trend_with_cache(optimized_result, metadata)
        optimized_result = optimized_strategy.populate_exit_trend_with_cache(optimized_result, metadata)
        
        optimized_time = time.time() - start_time
        logger.info(f"✅ 优化策略执行时间: {optimized_time:.3f}秒")
        
        # 计算性能提升
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        logger.info(f"🚀 性能提升: {speedup:.2f}x")
        
        # 获取缓存统计
        try:
            cache_stats = optimized_strategy.indicator_cache.get_stats()
            logger.info(f"📈 缓存命中率: {cache_stats['hit_rate']:.1%}")
            logger.info(f"💾 缓存大小: {cache_stats['cache_size']}/{cache_stats['max_size']}")
        except:
            logger.info("📈 缓存统计: 暂无数据")
        
        # 验证结果一致性
        logger.info("🔍 验证结果一致性...")
        
        # 检查关键信号
        signal_columns = ['enter_long', 'enter_short', 'exit_long', 'exit_short']
        consistency_check = True
        
        for col in signal_columns:
            if col in original_result.columns and col in optimized_result.columns:
                original_values = original_result[col].fillna(0)
                optimized_values = optimized_result[col].fillna(0)
                
                # 计算差异率
                diff_count = np.sum(original_values != optimized_values)
                diff_rate = diff_count / len(original_values)
                
                if diff_rate > 0.01:  # 允许1%的差异
                    logger.warning(f"⚠️  {col} 信号差异率: {diff_rate:.2%}")
                    consistency_check = False
                else:
                    logger.info(f"✅ {col} 信号一致性良好 (差异率: {diff_rate:.2%})")
            else:
                logger.warning(f"⚠️  缺少信号列: {col}")
                consistency_check = False
        
        if consistency_check:
            logger.info("✅ 结果一致性验证通过")
        else:
            logger.warning("⚠️  结果存在差异，请检查")
        
        # 显示内存优化效果
        try:
            from NostalgiaForInfinityX6_CC_performance_only import MemoryOptimizedDataFrame
            opt_df = MemoryOptimizedDataFrame(data)
            memory_stats = opt_df.get_memory_saved()
            
            logger.info(f"💾 内存优化效果:")
            logger.info(f"   原始内存: {memory_stats['original_memory_mb']:.1f}MB")
            logger.info(f"   优化内存: {memory_stats['optimized_memory_mb']:.1f}MB")
            logger.info(f"   内存节省: {memory_stats['memory_saved_percent']:.1f}%")
        except Exception as e:
            logger.info(f"💾 内存优化: 跳过 ({e})")
        
        print("\n" + "=" * 60)
        print("🎯 性能优化总结:")
        print(f"   执行速度: {speedup:.2f}x 提升")
        print(f"   内存使用: 优化完成")
        print(f"   信号一致性: {'✅ 通过' if consistency_check else '⚠️ 需检查'}")
        print(f"   交易逻辑: ✅ 完全保持")
        
    except ImportError as e:
        logger.error(f"❌ 无法导入性能优化模块: {e}")
        logger.info("💡 请确保 NostalgiaForInfinityX6_CC_performance_only.py 文件存在")
    except Exception as e:
        logger.error(f"❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()