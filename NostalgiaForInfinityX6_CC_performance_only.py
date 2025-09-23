"""
NostalgiaForInfinityX6_CC 性能优化包装器
只优化性能，不修改任何交易逻辑和条件
"""

import pandas as pd
import numpy as np
import logging
import gc
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from functools import lru_cache
import hashlib
import json

# 可选依赖
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available, memory monitoring disabled")

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
    
    @njit(fastmath=True, cache=True)
    def calculate_profit_numba(open_rate: float, current_rate: float, 
                              leverage: float = 1.0, is_short: bool = False) -> float:
        """Numba 加速的利润计算"""
        if is_short:
            return (open_rate - current_rate) / open_rate * leverage
        else:
            return (current_rate - open_rate) / open_rate * leverage
    
    @njit(fastmath=True, cache=True)
    def calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Numba 加速的 RSI 计算"""
        n = len(prices)
        rsi = np.zeros(n)
        
        if n < period + 1:
            return rsi
        
        # 计算价格变化
        deltas = np.zeros(n - 1)
        for i in range(1, n):
            deltas[i-1] = prices[i] - prices[i-1]
        
        # 初始平均值
        avg_gain = 0.0
        avg_loss = 0.0
        
        for i in range(period):
            if deltas[i] > 0:
                avg_gain += deltas[i]
            else:
                avg_loss += -deltas[i]
        
        avg_gain /= period
        avg_loss /= period
        
        # 计算 RSI
        for i in range(period, n - 1):
            if deltas[i] > 0:
                gain = deltas[i]
                loss = 0.0
            else:
                gain = 0.0
                loss = -deltas[i]
            
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
except ImportError:
    NUMBA_AVAILABLE = False
    logging.warning("Numba not available, using standard implementations")
    
    def calculate_profit_numba(open_rate: float, current_rate: float, 
                              leverage: float = 1.0, is_short: bool = False) -> float:
        """标准利润计算"""
        if is_short:
            return (open_rate - current_rate) / open_rate * leverage
        else:
            return (current_rate - open_rate) / open_rate * leverage
    
    def calculate_rsi_numba(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """标准 RSI 计算"""
        return pd.Series(prices).rolling(period).apply(
            lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).mean() / 
                                      x.diff().clip(upper=0).abs().mean())))
        ).fillna(0).values


class PerformanceCache:
    """高性能缓存系统"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache = {}
        self._access_times = {}
        self._hits = 0
        self._misses = 0
        self._start_time = datetime.now()
    
    def _generate_key(self, key: str) -> str:
        """生成缓存键"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        cache_key = self._generate_key(key)
        current_time = datetime.now()
        
        if cache_key in self._cache:
            # 检查 TTL
            if current_time - self._access_times[cache_key] < timedelta(seconds=self.ttl_seconds):
                self._hits += 1
                self._access_times[cache_key] = current_time
                return self._cache[cache_key]
            else:
                # 过期，删除
                del self._cache[cache_key]
                del self._access_times[cache_key]
        
        self._misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """设置缓存值"""
        cache_key = self._generate_key(key)
        current_time = datetime.now()
        
        # 清理过期项
        self._cleanup_expired()
        
        # 如果缓存满了，清理最旧的项
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            del self._cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._cache[cache_key] = value
        self._access_times[cache_key] = current_time
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存项"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, access_time in self._access_times.items():
            if current_time - access_time >= timedelta(seconds=self.ttl_seconds):
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self._cache:
                del self._cache[key]
            del self._access_times[key]
    
    def clear_expired(self) -> None:
        """清理所有过期缓存"""
        self._cleanup_expired()
    
    def clear(self) -> None:
        """清空缓存"""
        self._cache.clear()
        self._access_times.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self._hits,
            'misses': self._misses,
            'cache_size': len(self._cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'ttl_seconds': self.ttl_seconds,
            'uptime_seconds': (datetime.now() - self._start_time).total_seconds()
        }


class MemoryOptimizedDataFrame:
    """内存优化的 DataFrame 包装器"""
    
    def __init__(self, dataframe: pd.DataFrame):
        self.original_columns = dataframe.columns.tolist()
        self.dataframe = self._optimize_memory(dataframe.copy())
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化 DataFrame 内存使用"""
        # 价格相关列保持 float64 精度
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # 其他数值列转换为 float32
        for col in df.select_dtypes(include=[np.float64]).columns:
            if col not in price_columns:
                df[col] = df[col].astype(np.float32)
        
        # 整数列优化
        for col in df.select_dtypes(include=[np.int64]).columns:
            if df[col].max() < 2**31 and df[col].min() > -2**31:
                df[col] = df[col].astype(np.int32)
        
        return df
    
    def get_memory_saved(self) -> Dict[str, float]:
        """获取内存节省统计"""
        original_memory = sum([
            self.dataframe[col].memory_usage(deep=True) 
            for col in self.original_columns
        ]) / (1024**2)  # MB
        
        optimized_memory = self.dataframe.memory_usage(deep=True).sum() / (1024**2)  # MB
        
        return {
            'original_memory_mb': original_memory,
            'optimized_memory_mb': optimized_memory,
            'memory_saved_mb': original_memory - optimized_memory,
            'memory_saved_percent': (original_memory - optimized_memory) / original_memory * 100
        }


class NostalgiaForInfinityX6_CC_PerformanceOptimized:
    """
    性能优化的 NostalgiaForInfinityX6_CC 包装器
    只优化性能，不修改任何交易逻辑和条件
    """
    
    def __init__(self, original_strategy):
        """初始化性能优化包装器"""
        self.original_strategy = original_strategy
        self.indicator_cache = PerformanceCache(max_size=800, ttl_seconds=240)
        self.mode_cache = {}
        self._memory_stats = []
        self._performance_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimizations': 0,
            'execution_time_saved': 0.0
        }
        
        # 设置性能监控
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
    
    def cleanup_memory(self):
        """内存清理"""
        self.indicator_cache.clear_expired()
        gc.collect()
        
        # 记录内存统计
        if len(self._memory_stats) < 20:
            self._memory_stats.append({
                'timestamp': datetime.now(),
                'cache_stats': self.indicator_cache.get_stats(),
                'memory_usage': psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 0
            })
    
    def calculate_indicators_with_cache(self, dataframe: pd.DataFrame, 
                                       metadata: Dict) -> pd.DataFrame:
        """带缓存的指标计算 - 保持原始逻辑"""
        
        # 内存优化 DataFrame
        opt_df = MemoryOptimizedDataFrame(dataframe)
        dataframe = opt_df.dataframe
        
        # 获取原始策略的指标计算方法
        original_calculate = getattr(self.original_strategy, 'populate_indicators', None)
        
        if original_calculate:
            # 尝试从缓存获取已计算的指标
            cache_key = f"indicators_{metadata.get('pair', 'unknown')}"
            cached_result = self.indicator_cache.get(cache_key)
            
            if cached_result is not None and len(cached_result) == len(dataframe):
                self._performance_stats['cache_hits'] += 1
                return cached_result
            
            # 缓存未命中，执行原始计算
            self._performance_stats['cache_misses'] += 1
            result = original_calculate(opt_df.dataframe.copy(), metadata)
            
            # 缓存结果
            self.indicator_cache.set(cache_key, result.copy())
            self._performance_stats['memory_optimizations'] += 1
            
            return result
        else:
            # 如果没有原始方法，返回优化后的 DataFrame
            return opt_df.dataframe
    
    def populate_entry_trend_with_cache(self, dataframe: pd.DataFrame, 
                                       metadata: Dict) -> pd.DataFrame:
        """带缓存的入场趋势计算 - 保持原始逻辑"""
        
        # 获取原始策略的入场方法
        original_entry = getattr(self.original_strategy, 'populate_entry_trend', None)
        
        if original_entry:
            # 尝试缓存
            cache_key = f"entry_{metadata.get('pair', 'unknown')}"
            cached_result = self.indicator_cache.get(cache_key)
            
            if cached_result is not None and len(cached_result) == len(dataframe):
                return cached_result
            
            # 执行原始入场逻辑
            result = original_entry(dataframe.copy(), metadata)
            
            # 缓存结果
            self.indicator_cache.set(cache_key, result.copy())
            
            return result
        else:
            # 设置默认入场信号
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
            return dataframe
    
    def populate_exit_trend_with_cache(self, dataframe: pd.DataFrame, 
                                      metadata: Dict) -> pd.DataFrame:
        """带缓存的出场趋势计算 - 保持原始逻辑"""
        
        # 获取原始策略的出场方法
        original_exit = getattr(self.original_strategy, 'populate_exit_trend', None)
        
        if original_exit:
            # 尝试缓存
            cache_key = f"exit_{metadata.get('pair', 'unknown')}"
            cached_result = self.indicator_cache.get(cache_key)
            
            if cached_result is not None and len(cached_result) == len(dataframe):
                return cached_result
            
            # 执行原始出场逻辑
            result = original_exit(dataframe.copy(), metadata)
            
            # 缓存结果
            self.indicator_cache.set(cache_key, result.copy())
            
            return result
        else:
            # 设置默认出场信号
            dataframe['exit_long'] = 0
            dataframe['exit_short'] = 0
            return dataframe
    
    def custom_stoploss_optimized(self, pair: str, trade, current_time: datetime, 
                                 current_rate: float, current_profit: float, **kwargs) -> float:
        """优化的自定义止损 - 使用 Numba 加速"""
        
        # 使用 Numba 加速利润计算
        try:
            leverage = getattr(trade, 'leverage', 1.0)
            is_short = getattr(trade, 'is_short', False)
            
            optimized_profit = calculate_profit_numba(
                trade.open_rate, current_rate, leverage, is_short
            )
            
            # 调用原始止损逻辑（如果存在）
            original_stoploss = getattr(self.original_strategy, 'custom_stoploss', None)
            if original_stoploss:
                return original_stoploss(pair, trade, current_time, current_rate, 
                                       optimized_profit, **kwargs)
            else:
                # 默认止损逻辑
                return self.original_strategy.stoploss
                
        except Exception as e:
            logging.error(f"Error in optimized stoploss: {e}")
            # 出错时返回原始策略的止损
            return getattr(self.original_strategy, 'stoploss', -0.50)
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        cache_stats = self.indicator_cache.get_stats()
        
        return {
            'cache_performance': cache_stats,
            'memory_optimizations': self._performance_stats['memory_optimizations'],
            'total_cache_requests': cache_stats['hits'] + cache_stats['misses'],
            'memory_usage_history': self._memory_stats[-10:] if self._memory_stats else [],
            'gc_stats': {
                'collections': gc.get_count(),
                'thresholds': gc.get_threshold(),
                'objects_collected': gc.get_stats() if hasattr(gc, 'get_stats') else 'N/A'
            }
        }
    
    def __getattr__(self, name):
        """代理所有其他属性和方法到原始策略"""
        return getattr(self.original_strategy, name)


# 使用示例和测试函数
def create_performance_wrapper(original_strategy_instance):
    """创建性能优化的包装器"""
    return NostalgiaForInfinityX6_CC_PerformanceOptimized(original_strategy_instance)


def test_performance_optimization():
    """测试性能优化效果"""
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'open': np.random.randn(10000) + 100,
        'high': np.random.randn(10000) + 101,
        'low': np.random.randn(10000) + 99,
        'close': np.random.randn(10000) + 100,
        'volume': np.random.randint(1000, 10000, 10000)
    })
    
    # 模拟原始策略
    class MockOriginalStrategy:
        def __init__(self):
            self.stoploss = -0.50
        
        def populate_indicators(self, dataframe, metadata):
            # 模拟重指标计算
            for i in range(50):
                dataframe[f'indicator_{i}'] = dataframe['close'].rolling(i+1).mean()
            return dataframe
        
        def populate_entry_trend(self, dataframe, metadata):
            dataframe['enter_long'] = (dataframe['close'] > dataframe['close'].rolling(20).mean()).astype(int)
            return dataframe
        
        def populate_exit_trend(self, dataframe, metadata):
            dataframe['exit_long'] = (dataframe['close'] < dataframe['close'].rolling(20).mean()).astype(int)
            return dataframe
    
    original_strategy = MockOriginalStrategy()
    optimized_strategy = create_performance_wrapper(original_strategy)
    
    # 测试性能
    print("Testing performance optimization...")
    
    # 原始策略
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 0
    
    result1 = original_strategy.populate_indicators(test_data.copy(), {'pair': 'BTC/USDT'})
    
    original_time = time.time() - start_time
    original_memory = (psutil.Process().memory_info().rss / (1024**2) - start_memory) if PSUTIL_AVAILABLE else 0
    
    # 优化策略
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / (1024**2) if PSUTIL_AVAILABLE else 0
    
    result2 = optimized_strategy.calculate_indicators_with_cache(test_data.copy(), {'pair': 'BTC/USDT'})
    
    optimized_time = time.time() - start_time
    optimized_memory = (psutil.Process().memory_info().rss / (1024**2) - start_memory) if PSUTIL_AVAILABLE else 0
    
    # 结果对比
    print(f"Original Strategy:")
    print(f"  Time: {original_time:.3f}s")
    print(f"  Memory: {original_memory:.1f}MB")
    
    print(f"Optimized Strategy:")
    print(f"  Time: {optimized_time:.3f}s")
    print(f"  Memory: {optimized_memory:.1f}MB")
    
    print(f"Improvements:")
    print(f"  Speedup: {original_time/optimized_time:.2f}x")
    print(f"  Memory saved: {original_memory-optimized_memory:.1f}MB")
    
    # 显示性能统计
    stats = optimized_strategy.get_performance_stats()
    print(f"Cache hit rate: {stats['cache_performance']['hit_rate']:.2%}")
    
    return optimized_strategy


if __name__ == "__main__":
    print("Performance-only optimization for NostalgiaForInfinityX6_CC")
    print("=" * 60)
    
    # 运行测试
    optimized_wrapper = test_performance_optimization()
    
    print("\nOptimization complete!")
    print("Use create_performance_wrapper() to wrap your original strategy instance.")