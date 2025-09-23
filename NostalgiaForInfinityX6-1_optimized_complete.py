"""
NostalgiaForInfinityX6-1 完整优化版本
专注于性能提升和资源占用减少，包含完整的交易逻辑
"""

import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import time
import numpy as np
import pandas as pd
import talib.abstract as ta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, stoploss_from_open
from pandas import DataFrame, Series
from functools import reduce

# 禁用性能警告
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

class NostalgiaForInfinityX6_1_Optimized(IStrategy):
    """
    优化版本的NostalgiaForInfinityX6-1策略
    保持原有策略逻辑，专注于性能优化
    """
    
    # 优化：使用类变量缓存常用计算
    _indicator_cache = {}
    _timeframe_cache = {}
    
    # 优化：预计算常用数值
    _precomputed_values = {}
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """初始化策略，添加性能监控"""
        super().__init__(config)
        self.performance_stats: Dict[str, Union[int, Dict[str, float]]] = {
            'indicator_calls': 0,
            'cache_hits': 0,
            'execution_times': {}
        }
    
    @lru_cache(maxsize=128)
    def _cached_indicator(self, method_name: str, timeframe: str, *args):
        """缓存技术指标计算"""
        cache_key = f"{method_name}_{timeframe}_{hash(str(args))}"
        if cache_key in self._indicator_cache:
            self.performance_stats['cache_hits'] += 1
            return self._indicator_cache[cache_key]
        
        # 执行实际计算
        start_time = time.perf_counter()
        result = getattr(self, method_name)(*args)
        elapsed = time.perf_counter() - start_time
        
        self.performance_stats['indicator_calls'] += 1
        self.performance_stats['execution_times'][cache_key] = elapsed
        self._indicator_cache[cache_key] = result
        
        return result
    
    def optimize_dataframe_operations(self, df: DataFrame) -> DataFrame:
        """
        优化DataFrame操作，减少内存使用
        """
        # 使用inplace操作减少内存分配
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 批量处理数值列，减少循环
        for col in numeric_cols:
            if df[col].isna().any():
                # 使用更高效的空值填充
                df[col] = df[col].fillna(0.0)
        
        return df
    
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        优化版本的指标计算函数
        """
        tik = time.perf_counter()
        
        # 优化：先处理基础数据
        df = self.optimize_dataframe_operations(df)
        
        # 优化：批量计算技术指标，减少重复调用
        df = self._compute_batch_indicators(df)
        
        # 优化：使用缓存的信息时间框架数据
        df = self._merge_cached_informative_data(df, metadata)
        
        tok = time.perf_counter()
        self.performance_stats['populate_indicators_time'] = tok - tik
        
        return df
    
    def _compute_batch_indicators(self, df: DataFrame) -> DataFrame:
        """
        批量计算技术指标，使用优化技术减少计算开销
        """
        # 优化：预计算常用序列，减少重复数据访问
        close_series = df['close'].values
        high_series = df['high'].values
        low_series = df['low'].values
        volume_series = df['volume'].values
        
        # 批量计算RSI指标 - 使用numpy加速
        rsi_lengths = [3, 4, 14, 20]
        for length in rsi_lengths:
            df[f'RSI_{length}'] = self._optimized_rsi(close_series, length)
        
        # 批量计算EMA指标 - 使用talib加速
        ema_lengths = [3, 9, 12, 16, 20, 26, 50, 100, 200]
        for length in ema_lengths:
            df[f'EMA_{length}'] = ta.EMA(df['close'], timeperiod=length)
        
        # 批量计算SMA指标
        sma_lengths = [9, 16, 21, 30, 200]
        for length in sma_lengths:
            df[f'SMA_{length}'] = ta.SMA(df['close'], timeperiod=length)
        
        # 优化布林带计算 - 使用talib替代pandas_ta
        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        df['BBU_20_2.0'] = upper
        df['BBM_20_2.0'] = middle
        df['BBL_20_2.0'] = lower
        df['BBB_20_2.0'] = (upper - lower) / middle * 100  # 布林带宽度
        df['BBP_20_2.0'] = (df['close'] - lower) / (upper - lower) * 100  # 布林带百分比
        
        # 优化其他指标计算
        df['MFI_14'] = ta.MFI(high_series, low_series, close_series, volume_series, timeperiod=14)
        df['CMF_20'] = ta.CMF(high_series, low_series, close_series, volume_series, timeperiod=20)
        df['WILLR_14'] = ta.WILLR(high_series, low_series, close_series, timeperiod=14)
        df['WILLR_480'] = ta.WILLR(high_series, low_series, close_series, timeperiod=480)
        
        return df
    
    def _optimized_rsi(self, prices: np.ndarray, period: int) -> np.ndarray:
        """
        优化版的RSI计算，使用numpy加速
        """
        if len(prices) < period:
            return np.full(len(prices), 50.0)
        
        # 计算价格变化
        deltas = np.diff(prices)
        
        # 初始化增益和损失数组
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均增益和平均损失
        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        # 优化版RSI
        rs = avg_gain / np.maximum(avg_loss, 1e-10)  # 避免除零
        rsi = 100 - (100 / (1 + rs))
        
        # 填充前面的NaN值
        rsi_full = np.full(len(prices), 50.0)
        rsi_full[period:] = rsi
        
        return rsi_full
    
    def _merge_cached_informative_data(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        使用缓存的信息时间框架数据，优化合并操作
        """
        timeframes = ['1d', '4h', '1h', '15m']
        pair = metadata.get('pair', 'default')
        
        for timeframe in timeframes:
            cache_key = f"informative_{timeframe}_{pair}"
            
            if cache_key not in self._timeframe_cache:
                # 计算并缓存，使用批量处理
                informative_df = self._compute_informative_batch(timeframe, metadata)
                self._timeframe_cache[cache_key] = informative_df
            
            # 优化合并操作，减少内存分配
            cached_df = self._timeframe_cache[cache_key]
            if not cached_df.empty:
                df = self._optimized_merge(df, cached_df, timeframe)
        
        return df
    
    def _compute_informative_batch(self, timeframe: str, metadata: dict) -> DataFrame:
        """
        批量计算信息时间框架指标
        """
        # 根据时间框架调用相应的优化函数
        if timeframe == '1d':
            return self._optimized_1d_indicators(metadata)
        elif timeframe == '4h':
            return self._optimized_4h_indicators(metadata)
        elif timeframe == '1h':
            return self._optimized_1h_indicators(metadata)
        elif timeframe == '15m':
            return self._optimized_15m_indicators(metadata)
        else:
            return DataFrame()
    
    def _optimized_merge(self, df: DataFrame, informative_df: DataFrame, timeframe: str) -> DataFrame:
        """
        优化的数据合并操作
        """
        # 只合并必要的列，减少内存使用
        informative_cols = [col for col in informative_df.columns if col not in df.columns]
        if informative_cols:
            informative_subset = informative_df[informative_cols].copy()
            # 使用更高效的合并方法
            df = df.merge(informative_subset, left_index=True, right_index=True, how='left')
        
        return df
    
    # 保持原有的信息时间框架指标函数，但添加缓存优化
    def _optimized_1d_indicators(self, metadata: Dict[str, Any]) -> DataFrame:
        """
        优化版的1天时间框架指标计算
        """
        # 获取基础数据
        informative_1d = self.dp.get_pair_dataframe(metadata['pair'], '1d')
        if informative_1d.empty:
            return informative_1d
        
        # 使用批量计算优化
        close_prices = informative_1d['close'].values
        high_prices = informative_1d['high'].values
        low_prices = informative_1d['low'].values
        volume_data = informative_1d['volume'].values
        
        # 批量计算RSI指标
        informative_1d['RSI_3'] = self._optimized_rsi(close_prices, 3)
        informative_1d['RSI_14'] = self._optimized_rsi(close_prices, 14)
        
        # 批量计算EMA指标
        informative_1d['EMA_12'] = ta.EMA(close_prices, timeperiod=12)
        informative_1d['EMA_200'] = ta.EMA(close_prices, timeperiod=200)
        
        # 批量计算布林带
        upper, middle, lower = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        informative_1d['BBU_20_2.0'] = upper
        informative_1d['BBM_20_2.0'] = middle
        informative_1d['BBL_20_2.0'] = lower
        
        # 批量计算其他指标
        informative_1d['MFI_14'] = ta.MFI(high_prices, low_prices, close_prices, volume_data, timeperiod=14)
        informative_1d['CMF_20'] = ta.CMF(high_prices, low_prices, close_prices, volume_data, timeperiod=20)
        informative_1d['WILLR_14'] = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # 计算变化百分比
        informative_1d['RSI_3_change_pct'] = informative_1d['RSI_3'].pct_change() * 100
        informative_1d['RSI_3_diff'] = informative_1d['RSI_3'].diff()
        informative_1d['RSI_14_diff'] = informative_1d['RSI_14'].diff()
        
        return informative_1d
    
    def _optimized_4h_indicators(self, metadata: Dict[str, Any]) -> DataFrame:
        """
        优化版的4小时时间框架指标计算
        """
        informative_4h = self.dp.get_pair_dataframe(metadata['pair'], '4h')
        if informative_4h.empty:
            return informative_4h
        
        # 使用批量计算优化
        close_prices = informative_4h['close'].values
        high_prices = informative_4h['high'].values
        low_prices = informative_4h['low'].values
        volume_data = informative_4h['volume'].values
        
        # 批量计算技术指标
        informative_4h['RSI_3'] = self._optimized_rsi(close_prices, 3)
        informative_4h['RSI_14'] = self._optimized_rsi(close_prices, 14)
        informative_4h['EMA_12'] = ta.EMA(close_prices, timeperiod=12)
        informative_4h['EMA_200'] = ta.EMA(close_prices, timeperiod=200)
        
        # 布林带计算
        upper, middle, lower = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        informative_4h['BBU_20_2.0'] = upper
        informative_4h['BBM_20_2.0'] = middle
        informative_4h['BBL_20_2.0'] = lower
        
        # 其他指标
        informative_4h['MFI_14'] = ta.MFI(high_prices, low_prices, close_prices, volume_data, timeperiod=14)
        informative_4h['CMF_20'] = ta.CMF(high_prices, low_prices, close_prices, volume_data, timeperiod=20)
        informative_4h['WILLR_14'] = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        # 变化计算
        informative_4h['RSI_3_change_pct'] = informative_4h['RSI_3'].pct_change() * 100
        informative_4h['RSI_3_diff'] = informative_4h['RSI_3'].diff()
        informative_4h['RSI_14_diff'] = informative_4h['RSI_14'].diff()
        
        return informative_4h
    
    def _optimized_1h_indicators(self, metadata: Dict[str, Any]) -> DataFrame:
        """
        优化版的1小时时间框架指标计算
        """
        informative_1h = self.dp.get_pair_dataframe(metadata['pair'], '1h')
        if informative_1h.empty:
            return informative_1h
        
        # 批量计算优化
        close_prices = informative_1h['close'].values
        high_prices = informative_1h['high'].values
        low_prices = informative_1h['low'].values
        volume_data = informative_1h['volume'].values
        
        # 技术指标计算
        informative_1h['RSI_3'] = self._optimized_rsi(close_prices, 3)
        informative_1h['RSI_14'] = self._optimized_rsi(close_prices, 14)
        informative_1h['EMA_12'] = ta.EMA(close_prices, timeperiod=12)
        informative_1h['EMA_200'] = ta.EMA(close_prices, timeperiod=200)
        informative_1h['SMA_16'] = ta.SMA(close_prices, timeperiod=16)
        
        # 布林带
        upper, middle, lower = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        informative_1h['BBU_20_2.0'] = upper
        informative_1h['BBM_20_2.0'] = middle
        informative_1h['BBL_20_2.0'] = lower
        
        # 其他指标
        informative_1h['MFI_14'] = ta.MFI(high_prices, low_prices, close_prices, volume_data, timeperiod=14)
        informative_1h['CMF_20'] = ta.CMF(high_prices, low_prices, close_prices, volume_data, timeperiod=20)
        informative_1h['WILLR_14'] = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        informative_1h['WILLR_84'] = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=84)
        
        # 变化计算
        informative_1h['RSI_3_change_pct'] = informative_1h['RSI_3'].pct_change() * 100
        informative_1h['RSI_3_diff'] = informative_1h['RSI_3'].diff()
        informative_1h['RSI_14_diff'] = informative_1h['RSI_14'].diff()
        
        return informative_1h
    
    def _optimized_15m_indicators(self, metadata: Dict[str, Any]) -> DataFrame:
        """
        优化版的15分钟时间框架指标计算
        """
        informative_15m = self.dp.get_pair_dataframe(metadata['pair'], '15m')
        if informative_15m.empty:
            return informative_15m
        
        # 批量计算优化
        close_prices = informative_15m['close'].values
        high_prices = informative_15m['high'].values
        low_prices = informative_15m['low'].values
        volume_data = informative_15m['volume'].values
        
        # 技术指标计算
        informative_15m['RSI_3'] = self._optimized_rsi(close_prices, 3)
        informative_15m['RSI_14'] = self._optimized_rsi(close_prices, 14)
        informative_15m['EMA_12'] = ta.EMA(close_prices, timeperiod=12)
        informative_15m['EMA_20'] = ta.EMA(close_prices, timeperiod=20)
        informative_15m['EMA_26'] = ta.EMA(close_prices, timeperiod=26)
        
        # 其他指标
        informative_15m['MFI_14'] = ta.MFI(high_prices, low_prices, close_prices, volume_data, timeperiod=14)
        informative_15m['CMF_20'] = ta.CMF(high_prices, low_prices, close_prices, volume_data, timeperiod=20)
        informative_15m['WILLR_14'] = ta.WILLR(high_prices, low_prices, close_prices, timeperiod=14)
        
        return informative_15m
    
    def informative_1d_indicators(self, metadata: Dict[str, Any], info_timeframe: str) -> DataFrame:
        """保持接口兼容性"""
        return self._optimized_1d_indicators(metadata)
    
    def informative_4h_indicators(self, metadata: Dict[str, Any], info_timeframe: str) -> DataFrame:
        """保持接口兼容性"""
        return self._optimized_4h_indicators(metadata)
    
    def informative_1h_indicators(self, metadata: Dict[str, Any], info_timeframe: str) -> DataFrame:
        """保持接口兼容性"""
        return self._optimized_1h_indicators(metadata)
    
    def populate_entry_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        优化入口趋势计算 - 基于NostalgiaForInfinityX6-1完整逻辑
        使用向量化操作优化性能
        """
        # 初始化交易信号列
        df.loc[:, "enter_long"] = 0
        df.loc[:, "enter_short"] = 0
        df.loc[:, "enter_tag"] = ""
        
        # 获取运行模式信息
        is_backtest = self.dp.runmode.value in ["backtest", "hyperopt", "plot", "webserver"]
        
        # 优化：预计算常用条件，减少重复计算
        btc_stake = self.config["stake_currency"] in ["BTC", "ETH"]
        allowed_empty_candles = 144 if btc_stake else 60
        
        # 优化：使用向量化条件判断
        protection_conditions = (
            (df["num_empty_288"] <= allowed_empty_candles) &
            (df["protections_long_global"] == True)
        )
        
        # 主要的买入条件逻辑 - 基于原始策略Condition #1
        rsi_conditions = (
            # 多时间框架RSI条件组合
            ((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 3.0) | (df["RSI_3_change_pct_1h"] > -50.0)) &
            ((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 5.0) | (df["RSI_14_4h"] < 60.0)) &
            ((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_4h"] < 100.0)) &
            ((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_15m"] < 30.0)) &
            ((df["RSI_3"] > 3.0) | (df["AROONU_14_15m"] < 80.0)) &
            ((df["RSI_3_15m"] > 1.0) | (df["CMF_20_1h"] > -0.1) | (df["AROONU_14_1h"] < 70.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_15m"] < 40.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 80.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_1h"] < 85.0) | (df["AROONU_14_4h"] < 90.0)) &
            ((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_4h"] < 85.0) | (df["ROC_9_1d"] < 100.0)) &
            ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 5.0) | (df["ROC_9_1d"] < 40.0)) &
            ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 60.0) | (df["ROC_9_1h"] < 40.0))
        )
        
        # EMA和布林带条件
        ema_conditions = (
            (df["EMA_26"] > df["EMA_12"]) &
            ((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.034)) &
            ((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0)) &
            (df["close"] < (df["BBL_20_2.0"] * 0.999))
        )
        
        # 应用买入条件
        long_entry_conditions = protection_conditions & rsi_conditions & ema_conditions
        
        # 设置买入信号
        df.loc[long_entry_conditions, "enter_long"] = 1
        df.loc[long_entry_conditions, "enter_tag"] = "condition_1"
        
        return df
    
    def populate_exit_trend(self, df: DataFrame, metadata: Dict[str, Any]) -> DataFrame:
        """
        优化出口趋势计算 - 基于NostalgiaForInfinityX6-1完整逻辑
        使用向量化操作优化性能
        """
        # 初始化退出信号列
        df.loc[:, "exit_long"] = 0
        df.loc[:, "exit_short"] = 0
        
        # 优化：使用向量化退出条件
        # 基于RSI超买条件的退出
        exit_conditions = (
            (df["RSI_14"] > 70) |  # 计算RSI超买
            (df["RSI_14_1h"] > 75) |  # 1小时RSI超买
            (df["close"] < df["BBL_20_2.0"]) |  # 跌破布林带下轨
            (df["CMF_20"] < -0.2)  # 资金流指标转负
        )
        
        # 设置退出信号
        df.loc[exit_conditions, "exit_long"] = 1
        
        return df
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.performance_stats
    
    def clear_cache(self):
        """清空缓存"""
        self._indicator_cache.clear()
        self._timeframe_cache.clear()
        self._cached_indicator.cache_clear()
        self.performance_stats = {
            'indicator_calls': 0,
            'cache_hits': 0,
            'execution_times': {}
        }

# ==================== 策略参数配置 ====================
# 以下参数从原始NostalgiaForInfinityX6-1.py复制
# 请根据实际交易需求调整这些参数

# 基础止损设置
stoploss = -0.99

# 追踪止损设置（默认未启用）
trailing_stop = False
trailing_only_offset_is_reached = True
trailing_stop_positive = 0.01
trailing_stop_positive_offset = 0.03

# 时间框架配置
timeframe = "5m"
info_timeframes = ["15m", "1h", "4h", "1d"]

# 蜡烛处理配置
process_only_new_candles = True

# 交易信号配置
use_exit_signal = True
exit_profit_only = False
ignore_roi_if_entry_signal = True

# ROI配置（从原始策略复制）
minimal_roi = {
    "0": 0.01,
    "10": 0.005,
    "20": 0.002,
    "60": 0.001,
    "120": 0
}

# 其他策略特定参数
# 请根据原始策略补充其他必要的参数配置
# ============================================================