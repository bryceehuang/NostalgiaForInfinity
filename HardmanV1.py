from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy.strategy_wrapper import strategy_safe_wrapper
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd
import logging
from typing import Dict, Callable, TypeVar, Any
from functools import wraps

F = TypeVar("F", bound=Callable[..., Any])

def safe_wrapper(message: str = "", default_retval=None):
    """
    Decorator factory for wrapping strategy methods with error handling
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Call strategy_safe_wrapper with proper parameters
            wrapped_func = strategy_safe_wrapper(func, message=message, default_retval=default_retval)
            return wrapped_func(*args, **kwargs)
        return wrapper
    return decorator

logger = logging.getLogger(__name__)

class HardmanV1(IStrategy):
    """
    HardmanV1 多空双向趋势跟踪策略
    核心理念： 
      - 4小时定多空方向，1小时精确定位买卖点
      - 3分钟EMA20跟踪止盈，让利润在趋势中充分奔跑
      - 动态ATR止损，截断亏损
    """
    
    # === 策略基础配置 ===
    INTERFACE_VERSION = 3
    timeframe = '1h'  # 主交易周期（入场/出场判断）
    can_short = True
    RESONANCE_TIMEFRAME = '4h'  # 趋势确认周期
    TRAILING_TIMEFRAME = '3m'  # 趋势跟踪止盈周期[1,3](@ref)
    startup_candle_count = 200
    
    # === 风险管理参数 ===
    stoploss = -0.02
    use_custom_stoploss = True

    # 分层止盈（作为后备保护，主要依赖趋势跟踪止盈）
    minimal_roi = {
        "0": 0.02,   # 立即止盈2%（更现实的短期目标）
        "60": 0.015, # 1小时后止盈1.5%
        "240": 0.005, # 4小时后止盈0.5%
        "1440": 0.00  # 24小时后保本退出
    }
    
    # === 信号控制 ===
    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.005  # 0.5% profit offset to allow exits

    @safe_wrapper(message="多周期指标计算异常", default_retval=pd.DataFrame())
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        计算多周期技术指标（4h趋势 + 1h入场 + 3min止盈）
        """
        try:
            # === 1小时周期指标（交易执行层）===
            dataframe['ema_short_1h'] = ta.EMA(dataframe, timeperiod=12)
            dataframe['ema_long_1h'] = ta.EMA(dataframe, timeperiod=36)
            dataframe['rsi_1h'] = ta.RSI(dataframe, timeperiod=14)
            dataframe['atr_1h'] = ta.ATR(dataframe, timeperiod=14)

            # Initialize multi-timeframe columns with NaN to ensure they exist
            dataframe['ema_short_4h'] = float('nan')
            dataframe['ema_long_4h'] = float('nan')
            dataframe['rsi_4h'] = float('nan')
            dataframe['trend_strength_4h'] = float('nan')
            dataframe['ema20_3min'] = float('nan')

            # === 4小时共振周期指标（趋势过滤层）===
            if self.dp:
                try:
                    resampled_4h = self.dp.get_pair_dataframe(
                        pair=metadata['pair'],
                        timeframe=self.RESONANCE_TIMEFRAME
                    )
                    if not resampled_4h.empty:
                        resampled_4h['ema_short_4h'] = ta.EMA(resampled_4h, timeperiod=20)
                        resampled_4h['ema_long_4h'] = ta.EMA(resampled_4h, timeperiod=50)
                        resampled_4h['rsi_4h'] = ta.RSI(resampled_4h, timeperiod=14)

                        merge_columns = ['ema_short_4h', 'ema_long_4h', 'rsi_4h']
                        dataframe = dataframe.merge(
                            resampled_4h[merge_columns],
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                        dataframe[merge_columns] = dataframe[merge_columns].ffill()

                        # 计算趋势强度指标
                        dataframe['trend_strength_4h'] = (
                            (dataframe['ema_short_4h'] - dataframe['ema_long_4h']) /
                            dataframe['ema_long_4h']
                        ) * 100
                except Exception as e_4h:
                    logger.warning(f"4小时周期数据获取失败 {metadata['pair']}: {str(e_4h)}")

            # === 3分钟趋势跟踪指标（止盈判断层）===
            if self.dp:
                try:
                    resampled_3min = self.dp.get_pair_dataframe(
                        pair=metadata['pair'],
                        timeframe=self.TRAILING_TIMEFRAME
                    )
                    if not resampled_3min.empty:
                        # 计算3分钟EMA20作为趋势跟踪线[1,8](@ref)
                        resampled_3min['ema20_3min'] = ta.EMA(resampled_3min, timeperiod=20)

                        # 将3分钟指标合并到1小时数据（前向填充）
                        dataframe = dataframe.merge(
                            resampled_3min[['ema20_3min']],
                            left_index=True,
                            right_index=True,
                            how='left'
                        )
                        dataframe['ema20_3min'] = dataframe['ema20_3min'].ffill()
                except Exception as e_3min:
                    logger.warning(f"3分钟周期数据获取失败 {metadata['pair']}: {str(e_3min)}")

            return dataframe

        except Exception as e:
            logger.error(f"多周期指标计算异常 {metadata.get('pair', 'unknown')}: {str(e)}")
            # Return the original dataframe to maintain all required columns
            return dataframe

    @safe_wrapper(message="入场信号生成异常", default_retval=pd.DataFrame())
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        多周期共振入场信号[2,4](@ref)
        """
        try:
            if len(dataframe) < 50:
                return dataframe

            # Check if multi-timeframe indicators are available, otherwise fallback to 1h indicators only
            if ('ema_short_4h' in dataframe.columns and not dataframe['ema_short_4h'].isna().all() and
                'ema_long_4h' in dataframe.columns and not dataframe['ema_long_4h'].isna().all() and
                'rsi_4h' in dataframe.columns and not dataframe['rsi_4h'].isna().all()):

                # === 4小时趋势过滤 ===
                major_uptrend = (dataframe['ema_short_4h'] > dataframe['ema_long_4h'])
                major_downtrend = (dataframe['ema_short_4h'] < dataframe['ema_long_4h'])
                uptrend_healthy = (dataframe['rsi_4h'] < 65) & (dataframe['rsi_4h'] > 25)
                downtrend_healthy = (dataframe['rsi_4h'] < 75) & (dataframe['rsi_4h'] > 35)

                uptrend_confirmed = major_uptrend & uptrend_healthy
                downtrend_confirmed = major_downtrend & downtrend_healthy
            else:
                # Fallback to 1h EMA crossover when 4h data not available
                uptrend_confirmed = (dataframe['ema_short_1h'] > dataframe['ema_long_1h'])
                downtrend_confirmed = (dataframe['ema_short_1h'] < dataframe['ema_long_1h'])

            # === 1小时精确入场 ===
            # EMA crossover signals (only when EMAs cross, which happens less frequently)
            ema_long_signal = qtpylib.crossed_above(dataframe['ema_short_1h'], dataframe['ema_long_1h'])
            ema_short_signal = qtpylib.crossed_below(dataframe['ema_short_1h'], dataframe['ema_long_1h'])

            # Additional RSI conditions for more precise entry
            rsi_long_condition = (dataframe['rsi_1h'] < 40) & (dataframe['rsi_1h'] > dataframe['rsi_1h'].shift(1))
            rsi_short_condition = (dataframe['rsi_1h'] > 60) & (dataframe['rsi_1h'] < dataframe['rsi_1h'].shift(1))

            # RSI-only signals as additional options (when no EMA crossover available)
            rsi_only_long = (dataframe['rsi_1h'].shift(1) < 30) & (dataframe['rsi_1h'] > dataframe['rsi_1h'].shift(1))
            rsi_only_short = (dataframe['rsi_1h'].shift(1) > 70) & (dataframe['rsi_1h'] < dataframe['rsi_1h'].shift(1))

            long_signal_1h = (ema_long_signal & rsi_long_condition) | (rsi_only_long & uptrend_confirmed)
            short_signal_1h = (ema_short_signal & rsi_short_condition) | (rsi_only_short & downtrend_confirmed)

            # 成交量确认（如果volume数据有效，则进行确认，否则跳过）
            volume_mean = dataframe['volume'].rolling(20).mean()
            volume_confirmation = (dataframe['volume'] > volume_mean) if not volume_mean.isna().all() else True

            # === 多周期共振入场 ===[3,8](@ref)
            dataframe.loc[uptrend_confirmed & long_signal_1h & volume_confirmation, 'enter_long'] = 1
            dataframe.loc[downtrend_confirmed & short_signal_1h & volume_confirmation, 'enter_short'] = 1

            return dataframe

        except Exception as e:
            logger.error(f"入场信号异常 {metadata.get('pair', 'unknown')}: {str(e)}")
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
            return dataframe

    @safe_wrapper(message="出场信号生成异常", default_retval=pd.DataFrame())
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        趋势跟踪出场策略[1,3,4](@ref)
        核心逻辑：趋势延续时持有，趋势反转时离场
        """
        try:
            if len(dataframe) < 50:
                return dataframe

            # === 技术指标出场（辅助判断）===
            long_technical_exit = (
                qtpylib.crossed_below(dataframe['ema_short_1h'], dataframe['ema_long_1h']) |
                (dataframe['rsi_1h'] > 70)
            )
            short_technical_exit = (
                qtpylib.crossed_above(dataframe['ema_short_1h'], dataframe['ema_long_1h']) |
                (dataframe['rsi_1h'] < 30)
            )

            # Initialize trend exit signals as False
            long_trend_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
            short_trend_exit = pd.Series([False] * len(dataframe), index=dataframe.index)

            # === 趋势跟踪止盈策略（仅当3分钟数据可用时）===
            if 'ema20_3min' in dataframe.columns and not dataframe['ema20_3min'].isna().all():
                # 多头出场：3分钟价格跌破EMA20（趋势跟踪止盈）[1](@ref)
                long_trend_exit = (dataframe['close'] < dataframe['ema20_3min'])

                # 空头出场：3分钟价格升破EMA20（趋势跟踪止盈）
                short_trend_exit = (dataframe['close'] > dataframe['ema20_3min'])

            # === 综合出场条件：趋势跟踪优先 ===[4,5](@ref)
            dataframe.loc[long_trend_exit | long_technical_exit, 'exit_long'] = 1
            dataframe.loc[short_trend_exit | short_technical_exit, 'exit_short'] = 1

            logger.debug(f"趋势跟踪信号 - 多头出场: {long_trend_exit.sum()}, 空头出场: {short_trend_exit.sum()}")

            return dataframe

        except Exception as e:
            logger.error(f"出场信号异常 {metadata.get('pair', 'unknown')}: {str(e)}")
            dataframe['exit_long'] = 0
            dataframe['exit_short'] = 0
            return dataframe

    @safe_wrapper(message="动态止损计算异常", default_retval=-0.02)
    def custom_stoploss(self, pair: str, trade, current_time, current_rate: float,
                       current_profit: float, **kwargs) -> float:
        """
        基于ATR的动态止损（截断亏损）[6,9](@ref)
        """
        try:
            if not self.dp:
                return self.stoploss

            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or 'atr_1h' not in dataframe.columns or dataframe['atr_1h'].isna().all():
                return self.stoploss

            atr_current = dataframe['atr_1h'].iloc[-1]
            atr_mean = dataframe['atr_1h'].replace([float('inf'), float('-inf')], float('nan')).mean()

            # Check if atr_mean is valid
            if pd.isna(atr_mean) or atr_mean <= 0:
                return self.stoploss

            # 动态止损逻辑[5,6](@ref)
            if atr_current > atr_mean * 1.3:  # 高波动环境
                return -0.015  # 收紧止损
            elif atr_current < atr_mean * 0.7:  # 低波动环境
                return -0.035  # 放宽止损
            else:
                return self.stoploss

        except Exception as e:
            logger.error(f"动态止损计算异常 {pair}: {str(e)}")
            return self.stoploss

    def leverage(self, pair: str, current_time, current_rate: float,
                 proposed_leverage: float, max_leverage: float, 
                 entry_tag: str, side: str, **kwargs) -> float:
        """
        动态杠杆控制：趋势强时适度增加杠杆[3](@ref)
        """
        try:
            if not self.dp:
                return 1.0

            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or 'trend_strength_4h' not in dataframe.columns or dataframe['trend_strength_4h'].isna().all():
                return 1.0

            trend_strength = abs(dataframe['trend_strength_4h'].iloc[-1])

            # Check if trend_strength is valid
            if pd.isna(trend_strength):
                return 1.0

            if trend_strength > 3.0:
                return min(3.0, max_leverage)
            elif trend_strength > 1.5:
                return min(2.0, max_leverage)
            else:
                return 1.0

        except Exception as e:
            logger.error(f"杠杆计算异常 {pair}: {str(e)}")
            return 1.0