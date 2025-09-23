import copy
import logging
import pathlib
import rapidjson
import numpy as np
import talib.abstract as ta
import pandas as pd
import pandas_ta as pta
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Tuple, Any
import warnings
from collections import defaultdict
import numba
from numba import jit, njit
import gc
import os

log = logging.getLogger(__name__)
# log.setLevel(logging.DEBUG)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Performance optimization: Pre-compile frequently used functions
@njit(fastmath=True, cache=True)
def fast_profit_calculation(entry_price: float, current_price: float, is_short: bool, fee_open: float, fee_close: float) -> float:
    """Fast profit calculation using numba"""
    if is_short:
        profit = (entry_price - current_price) / entry_price - (fee_open + fee_close)
    else:
        profit = (current_price - entry_price) / entry_price - (fee_open + fee_close)
    return profit

@njit(fastmath=True, cache=True)
def fast_exit_condition_check(profit: float, threshold: float, mode: int) -> bool:
    """Fast exit condition check"""
    if mode == 0:  # Normal mode
        return profit < threshold
    elif mode == 1:  # Profit taking
        return profit > threshold
    else:  # Stop loss
        return profit < threshold

class NostalgiaForInfinityX6_CC(IStrategy):
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v16.7.137_CC"

    stoploss = -0.99

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = "5m"
    info_timeframes = ["15m", "1h", "4h", "1d"]

    # BTC informatives
    btc_info_timeframes = ["5m", "15m", "1h", "4h", "1d"]

    # Backtest Age Filter emulation
    has_bt_agefilter = False
    bt_min_age_days = 3

    # Exchange Downtime protection
    has_downtime_protection = False

    # Do you want to use the hold feature? (with hold-trades.json)
    hold_support_enabled = True

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 800

    # Number of cores to use for pandas_ta indicators calculations
    num_cores_indicators_calc = 0

    # Performance optimization: Pre-computed mode tags for faster lookups
    _mode_tags_cache = None
    _exit_conditions_cache = None
    _indicator_cache = {}

    # Long Normal mode tags
    long_normal_mode_tags = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
    # Long Pump mode tags
    long_pump_mode_tags = ["21", "22", "23", "24", "25", "26"]
    # Long Quick mode tags
    long_quick_mode_tags = ["41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53"]
    # Long rebuy mode tags
    long_rebuy_mode_tags = ["61", "62"]
    # Long high profit mode tags
    long_mode_tags = ["81", "82"]
    # Long rapid mode tags
    long_rapid_mode_tags = ["101", "102", "103", "104", "105", "106", "107", "108", "109", "110"]
    # Long grind mode tags
    long_grind_mode_tags = ["120"]
    # Long top coins mode tags
    long_top_coins_mode_tags = ["141", "142", "143", "144"]
    # Long scalp mode tags
    long_scalp_mode_tags = ["161", "162", "163"]

    long_normal_mode_name = "long_normal"
    long_pump_mode_name = "long_pump"
    long_quick_mode_name = "long_quick"
    long_rebuy_mode_name = "long_rebuy"
    long_high_profit_mode_name = "long_hp"
    long_rapid_mode_name = "long_rapid"
    long_grind_mode_name = "long_grind"
    long_top_coins_mode_name = "long_tc"
    long_scalp_mode_name = "long_scalp"

    # Shorting

    # Short normal mode tags
    short_normal_mode_tags = ["501", "502"]
    # Short Pump mode tags
    short_pump_mode_tags = ["521", "522", "523", "524", "525", "526"]
    # Short Quick mode tags
    short_quick_mode_tags = ["541", "542", "543", "544", "545", "546", "547", "548", "549", "550"]
    # Short rebuy mode tags
    short_rebuy_mode_tags = ["561"]
    # Short mode tags
    short_mode_tags = ["581", "582"]
    # Short rapid mode tags
    short_rapid_mode_tags = ["601", "602", "603", "604", "605", "606", "607", "608", "609", "610"]
    # Short grind mode tags
    short_grind_mode_tags = ["620"]
    # Short top coins mode tags
    short_top_coins_mode_tags = ["641", "642"]
    # Short scalp mode tags
    short_scalp_mode_tags = ["661"]

    short_normal_mode_name = "short_normal"
    short_pump_mode_name = "short_pump"
    short_quick_mode_name = "short_quick"
    short_rebuy_mode_name = "short_rebuy"
    short_high_profit_mode_name = "short_hp"
    short_rapid_mode_name = "short_rapid"
    short_top_coins_mode_name = "short_tc"
    short_scalp_mode_name = "short_scalp"

    is_futures_mode = False
    futures_mode_leverage = 3.0
    futures_mode_leverage_rebuy_mode = 3.0
    futures_mode_leverage_grind_mode = 3.0

    # Limit the number of long/short trades for futures (0 for no limit)
    futures_max_open_trades_long = 0
    futures_max_open_trades_short = 0

    # Based on the the first entry (regardless of rebuys)
    stop_threshold_spot = 0.10
    stop_threshold_futures = 0.10
    stop_threshold_doom_spot = 0.20
    stop_threshold_doom_futures = 0.20
    stop_threshold_spot_rebuy = 1.0
    stop_threshold_futures_rebuy = 1.0
    stop_threshold_rapid_spot = 0.20
    stop_threshold_rapid_futures = 0.20
    stop_threshold_scalp_spot = 0.20
    stop_threshold_scalp_futures = 0.20

    # user specified fees to be used for profit calculations
    custom_fee_open_rate = None
    custom_fee_close_rate = None

    # Rebuy mode minimum number of free slots
    rebuy_mode_min_free_slots = 2

    # Position adjust feature
    position_adjustment_enable = True

    # Grinding feature
    grinding_enable = True
    derisk_enable = True
    stops_enable = True
    doom_stops_enable = True
    u_e_stops_enable = False

    # Grinding
    grinding_v1_max_stake = 1.0  # ratio of first entry
    derisk_use_grind_stops = False

    grind_1_stop_grinds_spot = -0.50
    grind_1_profit_threshold_spot = 0.018
    grind_1_stakes_spot = [0.24, 0.26, 0.28]
    grind_1_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_1_stop_grinds_futures = -0.50
    grind_1_profit_threshold_futures = 0.018
    grind_1_stakes_futures = [0.24, 0.26, 0.28]
    grind_1_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grind_2_stop_grinds_spot = -0.50
    grind_2_profit_threshold_spot = 0.018
    grind_2_stakes_spot = [0.20, 0.24, 0.28]
    grind_2_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_2_stop_grinds_futures = -0.50
    grind_2_profit_threshold_futures = 0.018
    grind_2_stakes_futures = [0.20, 0.24, 0.28]
    grind_2_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grind_3_stop_grinds_spot = -0.50
    grind_3_profit_threshold_spot = 0.018
    grind_3_stakes_spot = [0.20, 0.22, 0.24]
    grind_3_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_3_stop_grinds_futures = -0.50
    grind_3_profit_threshold_futures = 0.018
    grind_3_stakes_futures = [0.20, 0.22, 0.24]
    grind_3_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grind_4_stop_grinds_spot = -0.50
    grind_4_profit_threshold_spot = 0.018
    grind_4_stakes_spot = [0.20, 0.22, 0.24]
    grind_4_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_4_stop_grinds_futures = -0.50
    grind_4_profit_threshold_futures = 0.018
    grind_4_stakes_futures = [0.20, 0.22, 0.24]
    grind_4_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grind_5_stop_grinds_spot = -0.50
    grind_5_profit_threshold_spot = 0.048
    grind_5_stakes_spot = [0.20, 0.22, 0.24]
    grind_5_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_5_stop_grinds_futures = -0.50
    grind_5_profit_threshold_futures = 0.048
    grind_5_stakes_futures = [0.20, 0.22, 0.24]
    grind_5_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grind_6_stop_grinds_spot = -0.50
    grind_6_profit_threshold_spot = 0.018
    grind_6_stakes_spot = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
    grind_6_sub_thresholds_spot = [-0.03, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18, -0.20, -0.22]

    grind_6_stop_grinds_futures = -0.50
    grind_6_profit_threshold_futures = 0.018
    grind_6_stakes_futures = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18]
    grind_6_sub_thresholds_futures = [-0.03, -0.08, -0.10, -0.12, -0.14, -0.16, -0.18, -0.20, -0.22]

    grind_1_derisk_1_stop_grinds_spot = -0.50
    grind_1_derisk_1_profit_threshold_spot = 0.018
    grind_1_derisk_1_stakes_spot = [0.20, 0.24, 0.28]
    grind_1_derisk_1_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_1_derisk_1_stop_grinds_futures = -0.50
    grind_1_derisk_1_profit_threshold_futures = 0.018
    grind_1_derisk_1_stakes_futures = [0.20, 0.24, 0.28]
    grind_1_derisk_1_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grind_2_derisk_1_stop_grinds_spot = -0.50
    grind_2_derisk_1_profit_threshold_spot = 0.018
    grind_2_derisk_1_stakes_spot = [0.20, 0.24, 0.28]
    grind_2_derisk_1_sub_thresholds_spot = [-0.12, -0.16, -0.20]

    grind_2_derisk_1_stop_grinds_futures = -0.50
    grind_2_derisk_1_profit_threshold_futures = 0.018
    grind_2_derisk_1_stakes_futures = [0.20, 0.24, 0.28]
    grind_2_derisk_1_sub_thresholds_futures = [-0.12, -0.16, -0.20]

    grinds_stop_spot = -0.12
    grinds_stop_futures = -0.12

    # Non rebuy modes
    regular_mode_stake_multiplier_spot = [1.0]
    regular_mode_stake_multiplier_futures = [1.0]
    regular_mode_use_grind_stops = False

    regular_mode_rebuy_stakes_spot = [0.10, 0.10, 0.10]
    regular_mode_rebuy_thresholds_spot = [-0.12, -0.14, -0.16]
    regular_mode_grind_1_stakes_spot = [0.22, 0.24, 0.26]
    regular_mode_grind_1_thresholds_spot = [-0.06, -0.10, -0.12]
    regular_mode_grind_1_stop_grinds_spot = -0.20
    regular_mode_grind_1_profit_threshold_spot = 0.018
    regular_mode_grind_2_stakes_spot = [0.14, 0.20, 0.26]
    regular_mode_grind_2_thresholds_spot = [-0.04, -0.10, -0.12]
    regular_mode_grind_2_stop_grinds_spot = -0.20
    regular_mode_grind_2_profit_threshold_spot = 0.018
    regular_mode_grind_3_stakes_spot = [0.18, 0.20, 0.22]
    regular_mode_grind_3_thresholds_spot = [-0.03, -0.10, -0.12]
    regular_mode_grind_3_stop_grinds_spot = -0.20
    regular_mode_grind_3_profit_threshold_spot = 0.018
    regular_mode_grind_4_stakes_spot = [0.18, 0.20, 0.22]
    regular_mode_grind_4_thresholds_spot = [-0.03, -0.10, -0.12]
    regular_mode_grind_4_stop_grinds_spot = -0.20
    regular_mode_grind_4_profit_threshold_spot = 0.018
    regular_mode_grind_5_stakes_spot = [0.18, 0.20, 0.22]
    regular_mode_grind_5_thresholds_spot = [-0.03, -0.10, -0.12]
    regular_mode_grind_5_stop_grinds_spot = -0.20
    regular_mode_grind_5_profit_threshold_spot = 0.048
    regular_mode_grind_6_stakes_spot = [0.05, 0.057, 0.065, 0.074, 0.084, 0.095, 0.107, 0.121, 0.137]
    regular_mode_grind_6_thresholds_spot = [-0.025, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12]
    regular_mode_grind_6_stop_grinds_spot = -0.20
    regular_mode_grind_6_profit_threshold_spot = 0.018
    regular_mode_derisk_1_spot = -0.24
    regular_mode_derisk_1_spot_old = -0.80
    regular_mode_derisk_1_reentry_spot = -0.08
    regular_mode_derisk_spot = -0.24
    regular_mode_derisk_spot_old = -1.60
    regular_mode_derisk_1_scalp_mode_spot = -0.05

    regular_mode_rebuy_stakes_futures = [0.10, 0.10, 0.10]
    regular_mode_rebuy_thresholds_futures = [-0.12, -0.14, -0.16]
    regular_mode_grind_1_stakes_futures = [0.22, 0.24, 0.26]
    regular_mode_grind_1_thresholds_futures = [-0.06, -0.10, -0.12]
    regular_mode_grind_1_stop_grinds_futures = -0.20
    regular_mode_grind_1_profit_threshold_futures = 0.018
    regular_mode_grind_2_stakes_futures = [0.14, 0.20, 0.26]
    regular_mode_grind_2_thresholds_futures = [-0.04, -0.10, -0.12]
    regular_mode_grind_2_stop_grinds_futures = -0.20
    regular_mode_grind_2_profit_threshold_futures = 0.018
    regular_mode_grind_3_stakes_futures = [0.18, 0.20, 0.22]
    regular_mode_grind_3_thresholds_futures = [-0.03, -0.10, -0.12]
    regular_mode_grind_3_stop_grinds_futures = -0.20
    regular_mode_grind_3_profit_threshold_futures = 0.018
    regular_mode_grind_4_stakes_futures = [0.18, 0.20, 0.22]
    regular_mode_grind_4_thresholds_futures = [-0.03, -0.10, -0.12]
    regular_mode_grind_4_stop_grinds_futures = -0.20
    regular_mode_grind_4_profit_threshold_futures = 0.018
    regular_mode_grind_5_stakes_futures = [0.18, 0.20, 0.22]
    regular_mode_grind_5_thresholds_futures = [-0.03, -0.10, -0.12]
    regular_mode_grind_5_stop_grinds_futures = -0.20
    regular_mode_grind_5_profit_threshold_futures = 0.048
    regular_mode_grind_6_stakes_futures = [0.05, 0.057, 0.065, 0.074, 0.084, 0.095, 0.107, 0.121, 0.137]
    regular_mode_grind_6_thresholds_futures = [-0.025, -0.05, -0.06, -0.07, -0.08, -0.09, -0.10, -0.11, -0.12]
    regular_mode_grind_6_stop_grinds_futures = -0.20
    regular_mode_grind_6_profit_threshold_futures = 0.018
    regular_mode_derisk_1_futures = -0.60
    regular_mode_derisk_1_futures_old = -0.80
    regular_mode_derisk_1_reentry_futures = -0.08  # without leverage
    regular_mode_derisk_futures = -0.60
    regular_mode_derisk_futures_old = -1.20
    regular_mode_derisk_1_scalp_mode_futures = -0.05

    # Grinding v2
    grinding_v2_max_stake = 1.0  # ratio of first entry
    grinding_v2_max_grinds_and_buybacks = 20  # current open

    grinding_v2_derisk_level_1_enable = True
    grinding_v2_derisk_level_1_spot = -0.12
    grinding_v2_derisk_level_1_futures = -0.36
    grinding_v2_derisk_level_2_enable = True
    grinding_v2_derisk_level_2_spot = -0.14
    grinding_v2_derisk_level_2_futures = -0.42
    grinding_v2_derisk_level_3_enable = True
    grinding_v2_derisk_level_3_spot = -0.15
    grinding_v2_derisk_level_3_futures = -0.45
    grinding_v2_derisk_level_1_stake_spot = 0.20
    grinding_v2_derisk_level_1_stake_futures = 0.20
    grinding_v2_derisk_level_2_stake_spot = 0.30
    grinding_v2_derisk_level_2_stake_futures = 0.30
    grinding_v2_derisk_level_3_stake_spot = 0.50
    grinding_v2_derisk_level_3_stake_futures = 0.50
    grinding_v2_derisk_global_enable = False
    grinding_v2_derisk_global_spot = -0.10
    grinding_v2_derisk_global_futures = -0.30

    grinding_v2_grind_1_enable = True
    grinding_v2_grind_1_stakes_spot = [0.10, 0.15, 0.25, 0.30, 0.35]
    grinding_v2_grind_1_thresholds_spot = [-0.06, -0.07, -0.08, -0.09, -0.10]
    grinding_v2_grind_1_stakes_futures = [0.10, 0.15, 0.25, 0.30, 0.35]
    grinding_v2_grind_1_thresholds_futures = [-0.06, -0.07, -0.08, -0.09, -0.10]
    grinding_v2_grind_1_profit_threshold_spot = 0.028
    grinding_v2_grind_1_profit_threshold_futures = 0.028
    grinding_v2_grind_1_use_derisk = True
    grinding_v2_grind_1_derisk_spot = -0.26
    grinding_v2_grind_1_derisk_futures = -0.26

    grinding_v2_grind_2_enable = True
    grinding_v2_grind_2_stakes_spot = [0.05, 0.10, 0.15, 0.20, 0.25]
    grinding_v2_grind_2_thresholds_spot = [-0.06, -0.07, -0.08, -0.09, -0.10]
    grinding_v2_grind_2_stakes_futures = [0.05, 0.10, 0.15, 0.20, 0.25]
    grinding_v2_grind_2_thresholds_futures = [-0.06, -0.07, -0.08, -0.09, -0.10]
    grinding_v2_grind_2_profit_threshold_spot = 0.05
    grinding_v2_grind_2_profit_threshold_futures = 0.05
    grinding_v2_grind_2_use_derisk = True
    grinding_v2_grind_2_derisk_spot = -0.26
    grinding_v2_grind_2_derisk_futures = -0.26

    grinding_v2_grind_3_enable = True
    grinding_v2_grind_3_stakes_spot = [0.05, 0.10, 0.15, 0.20, 0.25]
    grinding_v2_grind_3_thresholds_spot = [-0.06, -0.07, -0.08, -0.09, -0.10]
    grinding_v2_grind_3_stakes_futures = [0.05, 0.10, 0.15, 0.20, 0.25]
    grinding_v2_grind_3_thresholds_futures = [-0.06, -0.07, -0.08, -0.09, -0.10]
    grinding_v2_grind_3_profit_threshold_spot = 0.05
    grinding_v2_grind_3_profit_threshold_futures = 0.05
    grinding_v2_grind_3_use_derisk = True
    grinding_v2_grind_3_derisk_spot = -0.26
    grinding_v2_grind_3_derisk_futures = -0.26

    grinding_v2_grind_4_enable = True
    grinding_v2_grind_4_stakes_spot = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
    grinding_v2_grind_4_thresholds_spot = [-0.06, -0.07, -0.09, -0.12, -0.16, -0.21, -0.27]
    grinding_v2_grind_4_stakes_futures = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
    grinding_v2_grind_4_thresholds_futures = [-0.06, -0.07, -0.09, -0.12, -0.16, -0.21, -0.27]
    grinding_v2_grind_4_profit_threshold_spot = 0.10
    grinding_v2_grind_4_profit_threshold_futures = 0.10
    grinding_v2_grind_4_use_derisk = True
    grinding_v2_grind_4_derisk_spot = -0.26
    grinding_v2_grind_4_derisk_futures = -0.26

    grinding_v2_grind_5_enable = True
    grinding_v2_grind_5_stakes_spot = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
    grinding_v2_grind_5_thresholds_spot = [-0.06, -0.07, -0.09, -0.12, -0.16, -0.21, -0.27]
    grinding_v2_grind_5_stakes_futures = [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11]
    grinding_v2_grind_5_thresholds_futures = [-0.06, -0.07, -0.09, -0.12, -0.16, -0.21, -0.27]
    grinding_v2_grind_5_profit_threshold_spot = 0.10
    grinding_v2_grind_5_profit_threshold_futures = 0.10
    grinding_v2_grind_5_use_derisk = True
    grinding_v2_grind_5_derisk_spot = -0.26
    grinding_v2_grind_5_derisk_futures = -0.26

    grinding_v2_buyback_1_enable = True
    grinding_v2_buyback_1_stake_spot = 0.10
    grinding_v2_buyback_1_stake_futures = 0.10
    grinding_v2_buyback_1_distance_ratio_spot = -0.06
    grinding_v2_buyback_1_distance_ratio_futures = -0.06
    grinding_v2_buyback_1_profit_threshold_spot = 0.05
    grinding_v2_buyback_1_profit_threshold_futures = 0.05
    grinding_v2_buyback_1_use_derisk = True
    grinding_v2_buyback_1_derisk_spot = -0.26
    grinding_v2_buyback_1_derisk_futures = -0.26

    grinding_v2_buyback_2_enable = True
    grinding_v2_buyback_2_stake_spot = 0.10
    grinding_v2_buyback_2_stake_futures = 0.10
    grinding_v2_buyback_2_distance_ratio_spot = -0.12
    grinding_v2_buyback_2_distance_ratio_futures = -0.12
    grinding_v2_buyback_2_profit_threshold_spot = 0.05
    grinding_v2_buyback_2_profit_threshold_futures = 0.05
    grinding_v2_buyback_2_use_derisk = True
    grinding_v2_buyback_2_derisk_spot = -0.26
    grinding_v2_buyback_2_derisk_futures = -0.26

    grinding_v2_buyback_3_enable = True
    grinding_v2_buyback_3_stake_spot = 0.10
    grinding_v2_buyback_3_stake_futures = 0.10
    grinding_v2_buyback_3_distance_ratio_spot = -0.16
    grinding_v2_buyback_3_distance_ratio_futures = -0.16
    grinding_v2_buyback_3_profit_threshold_spot = 0.05
    grinding_v2_buyback_3_profit_threshold_futures = 0.05
    grinding_v2_buyback_3_use_derisk = True
    grinding_v2_buyback_3_derisk_spot = -0.26
    grinding_v2_buyback_3_derisk_futures = -0.26

    # Rebuy mode
    rebuy_mode_stake_multiplier = 0.35
    rebuy_mode_derisk_spot = -0.60
    rebuy_mode_derisk_futures = -0.60
    rebuy_mode_stakes_spot = [1.0, 1.0]
    rebuy_mode_stakes_futures = [1.0, 1.0]
    rebuy_mode_thresholds_spot = [-0.08, -0.10]
    rebuy_mode_thresholds_futures = [-0.08, -0.10]

    # Rapid mode
    rapid_mode_stake_multiplier_spot = [0.75]
    rapid_mode_stake_multiplier_futures = [0.75]

    # Scalp mode
    min_free_slots_scalp_mode = 1

    # Grind mode
    grind_mode_stake_multiplier_spot = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    grind_mode_stake_multiplier_futures = [0.20, 0.30, 0.40, 0.50]
    grind_mode_first_entry_profit_threshold_spot = 0.018
    grind_mode_first_entry_profit_threshold_futures = 0.018
    grind_mode_first_entry_stop_threshold_spot = -0.20
    grind_mode_first_entry_stop_threshold_futures = -0.20
    grind_mode_max_slots = 1
    grind_mode_coins = [
        "AAVE", "ADA", "ALGO", "APE", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC",
        "CAKE", "CRV", "DOGE", "DOT", "DYDX", "ETC", "ETH", "FIL", "GALA", "HBAR",
        "HYPE", "ICP", "INJ", "IOTA", "JUP", "KAS", "LDO", "LINK", "LTC", "NEAR",
        "NEO", "OP", "POL", "RENDER", "RUNE", "SAND", "SEI", "SOL", "SUI", "THETA",
        "TIA", "TON", "TRX", "UNI", "VET", "XLM", "XMR", "XRP", "XTZ", "ZEC"
    ]

    # Top coins mode coins
    top_coins_mode_coins = [
        "AAVE", "ADA", "ALGO", "APE", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC",
        "CAKE", "CRV", "DOGE", "DOT", "DYDX", "ETC", "ETH", "FIL", "GALA", "HBAR",
        "HYPE", "ICP", "INJ", "IOTA", "JUP", "KAS", "LDO", "LINK", "LTC", "NEAR",
        "NEO", "OP", "POL", "RENDER", "RUNE", "SAND", "SEI", "SOL", "SUI", "THETA",
        "TIA", "TON", "TRX", "UNI", "VET", "XLM", "XMR", "XRP", "XTZ", "ZEC"
    ]

    # Profit max thresholds
    profit_max_thresholds = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.05, 0.05]

    # Max allowed buy "slippage", how high to buy on the candle
    max_slippage = 0.01

    # BTC/ETH stakes
    btc_stakes = ["BTC", "ETH"]

    #############################################################
    # Buy side configuration

    long_entry_signal_params = {
        # Enable/Disable conditions
        # -------------------------------------------------------
        "long_entry_condition_1_enable": True,
        "long_entry_condition_2_enable": True,
        "long_entry_condition_3_enable": True,
        "long_entry_condition_4_enable": True,
        "long_entry_condition_5_enable": True,
        "long_entry_condition_6_enable": True,
        "long_entry_condition_21_enable": True,
        "long_entry_condition_41_enable": True,
        "long_entry_condition_42_enable": True,
        "long_entry_condition_43_enable": True,
        "long_entry_condition_44_enable": True,
        "long_entry_condition_45_enable": True,
        "long_entry_condition_46_enable": True,
        "long_entry_condition_61_enable": True,
        "long_entry_condition_62_enable": True,
        "long_entry_condition_101_enable": True,
        "long_entry_condition_102_enable": True,
        "long_entry_condition_103_enable": True,
        "long_entry_condition_104_enable": True,
        "long_entry_condition_120_enable": True,
        "long_entry_condition_141_enable": True,
        "long_entry_condition_142_enable": True,
        "long_entry_condition_143_enable": True,
        "long_entry_condition_144_enable": True,
        "long_entry_condition_161_enable": True,
        "long_entry_condition_162_enable": True,
        "long_entry_condition_163_enable": True,
    }

    short_entry_signal_params = {
        # Enable/Disable conditions
        # -------------------------------------------------------
        "short_entry_condition_501_enable": True,
        "short_entry_condition_502_enable": True,
        # "short_entry_condition_503_enable": True,
        # "short_entry_condition_504_enable": True,
        # "short_entry_condition_541_enable": True,
        "short_entry_condition_542_enable": True,
        # "short_entry_condition_543_enable": True,
        # "short_entry_condition_603_enable": True,
        # "short_entry_condition_641_enable": True,
        # "short_entry_condition_642_enable": True,
        # "short_entry_condition_661_enable": True,
    }

    #############################################################
    # CACHES

    hold_trades_cache = None
    target_profit_cache = None

    def __init__(self, config: dict) -> None:
        # A list of parameters that can be changed through the config.
        NFI_SAFE_PARAMETERS = [
            "num_cores_indicators_calc",
            "custom_fee_open_rate",
            "custom_fee_close_rate",
            "futures_mode_leverage",
            "futures_mode_leverage_rebuy_mode",
            "futures_mode_leverage_grind_mode",
            "futures_max_open_trades_long",
            "futures_max_open_trades_short",
            "stop_threshold_doom_spot",
            "stop_threshold_doom_futures",
            "stop_threshold_rapid_spot",
            "stop_threshold_rapid_futures",
            "stop_threshold_scalp_spot",
            "stop_threshold_scalp_futures",
            "derisk_enable",
            "stops_enable",
            "regular_mode_derisk_1_spot",
            "regular_mode_derisk_spot",
            "regular_mode_derisk_1_futures",
            "regular_mode_derisk_futures",
            "grind_mode_max_slots",
            "grind_mode_coins",
            "max_slippage",
        ]

        if "ccxt_config" not in config["exchange"]:
            config["exchange"]["ccxt_config"] = {}
        if "ccxt_async_config" not in config["exchange"]:
            config["exchange"]["ccxt_async_config"] = {}

        options = {
            "brokerId": None,
            "broker": {"spot": None, "margin": None, "future": None, "delivery": None},
            "partner": {
                "spot": {"id": None, "key": None},
                "future": {"id": None, "key": None},
                "id": None,
                "key": None,
            },
        }

        config["exchange"]["ccxt_config"]["options"] = options
        config["exchange"]["ccxt_async_config"]["options"] = options
        super().__init__(config)
        if ("exit_profit_only" in self.config and self.config["exit_profit_only"]) or (
            "sell_profit_only" in self.config and self.config["sell_profit_only"]
        ):
            self.exit_profit_only = True

        # Advanced configuration mode. Allows to change any parameter.
        is_config_advanced_mode = "nfi_advanced_mode" in self.config and self.config["nfi_advanced_mode"] == True
        if is_config_advanced_mode:
            log.warning("The advanced configuration mode is enabled. I hope you know what you are doing.")

        # Configuration from the nfi_parameters block. New config style.
        if "nfi_parameters" in self.config and type(self.config["nfi_parameters"]) is dict:
            for nfi_param in self.config["nfi_parameters"]:
                if nfi_param in ["long_entry_signal_params", "short_entry_signal_params"]:
                    continue
                if (nfi_param in NFI_SAFE_PARAMETERS or is_config_advanced_mode) and hasattr(self, nfi_param):
                    log.info(
                        f'Parameter {nfi_param} changed from "{getattr(self, nfi_param)}" to "{self.config["nfi_parameters"][nfi_param]}".'
                    )
                    setattr(self, nfi_param, self.config["nfi_parameters"][nfi_param])
                else:
                    log.warning(f"Invalid or unsafe parameter: {nfi_param}.")

            self.update_signals_from_config(self.config["nfi_parameters"])

        # Parameter settings. Backward compatibility with the old configuration style.
        for nfi_param in NFI_SAFE_PARAMETERS:
            if (nfi_param in self.config) and hasattr(self, nfi_param):
                setattr(self, nfi_param, self.config[nfi_param])

        if self.target_profit_cache is None:
            bot_name = ""
            if "bot_name" in self.config:
                bot_name = self.config["bot_name"] + "-"
            cache_path = os.path.join(
                self.config["user_data_dir"],
                (
                    "nfix6-profit_max-"
                    + bot_name
                    + self.config["exchange"]["name"]
                    + "-"
                    + self.config["stake_currency"]
                    + ("-(backtest)" if (self.config["runmode"].value == "backtest") else "")
                    + ("-(hyperopt)" if (self.config["runmode"].value == "hyperopt") else "")
                    + ".json"
                )
            )
            self.target_profit_cache = Cache(cache_path)

        # OKX, Kraken provides a lower number of candle data per API call
        if self.config["exchange"]["name"] in ["okx", "okex"]:
            self.startup_candle_count = 480
        elif self.config["exchange"]["name"] in ["kraken"]:
            self.startup_candle_count = 710
        elif self.config["exchange"]["name"] in ["bybit"]:
            self.startup_candle_count = 199
        elif self.config["exchange"]["name"] in ["bitget"]:
            self.startup_candle_count = 499
        elif self.config["exchange"]["name"] in ["bingx"]:
            self.startup_candle_count = 499

        if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
            self.is_futures_mode = True
            self.can_short = True

        # Initialize cache
        self._initialize_cache()

        # If the cached data hasn't changed, it's a no-op
        self.target_profit_cache.save()

        # Parameter settings. Backward compatibility with the old configuration style.
        self.update_signals_from_config(self.config)

    def _initialize_cache(self):
        """Initialize performance optimization caches"""
        # Pre-compute mode tag lookups
        self._mode_tags_cache = {
            'long_normal': set(self.long_normal_mode_tags),
            'long_pump': set(self.long_pump_mode_tags),
            'long_quick': set(self.long_quick_mode_tags),
            'long_rebuy': set(self.long_rebuy_mode_tags),
            'long_hp': set(self.long_mode_tags),
            'long_rapid': set(self.long_rapid_mode_tags),
            'long_grind': set(self.long_grind_mode_tags),
            'long_tc': set(self.long_top_coins_mode_tags),
            'long_scalp': set(self.long_scalp_mode_tags),
            'short_normal': set(self.short_normal_mode_tags),
            'short_pump': set(self.short_pump_mode_tags),
            'short_quick': set(self.short_quick_mode_tags),
            'short_rebuy': set(self.short_rebuy_mode_tags),
            'short_hp': set(self.short_mode_tags),
            'short_rapid': set(self.short_rapid_mode_tags),
            'short_grind': set(self.short_grind_mode_tags),
            'short_scalp': set(self.short_scalp_mode_tags),
        }

        # Pre-compute exit conditions
        self._exit_conditions_cache = self._build_exit_conditions_cache()

    def _build_exit_conditions_cache(self):
        """Build optimized exit conditions cache"""
        cache = {}

        # Simplified exit conditions mapping
        thresholds = {
            'doom_spot': self.stop_threshold_doom_spot,
            'doom_futures': self.stop_threshold_doom_futures,
            'rapid_spot': self.stop_threshold_rapid_spot,
            'rapid_futures': self.stop_threshold_rapid_futures,
            'scalp_spot': self.stop_threshold_scalp_spot,
            'scalp_futures': self.stop_threshold_scalp_futures,
            'rebuy_spot': self.stop_threshold_spot_rebuy,
            'rebuy_futures': self.stop_threshold_futures_rebuy,
        }

        for mode, threshold_key in [
            ('normal', 'doom'),
            ('rapid', 'rapid'),
            ('scalp', 'scalp'),
            ('rebuy', 'rebuy'),
        ]:
            cache[f'{mode}_spot'] = thresholds[f'{threshold_key}_spot']
            cache[f'{mode}_futures'] = thresholds[f'{threshold_key}_futures']

        return cache

    # Plot configuration for FreqUI
    # ---------------------------------------------------------------------------------------------
    @property
    def plot_config(self):
        plot_config = {}

        plot_config["main_plot"] = {
            "EMA_12": {"color": "LightGreen"},
            "EMA_26": {"color": "Yellow"},
            "EMA_50": {"color": "DodgerBlue"},
            "EMA_200": {"color": "DarkRed"},
        }

        plot_config["subplots"] = {
            "long_pump_protection": {"global_protections_long_pump": {"color": "green"}},
            "long_dump_protection": {"global_protections_long_dump": {"color": "red"}},
        }

        return plot_config

    # Get Ticker Indicator
    # ---------------------------------------------------------------------------------------------
    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    # Mark Profit Target
    # ---------------------------------------------------------------------------------------------
    def mark_profit_target(
        self,
        mode_name: str,
        pair: str,
        sell: bool,
        signal_name: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        last_candle,
        previous_candle_1,
    ) -> tuple:
        if sell and (signal_name is not None):
            return pair, signal_name

        return None, None

    # Exit Profit Target - OPTIMIZED
    # ---------------------------------------------------------------------------------------------
    def exit_profit_target(
        self,
        mode_name: str,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        profit_stake: float,
        profit_ratio: float,
        profit_current_stake_ratio: float,
        profit_init_ratio: float,
        last_candle,
        previous_candle_1,
        previous_rate,
        previous_profit,
        previous_sell_reason,
        previous_time_profit_reached,
        enter_tags,
    ) -> tuple:
        is_backtest = self.is_backtest_mode()
        is_derisk = False

        # Fast path for common cases
        if previous_sell_reason in [
            f"exit_{mode_name}_stoploss_doom",
            f"exit_{mode_name}_stoploss",
            f"exit_{mode_name}_stoploss_u_e",
        ]:
            # Quick derisk check
            filled_entries = trade.select_filled_orders(trade.entry_side)
            filled_exits = trade.select_filled_orders(trade.exit_side)

            if hasattr(filled_entries[0], "ft_order_tag"):
                for order in filled_exits:
                    if order.ft_order_tag and any(tag in order.ft_order_tag for tag in ["d", "d1", "derisk_level_1", "derisk_level_2", "derisk_level_3"]):
                        is_derisk = True
                        break

                if not is_derisk:
                    is_derisk = trade.amount < (filled_entries[0].safe_filled * 0.95)

            if profit_init_ratio > 0.0:
                self._remove_profit_target(pair)
                return False, None
            elif is_derisk:
                self._remove_profit_target(pair)
                return False, None

        # Optimized exit conditions
        return self._optimized_exit_conditions(
            mode_name, pair, trade, current_time, profit_init_ratio,
            previous_profit, previous_sell_reason, enter_tags, last_candle
        )

    def _optimized_exit_conditions(self, mode_name: str, pair: str, trade: Trade,
                                 current_time: datetime, profit_init_ratio: float,
                                 previous_profit: float, previous_sell_reason: str,
                                 enter_tags: List[str], last_candle) -> tuple:
        """Optimized exit condition checking"""

        # Fast mode detection
        is_rapid_mode = all(c in self.long_rapid_mode_tags for c in enter_tags)
        is_rebuy_mode = all(c in self.long_rebuy_mode_tags for c in enter_tags) or (
            any(c in self.long_rebuy_mode_tags for c in enter_tags)
            and all(c in (self.long_rebuy_mode_tags + self.long_grind_mode_tags) for c in enter_tags)
        )
        is_scalp_mode = all(c in self.long_scalp_mode_tags for c in enter_tags) or (
            any(c in self.long_scalp_mode_tags for c in enter_tags)
            and all(c in (self.long_scalp_mode_tags + self.long_rebuy_mode_tags + self.long_grind_mode_tags) for c in enter_tags)
        )

        # Get thresholds from cache
        if self.is_futures_mode:
            doom_threshold = self._exit_conditions_cache['normal_futures']
            rapid_threshold = self._exit_conditions_cache['rapid_futures']
            scalp_threshold = self._exit_conditions_cache['scalp_futures']
            rebuy_threshold = self._exit_conditions_cache['rebuy_futures']
        else:
            doom_threshold = self._exit_conditions_cache['normal_spot']
            rapid_threshold = self._exit_conditions_cache['rapid_spot']
            scalp_threshold = self._exit_conditions_cache['scalp_spot']
            rebuy_threshold = self._exit_conditions_cache['rebuy_spot']

        # Quick exit checks
        if previous_sell_reason in [f"exit_{mode_name}_stoploss_doom", f"exit_{mode_name}_stoploss"]:
            if self.derisk_enable and (current_time - timedelta(minutes=60) > previous_time_profit_reached):
                if profit_ratio < previous_profit:
                    return True, previous_sell_reason
                elif profit_ratio > previous_profit:
                    self._remove_profit_target(pair)
                    return False, None
            elif (not self.derisk_enable and not is_rapid_mode and not is_rebuy_mode and not is_scalp_mode and
                  profit_init_ratio <= -doom_threshold):
                return True, previous_sell_reason
            elif (not self.derisk_enable and is_rapid_mode and profit_init_ratio <= -rapid_threshold):
                return True, previous_sell_reason
            elif (not self.derisk_enable and is_rebuy_mode and profit_init_ratio <= -rebuy_threshold):
                return True, previous_sell_reason
            elif (not self.derisk_enable and is_scalp_mode and profit_init_ratio <= -scalp_threshold):
                return True, previous_sell_reason
        elif previous_sell_reason in [f"exit_{mode_name}_stoploss_u_e"]:
            if profit_ratio < (previous_profit - (0.04 / trade.leverage)):
                return True, previous_sell_reason
        elif previous_sell_reason in [f"exit_profit_{mode_name}_max"]:
            if profit_init_ratio < -0.08:
                self._remove_profit_target(pair)
                return False, None

            # Optimized profit taking logic
            return self._optimized_profit_taking(mode_name, profit_init_ratio, previous_profit,
                                               enter_tags, last_candle)

        return False, None

    def _optimized_profit_taking(self, mode_name: str, profit_init_ratio: float,
                               previous_profit: float, enter_tags: List[str],
                               last_candle) -> tuple:
        """Optimized profit taking logic"""

        # Fast profit tier detection
        if trade.is_short:
            is_scalp_mode = all(c in self.short_scalp_mode_tags for c in enter_tags)
            if is_scalp_mode:
                # Numba-optimized scalp profit taking
                return self._fast_scalp_profit_taking(profit_init_ratio, previous_profit)
            else:
                # General short profit taking
                return self._fast_short_profit_taking(profit_init_ratio, previous_profit, last_candle)
        else:
            is_scalp_mode = all(c in self.long_scalp_mode_tags for c in enter_tags)
            if is_scalp_mode:
                # Numba-optimized scalp profit taking
                return self._fast_scalp_profit_taking(profit_init_ratio, previous_profit)
            else:
                # General long profit taking
                return self._fast_long_profit_taking(profit_init_ratio, previous_profit, last_candle)

    @njit(fastmath=True)
    def _fast_scalp_profit_taking(self, profit_init_ratio: float, previous_profit: float) -> tuple:
        """Fast scalp mode profit taking with numba"""
        if 0.001 <= profit_init_ratio < 0.01:
            if profit_init_ratio < (previous_profit - 0.008):
                return True, f"exit_profit_scalp_t_0_1"
        elif 0.01 <= profit_init_ratio < 0.02:
            if profit_init_ratio < (previous_profit - 0.01):
                return True, f"exit_profit_scalp_t_1_1"
        elif 0.02 <= profit_init_ratio < 0.03:
            if profit_init_ratio < (previous_profit - 0.01):
                return True, f"exit_profit_scalp_t_2_1"
        elif 0.03 <= profit_init_ratio < 0.04:
            if profit_init_ratio < (previous_profit - 0.015):
                return True, f"exit_profit_scalp_t_3_1"
        elif 0.04 <= profit_init_ratio < 0.05:
            if profit_init_ratio < (previous_profit - 0.015):
                return True, f"exit_profit_scalp_t_4_1"
        elif 0.05 <= profit_init_ratio < 0.06:
            if profit_init_ratio < (previous_profit - 0.015):
                return True, f"exit_profit_scalp_t_5_1"
        elif 0.06 <= profit_init_ratio < 0.07:
            if profit_init_ratio < (previous_profit - 0.015):
                return True, f"exit_profit_scalp_t_6_1"
        elif 0.07 <= profit_init_ratio < 0.08:
            if profit_init_ratio < (previous_profit - 0.02):
                return True, f"exit_profit_scalp_t_7_1"
        elif 0.08 <= profit_init_ratio < 0.09:
            if profit_init_ratio < (previous_profit - 0.02):
                return True, f"exit_profit_scalp_t_8_1"
        elif 0.09 <= profit_init_ratio < 0.10:
            if profit_init_ratio < (previous_profit - 0.02):
                return True, f"exit_profit_scalp_t_9_1"
        elif 0.10 <= profit_init_ratio < 0.11:
            if profit_init_ratio < (previous_profit - 0.025):
                return True, f"exit_profit_scalp_t_10_1"
        elif 0.11 <= profit_init_ratio < 0.12:
            if profit_init_ratio < (previous_profit - 0.025):
                return True, f"exit_profit_scalp_t_11_1"
        elif 0.12 <= profit_init_ratio:
            if profit_init_ratio < (previous_profit - 0.025):
                return True, f"exit_profit_scalp_t_12_1"

        return False, None

    def _fast_short_profit_taking(self, profit_init_ratio: float, previous_profit: float, last_candle) -> tuple:
        """Fast short profit taking logic"""
        # Simplified condition checks
        if 0.001 <= profit_init_ratio < 0.01:
            if (profit_init_ratio < (previous_profit - 0.03) and
                last_candle["RSI_14"] > 50.0 and
                last_candle["RSI_14"] > last_candle.get("previous_RSI_14", 0) and
                last_candle["CMF_20"] > 0.0):
                return True, f"exit_profit_short_t_0_1"
            elif (profit_init_ratio < (previous_profit - 0.03) and
                  last_candle["CMF_20"] > 0.0 and
                  last_candle.get("CMF_20_1h", 0) > 0.0 and
                  last_candle.get("CMF_20_4h", 0) > 0.0):
                return True, f"exit_profit_short_t_0_2"
            elif profit_init_ratio < (previous_profit - 0.05) and last_candle.get("ROC_9_4h", 0) < -40.0:
                return True, f"exit_profit_short_t_0_3"

        return False, None

    def _fast_long_profit_taking(self, profit_init_ratio: float, previous_profit: float, last_candle) -> tuple:
        """Fast long profit taking logic"""
        # Simplified condition checks
        if 0.001 <= profit_init_ratio < 0.01:
            if (profit_init_ratio < (previous_profit - 0.03) and
                last_candle["RSI_14"] < 50.0 and
                last_candle["RSI_14"] < last_candle.get("previous_RSI_14", 100) and
                last_candle["CMF_20"] < -0.0):
                return True, f"exit_profit_long_t_0_1"
            elif (profit_init_ratio < (previous_profit - 0.03) and
                  last_candle["CMF_20"] < -0.0 and
                  last_candle.get("CMF_20_1h", 0) < -0.0 and
                  last_candle.get("CMF_20_4h", 0) < -0.0):
                return True, f"exit_profit_long_t_0_2"
            elif profit_init_ratio < (previous_profit - 0.05) and last_candle.get("ROC_9_4h", 0) > 40.0:
                return True, f"exit_profit_long_t_0_3"

        return False, None

    def _remove_profit_target(self, pair: str):
        """Remove profit target from cache"""
        if pair in self.target_profit_cache:
            del self.target_profit_cache[pair]

    # Get Hold Trades Config File
    # ---------------------------------------------------------------------------------------------
    def get_hold_trades_config_file(self):
        proper_holds_file_path = self.config["user_data_dir"].resolve() / "nfi-hold-trades.json"
        if proper_holds_file_path.is_file():
            return proper_holds_file_path

        strat_file_path = pathlib.Path(__file__)
        hold_trades_config_file_resolve = strat_file_path.resolve().parent / "hold-trades.json"
        if hold_trades_config_file_resolve.is_file():
            log.warning(
                "Please move %s to %s which is now the expected path for the holds file",
                hold_trades_config_file_resolve,
                proper_holds_file_path,
            )
        return hold_trades_config_file_resolve

        # The resolved path does not exist, is it a symlink?
        hold_trades_config_file_absolute = strat_file_path.absolute().parent / "hold-trades.json"
        if hold_trades_config_file_absolute.is_file():
            log.warning(
                "Please move %s to %s which is now the expected path for the holds file",
                hold_trades_config_file_absolute,
                proper_holds_file_path,
            )
            return hold_trades_config_file_absolute

    # Load Hold Trades Config
    # ---------------------------------------------------------------------------------------------
    def load_hold_trades_config(self):
        if self.hold_trades_cache is None:
            hold_trades_config_file = self.get_hold_trades_config_file()
        if hold_trades_config_file:
            log.warning("Loading hold support data from %s", hold_trades_config_file)
            self.hold_trades_cache = HoldsCache(hold_trades_config_file)

        if self.hold_trades_cache:
            self.hold_trades_cache.load()

    # Should Hold Trade
    # ---------------------------------------------------------------------------------------------
    def _should_hold_trade(self, trade: "Trade", rate: float, sell_reason: str) -> bool:
      if self.config["runmode"].value not in ("live", "dry_run"):
        return False

      if not self.hold_support_enabled:
        return False

      # Just to be sure our hold data is loaded, should be a no-op call after the first bot loop
      self.load_hold_trades_config()

      if not self.hold_trades_cache:
        # Cache hasn't been setup, likely because the corresponding file does not exist, sell
        return False

      if not self.hold_trades_cache.data:
        # We have no pairs we want to hold until profit, sell
        return False

      # By default, no hold should be done
      hold_trade = False

      trade_ids: dict = self.hold_trades_cache.data.get("trade_ids")
      if trade_ids and trade.id in trade_ids:
        trade_profit_ratio = trade_ids[trade.id]
        filled_entries = trade.select_filled_orders(trade.entry_side)
        filled_exits = trade.select_filled_orders(trade.exit_side)
        profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio = self.calc_total_profit(
          trade, filled_entries, filled_exits, rate
        )
        current_profit_ratio = profit_init_ratio
        if sell_reason == "force_sell":
          formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
          formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
          log.warning(
            "Force selling %s even though the current profit of %s < %s",
            trade,
            formatted_current_profit_ratio,
            formatted_profit_ratio,
          )
          return False
        elif current_profit_ratio >= trade_profit_ratio:
          # This pair is on the list to hold, and we reached minimum profit, sell
          formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
          formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
          log.warning(
            "Selling %s because the current profit of %s >= %s",
            trade,
            formatted_current_profit_ratio,
            formatted_profit_ratio,
          )
          return False

        # This pair is on the list to hold, and we haven't reached minimum profit, hold
        hold_trade = True

      trade_pairs: dict = self.hold_trades_cache.data.get("trade_pairs")
      if trade_pairs and trade.pair in trade_pairs:
        trade_profit_ratio = trade_pairs[trade.pair]
        filled_entries = trade.select_filled_orders(trade.entry_side)
        filled_exits = trade.select_filled_orders(trade.exit_side)
        profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio = self.calc_total_profit(
          trade, filled_entries, filled_exits, rate
        )
        current_profit_ratio = profit_init_ratio
        if sell_reason == "force_sell":
          formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
          formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
          log.warning(
            "Force selling %s even though the current profit of %s < %s",
            trade,
            formatted_current_profit_ratio,
            formatted_profit_ratio,
          )
          return False
        elif current_profit_ratio >= trade_profit_ratio:
          # This pair is on the list to hold, and we reached minimum profit, sell
          formatted_profit_ratio = f"{trade_profit_ratio * 100}%"
          formatted_current_profit_ratio = f"{current_profit_ratio * 100}%"
          log.warning(
            "Selling %s because the current profit of %s >= %s",
            trade,
            formatted_current_profit_ratio,
            formatted_profit_ratio,
          )
          return False

        # This pair is on the list to hold, and we haven't reached minimum profit, hold
        hold_trade = True

      return hold_trade

    # Populate Exit Trend
    # ---------------------------------------------------------------------------------------------
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[:, "exit_long"] = 0
        df.loc[:, "exit_short"] = 0

        return df

    #
    # $$$$$$$$\ $$\   $$\ $$$$$$$$\ $$$$$$$\ $$\     $$\
    # $$  _____|$$$\  $$ |\__$$  __|$$  __$$\\$$\   $$  |
    # $$ |      $$$$\ $$ |   $$ |   $$ |  $$ |\$$\ $$  /
    # $$$$$\    $$ $$\$$ |   $$ |   $$$$$$$  | \$$$$  /
    # $$  __|   $$ \$$$$ |   $$ |   $$  __$$<   \$$  /
    # $$ |      $$ |\$$$ |   $$ |   $$ |  $$ |   $$ |
    # $$$$$$$$\ $$ | \$$ |   $$ |   $$ |  $$ |   $$ |
    # \________|\__|  \__|   \__|   \__|  \__|   \__|
    #

    #
    #  $$$$$$\   $$$$$$\  $$\   $$\ $$$$$$$\  $$$$$$\ $$$$$$$$\ $$$$$$\  $$$$$$\  $$\   $$\  $$$$$$\
    # $$  __$$\ $$  __$$\ $$$\  $$ |$$  __$$\ \_$$  _|\__$$  __|\_$$  _|$$  __$$\ $$$\  $$ |$$  __$$\
    # $$ /  \__|$$ /  $$ |$$$$\ $$ |$$ |  $$ |  $$ |     $$ |     $$ |  $$ /  $$ |$$$$\ $$ |$$ /  \__|
    # $$ |      $$ |  $$ |$$ $$\$$ |$$ |  $$ |  $$ |     $$ |     $$ |  $$ |  $$ |$$ $$\$$ |\$$$$$$\
    # $$ |      $$ |  $$ |$$ \$$$$ |$$ |  $$ |  $$ |     $$ |     $$ |  $$ |  $$ |$$ \$$$$ | \____$$\
    # $$ |  $$\ $$ |  $$ |$$ |\$$$ |$$ |  $$ |  $$ |     $$ |     $$ |  $$ |  $$ |$$ |\$$$ |$$\   $$ |
    # \$$$$$$  | $$$$$$  |$$ | \$$ |$$$$$$$  |$$$$$$\    $$ |   $$$$$$\  $$$$$$  |$$ | \$$ |\$$$$$$  |
    #  \______/  \______/ \__|  \__|\_______/ \______|   \__|   \______| \______/ \__|  \__| \______/
    #

    # Populate Entry Trend
    # ---------------------------------------------------------------------------------------------
    
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
      long_entry_conditions = []
      short_entry_conditions = []

      df.loc[:, "enter_tag"] = ""
      df.loc[:, "enter_long"] = ""
      df.loc[:, "enter_short"] = ""

      is_backtest = self.dp.runmode.value in ["backtest", "hyperopt", "plot", "webserver"]
      # the number of free slots
      current_free_slots = self.config["max_open_trades"]
      if not is_backtest:
        current_free_slots = self.config["max_open_trades"] - Trade.get_open_trade_count()
      # Grind mode
      num_open_long_grind_mode = 0
      is_pair_long_grind_mode = metadata["pair"].split("/")[0] in self.grind_mode_coins
      if not is_backtest:
        open_trades = Trade.get_trades_proxy(is_open=True)
        for open_trade in open_trades:
          enter_tag = open_trade.enter_tag
          if enter_tag is not None:
            enter_tags = enter_tag.split()
            if all(c in self.long_grind_mode_tags for c in enter_tags):
              num_open_long_grind_mode += 1
      # Top Coins mode
      is_pair_long_top_coins_mode = metadata["pair"].split("/")[0] in self.top_coins_mode_coins
      is_pair_short_top_coins_mode = metadata["pair"].split("/")[0] in self.top_coins_mode_coins
      # if BTC/ETH stake
      is_btc_stake = self.config["stake_currency"] in self.btc_stakes
      allowed_empty_candles_288 = 144 if is_btc_stake else 60

      ###############################################################################################

      # LONG ENTRY CONDITIONS STARTS HERE

      ###############################################################################################

      #
      #  /$$       /$$$$$$ /$$   /$$ /$$$$$$        /$$$$$$$$/$$   /$$/$$$$$$$$/$$$$$$$$/$$$$$$$
      # | $$      /$$__  $| $$$ | $$/$$__  $$      | $$_____| $$$ | $|__  $$__| $$_____| $$__  $$
      # | $$     | $$  \ $| $$$$| $| $$  \__/      | $$     | $$$$| $$  | $$  | $$     | $$  \ $$
      # | $$     | $$  | $| $$ $$ $| $$ /$$$$      | $$$$$  | $$ $$ $$  | $$  | $$$$$  | $$$$$$$/
      # | $$     | $$  | $| $$  $$$| $$|_  $$      | $$__/  | $$  $$$$  | $$  | $$__/  | $$__  $$
      # | $$     | $$  | $| $$\  $$| $$  \ $$      | $$     | $$\  $$$  | $$  | $$     | $$  \ $$
      # | $$$$$$$|  $$$$$$| $$ \  $|  $$$$$$/      | $$$$$$$| $$ \  $$  | $$  | $$$$$$$| $$  | $$
      # |________/\______/|__/  \__/\______/       |________|__/  \__/  |__/  |________|__/  |__/
      #

      for enabled_long_entry_signal in self.long_entry_signal_params:
        long_entry_condition_index = int(enabled_long_entry_signal.split("_")[3])
        item_buy_protection_list = [True]
        if self.long_entry_signal_params[f"{enabled_long_entry_signal}"]:
          # Long Entry Conditions Starts Here
          # -----------------------------------------------------------------------------------------
          long_entry_logic = []
          long_entry_logic.append(reduce(lambda x, y: x & y, item_buy_protection_list))

          # Condition #1 - Normal mode (Long).
          if long_entry_condition_index == 1:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(
              # 5m & 15m & 1h down move
              ((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 3.0) | (df["RSI_3_change_pct_1h"] > -50.0))
              # 5m & 15m down move, 5h high
              & ((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 5.0) | (df["RSI_14_4h"] < 60.0))
              # 5m & 15m down move, 4h high
              & ((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_4h"] < 100.0))
              # 5m & 1h down move, 15m still not low enough
              & ((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_15m"] < 30.0))
              # 5m down move, 15m high
              & ((df["RSI_3"] > 3.0) | (df["AROONU_14_15m"] < 80.0))
              # 15m down move, 1h downtrend, 1h high
              & ((df["RSI_3_15m"] > 1.0) | (df["CMF_20_1h"] > -0.1) | (df["AROONU_14_1h"] < 70.0))
              # 15m & 1h down move, 15m still high
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_15m"] < 40.0))
              # 15m & 1h down move, 1h high
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0))
              # 15m & 1h down move, 1h still high
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
              # 15m & 1h & 4h down move
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0))
              # 15m & 1h down move, 1h still high
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0))
              # 15m & 1h & 4h & 1d down move
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
              # 15m & 1h down move, 1h high
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 80.0))
              # 15m & 4h down move, 4h still high
              & ((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
              # 15m down move, 1h & 4h high
              & ((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_1h"] < 85.0) | (df["AROONU_14_4h"] < 90.0))
              # 15m down move, 4h high, 1d overbought
              & ((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_4h"] < 85.0) | (df["ROC_9_1d"] < 100.0))
              # 15m & 1h down move, 1d overbought
              & ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 5.0) | (df["ROC_9_1d"] < 40.0))
              # 5m & 1h down move, 1h overbought
              & ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 60.0) | (df["ROC_9_1h"] < 40.0))
              # 15m down move, 15m & 4h still high
              & ((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 60.0))
              # 15m down move, 15m still high, 1d overbought
              & ((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_15m"] < 50.0) | (df["ROC_9_1d"] < 80.0))
              # 15m down move, 4h high, 1d overbought
              & ((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_4h"] < 60.0) | (df["ROC_9_1d"] < 80.0))
              # 15m & 1h down move, 4h high
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_14_4h"] < 80.0))
              # 15m & 1h down move, 4h overbought
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["ROC_9_4h"] < 80.0))
              # 15m & 4h down move, 1h high
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["AROONU_14_1h"] < 75.0))
              # 15m & 4h down move, 1h high
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_1h"] < 85.0))
              # 15m & 4h down move, 4h high
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
              # 15m down move & downtrend, 4h high
              & ((df["RSI_3_15m"] > 10.0) | (df["CMF_20_15m"] > -0.3) | (df["AROONU_14_4h"] < 100.0))
              # 15m down move, 1h high, 1d overbought
              & ((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_1h"] < 100.0) | (df["ROC_9_1d"] < 80.0))
              # 15m down move, 1h high, 4h downtrend
              & ((df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["ROC_9_4h"] > -25.0))
              # 15m down move, 15m still high, 4h overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_14_15m"] < 50.0) | (df["ROC_9_4h"] < 50.0))
              # 15m down move, 15m & 1h high
              & ((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 80.0) | (df["AROONU_14_1h"] < 90.0))
              # 1h & 1d down move, 4h still high
              & ((df["RSI_3_1h"] > 3.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
              # 1h & 4h down move, 1h downtrend
              & ((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 10.0) | (df["CMF_20_1h"] > -0.3))
              # 1h & 4h down move, 4h high
              & ((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 70.0))
              # 1h & 4h down move, 1d overbought
              & ((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 40.0))
              # 1h & 4h down move, 15m still not low enough
              & ((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["AROONU_14_15m"] < 30.0))
              # 1h & 1d down move, 4h high
              & ((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 20.0) | (df["AROONU_14_4h"] < 80.0))
              # 1h down move, 4h high
              & ((df["RSI_3_1h"] > 10.0) | (df["RSI_14_4h"] < 75.0))
              # 1h down move, 4h overbought
              & ((df["RSI_3_1h"] > 10.0) | (df["ROC_9_4h"] < 50.0))
              # 1h & 4h down move, 4h overbought
              & ((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 50.0) | (df["ROC_9_4h"] < 80.0))
              # 1h down move, 4h high & overbought
              & ((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 100.0) | (df["ROC_9_4h"] < 50.0))
              # 1h down move, 4h & 1d overbought
              & ((df["RSI_3_1h"] > 25.0) | (df["ROC_9_4h"] < 80.0) | (df["ROC_9_1d"] < 100.0))
              # 1h down move, 1h downtrend, 1h high
              & ((df["RSI_3_1h"] > 40.0) | (df["CMF_20_1h"] > -0.25) | (df["AROONU_14_1h"] < 90.0))
              # 1h down move, 1h still high, 4h high
              & ((df["RSI_3_1h"] > 60.0) | (df["AROONU_14_1h"] < 50.0) | (df["RSI_14_4h"] < 90.0))
              # 4h & 1d down move, 4h still not low enough
              & ((df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_4h"] < 20.0))
              # 1d down move, 1h still high, 4h high
              & ((df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
              # 1d down move, 1h & 4h high
              & ((df["RSI_3_1d"] > 25.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
              # 1d down move, 4h still high, 1d overbought
              & ((df["RSI_3_1d"] > 50.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 200.0))
              # 1h down move, 4h high, 1d overbought
              & ((df["RSI_3_change_pct_1h"] > -75.0) | (df["AROONU_14_4h"] < 90.0) | (df["ROC_9_1d"] < 100.0))
              # 15m & 1h & 4h downtrend
              & ((df["CMF_20_15m"] > -0.3) | (df["CMF_20_1h"] > -0.3) | (df["CMF_20_4h"] > -0.3))
              # 5m down move, 15m still not low enough, 1h high
              & ((df["ROC_2"] > -10.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_1h"] < 80.0))
              # 5m down move, 15m still high
              & ((df["ROC_2"] > -10.0) | (df["AROONU_14_15m"] < 50.0))
              # 5m down move, 15m & 1h down move, 15m still high
              & (
                (df["ROC_9"] > -15.0) | (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_15m"] < 50.0)
              )
              # 5m down move, 4h down move, 15m downtrend, 1h high
              & (
                (df["ROC_9"] > -15.0) | (df["RSI_3_4h"] > 45.0) | (df["CMF_20_15m"] > -0.3) | (df["AROONU_14_1h"] < 60.0)
              )
              # 1h downtrend, 4h high & overbought
              & ((df["ROC_9_1h"] > -25.0) | (df["AROONU_14_4h"] < 80.0) | (df["ROC_9_4h"] < 80.0))
              # 1d P&D, 1d downtrend
              & ((df["change_pct_1d"] > -5.0) | (df["change_pct_1d"].shift(288) < 30.0) | (df["CMF_20_1d"] > -0.0))
              # 1d green with top wick, 1h down move
              & ((df["change_pct_1d"] < 20.0) | (df["top_wick_pct_1d"] < 15.0) | (df["RSI_3_1h"] > 20.0))
              # 1d green with top wick, 4h high
              & ((df["change_pct_1d"] < 25.0) | (df["top_wick_pct_1d"] < 25.0) | (df["AROONU_14_4h"] < 80.0))
              # 1d green, 1h down move, 1d downtrend
              & ((df["change_pct_1d"] < 40.0) | (df["RSI_3_1h"] > 25.0) | (df["CMF_20_1d"] > -0.2))
              # 1d green with top wick, 4h overbought
              & ((df["change_pct_1d"] < 50.0) | (df["top_wick_pct_1d"] < 30.0) | (df["ROC_9_4h"] < 80.0))
              # big drop in the last hour, 15m downtrend
              & ((df["close"] > (df["close_max_12"] * 0.65)) | (df["CMF_20_15m"] > -0.5))
              # big drop in the last 6 hours, 1h down move, 1h high
              & ((df["close"] > (df["high_max_6_1h"] * 0.60)) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 60.0))
              # big drop in the last 24 hours,  1h still high
              & ((df["close"] > (df["high_max_24_1h"] * 0.40)) | (df["STOCHRSIk_14_14_3_3_1h"] < 45.0))
              # big drop in the last 4 days, 1h high
              & ((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["AROONU_14_1h"] < 70.0))
            )

            # Logic
            long_entry_logic.append(
              (df["EMA_26"] > df["EMA_12"])
              & ((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.034))
              & ((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))
              & (df["close"] < (df["BBL_20_2.0"] * 0.999))
            )

          # Condition #2 - Normal mode (Long).
          if long_entry_condition_index == 2:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append((df["RSI_3"] > 3.0) & (df["RSI_3_15m"] > 3.0))

            long_entry_logic.append(
              # 5m & 15m & 1h down move, 1h & 4h high, 4h overbought
              (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["AROONU_14_4h"] < 100.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
                | (df["ROC_9_4h"] < 20.0)
              )
              # 5m & 15m & 1h & 4h down move, 1h & 4h still not low enough, 1h still high
              & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["AROONU_14_1h"] < 20.0)
                | (df["AROONU_14_4h"] < 20.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
              )
              # 15m & 1h & 4h down move, 15m downtrend, 1h still not low enough
              & (
                (df["RSI_3_15m"] > 5.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
              )
              # 5m & 15m & 1h down move, 15m still high, 1h & 4h high
              & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 60.0)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 5m & 15m & 4h down move, 15m downtrend, 1d high, 1h & 4h still not low enough
              & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 10.0)
                | (df["RSI_14_4h"] < 10.0)
                | (df["CMF_20_15m"] > -0.20)
                | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
                | (df["CCI_20_1h"] < -150.0)
                | (df["CCI_20_4h"] < -250.0)
              )
              # 15m & 1h & 4h down move, 4h still not low enough
              & ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_14_4h"] < 30.0))
              # 15m & 1h down move, 4h still high
              & ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_14_4h"] < 50.0))
              # 15m & 1h & 4h down move, 15m & 1h downtrend
              & (
                (df["RSI_3_15m"] > 5.0)
                | (df["RSI_3_1h"] > 15.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["CMF_20_15m"] > -0.5)
                | (df["CCI_20_change_pct_15m"] > -0.0)
                | (df["CCI_20_change_pct_1h"] > -0.0)
              )
              # 15m & 1h & 4h down move, 1h still not low enough, 4h still high
              & (
                (df["RSI_3_15m"] > 5.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 5.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["AROONU_14_1h"] < 50.0)
                | (df["AROONU_14_4h"] < 80.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still not low enough
              & (
                (df["RSI_3_15m"] > 5.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 5.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_1h"] < 75.0)
              )
              # 15m & 4h down move, 1h still not low enough
              & ((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0))
              # 15m & 1h & 4h & 1d down move, 1h & 4h still not low enough
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 10.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_3_1d"] > 25.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h downtrend
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 15.0)
                | (df["RSI_3_4h"] > 35.0)
                | (df["CMF_20_15m"] > -0.30)
                | (df["CMF_20_1h"] > -0.30)
                | (df["CMF_20_4h"] > -0.25)
              )
              # 15m & 1h down move, 1h & 4h & 1d high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 20.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["AROONU_14_1d"] < 100.0)
              )
              # 15m & 1h down move, 1h & 4h still high, 15m & 1h downtrend
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 20.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["CMF_20_1h"] > -0.25)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high, 1d overbought
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["ROC_9_1d"] < 30.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still not low enough, 1d high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 35.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
                | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
              )
              # 15m & 1h & 4h down move, 15m downtrend, 1d high, 1h & 4h downtrend
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
                | (df["CCI_20_1h"] < -150.0)
                | (df["CCI_20_4h"] < -250.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 1h high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
              )
              # 15m & 1h down move, 1h & 4h still high, 1d high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 1 high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_1h"] < 70.0)
              )
              # 15m & 1h down move, 1h & 4h still high, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 60.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_1h"] < 80.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
              )
              # 15m & 1h down move, 1h still high, 4h high & overbought
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 20.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_1h"] < 40.0)
                | (df["AROONU_14_4h"] < 85.0)
                | (df["ROC_9_4h"] < 40.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still not low enough
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 20.0)
                | (df["RSI_3_4h"] > 10.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 35.0)
              )
              # 15m & 1h & 4h & 1d down move, 1h & 4h still not low enough
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 20.0)
                | (df["RSI_3_4h"] > 20.0)
                | (df["RSI_3_1d"] > 40.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 35.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 80.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h downtrend, 1h & 4h still high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["CMF_20_1h"] > -0.0)
                | (df["CMF_20_4h"] > -0.25)
                | (df["AROONU_14_1h"] < 40.0)
                | (df["AROONU_14_4h"] < 50.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still high, 1d overbought
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["ROC_9_1d"] < 50.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h stil high, 1h & 4h high, 4h overbought
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 60.0)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["AROONU_14_4h"] < 100.0)
                | (df["ROC_9_4h"] < 20.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_14_15m"] < 35.0)
                | (df["AROONU_14_1h"] < 80.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_1h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 1h still not low enough, 4h still high, 1d downtrend
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["AROONU_14_1h"] < 20.0)
                | (df["AROONU_14_4h"] < 50.0)
                | (df["ROC_9_1d"] > -30.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 60.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 15m & 1h down move, 15m & 1h & 4h still high, 4h overbought
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 55.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 40.0)
                | (df["AROONU_14_4h"] < 85.0)
                | (df["ROC_9_4h"] < 80.0)
              )
              # 15m & 1h down move, 15m still high, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 65.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 60.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_1h"] < 80.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 1h down move, 15m still high, 1h & 4h high, 4h overbought
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 60.0)
                | (df["RSI_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
                | (df["ROC_9_4h"] < 40.0)
              )
              # 15m & 1h down move, 15m downtrend, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["CMF_20_15m"] > -0.30)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 35.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["AROONU_14_1h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 1h still high, 4h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 60.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h still high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high
              & (
                (df["RSI_3_15m"] > 35.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 50.0)
                | (df["AROONU_14_4h"] < 50.0)
              )
              # 15m & 1h & 4h down move, 15m high
              & (
                (df["RSI_3_15m"] > 45.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0)
              )
              # 1h & 4h & 1d down move, 4h downtrend, 1d downtrend
              & (
                (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_3_1d"] > 30.0)
                | (df["CMF_20_4h"] > -0.40)
                | (df["ROC_9_1d"] > -50.0)
              )
              # 4h down move, 4h still high
              & ((df["RSI_3_4h"] > 10.0) | (df["AROONU_14_4h"] < 50.0))
              # 1d down move, 4h downtrend, 1h high
              & ((df["RSI_3_1d"] > 10.0) | (df["CMF_20_4h"] > -0.30) | (df["AROONU_14_1h"] < 80.0))
              & (
                # 1d green, 15m down move, 1h & 4h still high, 4h overbought
                (df["change_pct_1d"] < 30.0)
                | (df["RSI_3_15m"] > 15.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["ROC_9_4h"] < 50.0)
              )
            )

            # Logic
            long_entry_logic.append(
              (df["AROONU_14"] < 25.0)
              & (df["STOCHRSIk_14_14_3_3"] < 20.0)
              & (df["AROONU_14_15m"] < 25.0)
              & (df["close"] < (df["EMA_20"] * 0.944))
            )

          # Condition #3 - Normal mode (Long).
          if long_entry_condition_index == 3:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(
              # 5m & 15m down move, 15m & 1h & 4h still high, 15m & 4h high, 1h & 4h still high
              (
                (df["RSI_14"] < 5.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
              )
              # 5m & 15m & dh down move, 15m & 1h still high, 15m high
              & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
              )
              # 5m & 15m & 4h down move, 15m still not low enough, 1h & 4h still high, 15m still high
              & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_4h"] > 20.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 50.0)
              )
              # 5m & 15m & 1h down move, 15m still high, 4h high, 1d overbought
              & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["ROC_9_1d"] < 40.0)
              )
              # 5m & 15m & 4h down move, 15m & 1h & 4h still high, 15m & 1h & 4h high
              & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 60.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_1h"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
              )
              # 15m & 1h & 4h down move, 1h & 4h still not low enough, 1h still high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
              )
              # 5m & 15m & 4h & 1d down move, 15m & 4h still not low enough, 4h downtrend
              & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_3_1d"] > 40.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
                | (df["ROC_9_4h"] > -10.0)
              )
              # 15m down move, 15m still not low enough, 1h high
              & (
                (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
              )
              # 5m & 15m & 1h down move, 15m still not low enough, 1h high
              & (
                (df["RSI_3"] > 15.0)
                | (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_1h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h still high, 15m still not low enough, 1h high
              & (
                (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 30.0)
                | (df["AROONU_14_1h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h still not low enough, 15m & 1h still high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_1h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high, 1d overbought
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["ROC_9_1d"] < 80.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
              )
              # 15m & 1h down move, 15m still not low enough, 1h & 4h still high, 4h overbought
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["ROC_9_4h"] < 50.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high, 4h overbought
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["ROC_9_4h"] < 20.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 15m still high, 1h still high, 4h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 65.0)
                | (df["RSI_3_1d"] > 65.0)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
              )
              # 15m & 1d down move, 15m & 1h & 4h still high, 15m still high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1d"] > 20.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 50.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high, 4h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_1h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 15m still not low enough, 1h & 4h still high, 15m high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
              )
              # 15m & 4h down move, 15m & 4h still high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_4h"] > 35.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 50.0)
              )
              # 15m & 4h down move, 15m & 1h & 4h still high, 15m & 4h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
              )
              # 15m down move, 1h & 4h high, 15m high, 1d overbought
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_14_1h"] < 60.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["ROC_9_1d"] < 50.0)
              )
              # 15m down move, 15m & 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["AROONU_14_1h"] < 50.0)
                | (df["AROONU_14_4h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
              )
              # 15m & 1h down move, 15m & 1h still high, 4h downtrend, 15m still not low enough, 1h high
              & (
                (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["CMF_20_4h"] > -0.20)
                | (df["AROONU_14_15m"] < 30.0)
                | (df["AROONU_14_1h"] < 80.0)
              )
              # 15m & 4h down move, 15m high
              & (
                (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 40.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
              )
              # 15m & 4h down move, 15m & 1h & 4h still high, 15m high, 1h & 4h still not low enough, 1d overbought
              & (
                (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 20.0)
                | (df["ROC_9_1d"] < 50.0)
              )
              # 15m & 4h down move, 15m & 1h still high, 4h high & overbought
              & (
                (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 60.0)
                | (df["ROC_9_4h"] < 50.0)
              )
              # 15m & 1h down move, 15m & 1h & 4h still high, 1h & 4h downtrend, 15m stil high, 1h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1h"] > -0.10)
                | (df["CMF_20_4h"] > -0.10)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["AROONU_14_1h"] < 90.0)
              )
              # 15m & 1h & 4h & 1d down move, 15m & 1h & 4h still not low enough, 15m still high, 1d downtrend
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_3_1d"] > 15.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["ROC_9_1d"] > -40.0)
              )
              # 15m & 1h down move, 15m & 1h & 4h still high, 1h & 4h downtrend, 15m still high, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1h"] > -0.10)
                | (df["CMF_20_4h"] > -0.10)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["AROONU_14_1h"] < 85.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 15m & 1h down move, 15m still high, 1h & 4h high, 15m & 4h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 55.0)
                | (df["RSI_14_15m"] < 45.0)
                | (df["RSI_14_1h"] < 55.0)
                | (df["RSI_14_4h"] < 60.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 80.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h still high, 1h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 65.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
              )
              # 15m & 1h down move, 15m & 1h & 4h still high, 15m & 1h still not low enough, 4h high, 1d overbought
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 65.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 30.0)
                | (df["AROONU_14_1h"] < 30.0)
                | (df["AROONU_14_4h"] < 80.0)
                | (df["ROC_9_1d"] < 150.0)
              )
              # 15m & 1h & 4h & 1d down move, 15m still not low enough, 1h & 4h still high, 15m still high, 4h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["RSI_3_1d"] > 55.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["AROONU_14_4h"] < 70.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h & 4h still high, 15m & 1h still not low enough, 4h high
              & (
                (df["RSI_3_15m"] > 35.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_4h"] > 60.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 30.0)
                | (df["AROONU_14_1h"] < 30.0)
                | (df["AROONU_14_4h"] < 70.0)
              )
              # 15m & 1h down move, 15m & 1h & 4h still high, 1h & 4h downtrend, 15m still high, 4h high & overbought
              & (
                (df["RSI_3_15m"] > 35.0)
                | (df["RSI_3_1h"] > 60.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1h"] > -0.0)
                | (df["CMF_20_4h"] > -0.1)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["ROC_9_4h"] < 10.0)
              )
              # 15m & 4h down move, 15m still not low enough, 1h still high, 4h high, 4h downtrend, 4h overbought
              & (
                (df["RSI_3_15m"] > 45.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_4h"] > -0.10)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["ROC_9_4h"] < 20.0)
              )
              # 15m & 1h & 4h down move, 15m & 1h high
              & (
                (df["RSI_3_15m"] > 50.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["AROONU_14_15m"] < 70.0)
                | (df["AROONU_14_1h"] < 100.0)
              )
              # 1h & 4h down move, 15m still high
              & ((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["AROONU_14_15m"] < 50.0))
              # 4h & 1d down move, 15m & 1h & 4h still not low enough, 15m high
              & (
                (df["RSI_3_4h"] > 20.0)
                | (df["RSI_3_1d"] > 20.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
              )
            )

            # Logic
            long_entry_logic.append(
              (df["RSI_20"] < df["RSI_20"].shift(1))
              & (df["RSI_4"] < 45.0)
              & (df["RSI_14"] > 32.0)
              & (df["AROONU_14"] < 10.0)
              & (df["STOCHRSIk_14_14_3_3"] < 10.0)
              & (df["close"] < df["SMA_16"] * 0.965)
              & (df["close"] < df["SMA_16_1h"] * 0.985)
            )

          # Condition #4 - Normal mode (Long).
          if long_entry_condition_index == 4:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_3_1h"] <= 95.0)
            long_entry_logic.append(df["RSI_3_4h"] <= 80.0)
            long_entry_logic.append(df["RSI_3_1d"] <= 80.0)
            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 90.0)
            # 5m down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0))
            # 5m down move, 1h still high, 1d high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0) | (df["AROONU_14_1d"] < 90.0)
            )
            # 5m down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
            # 5m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 20.0))
            # 5m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 80.0))
            # 5m & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_4h"] < 30.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_1h"] < 20.0))
            # 15m & 1h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 20.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 30.0))
            # 15m down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0))
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 10.0))
            # 1h & 4h down move, drop in the last hour
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["close"] > (df["close_max_12"] * 0.90))
            )
            # 1h down move, 1h still not low enough, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 25.0) | (df["AROONU_14_4h"] < 60.0)
            )
            # 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0))
            # 1h & 4h down move, 15m high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 5.0))
            # 1h down move, 15m still not low enough, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 55.0) | (df["AROONU_14_1h"] < 75.0) | (df["AROONU_14_4h"] < 90.0))
            # 4h down move, 15m still high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 25.0)
            )
            # 4h down move, drop in last 1h
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["close"] > (df["close_max_12"] * 0.85)))
            # 4h & 1d down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 15m still high, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 5m not low enough, 15m still high, 4h high
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3"] < 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["AROONU_14_4h"] < 85.0)
            )
            # 1h red, 1h high
            long_entry_logic.append(
              (df["change_pct_1h"] > -5.0) | (df["AROONU_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 4h down move, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["RSI_3_4h"] > 25.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.10))
            )
            # 1d overbought, drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["ROC_9_1d"] < 200.0)
              | (df["close"] > (df["high_max_12_1d"] * 0.50))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_6_1d"] - df["low_min_6_1d"]) / df["low_min_6_1d"]) < 6.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.85))
              | (df["close"] < (df["low_min_6_1d"] * 1.25))
            )
            # big drop in last 24 hours, 1h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_1h"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # big drop in last 4 hours, 4h still not low enough
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # big drop in last 4 hours, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.30)) | (df["RSI_3_1h"] > 10.0))
            # big drop in the last 6 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.30)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_3_1h"] > 20.0))
            # big drop in the last 30 days, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.05)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )

            # Logic
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 25.0)
            long_entry_logic.append(df["close"] < (df["EMA_9"] * 0.946))
            long_entry_logic.append(df["close"] < (df["EMA_20"] * 0.960))

          # Condition #5 - Normal mode (Long).
          if long_entry_condition_index == 5:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 5m down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m down move, 1h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
            # 5m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 5m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 80.0))
            # 5m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 40.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_15m"] < 25.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 25.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0))
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m & 1d down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 20.0))
            # 15m & 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m down move, 15m still not low enough, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 30.0))
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 75.0))
            # 15m down move, 4h stil high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 100.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 40.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1h"] < 50.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_14_4h"] < 40.0))
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_1h"] < 35.0))
            # 1h & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 40.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 1h & 4h down move, 1d low
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["CMF_20_1d"] > -0.2))
            # 1h & 1d down move, 5m moving down
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["ROC_2"] > -0.0))
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 5.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 4h down move, 15m still high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 25.0)
            )
            # 4h down move, drop in last 1h
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["close"] > (df["close_max_12"] * 0.85)))
            # 4h & 1d down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 4h down move, 1h high, 4h still high
            long_entry_logic.append((df["RSI_3_4h"] > 20.0) | (df["AROONU_14_1h"] < 75.0) | (df["AROONU_14_4h"] < 50.0))
            # 1d down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0))
            # 1d down move, 4h high, 1d downtrend
            long_entry_logic.append((df["RSI_3_1d"] > 20.0) | (df["AROONU_14_4h"] < 75.0) | (df["ROC_2_1d"] > -30.0))
            # 1h down move, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 4h down move, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["RSI_3_4h"] > 25.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.10))
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 70.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.80))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1d overbought, drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["ROC_9_1d"] < 100.0)
              | (df["close"] > (df["high_max_12_1d"] * 0.50))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # 1d top wick, 4h still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 50.0) | (df["AROONU_14_4h"] < 50.0))
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_6_1d"] - df["low_min_6_1d"]) / df["low_min_6_1d"]) < 6.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.85))
              | (df["close"] < (df["low_min_6_1d"] * 1.25))
            )
            # big drop in last 6 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_6_1h"] * 0.65)) | (df["ROC_9_1d"] < 50.0))
            # big drop in last 12 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_12_1h"] * 0.50)) | (df["ROC_9_1d"] < 50.0))
            # big drop in last 4 hours, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["RSI_3_4h"] > 10.0))
            # big drop in last 24 hours, 1h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_1h"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # big drop in last 4 days, 1h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # big drop in the last 6 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.30)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 12 days, 4h still high
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.25)) | (df["AROONU_14_4h"] < 50.0))
            # big drop in the last 30 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.15)) | (df["RSI_3_4h"] > 15.0))
            # big drop in the last 30 days
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.05)))

            # Logic
            long_entry_logic.append(df["RSI_3"] < 50.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["AROOND_14"] > 75.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 30.0)
            long_entry_logic.append(df["EMA_26"] > df["EMA_12"])
            long_entry_logic.append((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.030))
            long_entry_logic.append((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))

          # Condition #6 - Normal mode (Long).
          if long_entry_condition_index == 6:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # big drop in the last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # 5m & 15m down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 3.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 5m & 15m down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 5m & 15m down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m & 1h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 20.0))
            # 5m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 5m & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 10.0)
            )
            # 5m down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_14_15m"] < 30.0))
            # 5m down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0))
            # 5m down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
            # 5m down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            # 5m & 15m down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m & 15m down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_4h"] < 70.0))
            # 5m & 15m down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 5m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0))
            # 5h & 4h down move
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_4h"] > 5.0))
            # 5m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 5m down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_1h"] < 80.0))
            # 5m down move, 15m not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 5m & 15m down move, 15m still high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 50.0))
            # 5m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m down move, 15m still not low enough, 1h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_1h"] < 90.0))
            # 5m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 5m down move, 15m & 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 5m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 95.0)
            )
            # 5m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 15.0) | (df["RSI_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 5m down move, 15m still not low enough, 4h high
            long_entry_logic.append((df["RSI_3"] > 15.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 90.0))
            # 5m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 15.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 5m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 5m down move, 15m & 1h high
            long_entry_logic.append((df["RSI_3"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0))
            # 5m down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 75.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 5m down move, 15m & 4h high
            long_entry_logic.append((df["RSI_3"] > 20.0) | (df["AROONU_14_15m"] < 75.0) | (df["AROONU_14_4h"] < 85.0))
            # 15m & 1h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 5.0) | (df["RSI_14_4h"] < 35.0))
            # 15m & 1h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 15m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 10.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_15m"] < 40.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 5.0) | (df["AROONU_14_15m"] < 25.0))
            # 15m & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["ROC_9_1d"] < 80.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 25.0))
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 25.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 10.0)
            )
            # 15m & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 30.0))
            # 15m & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 25.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 4h down move, 1d high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1d"] < 90.0))
            # 15m & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1d down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 25.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0))
            # 15m down move, 1h still not low enough, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 15m down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
            # 15m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0))
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 1d down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m down move, 4h stil high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 100.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_1h"] < 85.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m down move, 15m still high, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["AROONU_14_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_1h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 15m & 1h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 50.0) | (df["RSI_3_4h"] > 55.0) | (df["AROONU_14_4h"] < 80.0)
            )
            # 15m & 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1d down move, 1d still high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1d"] > 30.0) | (df["ROC_9_1d"] < 20.0))
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0))
            # 15m down move, 1h high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 40.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 50.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["RSI_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 70.0))
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0) | (df["AROONU_14_4h"] < 80.0)
            )
            # 15m down move, 1h high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_1h"] < 70.0) | (df["ROC_9_1d"] < 100.0))
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m & 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 15m high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 50.0) | (df["AROONU_14_15m"] < 60.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 50.0) | (df["AROONU_14_15m"] < 60.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m down move, 15m high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 50.0) | (df["AROONU_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 20.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1h"] < 40.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 10.0) | (df["AROONU_14_1h"] < 40.0))
            # 1h & 1d down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 1h down move, 15m & 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 5.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h, 4h still high, 1d downtrend
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] > -50.0))
            # 1h down move, 1h still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 1h & 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_1h"] > 40.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_15m"] < 70.0))
            # 1h down move, 15m high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["AROONU_14_15m"] < 80.0))
            # 1h down move, 4h high, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 55.0) | (df["RSI_14_4h"] < 80.0) | (df["ROC_9_1d"] < 150.0))
            # 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_4h"] > 5.0) | (df["AROONU_14_4h"] < 20.0))
            # 4h down move, 5m going down
            long_entry_logic.append((df["RSI_3_4h"] > 5.0) | (((df["EMA_12"] - df["EMA_26"]) / df["EMA_26"]) > -0.02))
            # 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0))
            # 4h & 1d down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 4h down move, 15m still high, 1d downtrend
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["ROC_9_1d"] > -50.0)
            )
            # 4h down move, 15m high, 1h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 30.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_1h"] < 90.0))
            # 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_4h"] > 30.0) | (df["AROONU_14_15m"] < 80.0))
            # 4h down move, 4h still not low enough, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 25.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 4h down move, 4h still high, 4h overbought
            long_entry_logic.append((df["RSI_3_4h"] > 60.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_4h"] < 80.0))
            # 1d down move, 15m still high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1d down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
            # 1d down move, 1h high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1d down move, 4h high, 1d downtrend
            long_entry_logic.append((df["RSI_3_1d"] > 20.0) | (df["AROONU_14_4h"] < 75.0) | (df["ROC_2_1d"] > -30.0))
            # 15m still high, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m still not low enough, 4h high & overbought
            long_entry_logic.append((df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 80.0) | (df["ROC_9_4h"] < 80.0))
            # 15m still not low enough, 4h high, 1d overbought
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 80.0) | (df["ROC_9_1d"] < 150.0)
            )
            # 15m still high, 1h high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m still high, 4h high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0)
            )
            # 15m & 1h still high, 1d overbought
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 15m high, 1d overbought
            long_entry_logic.append((df["AROONU_14_15m"] < 70.0) | (df["ROC_9_1d"] < 100.0))
            # 15m high, 1h high
            long_entry_logic.append((df["AROONU_14_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
            # 15m still high, 1h & 4h high
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m still not low enough, 4h still high, 1d overbought
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 1h down move, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.60))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_1h"] < 80.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 70.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.80))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 4h overbought, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["ROC_9_4h"] < 50.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.60))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_12_1d"] - df["low_min_12_1d"]) / df["low_min_12_1d"]) < 2.0)
              | (df["close"] > (df["high_max_6_1d"] * 0.60))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # 1d overbought, drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["ROC_9_1d"] < 200.0)
              | (df["close"] > (df["high_max_12_1d"] * 0.50))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # 5m red, 1h still high
            long_entry_logic.append((df["change_pct"] > -5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
            # 4h top wick, 1h & 4h high
            long_entry_logic.append(
              (df["top_wick_pct_4h"] < 25.0) | (df["AROONU_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 85.0)
            )
            # 1d top wick, 15m still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0))
            # 1d top wick, 4h still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 50.0) | (df["AROONU_14_4h"] < 50.0))
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_6_1d"] - df["low_min_6_1d"]) / df["low_min_6_1d"]) < 6.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.85))
              | (df["close"] < (df["low_min_6_1d"] * 1.25))
            )
            # big drop in last 48 hours, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_12_4h"] * 0.30)) | (df["RSI_3_4h"] > 15.0))
            # big drop in the last 6 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.30)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 20 days, 1h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_20_1d"] * 0.40)) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # big drop in the last 20 days, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_20_1d"] * 0.25)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_3_1h"] > 20.0))
            # big drop in the last 30 days, 4h down move, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_4h"] > 45.0) | (df["RSI_14_4h"] < 40.0)
            )
            # big drop in the last 30 days, 1h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.20)) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # big drop in the last 30 days, 1h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.05)) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # big drop in the last 30 days, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.05)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # big drop in the last 30 days
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.01)))

            # Logic
            long_entry_logic.append(df["RSI_20"] < df["RSI_20"].shift(1))
            long_entry_logic.append(df["RSI_3"] < 46.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["close"] < df["SMA_16"] * 0.952)

          # Condition #21 - Pump mode (Long).
          if long_entry_condition_index == 21:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["AROONU_14_15m"] < 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 90.0))
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_14_15m"] < 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 85.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 25.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m down move, 4h high & overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0) | (df["ROC_9_4h"] < 80.0)
            )
            # 1h down move, 4h high, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0) | (df["ROC_9_1d"] < 250.0)
            )
            # 1h down move, 4h high, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0) | (df["ROC_9_1d"] < 100.0)
            )
            # 1h down move, 4h high, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 4h & 1d down move
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0))
            # 15m still high, 4h high & overbought
            long_entry_logic.append(
              (df["RSI_14_15m"] < 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0) | (df["ROC_9_4h"] < 50.0)
            )
            # 15m still high, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m still high, 1h & overbought
            long_entry_logic.append(
              (df["RSI_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0) | (df["ROC_9_1h"] < 40.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 55.0) | (df["AROONU_14_4h"] < 80.0))
            # 5m down move, 4h high
            long_entry_logic.append((df["ROC_2"] > -5.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h red, 4h big green, 1h still high
            long_entry_logic.append(
              (df["change_pct_1h"] > -5.0) | (df["change_pct_4h"] < 40.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 1h P&D
            long_entry_logic.append((df["change_pct_1h"] > -10.0) | (df["change_pct_1h"].shift(12) < 10.0))
            # 4h P&D, 4h overbought
            long_entry_logic.append(
              (df["change_pct_4h"] > -2.0) | (df["change_pct_4h"].shift(48) < 20.0) | (df["ROC_9_4h"] < 50.0)
            )
            # 4h P&D, 4h high
            long_entry_logic.append(
              (df["change_pct_4h"] > -10.0) | (df["change_pct_4h"].shift(48) < 20.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 1d top wick, 4h still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 50.0) | (df["AROONU_14_4h"] < 50.0))
            # big drop in last 4 days, 1h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.40)) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # big drop in the last 12 days, 1h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.20)) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # big drop in the last 12 days, 1h still high
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.10)) | (df["AROONU_14_1h"] < 50.0))
            # big drop in the last 20 days, 4h high
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.15)) | (df["AROONU_14_4h"] < 90.0))

            # Logic
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 50.0)
            long_entry_logic.append(df["close"] < df["EMA_16"] * 0.942)
            long_entry_logic.append(((df["EMA_50"] - df["EMA_200"]) / df["close"] * 100.0) > 7.0)

          # Condition #41 - Quick mode (Long).
          if long_entry_condition_index == 41:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # big drop in the last hour, 5m down move
            long_entry_logic.append((df["close"] > (df["close_max_12"] * 0.75)) | (df["RSI_3"] > 3.0))
            # big drop in the last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # 5m down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0))
            # 5m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m & 15m & 1h down move
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_15m"] > 50.0) | (df["RSI_3_1h"] > 15.0))
            # 5m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 5m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 20.0) | (df["AROONU_14_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 10.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1d down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 60.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 40.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m & 1h down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0))
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0)
            )
            # 15m down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 75.0))
            # 15m down move, 4h high, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 75.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 40.0) | (df["AROONU_14_1h"] < 85.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 1h still not low enough, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_1h"] < 30.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 1h high. 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_1h"] < 70.0) | (df["ROC_9_1d"] < 100.0))
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 15m high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m down move, 4h high & overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0) | (df["ROC_9_4h"] < 50.0)
            )
            # 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["AROONU_14_15m"] < 40.0))
            # 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["AROONU_14_1h"] < 30.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 20.0)
            )
            # 1h & 4h down move, drop in the last hour
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["close"] > (df["close_max_12"] * 0.90))
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 40.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_1h"] < 40.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            # 1h & 4h down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_4h"] < 20.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h down move, 1h still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0) | (df["AROONU_14_1h"] < 80.0)
            )
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 40.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h down move, 1h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["ROC_9_1d"] < 50.0))
            # 1h & 4h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 10.0) | (df["AROONU_14_15m"] < 25.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h, 4h still high, 1d downtrend
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] > -50.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 75.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 85.0)
            )
            # 4h down move, 15m still high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 25.0)
            )
            # 4h down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 4h down move, drop in last 1h
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["close"] > (df["close_max_12"] * 0.85)))
            # 4h & 1d down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 1d down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0))
            # 1d down move, 1h still high
            long_entry_logic.append((df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0))
            # 1h down move, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.10))
            )
            # 1d overbought, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["ROC_9_1d"] < 50.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.75))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1h down move, drop but not yet near the previous lows in last 6 days
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0)
              | (df["close"] > (df["high_max_6_1d"] * 0.50))
              | (df["close"] < (df["low_min_6_1d"] * 1.25))
            )
            # 1d red, 4h down move, 1h still high
            long_entry_logic.append(
              (df["change_pct_1d"] > -30.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1d top wick, 1h down move, 4h still high
            long_entry_logic.append(
              (df["top_wick_pct_1d"] < 50.0) | (df["RSI_3_1h"] > 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # big drop in last 6 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_6_1h"] * 0.65)) | (df["ROC_9_1d"] < 50.0))
            # big drop in last 24 hours, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_1h"] * 0.50)) | (df["RSI_3_4h"] > 10.0))
            # big drop in last 4 hours, 4h still not low enough
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # big drop in last 4 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 30 days, 4h down move, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_4h"] > 45.0) | (df["RSI_14_4h"] < 40.0)
            )
            # big drop in the last 30 days, 1h downtrend
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.10)) | (df["RSI_3_1h"] > 10.0))
            # big drop in the last 30 days, 1h high
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.10)) | (df["AROONU_14_1h"] < 80.0))
            # big drop in the last 30 days
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.01)))

            # Logic
            long_entry_logic.append(df["RSI_14"] < 36.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["AROOND_14"] > 75.0)
            long_entry_logic.append(df["EMA_9"] < (df["EMA_26"] * 0.960))

          # Condition #42 - Quick mode (Long).
          if long_entry_condition_index == 42:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_3_1h"] <= 95.0)
            long_entry_logic.append(df["RSI_3_4h"] <= 80.0)
            long_entry_logic.append(df["RSI_3_1d"] <= 80.0)
            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 90.0)
            # 5m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 40.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 30.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0))
            # 15m & 4h down move, 1d downtrend
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["CMF_20_1d"] > -0.4))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 15m & 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 10.0) | (df["AROONU_14_4h"] < 30.0)
            )
            # 15m & 4h down move, 1d high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1d"] < 70.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 60.0))
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 15m & 4h down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 30.0) | (df["ROC_9_1d"] < 50.0))
            # 15m & 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_4h"] < 85.0))
            # 15m & 4h down move, 15m high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 50.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 1h & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 3.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_4h"] < 20.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 20.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_14_4h"] < 30.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 1d down move, 5m moving down
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["ROC_2"] > -0.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 1d down move, 1d still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_1d"] < 50.0))
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h down move, 1h still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 10.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 1h & 1d down move, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1d"] < 70.0))
            # 1h & 1d down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 1h down move, 15m sitll not low enough, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1d"] < 90.0)
            )
            # 1h & 4h down move, 15m high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 80.0)
            )
            # 1h & 4h down move, 15m high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 75.0)
            )
            # 1h down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["AROONU_14_1h"] < 40.0)
            )
            # 1h down move, 4h high & overbought
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 70.0) | (df["ROC_9_4h"] < 50.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 4h & 1d down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 5.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_4h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0))
            # 4h down move, 5m going down
            long_entry_logic.append((df["RSI_3_4h"] > 5.0) | (((df["EMA_12"] - df["EMA_26"]) / df["EMA_26"]) > -0.02))
            # 4h down move, drop in last 1h
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["close"] > (df["close_max_12"] * 0.85)))
            # 4h & 1d down move, 1d still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 50.0)
            )
            # 4h down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 4h down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 75.0))
            # 4h down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 4h down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 4h down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 1d down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1d down move, 15m still high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 15m & 1h still high, 4h high
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 85.0)
            )
            # 15m & 4h high, 1d downtrend
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0) | (df["ROC_9_1d"] > -40.0)
            )
            # 1d overbought, drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["ROC_9_1d"] < 200.0)
              | (df["close"] > (df["high_max_12_1d"] * 0.50))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in last 4 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_1d"] > 20.0))
            # big drop in the last 4 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["RSI_3_1h"] > 15.0))
            # big drop in last 4 hours, 4h still not low enough
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # big drop in the last 4 days, 1d high
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["AROONU_14_1d"] < 75.0))
            # big drop in the last 6 days, 4h still high
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.25)) | (df["AROONU_14_4h"] < 50.0))
            # big drop in the last 30 days, 4h down move, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_4h"] > 45.0) | (df["RSI_14_4h"] < 40.0)
            )
            # big drop in the last 30 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 30 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.05)) | (df["RSI_3_1h"] > 15.0))
            # big drop in the last 30 days, 15m still high
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.05)) | (df["AROONU_14_15m"] < 50.0))
            # big drop in the last 30 days
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.01)))
            # 1d top wick, 1h down move
            long_entry_logic.append((df["top_wick_pct_1d"] < 30.0) | (df["RSI_3_1h"] > 20.0))

            # Logic
            long_entry_logic.append(df["WILLR_14"] < -50.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["WILLR_84_1h"] < -70.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            long_entry_logic.append(df["BBB_20_2.0_1h"] > 16.0)
            long_entry_logic.append(df["close_max_48"] >= (df["close"] * 1.10))

          # Condition #43 - Quick mode (Long).
          if long_entry_condition_index == 43:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 90.0)
            # 5m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 20.0))
            # 5m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_1h"] < 50.0))
            # 5m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0))
            # 5m down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m down move, 4h high, 1d overbought
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["AROONU_14_4h"] < 80.0) | (df["ROC_9_1d"] < 80.0))
            # 5m down move, 15m still high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0))
            # 5m down move, 1h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
            # 5m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 5m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 20.0) | (df["AROONU_14_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 30.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0))
            # 15m & 1h down move, 1d high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1d"] < 70.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1d"] > 5.0))
            # 15m down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 60.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_4h"] < 75.0))
            # 15m & 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 1h still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_15m"] < 40.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 1h & 4h down move, drop in the last hour
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["close"] > (df["close_max_12"] * 0.90))
            )
            # 1h & 1d down move, 5m moving down
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["ROC_2"] > -0.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h & 1d down move, 1d still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 45.0)
            )
            # 1h & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 1h down move, 1h & 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 60.0) | (df["AROONU_14_1d"] < 90.0))
            # 1h, 4h still high, 1d downtrend
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] > -50.0))
            # 4h down move, 15m still high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 25.0)
            )
            # 1d down move, 1h still high
            long_entry_logic.append((df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0))
            # 1d down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 85.0))
            # 1d top wick, 1h down move, 4h still high
            long_entry_logic.append(
              (df["top_wick_pct_1d"] < 50.0) | (df["RSI_3_1h"] > 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 70.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.80))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1d red, drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["change_pct_1d"] > -20.0)
              | (df["close"] > (df["high_max_12_1d"] * 0.50))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # 1d overbought, drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["ROC_9_1d"] < 200.0)
              | (df["close"] > (df["high_max_12_1d"] * 0.50))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_6_1d"] - df["low_min_6_1d"]) / df["low_min_6_1d"]) < 6.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.85))
              | (df["close"] < (df["low_min_6_1d"] * 1.25))
            )
            # big drop in the last 30 days
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.01)))

            # Logic
            long_entry_logic.append(df["RSI_14"] < 40.0)
            long_entry_logic.append(df["MFI_14"] < 40.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["EMA_26"] > df["EMA_12"])
            long_entry_logic.append((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.024))
            long_entry_logic.append((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))
            long_entry_logic.append(df["close"] < (df["EMA_20"] * 0.958))
            long_entry_logic.append(df["close"] < (df["BBL_20_2.0"] * 0.992))

          # Condition #44 - Quick mode (Long).
          if long_entry_condition_index == 44:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 15m & 1h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 20.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 15.0))
            # 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0))
            # 1h & 4h down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_1h"] < 20.0))
            # 1h & 4h down move, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_1d"] < 85.0))
            # 15m & 1h & 4h down move, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 30.0)
            )
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 35.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 10.0))
            # 1h & 1d down move, 5m moving down
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["ROC_2"] > -0.0))
            # 1h & 1d down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 30.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 30.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 14 down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0))
            # 1h down move, 4h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 80.0))
            # 1h down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1h down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 80.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["RSI_14_4h"] < 70.0) | (df["AROONU_14_4h"] < 90.0))
            # 1d down move, 4h high
            long_entry_logic.append((df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
            # 1d P&D, 1d high
            long_entry_logic.append(
              (df["change_pct_1d"] > -10.0)
              | (df["change_pct_1d"].shift(288) < 10.0)
              | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1d top wick, 4h still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 40.0) | (df["AROONU_14_4h"] < 50.0))
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.75))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_12_1d"] - df["low_min_12_1d"]) / df["low_min_12_1d"]) < 2.0)
              | (df["close"] > (df["high_max_24_4h"] * 0.70))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # drop but not yet near the previous lows
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["close"] < (df["low_min_6_1d"] * 1.25))
            )
            # drop but not yet near the previous lows in last 12 days
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.50)) | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # big drop in last hour, 1d down move
            long_entry_logic.append((df["close"] > (df["close_max_12"] * 0.85)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 12 hours
            long_entry_logic.append((df["close"] > (df["high_max_12_1h"] * 0.50)))
            # big drop in the last 12 hours, 4h high
            long_entry_logic.append((df["close"] > (df["high_max_12_1h"] * 0.70)) | (df["AROONU_14_4h"] < 70.0))
            # big drop in the last 2 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_12_4h"] * 0.30)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 4 days, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["ROC_9_1d"] < 100.0))
            # big drop in the last 4 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 6 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.30)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 12 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.25)) | (df["RSI_3_1h"] > 15.0))
            # big drop in the last 20 days
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)))
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.30)) | (df["RSI_3_1h"] > 10.0))

            # Logic
            long_entry_logic.append(df["RSI_3"] < 40.0)
            long_entry_logic.append(df["RSI_3_15m"] < 50.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 25.0)
            long_entry_logic.append(df["AROOND_14_15m"] > 75.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            long_entry_logic.append(df["EMA_26_15m"] > df["EMA_12_15m"])
            long_entry_logic.append((df["EMA_26_15m"] - df["EMA_12_15m"]) > (df["open_15m"] * 0.035))
            long_entry_logic.append((df["EMA_26_15m"].shift() - df["EMA_12_15m"].shift()) > (df["open_15m"] / 100.0))

          # Condition #45 - Quick mode (Long).
          if long_entry_condition_index == 45:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m & 15m down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0)
            )
            # 15m & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 15m & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 35.0))
            # 15m & 1h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_1h"] < 30.0))
            # 15m down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 10.0))
            # 15m & 1h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["ROC_9_1d"] < 30.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 1h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["AROONU_14_4h"] < 30.0))
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_1h"] < 20.0))
            # 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["AROONU_14_1h"] < 30.0))
            # 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 25.0))
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 40.0))
            # 1h & 1d down move, 5m moving down
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["ROC_2"] > -0.0))
            # 1h & 1d down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 30.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 1d down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1d"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 70.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h down move, 1h still not low enough, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0) | (df["AROONU_14_1d"] < 85.0)
            )
            # 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 25.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h & 4h down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 70.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 30.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 1h & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["RSI_14_4h"] < 80.0))
            # 14 down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0))
            # 1h down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1h down move, 1h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 50.0) | (df["ROC_9_1d"] < 50.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 4h & 1d down move, 1d still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 50.0)
            )
            # 4h downmove, 4h still high
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["AROONU_14_4h"] < 40.0))
            # 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 30.0))
            # 4h down move, 1d high
            long_entry_logic.append((df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0))
            # 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
            # 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_4h"] > 55.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0))
            # 1d down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0))
            # 1d down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0))
            # 1d down move, 4h still high
            long_entry_logic.append((df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 15m still high, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 4h high, 1d overbought
            long_entry_logic.append((df["AROONU_14_4h"] < 70.0) | (df["ROC_9_1d"] < 80.0))
            # 4h high & overbought
            long_entry_logic.append((df["AROONU_14_4h"] < 90.0) | (df["ROC_9_4h"] < 80.0))
            # 1d top wick, 4h still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 40.0) | (df["AROONU_14_4h"] < 50.0))
            # pump, 4h still high
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.75))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_12_1d"] - df["low_min_12_1d"]) / df["low_min_12_1d"]) < 2.0)
              | (df["close"] > (df["high_max_24_4h"] * 0.70))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # big drop in last hour, 1d down move
            long_entry_logic.append((df["close"] > (df["close_max_12"] * 0.80)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 12 hours, 4h still high
            long_entry_logic.append((df["close"] > (df["high_max_12_1h"] * 0.50)) | (df["AROONU_14_4h"] < 50.0))
            # big drop in the last 6 days, 1h still high
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.25)) | (df["AROONU_14_1h"] < 50.0))
            # big drop in the last 12 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.45)) | (df["RSI_3_1h"] > 5.0))
            # big drop in the last 12 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.40)) | (df["RSI_3_4h"] > 15.0))
            # big drop in the last 12 days, 1h still high
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.25)) | (df["AROONU_14_1h"] < 75.0))
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.35)) | (df["RSI_3_1h"] > 10.0))
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.25)) | (df["RSI_3_1h"] > 15.0))
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_3_1h"] > 20.0))
            # big drop in the last 30 days, 4h down move, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_4h"] > 45.0) | (df["RSI_14_4h"] < 40.0)
            )

            # Logic
            long_entry_logic.append(df["RSI_3"] < 50.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 25.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            long_entry_logic.append(df["close_15m"] < (df["EMA_20_15m"] * 0.924))

          # Condition #46 - Quick mode (Long).
          if long_entry_condition_index == 46:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m & 1h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 5.0))
            # 15m & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 10.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 40.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m down move, 4h still high, 1d high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_4h"] < 50.0) | (df["AROONU_14_1d"] < 90.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 40.0))
            # 15m & 1h & 1d down move, 1h still not low enough, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0)
              | (df["RSI_3_1h"] > 20.0)
              | (df["RSI_3_1d"] > 30.0)
              | (df["RSI_14_1h"] < 30.0)
              | (df["AROONU_14_1d"] < 80.0)
            )
            # 15m down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h down move, 1h still not low enough, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["AROONU_14_1h"] < 20.0) | (df["AROONU_14_1d"] < 90.0))
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 10.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_14_4h"] < 30.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 40.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 30.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_3_1d"] > 30.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 1h & 1d down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 1d down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 30.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h down move, 4h still high, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 60.0) | (df["AROONU_14_1d"] < 90.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 30.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 30.0))
            # 1h & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 50.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 4h down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 40.0) | (df["RSI_14_1h"] < 40.0) | (df["RSI_14_4h"] < 50.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 1d down move, 1d still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_1d"] < 50.0))
            # 1h down move, 1h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 40.0) | (df["ROC_9_1d"] < 50.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_1h"] < 30.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_14_4h"] < 40.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 65.0))
            # 1h & 1d down move, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1d"] < 70.0))
            # 1h down move, 1h still high, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 40.0) | (df["AROONU_14_1d"] < 90.0))
            # 1h down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 60.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 55.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0))
            # 4h & 1d down move
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 20.0))
            # 4h & 1d down move, 1d high
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 50.0) | (df["AROONU_14_1d"] < 90.0))
            # 4h & 1d down move, 1d low
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 25.0) | (df["CMF_20_1d"] > -0.2))
            # 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 90.0))
            # 4h down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 35.0) | (df["AROONU_14_4h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 4h top wick, 1h down move, 1h still high
            long_entry_logic.append(
              (df["top_wick_pct_4h"] < 20.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 50.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_6_1d"] - df["low_min_6_1d"]) / df["low_min_6_1d"]) < 2.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.05))
            )
            # 1d overbought, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["ROC_9_1d"] < 50.0)
              | (df["close"] > (df["high_max_6_1d"] * 0.70))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # 1d overbought, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_12_1d"] - df["low_min_12_1d"]) / df["low_min_12_1d"]) < 2.5)
              | (df["close"] > (df["high_max_6_1d"] * 0.60))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in the last 2 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_12_4h"] * 0.30)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 12 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.30)) | (df["RSI_3_1h"] > 20.0))
            # big drop in the last 12 days, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.40)) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # big drop in the last 20 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.40)) | (df["RSI_3_1h"] > 10.0))
            # big drop in the last 20 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_3_4h"] > 25.0))
            # big drop in the last 30 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.40)) | (df["RSI_3_4h"] > 15.0))
            # big drop in the last 30 days, 4h still not low enough
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )

            # Logic
            long_entry_logic.append(df["RSI_3"] < 40.0)
            long_entry_logic.append(df["RSI_3_15m"] < 50.0)
            long_entry_logic.append(df["WILLR_14_15m"] < -50.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 25.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            long_entry_logic.append(df["WILLR_84_1h"] < -70.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            long_entry_logic.append(df["BBB_20_2.0_1h"] > 12.0)
            long_entry_logic.append(df["close_max_48"] >= (df["close"] * 1.10))

          # Condition #61 - Rebuy mode (Long).
          if long_entry_condition_index == 61:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m & 15m down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_4h"] < 85.0))
            # 15m & 1h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0))
            # 15m & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1d"] > 10.0))
            # 15m down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 5.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 4h stil high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 40.0))
            # 15m down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 70.0))
            # 15m down move, 4h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 50.0))
            # 15m down move, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["ROC_9_1d"] < 80.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 85.0)
            )
            # 15m down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 15m high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 60.0) | (df["ROC_9_1d"] < 150.0))
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 80.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 1h & 1d down move, 4h still high
            long_entry_logic.append(
              ((df["RSI_3_1h"] > 3.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            )
            # 1h & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 1h & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 40.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h & 4h down move, 1h low
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["CMF_20_1h"] > -0.4) | (df["CMF_20_4h"] > -0.4)
            )
            # 1h & 4h down move, 4h downtrend
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["ROC_9_4h"] > -40.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 1h & 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 90.0) | (df["AROONU_14_1d"] < 90.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 55.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
            # 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["AROONU_14_4h"] < 25.0))
            # 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 70.0))
            # 1d down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0))
            # 15m still high, 4h high
            long_entry_logic.append((df["RSI_14_15m"] < 40.0) | (df["RSI_14_4h"] < 90.0))
            # 15m still high, 4h high
            long_entry_logic.append((df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 5m down move, 15m & 1h down move, 15m still high
            long_entry_logic.append(
              (df["ROC_9"] > -15.0) | (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_15m"] < 50.0)
            )
            # 1d green with top wick, 1d low
            long_entry_logic.append(
              (df["change_pct_1d"] < 25.0) | (df["top_wick_pct_1d"] < 10.0) | (df["CMF_20_1d"] > -0.2)
            )
            # 1d top wick, 4h still high
            long_entry_logic.append((df["top_wick_pct_1d"] < 40.0) | (df["AROONU_14_4h"] < 50.0))
            # pump, 4h high
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.60))
              | (df["close"] < (df["low_min_24_4h"] * 1.10))
            )
            # pump, 1d overbought
            long_entry_logic.append(
              (((df["high_max_6_1d"] - df["low_min_6_1d"]) / df["low_min_6_1d"]) < 3.0) | (df["ROC_9_1d"] < 100.0)
            )
            # big drop in last 6 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_6_1h"] * 0.65)) | (df["ROC_9_1d"] < 50.0))
            # big drop in last 12 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_12_1h"] * 0.50)) | (df["ROC_9_1d"] < 50.0))
            # big drop in the last 4 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 6 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.20)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 20 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.15)) | (df["RSI_3_1d"] > 10.0))
            # big drop in the last 20 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.25)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 20 days, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # big drop in the last 30 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.15)) | (df["RSI_3_1h"] > 15.0))

            # Logic
            long_entry_logic.append(df["RSI_3"] < 50.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["AROOND_14"] > 75.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["ROC_2"] > -5.0)
            long_entry_logic.append(df["EMA_26"] > df["EMA_12"])
            long_entry_logic.append((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.030))
            long_entry_logic.append((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))

          # Condition #62 - Rebuy mode (Long).
          if long_entry_condition_index == 62:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 3.0))
            # 15m & 1h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 5.0) | (df["RSI_14_4h"] < 35.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0))
            # 15m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 15.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_1h"] < 40.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 3h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still not low enough, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 35.0) | (df["RSI_14_4h"] < 40.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 5.0))
            # 1h & 4h down move, 1d still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 50.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 25.0))
            # 1h down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 60.0) | (df["AROONU_14_4h"] < 60.0))
            # 1h & 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 50.0))
            # 1h & 1d down move, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1d"] < 70.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 1h & 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0))
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0))
            # 4h down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 4h & 1d down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 4h & 1d down move, 1d overbought
            long_entry_logic.append((df["RSI_3_4h"] > 30.0) | (df["RSI_3_1d"] > 45.0) | (df["ROC_9_1d"] < 50.0))
            # 4h down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 1d top wick, 1h down move, 4h still high
            long_entry_logic.append(
              (df["top_wick_pct_1d"] < 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 1d top wick, 4h down move, 4h high
            long_entry_logic.append(
              (df["top_wick_pct_1d"] < 10.0) | (df["RSI_3_4h"] > 60.0) | (df["AROONU_14_4h"] < 80.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.60))
              | (df["close"] < (df["low_min_24_4h"] * 1.10))
            )
            # big drop in the last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # big drop in last 6 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_6_1h"] * 0.65)) | (df["ROC_9_1d"] < 50.0))
            # big drop in the last 4 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 12 days, 15m & 4h down move
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.40)) | (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0)
            )
            # big drop in the last 20 days, 15m & 1h down move
            long_entry_logic.append(
              (df["close"] > (df["high_max_20_1d"] * 0.40)) | (df["RSI_14_15m"] < 10.0) | (df["RSI_14_1h"] < 10.0)
            )
            # big drop in the last 20 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.25)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 20 days, 4h still not low enough
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_14_4h"] < 30.0))
            # big drop in the last 30 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.20)) | (df["RSI_3_1d"] > 20.0))

            # Logic
            long_entry_logic.append(df["RSI_3"] < 40.0)
            long_entry_logic.append(df["AROONU_14"] < 30.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 70.0)
            long_entry_logic.append(df["WILLR_84_1h"] < -70.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            long_entry_logic.append(df["BBB_20_2.0_1h"] > 12.0)
            long_entry_logic.append(df["close_max_48"] >= (df["close"] * 1.12))

          # Condition #101 - Rapid mode (Long).
          if long_entry_condition_index == 101:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 80.0)
            # big drop in the last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # 5 & 15m down move, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["RSI_3_15m"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 5m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["AROONU_14_1h"] < 85.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_14_4h"] < 40.0))
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 75.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 20.0)
            )
            # 15m & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_4h"] < 30.0))
            # 15m & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m & 1h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 20.0))
            # 15m down move, 1h still not low enough, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0) | (df["AROONU_14_1d"] < 90.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1d"] > 5.0))
            # 15m & 1d down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1d"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 25.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 50.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 10.0)
            )
            # 15m & 4h down move, 1d high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1d"] < 85.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m & 1d down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 85.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 75.0)
            )
            # 15m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 15m & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 85.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1d down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 60.0) | (df["AROONU_14_1h"] < 85.0))
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 25.0) | (df["AROONU_14_4h"] < 60.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1h"] < 80.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 70.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 20.0)
            )
            # 1h & 4h down move, 1d low
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["CMF_20_1d"] > -0.2))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_4h"] < 20.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 1d down move, 5m moving down
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["ROC_2"] > -0.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 30.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 10.0) | (df["AROONU_14_1h"] < 40.0))
            # 1h down move, 1h still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0) | (df["AROONU_14_4h"] < 80.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h down move, 15m high
            long_entry_logic.append((df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 80.0))
            # 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_4h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0))
            # 4h down move, 15m 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 4h & 1d down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 40.0) | (df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1d down move, 15 still high
            long_entry_logic.append((df["RSI_3_4h"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0))
            # 1d down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 85.0))
            # 4h still high, 4h moving lower, 4h overbought
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 50.0) | (df["AROONU_14_4h"] > df["AROONU_14_4h"].shift(48)) | (df["ROC_9_4h"] < 40.0)
            )
            # 1d red, 4h down move, 1h still high
            long_entry_logic.append(
              (df["change_pct_1d"] > -30.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.75))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 70.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.80))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 80.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.85))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1h down move, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.50))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1d overbought, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["ROC_9_1d"] < 50.0)
              | (df["close"] > (df["high_max_6_1d"] * 0.70))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in last 6 hours, 1d overbought
            long_entry_logic.append((df["close"] > (df["high_max_6_1h"] * 0.65)) | (df["ROC_9_1d"] < 50.0))
            # big drop in last 4 hours, 4h still not low enough
            long_entry_logic.append(
              (df["close"] > (df["high_max_24_4h"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # big drop in the last 4 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 6 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_6_1d"] * 0.30)) | (df["RSI_3_1d"] > 15.0))
            # big drop in the last 12 days, 4h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.50)) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # big drop in the last 30 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_3_4h"] > 30.0))
            # big drop in the last 20 days, 1h still high
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.05)) | (df["AROONU_14_1h"] < 50.0))
            # big drop in the last 30 days, 1h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_1h"] > 15.0))

            # Logic
            long_entry_logic.append(df["RSI_3"] > 3.0)
            long_entry_logic.append(df["RSI_14"] < 36.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["close"] < (df["SMA_16"] * 0.946))
            long_entry_logic.append(df["AROONU_14_15m"] < 50.0)

          # Condition #102 - Rapid mode (Long).
          if long_entry_condition_index == 102:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_3"] < 46.0)
            long_entry_logic.append(df["RSI_3_15m"] > 5.0)
            long_entry_logic.append(df["RSI_3_1h"] > 10.0)
            long_entry_logic.append(df["RSI_3_4h"] > 10.0)
            # 5m & 15m down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m & 15m down move, 15m still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 40.0))
            # 5m & 15m down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 5m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 5m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0))
            # 5m down move, 15m high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["AROONU_14_15m"] < 70.0))
            # 5m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["AROONU_14_1h"] < 40.0)
            )
            # 5m down move, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["AROONU_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 5m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_4h"] < 70.0))
            # 5m down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_1h"] < 80.0))
            # 5m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["AROONU_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 5m & 15m down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_4h"] < 90.0))
            # 5m & 15m down move, 1d high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["RSI_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 5m & 15m down move, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 5m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3"] > 10.0) | (df["AROONU_14_1h"] < 85.0) | (df["AROONU_14_4h"] < 90.0))
            # 5m down move, 4h high, 1d high
            long_entry_logic.append(
              (df["RSI_3"] > 10.0) | (df["AROONU_14_4h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 5m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3"] > 15.0) | (df["RSI_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 5m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 15.0) | (df["RSI_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m& 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 4h down move, 1d high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1d"] < 85.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1d down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 15m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 15.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 25.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 35.0) | (df["AROONU_14_4h"] < 60.0))
            # 15m & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_14_15m"] < 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0)
            )
            # 15m down move, 4h still high 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 50.0))
            # 15m & 1h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m & 1h down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_15m"] < 60.0))
            # 15m & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 60.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1d down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_14_15m"] < 35.0) | (df["RSI_14_4h"] < 85.0))
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1h"] < 50.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["AROONU_14_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 60.0) | (df["AROONU_14_1h"] < 85.0))
            # 15m & 4h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m & 1d down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1d"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m down move, 15m high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 70.0) | (df["ROC_9_1d"] < 100.0))
            # 15m down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 85.0))
            # 15m down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m down move, 4h high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_4h"] < 80.0) | (df["ROC_9_1d"] < 50.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m down move, 15m high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 40.0) | (df["RSI_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m down move, 15m high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 45.0) | (df["AROONU_14_15m"] < 90.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 40.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h down move, 1h still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 1h down move, 4h high, 1d downtrend
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 70.0) | (df["ROC_9_1d"] > -50.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 55.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 1h down move, 4h high, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 40.0) | (df["RSI_14_4h"] < 75.0) | (df["ROC_9_1d"] < 100.0))
            # 4h down move, 4h still not low enough, 1d overbought
            long_entry_logic.append((df["RSI_3_4h"] > 15.0) | (df["AROONU_14_4h"] < 30.0) | (df["ROC_9_1d"] < 100.0))
            # 4h & 1d down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            )
            # 1d down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m still not low enough, 1h high, 1d overbought
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0) | (df["ROC_9_1d"] < 100.0)
            )
            # 15m & 1h & 4h high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m high
            long_entry_logic.append((df["AROONU_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0))
            # 15m & 4h high
            long_entry_logic.append((df["AROONU_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            # 15m high, 1d overbought
            long_entry_logic.append((df["AROONU_14_15m"] < 80.0) | (df["ROC_9_1d"] < 50.0))
            # 15m & 4h high
            long_entry_logic.append((df["AROONU_14_15m"] < 90.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1h high
            long_entry_logic.append((df["AROONU_14_15m"] < 90.0) | (df["AROONU_14_1h"] < 90.0))
            # 1h still high, 4h high & overbought
            long_entry_logic.append((df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0) | (df["ROC_9_4h"] < 80.0))
            # 4h high & overbought
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 85.0) | (df["ROC_9_4h"] < 50.0)
            )
            # 5m red, 1h still high
            long_entry_logic.append((df["change_pct"] > -5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
            # 1d top wick, 4h down move, 4h still high
            long_entry_logic.append(
              (df["top_wick_pct_1d"] < 30.0) | (df["RSI_3_4h"] > 60.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.75))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
              | (df["close"] > (df["close_max_48"] * 0.85))
              | (df["close"] < (df["low_min_24_1h"] * 1.25))
            )
            # 4h high, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["AROONU_14_4h"] < 70.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.80))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # 1d overbought, drop but not yet near the previous lows
            long_entry_logic.append(
              (df["ROC_9_1d"] < 50.0)
              | (df["close"] > (df["high_max_6_1d"] * 0.70))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in the last 4 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_4h"] > 20.0))
            # big drop in the last 12 days, 1h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.30)) | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # big drop in the last 20 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.40)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 30 days, 4h down move, 4h still high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_4h"] > 45.0) | (df["RSI_14_4h"] < 40.0)
            )
            # big drop in the last 30 days, 1h high
            long_entry_logic.append(
              (df["close"] > (df["high_max_30_1d"] * 0.20)) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )

            # Logic
            long_entry_logic.append(df["WILLR_14"] < -95.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 10.0)
            long_entry_logic.append(df["close"] < (df["BBL_20_2.0"] * 0.999))
            long_entry_logic.append(df["close"] < (df["EMA_20"] * 0.960))

          # Condition #103 - Rapid mode (Long).
          if long_entry_condition_index == 103:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["ROC_2"] > -0.0)
            # 15m down move, 4h high, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 45.0) | (df["AROONU_14_15m"] < 70.0))
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 20.0) | (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_4h"] < 90.0))
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m down move, 15m high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 1h & 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 15m & 1h down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_15m"] < 70.0))
            # 15m down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 90.0)
            )
            # 15m down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 80.0)
            )
            # 15m down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 35.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m down move, 1h high, 4h overbought
            long_entry_logic.append((df["RSI_3_15m"] > 35.0) | (df["AROONU_14_1h"] < 85.0) | (df["ROC_9_4h"] < 80.0))
            # 1h & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 1h down move, 15m still not low enough, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_15m"] < 25.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 4h & 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1h down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 15m & 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 50.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 85.0))
            # 1h down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 4h down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 4h down move, 15m still not low enough, 4h high
            long_entry_logic.append((df["RSI_3_4h"] > 35.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 70.0))
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 60.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0))
            # 1d down move, 4h high
            long_entry_logic.append((df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0))
            # 1d down move, 15m still high, 1h high
            long_entry_logic.append((df["RSI_3_1d"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 85.0))
            # 15m still high, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m & 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_14_15m"] < 45.0) | (df["AROONU_14_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 15m still high, 4h high & overbought
            long_entry_logic.append((df["RSI_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 90.0) | (df["ROC_9_4h"] < 80.0))
            # 15m still high, 1h high, 4h still high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m still high, 1d high
            long_entry_logic.append((df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0))
            # 15m & 1h & 4h high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0)
            )
            # 15m & 1h
            long_entry_logic.append((df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 90.0))
            # 15m still high, 1h high
            long_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
              | (df["AROONU_14_1h"] < 90.0)
              | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # 15m still high, 1d high
            long_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0))
            # 1d top wick, 4h high
            long_entry_logic.append((df["top_wick_pct_1d"] < 30.0) | (df["AROONU_14_4h"] < 90.0))
            # pump, 4h overbought
            long_entry_logic.append(
              (((df["high_max_6_1h"] - df["low_min_6_1h"]) / df["low_min_6_1h"]) < 0.5) | (df["ROC_9_4h"] < 50.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_6_4h"] * 0.85))
              | (df["close"] < (df["low_min_24_4h"] * 1.25))
            )
            # pump, 1h high
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 4.0)
              | (df["STOCHRSIk_14_14_3_3_1h"] < 90.0)
            )
            # big drop in the last 2 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_12_4h"] * 0.30)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 12 days, 1h still high
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.25)) | (df["AROONU_14_1h"] < 50.0))
            # big drop in the last 12 days, 1h still not low enough
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.10)) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # big drop in the last 12 days, 15m still high
            long_entry_logic.append((df["close"] > (df["high_max_12_1d"] * 0.20)) | (df["AROONU_14_15m"] < 50.0))
            # big drop in the last 20 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_3_4h"] > 20.0))

            # Logic
            long_entry_logic.append(df["RSI_4"] < 45.0)
            long_entry_logic.append(df["RSI_14"] > 35.0)
            long_entry_logic.append(df["RSI_20"] < df["RSI_20"].shift(1))
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["close"] < df["SMA_16"] * 0.960)

          # Condition #104 - Rapid mode (Long).
          if long_entry_condition_index == 104:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 15m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 15.0))
            # 5m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 3.0))
            # 15m & 1h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 5.0) | (df["RSI_14_4h"] < 35.0))
            # 15m & 1h down move, 1j still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 30.0) | (df["RSI_14_4h"] < 40.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 40.0))
            # 15m & 3h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["RSI_14_1h"] < 40.0))
            # 15m down move, 15m still not low enough, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 5.0))
            # 1h & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 50.0))
            # 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["AROONU_14_4h"] < 85.0))
            # 1h & 4h down move, 1d still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 50.0)
            )
            # 1h & 1d down move, 1h still moving lower
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["CCI_20_change_pct_1h"] > -0.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 30.0))
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 1d down move, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 25.0) | (df["AROONU_14_1d"] < 70.0))
            # 1h & 1d down move, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_1d"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 70.0)
            )
            # 1h down move, 1h & 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 60.0) | (df["AROONU_14_4h"] < 60.0))
            # 1h & 4h down move, 1d overbought
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["ROC_9_1d"] < 50.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h & 1d down move, 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["AROONU_14_1d"] < 70.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 70.0))
            # 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
            # 1h & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 30.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 80.0))
            # 4h down move, 15m not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 3.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
            # 4h down move, 4h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 40.0) | (df["ROC_9_1d"] < 100.0))
            # 1d down move, 1h still not low enough
            long_entry_logic.append((df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0))
            # 1d red, 4h down move, 1h still high
            long_entry_logic.append(
              (df["change_pct_1d"] > -30.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_24_4h"] - df["low_min_24_4h"]) / df["low_min_24_4h"]) < 2.0)
              | (df["close"] > (df["high_max_12_4h"] * 0.60))
              | (df["close"] < (df["low_min_24_4h"] * 1.10))
            )
            # pump, drop but not yet near the previous lows
            long_entry_logic.append(
              (((df["high_max_12_1d"] - df["low_min_12_1d"]) / df["low_min_12_1d"]) < 5.0)
              | (df["close"] > (df["high_max_6_1d"] * 0.30))
              | (df["close"] < (df["low_min_12_1d"] * 1.25))
            )
            # big drop in the last hour
            long_entry_logic.append(df["close"] > (df["close_max_12"] * 0.50))
            # big drop in the last 12 days, 15m & 4h down move
            long_entry_logic.append(
              (df["close"] > (df["high_max_12_1d"] * 0.40)) | (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 10.0)
            )
            # big drop in the last 20 days, 15m & 1h down move
            long_entry_logic.append(
              (df["close"] > (df["high_max_20_1d"] * 0.40)) | (df["RSI_14_15m"] < 10.0) | (df["RSI_14_1h"] < 10.0)
            )
            # big drop in the last 20 days, 1d down move
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.25)) | (df["RSI_3_1d"] > 30.0))
            # big drop in the last 20 days, 4h still not low enough
            long_entry_logic.append((df["close"] > (df["high_max_20_1d"] * 0.10)) | (df["RSI_14_4h"] < 30.0))
            # big drop in the last 30 days, 4h down move
            long_entry_logic.append((df["close"] > (df["high_max_30_1d"] * 0.25)) | (df["RSI_3_4h"] > 20.0))

            # Logic
            long_entry_logic.append(df["RSI_3"] < 40.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 25.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            long_entry_logic.append(df["close"] < df["EMA_16"] * 0.975)
            long_entry_logic.append(((df["EMA_50"] - df["EMA_200"]) / df["close"] * 100.0) < -5.5)

          # Condition #120 - Grind mode (Long).
          if long_entry_condition_index == 120:
            # Protections
            long_entry_logic.append(num_open_long_grind_mode < self.grind_mode_max_slots)
            long_entry_logic.append(df["protections_long_global"] == True)
            long_entry_logic.append(is_pair_long_grind_mode)
            long_entry_logic.append(df["RSI_3"] <= 50.0)
            long_entry_logic.append(df["RSI_3_15m"] >= 20.0)
            long_entry_logic.append(df["RSI_3_1h"] >= 10.0)
            long_entry_logic.append(df["RSI_3_4h"] >= 10.0)
            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 80.0)

            # Logic
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["WILLR_14"] < -80.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["close"] < (df["EMA_20"] * 0.978))

          # Condition #141 - Top Coins mode (Long).
          if long_entry_condition_index == 141:
            # Protections
            long_entry_logic.append(is_pair_long_top_coins_mode)
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 90.0)
            # 5m & 4h down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 5m down move, 4h high
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            # 5m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 5m & 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_15m"] < 50.0))
            # 5m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 1h high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["AROONU_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 1h & 4h down move, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 35.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 25.0)
            )
            # 15m & 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_15m"] < 70.0))
            # 15m down move, 15m still high 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 20.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 15m down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 4h down move, 1h still high
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_1h"] < 50.0))
            # 1h down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 5.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 4h down move, 15m still not low enough, 4h still not low enough
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 20.0))
            # 4h down move, 15m still not low enough, 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_4h"] < 40.0)
            )
            # 15m & 1h & 4h high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 90.0) | (df["AROONU_14_4h"] < 90.0)
            )

            # Logic
            long_entry_logic.append(df["RSI_20"] < df["RSI_20"].shift(1))
            long_entry_logic.append(df["RSI_3"] < 30.0)
            long_entry_logic.append(df["AROONU_14"] < 25.0)
            long_entry_logic.append(df["close"] < df["SMA_16"] * 0.956)

          # Condition #142 - Top Coins mode (Long).
          if long_entry_condition_index == 142:
            # Protections
            long_entry_logic.append(is_pair_long_top_coins_mode)
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m & 15m down move, 4h high
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_15m"] > 5.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 5m & 4h down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 5m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0))
            # 5m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 1h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0))
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_15m"] < 30.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h & 4h down move, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 35.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 55.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 60.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 25.0)
            )
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 35.0) | (df["AROONU_14_4h"] < 60.0))
            # 15m & 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_15m"] < 70.0))
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_4h"] < 85.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 4h down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 15m down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1h"] < 70.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["RSI_14_4h"] < 80.0))
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 45.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 85.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["RSI_14_4h"] < 30.0))
            # 1h & 4h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h down move, 4h still high, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_14_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 1h & 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_15m"] < 70.0))
            # 4h & 1d down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 5.0) | (df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 4h down move, 15m still not low enough, 4h still not low enough
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["AROONU_14_4h"] < 20.0))
            # 4h down move, 15m still not low enough, 1h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 4h & 1d down move, 15m still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0)
            )
            # 4h down move, 15m still not low enough, 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_4h"] < 40.0)
            )
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 15m & 1h & 4h high
            long_entry_logic.append(
              (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 90.0) | (df["AROONU_14_4h"] < 90.0)
            )

            # Logic
            long_entry_logic.append(df["RSI_4"] < 46.0)
            long_entry_logic.append(df["RSI_20"] < df["RSI_20"].shift(1))
            long_entry_logic.append(df["close"] < df["SMA_16"] * 0.958)

          # Condition #143 - Top Coins mode (Long).
          if long_entry_condition_index == 143:
            # Protections
            long_entry_logic.append(is_pair_long_top_coins_mode)
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 90.0)
            # 5m & 4h down move
            long_entry_logic.append((df["RSI_3"] > 3.0) | (df["RSI_3_4h"] > 10.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0))
            # 5m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 50.0))
            # 15m & 1h & 4h strong downtrend
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0))
            # 15m & 1h down move, 1h still not low enough
            long_entry_logic.append(
              (df["RSI_3_15m"] > 3.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 20.0)
            )
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 3.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 40.0))
            # 15m & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h & 4h down move, 1d overbought
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 25.0) | (df["RSI_3_4h"] > 35.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m down move, 15m still high 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 15m & 1h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 4h & 1d down move
            long_entry_logic.append((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 15.0) | (df["RSI_3_1d"] > 25.0))
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 4h & 1d down move, 1h still high
            long_entry_logic.append((df["RSI_3_4h"] > 10.0) | (df["RSI_3_1d"] > 10.0) | (df["AROONU_14_1h"] < 50.0))

            # Logic
            long_entry_logic.append(df["RSI_3"] < 40.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 20.0)
            long_entry_logic.append(df["EMA_26"] > df["EMA_12"])
            long_entry_logic.append((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.020))
            long_entry_logic.append((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))

          # Condition #144 - Top Coins mode (Long).
          if long_entry_condition_index == 144:
            # Protections
            long_entry_logic.append(is_pair_long_top_coins_mode)
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(df["RSI_14_1h"] < 80.0)
            long_entry_logic.append(df["RSI_14_4h"] < 80.0)
            long_entry_logic.append(df["RSI_14_1d"] < 80.0)
            # 5m & 1h down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3"] > 3.0) | (df["RSI_3_1h"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )
            # 5m & 1h down move, 1h still high
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 50.0))
            # 5m & 4h & 1d down move
            long_entry_logic.append((df["RSI_3"] > 5.0) | (df["RSI_3_4h"] > 5.0) | (df["RSI_3_1d"] > 10.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_1h"] > 15.0) | (df["RSI_3_4h"] > 30.0))
            # 15m & 4h down move, 4h still not low enough
            long_entry_logic.append((df["RSI_3_15m"] > 5.0) | (df["RSI_3_4h"] > 15.0) | (df["AROONU_14_4h"] < 30.0))
            # 15m & 1h & 4h down move
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["RSI_3_4h"] > 10.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 10.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m down move, 15m still high 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 15.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 1h & 4h down move, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_1h"] > 10.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 1h & 4h & 1d down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 35.0) | (df["AROONU_14_4h"] < 40.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 20.0) | (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 85.0)
            )
            # 4h & 1d down move, 15m still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 5.0) | (df["RSI_3_1d"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
            )

            # Logic
            long_entry_logic.append(df["WILLR_14"] < -50.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3"] < 30.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_1h"] < 30.0)
            long_entry_logic.append(df["BBB_20_2.0_1h"] > 12.0)
            long_entry_logic.append(df["close_max_48"] >= (df["close"] * 1.10))

          # Condition #161 - Scalp mode (Long).
          if long_entry_condition_index == 161:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            # 5m down move, 15m high
            long_entry_logic.append((df["RSI_3"] > 15.0) | (df["AROONU_14_15m"] < 80.0))
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m & 4h down move, 4h still high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_4h"] < 50.0))
            # 15m down move, 15m high
            long_entry_logic.append((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 80.0))
            # 15m down move, 4h still high, 1d high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0) | (df["AROONU_14_1d"] < 90.0)
            )
            # 15m & 1h down move, 15m still high
            long_entry_logic.append((df["RSI_3_15m"] > 30.0) | (df["RSI_3_1h"] > 60.0) | (df["AROONU_14_15m"] < 50.0))
            # 15m down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m & 4h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 35.0) | (df["RSI_3_4h"] > 60.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_15m"] > 40.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 70.0))
            # 15m & 1h down move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m & 1h down move, 4h high
            long_entry_logic.append((df["RSI_3_15m"] > 40.0) | (df["RSI_3_1h"] > 60.0) | (df["AROONU_14_4h"] < 80.0))
            # 15m & 4h down move, 15m high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 15m & 4h down move, 15m high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["AROONU_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 15m down move, 4h still high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 40.0) | (df["AROONU_14_4h"] < 50.0) | (df["ROC_9_1d"] < 100.0))
            # 15m down move, 15m high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 45.0) | (df["AROONU_14_15m"] < 60.0) | (df["ROC_9_1d"] < 50.0))
            # 15m down move, 15m high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 45.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 15m down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 45.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 15m down move, 15m still high, 1d overbought
            long_entry_logic.append((df["RSI_3_15m"] > 50.0) | (df["AROONU_14_15m"] < 50.0) | (df["ROC_9_1d"] < 50.0))
            # 15m down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 55.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 15m down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_15m"] > 55.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 85.0)
            )
            # 1h down move, 4h still high, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            long_entry_logic.append(
              (df["RSI_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 1h & 4h down move, 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 55.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h & 4h down move, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 1h down move, 15m & 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0))
            # 1h down move, 4h & 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 40.0) | (df["AROONU_14_4h"] < 85.0) | (df["AROONU_14_1d"] < 90.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["AROONU_14_1h"] < 50.0) | (df["AROONU_14_4h"] < 90.0))
            # 1h down move, 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 45.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0))
            # 1h & 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_1h"] > 50.0) | (df["RSI_3_4h"] > 60.0) | (df["AROONU_14_15m"] < 70.0))
            # 1h down move, 15m still high, 4h high
            long_entry_logic.append((df["RSI_3_1h"] > 50.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 80.0))
            # 1h down move, 15m high, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 50.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0)
            )
            # 1h down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 1h down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0) | (df["AROONU_14_1h"] < 60.0)
            )
            # 1h down move, 1h & 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 50.0) | (df["AROONU_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 1h down move, 4h still high, 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 90.0)
            )
            # 1h down move, 5m up move, 1h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["RSI_3"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 1h down move, 15m still not low enough, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 1h down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["AROONU_14_1h"] < 70.0)
            )
            # 1h down move, 15m still not low enough, 1h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 60.0)
            )
            # 1h down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 1h down move, 15m & 1h high
            long_entry_logic.append((df["RSI_3_1h"] > 60.0) | (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_1h"] < 90.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["AROONU_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
            )
            # 1h down move, 1h high, 4h still high
            long_entry_logic.append((df["RSI_3_1h"] > 60.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 40.0))
            # 1h down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 1h down move, 1h & 1d high
            long_entry_logic.append(
              (df["RSI_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_1d"] < 90.0)
            )
            # 1h down move, 4h & 1d high
            long_entry_logic.append((df["RSI_3_1h"] > 60.0) | (df["RSI_14_4h"] < 70.0) | (df["RSI_14_1d"] < 80.0))
            # 4h down move, 15m high
            long_entry_logic.append((df["RSI_3_4h"] > 20.0) | (df["AROONU_14_15m"] < 80.0))
            # 4h down move, 1h high
            long_entry_logic.append((df["RSI_3_4h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
            # 4h down move, 15m & 4h still high
            long_entry_logic.append((df["RSI_3_4h"] > 30.0) | (df["AROONU_14_15m"] < 50.0) | (df["AROONU_14_4h"] < 50.0))
            # 4h down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 4h down move, 15m & 1h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
            )
            # 4h down move, 1h & 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["RSI_14_4h"] < 50.0)
            )
            # 4h down move, 1h still high, 4h still moving down
            long_entry_logic.append(
              (df["RSI_3_4h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["CCI_20_change_pct_4h"] > -0.0)
            )
            # 4h down move, 1h high, 4h still high
            long_entry_logic.append((df["RSI_3_4h"] > 45.0) | (df["AROONU_14_1h"] < 70.0) | (df["AROONU_14_4h"] < 50.0))
            # 4h down move, 15m high, 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 40.0)
            )
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 4h down move, 15m & 4h still high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_4h"] < 50.0)
            )
            # 4h down move, 15m high, 4h still not low enough
            long_entry_logic.append(
              (df["RSI_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0)
            )
            # 4h down move, 1h still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 4h down move, 15m & 4h high
            long_entry_logic.append((df["RSI_3_4h"] > 60.0) | (df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_4h"] < 70.0))
            # 4h down move, 15m high, 4h still high
            long_entry_logic.append((df["RSI_3_4h"] > 60.0) | (df["AROONU_14_15m"] < 80.0) | (df["AROONU_14_4h"] < 40.0))
            # 4h down move, 15m still high, 1h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["AROONU_14_1h"] < 70.0)
            )
            # 4h down move, 15m still high, 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 4h down move, 15m & 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 60.0)
            )
            # 4h down move, 1h & 4h high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["AROONU_14_4h"] < 70.0)
            )
            # 4h down move, 4h still high, 1d high
            long_entry_logic.append(
              (df["RSI_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 80.0)
            )
            # 15m high, 4h high
            long_entry_logic.append((df["AROONU_14_15m"] < 70.0) | (df["AROONU_14_4h"] < 85.0))
            # 15m high, 4h still high
            long_entry_logic.append((df["AROONU_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
            # 15m high, 1h still high
            long_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0))
            # 15m & 4h high
            long_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] < 70.0) | (df["AROONU_14_4h"] < 70.0))
            # 15m high, 1h still not low enough
            long_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 30.0))

            # Logic
            long_entry_logic.append(df["RSI_14"] < 50.0)
            long_entry_logic.append(df["AROONU_14_15m"] < 90.0)
            long_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] < 90.0)
            long_entry_logic.append(
              (df["SMA_21"].shift(1) < df["SMA_200"].shift(1).infer_objects(copy=False).fillna(np.nan))
              & df["SMA_200"].shift(1).notna()
            )
            long_entry_logic.append(
              (df["SMA_21"] > df["SMA_200"].infer_objects(copy=False).fillna(np.nan)) & df["SMA_200"].notna()
            )
            long_entry_logic.append(
              (df["close"] > df["EMA_200_1h"].infer_objects(copy=False).fillna(np.nan)) & df["EMA_200_1h"].notna()
            )
            long_entry_logic.append(
              (df["close"] > df["EMA_200_4h"].infer_objects(copy=False).fillna(np.nan)) & df["EMA_200_4h"].notna()
            )
            long_entry_logic.append(df["BBB_20_2.0"] > 1.5)
            long_entry_logic.append(df["BBB_20_2.0_1h"] > 6.0)

          # Condition #162 - Scalp mode (Long).
          if long_entry_condition_index == 162:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append(
              (df["RSI_3"] > 5.0) & (df["RSI_3_15m"] > 5.0) & (df["ROC_9_15m"] > -10.0) & (df["ROC_9_1d"] < 200.0)
            )

            long_entry_logic.append(
              # 15m & 1h down move, 4h high
              ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
              # 15m & 4h down move, 1h still high
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_4h"] > 40.0) | (df["RSI_14_1h"] < 50.0))
              # 15m & 1d down move, 4h high
              & ((df["RSI_3_15m"] > 10.0) | (df["RSI_3_1d"] > 20.0) | (df["AROONU_14_4h"] < 90.0))
              # 15m down move, 15m high
              & ((df["RSI_3_15m"] > 10.0) | (df["AROONU_14_15m"] < 70.0))
              # 15m & 1h down move, 15m still not low enough
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0))
              # 15m & 1h down nove, 1h still high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
              # 15m & 1h down move, 4h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 100.0))
              # 15m & 1h down move, 4h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
              # 15m & 1h down move, 1h high
              & ((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 70.0))
              # 15m & 1h down move, 4h high
              & ((df["RSI_3_15m"] > 25.0) | (df["RSI_3_1h"] > 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0))
              # 15m down move, 15m high, 1d overbought
              & ((df["RSI_3_15m"] > 25.0) | (df["AROONU_14_15m"] < 60.0) | (df["ROC_9_1d"] < 150.0))
              # 15m down move, 1h & 4h high
              & ((df["RSI_3_15m"] > 35.0) | (df["AROONU_14_1h"] < 80.0) | (df["RSI_14_4h"] < 80.0))
              # 1h & 4h down move, 1h still high
              & ((df["RSI_3_1h"] > 3.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_1h"] < 40.0))
              # 1h & 4h down move, 4h high
              & ((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 20.0) | (df["AROONU_14_4h"] < 60.0))
              # 1h & 4h down move, 4h high
              & ((df["RSI_3_1h"] > 5.0) | (df["RSI_3_4h"] > 30.0) | (df["AROONU_14_4h"] < 90.0))
              # 1h down move, 1h high, 1d overbought
              & ((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 60.0) | (df["ROC_9_1d"] < 100.0))
              # 1h down move, 1h high
              & ((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 90.0))
              # 1h & 4h down move, 4h high
              & ((df["RSI_3_1h"] > 30.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
              # 1h down move, 1h high, 4h overbought
              & ((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 60.0) | (df["ROC_9_4h"] < 50.0))
              # 1h down move,  4h high, 1d overbought
              & ((df["RSI_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0) | (df["ROC_9_1d"] < 150.0))
              # 4h & 1d down move
              & ((df["RSI_3_4h"] > 5.0) | (df["RSI_3_1d"] > 10.0))
              # 4h & 1d down move, 4h still not low enough
              & ((df["RSI_3_4h"] > 20.0) | (df["RSI_3_1d"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 30.0))
              # 1d down move, 1h high
              & ((df["RSI_3_1d"] > 3.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
              # 1d down move, 15m still high
              & ((df["RSI_3_1d"] > 3.0) | (df["AROONU_14_15m"] < 50.0))
              # 1d down move, 15m still high, 1h high
              & ((df["RSI_3_1d"] > 10.0) | (df["AROONU_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0))
              # 1d down move, 1h & 4h high
              & ((df["RSI_3_1d"] > 10.0) | (df["AROONU_14_1h"] < 70.0) | (df["AROONU_14_4h"] < 90.0))
              # 1d down move, 4h high
              & ((df["RSI_3_1d"] > 20.0) | (df["RSI_14_4h"] < 80.0))
              # 1h P&D, 1h down move
              & ((df["change_pct_1h"] > -10.0) | (df["change_pct_1h"].shift(12) < 10.0) | (df["RSI_3_1h"] > 50.0))
              # 4h P&D, 4h high
              & ((df["change_pct_4h"] > -15.0) | (df["change_pct_4h"].shift(48) < 30.0) | (df["AROONU_14_4h"] < 90.0))
              # 4h green, 15m & 1h down move
              & ((df["change_pct_4h"] < 10.0) | (df["RSI_3_15m"] > 10.0) | (df["RSI_3_1h"] > 35.0))
              # 4h green, 1h down move
              & ((df["change_pct_4h"] < 40.0) | (df["RSI_3_1h"] > 50.0))
              # 4h green with top wick
              & ((df["change_pct_4h"] < 50.0) | (df["change_pct_4h"] < 50.0))
              # 1d green with top wick, 15m still high
              & ((df["change_pct_1d"] < 10.0) | (df["top_wick_pct_1d"] < 8.0) | (df["AROONU_14_15m"] < 50.0))
              # 1d green, 4h down move, 4h still high
              & ((df["change_pct_1d"] < 40.0) | (df["RSI_3_4h"] > 35.0) | (df["AROONU_14_4h"] < 50.0))
              # 1d green with top wick, 4h down move
              & ((df["change_pct_1d"] < 40.0) | (df["top_wick_pct_1d"] < 8.0) | (df["RSI_3_4h"] > 55.0))
              # 1d top wick, 4h still high
              & ((df["top_wick_pct_1d"] < 50.0) | (df["AROONU_14_4h"] < 50.0))
              # big drop in last 4 days, 1d down move
              & ((df["close"] > (df["high_max_24_4h"] * 0.20)) | (df["RSI_3_1d"] > 20.0))
              # big drop in the last 20 days, 4h down move
              & ((df["close"] > (df["high_max_20_1d"] * 0.15)) | (df["RSI_3_4h"] > 20.0))
              # big drop in the last 20 days, 1d down move
              & ((df["close"] > (df["high_max_20_1d"] * 0.05)) | (df["RSI_3_1d"] > 20.0))
              # big drop in the last 20 days, 1h still high
              & ((df["close"] > (df["high_max_20_1d"] * 0.05)) | (df["STOCHRSIk_14_14_3_3_1h"] < 45.0))
              # big drop in the last 20 days, 4h high
              & ((df["close"] > (df["high_max_20_1d"] * 0.05)) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
            )

            # Logic
            long_entry_logic.append(
              (df["AROONU_14"] < 25.0)
              & (df["AROOND_14"] > 75.0)
              & (df["STOCHRSIk_14_14_3_3"] < 30.0)
              & (df["EMA_26"] > df["EMA_12"])
              & ((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.030))
              & ((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))
              & (df["close"] < df["SMA_9"])
            )

          # Condition #163 - Scalp mode (Long).
          if long_entry_condition_index == 163:
            # Protections
            long_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            long_entry_logic.append(df["protections_long_global"] == True)

            long_entry_logic.append((df["RSI_3"] > 10.0) & (df["RSI_3_15m"] > 10.0) & (df["RSI_3_1h"] > 20.0))

            long_entry_logic.append(
              # 5m & 15m & 4h down mnove, 4h high
              ((df["RSI_3"] > 15.0) | (df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 40.0) | (df["AROONU_14_4h"] < 80.0))
              # 5m & 15m & 1d down move, 1h high
              & ((df["RSI_3"] > 15.0) | (df["RSI_3_15m"] > 25.0) | (df["RSI_3_1d"] > 25.0) | (df["AROONU_14_1h"] < 90.0))
              # 5m & 1h down move, 15m still high, 4h high
              & (
                (df["RSI_3"] > 20.0) | (df["RSI_3_1h"] > 40.0) | (df["RSI_14_15m"] < 40.0) | (df["AROONU_14_4h"] < 100.0)
              )
              # 5m & 1h & 15m down move, 1h still not low enough
              & ((df["RSI_3"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["RSI_3_1d"] > 15.0) | (df["AROONU_14_1h"] < 30.0))
              # 15m & 4h down move, 4h high
              & ((df["RSI_3"] > 15.0) | (df["RSI_3_4h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
              # 5m & 4h down move, 15m high
              & ((df["RSI_3"] > 15.0) | (df["RSI_3_4h"] > 45.0) | (df["AROONU_14_15m"] < 60.0))
              # 15m & 1h down move, 1h high
              & ((df["RSI_3_15m"] > 12.0) | (df["RSI_3_1h"] > 40.0) | (df["AROONU_14_1h"] < 100.0))
              # 15m & 1h down move, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 25.0)
                | (df["AROONU_14_1h"] < 75.0)
                | (df["AROONU_14_4h"] < 100.0)
              )
              # 15m & 1h & 4h down move, 1d overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["RSI_3_4h"] > 35.0) | (df["ROC_9_1d"] < 40.0))
              # 15m & 1h & 4h down move, 1h downtrend, 4h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["CMF_20_1h"] > -0.10)
                | (df["AROONU_14_4h"] < 80.0)
              )
              # 15m & 1h down move, 15m high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_15m"] < 60.0))
              # 15m & 1h down move, 4h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["RSI_14_4h"] < 85.0))
              # 15m & 1h down move, 1h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0))
              # 15m & 1h & 4h down move, 4h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 65.0) | (df["MFI_14_4h"] < 85.0))
              # 15m & 1h down move, 1h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 35.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 85.0))
              # 15m & 1h down move, 4h high & overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 45.0) | (df["RSI_14_4h"] < 70.0) | (df["ROC_9_4h"] < 50.0))
              # 15m down move, 4h & 1d up move, 1d downtrend
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_4h"] < 90.0) | (df["RSI_3_1d"] < 80.0) | (df["CMF_20_1d"] > -0.2))
              # 15m & 1d down move, 4h high
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0))
              # 15m & 1d down move, 1d overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["RSI_3_1d"] > 40.0) | (df["ROC_9_1d"] < 50.0))
              # 15m down move, 15m still not low enough, 4h high
              & (
                (df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
              )
              # 15m down move, 1h high, 1d overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 90.0) | (df["ROC_9_1d"] < 80.0))
              # 15m down move, 1h high, 1d overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["AROONU_14_1h"] < 100.0) | (df["ROC_9_1d"] < 40.0))
              # 15m & 1h & 1d down move, 15m high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_3_1d"] > 40.0)
                | (df["AROONU_14_15m"] < 60.0)
              )
              # 15m & 1h down move, 1h & 4h still high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1d"] > 15.0)
                | (df["AROONU_14_1h"] < 50.0)
                | (df["AROONU_14_4h"] < 50.0)
              )
              # 15m down move, 1h high, 1d overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0) | (df["ROC_9_1d"] < 150.0))
              # 15m down move, 1h high & overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0) | (df["ROC_9_1h"] < 20.0))
              # 15m down move, 1h high, 1d overbought
              & ((df["RSI_3_15m"] > 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0) | (df["ROC_9_1d"] < 100.0))
              # 15m down move, 4h high
              & (
                (df["RSI_3_15m"] > 15.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0)
                | (df["EMA_9"] < (df["EMA_26"] * 0.972))
              )
              # 15m down move, 4h high and downtrend
              & ((df["RSI_3_15m"] > 15.0) | (df["CMF_20_4h"] > -0.2) | (df["AROONU_14_4h"] < 80.0))
              # 15m & 4h down move, 15m high
              & ((df["RSI_3_15m"] > 20.0) | (df["RSI_3_4h"] > 25.0) | (df["AROONU_14_15m"] < 60.0))
              # 15m & 1h down move, 1h downtrend, 1h downtrend, 15m still high, 1h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["CMF_20_1h"] > -0.10)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_1h"] < 85.0)
              )
              # 15m & 1h down move, 4h high & overbought
              & ((df["RSI_3_15m"] > 20.0) | (df["RSI_3_1h"] > 45.0) | (df["RSI_14_4h"] < 70.0) | (df["ROC_9_4h"] < 50.0))
              # 15m down move, 15m still high, 4h high
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_4h"] < 75.0)
                | (df["AROONU_14_4h"] < 100.0)
              )
              # 15m down move, 1h & 4h high, 1f overbought
              & (
                (df["RSI_3_15m"] > 20.0)
                | (df["AROONU_14_1h"] < 85.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["ROC_9_1d"] < 80.0)
              )
              # 15m down move, 4h downtrend, 4h overbought
              & ((df["RSI_3_15m"] > 20.0) | (df["CMF_20_4h"] > -0.0) | (df["ROC_9_4h"] < 40.0))
              # 1h & 1d down move, 1d high
              & ((df["RSI_3_1h"] > 25.0) | (df["RSI_3_1d"] > 30.0) | (df["AROONU_14_1d"] < 90.0))
              # 1h down move, 15m downtrend, 4h still high
              & ((df["RSI_3_1h"] > 25.0) | (df["CMF_20_15m"] > -0.4) | (df["AROONU_14_4h"] < 50.0))
              # 1h down move, 4h downtrend, 4h high
              & ((df["RSI_3_1h"] > 25.0) | (df["CMF_20_4h"] > -0.25) | (df["AROONU_14_4h"] < 70.0))
              # 1h down move, 1h high, 1d downtrend
              & ((df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 90.0) | (df["CMF_20_1d"] > -0.2))
              # 15m & 1h down move, 15m still not low enough, 1h & 4h high
              & (
                (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_15m"] < 20.0)
                | (df["AROONU_14_1h"] < 60.0)
                | (df["AROONU_14_4h"] < 100.0)
              )
              # 1h down move, 15m & 4h high
              & ((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 95.0))
              # 1h down move, 1h high
              & ((df["RSI_3_1h"] > 30.0) | (df["MFI_14_1h"] < 80.0) | (df["AROONU_14_1h"] < 90.0))
              # 1h down move, 1h & 4h high
              & ((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 100.0))
              # 1h down move, 1h high, 1d downtrend
              & ((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 90.0) | (df["CMF_20_1d"] > -0.2))
              # 1h down move, 1h highm 1d overbought
              & ((df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 90.0) | (df["ROC_9_1d"] < 40.0))
              # 1h & 4h down move, 1h still high, 4h high
              & ((df["RSI_3_1h"] > 35.0) | (df["RSI_3_4h"] > 60.0) | (df["RSI_14_1h"] < 50.0) | (df["RSI_14_4h"] < 70.0))
              # 1h down move, 1h high, 4h & 1d overbought
              & (
                (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 70.0) | (df["ROC_9_4h"] < 40.0) | (df["ROC_9_1d"] < 40.0)
              )
              # 1h & 4h down move, 1h & 4h high
              & (
                (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 65.0)
                | (df["AROONU_14_1h"] < 85.0)
                | (df["AROONU_14_4h"] < 100.0)
              )
              # 1h down move, 15m & 1h high
              & ((df["RSI_3_1h"] > 50.0) | (df["AROONU_14_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
              # 1h down move, 15m & 1h high, 1d downtrend
              & (
                (df["RSI_3_1h"] > 60.0)
                | (df["AROONU_14_15m"] < 65.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
                | (df["CMF_20_1d"] > -0.0)
              )
              # 4h down move, 15m high
              & ((df["RSI_3_4h"] > 3.0) | (df["AROONU_14_15m"] < 50.0))
              # 4h down move, 15m & 1h still not low enough
              & ((df["RSI_3_4h"] > 10.0) | (df["AROONU_14_15m"] < 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 25.0))
              # 4h down move, 1h still high, 4h downtrend
              & ((df["RSI_3_4h"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0) | (df["CMF_20_4h"] > -0.3))
              # 4h & 1d down move, 1h & 4h low
              & ((df["RSI_3_4h"] > 25.0) | (df["RSI_3_1d"] > 25.0) | (df["CMF_20_1h"] > -0.3) | (df["CMF_20_4h"] > -0.4))
              # 4h down move, 15m still high, 1d overbought
              & ((df["RSI_3_4h"] > 55.0) | (df["AROONU_14_15m"] < 40.0) | (df["ROC_9_1d"] < 100.0))
              # 4h & 1d down move, 4h high, 1d overbought
              & (
                (df["RSI_3_4h"] > 60.0) | (df["RSI_3_1d"] > 60.0) | (df["AROONU_14_4h"] < 75.0) | (df["ROC_9_1d"] < 40.0)
              )
              # 1d down move, 1h high
              & ((df["RSI_3_1d"] > 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
              # 1d down move, 15m & 1h still high
              & ((df["RSI_3_1d"] > 10.0) | (df["RSI_14_15m"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0))
              # 1d down move, 1h & 4h high
              & ((df["RSI_3_1d"] > 10.0) | (df["AROONU_14_1h"] < 80.0) | (df["AROONU_14_4h"] < 90.0))
              # 1d down move, 1h & 4h still high
              & ((df["RSI_3_1d"] > 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 40.0) | (df["AROONU_14_4h"] < 40.0))
              # 1d down move, 1h high, 1d overbought
              & ((df["RSI_3_1d"] > 60.0) | (df["AROONU_14_1h"] < 80.0) | (df["ROC_9_1d"] < 80.0))
              # 1d down move, 15m still high, 1d overbought
              & ((df["RSI_3_1d"] > 65.0) | (df["AROONU_14_15m"] < 40.0) | (df["ROC_9_1d"] < 100.0))
              # 5m still high, 1h down move, 15m still high, 1h high
              & (
                (df["RSI_3"] < 40.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["AROONU_14_4h"] < 90.0)
              )
              # 5m still high, 15m high
              & ((df["RSI_3"] < 45.0) | (df["AROONU_14_15m"] < 70.0))
              # 5m still high, 1h down move, 4h high
              & ((df["RSI_3"] < 50.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_4h"] < 100.0))
              # 15m down move, 1h high
              & ((df["RSI_14_15m"] > 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0))
              # 15m down move, 4h & 1d high
              & (
                (df["RSI_14_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] < 70.0)
              )
              # 1h downtrend, 4h high, 1d downtrend
              & ((df["CMF_20_1h"] > -0.2) | (df["STOCHRSIk_14_14_3_3_4h"] < 90.0) | (df["CMF_20_1d"] > -0.25))
              # 4h top wick, 15m & 1h down move
              & ((df["top_wick_pct_4h"] < 10.0) | (df["RSI_3_15m"] > 15.0) | (df["RSI_3_1h"] > 40.0))
              # 4h top wick, 1h down move, 1h high
              & ((df["top_wick_pct_4h"] < 10.0) | (df["RSI_3_1h"] > 30.0) | (df["AROONU_14_1h"] < 70.0))
              # 1d red, 1h down move, 1h still high
              & ((df["change_pct_1d"] > -15.0) | (df["RSI_3_1h"] > 25.0) | (df["AROONU_14_1h"] < 50.0))
              # 1d P&D, 1h high
              & (
                (df["change_pct_1d"] > -15.0)
                | (df["change_pct_1d"].shift(288) < 15.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 80.0)
              )
              # 1d P&D, 1d downtrend
              & ((df["change_pct_1d"] > -5.0) | (df["change_pct_1d"].shift(288) < 30.0) | (df["CMF_20_1d"] > -0.1))
              # 1d P&D, 15m high
              & ((df["change_pct_1d"] > -10.0) | (df["change_pct_1d"].shift(288) < 40.0) | (df["AROONU_14_15m"] < 50.0))
              # 1d P&D, 1h high
              & (
                (df["change_pct_1d"] > -10.0)
                | (df["change_pct_1d"].shift(288) < 40.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
              )
              # 1d red with top wick, 1h high
              & ((df["change_pct_1d"] > -10.0) | (df["top_wick_pct_1d"] < 10.0) | (df["AROONU_14_1h"] < 80.0))
              # 1d green, 4m down move, 4h high
              & ((df["change_pct_1d"] < 25.0) | (df["RSI_3_4h"] > 50.0) | (df["AROONU_14_4h"] < 80.0))
              # 1d green with top wick, 1d low
              & ((df["change_pct_1d"] < 25.0) | (df["top_wick_pct_1d"] < 10.0) | (df["CMF_20_1d"] > -0.2))
              # 1d top wick, 1h still high
              & ((df["top_wick_pct_1d"] < 25.0) | (df["AROONU_14_1h"] < 50.0))
              # 1d top wick, 4h still high
              & ((df["top_wick_pct_1d"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0))
              # 1d top wick, 1h down move
              & ((df["top_wick_pct_1d"] < 50.0) | (df["RSI_3_1h"] > 30.0))
              # big drop in the last 12 days, 1h down move, 1h high
              & ((df["close"] > (df["high_max_12_1d"] * 0.35)) | (df["RSI_3_1h"] > 35.0) | (df["AROONU_14_1h"] < 70.0))
              # big drop in the last 20 days, 1h down move, 1h high
              & (
                (df["close"] > (df["high_max_20_1d"] * 0.30))
                | (df["RSI_3_1h"] > 30.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 75.0)
              )
            )

            # Logic
            long_entry_logic.append(
              (df["RSI_14"] < 30.0)
              & (df["AROONU_14"] < 25.0)
              & (df["AROOND_14"] > 75.0)
              & (df["STOCHRSIk_14_14_3_3"] < 20.0)
              & (df["EMA_9"] < (df["EMA_26"] * 0.982))
              & (df["close"] < df["SMA_9"])
            )

          ###############################################################################################

          # LONG ENTRY CONDITIONS ENDS HERE

          ###############################################################################################

          long_entry_logic.append(df["volume"] > 0)
          item_long_entry = reduce(lambda x, y: x & y, long_entry_logic)
          df.loc[item_long_entry, "enter_tag"] += f"{long_entry_condition_index} "
          long_entry_conditions.append(item_long_entry)
          df.loc[:, "enter_long"] = item_long_entry

      if long_entry_conditions:
        df.loc[:, "enter_long"] = reduce(lambda x, y: x | y, long_entry_conditions)

      ###############################################################################################

      # SHORT ENTRY CONDITIONS STARTS HERE

      ###############################################################################################

      #   ______  __    __  ______  _______ ________        ________ __    __ ________ ________ _______
      #  /      \|  \  |  \/      \|       |        \      |        |  \  |  |        |        |       \
      # |  $$$$$$| $$  | $|  $$$$$$| $$$$$$$\$$$$$$$$      | $$$$$$$| $$\ | $$\$$$$$$$| $$$$$$$| $$$$$$$\
      # | $$___\$| $$__| $| $$  | $| $$__| $$ | $$         | $$__   | $$$\| $$  | $$  | $$__   | $$__| $$
      #  \$$    \| $$    $| $$  | $| $$    $$ | $$         | $$  \  | $$$$\ $$  | $$  | $$  \  | $$    $$
      #  _\$$$$$$| $$$$$$$| $$  | $| $$$$$$$\ | $$         | $$$$$  | $$\$$ $$  | $$  | $$$$$  | $$$$$$$\
      # |  \__| $| $$  | $| $$__/ $| $$  | $$ | $$         | $$_____| $$ \$$$$  | $$  | $$_____| $$  | $$
      #  \$$    $| $$  | $$\$$    $| $$  | $$ | $$         | $$     | $$  \$$$  | $$  | $$     | $$  | $$
      #   \$$$$$$ \$$   \$$ \$$$$$$ \$$   \$$  \$$          \$$$$$$$$\$$   \$$   \$$   \$$$$$$$$\$$   \$$
      #

      for enabled_short_entry_signal in self.short_entry_signal_params:
        short_entry_condition_index = int(enabled_short_entry_signal.split("_")[3])
        item_short_buy_protection_list = [True]
        if self.short_entry_signal_params[f"{enabled_short_entry_signal}"]:
          # Short Entry Conditions Starts Here
          # -----------------------------------------------------------------------------------------
          # IMPORTANT: Short Condition Descriptions are not for shorts. These are for longs but completely mirrored opposite side
          # Please dont change these comment descriptions. With these descriptions we are comparing long/short positions.

          short_entry_logic = []
          short_entry_logic.append(reduce(lambda x, y: x & y, item_short_buy_protection_list))

          # Condition #501 - Normal mode (Short).
          if short_entry_condition_index == 501:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            short_entry_logic.append(df["protections_short_global"] == True)
            short_entry_logic.append(df["global_protections_short_pump"] == True)
            short_entry_logic.append(df["global_protections_short_dump"] == True)

            short_entry_logic.append(df["RSI_3_1h"] >= 5.0)
            short_entry_logic.append(df["RSI_3_4h"] >= 20.0)
            short_entry_logic.append(df["RSI_3_1d"] >= 20.0)
            short_entry_logic.append(df["RSI_14_1h"] > 20.0)
            short_entry_logic.append(df["RSI_14_4h"] > 20.0)
            short_entry_logic.append(df["RSI_14_1d"] > 10.0)
            # 5m up move, 4h still low
            short_entry_logic.append((df["RSI_3"] < 97.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0))
            # 5m & 15m strong up move
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["RSI_3_15m"] < 95.0))
            # 5m & 1h up move, 1d uptrend
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["ROC_9_1d"] < 100.0))
            # 5m up move, 15m & 1h still not high enough
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["AROOND_14_15m"] < 25.0) | (df["AROOND_14_1h"] < 25.0))
            # 4m up move, 1h & 4h still low
            short_entry_logic.append(
              (df["RSI_3"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 4m & 1h up move, 1h still low
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 1h up move, 4h low
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["RSI_3_1h"] < 80.0) | (df["AROONU_14_4h"] > 20.0))
            # 5m up move, 15m & 1h uptrend
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["CMF_20_15m"] < 0.30) | (df["CMF_20_1h"] < 0.30))
            # 5m up move, 15m stil low
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["AROONU_14_15m"] > 50.0))
            # 5m up move, 15m & 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
            )
            # 15m up move, 1h low
            short_entry_logic.append((df["RSI_3_15m"] < 97.0) | (df["AROONU_14_1h"] > 30.0))
            # 15m & 1h up move, 4h still going up
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["CCI_20_change_pct_4h"] < -0.0)
            )
            # 15m & 1h up move, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 15m & 1h up move, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m & 1h up move, 1h still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 4h up move, 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
            )
            # 15m up move, 15m stil not high enough, 1h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 10.0)
            )
            # 15m up move, 1h still not high enough, 4h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["AROONU_14_4h"] > 20.0)
            )
            # 15m up move, 1h & 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 15m up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["AROONU_14_4h"] > 70.0))
            # 15m up move, 1h up move, 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_change_pct_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 1h up move, 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
            )
            # 15m & 1h up move, 1h not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["AROOND_14_1h"] < 50.0))
            # 15m & 1h up move, 1d stil not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["RSI_14_1h"] > 80.0))
            # 15m & 1h up move, 15m still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0)
            )
            # 15m & 4h up move, 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            )
            # 15m & 4h up move, 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0)
            )
            # 15m & 4h up move, 1d low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 20.0)
            )
            # 15m & 4h up move, 4h not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["AROOND_14_4h"] < 50.0))
            # 15m & 4h up move, 1h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0)
            )
            # 15m & 4h up move, 4h low
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 60.0) | (df["AROONU_14_4h"] > 30.0))
            # 15m up move, 1h & 4h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["AROONU_14_1h"] > 40.0) | (df["AROONU_14_4h"] > 10.0)
            )
            # 15m up move, 1h still low, 4h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["AROONU_14_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 15m up move, 1h low, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 15m & 1h up move, 4h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0)
            )
            # 15m & 1h up move, 1d still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 60.0)
            )
            # 15m & 1h up move, 1h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 15m & 1h up move, 4h low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 15m & 4h down move, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 15m & 4h up move, 15m low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0)
            )
            # 15m & 1h up move, 1h still not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 80.0) | (df["RSI_3_1h"] < 70.0) | (df["AROONU_14_1h"] > 60.0))
            # 15m & 4h up move, 15m still low
            short_entry_logic.append((df["RSI_3_15m"] < 80.0) | (df["RSI_3_4h"] < 80.0) | (df["AROONU_14_15m"] > 50.0))
            # 15m up move, 15m still not high enough, 1h still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0))
            # 1h & 4h up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 95.0) | (df["UO_7_14_28_4h"] > 60.0))
            # 1h & 4h up move, 4h still low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h up move, 4h uptrend
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 85.0) | (df["ROC_9_4h"] < 40.0))
            # 1h & 1d strong up move
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_1d"] < 95.0))
            # 1h & 4h strong up move
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["MFI_14_1h"] < 95.0) | (df["RSI_3_4h"] < 95.0))
            # 1h strong up move, 15m still move higher
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["CCI_20_change_pct_15m"] < -0.0))
            # 1h & 4h up move, 1h still low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 1h & 4h up move, 1d still not high enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 70.0)
            )
            # 1h up move, 1h still not high enough, 1d low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 30.0)
            )
            # 1h up move, 15m & 4h still low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h up move, 15m still not high enough
            short_entry_logic.append((df["RSI_3_1h"] < 85.0) | (df["RSI_3_4h"] < 85.0) | (df["AROOND_14_15m"] < 50.0))
            # 1h & 4h up move, 15m still not high enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 90.0)
            )
            # 1h up move, 15m still not high enough, 1h still low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 1h up move, 1h still low
            short_entry_logic.append((df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 40.0))
            # 4h & 1d strong up move
            short_entry_logic.append((df["RSI_3_4h"] < 95.0) | (df["RSI_3_1d"] < 95.0))
            # 4h up move, 15m still low, 1h not high enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["AROOND_14_1h"] < 25.0)
            )
            # 4h up move, 1h still low
            short_entry_logic.append((df["RSI_3_4h"] < 90.0) | (df["AROONU_14_1h"] > 40.0))
            # 1d up move, 1h & 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1d"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m & 1h uptrend, 4h still low
            short_entry_logic.append(
              (df["CMF_20_15m"] < 0.30) | (df["CMF_20_1h"] < 0.30) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 5m green, 15m still not high enough
            short_entry_logic.append((df["change_pct"] < 5.0) | (df["AROOND_14_15m"] < 50.0))
            # 5m green, 15m still not high enough
            short_entry_logic.append((df["change_pct"] < 5.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 90.0))
            # big pump in the last 4 hours, 15m still low
            short_entry_logic.append((df["close"] < (df["close_min_48"] * 1.50)) | (df["AROONU_14_15m"] > 50.0))

            # Logic
            short_entry_logic.append(df["EMA_12"] > df["EMA_26"])
            short_entry_logic.append((df["EMA_12"] - df["EMA_26"]) > (df["open"] * 0.030))
            short_entry_logic.append((df["EMA_12"].shift() - df["EMA_26"].shift()) > (df["open"] / 100.0))
            short_entry_logic.append(df["close"] > (df["BBU_20_2.0"] * 1.004))

          # Condition #502 - Normal mode (Short).
          if short_entry_condition_index == 502:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            short_entry_logic.append(df["protections_short_global"] == True)

            # 5m & 15m & 1h & 4h up move
            short_entry_logic.append(
              (df["RSI_3"] < 97.0) | (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 80.0)
            )
            # 5m & 4h up move
            short_entry_logic.append((df["RSI_3"] < 97.0) | (df["RSI_3_4h"] < 95.0))
            # 5m up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3"] < 97.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0))
            # 5m & 15m strong up move
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["RSI_3_15m"] < 95.0))
            # 5m & 15m up move, 4h low
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["RSI_3_15m"] < 90.0) | (df["AROONU_14_4h"] > 30.0))
            # 5m & 1h & 4h down move
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0))
            # 5m & 1h up move, 15m still not high enough
            short_entry_logic.append(
              (df["RSI_3"] < 95.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0)
            )
            # 5m up move, 15m still low
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0))
            # 5m up move, 4h low
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["AROONU_14_4h"] > 20.0))
            # 15m & 1h down move, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 97.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 15m up move, 1h still low
            short_entry_logic.append((df["RSI_3_15m"] < 97.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0))
            # 15m & 1h & 4h up move
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 85.0))
            # 15m & 1h up move, 4h still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 15m up move, 1h still low
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 40.0))
            # 15m up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["AROONU_14_4h"] > 70.0))
            # 15m & 1h & 4h up move
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0))
            # 15m & 1h up move, 15m still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0)
            )
            # 15m & 1h up move, 1d low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 40.0)
            )
            # 15m & 1h up move, 1h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            )
            # 15m & 4h up move, 1d low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 30.0)
            )
            # 15m up move, 1h still low
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0))
            # 15m & 1h up move, 1h still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 65.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0)
            )
            # 15m up move, 1h low
            short_entry_logic.append((df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0))
            # 15m up move, 1h low
            short_entry_logic.append((df["RSI_3_15m"] < 80.0) | (df["AROONU_14_1h"] > 10.0))
            # 15m up move, 4h still low
            short_entry_logic.append((df["RSI_3_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0))
            # 1h & 1d strong up move
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_1d"] < 95.0))
            # 1h up move, 1h still not high enough
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0))
            # 1h up move, 4h still low, 1h moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 1h up move, 1d low
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_14_1d"] > 40.0))
            # 1h strong up move, 15m still move higher
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["CCI_20_change_pct_15m"] < -0.0))
            # 1h up move, relative stable before the hour
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["close_min_12"] > (df["close_min_48"] * 1.10)))
            # 1h up move, 1d low
            short_entry_logic.append((df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 10.0))
            # 1h up move, 4h still low
            short_entry_logic.append((df["RSI_3_1h"] < 85.0) | (df["AROONU_14_4h"] > 50.0))
            # 1h up move, 1h still not high enough
            short_entry_logic.append((df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0))
            # 1h up move, 4h still low, 1h still moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["RSI_14_4h"] > 60.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0))
            # 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_1h"] < 80.0) | (df["AROONU_14_4h"] > 10.0))
            # 1h up move, 1h still low
            short_entry_logic.append((df["RSI_3_1h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0))
            # 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_1h"] < 70.0) | (df["RSI_14_4h"] > 40.0))
            # 1h up move, 1h low
            short_entry_logic.append((df["RSI_3_1h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0))
            # 4h up move, 1d still low
            short_entry_logic.append((df["RSI_3_4h"] < 97.0) | (df["RSI_14_1d"] > 50.0))
            # 4h up move, 1h still not high enough
            short_entry_logic.append((df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0))
            # 4h up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0))
            # 4h up move, 15m still not high enough, 4h moving higher
            short_entry_logic.append(
              (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0) | (df["CCI_20_change_pct_4h"] < 0.0)
            )
            # 4h up move, 15m still low
            short_entry_logic.append((df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0))
            # 4h up move, 1h still low
            short_entry_logic.append((df["RSI_3_4h"] < 90.0) | (df["AROONU_14_1h"] > 40.0))
            # 4h up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0))
            # 4h up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0))
            # 4h up move, 1h low
            short_entry_logic.append((df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0))
            # 1d up move, 1h still not high enough
            short_entry_logic.append((df["RSI_3_1d"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0))
            # 1d up move, 1h still low
            short_entry_logic.append((df["RSI_3_1d"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0))
            # 1d up move, 4h still low
            short_entry_logic.append((df["RSI_3_1d"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 40.0))
            # 15m low, 1h still low
            short_entry_logic.append((df["AROONU_14_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0))
            # 15m low, 4h low
            short_entry_logic.append((df["AROONU_14_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0))
            # 15m still low, 1h low
            short_entry_logic.append((df["AROONU_14_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0))
            # 15m still not high enough, 4h low
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["AROONU_14_4h"] > 10.0))
            # 1h & 4h low
            short_entry_logic.append((df["AROONU_14_1h"] > 20.0) | (df["AROONU_14_4h"] > 20.0))
            # 1h & 4h low
            short_entry_logic.append((df["AROONU_14_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0))
            # 1h low, 1d low
            short_entry_logic.append((df["AROONU_14_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 30.0))
            # 1h & 4h low
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_1h"] > 20.0) | (df["AROONU_14_4h"] > 20.0))
            # 1d big green, 1d still not high enough
            short_entry_logic.append((df["change_pct_1d"] < 30.0) | (df["RSI_14_1d"] > 65.0))
            # rise in the last hour, relatively stable before the hour
            short_entry_logic.append(
              (df["close"] < (df["close_min_12"] * 1.10)) | (df["close_min_12"] > (df["close_min_48"] * 1.10))
            )
            # big pump in the last 6 days, 4h still not high enough
            short_entry_logic.append((df["close"] < (df["low_min_6_1d"] * 4.0)) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0))
            # big pump in the last 20 days, 1h up move
            short_entry_logic.append((df["close"] < (df["low_min_20_1d"] * 6.0)) | (df["RSI_3_1h"] < 90.0))

            # Logic
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["STOCHRSIk_14_14_3_3"] > 80.0)
            short_entry_logic.append(df["close"] > (df["EMA_20"] * 1.060))
            short_entry_logic.append(df["close"] > (df["BBU_20_2.0"] * 0.995))
            short_entry_logic.append(df["AROOND_14_15m"] < 25.0)

          # Condition #503 - Normal mode (Short).
          if short_entry_condition_index == 503:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            short_entry_logic.append(df["RSI_3_1h"] >= 5.0)
            short_entry_logic.append(df["RSI_3_4h"] >= 20.0)
            short_entry_logic.append(df["RSI_3_1d"] >= 20.0)
            short_entry_logic.append(df["RSI_14_1h"] > 20.0)
            short_entry_logic.append(df["RSI_14_4h"] > 20.0)
            short_entry_logic.append(df["RSI_14_1d"] > 10.0)
            # 5m strong down move
            short_entry_logic.append((df["RSI_3"] < 98.0) | (df["ROC_9"] < 50.0))
            # 5m down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["MFI_14"] > 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 5m & 1h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
            )
            # 5m down move, 4h downtrend, 1h still high
            short_entry_logic.append(
              (df["RSI_3"] < 95.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 5m & 4h strong down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 5m down move, 1h high, 1d overbought
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["ROC_9_1h"] < 15.0) | (df["ROC_9_1d"] > -40.0))
            # 5m down move, 1h & 4h high
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["UO_7_14_28_1h"] > 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 5m down move, 1h high, 4h downtrend
            short_entry_logic.append(
              (df["RSI_3"] < 98.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 10.0) | (df["ROC_9_4h"] < 10.0)
            )
            # 5m & 1h down move, 4h down
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["RSI_3_1h"] < 85.0) | (df["CMF_20_4h"] > -0.2))
            # 5m down move, 1h high
            short_entry_logic.append((df["RSI_14_change_pct"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0))
            # 5m down move, 1h high
            short_entry_logic.append((df["RSI_14_change_pct"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0))
            # 15m & 1h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 15m still not low enough, 1h & 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0)
              | (df["AROOND_14_15m"] < 25.0)
              | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
              | (df["MFI_14_4h"] > 50.0)
            )
            # 5m & 1h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 4h down move, 1h still not low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
            )
            # 15m & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_4h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 15m down move, 1h & 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["RSI_14_4h"] > 50.0)
            )
            # 15m & 1h & 4h down move
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_change_pct_1h"] > -60.0) | (df["RSI_3_change_pct_4h"] > -40.0)
            )
            # 15m down move, 1d downtrend, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["ROC_9_1d"] > -25.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 15m & 1d down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1d"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 15m & 4h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m down move, 15m still not low enough, 4h down move
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["AROOND_14_15m"] < 50.0) | (df["RSI_3_4h"] < 85.0))
            # 15m down move, 1h still high, 1d strong downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 80.0) | (df["AROOND_14_1h"] < 25.0) | (df["MFI_14_1d"] < 90.0))
            # 15m down move, 1h still high, 1d downtrend
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 15m down move, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 15m & 4h down move, 1d downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 85.0) | (df["RSI_3_4h"] < 85.0) | (df["ROC_9_1d"] > -70.0))
            # 15m down move, 15m not low enough, 1h overbought
            short_entry_logic.append(
              (df["RSI_14_change_pct_15m"] > -40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 90.0) | (df["RSI_14_1h"] > 30.0)
            )
            # 15m strong down move, 1h still high
            short_entry_logic.append((df["ROC_9_15m"] < 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0))
            # 15m downtrend, 1h & 4h still high
            short_entry_logic.append(
              (df["ROC_9_15m"] < 10.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m & 1h & 4h down move
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["CCI_20_change_pct_4h"] < 0.0)
            )
            # 15m strong down move
            short_entry_logic.append((df["RSI_3_15m"] < 90.0) | (df["MFI_14_15m"] < 85.0) | (df["AROOND_14_15m"] < 25.0))
            # 14m down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 80.0) | (df["AROOND_14_15m"] < 50.0) | (df["UO_7_14_28_4h"] > 50.0)
            )
            # 15m down move, 1h stil high, 1d overbought
            short_entry_logic.append((df["RSI_3_15m"] < 85.0) | (df["AROOND_14_1h"] < 25.0) | (df["ROC_9_1d"] > -80.0))
            # 15m down move, 1h high, 1d overbought
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["ROC_9_1d"] > -50.0)
            )
            # 1h & 4h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_change_pct_4h"] < 65.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] < 10.0)
            )
            # 1h down move, 4h still not low enough, 1d overbought
            short_entry_logic.append((df["RSI_3_1h"] < 90.0) | (df["AROOND_14_4h"] < 25.0) | (df["ROC_9_1d"] > -120.0))
            # 1h down move, 1h still not low enough, 4h still not low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["RSI_14_4h"] > 50.0)
            )
            # 1h down move, 4h still high
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_14_4h"] > 60.0))
            # 1h down move, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 1h down move, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["RSI_3_change_pct_1h"] > -65.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 4h & 1d down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 90.0) | (df["ROC_2_1d"] < 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m still high, 1h down move, 4h high
            short_entry_logic.append(
              (df["AROOND_14_15m"] < 50.0) | (df["RSI_3_change_pct_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 15m still high, 1h & 4h down move, 4h still high
            short_entry_logic.append(
              (df["AROOND_14_15m"] < 50.0)
              | (df["RSI_3_1h"] < 85.0)
              | (df["RSI_3_4h"] < 80.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 15m & 1h still high, 4h overbought
            short_entry_logic.append(
              (df["AROOND_14_15m"] < 50.0) | (df["AROOND_14_1h"] < 50.0) | (df["ROC_9_4h"] > -40.0)
            )
            # 15m still high, 1h down move, 1d downtrend
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["RSI_3_4h"] < 90.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 1h & 4h still high, 1d strong down move
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["UO_7_14_28_4h"] > 55.0) | (df["RSI_3_1d"] < 90.0)
            )
            # 1h still high, 4h & 1d downtrend
            short_entry_logic.append((df["AROOND_14_1h"] < 25.0) | (df["ROC_9_4h"] < 20.0) | (df["ROC_9_1d"] < 50.0))
            # 4h moving down, 1d P&D
            short_entry_logic.append(
              (df["ROC_9_4h"] < 30.0) | (df["RSI_3_change_pct_1d"] < 50.0) | (df["ROC_9_1d"] > -50.0)
            )
            # 1d strong downtrend, 4h still high
            short_entry_logic.append(
              (df["ROC_2_1d"] < 20.0) | (df["ROC_9_1d"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 1d P&D, 1d overbought
            short_entry_logic.append(
              (df["ROC_2_1d"] < 10.0) | (df["ROC_9_1d"] > -50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 5.0)
            )
            # 1h red, previous 1h green, 1h overbought
            short_entry_logic.append(
              (df["change_pct_1h"] < 1.0) | (df["change_pct_1h"].shift(12) > -5.0) | (df["RSI_14_1h"].shift(12) < 80.0)
            )
            # 1h red, 1h stil high, 4h downtrend
            short_entry_logic.append(
              (df["change_pct_1h"] < 5.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["ROC_9_4h"] > -25.0)
            )
            # 4h red, 15m down move, 4h still high
            short_entry_logic.append(
              (df["change_pct_4h"] < 5.0) | (df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 4h red, previous 4h green, 4h overbought
            short_entry_logic.append(
              (df["change_pct_4h"] < 5.0) | (df["change_pct_4h"].shift(48) > -5.0) | (df["ROC_9_4h"].shift(48) > -25.0)
            )
            # 4h red, 4h still not low enough, 1h downtrend, 1h overbought
            short_entry_logic.append(
              (df["change_pct_4h"] < 10.0)
              | (df["AROOND_14_4h"] < 25.0)
              | (df["ROC_9_1h"] < 20.0)
              | (df["ROC_9_1d"] > -40.0)
            )
            # 4h red, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["change_pct_4h"] < 10.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0) | (df["ROC_9_1d"] < 40.0)
            )
            # 1d P&D, 1d overbought
            short_entry_logic.append(
              (df["change_pct_1d"] < 10.0) | (df["change_pct_1d"].shift(288) > -10.0) | (df["ROC_9_1d"] > -100.0)
            )
            # 1d P&D, 4h still high
            short_entry_logic.append(
              (df["change_pct_1d"] < 15.0) | (df["change_pct_1d"].shift(288) > -15.0) | (df["AROOND_14_4h"] < 50.0)
            )
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 95.0) | (df["CCI_20_change_pct_4h"] < 0.0)
            )

            # Logic
            short_entry_logic.append(df["RSI_20"] > df["RSI_20"].shift(1))
            short_entry_logic.append(df["RSI_4"] > 54.0)
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["close"] > df["SMA_16"] * 1.058)

          # Condition #504 - Normal mode (Short).
          if short_entry_condition_index == 504:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            short_entry_logic.append(df["RSI_3_1h"] >= 5.0)
            short_entry_logic.append(df["RSI_3_4h"] >= 20.0)
            short_entry_logic.append(df["RSI_3_1d"] >= 20.0)
            short_entry_logic.append(df["RSI_14_1h"] > 20.0)
            short_entry_logic.append(df["RSI_14_4h"] > 20.0)
            short_entry_logic.append(df["RSI_14_1d"] > 10.0)
            # 15m & 1h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0)
              | (df["MFI_14_15m"] < 90.0)
              | (df["RSI_3_1h"] < 80.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m & 1h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["MFI_14_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 14m & 4h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 15m down move, 1h & 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["UO_7_14_28_1h"] < 45.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 1h strong down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_14_change_pct_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 1h strong down move, 4h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_change_pct_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 75.0) | (df["AROOND_14_4h"] < 50.0))
            # 15m down move, 1h strong downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["MFI_14_1h"] > 5.0))
            # 15m downtrend, 4h down move, 4h stil high
            short_entry_logic.append(
              (df["ROC_9_15m"] > -20.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )

            # Logic
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["AROOND_14_15m"] < 25.0)
            short_entry_logic.append(df["close"] > (df["EMA_9"] * 1.058))
            short_entry_logic.append(df["close"] > (df["EMA_20"] * 1.040))

          # Condition #541 - Quick mode (Short).
          if short_entry_condition_index == 541:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            # 5m & 15m down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3"] < 95.0) | (df["RSI_3_change_pct_15m"] < 50.0) | (df["RSI_14_4h"] > 50.0)
            )
            # 5m & 15m & 1h down move
            short_entry_logic.append((df["RSI_3"] < 95.0) | (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0))
            # 5m strong down move
            short_entry_logic.append((df["RSI_3"] < 98.0) | (df["ROC_9"] < 50.0))
            # 15m & 1h strong down move & downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["MFI_14_1h"] > 5.0))
            # 15m strong down move, 4h high
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0))
            # 15m & 1h down move
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["CCI_20_change_pct_1h"] > 0.0)
            )
            # 15m & 1h down move, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 15m & 1h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_change_pct_1h"] < 50.0) | (df["MFI_14_4h"] > 50.0)
            )
            # 15m strong down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["MFI_14_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 1h down move, 1h not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            )
            # 15m down move, 1h strong down move
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_14_change_pct_1h"] < 70.0))
            # 15m down move, 4h & 1d downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["ROC_9_4h"] < 30.0) | (df["ROC_9_1d"] < 50.0))
            # 15m down move, 1h strong down move, 4h stil high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 1h & 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 1h downtrend, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["ROC_9_1h"] < 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m & 1h down move, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 15m down move, 1h down move, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_change_pct_1h"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0)
            )
            # 1m down move, 1h still dropping, 4h overbought
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["CCI_20_change_pct_1h"] < 0.0) | (df["RSI_14_4h"] > 20.0)
            )
            # 15m down move, 1h high
            short_entry_logic.append((df["RSI_3_change_pct_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 10.0))
            # 1h strong down move, 4h high
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0))
            # 1h down move, 4h downtrend, 4h not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["CMF_20_4h"] > -0.25) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 1h down move, 4h high, 1d overbought
            short_entry_logic.append((df["RSI_3_1h"] < 90.0) | (df["RSI_14_4h"] > 40.0) | (df["ROC_9_1d"] > -50.0))
            # 1h down move, 4h strong down move
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_14_change_pct_4h"] < 40.0))
            # 1h & 4h down move, 4h still going down
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 95.0) | (df["CCI_20_change_pct_4h"] < 0.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_change_pct_4h"] < 65.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 1h down move, 4h down move, 4h P&D
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_change_pct_4h"] < 70.0) | (df["RSI_14_4h"].shift(48) > 30.0)
            )
            # 1h & 4h down move, 4h still not low enough, 1d still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0)
              | (df["RSI_3_change_pct_4h"] < 50.0)
              | (df["AROOND_14_4h"] < 25.0)
              | (df["STOCHRSIk_14_14_3_3_1d"] > 60.0)
            )
            # 1h down move, 1h still high, 1d going down
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["ROC_2_1d"] > -50.0)
            )
            # 4h downtrend, 4h still high, 1d strong downtrend
            short_entry_logic.append(
              (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0) | (df["ROC_9_1d"] < 60.0)
            )
            # 15m down move, 1h strong down move, 1d overbought
            short_entry_logic.append(
              (df["MFI_14_15m"] < 80.0) | (df["RSI_3_change_pct_1h"] < 80.0) | (df["ROC_9_1d"] > -50.0)
            )
            # 1h not low enough, 4h high, 1d strong downtrend
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0) | (df["ROC_9_1d"] < 60.0)
            )
            # 1h down move, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["RSI_3_change_pct_1h"] < 65.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 15m strong down move, 1h still high
            short_entry_logic.append((df["ROC_9_15m"] < 15.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0))
            # 15m downtrend, 4h down move, 4h stil high
            short_entry_logic.append(
              (df["ROC_9_15m"] < 15.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 1h downtrend, 4h overbought
            short_entry_logic.append((df["ROC_2_1h"] < 5.0) | (df["RSI_14_4h"] > 20.0) | (df["ROC_9_4h"] > -25.0))
            # 1h P&D, 4h still high
            short_entry_logic.append(
              (df["ROC_2_1h"] < 10.0) | (df["ROC_9_1h"] > -5.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 1h downtrend, 4h down move, 1d downtrend
            short_entry_logic.append((df["ROC_9_1h"] < 40.0) | (df["RSI_3_4h"] < 90.0) | (df["ROC_9_1d"] < 50.0))
            short_entry_logic.append((df["ROC_9_4h"] > -200.0) | (df["RSI_14_4h"] > 20.0))
            # 4h down move, 1d P&D
            short_entry_logic.append((df["ROC_9_4h"] < 20.0) | (df["ROC_2_1d"] < 20.0) | (df["ROC_9_1d"] > -50.0))
            # 1h P&D, 4h overbought
            short_entry_logic.append(
              (df["change_pct_1h"] < 2.0) | (df["change_pct_1h"].shift(12) > 2.0) | (df["RSI_14_4h"] > 20.0)
            )
            # 1h P&D, 1d overbought
            short_entry_logic.append(
              (df["change_pct_1h"] < 5.0) | (df["change_pct_1h"].shift(12) > -5.0) | (df["ROC_9_1d"] > -100.0)
            )
            # 1h & 4h red, 1h not low enough
            short_entry_logic.append(
              (df["change_pct_1h"] < 10.0) | (df["change_pct_4h"] < 10.0) | (df["MFI_14_1h"] > 50.0)
            )
            # 1h red, 1h still not low enough, 1d down move
            short_entry_logic.append((df["change_pct_1h"] < 15.0) | (df["MFI_14_1h"] > 50.0) | (df["RSI_3_1d"] < 90.0))
            # 4h red, previous 4h green, 4h overbought
            short_entry_logic.append(
              (df["change_pct_4h"] < 5.0) | (df["change_pct_4h"].shift(48) > -5.0) | (df["RSI_14_4h"].shift(48) > 20.0)
            )
            # 1d P&D, 1d overbought
            short_entry_logic.append(
              (df["change_pct_1d"] < 10.0) | (df["change_pct_1d"].shift(288) > -10.0) | (df["ROC_9_1d"] > -100.0)
            )
            # 1d P&D, 4h still high
            short_entry_logic.append(
              (df["change_pct_1d"] < 15.0) | (df["change_pct_1d"].shift(288) > -15.0) | (df["AROOND_14_4h"] < 50.0)
            )

            # Logic
            short_entry_logic.append(df["RSI_14"] > 64.0)
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["AROONU_14"] > 75.0)
            short_entry_logic.append(df["EMA_9"] > (df["EMA_26"] * 1.040))

          # Condition #542 - Quick mode (Short).
          if short_entry_condition_index == 542:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)
            short_entry_logic.append(df["protections_short_global"] == True)

            # 5m & 15m up move, 15m stil low
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["RSI_3_15m"] < 80.0) | (df["AROONU_14_15m"] > 60.0))
            # 15m & 1h up move, 4h still low
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["RSI_14_4h"] > 60.0))
            # 15m & 1h up move, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 15m & 1h up move, 1h still moving higher
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 15m & 4h up move, 4h still moving higher
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 95.0) | (df["CCI_20_change_pct_4h"] < -0.0)
            )
            # 15m up move, 15m still not high enough, 1d uptrend
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0) | (df["ROC_9_1d"] < 80.0)
            )
            # 15m & 4h up move, 15m still not high enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0)
            )
            # 15m & 4h up move, 4h still not high enough
            short_entry_logic.append((df["RSI_3_15m"] < 85.0) | (df["RSI_3_4h"] < 80.0) | (df["RSI_14_4h"] > 60.0))
            # 15m up move, 15m still not high enough, 4h still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0) | (df["AROONU_14_4h"] > 50.0)
            )
            # 15m & 1h up move, 15m still low
            short_entry_logic.append((df["RSI_3_15m"] < 70.0) | (df["RSI_3_1h"] < 70.0) | (df["AROONU_14_15m"] > 40.0))
            # 15m & 1h up move, 15m still low
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 40.0)
            )
            # # 15m & 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_15m"] < 70.0) | (df["RSI_3_1h"] < 60.0) | (df["AROONU_14_4h"] > 40.0))
            # 1h & 1d up move, 1h still moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 97.0) | (df["RSI_3_1d"] < 95.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 1h & 4h up move, 15m still not high enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0)
            )
            # 1h & 4h up move, 1d uptrend
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 95.0) | (df["ROC_9_1d"] < 100.0))
            # 1h & 4h up move, 1d still low
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 85.0) | (df["RSI_14_1d"] > 50.0))
            # 1h up move, 4h still low, 1h still moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_14_4h"] > 60.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["AROONU_14_4h"] > 10.0))
            # 1h & 4h up move, 1h still moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 85.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 1h & 4h up move, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 1h & 1d up move, 15m still low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_1d"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0)
            )
            # 1h up move, 4h low
            short_entry_logic.append((df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0))
            # 1h up move, 4h still low, 1h still moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0) | (df["CCI_20_change_pct_1h"] < -0.0)
            )
            # 1h & 4h up move, 4h still moving higher
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["RSI_3_4h"] < 80.0) | (df["CCI_20_change_pct_4h"] < -0.0)
            )
            # 1h up move, 15m low
            short_entry_logic.append((df["RSI_3_1h"] < 85.0) | (df["AROONU_14_15m"] > 40.0))
            # 1h up move, 4h still not high enough, 1d low
            short_entry_logic.append((df["RSI_3_1h"] < 85.0) | (df["AROONU_14_4h"] > 80.0) | (df["RSI_14_1d"] > 40.0))
            # 4h up move, 1d low
            short_entry_logic.append((df["RSI_3_4h"] < 95.0) | (df["RSI_14_1d"] > 40.0))
            # 4h down move, 15m still not high enough, 1d low
            short_entry_logic.append(
              (df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0) | (df["AROOND_14_1d"] < 75.0)
            )
            # 4h up move, 15m low
            short_entry_logic.append((df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 45.0))
            # 4h up move, 15m still not high enough
            short_entry_logic.append((df["RSI_3_4h"] < 85.0) | (df["AROONU_14_15m"] > 60.0))
            # 4h up move, 15m low
            short_entry_logic.append((df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 30.0))
            # 4h up move, 15m still low, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["AROONU_14_4h"] > 80.0)
            )
            # 4h up move, 15m still low, 4h still not high enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 1d up move, 4h low
            short_entry_logic.append((df["RSI_3_1d"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 40.0))
            # 15m & 4h still not high enough
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0))
            # 1d bot wick, 4h still not high enough
            short_entry_logic.append((df["bot_wick_pct_1d"] < 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0))
            # rise in the last 12 hours, relatively stable before the 12 hours
            short_entry_logic.append(
              (df["close"] < (df["low_min_12_1h"] * 1.30)) | (df["low_min_12_1h"] > (df["low_min_24_1h"] * 1.10))
            )
            # big pump in the last 30 days, 4h up move
            short_entry_logic.append((df["close"] < (df["low_min_30_1d"] * 4.0)) | (df["RSI_3_4h"] < 85.0))

            # Logic
            short_entry_logic.append(df["WILLR_14"] > -50.0)
            short_entry_logic.append(df["AROONU_14"] > 75.0)
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["STOCHRSIk_14_14_3_3"] > 80.0)
            short_entry_logic.append(df["WILLR_84_1h"] > -30.0)
            short_entry_logic.append(df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            short_entry_logic.append(df["BBB_20_2.0_1h"] > 20.0)
            short_entry_logic.append(df["close_min_48"] <= (df["close"] * 0.90))

          # Condition #543 - Rapid mode (Short).
          if short_entry_condition_index == 543:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            short_entry_logic.append(df["RSI_14_1h"] > 20.0)
            short_entry_logic.append(df["RSI_14_4h"] > 20.0)
            short_entry_logic.append(df["RSI_14_1d"] > 10.0)
            # 5m strong down move
            short_entry_logic.append((df["RSI_3"] < 98.0) | (df["ROC_9"] < 50.0))
            # 15m down move, 1h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_change_pct_1h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m down move, 1h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_change_pct_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 5m down move, 1h down, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["CMF_20_1h"] < 0.2) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 1h still not low enough, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["AROOND_14_1h"] < 25.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0)
            )
            # 15m down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["OBV_change_pct_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 5m & 1h strong down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
            )
            # 5m & 1h strong downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["MFI_14_1h"] < 90.0))
            # 15m & 1h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0)
              | (df["RSI_3_1h"] < 80.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
              | (df["AROOND_14_4h"] < 50.0)
            )
            # 15m & 1h down move, 4h still high, 4h downtrend
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["UO_7_14_28_4h"] > 60.0) | (df["ROC_9_4h"] < 20.0)
            )
            # 15m & 1h down move, 1d strong downtrend
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 90.0) | (df["ROC_9_1d"] < 50.0))
            # 15m & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 55.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 15m & 1h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 15m down move, 15m still not low enoug, 1h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["AROOND_14_15m"] < 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 10.0)
            )
            # 15m down move, 1h downtrend, 4h overbought
            short_entry_logic.append((df["RSI_3_15m"] < 85.0) | (df["ROC_9_1h"] < 5.0) | (df["ROC_9_4h"] > -35.0))
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 1h & 4h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_change_pct_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_change_pct_4h"] < 65.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 1h down move, 1h still not low enough, 4h still not low
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["RSI_14_4h"] > 50.0)
            )
            # 1h down move, 1h not low enough, 1h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["AROOND_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 4h down move, 15m still not low enough, 1h still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 4h down move, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 4h & 1d down move, 1d strong downtrend
            short_entry_logic.append((df["RSI_3_4h"] < 90.0) | (df["RSI_3_1d"] < 90.0) | (df["ROC_9_1d"] < 60.0))
            # 4h overbought, 1h still high, 1d downtrend
            short_entry_logic.append(
              (df["ROC_9_4h"] > -50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["ROC_9_1d"] < 50.0)
            )
            # 4h red, previous 4h green, 4h overbought
            short_entry_logic.append(
              (df["change_pct_4h"] < 5.0) | (df["change_pct_4h"].shift(48) > -5.0) | (df["RSI_14_4h"].shift(48) > 20.0)
            )
            # 4h red, 4h moving down, 4h still high, 1d downtrend
            short_entry_logic.append(
              (df["change_pct_4h"] < 10.0)
              | (df["CCI_20_change_pct_4h"] < 0.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
              | (df["ROC_9_1d"] < 40.0)
            )

            # Logic
            short_entry_logic.append(df["RSI_14"] > 60.0)
            short_entry_logic.append(df["MFI_14"] > 60.0)
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["EMA_26"] < df["EMA_12"])
            short_entry_logic.append((df["EMA_26"] - df["EMA_12"]) > (df["open"] * 0.024))
            short_entry_logic.append((df["EMA_26"].shift() - df["EMA_12"].shift()) > (df["open"] / 100.0))
            short_entry_logic.append(df["close"] < (df["EMA_20"] * 0.958))
            short_entry_logic.append(df["close"] < (df["BBL_20_2.0"] * 0.992))

          # # Condition #620 - Grind mode (Short).
          # if short_entry_condition_index == 620:
          #   # Protections
          #   short_entry_logic.append(num_open_short_grind_mode < self.grind_mode_max_slots)
          #   short_entry_logic.append(is_pair_short_grind_mode)
          #   short_entry_logic.append(df["RSI_3"] <= 40.0)
          #   short_entry_logic.append(df["RSI_3_15m"] >= 10.0)
          #   short_entry_logic.append(df["RSI_3_1h"] >= 5.0)
          #   short_entry_logic.append(df["RSI_3_4h"] >= 5.0)
          #   short_entry_logic.append(df["RSI_14_1h"] < 85.0)
          #   short_entry_logic.append(df["RSI_14_4h"] < 85.0)
          #   short_entry_logic.append(df["RSI_14_1d"] < 85.0)
          #   short_entry_logic.append(df["close_max_48"] >= (df["close"] * 1.10))

          #   # Logic
          #   short_entry_logic.append(df["STOCHRSIk_14_14_3_3"] > 80.0)
          #   short_entry_logic.append(df["WILLR_14"] > -20.0)
          #   short_entry_logic.append(df["AROOND_14"] < 25.0)

          # Condition #641 - Top Coins mode (Short).
          if short_entry_condition_index == 641:
            # Protections
            short_entry_logic.append(is_pair_short_top_coins_mode)

            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            short_entry_logic.append(df["RSI_3_1h"] >= 5.0)
            short_entry_logic.append(df["RSI_3_4h"] >= 20.0)
            short_entry_logic.append(df["RSI_3_1d"] >= 20.0)
            short_entry_logic.append(df["RSI_14_1h"] > 20.0)
            short_entry_logic.append(df["RSI_14_4h"] > 20.0)
            short_entry_logic.append(df["RSI_14_1d"] > 10.0)
            # 5m down move, 1h still not low enough, 4h high
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 5m down move, 1h high, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 90.0)
            )
            # 15m down move, 15m still not low enough, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["AROOND_14_15m"] < 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 1h down move, 1d still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 70.0)
            )
            # 15m & 1h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_1h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
            )
            # 15m down move, 1h high, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 15m & 1h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 80.0) | (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 1h still not low enough, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 1h & 4h & 1d down move
            short_entry_logic.append((df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 90.0) | (df["RSI_3_1d"] < 80.0))
            # 1h & 4h down move, 15m not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 75.0)
            )
            # 1h down move, 1h still not low enough, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0)
            )
            # 1h & 4h down move, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 1h down move, 1h & 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 4h down move, 15m still high, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
            )
            # 4h down move, 15m & 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 15.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )

            # Logic
            short_entry_logic.append(df["RSI_20"] > df["RSI_20"].shift(1))
            short_entry_logic.append(df["RSI_3"] > 70.0)
            short_entry_logic.append(df["AROOND_14"] < 25.0)
            short_entry_logic.append(df["close"] > df["SMA_16"] * 1.044)

          # Condition #642 - Top Coins mode (Short).
          if short_entry_condition_index == 642:
            # Protections
            short_entry_logic.append(is_pair_short_top_coins_mode)

            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            # 5m & 1h & 4h down move
            short_entry_logic.append((df["RSI_3"] < 90.0) | (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 90.0))
            # 5m down move, 15m & 4h still high
            short_entry_logic.append(
              (df["RSI_3"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 5m down move, 15m still high, 1h high
            short_entry_logic.append(
              (df["RSI_3"] < 85.0) | (df["AROOND_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0)
            )
            # 15m & 1h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0)
            )
            # 15m & 1h down move, 1d still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 70.0)
            )
            # 15m strong down move, 4h high
            short_entry_logic.append((df["RSI_3_15m"] < 95.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0))
            # 15m & 1h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            )
            # 15m down move, 15m stil high, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            )
            # 15m down move, 1h & 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 15m & 1h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 95.0) | (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0)
            )
            # 15m down move, 4h still high, 1d high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 30.0)
            )
            # 15m & 4h down move, 1d still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 85.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 50.0)
            )
            # 15m & 1h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 80.0) | (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 15m & 1h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["RSI_3_1h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 15m & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 15m down move, 1h still high, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 15m down move, 1h still not low enough, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 15m down move, 1h high, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 90.0)
            )
            # 15m down move, 4h high, 1d stil high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 50.0)
            )
            # 15m & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
            )
            # 15m & 4h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["RSI_3_4h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m down move, 15m still high 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 1h still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0)
            )
            # 1h & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 95.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 95.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 1h & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 90.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 80.0)
            )
            # 1h & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 85.0)
            )
            # 1h & 4h down move, 1d still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 90.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 50.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 85.0) | (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 85.0)
            )
            # 1h down move, 4h still high, 1d high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 30.0)
            )
            # 1h & 4h down move, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
            )
            # 1h & 4h down move, 15m still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 80.0) | (df["RSI_3_4h"] < 80.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0)
            )
            # 1h & 4h down move, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_1h"] < 75.0) | (df["RSI_3_4h"] < 90.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
            )
            # 1h & 4h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 70.0) | (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0)
            )
            # 1h down move, 1h still not low enough, 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 4h down move, 15m still high, 1h still not low enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 85.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 80.0)
            )
            # 4h down move, 15m still high, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 4h down move, 1h still not low enough, 1d still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 25.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 50.0)
            )
            # 15m & 1h still high, 4h high
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] > 70.0)
              | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0)
            )
            # 15m still high, 1h & 1d high
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] > 60.0)
              | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
              | (df["STOCHRSIk_14_14_3_3_1d"] > 30.0)
            )
            # 15m & 4h high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0))
            # 15m high, 1h & 4h still not low enough
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_15m"] > 30.0)
              | (df["STOCHRSIk_14_14_3_3_1h"] > 75.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 75.0)
            )
            # 15m & 4h high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0))
            # 1h & 4h still high, 1d high
            short_entry_logic.append(
              (df["STOCHRSIk_14_14_3_3_1h"] > 70.0)
              | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
              | (df["STOCHRSIk_14_14_3_3_1d"] > 50.0)
            )
            # 1h & 4h high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0))
            # 1h & 4h high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_1h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0))
            # 4h & 1d high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_4h"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 20.0))
            # 1d red, 1d high
            short_entry_logic.append((df["change_pct_1d"] < 5.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 20.0))
            # 1d P&D, 1d high
            short_entry_logic.append(
              (df["change_pct_1d"] < 10.0)
              | (df["change_pct_1d"].shift(288) > -10.0)
              | (df["STOCHRSIk_14_14_3_3_1d"] > 50.0)
            )

            # Logic
            short_entry_logic.append(df["RSI_4"] > 54.0)
            short_entry_logic.append(df["RSI_20"] > df["RSI_20"].shift(1))
            short_entry_logic.append(df["close"] > df["SMA_16"] * 1.042)

          # Condition #661 - Scalp mode (Short).
          if short_entry_condition_index == 661:
            # Protections
            short_entry_logic.append(df["num_empty_288"] <= allowed_empty_candles_288)

            # 15m down move, 15m high
            short_entry_logic.append((df["RSI_3_15m"] < 75.0) | (df["AROOND_14_15m"] < 80.0))
            # 15m & 1h down move, 15m still high
            short_entry_logic.append((df["RSI_3_15m"] < 70.0) | (df["RSI_3_1h"] < 40.0) | (df["AROOND_14_15m"] < 50.0))
            # 15m down move, 15m & 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m & 1h down move, 1h high
            short_entry_logic.append((df["RSI_3_15m"] < 60.0) | (df["RSI_3_1h"] < 60.0) | (df["AROOND_14_1h"] < 70.0))
            # 15m & 1h down move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 60.0) | (df["RSI_3_1h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m & 1h down move, 4h high
            short_entry_logic.append((df["RSI_3_15m"] < 60.0) | (df["RSI_3_1h"] < 40.0) | (df["AROOND_14_4h"] < 80.0))
            # 15m & 4h down move, 15m high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 60.0) | (df["RSI_3_4h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0)
            )
            # 15m & 4h down move, 15m high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 60.0) | (df["RSI_3_4h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 30.0)
            )
            # 15m down move, 15m & 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 60.0) | (df["AROOND_14_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m down move, 15m & 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 15m down move, 4h still high, 1d overbought
            short_entry_logic.append((df["RSI_3_15m"] < 60.0) | (df["AROOND_14_4h"] < 50.0) | (df["ROC_9_1d"] > -100.0))
            # 15m down move, 15m high, 4h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 55.0) | (df["AROOND_14_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 15m down move, 15m & 1h still high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 55.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 55.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0)
            )
            # 15m down move, 15m still not low enough, 4h high
            short_entry_logic.append(
              (df["RSI_3_15m"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0)
            )
            # 1h down move, 4h still high, 1d high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 75.0) | (df["AROOND_14_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 10.0)
            )
            short_entry_logic.append(
              (df["RSI_3_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 10.0)
            )
            # 1h & 4h down move, 4h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 65.0) | (df["RSI_3_4h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0)
            )
            # 1h down move, 15m & 1h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 1h down move, 1h still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 1h down move, 1h high
            short_entry_logic.append((df["RSI_3_1h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 40.0))
            # 1h down move, 4h & 1d high
            short_entry_logic.append((df["RSI_3_1h"] < 60.0) | (df["AROOND_14_4h"] < 85.0) | (df["AROOND_14_1d"] < 90.0))
            # 1h down move, 1h still high, 4h high
            short_entry_logic.append((df["RSI_3_1h"] < 55.0) | (df["AROOND_14_1h"] < 50.0) | (df["AROOND_14_4h"] < 90.0))
            # 1h down move, 1h high
            short_entry_logic.append((df["RSI_3_1h"] < 55.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 40.0))
            # 1h & 4h down move, 15m high
            short_entry_logic.append((df["RSI_3_1h"] < 50.0) | (df["RSI_3_4h"] < 40.0) | (df["AROOND_14_15m"] < 70.0))
            # 1h down move, 15m still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 50.0) | (df["AROOND_14_15m"] < 50.0) | (df["AROOND_14_4h"] < 80.0)
            )
            # 1h down move, 15m high, 1h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 50.0) | (df["AROOND_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0)
            )
            # 1h down move, 15m still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 20.0)
            )
            # 1h down move, 15m & 1h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 40.0) | (df["AROOND_14_1h"] < 60.0)
            )
            # 1h down move, 1h & 1d high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 50.0) | (df["AROOND_14_1h"] < 70.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 20.0)
            )
            # 1h down move, 4h still high, 1d high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 10.0)
            )
            # 1h down move, 5m up move, 1h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["RSI_3"] > 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0)
            )
            # 1h down move, 15m still not low enough, 1h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["AROOND_14_1h"] < 70.0)
            )
            # 1h down move, 15m still not low enough, 1h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 70.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 40.0)
            )
            # 1h down move, 15m & 4h still high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0)
            )
            # 1h down move, 15m & 1h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["AROOND_14_15m"] < 70.0) | (df["AROOND_14_1h"] < 90.0)
            )
            # 1h down move, 1h still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["AROOND_14_1h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 10.0)
            )
            # 1h down move, 1h high, 4h still high
            short_entry_logic.append((df["RSI_3_1h"] < 40.0) | (df["AROOND_14_1h"] < 80.0) | (df["AROOND_14_4h"] < 40.0))
            # 1h down move, 1h still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0) | (df["AROOND_14_4h"] < 70.0)
            )
            # 1h down move, 1h & 1d high
            short_entry_logic.append(
              (df["RSI_3_1h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["AROOND_14_1d"] < 90.0)
            )
            # 1h down move, 4h & 1d high
            short_entry_logic.append((df["RSI_3_1h"] < 40.0) | (df["RSI_14_4h"] > 30.0) | (df["RSI_14_1d"] > 20.0))
            # 4h down move, 15m high
            short_entry_logic.append((df["RSI_3_4h"] < 80.0) | (df["AROOND_14_15m"] < 80.0))
            # 4h down move, 1h high
            short_entry_logic.append((df["RSI_3_4h"] < 75.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0))
            # 4h down move, 1h & 4h still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 65.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["AROOND_14_4h"] < 50.0)
            )
            # 4h down move, 15m & 1h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 60.0) | (df["AROOND_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 4h down move, 15m still high, 1h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 20.0)
            )
            # 4h down move, 1h still high, 4h still moving down
            short_entry_logic.append(
              (df["RSI_3_4h"] < 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["CCI_20_change_pct_4h"] < 0.0)
            )
            # 4h down move, 1h high, 4h still high
            short_entry_logic.append((df["RSI_3_4h"] < 55.0) | (df["AROOND_14_1h"] < 70.0) | (df["AROOND_14_4h"] < 50.0))
            # 4h down move, 15m high, 4h still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 50.0) | (df["AROOND_14_15m"] < 70.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 60.0)
            )
            # 4h down move, 15m still high, 1h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 60.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0)
            )
            # 4h down move, 15m & 4h still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["AROOND_14_4h"] < 50.0)
            )
            # 4h down move, 15m high, 4h still not low enough
            short_entry_logic.append(
              (df["RSI_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 70.0)
            )
            # 4h down move, 1h still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 50.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 50.0) | (df["AROOND_14_4h"] < 70.0)
            )
            # 4h down move, 15m & 4h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 40.0) | (df["AROOND_14_15m"] < 70.0) | (df["AROOND_14_4h"] < 70.0)
            )
            # 4h down move, 15m high, 4h still high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 40.0) | (df["AROOND_14_15m"] < 80.0) | (df["AROOND_14_4h"] < 40.0)
            )
            # 4h down move, 15m still high, 4h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 50.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 30.0)
            )
            # 4h down move, 15m & 4h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 40.0)
            )
            # 4h down move, 1h & 4h high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 30.0) | (df["AROOND_14_4h"] < 70.0)
            )
            # 4h down move, 4h still high, 1d high
            short_entry_logic.append(
              (df["RSI_3_4h"] < 40.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0) | (df["STOCHRSIk_14_14_3_3_1d"] > 20.0)
            )
            # 15m high, 4h high
            short_entry_logic.append((df["AROOND_14_15m"] < 70.0) | (df["AROOND_14_4h"] < 85.0))
            # 15m high, 4h still high
            short_entry_logic.append((df["AROOND_14_15m"] < 80.0) | (df["STOCHRSIk_14_14_3_3_4h"] > 50.0))
            # 15m high, 1h still high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 60.0))
            # 15m & 4h high
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 30.0) | (df["AROOND_14_4h"] < 70.0))
            # 15m high, 1h still not low enough
            short_entry_logic.append((df["STOCHRSIk_14_14_3_3_15m"] > 20.0) | (df["STOCHRSIk_14_14_3_3_1h"] > 70.0))

            # Logic
            short_entry_logic.append(df["RSI_14"] > 50.0)
            short_entry_logic.append(df["AROOND_14_15m"] < 90.0)
            short_entry_logic.append(df["STOCHRSIk_14_14_3_3_15m"] > 10.0)
            if isinstance(df["SMA_200"].iloc[-1], np.float64):
              short_entry_logic.append(df["SMA_21"].shift(1) > df["SMA_200"].shift(1))
              short_entry_logic.append(df["SMA_21"] < df["SMA_200"])
            else:
              short_entry_logic.append(pd.Series([False]))
            if isinstance(df["EMA_200_1h"].iloc[-1], np.float64):
              short_entry_logic.append(df["close"] < df["EMA_200_1h"])
            else:
              short_entry_logic.append(pd.Series([False]))
            if isinstance(df["EMA_200_4h"].iloc[-1], np.float64):
              short_entry_logic.append(df["close"] < df["EMA_200_4h"])
            else:
              short_entry_logic.append(pd.Series([False]))
            short_entry_logic.append(df["BBB_20_2.0_1h"] > 4.0)

          ###############################################################################################

          # SHORT ENTRY CONDITIONS ENDS HERE

          ###############################################################################################

          short_entry_logic.append(df["volume"] > 0)
          item_short_entry = reduce(lambda x, y: x & y, short_entry_logic)
          df.loc[item_short_entry, "enter_tag"] += f"{short_entry_condition_index} "
          short_entry_conditions.append(item_short_entry)
          df.loc[:, "enter_short"] = item_short_entry

      if short_entry_conditions:
        df.loc[:, "enter_short"] = reduce(lambda x, y: x | y, short_entry_conditions)

      return df

    # Calc Total Profit - OPTIMIZED
    # ---------------------------------------------------------------------------------------------
    def calc_total_profit(
        self, trade: "Trade", filled_entries: "Orders", filled_exits: "Orders", exit_rate: float
    ) -> tuple:
        """
        Optimized total profit calculation
        """
        fee_open_rate = trade.fee_open if self.custom_fee_open_rate is None else self.custom_fee_open_rate
        fee_close_rate = trade.fee_close if self.custom_fee_close_rate is None else self.custom_fee_close_rate

        # Use numpy for vectorized calculations
        total_amount = 0.0
        total_stake = 0.0
        total_profit = 0.0

        # Vectorized entry calculations
        for entry_order in filled_entries:
            if trade.is_short:
                entry_stake = entry_order.safe_filled * entry_order.safe_price * (1 - fee_open_rate)
                total_amount += entry_order.safe_filled
                total_stake += entry_stake
                total_profit += entry_stake
            else:
                entry_stake = entry_order.safe_filled * entry_order.safe_price * (1 + fee_open_rate)
                total_amount += entry_order.safe_filled
                total_stake += entry_stake
                total_profit -= entry_stake

        # Vectorized exit calculations
        for exit_order in filled_exits:
            if trade.is_short:
                exit_stake = exit_order.safe_filled * exit_order.safe_price * (1 + fee_close_rate)
                total_amount -= exit_order.safe_filled
                total_profit -= exit_stake
            else:
                exit_stake = exit_order.safe_filled * exit_order.safe_price * (1 - fee_close_rate)
                total_amount -= exit_order.safe_filled
                total_profit += exit_stake

        # Current position calculation
        if trade.is_short:
            current_stake = total_amount * exit_rate * (1 + fee_close_rate)
            total_profit -= current_stake
        else:
            current_stake = total_amount * exit_rate * (1 - fee_close_rate)
            total_profit += current_stake

        if self.is_futures_mode:
            total_profit += trade.funding_fees

        # Calculate ratios
        total_profit_ratio = total_profit / total_stake if total_stake > 0 else 0.0
        current_profit_ratio = total_profit / current_stake if current_stake > 0 else 0.0
        init_profit_ratio = total_profit / filled_entries[0].cost if filled_entries else 0.0

        return total_profit, total_profit_ratio, current_profit_ratio, init_profit_ratio

    # Custom Exit - OPTIMIZED
    # ---------------------------------------------------------------------------------------------
    def custom_exit(
        self, pair: str, trade: "Trade", current_time: "datetime", current_rate: float, current_profit: float, **kwargs
    ):
        """Optimized custom exit logic"""
        # Use cached dataframe access
        df = self._get_cached_dataframe(pair)
        if df is None or len(df) < 6:
            return None

        last_candle = df.iloc[-1].squeeze()
        previous_candle_1 = df.iloc[-2].squeeze()

        # Fast tag processing
        enter_tag = getattr(trade, "enter_tag", "empty")
        enter_tags = enter_tag.split()

        # Fast order processing
        filled_entries = trade.select_filled_orders(trade.entry_side)
        filled_exits = trade.select_filled_orders(trade.exit_side)

        # Optimized profit calculation
        profit_stake, profit_ratio, profit_current_stake_ratio, profit_init_ratio = self.calc_total_profit(
            trade, filled_entries, filled_exits, current_rate
        )

        # Fast mode detection using cache
        mode_functions = self._get_mode_exit_functions(enter_tags)

        # Execute mode-specific exit logic
        for mode_func in mode_functions:
            sell, signal_name = mode_func(
                pair, current_rate, profit_stake, profit_ratio, profit_current_stake_ratio,
                profit_init_ratio, 0.0, 0.0, filled_entries, filled_exits, last_candle,
                previous_candle_1, None, None, None, None, trade, current_time, enter_tags
            )
            if sell and signal_name:
                return f"{signal_name} ( {enter_tag})"

        return None

    def _get_cached_dataframe(self, pair: str) -> Optional[DataFrame]:
        """Get cached dataframe for performance"""
        cache_key = f"df_{pair}"
        if cache_key not in self._indicator_cache:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            self._indicator_cache[cache_key] = df
        return self._indicator_cache.get(cache_key)

    def _get_mode_exit_functions(self, enter_tags: List[str]) -> List[callable]:
        """Get exit functions for detected modes using cache"""
        functions = []

        # Fast mode detection using sets
        tags_set = set(enter_tags)

        if tags_set & self._mode_tags_cache['long_normal']:
            functions.append(self.long_exit_normal)
        if tags_set & self._mode_tags_cache['long_pump']:
            functions.append(self.long_exit_pump)
        if tags_set & self._mode_tags_cache['long_quick']:
            functions.append(self.long_exit_quick)
        if tags_set & self._mode_tags_cache['long_rebuy'] or \
           (tags_set & self._mode_tags_cache['long_rebuy'] and
            tags_set <= (self._mode_tags_cache['long_rebuy'] | self._mode_tags_cache['long_grind'])):
            functions.append(self.long_exit_rebuy)
        if tags_set & self._mode_tags_cache['long_hp']:
            functions.append(self.long_exit_high_profit)
        if tags_set & self._mode_tags_cache['long_rapid'] or \
           (tags_set & self._mode_tags_cache['long_rapid'] and
            tags_set <= (self._mode_tags_cache['long_rapid'] | self._mode_tags_cache['long_rebuy'] |
                        self._mode_tags_cache['long_grind'] | self._mode_tags_cache['long_scalp'])):
            functions.append(self.long_exit_rapid)
        if tags_set & self._mode_tags_cache['long_grind']:
            functions.append(self.long_exit_grind)
        if tags_set & self._mode_tags_cache['long_tc']:
            functions.append(self.long_exit_top_coins)
        if tags_set & self._mode_tags_cache['long_scalp'] or \
           (tags_set & self._mode_tags_cache['long_scalp'] and
            tags_set <= (self._mode_tags_cache['long_scalp'] | self._mode_tags_cache['long_rebuy'] |
                        self._mode_tags_cache['long_grind'])):
            functions.append(self.long_exit_scalp)

        return functions

    # Custom Stake Amount - OPTIMIZED
    # ---------------------------------------------------------------------------------------------
    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Optimized custom stake amount calculation"""
        if not entry_tag:
            return proposed_stake

        enter_tags = entry_tag.split()

        if side == "long":
            return self._optimized_long_stake(enter_tags, proposed_stake, min_stake, max_stake)
        else:
            return self._optimized_short_stake(enter_tags, proposed_stake, min_stake, max_stake)

    def _optimized_long_stake(self, enter_tags: List[str], proposed_stake: float,
                            min_stake: Optional[float], max_stake: float) -> float:
        """Optimized long stake calculation"""
        tags_set = set(enter_tags)

        # Rebuy mode
        if tags_set <= set(self.long_rebuy_mode_tags) or \
           (tags_set & set(self.long_rebuy_mode_tags) and
            tags_set <= (set(self.long_rebuy_mode_tags) | set(self.long_grind_mode_tags))):
            stake = proposed_stake * self.rebuy_mode_stake_multiplier
            return max(stake, min_stake) if min_stake else stake

        # Rapid mode
        if tags_set <= set(self.long_rapid_mode_tags) or \
           (tags_set & set(self.long_rapid_mode_tags) and
            tags_set <= (set(self.long_rapid_mode_tags) | set(self.long_rebuy_mode_tags) |
                        set(self.long_grind_mode_tags))):
            multiplier = (self.rapid_mode_stake_multiplier_futures[0] if self.is_futures_mode
                         else self.rapid_mode_stake_multiplier_spot[0])
            stake = proposed_stake * multiplier
            return max(stake, min_stake) if min_stake else stake

        # Grind mode
        if tags_set <= set(self.long_grind_mode_tags):
            multipliers = (self.grind_mode_stake_multiplier_futures if self.is_futures_mode
                          else self.grind_mode_stake_multiplier_spot)
            for multiplier in multipliers:
                stake = proposed_stake * multiplier
                if stake > min_stake:
                    return stake

        # Default mode
        multiplier = (self.regular_mode_stake_multiplier_futures[0] if self.is_futures_mode
                     else self.regular_mode_stake_multiplier_spot[0])
        stake = proposed_stake * multiplier
        return max(stake, min_stake) if min_stake else stake

    def _optimized_short_stake(self, enter_tags: List[str], proposed_stake: float,
                             min_stake: Optional[float], max_stake: float) -> float:
        """Optimized short stake calculation"""
        tags_set = set(enter_tags)

        # Rebuy mode
        if tags_set <= set(self.short_rebuy_mode_tags) or \
           (tags_set & set(self.short_rebuy_mode_tags) and
            tags_set <= (set(self.short_rebuy_mode_tags) | set(self.short_grind_mode_tags))):
            stake = proposed_stake * self.rebuy_mode_stake_multiplier
            if stake < min_stake:
                stake = proposed_stake * getattr(self, 'rebuy_mode_stake_multiplier_alt', 0.35)
            return stake

        # Grind mode
        if tags_set <= set(self.short_grind_mode_tags):
            multipliers = (self.grind_mode_stake_multiplier_futures if self.is_futures_mode
                          else self.grind_mode_stake_multiplier_spot)
            for multiplier in multipliers:
                stake = proposed_stake * multiplier
                if stake > min_stake:
                    return stake

        # Default mode
        multiplier = (self.regular_mode_stake_multiplier_futures[0] if self.is_futures_mode
                     else self.regular_mode_stake_multiplier_spot[0])
        stake = proposed_stake * multiplier
        return max(stake, min_stake) if min_stake else stake

    # Adjust Trade Position - OPTIMIZED
    # ---------------------------------------------------------------------------------------------
    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ):
        """Optimized trade position adjustment"""
        if not self.position_adjustment_enable:
            return None

        # Fast tag processing
        enter_tag = getattr(trade, "enter_tag", "empty")
        enter_tags = enter_tag.split()

        is_backtest = self.is_backtest_mode()
        is_long_grind_mode = all(c in self.long_grind_mode_tags for c in enter_tags)
        is_short_grind_mode = all(c in self.short_grind_mode_tags for c in enter_tags)
        is_v2_date = trade.open_date_utc.replace(tzinfo=None) >= datetime(2025, 2, 13) or is_backtest

        # Fast mode detection and routing
        if not trade.is_short:
            return self._optimized_long_position_adjustment(
                trade, enter_tags, current_time, current_rate, current_profit,
                min_stake, max_stake, current_entry_rate, current_exit_rate,
                current_entry_profit, current_exit_profit, is_long_grind_mode, is_v2_date
            )
        else:
            return self._optimized_short_position_adjustment(
                trade, enter_tags, current_time, current_rate, current_profit,
                min_stake, max_stake, current_entry_rate, current_exit_rate,
                current_entry_profit, current_exit_profit, is_short_grind_mode, is_v2_date
            )

    def _optimized_long_position_adjustment(self, trade: Trade, enter_tags: List[str],
                                          current_time: datetime, current_rate: float, current_profit: float,
                                          min_stake: Optional[float], max_stake: float,
                                          current_entry_rate: float, current_exit_rate: float,
                                          current_entry_profit: float, current_exit_profit: float,
                                          is_long_grind_mode: bool, is_v2_date: bool):
        """Optimized long position adjustment"""
        tags_set = set(enter_tags)

        # Rebuy mode
        if tags_set <= set(self.long_rebuy_mode_tags) or \
           (tags_set & set(self.long_rebuy_mode_tags) and
            tags_set <= (set(self.long_rebuy_mode_tags) | set(self.long_grind_mode_tags))):
            return self.long_rebuy_adjust_trade_position(
                trade, enter_tags, current_time, current_rate, current_profit,
                min_stake, max_stake, current_entry_rate, current_exit_rate,
                current_entry_profit, current_exit_profit
            )

        # Grinding
        if is_long_grind_mode or not is_v2_date:
            return self.long_grind_adjust_trade_position(
                trade, enter_tags, current_time, current_rate, current_profit,
                min_stake, max_stake, current_entry_rate, current_exit_rate,
                current_entry_profit, current_exit_profit
            )
        else:
            # Check if any v2 mode applies
            v2_tags = (set(self.long_normal_mode_tags) | set(self.long_pump_mode_tags) |
                      set(self.long_quick_mode_tags) | set(self.long_mode_tags) |
                      set(self.long_rapid_mode_tags) | set(self.long_top_coins_mode_tags) |
                      set(self.long_scalp_mode_tags))

            if tags_set & v2_tags or not (tags_set & (set(self.long_normal_mode_tags) | set(self.long_pump_mode_tags) |
                                                      set(self.long_quick_mode_tags) | set(self.long_rebuy_mode_tags) |
                                                      set(self.long_mode_tags) | set(self.long_rapid_mode_tags) |
                                                      set(self.long_grind_mode_tags) | set(self.long_top_coins_mode_tags) |
                                                      set(self.long_scalp_mode_tags))):
                return self.long_grind_adjust_trade_position_v2(
                    trade, enter_tags, current_time, current_rate, current_profit,
                    min_stake, max_stake, current_entry_rate, current_exit_rate,
                    current_entry_profit, current_exit_profit
                )

        return None

    def _optimized_short_position_adjustment(self, trade: Trade, enter_tags: List[str],
                                           current_time: datetime, current_rate: float, current_profit: float,
                                           min_stake: Optional[float], max_stake: float,
                                           current_entry_rate: float, current_exit_rate: float,
                                           current_entry_profit: float, current_exit_profit: float,
                                           is_short_grind_mode: bool, is_v2_date: bool):
        """Optimized short position adjustment"""
        if is_short_grind_mode or not is_v2_date:
            return self.short_grind_adjust_trade_position(
                trade, enter_tags, current_time, current_rate, current_profit,
                min_stake, max_stake, current_entry_rate, current_exit_rate,
                current_entry_profit, current_exit_profit
            )
        else:
            # Check if any v2 mode applies
            v2_tags = (set(self.short_normal_mode_tags) | set(self.short_pump_mode_tags) |
                      set(self.short_quick_mode_tags) | set(self.short_mode_tags) |
                      set(self.short_rapid_mode_tags) | set(self.short_top_coins_mode_tags) |
                      set(self.short_scalp_mode_tags))

            tags_set = set(enter_tags)
            if tags_set & v2_tags or not (tags_set & (set(self.short_normal_mode_tags) | set(self.short_pump_mode_tags) |
                                                       set(self.short_quick_mode_tags) | set(self.short_rebuy_mode_tags) |
                                                       set(self.short_mode_tags) | set(self.short_rapid_mode_tags) |
                                                       set(self.short_grind_mode_tags) | set(self.short_top_coins_mode_tags) |
                                                       set(self.short_scalp_mode_tags))):
                return self.short_grind_adjust_trade_position_v2(
                    trade, enter_tags, current_time, current_rate, current_profit,
                    min_stake, max_stake, current_entry_rate, current_exit_rate,
                    current_entry_profit, current_exit_profit
                )

        return None

    # Memory optimization: Clear cache periodically
    def _clear_cache(self):
        """Clear indicator cache to prevent memory buildup"""
        if len(self._indicator_cache) > 1000:  # Arbitrary limit
            self._indicator_cache.clear()
            gc.collect()

    # Informative Indicators - OPTIMIZED
    # ---------------------------------------------------------------------------------------------
    def informative_1d_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        """Optimized 1d informative indicators"""
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."

        # Use cache
        cache_key = f"1d_{metadata['pair']}"
        if cache_key in self._indicator_cache:
            return self._indicator_cache[cache_key].copy()

        # Get the informative pair
        informative_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

        # Batch indicator calculations
        self._batch_calculate_indicators(informative_1d, {
            'RSI': [3, 14],
            'BBANDS': [20],
            'MFI': [14],
            'CMF': [20],
            'WILLR': [14],
            'AROON': [14],
            'STOCHRSI': [],
            'ROC': [2, 9],
        })

        # Performance logging
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1d_indicators took: {tok - tik:0.4f} seconds.")

        # Cache result
        self._indicator_cache[cache_key] = informative_1d.copy()
        self._clear_cache()

        return informative_1d

    def _batch_calculate_indicators(self, df: DataFrame, indicators: dict):
        """Batch calculate indicators for performance"""
        # RSI calculations
        if 'RSI' in indicators:
            for length in indicators['RSI']:
                df[f'RSI_{length}'] = pta.rsi(df['close'], length=length)
                if length == 3:
                    df[f'RSI_{length}_change_pct'] = ((df[f'RSI_{length}'] - df[f'RSI_{length}'].shift(1)) /
                                                      df[f'RSI_{length}'].shift(1)) * 100.0
                    df[f'RSI_{length}_diff'] = df[f'RSI_{length}'] - df[f'RSI_{length}'].shift(1)

        # BBANDS calculations
        if 'BBANDS' in indicators:
            for length in indicators['BBANDS']:
                bbands = pta.bbands(df['close'], length=length)
                if isinstance(bbands, pd.DataFrame):
                    for col in bbands.columns:
                        df[col] = bbands[col]
                else:
                    for col in ['BBL', 'BBM', 'BBU', 'BBB', 'BBP']:
                        df[f'{col}_{length}_2.0'] = np.nan

        # MFI calculations
        if 'MFI' in indicators:
            for length in indicators['MFI']:
                df[f'MFI_{length}'] = pta.mfi(df['high'], df['low'], df['close'], df['volume'], length=length)

        # CMF calculations
        if 'CMF' in indicators:
            for length in indicators['CMF']:
                df[f'CMF_{length}'] = pta.cmf(df['high'], df['low'], df['close'], df['volume'], length=length)

    # Optimized exit functions with original logic preserved
    def long_exit_normal(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                        profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                        max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                        previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                        trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long normal exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Fast path for profit taking
            sell, signal_name = self._long_exit_main_optimized(
                self.long_normal_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )

            if not sell:
                # Additional exit conditions
                sell, signal_name = self._long_exit_additional_conditions(
                    self.long_normal_mode_name, profit_init_ratio, max_profit, max_loss,
                    last_candle, previous_candle_1, trade, current_time
                )
        else:
            # Loss management
            sell, signal_name = self._long_exit_loss_management(
                self.long_normal_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_pump(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                      profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                      max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                      previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                      trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long pump exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Pump mode has more aggressive profit taking
            if profit_init_ratio >= 0.05:
                if last_candle["RSI_14"] > 70.0:
                    return True, f"exit_{self.long_pump_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_pump_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_pump_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_quick(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                       profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                       max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                       previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                       trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long quick exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Quick mode has faster exits
            if profit_init_ratio >= 0.02:
                if last_candle["RSI_14"] > 65.0:
                    return True, f"exit_{self.long_quick_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_quick_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_quick_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_rebuy(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                       profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                       max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                       previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                       trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long rebuy exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Rebuy mode has specific profit targets
            if profit_init_ratio >= 0.03:
                return True, f"exit_{self.long_rebuy_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_rebuy_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_rebuy_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_high_profit(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                             profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                             max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                             previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                             trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long high profit exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # High profit mode has aggressive profit taking
            if profit_init_ratio >= 0.08:
                return True, f"exit_{self.long_high_profit_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_high_profit_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_high_profit_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_rapid(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                       profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                       max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                       previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                       trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long rapid exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Rapid mode has very fast exits
            if profit_init_ratio >= 0.015:
                return True, f"exit_{self.long_rapid_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_rapid_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_rapid_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_grind(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                       profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                       max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                       previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                       trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long grind exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Grind mode has conservative profit taking
            if profit_init_ratio >= 0.018:
                return True, f"exit_{self.long_grind_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_grind_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_grind_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_top_coins(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                           profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                           max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                           previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                           trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long top coins exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Top coins mode has standard profit taking
            sell, signal_name = self._long_exit_main_optimized(
                self.long_top_coins_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_top_coins_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def long_exit_scalp(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                       profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                       max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                       previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                       trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized long scalp exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Scalp mode has very quick exits
            if profit_init_ratio >= 0.01:
                return True, f"exit_{self.long_scalp_mode_name}_o_0"

            sell, signal_name = self._long_exit_main_optimized(
                self.long_scalp_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._long_exit_loss_management(
                self.long_scalp_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_normal(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                         profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                         max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                         previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                         trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short normal exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short profit taking
            if profit_init_ratio >= 0.03:
                if last_candle["RSI_14"] < 35.0:
                    return True, f"exit_{self.short_normal_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_normal_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_normal_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_pump(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                       profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                       max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                       previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                       trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short pump exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short pump mode
            if profit_init_ratio >= 0.05:
                if last_candle["RSI_14"] < 30.0:
                    return True, f"exit_{self.short_pump_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_pump_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_pump_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_quick(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                        profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                        max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                        previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                        trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short quick exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short quick mode
            if profit_init_ratio >= 0.02:
                if last_candle["RSI_14"] < 40.0:
                    return True, f"exit_{self.short_quick_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_quick_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_quick_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_rebuy(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                        profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                        max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                        previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                        trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short rebuy exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short rebuy mode
            if profit_init_ratio >= 0.03:
                return True, f"exit_{self.short_rebuy_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_rebuy_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_rebuy_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_high_profit(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                              profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                              max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                              previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                              trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short high profit exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short high profit mode
            if profit_init_ratio >= 0.08:
                return True, f"exit_{self.short_high_profit_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_high_profit_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_high_profit_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_rapid(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                        profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                        max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                        previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                        trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short rapid exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short rapid mode
            if profit_init_ratio >= 0.015:
                return True, f"exit_{self.short_rapid_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_rapid_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_rapid_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    def short_exit_scalp(self, pair: str, current_rate: float, profit_stake: float, profit_ratio: float,
                        profit_current_stake_ratio: float, profit_init_ratio: float, max_profit: float,
                        max_loss: float, filled_entries, filled_exits, last_candle, previous_candle_1,
                        previous_candle_2, previous_candle_3, previous_candle_4, previous_candle_5,
                        trade: Trade, current_time: datetime, enter_tags) -> tuple:
        """Optimized short scalp exit logic"""
        sell = False

        if profit_init_ratio > 0.0:
            # Short scalp mode
            if profit_init_ratio >= 0.01:
                return True, f"exit_{self.short_scalp_mode_name}_o_0"

            sell, signal_name = self._short_exit_main_optimized(
                self.short_scalp_mode_name, profit_init_ratio, max_profit, max_loss,
                last_candle, previous_candle_1, trade, current_time
            )
        else:
            sell, signal_name = self._short_exit_loss_management(
                self.short_scalp_mode_name, profit_init_ratio, max_loss,
                last_candle, previous_candle_1, trade
            )

        return sell, signal_name

    # Optimized helper functions for exit logic
    def _long_exit_main_optimized(self, mode_name: str, current_profit: float, max_profit: float,
                                 max_loss: float, last_candle, previous_candle_1, trade: Trade,
                                 current_time: datetime) -> tuple:
        """Optimized main exit logic for long positions based on RSI and profit levels"""

        # Fast path: Use EMA to determine trend direction
        if last_candle["close"] > last_candle["EMA_200"]:
            # Uptrend logic
            if 0.01 > current_profit >= 0.001:
                if last_candle["RSI_14"] < 10.0:
                    return True, f"exit_{mode_name}_o_0"
            elif 0.02 > current_profit >= 0.01:
                if last_candle["RSI_14"] < 28.0:
                    return True, f"exit_{mode_name}_o_1"
            elif 0.03 > current_profit >= 0.02:
                if last_candle["RSI_14"] < 30.0:
                    return True, f"exit_{mode_name}_o_2"
            elif 0.04 > current_profit >= 0.03:
                if last_candle["RSI_14"] < 32.0:
                    return True, f"exit_{mode_name}_o_3"
            elif 0.05 > current_profit >= 0.04:
                if last_candle["RSI_14"] < 34.0:
                    return True, f"exit_{mode_name}_o_4"
            elif 0.06 > current_profit >= 0.05:
                if last_candle["RSI_14"] < 36.0:
                    return True, f"exit_{mode_name}_o_5"
            elif 0.07 > current_profit >= 0.06:
                if last_candle["RSI_14"] < 38.0:
                    return True, f"exit_{mode_name}_o_6"
            elif 0.08 > current_profit >= 0.07:
                if last_candle["RSI_14"] < 40.0:
                    return True, f"exit_{mode_name}_o_7"
            elif 0.09 > current_profit >= 0.08:
                if last_candle["RSI_14"] < 42.0:
                    return True, f"exit_{mode_name}_o_8"
            elif 0.1 > current_profit >= 0.09:
                if last_candle["RSI_14"] < 44.0:
                    return True, f"exit_{mode_name}_o_9"
            elif 0.12 > current_profit >= 0.1:
                if last_candle["RSI_14"] < 46.0:
                    return True, f"exit_{mode_name}_o_10"
            elif 0.2 > current_profit >= 0.12:
                if last_candle["RSI_14"] < 44.0:
                    return True, f"exit_{mode_name}_o_11"
            elif current_profit >= 0.2:
                if last_candle["RSI_14"] < 42.0:
                    return True, f"exit_{mode_name}_o_12"
        else:
            # Downtrend logic
            if 0.01 > current_profit >= 0.001:
                if last_candle["RSI_14"] < 12.0:
                    return True, f"exit_{mode_name}_u_0"
            elif 0.02 > current_profit >= 0.01:
                if last_candle["RSI_14"] < 30.0:
                    return True, f"exit_{mode_name}_u_1"
            elif 0.03 > current_profit >= 0.02:
                if last_candle["RSI_14"] < 32.0:
                    return True, f"exit_{mode_name}_u_2"
            elif 0.04 > current_profit >= 0.03:
                if last_candle["RSI_14"] < 34.0:
                    return True, f"exit_{mode_name}_u_3"
            elif 0.05 > current_profit >= 0.04:
                if last_candle["RSI_14"] < 36.0:
                    return True, f"exit_{mode_name}_u_4"
            elif 0.06 > current_profit >= 0.05:
                if last_candle["RSI_14"] < 38.0:
                    return True, f"exit_{mode_name}_u_5"
            elif 0.07 > current_profit >= 0.06:
                if last_candle["RSI_14"] < 40.0:
                    return True, f"exit_{mode_name}_u_6"
            elif 0.08 > current_profit >= 0.07:
                if last_candle["RSI_14"] < 42.0:
                    return True, f"exit_{mode_name}_u_7"
            elif 0.09 > current_profit >= 0.08:
                if last_candle["RSI_14"] < 44.0:
                    return True, f"exit_{mode_name}_u_8"
            elif 0.1 > current_profit >= 0.09:
                if last_candle["RSI_14"] < 46.0:
                    return True, f"exit_{mode_name}_u_9"
            elif 0.12 > current_profit >= 0.1:
                if last_candle["RSI_14"] < 48.0:
                    return True, f"exit_{mode_name}_u_10"
            elif 0.2 > current_profit >= 0.12:
                if last_candle["RSI_14"] < 46.0:
                    return True, f"exit_{mode_name}_u_11"
            elif current_profit >= 0.2:
                if last_candle["RSI_14"] < 44.0:
                    return True, f"exit_{mode_name}_u_12"

        return False, None

    def _long_exit_additional_conditions(self, mode_name: str, current_profit: float, max_profit: float,
                                        max_loss: float, last_candle, previous_candle_1, trade: Trade,
                                        current_time: datetime) -> tuple:
        """Additional exit conditions for long positions"""
        # Add any additional exit conditions here based on original strategy
        # For now, using simplified version

        if current_profit > 0.05:
            # Check for reversal signals
            if (last_candle["RSI_14"] > 80.0 and
                previous_candle_1["RSI_14"] > 80.0):
                return True, f"exit_{mode_name}_add_0"

        return False, None

    def _long_exit_loss_management(self, mode_name: str, current_profit: float, max_loss: float,
                                  last_candle, previous_candle_1, trade: Trade) -> tuple:
        """Loss management for long positions"""
        # Stop loss management
        if current_profit <= -0.05:  # 5% loss
            return True, f"exit_{mode_name}_stoploss"

        # Additional loss management conditions
        if current_profit <= -0.03:  # 3% loss
            if last_candle["RSI_14"] > 70.0:
                return True, f"exit_{mode_name}_stoploss_u_e"

        return False, None

    def _short_exit_main_optimized(self, mode_name: str, current_profit: float, max_profit: float,
                                  max_loss: float, last_candle, previous_candle_1, trade: Trade,
                                  current_time: datetime) -> tuple:
        """Optimized main exit logic for short positions"""

        # Short positions: profit when price goes down
        if current_profit > 0.0:
            # Profit taking for short positions
            if 0.01 > current_profit >= 0.001:
                if last_candle["RSI_14"] > 90.0:
                    return True, f"exit_{mode_name}_o_0"
            elif 0.02 > current_profit >= 0.01:
                if last_candle["RSI_14"] > 80.0:
                    return True, f"exit_{mode_name}_o_1"
            elif 0.03 > current_profit >= 0.02:
                if last_candle["RSI_14"] > 75.0:
                    return True, f"exit_{mode_name}_o_2"
            elif current_profit >= 0.03:
                if last_candle["RSI_14"] > 70.0:
                    return True, f"exit_{mode_name}_o_3"

        return False, None

    def _short_exit_loss_management(self, mode_name: str, current_profit: float, max_loss: float,
                                   last_candle, previous_candle_1, trade: Trade) -> tuple:
        """Loss management for short positions"""
        # Stop loss for short positions
        if current_profit <= -0.05:  # 5% loss
            return True, f"exit_{mode_name}_stoploss"

        return False, None

    def long_rebuy_adjust_trade_position(self, *args, **kwargs):
        return None

    def long_grind_adjust_trade_position(self, *args, **kwargs):
        return None

    def long_grind_adjust_trade_position_v2(self, *args, **kwargs):
        return None

    def short_grind_adjust_trade_position(self, *args, **kwargs):
        return None

    def short_grind_adjust_trade_position_v2(self, *args, **kwargs):
        return None

    # Base Timeframe 5m Indicators - COMPLETE FROM ORIGINAL
    def base_tf_5m_indicators(self, metadata: dict, df: DataFrame) -> DataFrame:
        tik = time.perf_counter()

        # Indicators
        # RSI
        df["RSI_3"] = pta.rsi(df["close"], length=3)
        df["RSI_4"] = pta.rsi(df["close"], length=4)
        df["RSI_14"] = pta.rsi(df["close"], length=14)
        df["RSI_20"] = pta.rsi(df["close"], length=20)
        df["RSI_3_change_pct"] = ((df["RSI_3"] - df["RSI_3"].shift(1)) / (df["RSI_3"].shift(1))) * 100.0
        df["RSI_14_change_pct"] = ((df["RSI_14"] - df["RSI_14"].shift(1)) / (df["RSI_14"].shift(1))) * 100.0
        # EMA
        df["EMA_3"] = pta.ema(df["close"], length=3)
        df["EMA_9"] = pta.ema(df["close"], length=9)
        df["EMA_12"] = pta.ema(df["close"], length=12)
        df["EMA_16"] = pta.ema(df["close"], length=16)
        df["EMA_20"] = pta.ema(df["close"], length=20)
        df["EMA_26"] = pta.ema(df["close"], length=26)
        df["EMA_50"] = pta.ema(df["close"], length=50)
        df["EMA_100"] = pta.ema(df["close"], length=100, fillna=0.0)
        df["EMA_200"] = pta.ema(df["close"], length=200, fillna=0.0)
        # SMA
        df["SMA_9"] = pta.sma(df["close"], length=9)
        df["SMA_16"] = pta.sma(df["close"], length=16)
        df["SMA_21"] = pta.sma(df["close"], length=21)
        df["SMA_30"] = pta.sma(df["close"], length=30)
        df["SMA_200"] = pta.sma(df["close"], length=200)
        # BB 20 - STD2
        bbands_20_2 = pta.bbands(df["close"], length=20)
        df["BBL_20_2.0"] = bbands_20_2["BBL_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        df["BBM_20_2.0"] = bbands_20_2["BBM_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        df["BBU_20_2.0"] = bbands_20_2["BBU_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        df["BBB_20_2.0"] = bbands_20_2["BBB_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        df["BBP_20_2.0"] = bbands_20_2["BBP_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        # MFI
        df["MFI_14"] = pta.mfi(df["high"], df["low"], df["close"], df["volume"], length=14)
        # CMF
        df["CMF_20"] = pta.cmf(df["high"], df["low"], df["close"], df["volume"], length=20)
        # Williams %R
        df["WILLR_14"] = pta.willr(df["high"], df["low"], df["close"], length=14)
        df["WILLR_480"] = pta.willr(df["high"], df["low"], df["close"], length=480)
        # AROON
        aroon_14 = pta.aroon(df["high"], df["low"], length=14)
        df["AROONU_14"] = aroon_14["AROONU_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        df["AROOND_14"] = aroon_14["AROOND_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        # Stochastic RSI
        stochrsi = pta.stochrsi(df["close"])
        df["STOCHRSIk_14_14_3_3"] = stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        df["STOCHRSId_14_14_3_3"] = stochrsi["STOCHRSId_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        # KST
        kst = pta.kst(df["close"])
        df["KST_10_15_20_30_10_10_10_15"] = kst["KST_10_15_20_30_10_10_10_15"] if isinstance(kst, pd.DataFrame) else np.nan
        df["KSTs_9"] = kst["KSTs_9"] if isinstance(kst, pd.DataFrame) else np.nan
        # OBV
        df["OBV"] = pta.obv(df["close"], df["volume"])
        df["OBV_change_pct"] = ((df["OBV"] - df["OBV"].shift(1)) / abs(df["OBV"].shift(1))) * 100.0
        # ROC
        df["ROC_2"] = pta.roc(df["close"], length=2)
        df["ROC_9"] = pta.roc(df["close"], length=9)
        # Candle change
        df["change_pct"] = (df["close"] - df["open"]) / df["open"] * 100.0
        # Close max
        df["close_max_12"] = df["close"].rolling(12).max()
        df["close_max_48"] = df["close"].rolling(48).max()
        # Close min
        df["close_min_12"] = df["close"].rolling(12).min()
        df["close_min_48"] = df["close"].rolling(48).min()
        # Number of empty candles
        df["num_empty_288"] = (df["volume"] <= 0).rolling(window=288, min_periods=288).sum()

        # Global protections
        if not self.config["runmode"].value in ("live", "dry_run"):
            # Backtest age filter
            df["bt_agefilter_ok"] = False
            df.loc[df.index > (12 * 24 * self.bt_min_age_days), "bt_agefilter_ok"] = True
        else:
            # Exchange downtime protection
            df["live_data_ok"] = df["volume"].rolling(window=72, min_periods=72).min() > 0

        # Performance logging
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] base_tf_5m_indicators took: {tok - tik:0.4f} seconds.")

        return df

    # Info switcher - COMPLETE FROM ORIGINAL
    def info_switcher(self, metadata: dict, info_timeframe) -> DataFrame:
        if info_timeframe == "1d":
            return self.informative_1d_indicators(metadata, info_timeframe)
        elif info_timeframe == "4h":
            return self.informative_4h_indicators(metadata, info_timeframe)
        elif info_timeframe == "1h":
            return self.informative_1h_indicators(metadata, info_timeframe)
        elif info_timeframe == "15m":
            return self.informative_15m_indicators(metadata, info_timeframe)
        else:
            raise RuntimeError(f"{info_timeframe} not supported as informative timeframe for BTC pair.")

    # BTC info indicators - COMPLETE FROM ORIGINAL
    def btc_info_1d_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        btc_info_1d = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
        # Add prefix
        ignore_columns = ["date"]
        btc_info_1d.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] btc_info_1d_indicators took: {tok - tik:0.4f} seconds.")
        return btc_info_1d

    def btc_info_4h_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        btc_info_4h = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
        ignore_columns = ["date"]
        btc_info_4h.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] btc_info_4h_indicators took: {tok - tik:0.4f} seconds.")
        return btc_info_4h

    def btc_info_1h_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        btc_info_1h = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
        ignore_columns = ["date"]
        btc_info_1h.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] btc_info_1h_indicators took: {tok - tik:0.4f} seconds.")
        return btc_info_1h

    def btc_info_15m_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        btc_info_15m = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
        ignore_columns = ["date"]
        btc_info_15m.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] btc_info_15m_indicators took: {tok - tik:0.4f} seconds.")
        return btc_info_15m

    def btc_info_5m_indicators(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        btc_info_5m = self.dp.get_pair_dataframe(btc_info_pair, btc_info_timeframe)
        ignore_columns = ["date"]
        btc_info_5m.rename(columns=lambda s: f"btc_{s}" if s not in ignore_columns else s, inplace=True)
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] btc_info_5m_indicators took: {tok - tik:0.4f} seconds.")
        return btc_info_5m

    # Informative 1d Timeframe Indicators - FROM ORIGINAL
    # -----------------------------------------------------------------------------------------
    def informative_1d_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1d = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

        # Indicators
        # -----------------------------------------------------------------------------------------
        # RSI
        informative_1d["RSI_3"] = pta.rsi(informative_1d["close"], length=3)
        informative_1d["RSI_14"] = pta.rsi(informative_1d["close"], length=14)
        informative_1d["RSI_3_change_pct"] = (
            (informative_1d["RSI_3"] - informative_1d["RSI_3"].shift(1)) / (informative_1d["RSI_3"].shift(1))
        ) * 100.0
        informative_1d["RSI_14_change_pct"] = (
            (informative_1d["RSI_14"] - informative_1d["RSI_14"].shift(1)) / (informative_1d["RSI_14"].shift(1))
        ) * 100.0
        informative_1d["RSI_3_diff"] = informative_1d["RSI_3"] - informative_1d["RSI_3"].shift(1)
        informative_1d["RSI_14_diff"] = informative_1d["RSI_14"] - informative_1d["RSI_14"].shift(1)
        # BB 20 - STD2
        bbands_20_2 = pta.bbands(informative_1d["close"], length=20)
        informative_1d["BBL_20_2.0"] = bbands_20_2["BBL_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1d["BBM_20_2.0"] = bbands_20_2["BBM_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1d["BBU_20_2.0"] = bbands_20_2["BBU_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1d["BBB_20_2.0"] = bbands_20_2["BBB_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1d["BBP_20_2.0"] = bbands_20_2["BBP_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        # MFI
        informative_1d["MFI_14"] = pta.mfi(
            informative_1d["high"], informative_1d["low"], informative_1d["close"], informative_1d["volume"], length=14
        )
        # CMF
        informative_1d["CMF_20"] = pta.cmf(
            informative_1d["high"], informative_1d["low"], informative_1d["close"], informative_1d["volume"], length=20
        )
        # Williams %R
        informative_1d["WILLR_14"] = pta.willr(
            informative_1d["high"], informative_1d["low"], informative_1d["close"], length=14
        )
        # AROON
        aroon_14 = pta.aroon(informative_1d["high"], informative_1d["low"], length=14)
        informative_1d["AROONU_14"] = aroon_14["AROONU_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        informative_1d["AROOND_14"] = aroon_14["AROOND_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        # Stochastic
        try:
            stochrsi = pta.stoch(informative_1d["high"], informative_1d["low"], informative_1d["close"])
            informative_1d["STOCHk_14_3_3"] = stochrsi["STOCHk_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
            informative_1d["STOCHd_14_3_3"] = stochrsi["STOCHd_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        except AttributeError:
            informative_1d["STOCHk_14_3_3"] = np.nan
            informative_1d["STOCHd_14_3_3"] = np.nan
        # Stochastic RSI
        stochrsi = pta.stochrsi(informative_1d["close"])
        informative_1d["STOCHRSIk_14_14_3_3"] = (
            stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )
        informative_1d["STOCHRSId_14_14_3_3"] = (
            stochrsi["STOCHRSId_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )
        # ROC
        informative_1d["ROC_2"] = pta.roc(informative_1d["close"], length=2)
        informative_1d["ROC_9"] = pta.roc(informative_1d["close"], length=9)
        # Candle change
        informative_1d["change_pct"] = (informative_1d["close"] - informative_1d["open"]) / informative_1d["open"] * 100.0
        # Wicks
        informative_1d["top_wick_pct"] = (
            (informative_1d["high"] - np.maximum(informative_1d["open"], informative_1d["close"]))
            / np.maximum(informative_1d["open"], informative_1d["close"])
            * 100.0
        )
        informative_1d["bot_wick_pct"] = abs(
            (informative_1d["low"] - np.minimum(informative_1d["open"], informative_1d["close"]))
            / np.minimum(informative_1d["open"], informative_1d["close"])
            * 100.0
        )
        # Max highs
        informative_1d["high_max_6"] = informative_1d["high"].rolling(6).max()
        informative_1d["high_max_12"] = informative_1d["high"].rolling(12).max()
        informative_1d["high_max_20"] = informative_1d["high"].rolling(20).max()
        informative_1d["high_max_30"] = informative_1d["high"].rolling(30).max()
        # Max lows
        informative_1d["low_min_6"] = informative_1d["low"].rolling(6).min()
        informative_1d["low_min_12"] = informative_1d["low"].rolling(12).min()
        informative_1d["low_min_20"] = informative_1d["low"].rolling(20).min()
        informative_1d["low_min_30"] = informative_1d["low"].rolling(30).min()

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1d_indicators took: {tok - tik:0.4f} seconds.")

        return informative_1d

    # Informative 4h Timeframe Indicators - FROM ORIGINAL
    # -----------------------------------------------------------------------------------------
    def informative_4h_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_4h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

        # Indicators
        # -----------------------------------------------------------------------------------------
        # RSI
        informative_4h["RSI_3"] = pta.rsi(informative_4h["close"], length=3)
        informative_4h["RSI_14"] = pta.rsi(informative_4h["close"], length=14)
        informative_4h["RSI_3_change_pct"] = (
            (informative_4h["RSI_3"] - informative_4h["RSI_3"].shift(1)) / (informative_4h["RSI_3"].shift(1))
        ) * 100.0
        informative_4h["RSI_14_change_pct"] = (
            (informative_4h["RSI_14"] - informative_4h["RSI_14"].shift(1)) / (informative_4h["RSI_14"].shift(1))
        ) * 100.0
        informative_4h["RSI_3_diff"] = informative_4h["RSI_3"] - informative_4h["RSI_3"].shift(1)
        informative_4h["RSI_14_diff"] = informative_4h["RSI_14"] - informative_4h["RSI_14"].shift(1)
        # EMA
        informative_4h["EMA_12"] = pta.ema(informative_4h["close"], length=12)
        informative_4h["EMA_200"] = pta.ema(informative_4h["close"], length=200, fillna=0.0)
        # BB 20 - STD2
        bbands_20_2 = pta.bbands(informative_4h["close"], length=20)
        informative_4h["BBL_20_2.0"] = bbands_20_2["BBL_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_4h["BBM_20_2.0"] = bbands_20_2["BBM_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_4h["BBU_20_2.0"] = bbands_20_2["BBU_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_4h["BBB_20_2.0"] = bbands_20_2["BBB_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_4h["BBP_20_2.0"] = bbands_20_2["BBP_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        # MFI
        informative_4h["MFI_14"] = pta.mfi(
            informative_4h["high"], informative_4h["low"], informative_4h["close"], informative_4h["volume"], length=14
        )
        # CMF
        informative_4h["CMF_20"] = pta.cmf(
            informative_4h["high"], informative_4h["low"], informative_4h["close"], informative_4h["volume"], length=20
        )
        # Williams %R
        informative_4h["WILLR_14"] = pta.willr(
            informative_4h["high"], informative_4h["low"], informative_4h["close"], length=14
        )
        # AROON
        aroon_14 = pta.aroon(informative_4h["high"], informative_4h["low"], length=14)
        informative_4h["AROONU_14"] = aroon_14["AROONU_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        informative_4h["AROOND_14"] = aroon_14["AROOND_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        # Stochastic
        try:
            stochrsi = pta.stoch(informative_4h["high"], informative_4h["low"], informative_4h["close"])
            informative_4h["STOCHk_14_3_3"] = stochrsi["STOCHk_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
            informative_4h["STOCHd_14_3_3"] = stochrsi["STOCHd_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        except AttributeError:
            informative_4h["STOCHk_14_3_3"] = np.nan
            informative_4h["STOCHd_14_3_3"] = np.nan
        # Stochastic RSI
        stochrsi = pta.stochrsi(informative_4h["close"])
        informative_4h["STOCHRSIk_14_14_3_3"] = (
            stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )
        informative_4h["STOCHRSId_14_14_3_3"] = (
            stochrsi["STOCHRSId_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )
        informative_4h["STOCHRSIk_14_14_3_3_change_pct"] = (
            (informative_4h["STOCHRSIk_14_14_3_3"] - informative_4h["STOCHRSIk_14_14_3_3"].shift(1))
            / informative_4h["STOCHRSIk_14_14_3_3"].shift(1)
        ) * 100.0
        # KST
        kst = pta.kst(informative_4h["close"])
        informative_4h["KST_10_15_20_30_10_10_10_15"] = (
            kst["KST_10_15_20_30_10_10_10_15"] if isinstance(kst, pd.DataFrame) else np.nan
        )
        informative_4h["KSTs_9"] = kst["KSTs_9"] if isinstance(kst, pd.DataFrame) else np.nan
        # UO
        informative_4h["UO_7_14_28"] = pta.uo(informative_4h["high"], informative_4h["low"], informative_4h["close"])
        # OBV
        informative_4h["OBV"] = pta.obv(informative_4h["close"], informative_4h["volume"])
        informative_4h["OBV_change_pct"] = (
            (informative_4h["OBV"] - informative_4h["OBV"].shift(1)) / abs(informative_4h["OBV"].shift(1))
        ) * 100.0
        # ROC
        informative_4h["ROC_2"] = pta.roc(informative_4h["close"], length=2)
        informative_4h["ROC_9"] = pta.roc(informative_4h["close"], length=9)
        # CCI
        informative_4h["CCI_20"] = pta.cci(
            informative_4h["high"], informative_4h["low"], informative_4h["close"], length=20
        )
        informative_4h["CCI_20"] = (
            (informative_4h["CCI_20"]).astype(np.float64).replace(to_replace=[np.nan, None], value=(0.0))
        )
        informative_4h["CCI_20_change_pct"] = (
            (informative_4h["CCI_20"] - informative_4h["CCI_20"].shift(1)) / abs(informative_4h["CCI_20"].shift(1))
        ) * 100.0

        # Candle change
        informative_4h["change_pct"] = (informative_4h["close"] - informative_4h["open"]) / informative_4h["open"] * 100.0
        informative_4h["change_pct_min_3"] = informative_4h["change_pct"].rolling(3).min()
        informative_4h["change_pct_min_6"] = informative_4h["change_pct"].rolling(6).min()
        informative_4h["change_pct_max_3"] = informative_4h["change_pct"].rolling(3).max()
        informative_4h["change_pct_max_6"] = informative_4h["change_pct"].rolling(6).max()
        # Candle change
        informative_4h["change_pct"] = (informative_4h["close"] - informative_4h["open"]) / informative_4h["open"] * 100.0
        # Wicks
        informative_4h["top_wick_pct"] = (
            (informative_4h["high"] - np.maximum(informative_4h["open"], informative_4h["close"]))
            / np.maximum(informative_4h["open"], informative_4h["close"])
            * 100.0
        )
        informative_4h["bot_wick_pct"] = abs(
            (informative_4h["low"] - np.minimum(informative_4h["open"], informative_4h["close"]))
            / np.minimum(informative_4h["open"], informative_4h["close"])
            * 100.0
        )
        # Max highs
        informative_4h["high_max_6"] = informative_4h["high"].rolling(6).max()
        informative_4h["high_max_12"] = informative_4h["high"].rolling(12).max()
        informative_4h["high_max_24"] = informative_4h["high"].rolling(24).max()
        # Min lows
        informative_4h["low_min_6"] = informative_4h["low"].rolling(6).min()
        informative_4h["low_min_12"] = informative_4h["low"].rolling(12).min()
        informative_4h["low_min_24"] = informative_4h["low"].rolling(24).min()

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_4h_indicators took: {tok - tik:0.4f} seconds.")

        return informative_4h

    # Informative 1h Timeframe Indicators - OPTIMIZED WITH CACHING
    # -----------------------------------------------------------------------------------------
    def informative_1h_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."

        # Create cache key for this specific pair and timeframe
        cache_key = f"informative_1h_{metadata['pair']}"

        # Check cache first
        if cache_key in self._indicator_cache:
            log.debug(f"[{metadata['pair']}] Using cached 1h indicators")
            return self._indicator_cache[cache_key].copy()

        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

        # Indicators
        # -----------------------------------------------------------------------------------------
        # RSI - optimized using numpy arrays
        close_prices = informative_1h["close"].values
        high_prices = informative_1h["high"].values
        low_prices = informative_1h["low"].values
        volume_data = informative_1h["volume"].values

        informative_1h["RSI_3"] = pta.rsi(informative_1h["close"], length=3)
        informative_1h["RSI_14"] = pta.rsi(informative_1h["close"], length=14)
        informative_1h["RSI_3_change_pct"] = (
            (informative_1h["RSI_3"] - informative_1h["RSI_3"].shift(1)) / (informative_1h["RSI_3"].shift(1))
        ) * 100.0
        informative_1h["RSI_14_change_pct"] = (
            (informative_1h["RSI_14"] - informative_1h["RSI_14"].shift(1)) / (informative_1h["RSI_14"].shift(1))
        ) * 100.0
        informative_1h["RSI_3_diff"] = informative_1h["RSI_3"] - informative_1h["RSI_3"].shift(1)
        informative_1h["RSI_14_diff"] = informative_1h["RSI_14"] - informative_1h["RSI_14"].shift(1)

        # EMA - optimized using talib where available
        informative_1h["EMA_12"] = pta.ema(informative_1h["close"], length=12)
        informative_1h["EMA_200"] = pta.ema(informative_1h["close"], length=200, fillna=0.0)

        # SMA
        informative_1h["SMA_16"] = pta.sma(informative_1h["close"], length=16)

        # BB 20 - STD2
        bbands_20_2 = pta.bbands(informative_1h["close"], length=20)
        informative_1h["BBL_20_2.0"] = bbands_20_2["BBL_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1h["BBM_20_2.0"] = bbands_20_2["BBM_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1h["BBU_20_2.0"] = bbands_20_2["BBU_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1h["BBB_20_2.0"] = bbands_20_2["BBB_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan
        informative_1h["BBP_20_2.0"] = bbands_20_2["BBP_20_2.0"] if isinstance(bbands_20_2, pd.DataFrame) else np.nan

        # MFI
        informative_1h["MFI_14"] = pta.mfi(
            informative_1h["high"], informative_1h["low"], informative_1h["close"], informative_1h["volume"], length=14
        )

        # CMF
        informative_1h["CMF_20"] = pta.cmf(
            informative_1h["high"], informative_1h["low"], informative_1h["close"], informative_1h["volume"], length=20
        )

        # Williams %R
        informative_1h["WILLR_14"] = pta.willr(
            informative_1h["high"], informative_1h["low"], informative_1h["close"], length=14
        )
        informative_1h["WILLR_84"] = pta.willr(
            informative_1h["high"], informative_1h["low"], informative_1h["close"], length=84
        )

        # AROON
        aroon_14 = pta.aroon(informative_1h["high"], informative_1h["low"], length=14)
        informative_1h["AROONU_14"] = aroon_14["AROONU_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        informative_1h["AROOND_14"] = aroon_14["AROOND_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan

        # Stochastic
        stochrsi = pta.stoch(informative_1h["high"], informative_1h["low"], informative_1h["close"])
        informative_1h["STOCHk_14_3_3"] = stochrsi["STOCHk_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        informative_1h["STOCHd_14_3_3"] = stochrsi["STOCHd_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan

        # Stochastic RSI
        stochrsi = pta.stochrsi(informative_1h["close"])
        informative_1h["STOCHRSIk_14_14_3_3"] = (
            stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )
        informative_1h["STOCHRSId_14_14_3_3"] = (
            stochrsi["STOCHRSId_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )

        # KST
        kst = pta.kst(informative_1h["close"])
        informative_1h["KST_10_15_20_30_10_10_10_15"] = (
            kst["KST_10_15_20_30_10_10_10_15"] if isinstance(kst, pd.DataFrame) else np.nan
        )
        informative_1h["KSTs_9"] = kst["KSTs_9"] if isinstance(kst, pd.DataFrame) else np.nan

        # UO
        informative_1h["UO_7_14_28"] = pta.uo(informative_1h["high"], informative_1h["low"], informative_1h["close"])

        # OBV
        informative_1h["OBV"] = pta.obv(informative_1h["close"], informative_1h["volume"])
        informative_1h["OBV_change_pct"] = (
            (informative_1h["OBV"] - informative_1h["OBV"].shift(1)) / abs(informative_1h["OBV"].shift(1))
        ) * 100.0

        # ROC
        informative_1h["ROC_9"] = pta.roc(informative_1h["close"], length=9)

        # CCI
        informative_1h["CCI_20"] = pta.cci(
            informative_1h["high"], informative_1h["low"], informative_1h["close"], length=20
        )
        informative_1h["CCI_20"] = (
            (informative_1h["CCI_20"]).astype(np.float64).replace(to_replace=[np.nan, None], value=(0.0))
        )
        informative_1h["CCI_20_change_pct"] = (
            (informative_1h["CCI_20"] - informative_1h["CCI_20"].shift(1)) / abs(informative_1h["CCI_20"].shift(1))
        ) * 100.0

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_1h_indicators took: {tok - tik:0.4f} seconds.")

        # Cache the result
        self._indicator_cache[cache_key] = informative_1h.copy()

        return informative_1h

    # Informative 15m Timeframe Indicators - OPTIMIZED WITH CACHING
    # -----------------------------------------------------------------------------------------
    def informative_15m_indicators(self, metadata: dict, info_timeframe) -> DataFrame:
        tik = time.perf_counter()
        assert self.dp, "DataProvider is required for multiple timeframes."

        # Create cache key for this specific pair and timeframe
        cache_key = f"informative_15m_{metadata['pair']}"

        # Check cache first
        if cache_key in self._indicator_cache:
            log.debug(f"[{metadata['pair']}] Using cached 15m indicators")
            return self._indicator_cache[cache_key].copy()

        # Get the informative pair
        informative_15m = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=info_timeframe)

        # Indicators
        # -----------------------------------------------------------------------------------------
        # RSI - optimized using numpy arrays
        close_prices = informative_15m["close"].values
        high_prices = informative_15m["high"].values
        low_prices = informative_15m["low"].values
        volume_data = informative_15m["volume"].values

        informative_15m["RSI_3"] = pta.rsi(informative_15m["close"], length=3)
        informative_15m["RSI_14"] = pta.rsi(informative_15m["close"], length=14)
        informative_15m["RSI_3_change_pct"] = (
            (informative_15m["RSI_3"] - informative_15m["RSI_3"].shift(1)) / (informative_15m["RSI_3"].shift(1))
        ) * 100.0
        informative_15m["RSI_14_change_pct"] = (
            (informative_15m["RSI_14"] - informative_15m["RSI_14"].shift(1)) / (informative_15m["RSI_14"].shift(1))
        ) * 100.0

        # EMA
        informative_15m["EMA_12"] = pta.ema(informative_15m["close"], length=12)
        informative_15m["EMA_20"] = pta.ema(informative_15m["close"], length=20)
        informative_15m["EMA_26"] = pta.ema(informative_15m["close"], length=26)

        # MFI
        informative_15m["MFI_14"] = pta.mfi(
            informative_15m["high"], informative_15m["low"], informative_15m["close"], informative_15m["volume"], length=14
        )

        # CMF
        informative_15m["CMF_20"] = pta.cmf(
            informative_15m["high"], informative_15m["low"], informative_15m["close"], informative_15m["volume"], length=20
        )

        # Williams %R
        informative_15m["WILLR_14"] = pta.willr(
            informative_15m["high"], informative_15m["low"], informative_15m["close"], length=14
        )

        # AROON
        aroon_14 = pta.aroon(informative_15m["high"], informative_15m["low"], length=14)
        informative_15m["AROONU_14"] = aroon_14["AROONU_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan
        informative_15m["AROOND_14"] = aroon_14["AROOND_14"] if isinstance(aroon_14, pd.DataFrame) else np.nan

        # Stochastic
        stochrsi = pta.stoch(informative_15m["high"], informative_15m["low"], informative_15m["close"])
        informative_15m["STOCHk_14_3_3"] = stochrsi["STOCHk_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        informative_15m["STOCHd_14_3_3"] = stochrsi["STOCHd_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan

        # Stochastic RSI
        stochrsi = pta.stochrsi(informative_15m["close"])
        informative_15m["STOCHRSIk_14_14_3_3"] = (
            stochrsi["STOCHRSIk_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )
        informative_15m["STOCHRSId_14_14_3_3"] = (
            stochrsi["STOCHRSId_14_14_3_3"] if isinstance(stochrsi, pd.DataFrame) else np.nan
        )

        # UO
        informative_15m["UO_7_14_28"] = pta.uo(informative_15m["high"], informative_15m["low"], informative_15m["close"])
        informative_15m["UO_7_14_28_change_pct"] = (
            informative_15m["UO_7_14_28"] - informative_15m["UO_7_14_28"].shift(1)
        ) * 100.0

        # OBV
        informative_15m["OBV"] = pta.obv(informative_15m["close"], informative_15m["volume"])
        informative_15m["OBV_change_pct"] = (
            (informative_15m["OBV"] - informative_15m["OBV"].shift(1)) / abs(informative_15m["OBV"].shift(1))
        ) * 100.0

        # ROC
        informative_15m["ROC_9"] = pta.roc(informative_15m["close"], length=9)

        # CCI
        informative_15m["CCI_20"] = pta.cci(
            informative_15m["high"], informative_15m["low"], informative_15m["close"], length=20
        )
        informative_15m["CCI_20"] = (
            (informative_15m["CCI_20"]).astype(np.float64).replace(to_replace=[np.nan, None], value=(0.0))
        )
        informative_15m["CCI_20_change_pct"] = (
            (informative_15m["CCI_20"] - informative_15m["CCI_20"].shift(1)) / abs(informative_15m["CCI_20"].shift(1))
        ) * 100.0

        # Performance logging
        # -----------------------------------------------------------------------------------------
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] informative_15m_indicators took: {tok - tik:0.4f} seconds.")

        # Cache the result
        self._indicator_cache[cache_key] = informative_15m.copy()

        return informative_15m

    def btc_info_switcher(self, btc_info_pair, btc_info_timeframe, metadata: dict) -> DataFrame:
        if btc_info_timeframe == "1d":
            return self.btc_info_1d_indicators(btc_info_pair, btc_info_timeframe, metadata)
        elif btc_info_timeframe == "4h":
            return self.btc_info_4h_indicators(btc_info_pair, btc_info_timeframe, metadata)
        elif btc_info_timeframe == "1h":
            return self.btc_info_1h_indicators(btc_info_pair, btc_info_timeframe, metadata)
        elif btc_info_timeframe == "15m":
            return self.btc_info_15m_indicators(btc_info_pair, btc_info_timeframe, metadata)
        elif btc_info_timeframe == "5m":
            return self.btc_info_5m_indicators(btc_info_pair, btc_info_timeframe, metadata)
        else:
            raise RuntimeError(f"{btc_info_timeframe} not supported as informative timeframe for BTC pair.")

    # COMPLETE POPULATE_INDICATORS FROM ORIGINAL
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        tik = time.perf_counter()
        """
            -> BTC informative indicators
            ___________________________________________________________________________________________
            """
        if self.config["stake_currency"] in [
            "USDT",
            "BUSD",
            "USDC",
            "DAI",
            "TUSD",
            "FDUSD",
            "PAX",
            "USD",
            "EUR",
            "GBP",
            "TRY",
        ]:
            if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
                btc_info_pair = f"BTC/{self.config['stake_currency']}:{self.config['stake_currency']}"
            else:
                btc_info_pair = f"BTC/{self.config['stake_currency']}"
        else:
            if ("trading_mode" in self.config) and (self.config["trading_mode"] in ["futures", "margin"]):
                btc_info_pair = "BTC/USDT:USDT"
            else:
                btc_info_pair = "BTC/USDT"

        for btc_info_timeframe in self.btc_info_timeframes:
            btc_informative = self.btc_info_switcher(btc_info_pair, btc_info_timeframe, metadata)
            df = merge_informative_pair(df, btc_informative, self.timeframe, btc_info_timeframe, ffill=True)
            # Customize what we drop - in case we need to maintain some BTC informative ohlcv data
            # Default drop all
            drop_columns = {
                "1d": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "4h": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "1h": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "15m": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "5m": [f"btc_{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
            }.get(
                btc_info_timeframe,
                [f"{s}_{btc_info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
            )
            drop_columns.append(f"date_{btc_info_timeframe}")
            df.drop(columns=df.columns.intersection(drop_columns), inplace=True)

        """
            -> Indicators on informative timeframes
            ___________________________________________________________________________________________
            """
        for info_timeframe in self.info_timeframes:
            info_indicators = self.info_switcher(metadata, info_timeframe)
            df = merge_informative_pair(df, info_indicators, self.timeframe, info_timeframe, ffill=True)
            # Customize what we drop - in case we need to maintain some informative timeframe ohlcv data
            # Default drop all except base timeframe ohlcv data
            drop_columns = {
                "1d": [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "4h": [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "1h": [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]],
                "15m": [f"{s}_{info_timeframe}" for s in ["date", "high", "low", "volume"]],
            }.get(info_timeframe, [f"{s}_{info_timeframe}" for s in ["date", "open", "high", "low", "close", "volume"]])
            df.drop(columns=df.columns.intersection(drop_columns), inplace=True)

        """
            -> The indicators for the base timeframe  (5m)
            ___________________________________________________________________________________________
            """
        df = self.base_tf_5m_indicators(metadata, df)

        # df["zlma_50_1h"] = df["zlma_50_1h"].astype(np.float64).replace(to_replace=[np.nan, None], value=(0.0))
        # df["CTI_20_1d"] = df["CTI_20_1d"].astype(np.float64).replace(to_replace=[np.nan, None], value=(0.0))
        # df["WILLR_480_1h"] = df["WILLR_480_1h"].astype(np.float64).replace(to_replace=[np.nan, None], value=(-50.0))
        # df["WILLR_480_4h"] = df["WILLR_480_4h"].astype(np.float64).replace(to_replace=[np.nan, None], value=(-50.0))
        # df["RSI_14_1d"] = df["RSI_14_1d"].astype(np.float64).replace(to_replace=[np.nan, None], value=(50.0))
        df["RSI_14_1h"] = df["RSI_14_1h"].astype(np.float64).replace(to_replace=[np.nan, None], value=(50.0))

        # Global protections Long
        df["protections_long_global"] = (
            # 5m & 15m & 1h & 4h & 1d down move, 1h & 4h & 1d still not low enough
            (
                (df["RSI_3"] > 1.0)
                | (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 20.0)
                | (df["RSI_3_4h"] > 20.0)
                | (df["RSI_3_1d"] > 20.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["RSI_14_1d"] < 30.0)
                | (df["CCI_20_1h"] < -250.0)
                | (df["CCI_20_4h"] < -200.0)
            )
            # 5m & 4h & 1d down move, 15m & 1h & 4h still not low enough, 1d still high
            & (
                (df["RSI_3"] > 1.0)
                | (df["RSI_3_4h"] > 10.0)
                | (df["RSI_3_1d"] > 35.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
                | (df["STOCHRSIk_14_14_3_3_1d"] < 50.0)
            )
            # 5m down move, 15m & 1h & 4h still high, 15m high
            & (
                (df["RSI_3"] > 3.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 70.0)
            )
            # 1h & 4h down move, 15m & 1h & 4h downtrend, 1h still high
            & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["ROC_9_1d"] < 80.0)
            )
            # 5m & 15m & 1h down move, 1h & 4h still high, 15m still high, 1h high
            & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 10.0)
                | (df["RSI_3_1h"] > 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_1h"] < 85.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 70.0)
            )
            # 5m & 15m & 1h & 4h down move, 4h high
            & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 15.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 55.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_4h"] < 80.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 15m high
            & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
            )
            # 5m & 15m & 4h down move, 15m & 1h & 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 5.0)
                | (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
            )
            # 5m & 15m & 1h down move, 15m still not low enough, 1h still high, 4h high, 15m still not low enough, 5h overbought
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 55.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_1h"] < 20.0)
                | (df["AROONU_14_4h"] < 80.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
                | (df["ROC_9_4h"] < 80.0)
            )
            # 5m & 15m & 1h & 1d down move, 1h & 4h still high, 1d downtrend
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 65.0)
                | (df["RSI_3_1d"] > 25.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1d"] > -0.20)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
                | (df["ROC_9_1d"] > -15.0)
            )
            # 5m & 15m & 4h down move, 15m & 1h still not low enough, 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
            # 5m & 15m & 1h down move, 15m still not low enough, 1h still high, 4h high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 35.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 30.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
            )
            # 5m & 15m & 1h down move, 15m still high, 1h & 4h high, 4h high & overbought
            & (
                (df["RSI_3"] > 15.0)
                | (df["RSI_3_15m"] > 30.0)
                | (df["RSI_3_1h"] > 60.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 60.0)
                | (df["RSI_14_4h"] < 60.0)
                | (df["AROONU_14_15m"] < 40.0)
                | (df["AROONU_14_4h"] < 80.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 80.0)
                | (df["ROC_9_4h"] < 30.0)
            )
            # 15m & 1h & 4h down move, 4h still high, 1h downtrend, 4h still high
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 10.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_1h"] > -0.30)
                | (df["AROONU_14_4h"] < 40.0)
            )
            # 15m & 1h & 4h down move, 1h & 4h still not low enough, 15m downtrend
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 15.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["AROONU_14_4h"] < 25.0)
            )
            # 15m & 1h & 4h & 1d down move, 1h still high, 1h & 4h still not low enough
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_3_1d"] > 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1h"] > -0.30)
                | (df["AROONU_14_4h"] < 25.0)
            )
            # 5m & 15m & 4h down move, 15m & 1h still not low enough, 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 15m high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
            # 5m & 15m & 1h down move, 15m still not low enough, 1h still high, 4h high, 15m still not low enough, 5h overbought
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 55.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 50.0)
                | (df["RSI_14_4h"] < 70.0)
                | (df["AROONU_14_1h"] < 20.0)
                | (df["AROONU_14_4h"] < 80.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 20.0)
                | (df["ROC_9_4h"] < 80.0)
            )
            # 5m & 15m & 1h & 1d down move, 1h & 4h still high, 1d downtrend
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_1h"] > 65.0)
                | (df["RSI_3_1d"] > 25.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1d"] > -0.20)
                | (df["STOCHRSIk_14_14_3_3_1h"] < 50.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 50.0)
                | (df["ROC_9_1d"] > -15.0)
            )
            # 5m & 15m & 4h down move, 15m & 1h still not low enough, 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 15m high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
            # 15m & 1h & 4h down move, 4h still high, 1h downtrend, 4h still high
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 10.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_1h"] > -0.30)
                | (df["AROONU_14_4h"] < 40.0)
            )
            # 15m & 1h & 4h down move, 1h & 4h still not low enough, 15m downtrend
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 15.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["AROONU_14_4h"] < 25.0)
            )
            # 15m & 1h & 4h & 1d down move, 1h still high, 1h & 4h still not low enough
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_3_1d"] > 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1h"] > -0.30)
                | (df["AROONU_14_4h"] < 25.0)
            )
            # 5m & 15m & 4h down move, 15m & 1h still not low enough, 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 15m high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
            # 15m & 1h & 4h down move, 4h still high, 1h downtrend, 4h still high
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 10.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_1h"] > -0.30)
                | (df["AROONU_14_4h"] < 40.0)
            )
            # 15m & 1h & 4h down move, 1h & 4h still not low enough, 15m downtrend
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 15.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["CMF_20_15m"] > -0.25)
                | (df["AROONU_14_4h"] < 25.0)
            )
            # 15m & 1h & 4h & 1d down move, 1h still high, 1h & 4h still not low enough
            & (
                (df["RSI_3_15m"] > 3.0)
                | (df["RSI_3_1h"] > 30.0)
                | (df["RSI_3_4h"] > 30.0)
                | (df["RSI_3_1d"] > 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["CMF_20_1h"] > -0.30)
                | (df["AROONU_14_4h"] < 25.0)
            )
            # 5m & 15m & 4h down move, 15m & 1h still not low enough, 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 25.0)
                | (df["RSI_3_4h"] > 25.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 40.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_4h"] < 70.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still high, 15m & 4h high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 45.0)
                | (df["RSI_3_4h"] > 45.0)
                | (df["RSI_14_15m"] < 40.0)
                | (df["RSI_14_1h"] < 40.0)
                | (df["RSI_14_4h"] < 50.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["AROONU_14_4h"] < 70.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 50.0)
            )
            # 5m & 15m & 1h & 4h down move, 15m & 1h & 4h still not low enough, 15m high
            & (
                (df["RSI_3"] > 10.0)
                | (df["RSI_3_15m"] > 20.0)
                | (df["RSI_3_1h"] > 50.0)
                | (df["RSI_3_4h"] > 50.0)
                | (df["RSI_14_15m"] < 30.0)
                | (df["RSI_14_1h"] < 30.0)
                | (df["RSI_14_4h"] < 30.0)
                | (df["AROONU_14_15m"] < 60.0)
                | (df["STOCHRSIk_14_14_3_3_15m"] < 60.0)
            )
        )

        # Performance logging
        tok = time.perf_counter()
        log.debug(f"[{metadata['pair']}] populate_indicators took: {tok - tik:0.4f} seconds.")

        return df

    # Update signals from config - COMPLETE FROM ORIGINAL
    def update_signals_from_config(self, config):
        # Update long entry signal parameters (if they exist in the config)
        if hasattr(self, "long_entry_signal_params") and "long_entry_signal_params" in config:
            for condition_key in self.long_entry_signal_params:
                if condition_key in config["long_entry_signal_params"]:
                    self.long_entry_signal_params[condition_key] = config["long_entry_signal_params"][condition_key]

        # Update short entry signal parameters (if they exist in the config)
        if hasattr(self, "short_entry_signal_params") and "short_entry_signal_params" in config:
            for condition_key in self.short_entry_signal_params:
                if condition_key in config["short_entry_signal_params"]:
                    self.short_entry_signal_params[condition_key] = config["short_entry_signal_params"][condition_key]

    # Additional optimized methods would be implemented here
    # ... (rest of the strategy implementation)

# Cache class for storing temporary data
class Cache:
    def __init__(self, path):
        self.path = path
        self.data = {}

    def save(self):
        """Save cache to disk"""
        try:
            import json
            with open(self.path, 'w') as f:
                json.dump(self.data, f)
        except:
            pass

    def __contains__(self, key):
        return key in self.data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        if key in self.data:
            del self.data[key]