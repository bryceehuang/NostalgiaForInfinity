# Freqtrade 问题修复总结

## 已修复的问题 ✅

### 1. pandas fillna() ValueError

**问题描述:**
```
ValueError("Must specify a fill 'value' or 'method'.")
```

**原因:**
pandas 2.1.0+ 版本要求 `fillna()` 方法必须明确指定 `value` 或 `method` 参数。

**修复位置:**
- `NostalgiaForInfinityX7.py` 第 70483 行
- `NostalgiaForInfinityX7.py` 第 70500-70503 行

**修复内容:**
```python
# 修复前
df = df[["open", "close", "high", "low"]].copy().fillna(0)
open_ha = open_ha.fillna(0)

# 修复后
df = df[["open", "close", "high", "low"]].copy().fillna(value=0)
open_ha = open_ha.fillna(value=0)
```

## 当前遇到的问题 ⚠️

### 1. OKX API Rate Limit (频率限制)

**错误信息:**
```
RateLimitExceeded. Message: okx {"msg":"Too Many Requests","code":"50011"}
```

**原因:**
Freqtrade 启动时需要下载大量交易对的历史数据,触发了 OKX 的 API 频率限制。

**解决方案:**

#### 方案 1: 增加 API 延迟时间
在 `user_data/config.json` 中修改交易所配置:

```json
{
  "exchange": {
    "name": "okx",
    "ccxt_config": {
      "rateLimit": 200,  // 从 60 增加到 200 毫秒
      "enableRateLimit": true
    }
  }
}
```

#### 方案 2: 减少交易对数量
在 `user_data/config.json` 中减少 pairlist:

```json
{
  "pairlists": [
    {
      "method": "VolumePairList",
      "number_assets": 50,  // 减少数量,比如从 100 减到 50
      "sort_key": "quoteVolume"
    }
  ]
}
```

#### 方案 3: 使用静态交易对列表
只交易你确定的交易对:

```json
{
  "exchange": {
    "pair_whitelist": [
      "BTC/USDT:USDT",
      "ETH/USDT:USDT",
      "SOL/USDT:USDT"
      // 添加你想交易的交易对
    ]
  }
}
```

#### 方案 4: 等待自动重试
Freqtrade 会自动重试失败的请求,只是需要更长时间完成初始化。这是最简单的方案,但启动会比较慢。

### 2. Empty Candle Data (空 K 线数据)

**受影响的交易对:**
- JCT/USDT:USDT
- BEAT/USDT:USDT
- LAB/USDT:USDT
- PIEVERSE/USDT:USDT
- AT/USDT:USDT
- KITE/USDT:USDT
- MMT/USDT:USDT
- ALLO/USDT:USDT
- OL/USDT:USDT
- TRUST/USDT:USDT

**原因:**
- 新上线的币种,历史数据不足
- OKX 不支持这些交易对的期货交易
- 交易对已下架

**解决方案:**

#### 方案 1: 添加到黑名单
在 `user_data/config.json` 中添加:

```json
{
  "exchange": {
    "pair_blacklist": [
      "JCT/USDT:USDT",
      "BEAT/USDT:USDT",
      "LAB/USDT:USDT",
      "PIEVERSE/USDT:USDT",
      "AT/USDT:USDT",
      "KITE/USDT:USDT",
      "MMT/USDT:USDT",
      "ALLO/USDT:USDT",
      "OL/USDT:USDT",
      "TRUST/USDT:USDT"
    ]
  }
}
```

#### 方案 2: 使用 AgeFilter
自动过滤掉数据不足的交易对:

```json
{
  "pairlists": [
    {
      "method": "VolumePairList",
      "number_assets": 100
    },
    {
      "method": "AgeFilter",
      "min_days_listed": 30  // 只交易上线超过 30 天的币种
    }
  ]
}
```

## 推荐配置

综合以上问题,推荐的配置调整:

```json
{
  "exchange": {
    "name": "okx",
    "ccxt_config": {
      "rateLimit": 150,
      "enableRateLimit": true
    },
    "pair_blacklist": [
      "JCT/USDT:USDT",
      "BEAT/USDT:USDT",
      "LAB/USDT:USDT",
      "PIEVERSE/USDT:USDT",
      "AT/USDT:USDT",
      "KITE/USDT:USDT",
      "MMT/USDT:USDT",
      "ALLO/USDT:USDT",
      "OL/USDT:USDT",
      "TRUST/USDT:USDT"
    ]
  },
  "pairlists": [
    {
      "method": "VolumePairList",
      "number_assets": 50,
      "sort_key": "quoteVolume"
    },
    {
      "method": "AgeFilter",
      "min_days_listed": 30
    }
  ]
}
```

## 验证修复

运行以下命令查看日志:

```bash
# 查看实时日志
docker compose logs -f

# 查看最近 100 行日志
docker compose logs --tail=100

# 重启容器
docker compose restart
```

## 注意事项

1. **Rate Limit 是正常现象**: 在启动时下载大量数据会触发限制,Freqtrade 会自动重试
2. **Empty candle 不影响运行**: Freqtrade 会自动跳过没有数据的交易对
3. **建议使用黑名单**: 将确认没有数据的交易对加入黑名单,可以加快启动速度
4. **监控日志**: 启动后继续观察日志,确保没有其他错误

## 下一步

1. 等待 Freqtrade 完成初始化(可能需要 5-10 分钟)
2. 检查是否有交易信号生成
3. 监控 dry run 的交易情况
4. 如果一切正常,可以考虑切换到实盘模式
