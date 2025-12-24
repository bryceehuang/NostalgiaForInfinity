#!/bin/bash
set -euo pipefail

# -------------------------
# Usage check
# -------------------------
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <PAIR> <DAYS>"
  echo "Example: $0 RIVER/USDT:USDT 365"
  exit 1
fi

PAIR="$1"
DAYS="$2"

EXCHANGE="binance"
TRADING_MODE="futures"
TIMEFRAMES=("5m" "15m" "1h" "4h" "1d")

echo "Downloading data for:"
echo "  Pair:        $PAIR"
echo "  Days:        $DAYS"
echo "  Exchange:    $EXCHANGE"
echo "  TradingMode: $TRADING_MODE"
echo

# -------------------------
# Download loop
# -------------------------
for TF in "${TIMEFRAMES[@]}"; do
  echo ">>> Downloading timeframe: $TF"
  freqtrade download-data \
    --exchange "$EXCHANGE" \
    --trading-mode "$TRADING_MODE" \
    --timeframe "$TF" \
    --pairs "$PAIR" \
    --days "$DAYS" \
    --prepend
done

echo
echo "✅ Data download completed successfully."
