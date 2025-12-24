#!/bin/bash
set -euo pipefail

# Usage: ./download-whitelist-data.sh <DAYS> [path_to_downloader]
# Example: ./download-whitelist-data.sh 365 ./download-futures-data.sh

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <DAYS> [download_script_path]"
  echo "Example: $0 365 ./download-futures-data.sh"
  exit 1
fi

DAYS="$1"
DOWNLOADER="${2:-./download-futures-data.sh}"

if [ ! -x "$DOWNLOADER" ]; then
  echo "ERROR: downloader script not found or not executable: $DOWNLOADER"
  echo "Fix: chmod +x $DOWNLOADER"
  exit 1
fi

# Pairs copied from your log output
PAIRS=(
  "BTC/USDT:USDT" "ETH/USDT:USDT" "SOL/USDT:USDT" "ZEC/USDT:USDT" "XRP/USDT:USDT" "SOPH/USDT:USDT"
  "DOGE/USDT:USDT" "BCH/USDT:USDT" "SUI/USDT:USDT" "ASTER/USDT:USDT" "HYPE/USDT:USDT" "ANIME/USDT:USDT"
  "ADA/USDT:USDT" "AVAX/USDT:USDT" "UNI/USDT:USDT" "LINK/USDT:USDT" "ENA/USDT:USDT"
  "FARTCOIN/USDT:USDT" "AAVE/USDT:USDT" "FIL/USDT:USDT" "APT/USDT:USDT" "TAO/USDT:USDT" "NEAR/USDT:USDT"
  "ICNT/USDT:USDT" "WIF/USDT:USDT" "PUMP/USDT:USDT" "LTC/USDT:USDT" "SOON/USDT:USDT"
  "XPL/USDT:USDT" "DOT/USDT:USDT" "PENGU/USDT:USDT" "ARB/USDT:USDT" "WLD/USDT:USDT" "F/USDT:USDT"
  "CLO/USDT:USDT" "CRV/USDT:USDT" "LUNA2/USDT:USDT" "XLM/USDT:USDT" "TIA/USDT:USDT"
  "PEOPLE/USDT:USDT" "OP/USDT:USDT" "HBAR/USDT:USDT" "DASH/USDT:USDT" "VIRTUAL/USDT:USDT"
  "MON/USDT:USDT" "SEI/USDT:USDT" "NOM/USDT:USDT" "ICP/USDT:USDT" "SSV/USDT:USDT" "FET/USDT:USDT"
  "LDO/USDT:USDT" "ETC/USDT:USDT" "INJ/USDT:USDT" "PENDLE/USDT:USDT" "ZEN/USDT:USDT" "EVAA/USDT:USDT"
  "TURBO/USDT:USDT" "TON/USDT:USDT" "HMSTR/USDT:USDT" "EIGEN/USDT:USDT"
  "ATOM/USDT:USDT" "SOMI/USDT:USDT" "STRK/USDT:USDT" "GALA/USDT:USDT" "ETHFI/USDT:USDT"
  "LRC/USDT:USDT" "VVV/USDT:USDT" "MOODENG/USDT:USDT" "RENDER/USDT:USDT"
  "TNSR/USDT:USDT" "DOOD/USDT:USDT" "BOME/USDT:USDT" "ZORA/USDT:USDT" "BARD/USDT:USDT"
  "AVNT/USDT:USDT" "ALCH/USDT:USDT" "ZRC/USDT:USDT" "AXL/USDT:USDT" "ALGO/USDT:USDT"
)

# If you want to allow non-ascii pairs, set ALLOW_NON_ASCII=1
ALLOW_NON_ASCII="${ALLOW_NON_ASCII:-0}"

log() {
  echo "[$(date -Is)] $*"
}

is_ascii() {
  LC_ALL=C grep -q '^[ -~]*$' <<<"$1"
}

total="${#PAIRS[@]}"
ok=0
fail=0
skip=0

log "Starting whitelist download: pairs=$total days=$DAYS downloader=$DOWNLOADER"

for i in "${!PAIRS[@]}"; do
  pair="${PAIRS[$i]}"
  n=$((i+1))

  if [ "$ALLOW_NON_ASCII" -ne 1 ] && ! is_ascii "$pair"; then
    log "SKIP ($n/$total): non-ascii pair: $pair  (set ALLOW_NON_ASCII=1 to include)"
    skip=$((skip+1))
    continue
  fi

  log "RUN  ($n/$total): $pair (days=$DAYS)"
  if "$DOWNLOADER" "$pair" "$DAYS"; then
    ok=$((ok+1))
    log "OK   ($n/$total): $pair"
  else
    fail=$((fail+1))
    log "FAIL ($n/$total): $pair (continuing)"
  fi
done

log "Done. ok=$ok fail=$fail skip=$skip total=$total"
exit 0
