#!/bin/bash

# Freqtrade Binance Futures Historical Data Download Script (Optimized)
# Uses official freqtrade download-data command to download multiple timeframes in a single call
# Date Range: 2021-01-01 to 2023-12-31
# Timeframes: 5m, 15m, 1h, 4h, 1d
# File naming: SYMBOL_USDT_USDT-TIMEFRAME-futures.feather

set -e  # Exit immediately on error

# ==================== USER CONFIGURATION ====================
# Set these variables according to your needs

# Path to your pairs.json file
PAIRS_FILE="pairs2021.json"

# Directory where flattened Feather files will be saved
FLAT_DATADIR="user_data/data/binance/futures-2021"

# Temporary directory for standard freqtrade download
TEMP_DATADIR="./temp_freqtrade_data"

# Freqtrade configuration
EXCHANGE="binance"
TRADING_MODE="futures"
TIMERANGE="20210101-20231231"
# 关键修改：使用逗号分隔的时间框架列表，确保一次性下载
TIMEFRAMES="5m, 15m, 1h, 4h, 1d"
DATA_FORMAT="feather"

# ==================== SCRIPT MAIN ====================

echo "=== Binance Futures Historical Data Downloader (Multi-Timeframe) ==="
echo "Using official freqtrade download-data command"
echo "Starting at: $(date)"
echo ""

# Display configuration
echo "Configuration:"
echo "✓ Pairs file: $PAIRS_FILE"
echo "✓ Flat output directory: $FLAT_DATADIR"
echo "✓ Temporary download directory: $TEMP_DATADIR"
echo "✓ Exchange: $EXCHANGE"
echo "✓ Trading mode: $TRADING_MODE"
echo "✓ Date range: $TIMERANGE"
echo "✓ All timeframes: $TIMEFRAMES"
echo "✓ Data format: $DATA_FORMAT"
echo ""

# Validate inputs and check dependencies
validate_environment() {
    echo "=== Environment Validation ==="
    
    # Check if pairs file exists
    if [[ ! -f "$PAIRS_FILE" ]]; then
        echo "Error: Pairs JSON file not found: $PAIRS_FILE"
        exit 1
    fi
    
    # Check if jq is available for JSON parsing
    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed."
        echo "Install with: sudo apt-get install jq or brew install jq"
        exit 1
    fi
    
    # Create directories
    mkdir -p "$FLAT_DATADIR"
    mkdir -p "$TEMP_DATADIR"
    
    # Count trading pairs
    local pair_count=$(jq 'length' "$PAIRS_FILE")
    echo "✓ Valid JSON file with $pair_count trading pairs"
    echo "✓ Output directory created: $FLAT_DATADIR"
    echo "✓ Temporary directory created: $TEMP_DATADIR"
    echo "✓ Environment validation passed"
    echo ""
}

# 修改后的下载函数 - 使用循环
download_all_timeframes() {
    echo "=== Downloading ALL Timeframes Data with Freqtrade ==="
    echo "Start time: $(date)"
    
    # 定义要下载的时间框架列表
    local timeframes=("5m" "15m" "1h" "4h" "1d")
    
    # 循环下载每个时间框架
    for tf in "${timeframes[@]}"; do
        echo "--- Downloading $tf data ---"
        freqtrade download-data \
            --exchange "$EXCHANGE" \
            --trading-mode "$TRADING_MODE" \
            --timeframes "$tf" \
            --pairs-file "$PAIRS_FILE" \
            --datadir "$TEMP_DATADIR" \
            --timerange "$TIMERANGE" \
            --data-format-ohlcv "$DATA_FORMAT"
        
        # 可选：在每个时间框架下载后添加短暂延迟，避免向交易所API发送过多请求
        sleep 2
    done
    
    echo "Completion time: $(date)"
    echo "✓ All timeframes download completed sequentially."
}

# 修改：处理所有时间框架的文件
flatten_all_file_structures() {
    echo "=== Flattening ALL File Structures ==="
    
    # 处理所有时间框架的文件
    local timeframes=("5m" "15m" "1h" "4h" "1d")
    
    for timeframe in "${timeframes[@]}"; do
        echo "Processing $timeframe files..."
        
        # 模式1: 标准freqtrade路径模式
        local source_pattern="$TEMP_DATADIR/data/$EXCHANGE/*_${timeframe}-*.feather"
        local file_count=0
        
        for source_file in $source_pattern; do
            if [[ -f "$source_file" ]]; then
                local filename=$(basename "$source_file")
                local target_path="$FLAT_DATADIR/$filename"
                
                # 复制文件到扁平目录
                cp "$source_file" "$target_path"
                ((file_count++))
            fi
        done
        
        # 模式2: 替代路径模式（某些freqtrade版本可能使用子目录）
        local alt_pattern="$TEMP_DATADIR/data/$EXCHANGE/${timeframe}/*.feather"
        if ls $alt_pattern 1> /dev/null 2>&1; then
            for source_file in $alt_pattern; do
                if [[ -f "$source_file" ]]; then
                    local filename=$(basename "$source_file")
                    local pair_name=$(echo "$filename" | cut -d'.' -f1)
                    local target_name="${pair_name}_USDT-${timeframe}-futures.feather"
                    local target_path="$FLAT_DATADIR/$target_name"
                    
                    cp "$source_file" "$target_path"
                    ((file_count++))
                fi
            done
        fi
        
        echo "✓ $timeframe: Flattened $file_count files"
    done
}

# 修改：主要处理逻辑
process_all_data() {
    echo "=== Processing ALL Data ==="
    local start_time=$(date +%s)
    
    # 步骤1: 一次性下载所有时间框架
    download_all_timeframes
    
    # 步骤2: 扁平化所有文件结构
    flatten_all_file_structures
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "✓ All data processing completed in $duration seconds"
}

# Generate comprehensive report
generate_report() {
    echo ""
    echo "=== Download Completion Report ==="
    echo "Completed at: $(date)"
    echo "Flat data location: $FLAT_DATADIR"
    echo ""
    
    # Count total files and show distribution
    local total_files=0
    local timeframes=("5m" "15m" "1h" "4h" "1d")
    
    echo "File Distribution:"
    for timeframe in "${timeframes[@]}"; do
        local files=($(find "$FLAT_DATADIR" -name "*${timeframe}-futures.feather" 2>/dev/null))
        local count=${#files[@]}
        echo "  $timeframe: $count files"
        total_files=$((total_files + count))
        
        # Show first 2 files as examples
        if [[ $count -gt 0 ]]; then
            for ((i=0; i<2 && i<count; i++)); do
                echo "    📁 $(basename "${files[i]}")"
            done
            if [[ $count -gt 2 ]]; then
                echo "    ... and $((count - 2)) more files"
            fi
        fi
    done
    
    echo ""
    echo "Summary:"
    echo "  Total Feather files: $total_files"
    echo "  Data size: $(du -sh "$FLAT_DATADIR" 2>/dev/null | cut -f1 || echo 'Unknown')"
    echo ""
    echo "File naming convention:"
    echo "  SYMBOL_USDT_USDT-TIMEFRAME-futures.feather"
    echo "  Examples: 1INCH_USDT_USDT-1d-futures.feather, BTC_USDT_USDT-1h-futures.feather"
    echo ""
    echo "=== Multi-timeframe download completed successfully ==="
}

# Cleanup temporary files
cleanup() {
    echo ""
    echo "=== Cleaning up temporary files ==="
    if [[ -d "$TEMP_DATADIR" ]]; then
        rm -rf "$TEMP_DATADIR"
        echo "✓ Temporary directory removed: $TEMP_DATADIR"
    fi
}

# Main execution function
main() {
    local start_time=$(date +%s)
    
    # Validate environment
    validate_environment
    
    local pair_count=$(jq 'length' "$PAIRS_FILE")
    
    echo "=== Starting Multi-Timeframe Data Processing ==="
    echo "All timeframes: $TIMEFRAMES"
    echo "Trading pairs: $pair_count"
    echo "Method: Single freqtrade command for all timeframes"
    echo "Output: Flat file structure"
    echo "Estimated time: 10-45 minutes (more efficient than sequential download)"
    echo ""
    
    # 一次性处理所有数据
    process_all_data
    
    # Generate final report
    generate_report
    
    # Cleanup temporary files
    # cleanup
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    echo "Total execution time: $total_duration seconds ($(($total_duration / 60)) minutes)"
}

# Execute main function
main