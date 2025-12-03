import json
import sys
from datetime import datetime

file_path = "/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-12-01_03-24-06_20240101-20251101_1000-unlimited/backtest-result-2025-12-01_03-24-06.json"

try:
    with open(file_path, 'r') as f:
        data = json.load(f)
        
    strategy_name = list(data['strategy'].keys())[0]
    trades = data['strategy'][strategy_name]['trades']
    
    print("\n--- Top 10 Losses Analysis ---")
    # Sort by profit ratio (ascending, so worst losses first)
    sorted_losses = sorted([t for t in trades if t['profit_ratio'] <= 0], key=lambda x: x['profit_ratio'])[:10]
    
    for t in sorted_losses:
        # Calculate duration if key is missing
        if 'duration_min' in t:
            duration = t['duration_min']
        else:
            # Fallback: calculate from timestamps if available, or just print N/A
            # Assuming open_date and close_date are available in some format, but let's just skip for now to avoid error
            duration = "N/A"
            
        dca_count = len(t.get('orders', [])) - 2 # Subtract entry and exit
        if dca_count < 0: dca_count = 0
        
        print(f"Pair: {t['pair']}")
        print(f"  Loss: {t['profit_ratio']*100:.2f}%")
        print(f"  DCA Count: {dca_count}")
        print(f"  Exit Reason: {t['exit_reason']}")
        print(f"  Open: {t['open_date']}")
        print(f"  Close: {t['close_date']}")
        print("-" * 30)

except Exception as e:
    print(f"Error: {e}")
