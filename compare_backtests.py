import json
import sys

files = [
    "/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-12-03_23-15-33_20240101-20251101-1000-unlimited_short_661/backtest-result-2025-12-03_23-15-33.json",
    "/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-12-01_03-24-06_20240101-20251101_1000-unlimited/backtest-result-2025-12-01_03-24-06.json"
]

labels = ["Short 661 Enabled", "Baseline (Doom Stop 0.70)"]

for i, file_path in enumerate(files):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        strategy_name = list(data['strategy'].keys())[0]
        stats = data['strategy'][strategy_name]
        
        print(f"--- {labels[i]} ---")
        print(f"Total Profit (Abs): {stats.get('profit_total_abs', 'N/A')}")
        print(f"Total Profit (Ratio): {stats.get('profit_total', 0) * 100:.2f}%")
        print(f"Total Trades: {stats.get('total_trades', 'N/A')}")
        
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        draws = stats.get('draws', 0)
        total = wins + losses + draws
        
        print(f"Win Rate: {wins / total * 100:.2f}%" if total > 0 else "Win Rate: N/A")
        print(f"Max Drawdown (Abs): {stats.get('max_drawdown_abs', 'N/A')}")
        print(f"Max Drawdown (%): {stats.get('max_drawdown', 0) * 100:.2f}%")
        print(f"Sharpe Ratio: {stats.get('sharpe', 'N/A')}")
        print(f"Sortino Ratio: {stats.get('sortino', 'N/A')}")
        print(f"Profit Factor: {stats.get('profit_factor', 'N/A')}")
        print(f"Avg Profit: {stats.get('profit_mean', 0) * 100:.2f}%")
        print(f"Avg Duration: {stats.get('avg_duration', 'N/A')}")
        print(f"Wins: {wins}")
        print(f"Losses: {losses}")
        print(f"Draws: {draws}")
        print("\n")
        
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
