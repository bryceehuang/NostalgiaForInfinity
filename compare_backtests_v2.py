import json
import sys

files = [
    "/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-12-02_01-51-36_2024全年-大资金-unlimited_stop-doom-all_0.24/backtest-result-2025-12-02_01-51-36.json",
    "/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-12-02_00-43-37_2024全年-大资金-unlimited_stop-doom-all_0.35/backtest-result-2025-12-02_00-43-37.json",
    "/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-12-02_00-14-35_2024全年-大资金-unlimited_stop-doom_0.35/backtest-result-2025-12-02_00-14-35.json"
]

labels = ["All Stops 0.24 (New)", "All Stops 0.30 (Previous Best)", "Doom Stop 0.35 (Baseline)"]

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
