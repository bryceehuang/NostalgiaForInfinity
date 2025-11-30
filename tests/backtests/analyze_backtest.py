import json
import pandas as pd
import os

# Path to the backtest result file
file_path = '/Users/bryce/Documents/projects/NostalgiaForInfinity/user_data/backtest_results/backtest-result-2025-11-30_14-26-39_20250701-20251101_100/backtest-result-2025-11-30_14-26-39.json'

def analyze_backtest(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    # The structure seems to be data['strategy'][strategy_name]['trades']
    # Let's find the strategy name dynamically
    strategies = data.get('strategy', {})
    if not strategies:
        print("No strategy data found.")
        return

    strategy_name = list(strategies.keys())[0]
    print(f"Analyzing Strategy: {strategy_name}")
    
    strategy_data = strategies[strategy_name]
    trades = strategy_data.get('trades', [])
    
    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)
    
    # Basic Metrics
    total_trades = len(df)
    wins = df[df['profit_abs'] > 0]
    losses = df[df['profit_abs'] <= 0]
    
    win_rate = len(wins) / total_trades * 100
    
    total_profit_abs = df['profit_abs'].sum()
    avg_profit_abs = df['profit_abs'].mean()
    avg_profit_ratio = df['profit_ratio'].mean() * 100
    
    gross_profit = wins['profit_abs'].sum()
    gross_loss = abs(losses['profit_abs'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Duration
    # trade_duration is usually in minutes in the JSON export for 'trade_duration' field? 
    # Let's verify with timestamps if needed, but using the field is easier.
    # Actually freqtrade exports trade_duration in seconds usually? Or minutes?
    # User sample: "trade_duration": 40. "open_date": "00:35:00", "close_date": "01:15:00". That is 40 minutes.
    # So trade_duration is in minutes.
    avg_duration = df['trade_duration'].mean()
    
    print("\n--- General Performance ---")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Profit (Abs): {total_profit_abs:.2f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Avg Profit per Trade (Abs): {avg_profit_abs:.2f}")
    print(f"Avg Profit per Trade (%): {avg_profit_ratio:.2f}%")
    print(f"Avg Duration: {avg_duration:.2f} min")

    # Best and Worst
    best_trade = df.loc[df['profit_abs'].idxmax()]
    worst_trade = df.loc[df['profit_abs'].idxmin()]
    
    print("\n--- Best Trade ---")
    print(f"Pair: {best_trade['pair']}")
    print(f"Profit: {best_trade['profit_abs']:.2f} ({best_trade['profit_ratio']*100:.2f}%)")
    print(f"Open: {best_trade['open_date']}")
    
    print("\n--- Worst Trade ---")
    print(f"Pair: {worst_trade['pair']}")
    print(f"Profit: {worst_trade['profit_abs']:.2f} ({worst_trade['profit_ratio']*100:.2f}%)")
    print(f"Open: {worst_trade['open_date']}")

    # Group by Pair
    print("\n--- Top 5 Pairs by Profit ---")
    pair_stats = df.groupby('pair')['profit_abs'].sum().sort_values(ascending=False)
    print(pair_stats.head(5))
    
    print("\n--- Bottom 5 Pairs by Profit ---")
    print(pair_stats.tail(5))

    # Exit Reasons
    print("\n--- Exit Reasons ---")
    print(df['exit_reason'].value_counts().head(10))
    
    # Enter Tags
    print("\n--- Enter Tags ---")
    print(df['enter_tag'].value_counts().head(10))

    # Monthly/Weekly breakdown (simple)
    df['open_date'] = pd.to_datetime(df['open_date'])
    # df.set_index('open_date', inplace=True)
    # monthly_profit = df.resample('M')['profit_abs'].sum()
    # print("\n--- Monthly Profit ---")
    # print(monthly_profit)

if __name__ == "__main__":
    analyze_backtest(file_path)
