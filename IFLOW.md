# IFLOW Context for NostalgiaForInfinity

## Project Overview

This project contains **NostalgiaForInfinity**, a sophisticated trading strategy for the [Freqtrade](https://www.freqtrade.io) cryptocurrency trading bot. The strategy is implemented in Python and is designed for automated trading on various cryptocurrency exchanges. The main strategy files are `NostalgiaForInfinityX.py` and its subsequent versions (`NostalgiaForInfinityX2.py`, `NostalgiaForInfinityX3.py`, etc.).

Key features of the strategy include:
- Complex technical analysis using libraries like `ta-lib`, `pandas-ta`, and custom indicators.
- Position adjustment logic (rebuying).
- Hold support for specific trades or pairs.
- Configurable parameters for risk management (stoploss, ROI).
- Support for both spot and futures trading modes.
- Integration with Freqtrade's standard strategy interface (`IStrategy`).

## Building and Running

### Prerequisites

- Python 3.12 or higher (as specified in `pyproject.toml`).
- Docker (optional, but recommended for ease of use, as seen in `docker-compose.yml`).
- A Freqtrade installation.
- Required Python packages (see `tests/requirements.txt` for testing dependencies, main dependencies are `freqtrade` and `pandas_ta` as seen in strategy and `pyproject.toml`).

### Using Docker (Recommended)

The project is set up to be easily run using Docker and Docker Compose.

1.  **Setup Environment**: Copy `live-account-example.env` to `.env` and fill in your exchange API keys and other settings.
2.  **Configure**: Place your Freqtrade configuration JSON file (e.g., based on `configs/exampleconfig.json`) in `user_data/config.json`. Ensure settings like `exchange.name`, `exchange.key`, and `exchange.secret` are correctly set for live trading. Adjust `max_open_trades`, `stake_currency`, etc., as needed.
3.  **Run**: Execute `docker-compose up`. This will start the Freqtrade bot using the strategy specified by the `FREQTRADE__STRATEGY` environment variable (defaults to `NostalgiaForInfinityX6`). The strategy file is mounted into the container.

### Running Locally (Without Docker)

1.  **Install Freqtrade**: Follow the official Freqtrade installation guide for your OS.
2.  **Install Dependencies**: Ensure `pandas_ta` is installed (`pip install pandas_ta`).
3.  **Configure**: Create a Freqtrade configuration file (e.g., `user_data/config.json`) based on `configs/exampleconfig.json` and configure your exchange details.
4.  **Run**: Use the Freqtrade CLI to start the bot, specifying the strategy file and configuration:
    ```bash
    freqtrade trade --strategy NostalgiaForInfinityX6 --strategy-path . --config user_data/config.json
    ```
    (Replace `NostalgiaForInfinityX6` and `user_data/config.json` with your chosen strategy and config file).

### Backtesting

Backtesting scripts and configurations are located in the `tests/backtests/` directory.
1.  Ensure you have historical data for the pairs and timeframe you intend to backtest.
2.  Use Freqtrade's backtesting command:
    ```bash
    freqtrade backtesting --strategy NostalgiaForInfinityX6 --strategy-path . --config configs/exampleconfig.json --timerange 20230101-20231231
    ```
    (Adjust strategy, config, and timerange as needed).

## Development Conventions

- **Code Style**: The project uses `black` and `ruff` for code formatting and linting, configured in `pyproject.toml`. Line length is set to 119 characters.
- **Strategy Structure**: Strategies inherit from `IStrategy` and implement standard methods like `populate_indicators`, `populate_entry_tearget`, and `populate_exit_tearget`. Extensive use of `pandas` DataFrames for data manipulation.
- **Configuration**: Strategy parameters, rebuy settings, and hold configurations are often defined as class attributes within the strategy file.
- **Hold Support**: The strategy supports holding specific trades or pairs until a certain profit level is reached, managed via an external `nfi-hold-trades.json` file in the `user_data` directory.
- **Testing**: Tests are located in the `tests/` directory. Dependencies for testing are listed in `tests/requirements.txt`. The project uses `pytest`.