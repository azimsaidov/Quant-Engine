# QuantPortfolio: Multi-Asset Backtesting & Portfolio Optimization Engine

A production-grade, vectorized multi-asset backtesting and portfolio rebalancing engine written in Python. The system ingests disparate market data (cryptocurrencies, equities, ETFs, commodities) from the Yahoo Finance API, resolves calendar mismatches, simulates daily weight drift, and executes rolling portfolio optimizations (Max Sharpe and Risk Parity) under flat transaction friction. It also includes a parallelized out-of-sample parameter optimization framework for single-asset technical trading strategies.

---

## Codebase Architecture

```
quant_engine/
├── data_loader.py          # Data ingestion, cache management, and calendar alignment
├── portfolio_optimizer.py  # SciPy SLSQP solvers for MVO and Risk Parity
├── engine.py               # Vectorized single-asset simulator and multi-asset rebalancer
├── strategy.py             # Rule-based technical strategies (MA Crossover, RSI, BBands)
├── optimizer.py            # Multiprocessed grid-search parameter optimizer
├── metrics.py              # Stateless quantitative performance library
├── reporter.py             # Visual plot generation and markdown logging
├── run.py                  # Multi-asset portfolio execution script
└── evaluate_all_strategies.py # Single-asset out-of-sample strategy evaluator
```

---

## Technical Specifications & Algorithms

### 1. Cross-Asset Calendar Alignment
Stocks trade ~252 days/year; cryptocurrencies trade 365 days/year. To prevent alignment errors during cross-asset matrix operations, the `data_loader` pipeline:
1. Executes an outer join across all ticker time series.
2. Applies a forward-fill (`ffill`) to carry stock closing prices over weekends and market holidays, preserving correlation structure.
3. Applies a backward-fill (`bfill`) for asset initialization gaps.
4. Serializes aligned dataframes to local CSV files in `data_cache/` to eliminate API latency and rate-limiting.

### 2. Portfolio Optimization Models
At each $T = 30$ day rebalancing boundary, the portfolio optimizer uses SciPy's Sequential Least Squares Programming (`SLSQP`) algorithm to compute asset weights under linear constraints.

#### Mean-Variance Optimization (Max Sharpe Tangency)
$$\text{Maximize } \frac{w^T \mu - r_f}{\sqrt{w^T \Sigma w}}$$
$$\text{Subject to } \sum_{i=1}^N w_i = 1.0, \quad 0.0 \le w_i \le 1.0$$
* $\mu$: Rolling 90-day annualized mean asset returns vector.
* $\Sigma$: Rolling 90-day covariance matrix.
* $r_f$: Risk-free rate (assumed $0.0\%$).

#### Risk Parity (Equal Risk Contribution)
Equalizes the marginal risk contributions ($RC_i$) of all assets:
$$\text{Minimize } \sum_{i=1}^N \sum_{j=1}^N \left( RC_i - RC_j \right)^2$$
$$\text{Subject to } \sum_{i=1}^N w_i = 1.0, \quad 0.0 \le w_i \le 1.0$$
$$\text{Where } RC_i = w_i \frac{(\Sigma w)_i}{\sigma_{p}}$$
* $\sigma_p = \sqrt{w^T \Sigma w}$: Total portfolio volatility.

### 3. Drifting Weight & Friction Simulation
Between rebalancing intervals, daily asset weights drift based on asset price wiggles:
$$w_{i, t} = w_{i, t-1} \cdot \frac{1 + R_{i, t}}{1 + R_{p, t}}$$
* $R_{i, t}$: Daily return of asset $i$ at time $t$.
* $R_{p, t} = \sum_{i=1}^N w_{i, t-1} R_{i, t}$: Daily portfolio return.

At each rebalancing date, a $0.1\%$ (10 bps) fee friction is applied to the portfolio value based on absolute weight turnover:
$$\text{Fee}_t = 0.001 \cdot \sum_{i=1}^N |w_{i, t^+} - w_{i, t^-}|$$
* $w_{i, t^-}$: Drifted weight before rebalancing.
* $w_{i, t^+}$: Optimized target weight after rebalancing.

---

## Empirical Backtest Results

### Multi-Asset Portfolio (Jan 1, 2020 - May 28, 2026)
* **Assets**: `['BTC-USD', 'ETH-USD', 'SPY', 'GLD', 'TLT']`
* **Starting Capital**: \$10,000.00
* **Parameters**: 90-Day Lookback | 30-Day Rebalancing Frequency | 10 bps Transaction Friction

| Strategy | Final Value | Return (%) | CAGR (%) | Max Drawdown | Sharpe Ratio | Sortino Ratio | Total Fees |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Equal Weighted** | \$60,987.19 | +509.87% | +34.13% | -47.33% | 1.1651 | 1.6296 | \$227.92 |
| **Max Sharpe (MVO)** | \$45,487.85 | +354.88% | +27.89% | -52.83% | 0.9447 | 1.0915 | \$1,754.75 |
| **Risk Parity** | **\$24,514.88** | **+145.15%** | **+15.68%** | **-29.45%** | **1.2252** | **1.7118** | **\$142.83** |

### Single-Asset Technical Strategies (BTC-USD, Oct 1, 2025 - May 28, 2026)
* **Parameters**: Train/Test Out-of-Sample Split. Parameters optimized on Train (`2024-01-01` to `2025-09-30`) via multi-threaded grid search, evaluated on Test.
* **Friction**: 10 bps transaction fee.

| Strategy | Final Value | Return (%) | CAGR (%) | Max Drawdown | Sharpe Ratio | Total Trades | Fees Paid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Buy & Hold Baseline** | \$6,266.69 | -37.33% | -55.85% | -52.26% | N/A | N/A | \$0.00 |
| **Moving Average Crossover** | **\$9,960.33** | **-0.40%** | **-0.61%** | **-0.40%** | **-0.1065** | **1** | **\$10.00** |
| **RSI Mean Reversion** | \$8,088.16 | -19.12% | -28.91% | -25.23% | -0.7410 | 17 | \$163.66 |
| **Bollinger Bands Volatility**| \$8,154.51 | -18.45% | -27.97% | -24.47% | -0.7188 | 13 | \$123.82 |

---

## Visual Dashboard Outputs

Backtest outputs are saved automatically to the `reports/` directory:
1. `portfolio_max_sharpe_vs_equal.png`: 3-panel dashboard (Log equity curve, historical drawdown, dynamic MVO asset weight allocation).
2. `portfolio_risk_parity_vs_equal.png`: 3-panel dashboard (Log equity curve, historical drawdown, dynamic Risk Parity asset weight allocation).
3. `all_strategies_comparison.png`: Out-of-sample comparative equity curves for BTC technical strategies.

---

## Local Deployment & Execution

### Prerequisites
Install the required quantitative stack:
```bash
pip install pandas numpy matplotlib scipy yfinance
```

### Run Multi-Asset Portfolio Backtest
To execute the multi-asset dynamic rebalancing pipeline and generate the MPT reports:
```bash
python run.py
```

### Run Technical Strategies Backtest
To run the train/test parameter grid search and evaluate the optimized single-asset strategies:
```bash
python evaluate_all_strategies.py
```
