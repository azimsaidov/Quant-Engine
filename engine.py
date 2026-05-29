import numpy as np
import pandas as pd
import metrics
from portfolio_optimizer import optimize_max_sharpe, optimize_risk_parity

class PortfolioEngine:
    """
    High-Performance Multi-Asset Portfolio Rebalancing & Backtesting Engine.
    Simulates daily price wiggles, rolling quantitative optimization sweeps (MVO/Risk Parity),
    dynamic weight rebalancing, and transaction fee drag over time.
    """
    
    def __init__(self, starting_capital=10000.0, fee_rate=0.001):
        self.starting_capital = starting_capital
        self.fee_rate = fee_rate
        
    def run_backtest(self, df, method='max_sharpe', lookback_window=90, rebalance_interval=30):
        """
        Executes a multi-asset portfolio optimization backtest.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Cleaned multi-asset closing price DataFrame with a DatetimeIndex.
        method : str
            The portfolio optimization model: 'max_sharpe', 'risk_parity', or 'equal_weighted'.
        lookback_window : int
            Number of daily return history records used to calculate expected returns & cov matrix. Default is 90.
        rebalance_interval : int
            Number of days between rebalancing sweeps. Default is 30 days.
            
        Returns:
        --------
        dict
            A structured results dictionary containing portfolio wiggles, metrics, and weight histories.
        """
        print(f"[Engine] Running Backtest. Model: {method.upper()} | Lookback: {lookback_window}d | Rebalance: {rebalance_interval}d...")
        
        # 1. Calculate Daily Returns of all assets
        asset_returns = df.pct_change().fillna(0)
        tickers = list(df.columns)
        num_assets = len(tickers)
        
        # 2. Initial Setup
        portfolio_value = self.starting_capital
        # Start with equal weights
        weights = np.repeat(1.0 / num_assets, num_assets)
        # USD Cash allocated to each asset
        asset_values = portfolio_value * weights
        
        # Histories to log
        portfolio_history = []
        weight_history = []
        fees_paid_log = []
        
        # 3. Daily Simulation Loop
        for i in range(lookback_window, len(df)):
            date = df.index[i]
            # Daily returns for today
            day_returns = asset_returns.iloc[i].values
            
            # --- DYNAMIC PORTFOLIO REBALANCE ---
            # Every rebalance interval, we re-run our mathematical optimizations
            if (i - lookback_window) % rebalance_interval == 0:
                # Extract lookback return data
                lookback_df = asset_returns.iloc[i-lookback_window:i]
                
                # Annualize expected returns and covariance matrix
                # 365.25 is standard in crypto, but since we aligned stock calendars to crypto,
                # the dataset represents 365 days of active data.
                mean_returns = lookback_df.mean().values * 365.25
                cov_matrix = lookback_df.cov().values * 365.25
                
                # Compute new optimized weights
                if method == 'max_sharpe':
                    new_weights = optimize_max_sharpe(mean_returns, cov_matrix)
                elif method == 'risk_parity':
                    new_weights = optimize_risk_parity(cov_matrix)
                elif method == 'equal_weighted':
                    new_weights = np.repeat(1.0 / num_assets, num_assets)
                else:
                    raise ValueError(f"Unknown optimization method: {method}")
                    
                # Calculate transaction fee friction on rebalancing volume
                # Volume traded = sum( Value * |new_weight - current_weight| )
                usd_diffs = portfolio_value * np.abs(new_weights - weights)
                total_fees = np.sum(usd_diffs * self.fee_rate)
                
                # Subtract fees from overall capital
                portfolio_value -= total_fees
                fees_paid_log.append((date, total_fees))
                
                # Apply new weights to the adjusted portfolio value
                weights = new_weights
                asset_values = portfolio_value * weights
            
            # --- DAILY PRICE WIGGLE (FLOAT PORTFOLIO VALUE) ---
            # Float the value of each asset by today's return
            asset_values = asset_values * (1.0 + day_returns)
            
            # Re-sum portfolio value
            portfolio_value = np.sum(asset_values)
            
            # Recalculate weights based on price drift
            weights = asset_values / portfolio_value
            
            # Log today's results
            portfolio_history.append(portfolio_value)
            weight_history.append(weights.copy())
            
        # 4. Compile Results
        backtest_dates = df.index[lookback_window:]
        equity_series = pd.Series(portfolio_history, index=backtest_dates)
        weights_df = pd.DataFrame(weight_history, index=backtest_dates, columns=tickers)
        
        # Calculate daily portfolio returns
        portfolio_returns = equity_series.pct_change().fillna(0)
        
        # Calculate final metrics
        num_days = len(backtest_dates)
        cagr = metrics.calculate_cagr(self.starting_capital, portfolio_history[-1], num_days)
        max_dd, drawdown_series = metrics.calculate_max_drawdown(equity_series)
        sharpe = metrics.calculate_sharpe_ratio(portfolio_returns)
        sortino = metrics.calculate_sortino_ratio(portfolio_returns)
        
        total_fees = sum([f[1] for f in fees_paid_log])
        
        results = {
            'method': method,
            'starting_capital': self.starting_capital,
            'final_value': portfolio_history[-1],
            'total_return_pct': ((portfolio_history[-1] - self.starting_capital) / self.starting_capital) * 100,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'total_fees_paid': total_fees,
            'fee_drag_pct': (total_fees / self.starting_capital) * 100,
            'equity_curve': equity_series,
            'drawdown_curve': drawdown_series,
            'weights_history': weights_df,
            'tickers': tickers
        }
        
        print(f"[Engine] Backtest Complete. Return: {results['total_return_pct']:+.2f}% | Sharpe: {sharpe:.4f} | Max DD: {max_dd*100:.2f}%")
        return results

class BacktestEngine:
    """
    High-Performance Single-Asset Backtesting Engine.
    Simulates trading signals, execution, transaction fee friction, and metrics.
    """
    def __init__(self, starting_capital=10000.0, fee_rate=0.001):
        self.starting_capital = starting_capital
        self.fee_rate = fee_rate
        
    def run(self, df, strategy):
        """
        Executes backtest for a single asset strategy on the historical price DataFrame.
        """
        if 'Close' not in df.columns:
            df = df.copy()
            df['Close'] = df.iloc[:, 0]
            
        close = df['Close'].values
        dates = df.index
        
        # 1. Generate Strategy Signals
        signals = strategy.generate_signals(df)
        # Shift signals by 1 to execute on the next day's close (avoiding lookahead bias)
        position = pd.Series(signals, index=dates).shift(1).fillna(0).values
        
        # 2. Track Trades and Fees
        portfolio_values = np.zeros(len(df))
        portfolio_values[0] = self.starting_capital
        
        fees_paid = np.zeros(len(df))
        cash = self.starting_capital
        units = 0.0
        
        trades = []
        entry_price = 0.0
        
        for t in range(len(df)):
            price = close[t]
            prev_pos = position[t-1] if t > 0 else 0.0
            curr_pos = position[t]
            
            if t > 0:
                current_val = cash + units * price
            else:
                current_val = self.starting_capital
                
            if curr_pos != prev_pos:
                fee = current_val * self.fee_rate
                current_val -= fee
                fees_paid[t] = fee
                
                if curr_pos == 1.0: # Buy
                    units = current_val / price
                    cash = 0.0
                    entry_price = price
                else: # Sell
                    cash = current_val
                    units = 0.0
                    if entry_price > 0.0:
                        ret = (price - entry_price) / entry_price
                        trades.append(ret)
                        entry_price = 0.0
                        
            portfolio_values[t] = cash + units * price
            
        equity_series = pd.Series(portfolio_values, index=dates)
        portfolio_returns = equity_series.pct_change().fillna(0)
        
        # Buy & Hold Baseline calculation
        bh_units = self.starting_capital / close[0]
        bh_equity = bh_units * close
        bh_equity_series = pd.Series(bh_equity, index=dates)
        
        # Calculate statistics
        num_days = len(df)
        cagr = metrics.calculate_cagr(self.starting_capital, portfolio_values[-1], num_days)
        max_dd, drawdown_series = metrics.calculate_max_drawdown(equity_series)
        sharpe = metrics.calculate_sharpe_ratio(portfolio_returns)
        sortino = metrics.calculate_sortino_ratio(portfolio_returns)
        
        bh_cagr = metrics.calculate_cagr(self.starting_capital, bh_equity[-1], num_days)
        bh_max_dd, bh_drawdown_series = metrics.calculate_max_drawdown(bh_equity_series)
        
        total_trades = len(trades)
        if total_trades > 0:
            wins = [r for r in trades if r > 0]
            losses = [r for r in trades if r <= 0]
            win_rate = len(wins) / total_trades
            gross_profits = sum(wins)
            gross_losses = abs(sum(losses))
            profit_factor = gross_profits / (gross_losses + 1e-9) if gross_losses > 0 else float('inf')
        else:
            win_rate = 0.0
            profit_factor = 0.0
            
        results = {
            'strategy_name': strategy.__class__.__name__,
            'final_value': portfolio_values[-1],
            'total_return_pct': ((portfolio_values[-1] - self.starting_capital) / self.starting_capital) * 100,
            'cagr': cagr,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'equity_curve': equity_series,
            'drawdown_curve': drawdown_series,
            'total_fees_paid': np.sum(fees_paid),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'bh_final_value': bh_equity[-1],
            'bh_return_pct': ((bh_equity[-1] - self.starting_capital) / self.starting_capital) * 100,
            'bh_cagr': bh_cagr,
            'bh_max_drawdown': bh_max_dd,
            'bh_equity_curve': bh_equity_series
        }
        
        return results

if __name__ == "__main__":
    # Quick self-test on random mock data
    dates = pd.date_range("2026-01-01", periods=150)
    np.random.seed(42)
    btc_prices = 100.0 + np.cumsum(np.random.normal(0, 5.0, 150))
    spy_prices = 100.0 + np.cumsum(np.random.normal(0, 1.0, 150))
    
    mock_df = pd.DataFrame(index=dates, data={'BTC-USD': btc_prices, 'SPY': spy_prices})
    # ffill gaps
    mock_df = mock_df.ffill().bfill()
    
    engine = PortfolioEngine()
    
    # Test Max Sharpe
    res = engine.run_backtest(mock_df, method='max_sharpe', lookback_window=30, rebalance_interval=10)
    print("\nWeight History (Last 5 Rows):")
    print(res['weights_history'].tail(5))
