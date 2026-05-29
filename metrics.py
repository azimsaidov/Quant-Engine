import numpy as np
import pandas as pd

def calculate_cagr(start_val, end_val, num_days):
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    """
    if num_days <= 0 or start_val <= 0:
        return 0.0
    years = num_days / 365.25
    return (end_val / start_val) ** (1.0 / years) - 1.0

def calculate_max_drawdown(equity_series):
    """
    Calculates the maximum peak-to-trough drawdown from an equity curve series.
    Returns:
        float: The maximum drawdown (as a negative decimal, e.g., -0.25 for -25%).
        pd.Series: The drawdown percentage series over time.
    """
    peaks = equity_series.cummax()
    drawdowns = (equity_series - peaks) / peaks
    max_dd = drawdowns.min()
    return max_dd, drawdowns

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    """
    Calculates the annualized Sharpe Ratio.
    Assumes 365.25 trading days per year (since crypto trades 24/7).
    """
    avg_daily_return = daily_returns.mean()
    daily_std = daily_returns.std()
    
    if daily_std == 0 or pd.isna(daily_std):
        return 0.0
        
    # Annualize returns and standard deviation
    ann_return = avg_daily_return * 365.25
    ann_std = daily_std * np.sqrt(365.25)
    
    return (ann_return - risk_free_rate) / ann_std

def calculate_sortino_ratio(daily_returns, risk_free_rate=0.0):
    """
    Calculates the annualized Sortino Ratio (penalizing only downside volatility).
    Assumes 365.25 trading days per year.
    """
    avg_daily_return = daily_returns.mean()
    
    # Calculate downside standard deviation
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) == 0:
        return 0.0
        
    downside_std = downside_returns.std()
    if downside_std == 0 or pd.isna(downside_std):
        return 0.0
        
    ann_return = avg_daily_return * 365.25
    ann_downside_std = downside_std * np.sqrt(365.25)
    
    return (ann_return - risk_free_rate) / ann_downside_std

def calculate_trade_metrics(prices, signals, fee_rate=0.001):
    """
    Identifies individual trades from a signal series and calculates
    advanced metrics like Win Rate, Profit Factor, and average trade return.
    
    Parameters:
    -----------
    prices : np.ndarray
        Array of Close prices.
    signals : np.ndarray
        Array of strategy signals (1 = hold asset, 0 = cash).
    fee_rate : float
        Transaction fee rate per trade.
        
    Returns:
    --------
    dict
        A dictionary containing win rate, profit factor, total trades, average trade, etc.
    """
    trades = []
    holding = False
    buy_price = 0.0
    
    for t in range(len(signals)):
        # Buy Signal (0 -> 1)
        if t > 0 and signals[t] == 1 and signals[t-1] == 0 and not holding:
            buy_price = prices[t]
            holding = True
        # Initial Buy (if signal starts at 1)
        elif t == 0 and signals[t] == 1 and not holding:
            buy_price = prices[t]
            holding = True
        # Sell Signal (1 -> 0)
        elif t > 0 and signals[t] == 0 and signals[t-1] == 1 and holding:
            sell_price = prices[t]
            # Calculate return including round-trip fees
            trade_return = (sell_price * (1.0 - fee_rate)) / (buy_price * (1.0 + fee_rate)) - 1.0
            trades.append(trade_return)
            holding = False
            
    # If still holding at the very end, close out at the last price
    if holding:
        sell_price = prices[-1]
        trade_return = (sell_price * (1.0 - fee_rate)) / (buy_price * (1.0 + fee_rate)) - 1.0
        trades.append(trade_return)
        
    num_trades = len(trades)
    if num_trades == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_trade_return': 0.0
        }
        
    trades = np.array(trades)
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades <= 0]
    
    win_rate = len(winning_trades) / num_trades
    
    gross_profits = winning_trades.sum()
    gross_losses = np.abs(losing_trades.sum())
    
    # Avoid division by zero
    if gross_losses == 0:
        profit_factor = float('inf') if gross_profits > 0 else 1.0
    else:
        profit_factor = gross_profits / gross_losses
        
    return {
        'total_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_return': trades.mean()
    }

if __name__ == "__main__":
    # Test calculations
    equity = pd.Series([100, 105, 95, 110, 115, 100, 120])
    max_dd, dd_series = calculate_max_drawdown(equity)
    print(f"Test Max Drawdown: {max_dd*100:.2f}%")
    
    returns = equity.pct_change().dropna()
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    print(f"Test Sharpe: {sharpe:.4f}")
    print(f"Test Sortino: {sortino:.4f}")
