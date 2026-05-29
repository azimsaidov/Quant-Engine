import abc
import numpy as np
import pandas as pd

class Strategy(abc.ABC):
    """
    Abstract Base Class representing a trading strategy.
    All custom strategies must inherit from this class and implement the generate_signals method.
    """
    
    def __init__(self, **params):
        """
        Initializes the strategy with parameter overrides.
        """
        for param_name, value in params.items():
            setattr(self, param_name, value)
            
    @abc.abstractmethod
    def generate_signals(self, df):
        """
        Generates trading signals for the given historical price DataFrame.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing Close, High, Low prices.
            
        Returns:
        --------
        pd.Series (or np.ndarray)
            A series of binary signals where:
            1 = BUY/HOLD asset (Long)
            0 = SELL/HOLD cash (Out of position)
        """
        pass


class MovingAverageCrossover(Strategy):
    """
    Trend-Following Strategy: Buys when a fast Moving Average crosses above
    a slow Moving Average, and sells when it crosses below.
    """
    
    def __init__(self, fast_window=20, slow_window=50):
        super().__init__(fast_window=fast_window, slow_window=slow_window)
        
    def generate_signals(self, df):
        signals = pd.Series(index=df.index, data=0)
        
        # Calculate Moving Averages
        fast_ma = df['Close'].rolling(window=self.fast_window).mean()
        slow_ma = df['Close'].rolling(window=self.slow_window).mean()
        
        # Signal is 1 when fast MA is greater than slow MA, else 0
        signals[fast_ma > slow_ma] = 1
        
        # Fill any initial NaN periods (during rolling window warmup) with 0
        return signals.fillna(0).values


class RsiStrategy(Strategy):
    """
    Mean-Reversion Strategy: Buys when the Relative Strength Index (RSI) is oversold
    (drops below 30) and sells when it is overbought (climbs above 70).
    """
    
    def __init__(self, rsi_window=14, oversold=30, overbought=70):
        super().__init__(rsi_window=rsi_window, oversold=oversold, overbought=overbought)
        
    def generate_signals(self, df):
        # Calculate RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        
        # Identify triggers
        buy_trigger = rsi < self.oversold
        sell_trigger = rsi > self.overbought
        
        # Run state machine to track active position (long vs. cash)
        signals = np.zeros(len(df))
        state = 0
        for t in range(len(df)):
            if pd.isna(rsi.iloc[t]):
                state = 0
            elif buy_trigger.iloc[t]:
                state = 1
            elif sell_trigger.iloc[t]:
                state = 0
            signals[t] = state
            
        return signals


class BollingerBandsStrategy(Strategy):
    """
    Volatility Mean-Reversion Strategy: Buys when the price crosses below the Lower
    Bollinger Band and sells when it crosses above the Upper Bollinger Band.
    """
    
    def __init__(self, window=20, num_std=2.0):
        super().__init__(window=window, num_std=num_std)
        
    def generate_signals(self, df):
        # Calculate Bollinger Bands
        rolling_mean = df['Close'].rolling(window=self.window).mean()
        rolling_std = df['Close'].rolling(window=self.window).std()
        
        lower_band = rolling_mean - self.num_std * rolling_std
        upper_band = rolling_mean + self.num_std * rolling_std
        
        # Identify triggers
        buy_trigger = df['Close'] < lower_band
        sell_trigger = df['Close'] > upper_band
        
        # Run state machine to track active position (long vs. cash)
        signals = np.zeros(len(df))
        state = 0
        for t in range(len(df)):
            if pd.isna(lower_band.iloc[t]):
                state = 0
            elif buy_trigger.iloc[t]:
                state = 1
            elif sell_trigger.iloc[t]:
                state = 0
            signals[t] = state
            
        return signals

if __name__ == "__main__":
    # Small self-test
    dates = pd.date_range("2026-01-01", periods=10)
    prices = [100, 101, 102, 98, 97, 105, 106, 103, 102, 104]
    test_df = pd.DataFrame(index=dates, data={'Close': prices})
    
    ma_strat = MovingAverageCrossover(fast_window=2, slow_window=5)
    sigs = ma_strat.generate_signals(test_df)
    print("Moving Average Crossover Test Signals:")
    print(sigs)
