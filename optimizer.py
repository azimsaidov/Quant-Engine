import time
import multiprocessing
import itertools
import pandas as pd
import numpy as np
from engine import BacktestEngine

# Global helper function for multiprocessing.
# Must be at the top level of the module to be picklable across platforms (macOS/Windows).
def _run_single_backtest(task_args):
    """
    Executes a single backtest for a specific parameter combination.
    Helper function designed to run in parallel worker processes.
    """
    df, strategy_class, params, fee_rate, capital = task_args
    try:
        # Instantiate strategy with these specific parameters
        strategy = strategy_class(**params)
        engine = BacktestEngine(starting_capital=capital, fee_rate=fee_rate)
        results = engine.run(df, strategy)
        
        # Return a summarized dictionary of results
        summary = {
            'final_value': results['final_value'],
            'total_return_pct': results['total_return_pct'],
            'cagr': results['cagr'],
            'max_drawdown': results['max_drawdown'],
            'sharpe_ratio': results['sharpe_ratio'],
            'sortino_ratio': results['sortino_ratio'],
            'total_trades': results['total_trades'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor']
        }
        # Include the parameter keys in the output summary
        for k, v in params.items():
            summary[k] = v
        return summary
    except Exception as e:
        print(f"Error in worker thread for params {params}: {e}")
        return None

class GridSearchOptimizer:
    """
    High-Performance Parameter Grid Search Optimizer.
    Uses Python's multiprocessing pool to distribute backtesting iterations
    concurrently across all available CPU cores.
    """
    
    def __init__(self, df, fee_rate=0.001, capital=10000.0):
        self.df = df
        self.fee_rate = fee_rate
        self.capital = capital
        
    def optimize(self, strategy_class, param_grid, optimize_for='sharpe_ratio', run_benchmark=True):
        """
        Executes a parallel grid search over the strategy parameter combinations.
        
        Parameters:
        -----------
        strategy_class : class
            The uninstantiated Strategy class (e.g., MovingAverageCrossover).
        param_grid : dict
            A dictionary where keys are parameter names and values are lists of options to test.
            e.g. {'fast_window': [5, 10, 20], 'slow_window': [50, 100]}
        optimize_for : str
            The metric key to sort results by (default is 'sharpe_ratio').
        run_benchmark : bool
            Whether to run a performance comparison of parallel vs. sequential execution.
            
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing all backtest combinations sorted by the target metric.
        """
        print(f"\n[Optimizer] Initializing Grid Search for strategy: {strategy_class.__name__}...")
        
        # Generate all parameter combinations
        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        num_tasks = len(combinations)
        print(f"[Optimizer] Total combinations to test: {num_tasks}")
        
        # Prepare task arguments list for workers
        tasks = [(self.df, strategy_class, params, self.fee_rate, self.capital) for params in combinations]
        
        # Running Sequential Benchmark if requested
        sequential_time = 0.0
        if run_benchmark:
            print("[Optimizer] Running sequential benchmark (single-threaded) on a sample of 3 tasks...")
            start_seq = time.time()
            for task in tasks[:3]:
                _run_single_backtest(task)
            duration_seq = time.time() - start_seq
            # Extrapolate sequential time for all tasks
            sequential_time = (duration_seq / 3.0) * num_tasks
            print(f"[Optimizer] Extrapolated Single-Threaded Time: {sequential_time:.2f} seconds")
            
        # Running Parallel Search
        print(f"[Optimizer] Starting Parallel Execution across {multiprocessing.cpu_count()} CPU cores...")
        start_parallel = time.time()
        
        # Create multiprocessing pool
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(_run_single_backtest, tasks)
            
        parallel_time = time.time() - start_parallel
        print(f"[Optimizer] Parallel Execution completed in {parallel_time:.2f} seconds.")
        
        if run_benchmark and parallel_time > 0:
            speedup = sequential_time / parallel_time
            print(f"[Optimizer] CONCURRENCY SPEEDUP: {speedup:.2f}x faster!")
            
        # Clean and package results
        valid_results = [r for r in results if r is not None]
        results_df = pd.DataFrame(valid_results)
        
        # Sort by target optimization metric
        # Handle cases where Sharpe can be negative or NaN by sorting descending, placing NaNs last
        if optimize_for in results_df.columns:
            results_df = results_df.sort_values(by=optimize_for, ascending=False, na_position='last')
            
        print("\nTOP 5 PARAMETER COMBINATIONS:")
        print(results_df.head(5).to_string(index=False))
        
        return results_df

if __name__ == "__main__":
    # Test optimizer
    import strategy
    
    dates = pd.date_range("2026-01-01", periods=100)
    # Generate random walking prices
    np.random.seed(42)
    prices = 100.0 + np.cumsum(np.random.normal(0, 1.5, 100))
    test_df = pd.DataFrame(index=dates, data={'Close': prices, 'High': prices+1.0, 'Low': prices-1.0})
    
    optimizer = GridSearchOptimizer(test_df)
    param_grid = {
        'fast_window': [2, 5, 8],
        'slow_window': [10, 15, 20]
    }
    
    # Run optimization
    optimizer.optimize(strategy.MovingAverageCrossover, param_grid, run_benchmark=True)
