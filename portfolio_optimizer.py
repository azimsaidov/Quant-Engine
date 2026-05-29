import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_max_sharpe(expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Computes the weights that maximize the Sharpe Ratio (Tangency Portfolio)
    using SciPy's sequential quadratic programming solver.
    
    Parameters:
    -----------
    expected_returns : np.ndarray
        Array of expected annualized returns for each asset.
    cov_matrix : np.ndarray
        Annualized covariance matrix of asset returns.
    risk_free_rate : float
        Annualized risk-free rate. Default is 0.0.
        
    Returns:
    --------
    np.ndarray
        Optimized weights summing to 1.0.
    """
    num_assets = len(expected_returns)
    
    # 1. Target function: Minimize the NEGATIVE Sharpe Ratio
    def objective_func(weights):
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if portfolio_vol == 0:
            return 0.0
        # Negative Sharpe
        return - (portfolio_return - risk_free_rate) / portfolio_vol
        
    # 2. Constraints: Sum of weights must equal 1.0 (100% allocation)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 3. Bounds: No shorting (weights >= 0) and no leverage (weights <= 1)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # 4. Initial guess: Equal weighting
    init_guess = np.repeat(1.0 / num_assets, num_assets)
    
    # 5. Run the solver
    res = minimize(
        fun=objective_func,
        x0=init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not res.success:
        print("Warning: Max Sharpe solver failed to converge. Defaulting to equal weights.")
        return init_guess
        
    return res.x


def optimize_risk_parity(cov_matrix):
    """
    Computes the weights that achieve Equal Risk Contribution (Risk Parity)
    across all assets using SciPy's SLSQP solver.
    
    The algorithm minimizes the sum of squared differences between the
    risk contributions of each asset pair.
    
    Parameters:
    -----------
    cov_matrix : np.ndarray
        Annualized covariance matrix of asset returns.
        
    Returns:
    --------
    np.ndarray
        Optimized Risk Parity weights summing to 1.0.
    """
    num_assets = cov_matrix.shape[0]
    
    # 1. Target function: Minimize the variance of Risk Contributions
    def objective_func(weights):
        # Portfolio Volatility
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if portfolio_vol == 0:
            return 0.0
            
        # Marginal Risk Contribution of each asset: Sigma * w / vol
        marginal_risk_contrib = (cov_matrix @ weights) / portfolio_vol
        
        # Absolute Risk Contribution of each asset: w_i * MRC_i
        risk_contrib = weights * marginal_risk_contrib
        
        # We calculate the sum of squared differences of risk contributions
        # RC_diffs = [ (RC_i - RC_j)^2 ]
        diffs = []
        for i in range(num_assets):
            for j in range(num_assets):
                diffs.append((risk_contrib[i] - risk_contrib[j]) ** 2)
                
        return np.sum(diffs)
        
    # 2. Constraints: Sum of weights must equal 1.0
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # 3. Bounds: Long-only, no leverage
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # 4. Initial guess: Equal weighting
    init_guess = np.repeat(1.0 / num_assets, num_assets)
    
    # 5. Run the solver
    res = minimize(
        fun=objective_func,
        x0=init_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if not res.success:
        print("Warning: Risk Parity solver failed to converge. Defaulting to equal weights.")
        return init_guess
        
    return res.x

if __name__ == "__main__":
    # Self-test solvers
    print("Testing Solvers on a mock 3-asset portfolio (BTC, SPY, GLD)...")
    
    # Mock expected returns (annualized)
    mock_returns = np.array([0.45, 0.10, 0.05])  # BTC = 45%, SPY = 10%, GLD = 5%
    
    # Mock covariance matrix (annualized)
    # BTC has high vol, SPY moderate, GLD low
    mock_cov = np.array([
        [0.360, 0.010, 0.005],  # BTC variance/covariances
        [0.010, 0.040, 0.002],  # SPY variance/covariances
        [0.005, 0.002, 0.025]   # GLD variance/covariances
    ])
    
    w_sharpe = optimize_max_sharpe(mock_returns, mock_cov)
    w_parity = optimize_risk_parity(mock_cov)
    
    print("\nSolver Outputs:")
    print(f"Equal Weighting:   {[0.33, 0.33, 0.33]}")
    print(f"Max Sharpe Weights: {np.round(w_sharpe, 4)} -> (Expect BTC/SPY heavy due to high returns)")
    print(f"Risk Parity Weights: {np.round(w_parity, 4)} -> (Expect GLD/SPY heavy to equalize high BTC volatility)")
