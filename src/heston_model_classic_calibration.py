"""
Heston Model Calibration utilities.
==================================

This module groups together different calibration entry points for the Heston
framework so callers have a single place to import from.

Sections
--------
1) Imports and setup
2) Public API - price-based calibration (to option prices)
3) Public API - variance swap calibration (term-structure of expected variance)
"""

# ---------------------------------------------------------------------------
# 1) Imports and setup
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy import optimize
from typing import Dict, Optional, List
import logging
import sys
from pathlib import Path

try:
    from .heston_model import HestonModel
except ImportError:
    # Allow running as a script (python src/heston_calibration.py)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.heston_model import HestonModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 2) Public API - price-based calibration (to option prices)
# ---------------------------------------------------------------------------
def calibrate_heston_to_market(market_prices: pd.DataFrame, S0: float,
                               initial_params: Optional[Dict] = None,
                               bounds: Optional[List] = None,
                               pricing_method: str = 'fft') -> Dict[str, float]:
    """
    Calibrate Heston parameters to market option prices.
    
    Minimizes: Σ (C_market - C_heston)²
    
    Args:
        market_prices: DataFrame with columns ['strike', 'maturity', 'price', 'rate']
        S0: Current spot price
        initial_params: Initial parameter guesses
        bounds: Parameter bounds for optimization
        pricing_method: 'fft' (fast, stable), 'quad' (accurate), or 'rectangular' (fast)
        
    Returns:
        Dictionary with calibrated parameters
    """
    if initial_params is None:
        initial_params = {
            'v0': 0.1,
            'kappa': 3.0,
            'theta': 0.05,
            'sigma': 0.3,
            'rho': -0.8,
            'lambd': 0.03
        }
    
    if bounds is None:
        bounds = [
            (1e-3, 0.1),      # v0
            (1e-3, 5.0),      # kappa
            (1e-3, 0.1),      # theta
            (1e-2, 1.0),      # sigma
            (-1.0, 0.0),      # rho
            (-1.0, 1.0)       # lambd
        ]
    
    K = market_prices['strike'].values
    tau = market_prices['maturity'].values
    P_market = market_prices['price'].values
    
    # Handle rates
    if 'rate' in market_prices.columns:
        rates = market_prices['rate'].values
        if isinstance(rates[0], (list, np.ndarray)):
            rates = [r[0] if isinstance(r, np.ndarray) else r for r in rates]
        r_avg = np.mean(rates) if len(rates) > 0 else 0.05
    else:
        rates = None
        r_avg = 0.05
    
    x0 = [initial_params['v0'], initial_params['kappa'], initial_params['theta'],
          initial_params['sigma'], initial_params['rho'], initial_params['lambd']]
    
    def objective(x):
        v0, kappa, theta, sigma, rho, lambd = x
        
        # Create model with average rate (or use per-maturity rates if available)
        model = HestonModel(kappa, theta, sigma, rho, v0, r=r_avg, lambd=lambd)
        
        # Calculate model prices
        P_model = []
        for i, (k, t) in enumerate(zip(K, tau)):
            # Use specific rate if available
            if rates is not None:
                r_val = rates[i] if isinstance(rates[i], (int, float)) else r_avg
                model.r = r_val
            
            if pricing_method == 'fft':
                price = model.call_price_fft(S0, k, t)
            elif pricing_method == 'rectangular':
                price = model.call_price_rectangular(S0, k, t)
            else:  # quad
                price = model.call_price_quad(S0, k, t)
            
            P_model.append(price)
        
        P_model = np.array(P_model)
        
        # Squared error (mean squared error)
        err = np.mean((P_market - P_model)**2)
        
        return err
    
    # Optimize
    logger.info("Starting Heston calibration to market prices...")
    result = optimize.minimize(
        objective, x0, method='SLSQP', bounds=bounds,
        options={'maxiter': 10000, 'ftol': 1e-3}
    )
    
    if not result.success:
        logger.warning(f"Calibration may not have converged: {result.message}")
    
    v0, kappa, theta, sigma, rho, lambd = result.x
    
    logger.info(f"Calibration complete. Final error: {result.fun:.6f}")
    
    return {
        'v0': float(v0),
        'kappa': float(kappa),
        'theta': float(theta),
        'sigma': float(sigma),
        'rho': float(rho),
        'lambd': float(lambd),
        'r': float(r_avg),
        'q': 0.0,
        'calibration_error': float(result.fun),  # Include the calibration error
    }


# ---------------------------------------------------------------------------
# 3) Public API - variance swap calibration
#     (fit κ^Q, θ^Q to VS term-structure: annualized variance curve)
# ---------------------------------------------------------------------------
def calibrate_heston_variance_swap(
    v0: float,
    r: float,
    q: float,
    maturities: np.ndarray,
    vswap_vols: np.ndarray,
    kappa_q_bounds: tuple[float, float] = (1e-6, 10.0),
    theta_q_bounds: tuple[float, float] = (1e-8, 5.0),
    weights: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Calibrate (kappa_Q, theta_Q) to a variance swap term-structure.
    Inputs:
      - maturities: array of T (years)
      - vswap_vols: variance swap volatility levels (annualized, i.e., sqrt(variance)), same shape as maturities
      - v0: initial variance under Q
      - r, q: rates for completeness (not used in the variance curve formula but included for consistency)
    Returns:
      dict with fitted 'kappa_q' and 'theta_q'
    """
    # convert variance swap vol to annualized variance
    targets_var = np.asarray(vswap_vols, dtype=float).ravel() ** 2
    T = np.asarray(maturities, dtype=float).ravel()
    model = HestonModel(kappa=1.0, theta=0.04, sigma=0.3, rho=-0.7, v0=float(v0), r=float(r), q=float(q))
    fit = model.calibrate_variance_swap(
        maturities=T,
        targets_annualized_variance=targets_var,
        weights=weights,
        bounds=(kappa_q_bounds, theta_q_bounds),
        update_model=True,
    )
    return {
        "kappa_q": float(fit["kappa_q"]),
        "theta_q": float(fit["theta_q"]),
        "success": bool(fit["success"]),
        "v0": float(v0),
        "r": float(r),
        "q": float(q),
    }
