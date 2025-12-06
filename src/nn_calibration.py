"""
Neural Network-Based Calibration for Heston Model.

This module implements the calibration procedure using deep learning.
The calibration uses:

1. Surrogate Network: Fast pricing approximation for optimization
2. PAN + CCN Pipeline: Two-phase calibration with error correction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional, List, Union, Callable
import logging
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).parent))

from neural_networks import (
    HestonSurrogateNetwork,
    PriceApproximatorNetwork,
    CalibrationCorrectionNetwork,
    HestonCalibrationNetwork
)
from heston_model import HestonModel
from training_pipeline import (
    train_pan_for_calibration_set,
    train_ccn,
    generate_training_data,
    train_surrogate_network
)

logger = logging.getLogger(__name__)


# =============================================================================
# CALIBRATION USING SURROGATE NETWORK
# =============================================================================

class NeuralNetworkCalibrator:
    """
    Heston model calibration using neural network surrogate.
    
    This class provides fast calibration by replacing the expensive
    Heston pricing function with a pre-trained neural network.
    
    The calibration minimizes:
        J(η) = Σ ωᵢⱼ (V̂(η, Tᵢ, mⱼ) - V^Mkt(Tᵢ, mⱼ))²
    
    where V̂ is approximated by the neural network.
    """
    
    def __init__(
        self,
        surrogate_model: HestonSurrogateNetwork = None,
        model_path: str = None,
        device: str = None
    ):
        """
        Initialize calibrator.
        
        Args:
            surrogate_model: Pre-trained surrogate network
            model_path: Path to saved model checkpoint
            device: Computing device
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load or use provided model
        if surrogate_model is not None:
            self.model = surrogate_model.to(self.device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = None
        
        if self.model is not None:
            self.model.eval()
        
        # Normalization parameters (to be set)
        self.norm_params = None
        
        # Calibration results
        self.calibration_results = {}
        
    def _load_model(self, path: str) -> HestonSurrogateNetwork:
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Create model and load state
        model = HestonSurrogateNetwork()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Loaded surrogate model from {path}")
        return model.to(self.device)
    
    def set_normalization(self, norm_params: Dict):
        """Set normalization parameters for input features."""
        self.norm_params = norm_params
        
        if self.model is not None:
            self.model.set_normalization(
                input_mean=torch.tensor(norm_params['feature_mean']),
                input_std=torch.tensor(norm_params['feature_std'])
            )
    
    def _prepare_inputs(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: Union[float, np.ndarray],
        params: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare input features for the neural network.
        
        Args:
            S: Spot price
            strikes: Strike prices
            maturities: Times to maturity
            r: Risk-free rate(s)
            params: Heston parameters [v0, kappa, theta, sigma, rho]
            
        Returns:
            Input tensor for the network
        """
        n = len(strikes)
        v0, kappa, theta, sigma, rho = params
        
        if isinstance(r, (int, float)):
            r = np.full(n, r)
        
        # Build feature matrix
        log_moneyness = np.log(strikes / S)
        
        features = np.column_stack([
            log_moneyness,
            maturities,
            np.full(n, v0),
            np.full(n, kappa),
            np.full(n, theta),
            np.full(n, sigma),
            np.full(n, rho),
            r
        ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def predict_prices(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        r: Union[float, np.ndarray],
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float
    ) -> np.ndarray:
        """
        Predict option prices using the surrogate network.
        
        Args:
            S: Spot price
            strikes, maturities: Option characteristics
            r: Risk-free rate(s)
            v0, kappa, theta, sigma, rho: Heston parameters
            
        Returns:
            Array of predicted option prices
        """
        params = np.array([v0, kappa, theta, sigma, rho])
        x = self._prepare_inputs(S, strikes, maturities, r, params)
        x = x.to(self.device)
        
        with torch.no_grad():
            normalized_prices = self.model(x, normalize=True)
            prices = normalized_prices.cpu().numpy().flatten() * S
        
        return prices
    
    def calibrate(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        r: Union[float, np.ndarray] = 0.03,
        weights: np.ndarray = None,
        initial_params: Dict = None,
        bounds: Dict = None,
        method: str = 'differential_evolution',
        maxiter: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Calibrate Heston model to market prices using surrogate network.
        
        Args:
            S: Spot price
            strikes: Array of strike prices
            maturities: Array of maturities
            market_prices: Array of observed market prices
            r: Risk-free rate(s)
            weights: Calibration weights (optional)
            initial_params: Initial parameter guess
            bounds: Parameter bounds
            method: 'differential_evolution', 'SLSQP', or 'L-BFGS-B'
            maxiter: Maximum iterations
            verbose: Print progress
            
        Returns:
            Dictionary with calibrated parameters and diagnostics
        """
        if self.model is None:
            raise ValueError("Surrogate model not loaded. Train or load a model first.")
        
        start_time = time.time()
        
        strikes = np.asarray(strikes).flatten()
        maturities = np.asarray(maturities).flatten()
        market_prices = np.asarray(market_prices).flatten()
        
        n_options = len(strikes)
        
        if weights is None:
            weights = np.ones(n_options)
        weights = weights / weights.sum()
        
        # Default bounds
        if bounds is None:
            bounds = {
                'v0': (0.001, 0.5),
                'kappa': (0.01, 10.0),
                'theta': (0.001, 0.5),
                'sigma': (0.01, 2.0),
                'rho': (-0.99, -0.01)
            }
        
        # Default initial params
        if initial_params is None:
            initial_params = {
                'v0': 0.04,
                'kappa': 2.0,
                'theta': 0.04,
                'sigma': 0.3,
                'rho': -0.7
            }
        
        # Objective function
        def objective(params):
            v0, kappa, theta, sigma, rho = params
            
            # Feller condition penalty
            feller_violation = max(0, sigma**2 - 2 * kappa * theta)
            
            # Predict prices
            try:
                pred_prices = self.predict_prices(
                    S, strikes, maturities, r, v0, kappa, theta, sigma, rho
                )
                
                # Weighted MSE
                errors = (pred_prices - market_prices) ** 2
                mse = np.sum(weights * errors)
                
                # Add Feller penalty
                mse += 100.0 * feller_violation
                
                return mse
                
            except Exception as e:
                return 1e10
        
        # Optimization bounds
        opt_bounds = [
            bounds['v0'],
            bounds['kappa'],
            bounds['theta'],
            bounds['sigma'],
            bounds['rho']
        ]
        
        # Initial guess
        x0 = [
            initial_params['v0'],
            initial_params['kappa'],
            initial_params['theta'],
            initial_params['sigma'],
            initial_params['rho']
        ]
        
        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective,
                bounds=opt_bounds,
                seed=42,
                maxiter=maxiter,
                tol=1e-8,
                workers=1,
                updating='deferred'
            )
            optimal_params = result.x
            final_error = result.fun
            success = result.success
            n_evals = result.nfev
            
        else:  # Gradient-based methods
            result = minimize(
                objective,
                x0=x0,
                method=method,
                bounds=opt_bounds,
                options={'maxiter': maxiter, 'ftol': 1e-8}
            )
            optimal_params = result.x
            final_error = result.fun
            success = result.success
            n_evals = result.nfev
        
        elapsed_time = time.time() - start_time
        
        # Store results
        v0_opt, kappa_opt, theta_opt, sigma_opt, rho_opt = optimal_params
        
        self.calibration_results = {
            'v0': float(v0_opt),
            'kappa': float(kappa_opt),
            'theta': float(theta_opt),
            'sigma': float(sigma_opt),
            'rho': float(rho_opt),
            'r': float(np.mean(r)) if isinstance(r, np.ndarray) else float(r),
            'calibration_error': float(final_error),
            'success': success,
            'n_evaluations': n_evals,
            'elapsed_time': elapsed_time,
            'method': method,
            'n_options': n_options
        }
        
        if verbose:
            print("\nNeural Network Calibration Results:")
            print(f"  v0:    {v0_opt:.6f}")
            print(f"  kappa: {kappa_opt:.6f}")
            print(f"  theta: {theta_opt:.6f}")
            print(f"  sigma: {sigma_opt:.6f}")
            print(f"  rho:   {rho_opt:.6f}")
            print(f"  MSE:   {final_error:.6f}")
            print(f"  Time:  {elapsed_time:.2f}s")
            print(f"  Feller condition: {'Satisfied' if 2*kappa_opt*theta_opt > sigma_opt**2 else 'VIOLATED'}")
        
        return self.calibration_results


# =============================================================================
# TWO-PHASE CALIBRATION (PAN + CCN)
# =============================================================================

class TwoPhaseCalibrator:
    """
    Two-phase calibration:
    
    Phase 1: Train PAN to approximate market price surface
    Phase 2: Calibrate Heston using PAN, then train CCN to correct errors
    
    This approach:
    1. Uses PAN to learn the strike-price relationship from market data
    2. Calibrates Heston model traditionally
    3. Uses CCN to correct systematic pricing errors
    """
    
    def __init__(self, device: str = None):
        """Initialize two-phase calibrator."""
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        self.pan_model = None
        self.ccn_model = None
        self.pan_norm_params = None
        self.ccn_norm_params = None
        self.heston_params = None
        self.training_history = {}
    
    def fit_pan(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        hidden_size: int = 8,
        epochs: int = 1000,
        learning_rate: float = 0.01
    ) -> Dict:
        """
        Phase 1: Train PAN on market data.
        
        Args:
            strikes: Array of strike prices
            market_prices: Array of market option prices
            hidden_size: PAN hidden layer size
            epochs: Training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        logger.info("Phase 1: Training Price Approximator Network (PAN)")
        
        self.pan_model, self.pan_norm_params, history = train_pan_for_calibration_set(
            strikes=strikes,
            prices=market_prices,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=str(self.device)
        )
        
        self.training_history['pan'] = history
        
        # Evaluate PAN fit
        self.pan_model.eval()
        with torch.no_grad():
            x = torch.tensor(
                (strikes.reshape(-1, 1) - self.pan_norm_params['strike_mean']) 
                / self.pan_norm_params['strike_std'],
                dtype=torch.float32
            ).to(self.device)
            pan_prices = self.pan_model(x).cpu().numpy().flatten()
        
        pan_mse = np.mean((pan_prices - market_prices) ** 2)
        logger.info(f"  PAN MSE: {pan_mse:.6f}")
        
        return history
    
    def calibrate_heston(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        r: float = 0.03,
        method: str = 'SLSQP'
    ) -> Dict:
        """
        Calibrate traditional Heston model to get initial parameters.
        
        Args:
            S: Spot price
            strikes, maturities, market_prices: Option data
            r: Risk-free rate
            method: Optimization method
            
        Returns:
            Calibrated Heston parameters
        """
        logger.info("Calibrating Heston model (traditional method)")
        
        # Objective function using Heston pricing
        def objective(params):
            v0, kappa, theta, sigma, rho = params
            
            # Check Feller condition
            if 2 * kappa * theta <= sigma ** 2:
                return 1e10
            
            try:
                model = HestonModel(
                    kappa=kappa, theta=theta, sigma=sigma,
                    rho=rho, v0=v0, r=r, lambd=0.0
                )
                
                pred_prices = np.array([
                    model.call_price_fft(S, k, t)
                    for k, t in zip(strikes, maturities)
                ])
                
                mse = np.mean((pred_prices - market_prices) ** 2)
                return mse
                
            except Exception:
                return 1e10
        
        # Bounds
        bounds = [
            (0.001, 0.5),   # v0
            (0.01, 10.0),   # kappa
            (0.001, 0.5),   # theta
            (0.01, 2.0),    # sigma
            (-0.99, -0.01)  # rho
        ]
        
        # Initial guess
        x0 = [0.04, 2.0, 0.04, 0.3, -0.7]
        
        # Optimize
        result = minimize(
            objective, x0=x0, method=method, bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-8}
        )
        
        self.heston_params = {
            'v0': result.x[0],
            'kappa': result.x[1],
            'theta': result.x[2],
            'sigma': result.x[3],
            'rho': result.x[4],
            'r': r,
            'calibration_error': result.fun
        }
        
        logger.info(f"  Heston calibration MSE: {result.fun:.6f}")
        
        return self.heston_params
    
    def fit_ccn(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        hidden_size: int = 7,
        epochs: int = 1000,
        learning_rate: float = 0.01
    ) -> Dict:
        """
        Phase 2: Train CCN to correct Heston pricing errors.
        
        Args:
            S: Spot price
            strikes, maturities: Option characteristics
            market_prices: Actual market prices
            hidden_size: CCN hidden size
            epochs: Training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        if self.heston_params is None:
            raise ValueError("Must calibrate Heston model first")
        
        logger.info("Phase 2: Training Calibration Correction Network (CCN)")
        
        # Get Heston model prices with calibrated parameters
        model = HestonModel(
            kappa=self.heston_params['kappa'],
            theta=self.heston_params['theta'],
            sigma=self.heston_params['sigma'],
            rho=self.heston_params['rho'],
            v0=self.heston_params['v0'],
            r=self.heston_params['r'],
            lambd=0.0
        )
        
        heston_prices = np.array([
            model.call_price_fft(S, k, t)
            for k, t in zip(strikes, maturities)
        ])
        
        # Train CCN
        self.ccn_model, self.ccn_norm_params, history = train_ccn(
            heston_prices=heston_prices,
            market_prices=market_prices,
            hidden_size=hidden_size,
            epochs=epochs,
            learning_rate=learning_rate,
            device=str(self.device)
        )
        
        self.training_history['ccn'] = history
        
        # Evaluate CCN correction
        self.ccn_model.eval()
        with torch.no_grad():
            x = torch.tensor(
                (heston_prices.reshape(-1, 1) - self.ccn_norm_params['input_mean'])
                / self.ccn_norm_params['input_std'],
                dtype=torch.float32
            ).to(self.device)
            corrected_prices = self.ccn_model(x).cpu().numpy().flatten()
        
        # Metrics
        heston_mse = np.mean((heston_prices - market_prices) ** 2)
        corrected_mse = np.mean((corrected_prices - market_prices) ** 2)
        
        logger.info(f"  Heston MSE: {heston_mse:.6f}")
        logger.info(f"  Corrected MSE: {corrected_mse:.6f}")
        logger.info(f"  Improvement: {(1 - corrected_mse/heston_mse)*100:.1f}%")
        
        return history
    
    def predict(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        use_correction: bool = True
    ) -> np.ndarray:
        """
        Predict option prices using calibrated model + correction.
        
        Args:
            S: Spot price
            strikes, maturities: Option characteristics
            use_correction: Whether to apply CCN correction
            
        Returns:
            Predicted option prices
        """
        if self.heston_params is None:
            raise ValueError("Model not calibrated")
        
        # Get Heston prices
        model = HestonModel(
            kappa=self.heston_params['kappa'],
            theta=self.heston_params['theta'],
            sigma=self.heston_params['sigma'],
            rho=self.heston_params['rho'],
            v0=self.heston_params['v0'],
            r=self.heston_params['r'],
            lambd=0.0
        )
        
        heston_prices = np.array([
            model.call_price_fft(S, k, t)
            for k, t in zip(strikes, maturities)
        ])
        
        if use_correction and self.ccn_model is not None:
            # Apply CCN correction
            self.ccn_model.eval()
            with torch.no_grad():
                x = torch.tensor(
                    (heston_prices.reshape(-1, 1) - self.ccn_norm_params['input_mean'])
                    / self.ccn_norm_params['input_std'],
                    dtype=torch.float32
                ).to(self.device)
                prices = self.ccn_model(x).cpu().numpy().flatten()
            return prices
        
        return heston_prices
    
    def full_calibration(
        self,
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        market_prices: np.ndarray,
        r: float = 0.03,
        pan_epochs: int = 1000,
        ccn_epochs: int = 1000,
        verbose: bool = True
    ) -> Dict:
        """
        Run complete two-phase calibration pipeline.
        
        Args:
            S: Spot price
            strikes, maturities: Option data
            market_prices: Market option prices
            r: Risk-free rate
            pan_epochs, ccn_epochs: Training epochs
            verbose: Print progress
            
        Returns:
            Complete calibration results
        """
        start_time = time.time()
        
        # Phase 1: Train PAN
        self.fit_pan(strikes, market_prices, epochs=pan_epochs)
        
        # Calibrate Heston
        self.calibrate_heston(S, strikes, maturities, market_prices, r)
        
        # Phase 2: Train CCN
        self.fit_ccn(S, strikes, maturities, market_prices, epochs=ccn_epochs)
        
        # Get final predictions
        pred_heston = self.predict(S, strikes, maturities, use_correction=False)
        pred_corrected = self.predict(S, strikes, maturities, use_correction=True)
        
        # Compute metrics
        mse_heston = np.mean((pred_heston - market_prices) ** 2)
        mse_corrected = np.mean((pred_corrected - market_prices) ** 2)
        rmse_heston = np.sqrt(mse_heston)
        rmse_corrected = np.sqrt(mse_corrected)
        
        elapsed_time = time.time() - start_time
        
        results = {
            'heston_params': self.heston_params,
            'mse_heston': float(mse_heston),
            'mse_corrected': float(mse_corrected),
            'rmse_heston': float(rmse_heston),
            'rmse_corrected': float(rmse_corrected),
            'improvement_pct': float((1 - mse_corrected / mse_heston) * 100),
            'elapsed_time': elapsed_time,
            'training_history': self.training_history
        }
        
        if verbose:
            print("\n" + "=" * 50)
            print("Two-Phase Calibration Results")
            print("=" * 50)
            print(f"Heston Parameters:")
            for k, v in self.heston_params.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.6f}")
            print(f"\nPerformance Metrics:")
            print(f"  Heston RMSE:    {rmse_heston:.4f}")
            print(f"  Corrected RMSE: {rmse_corrected:.4f}")
            print(f"  Improvement:    {results['improvement_pct']:.1f}%")
            print(f"  Total Time:     {elapsed_time:.1f}s")
            print("=" * 50)
        
        return results


# =============================================================================
# COMBINED CALIBRATION COMPARISON
# =============================================================================

def compare_calibration_methods(
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    r: float = 0.03,
    surrogate_model: HestonSurrogateNetwork = None,
    norm_params: Dict = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare different calibration methods.
    
    Methods compared:
    1. Traditional Heston (FFT pricing)
    2. Neural Network Surrogate
    3. Two-Phase (PAN + CCN)
    
    Args:
        S: Spot price
        strikes, maturities, market_prices: Option data
        r: Risk-free rate
        surrogate_model: Pre-trained surrogate (optional)
        norm_params: Normalization parameters
        verbose: Print results
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    # 1. Traditional Calibration
    print("\n1. Traditional Heston Calibration...")
    start_time = time.time()
    
    def trad_objective(params):
        v0, kappa, theta, sigma, rho = params
        if 2 * kappa * theta <= sigma ** 2:
            return 1e10
        try:
            model = HestonModel(
                kappa=kappa, theta=theta, sigma=sigma,
                rho=rho, v0=v0, r=r, lambd=0.0
            )
            pred = np.array([model.call_price_fft(S, k, t) for k, t in zip(strikes, maturities)])
            return np.mean((pred - market_prices) ** 2)
        except:
            return 1e10
    
    trad_result = minimize(
        trad_objective,
        x0=[0.04, 2.0, 0.04, 0.3, -0.7],
        method='SLSQP',
        bounds=[(0.001, 0.5), (0.01, 10), (0.001, 0.5), (0.01, 2), (-0.99, -0.01)]
    )
    trad_time = time.time() - start_time
    
    results.append({
        'Method': 'Traditional',
        'MSE': trad_result.fun,
        'RMSE': np.sqrt(trad_result.fun),
        'Time (s)': trad_time,
        'v0': trad_result.x[0],
        'kappa': trad_result.x[1],
        'theta': trad_result.x[2],
        'sigma': trad_result.x[3],
        'rho': trad_result.x[4]
    })
    
    # 2. Neural Network Calibration (if model provided)
    if surrogate_model is not None:
        print("2. Neural Network Surrogate Calibration...")
        nn_calibrator = NeuralNetworkCalibrator(surrogate_model=surrogate_model)
        if norm_params:
            nn_calibrator.set_normalization(norm_params)
        
        nn_results = nn_calibrator.calibrate(
            S, strikes, maturities, market_prices, r,
            method='differential_evolution',
            verbose=False
        )
        
        results.append({
            'Method': 'NN Surrogate',
            'MSE': nn_results['calibration_error'],
            'RMSE': np.sqrt(nn_results['calibration_error']),
            'Time (s)': nn_results['elapsed_time'],
            'v0': nn_results['v0'],
            'kappa': nn_results['kappa'],
            'theta': nn_results['theta'],
            'sigma': nn_results['sigma'],
            'rho': nn_results['rho']
        })
    
    # 3. Two-Phase Calibration
    print("3. Two-Phase (PAN + CCN) Calibration...")
    two_phase = TwoPhaseCalibrator()
    tp_results = two_phase.full_calibration(
        S, strikes, maturities, market_prices, r,
        pan_epochs=500, ccn_epochs=500, verbose=False
    )
    
    results.append({
        'Method': 'Two-Phase (DL)',
        'MSE': tp_results['mse_corrected'],
        'RMSE': tp_results['rmse_corrected'],
        'Time (s)': tp_results['elapsed_time'],
        'v0': tp_results['heston_params']['v0'],
        'kappa': tp_results['heston_params']['kappa'],
        'theta': tp_results['heston_params']['theta'],
        'sigma': tp_results['heston_params']['sigma'],
        'rho': tp_results['heston_params']['rho']
    })
    
    df = pd.DataFrame(results)
    
    if verbose:
        print("\n" + "=" * 80)
        print("CALIBRATION METHOD COMPARISON")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
    
    return df


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Neural Network Calibration Demo")
    print("=" * 60)
    
    # Generate synthetic market data
    print("\n1. Generating synthetic market data...")
    np.random.seed(42)
    
    S0 = 100.0
    r = 0.03
    
    # True Heston parameters
    true_params = {
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma': 0.3,
        'rho': -0.7
    }
    
    # Generate option grid
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    maturities = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
    
    # True model
    true_model = HestonModel(
        kappa=true_params['kappa'],
        theta=true_params['theta'],
        sigma=true_params['sigma'],
        rho=true_params['rho'],
        v0=true_params['v0'],
        r=r,
        lambd=0.0
    )
    
    # Generate "market" prices (with small noise)
    market_prices = np.array([
        true_model.call_price_fft(S0, k, t)
        for k, t in zip(strikes, maturities)
    ])
    market_prices += np.random.normal(0, 0.1, len(market_prices))  # Add noise
    
    print(f"   True parameters: {true_params}")
    print(f"   Strikes: {strikes}")
    print(f"   Market prices: {np.round(market_prices, 2)}")
    
    # Run two-phase calibration
    print("\n2. Running Two-Phase Calibration...")
    calibrator = TwoPhaseCalibrator()
    results = calibrator.full_calibration(
        S=S0,
        strikes=strikes,
        maturities=maturities,
        market_prices=market_prices,
        r=r,
        pan_epochs=500,
        ccn_epochs=500,
        verbose=True
    )
    
    # Compare with true parameters
    print("\n3. Parameter Recovery:")
    print(f"   {'Parameter':<10} {'True':>10} {'Calibrated':>12} {'Error %':>10}")
    print("   " + "-" * 44)
    for key in ['v0', 'kappa', 'theta', 'sigma', 'rho']:
        true_val = true_params[key]
        calib_val = results['heston_params'][key]
        error_pct = abs(calib_val - true_val) / abs(true_val) * 100
        print(f"   {key:<10} {true_val:>10.4f} {calib_val:>12.4f} {error_pct:>10.1f}%")
    
    print("\nDemo completed!")
