#!/usr/bin/env python3
"""
Heston Model Deep Learning Calibration - Main Workflow

This script runs the complete per-security calibration workflow:
1. Load data (historical spot + options) from data/folder
2. Compute underlying statistics
3. Generate synthetic training data
4. Train surrogate neural network
5. Calibrate to market options
6. Save results

Usage:
    python main.py --tickers QQQ SPY TSLA
    python main.py --tickers QQQ --epochs 300 --samples 200000
    python main.py --tickers AAPL --skip-training  # Use existing model

Data must be fetched first using the data_collection.ipynb notebook.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import optimize
from datetime import datetime
from time import time
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.heston_model import HestonModel
from src.neural_networks import HestonSurrogateNetwork
from src.data_generation import (
    compute_underlying_statistics,
    derive_heston_parameter_ranges,
    generate_security_specific_data
)

# Device selection
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_underlying_history(ticker: str, data_dir: Path, 
                            months: int = None, years: float = None) -> pd.DataFrame:
    """
    Load historical spot prices for a ticker.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Data directory path
        months: If specified, use only the most recent N months of data
        years: If specified (and months is None), use most recent N years
        
    Returns:
        DataFrame with historical price data
    """
    hist_file = data_dir / 'raw' / 'underlying' / f'{ticker}_history.csv'
    
    if not hist_file.exists():
        raise FileNotFoundError(
            f"Historical data not found: {hist_file}\n"
            f"Please run data_collection.ipynb first to fetch data for {ticker}"
        )
    
    df = pd.read_csv(hist_file, parse_dates=['Date'])
    
    # Filter to specified time period if requested
    if months is not None or years is not None:
        # Convert months to days, or years to days
        if months is not None:
            days_to_keep = int(months * 30.44)  # Average days per month
            period_str = f"{months} months"
        else:
            days_to_keep = int(years * 365)
            period_str = f"{years} years"
        
        # Get most recent date and filter
        max_date = df['Date'].max()
        cutoff_date = max_date - pd.Timedelta(days=days_to_keep)
        df_filtered = df[df['Date'] >= cutoff_date].copy()
        
        logger.info(f"Filtered history to {period_str}: {len(df_filtered)} observations "
                   f"(from {len(df)} total)")
        logger.info(f"  Date range: {df_filtered['Date'].min().date()} to {df_filtered['Date'].max().date()}")
        df = df_filtered
    else:
        logger.info(f"Loaded {len(df)} historical observations for {ticker} (full history)")
    
    return df


def load_options_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load options data for calibration."""
    calib_file = data_dir / 'processed' / 'calibration' / f'{ticker}_calibration.csv'
    
    if not calib_file.exists():
        raise FileNotFoundError(
            f"Calibration data not found: {calib_file}\n"
            f"Please run data_collection.ipynb first to fetch data for {ticker}"
        )
    
    df = pd.read_csv(calib_file)
    logger.info(f"Loaded {len(df)} options for {ticker} calibration")
    return df


# =============================================================================
# TRAINING
# =============================================================================

def train_surrogate_network(
    synthetic_df: pd.DataFrame,
    hidden_layers: list = [128, 128, 64, 32],
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    patience: int = 25,
    device: torch.device = None
) -> tuple:
    """
    Train surrogate network on synthetic data.
    
    Returns: (model, norm_params, history)
    """
    if device is None:
        device = get_device()
    
    feature_cols = ['log_moneyness', 'tau', 'v0', 'kappa', 'theta', 'sigma', 'rho', 'r']
    
    X = torch.tensor(synthetic_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(synthetic_df['normalized_price'].values, dtype=torch.float32).reshape(-1, 1)
    
    # Normalization
    X_mean = X.mean(dim=0)
    X_std = X.std(dim=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    # Train/val split (90/10)
    n_train = int(0.9 * len(X))
    indices = torch.randperm(len(X))
    
    train_dataset = TensorDataset(X_norm[indices[:n_train]], y[indices[:n_train]])
    val_dataset = TensorDataset(X_norm[indices[n_train:]], y[indices[n_train:]])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = HestonSurrogateNetwork(
        hidden_layers=hidden_layers,
        batch_norm=True,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    logger.info(f"Training on {len(train_dataset):,} samples, validating on {len(val_dataset):,}")
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model.network(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(X_batch)
        train_loss /= len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model.network(X_batch)
                val_loss += criterion(pred, y_batch).item() * len(X_batch)
        val_loss /= len(val_dataset)
        
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 25 == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(f"  Epoch {epoch+1:3d}/{epochs}: train={train_loss:.6f}, val={val_loss:.6f}, lr={lr:.2e}")
        
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Load best state
    if best_state:
        model.load_state_dict(best_state)
    
    norm_params = {
        'X_mean': X_mean.numpy(),
        'X_std': X_std.numpy()
    }
    
    logger.info(f"  Best validation loss: {best_val_loss:.6f}")
    
    return model, norm_params, history


# =============================================================================
# CALIBRATION
# =============================================================================

def traditional_calibration(S, strikes, maturities, market_prices, r, param_ranges):
    """Traditional Heston calibration using SLSQP."""
    def objective(params):
        v0, kappa, theta, sigma, rho = params
        if 2 * kappa * theta <= sigma ** 2:
            return 1e10
        try:
            model = HestonModel(kappa=kappa, theta=theta, sigma=sigma,
                               rho=rho, v0=v0, r=r, lambd=0.0)
            pred = np.array([model.call_price_fft(S, k, t) 
                           for k, t in zip(strikes, maturities)])
            return np.mean((pred - market_prices) ** 2)
        except:
            return 1e10
    
    bounds = [
        param_ranges['v0'],
        param_ranges['kappa'],
        param_ranges['theta'],
        param_ranges['sigma'],
        param_ranges['rho']
    ]
    
    x0 = [
        param_ranges['v0_center'],
        param_ranges['kappa_center'],
        param_ranges['theta_center'],
        param_ranges['sigma_center'],
        param_ranges['rho_center']
    ]
    
    start = time()
    result = optimize.minimize(objective, x0=x0, method='SLSQP', bounds=bounds,
                              options={'maxiter': 500})
    elapsed = time() - start
    
    return {
        'v0': result.x[0], 'kappa': result.x[1], 'theta': result.x[2],
        'sigma': result.x[3], 'rho': result.x[4],
        'mse': result.fun, 'time': elapsed, 'success': result.success
    }


def nn_calibration(model, norm_params, S, strikes, maturities, market_prices, r, 
                   param_ranges, device):
    """Neural network surrogate calibration using differential evolution."""
    X_mean = norm_params['X_mean']
    X_std = norm_params['X_std']
    
    def objective(params):
        v0, kappa, theta, sigma, rho = params
        if 2 * kappa * theta <= sigma ** 2:
            return 1e10
        
        features = np.array([[np.log(k/S), t, v0, kappa, theta, sigma, rho, r]
                            for k, t in zip(strikes, maturities)])
        features_norm = (features - X_mean) / X_std
        
        with torch.no_grad():
            X = torch.tensor(features_norm, dtype=torch.float32).to(device)
            pred_norm = model.network(X).cpu().numpy().flatten()
        
        pred_prices = pred_norm * S
        return np.mean((pred_prices - market_prices) ** 2)
    
    bounds = [
        param_ranges['v0'],
        param_ranges['kappa'],
        param_ranges['theta'],
        param_ranges['sigma'],
        param_ranges['rho']
    ]
    
    start = time()
    result = optimize.differential_evolution(objective, bounds=bounds, maxiter=200, 
                                            seed=42, workers=1, disp=False)
    elapsed = time() - start
    
    return {
        'v0': result.x[0], 'kappa': result.x[1], 'theta': result.x[2],
        'sigma': result.x[3], 'rho': result.x[4],
        'mse': result.fun, 'time': elapsed, 'success': result.success
    }


def hybrid_calibration(model, norm_params, S, strikes, maturities, market_prices, r,
                       param_ranges, device):
    """
    Two-stage hybrid calibration:
    1. Fast NN calibration to find good initial point
    2. Traditional optimization to refine (fewer iterations)
    
    This combines NN speed with traditional accuracy.
    """
    start_total = time()
    
    # Stage 1: Fast NN calibration
    X_mean = norm_params['X_mean']
    X_std = norm_params['X_std']
    
    def nn_objective(params):
        v0, kappa, theta, sigma, rho = params
        if 2 * kappa * theta <= sigma ** 2:
            return 1e10
        
        features = np.array([[np.log(k/S), t, v0, kappa, theta, sigma, rho, r]
                            for k, t in zip(strikes, maturities)])
        features_norm = (features - X_mean) / X_std
        
        with torch.no_grad():
            X = torch.tensor(features_norm, dtype=torch.float32).to(device)
            pred_norm = model.network(X).cpu().numpy().flatten()
        
        pred_prices = pred_norm * S
        return np.mean((pred_prices - market_prices) ** 2)
    
    bounds = [
        param_ranges['v0'],
        param_ranges['kappa'],
        param_ranges['theta'],
        param_ranges['sigma'],
        param_ranges['rho']
    ]
    
    # Fast NN stage - reduced iterations, smaller population
    nn_result = optimize.differential_evolution(
        nn_objective, bounds=bounds, maxiter=50, seed=42, 
        workers=1, disp=False, popsize=10, tol=0.01
    )
    
    nn_time = time() - start_total
    
    # Stage 2: Refine with traditional FFT-based optimization
    def fft_objective(params):
        v0, kappa, theta, sigma, rho = params
        if 2 * kappa * theta <= sigma ** 2:
            return 1e10
        try:
            heston = HestonModel(kappa=kappa, theta=theta, sigma=sigma,
                                rho=rho, v0=v0, r=r, lambd=0.0)
            pred = np.array([heston.call_price_fft(S, k, t) 
                           for k, t in zip(strikes, maturities)])
            return np.mean((pred - market_prices) ** 2)
        except:
            return 1e10
    
    # Start from NN solution
    x0 = nn_result.x
    
    # Refine with SLSQP - fewer iterations since starting from good point
    refined = optimize.minimize(
        fft_objective, x0=x0, method='SLSQP', bounds=bounds,
        options={'maxiter': 50, 'ftol': 1e-8}  # Reduced from 100
    )
    
    total_time = time() - start_total
    
    return {
        'v0': refined.x[0], 'kappa': refined.x[1], 'theta': refined.x[2],
        'sigma': refined.x[3], 'rho': refined.x[4],
        'mse': refined.fun, 'time': total_time, 'success': refined.success,
        'nn_time': nn_time,  # Track NN stage time
        'nn_initial': nn_result.x.tolist()  # Track NN initial guess
    }


def compute_pricing_metrics(params, S, strikes, maturities, market_prices, r):
    """Compute pricing metrics for calibrated parameters."""
    model = HestonModel(
        v0=params['v0'], kappa=params['kappa'], theta=params['theta'],
        sigma=params['sigma'], rho=params['rho'], r=r
    )
    
    pred_prices = np.array([model.call_price_fft(S, k, t) 
                           for k, t in zip(strikes, maturities)])
    
    errors = pred_prices - market_prices
    abs_errors = np.abs(errors)
    
    return {
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mae': float(np.mean(abs_errors)),
        'mape': float(np.mean(abs_errors / market_prices) * 100),
        'max_error': float(np.max(abs_errors)),
        'pred_prices': pred_prices.tolist()
    }


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def process_ticker(
    ticker: str,
    data_dir: Path,
    output_dir: Path,
    n_samples: int = 100000,
    epochs: int = 200,
    hidden_layers: list = [128, 128, 64, 32],
    skip_training: bool = False,
    history_months: int = None,
    history_years: float = None,
    device: torch.device = None
) -> dict:
    """
    Run full workflow for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        data_dir: Data directory path
        output_dir: Output directory for models
        n_samples: Number of synthetic samples for training
        epochs: Maximum training epochs
        hidden_layers: Neural network architecture
        skip_training: If True, load existing model instead of training
        history_months: Use only recent N months of underlying history (default: all)
        history_years: Use only recent N years of underlying history (default: all)
        device: PyTorch device
    
    Returns dict with all results.
    """
    if device is None:
        device = get_device()
    
    logger.info("=" * 70)
    logger.info(f"PROCESSING: {ticker}")
    logger.info("=" * 70)
    
    # Log history period being used
    if history_months is not None:
        logger.info(f"Using {history_months} months of underlying history for parameter derivation")
    elif history_years is not None:
        logger.info(f"Using {history_years} years of underlying history for parameter derivation")
    else:
        logger.info("Using full underlying history for parameter derivation")
    
    results = {
        'ticker': ticker,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_samples': n_samples,
            'epochs': epochs,
            'hidden_layers': hidden_layers,
            'history_months': history_months,
            'history_years': history_years
        }
    }
    
    # -------------------------------------------------------------------------
    # STEP 1: Load Data
    # -------------------------------------------------------------------------
    logger.info("\n[STEP 1] Loading data...")
    
    hist_df = load_underlying_history(ticker, data_dir, 
                                       months=history_months, 
                                       years=history_years)
    options_df = load_options_data(ticker, data_dir)
    
    results['n_historical_obs'] = len(hist_df)
    results['n_options'] = len(options_df)
    results['spot'] = float(options_df['spot'].iloc[0])
    
    # -------------------------------------------------------------------------
    # STEP 2: Compute Statistics
    # -------------------------------------------------------------------------
    logger.info("\n[STEP 2] Computing underlying statistics...")
    
    stats = compute_underlying_statistics(hist_df)
    results['underlying_stats'] = stats
    
    # -------------------------------------------------------------------------
    # STEP 3: Derive Parameter Ranges
    # -------------------------------------------------------------------------
    logger.info("\n[STEP 3] Deriving Heston parameter ranges...")
    
    param_ranges = derive_heston_parameter_ranges(stats)
    results['param_ranges'] = {
        k: list(v) if isinstance(v, tuple) else float(v)
        for k, v in param_ranges.items()
    }
    
    # -------------------------------------------------------------------------
    # STEP 4: Train or Load Model
    # -------------------------------------------------------------------------
    model_path = output_dir / f'{ticker}_surrogate.pth'
    
    if skip_training and model_path.exists():
        logger.info(f"\n[STEP 4] Loading existing model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model = HestonSurrogateNetwork(
            hidden_layers=checkpoint['hidden_layers'],
            batch_norm=True,
            dropout=0.1
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        norm_params = {
            'X_mean': checkpoint['X_mean'],
            'X_std': checkpoint['X_std']
        }
        results['training_epochs'] = checkpoint.get('epochs_trained', 'loaded')
        
    else:
        logger.info(f"\n[STEP 4] Generating {n_samples:,} synthetic samples...")
        synthetic_df = generate_security_specific_data(param_ranges, n_samples=n_samples)
        results['n_synthetic_samples'] = len(synthetic_df)
        
        logger.info("\n[STEP 5] Training surrogate network...")
        model, norm_params, history = train_surrogate_network(
            synthetic_df,
            hidden_layers=hidden_layers,
            epochs=epochs,
            device=device
        )
        results['training_epochs'] = len(history['train_loss'])
        results['final_val_loss'] = float(history['val_loss'][-1])
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'hidden_layers': hidden_layers,
            'X_mean': norm_params['X_mean'],
            'X_std': norm_params['X_std'],
            'param_ranges': param_ranges,
            'underlying_stats': stats,
            'ticker': ticker,
            'epochs_trained': results['training_epochs']
        }, model_path)
        logger.info(f"  Model saved to: {model_path}")
    
    # -------------------------------------------------------------------------
    # STEP 5/6: Calibrate to Market
    # -------------------------------------------------------------------------
    logger.info("\n[STEP 6] Calibrating to market data...")
    
    S0 = options_df['spot'].iloc[0]
    r = options_df['r'].iloc[0] if 'r' in options_df.columns else 0.04
    strikes = options_df['strike'].values
    maturities = options_df['tau'].values
    market_prices = options_df['mid_price'].values
    
    logger.info(f"  Calibrating with {len(strikes)} options")
    
    # Traditional calibration
    logger.info("  Running traditional calibration...")
    trad_params = traditional_calibration(S0, strikes, maturities, market_prices, r, param_ranges)
    trad_metrics = compute_pricing_metrics(trad_params, S0, strikes, maturities, market_prices, r)
    
    # NN calibration
    logger.info("  Running NN calibration...")
    model.eval()
    nn_params = nn_calibration(model, norm_params, S0, strikes, maturities, market_prices,
                               r, param_ranges, device)
    nn_metrics = compute_pricing_metrics(nn_params, S0, strikes, maturities, market_prices, r)
    
    # Hybrid calibration (NN initial + traditional refinement)
    logger.info("  Running hybrid calibration (NN → Traditional refinement)...")
    hybrid_params = hybrid_calibration(model, norm_params, S0, strikes, maturities, market_prices,
                                       r, param_ranges, device)
    hybrid_metrics = compute_pricing_metrics(hybrid_params, S0, strikes, maturities, market_prices, r)
    
    # Store results
    results['traditional'] = {
        'params': {k: float(v) for k, v in trad_params.items() if k not in ['time', 'success']},
        'time': trad_params['time'],
        'success': trad_params['success'],
        **{k: v for k, v in trad_metrics.items() if k != 'pred_prices'}
    }
    
    results['nn_surrogate'] = {
        'params': {k: float(v) for k, v in nn_params.items() if k not in ['time', 'success']},
        'time': nn_params['time'],
        'success': nn_params['success'],
        **{k: v for k, v in nn_metrics.items() if k != 'pred_prices'}
    }
    
    results['hybrid'] = {
        'params': {k: float(v) for k, v in hybrid_params.items() 
                   if k not in ['time', 'success', 'nn_time', 'nn_initial']},
        'time': hybrid_params['time'],
        'nn_time': hybrid_params.get('nn_time', 0),
        'success': hybrid_params['success'],
        **{k: v for k, v in hybrid_metrics.items() if k != 'pred_prices'}
    }
    
    results['speedup'] = trad_params['time'] / nn_params['time'] if nn_params['time'] > 0 else 0
    results['hybrid_speedup'] = trad_params['time'] / hybrid_params['time'] if hybrid_params['time'] > 0 else 0
    
    # -------------------------------------------------------------------------
    # Print Results
    # -------------------------------------------------------------------------
    logger.info("\n" + "-" * 70)
    logger.info(f"RESULTS FOR {ticker}")
    logger.info("-" * 70)
    logger.info(f"\n{'Method':<18} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'Time (s)':>10} {'Speedup':>10}")
    logger.info("-" * 70)
    logger.info(f"{'Traditional':<18} ${trad_metrics['rmse']:>9.4f} ${trad_metrics['mae']:>9.4f} "
                f"{trad_metrics['mape']:>9.2f}% {trad_params['time']:>9.2f} {'1.0x':>10}")
    logger.info(f"{'NN Surrogate':<18} ${nn_metrics['rmse']:>9.4f} ${nn_metrics['mae']:>9.4f} "
                f"{nn_metrics['mape']:>9.2f}% {nn_params['time']:>9.2f} {results['speedup']:>9.1f}x")
    logger.info(f"{'Hybrid (NN→Trad)':<18} ${hybrid_metrics['rmse']:>9.4f} ${hybrid_metrics['mae']:>9.4f} "
                f"{hybrid_metrics['mape']:>9.2f}% {hybrid_params['time']:>9.2f} {results['hybrid_speedup']:>9.1f}x")
    
    # Identify best method
    rmse_vals = {'Traditional': trad_metrics['rmse'], 
                 'NN': nn_metrics['rmse'], 
                 'Hybrid': hybrid_metrics['rmse']}
    best = min(rmse_vals, key=rmse_vals.get)
    logger.info(f"\nBest Method: {best} (RMSE: ${rmse_vals[best]:.4f})")
    
    # Save results JSON
    results_path = output_dir / f'{ticker}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Heston Model Deep Learning Calibration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --tickers QQQ SPY TSLA
    python main.py --tickers QQQ --epochs 300 --samples 200000
    python main.py --tickers AAPL --skip-training
    python main.py --tickers QQQ --history-months 3  # Use 3 months of underlying data
    python main.py --tickers QQQ --history-years 0.5  # Use 6 months of underlying data

Note: Run data_collection.ipynb first to fetch market data.
        """
    )
    
    parser.add_argument('--tickers', nargs='+', required=True,
                        help='Ticker symbols to process (e.g., QQQ SPY TSLA)')
    parser.add_argument('--samples', type=int, default=100000,
                        help='Number of synthetic samples to generate (default: 100000)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs (default: 200)')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training and use existing model')
    parser.add_argument('--history-months', type=int, default=None,
                        help='Months of underlying history for parameter derivation (default: use all data)')
    parser.add_argument('--history-years', type=float, default=None,
                        help='Years of underlying history (default: use all data, ignored if --history-months set)')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Data directory (default: PROJECT_ROOT/data)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for models (default: PROJECT_ROOT/data/models)')
    
    args = parser.parse_args()
    
    # Setup directories
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / 'data'
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Process each ticker
    all_results = {}
    
    for ticker in args.tickers:
        try:
            results = process_ticker(
                ticker=ticker,
                data_dir=data_dir,
                output_dir=output_dir,
                n_samples=args.samples,
                epochs=args.epochs,
                skip_training=args.skip_training,
                history_months=args.history_months,
                history_years=args.history_years if args.history_months is None else None,
                device=device
            )
            all_results[ticker] = results
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if len(all_results) > 1:
        logger.info("\n" + "=" * 85)
        logger.info("SUMMARY: ALL TICKERS")
        logger.info("=" * 85)
        
        logger.info(f"\n{'Ticker':<8} {'Spot':>10} {'Trad RMSE':>12} {'NN RMSE':>12} {'Hybrid RMSE':>12} {'Hybrid Speedup':>15}")
        logger.info("-" * 75)
        
        for ticker, res in all_results.items():
            hybrid_rmse = res.get('hybrid', {}).get('rmse', 0)
            hybrid_speedup = res.get('hybrid_speedup', 0)
            logger.info(f"{ticker:<8} ${res['spot']:>9.2f} "
                       f"${res['traditional']['rmse']:>11.4f} "
                       f"${res['nn_surrogate']['rmse']:>11.4f} "
                       f"${hybrid_rmse:>11.4f} "
                       f"{hybrid_speedup:>14.1f}x")
    
    # Save master results
    master_path = output_dir / 'master_results.json'
    with open(master_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'tickers': args.tickers,
            'results': all_results
        }, f, indent=2, default=str)
    logger.info(f"\nMaster results saved to: {master_path}")
    
    return all_results


if __name__ == '__main__':
    main()
