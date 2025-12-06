"""
Market Data Fetcher for Heston Model Calibration with Neural Networks

This script provides functionality for:
1. Fetching current market option data (yfinance)
2. Fetching historical underlying spot prices
3. Computing underlying statistics for Heston parameter estimation
4. Generating synthetic training data across different market regimes
5. Preparing data for NN calibration and benchmarking

Usage:
    # Fetch real market data
    real_data = fetch_market_options('QQQ')
    
    # Fetch historical underlying prices
    hist_data = fetch_underlying_history('QQQ', months=3)  # Use 3 months
    # OR: hist_data = fetch_underlying_history('QQQ', years=2)  # Use 2 years (default)
    
    # Compute underlying statistics
    stats = compute_underlying_statistics(hist_data)
    
    # Derive Heston parameter ranges
    param_ranges = derive_heston_parameter_ranges(stats)
    
    # Generate synthetic training data
    synthetic_data = generate_synthetic_data(n_samples=100000, regime='normal')
    
    # Generate security-specific synthetic data
    synthetic_data = generate_security_specific_data(param_ranges, n_samples=100000)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.heston_model import HestonModel
from src.yield_curve import YieldCurve, get_treasury_yields

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# MARKET REGIMES DEFINITION
# =============================================================================

MARKET_REGIMES = {
    'normal': {
        'name': 'Normal Market',
        'kappa': (0.5, 3.0),      # Mean reversion speed
        'theta': (0.02, 0.08),    # Long-term variance
        'sigma': (0.2, 0.6),      # Vol of vol
        'rho': (-0.7, -0.3),      # Correlation
        'v0': (0.02, 0.08),       # Initial variance
        'r': (0.02, 0.05),        # Risk-free rate
    },
    'low_vol': {
        'name': 'Low Volatility (Bull Market)',
        'kappa': (1.0, 4.0),
        'theta': (0.01, 0.04),
        'sigma': (0.1, 0.4),
        'rho': (-0.5, -0.2),
        'v0': (0.01, 0.04),
        'r': (0.01, 0.04),
    },
    'high_vol': {
        'name': 'High Volatility (Bear Market)',
        'kappa': (0.3, 2.0),
        'theta': (0.08, 0.20),
        'sigma': (0.4, 1.0),
        'rho': (-0.9, -0.6),
        'v0': (0.08, 0.20),
        'r': (0.0, 0.03),
    },
    'crisis': {
        'name': 'Market Crisis (2008/2020 style)',
        'kappa': (0.2, 1.5),
        'theta': (0.15, 0.35),
        'sigma': (0.6, 1.5),
        'rho': (-0.95, -0.75),
        'v0': (0.20, 0.40),
        'r': (0.0, 0.02),
    },
    'recovery': {
        'name': 'Post-Crisis Recovery',
        'kappa': (1.0, 3.5),
        'theta': (0.05, 0.12),
        'sigma': (0.3, 0.8),
        'rho': (-0.8, -0.4),
        'v0': (0.08, 0.15),
        'r': (0.0, 0.03),
    }
}


# =============================================================================
# UNDERLYING SPOT PRICE DATA
# =============================================================================

def fetch_underlying_history(ticker: str, months: int = None, years: float = 2.0) -> pd.DataFrame:
    """
    Fetch historical spot prices for the underlying security.
    
    Args:
        ticker: Stock/ETF ticker symbol (e.g., 'QQQ', 'SPY', 'TSLA')
        months: Number of months of historical data (takes precedence if specified)
        years: Number of years of historical data to fetch (can be fractional, e.g., 0.25 for 3 months)
        
    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume, ticker
        
    Examples:
        fetch_underlying_history('QQQ', months=3)   # 3 months of data
        fetch_underlying_history('QQQ', years=0.25) # Also 3 months
        fetch_underlying_history('QQQ', years=2)    # 2 years (default)
    """
    # Convert months to years if specified
    if months is not None:
        years = months / 12.0
        period_str = f"{months} months"
    else:
        period_str = f"{years} years"
    
    logger.info(f"Fetching {period_str} of historical data for {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(years * 365))
    
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, end=end_date)
    
    # Reset index to have Date as column
    hist = hist.reset_index()
    hist['ticker'] = ticker
    
    logger.info(f"  Retrieved {len(hist)} daily observations")
    logger.info(f"  Date range: {hist['Date'].min().date()} to {hist['Date'].max().date()}")
    logger.info(f"  Price range: ${hist['Close'].min():.2f} - ${hist['Close'].max():.2f}")
    
    return hist


def compute_underlying_statistics(hist_df: pd.DataFrame) -> dict:
    """
    Compute statistics from historical spot prices to inform Heston parameter ranges.
    
    Args:
        hist_df: DataFrame with 'Close' column containing historical prices
        
    Returns:
        dict with:
        - realized_vol: annualized realized volatility (various windows)
        - vol_of_vol: volatility of volatility (informs σ parameter)
        - mean_vol: average realized volatility (informs θ)
        - current_vol: recent volatility (informs v0)
        - return_stats: skewness, kurtosis (informs ρ)
        - kappa_estimate: mean reversion speed estimate
    """
    logger.info("Computing underlying statistics...")
    
    # Daily log returns
    prices = hist_df['Close'].values
    log_returns = np.diff(np.log(prices))
    
    # Annualization factor
    trading_days = 252
    
    # Realized volatility function
    def realized_vol(returns, window):
        """Rolling realized volatility."""
        if len(returns) < window:
            return np.array([np.std(returns) * np.sqrt(trading_days)])
        rolling_std = pd.Series(returns).rolling(window).std().dropna()
        return (rolling_std * np.sqrt(trading_days)).values
    
    # Short-term vol (21 days ~ 1 month)
    vol_21d = realized_vol(log_returns, 21)
    
    # Medium-term vol (63 days ~ 3 months)
    vol_63d = realized_vol(log_returns, 63)
    
    # Current spot volatility (most recent 21 days)
    current_vol = np.std(log_returns[-21:]) * np.sqrt(trading_days) if len(log_returns) >= 21 else np.std(log_returns) * np.sqrt(trading_days)
    
    # Long-term average volatility
    mean_vol = np.std(log_returns) * np.sqrt(trading_days)
    
    # Volatility of volatility (standard deviation of rolling vol)
    vol_of_vol = np.std(vol_21d) if len(vol_21d) > 10 else 0.3
    
    # Return statistics
    skewness = pd.Series(log_returns).skew()
    kurtosis = pd.Series(log_returns).kurtosis()
    
    # Correlation between returns and volatility changes (leverage effect / rho proxy)
    if len(vol_21d) > 21:
        vol_changes = np.diff(vol_21d)
        returns_aligned = log_returns[21:21+len(vol_changes)]
        if len(returns_aligned) == len(vol_changes) and len(returns_aligned) > 0:
            leverage_corr = np.corrcoef(returns_aligned, vol_changes)[0, 1]
            if np.isnan(leverage_corr):
                leverage_corr = -0.5
        else:
            leverage_corr = -0.5
    else:
        leverage_corr = -0.5  # Default negative correlation for equities
    
    # Estimate mean reversion speed (from vol half-life)
    if len(vol_21d) > 50:
        vol_series = vol_21d
        vol_mean = np.mean(vol_series)
        vol_demeaned = vol_series - vol_mean
        if len(vol_demeaned) > 1:
            ar1_coef = np.corrcoef(vol_demeaned[:-1], vol_demeaned[1:])[0, 1]
            if not np.isnan(ar1_coef) and 0 < ar1_coef < 1:
                half_life_days = -np.log(2) / np.log(ar1_coef)
                kappa_estimate = np.log(2) / (half_life_days / trading_days)
                kappa_estimate = np.clip(kappa_estimate, 0.5, 10.0)
            else:
                kappa_estimate = 2.0
        else:
            kappa_estimate = 2.0
    else:
        kappa_estimate = 2.0
    
    stats = {
        # Volatility measures
        'current_vol': float(current_vol),
        'mean_vol': float(mean_vol),
        'vol_21d_mean': float(np.mean(vol_21d)) if len(vol_21d) > 0 else float(current_vol),
        'vol_21d_std': float(np.std(vol_21d)) if len(vol_21d) > 0 else 0.1,
        'vol_21d_min': float(np.min(vol_21d)) if len(vol_21d) > 0 else float(current_vol * 0.5),
        'vol_21d_max': float(np.max(vol_21d)) if len(vol_21d) > 0 else float(current_vol * 2.0),
        'vol_21d_p5': float(np.percentile(vol_21d, 5)) if len(vol_21d) > 10 else float(current_vol * 0.6),
        'vol_21d_p95': float(np.percentile(vol_21d, 95)) if len(vol_21d) > 10 else float(current_vol * 1.5),
        'vol_of_vol': float(vol_of_vol),
        
        # Return statistics
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'leverage_corr': float(leverage_corr),
        
        # Mean reversion
        'kappa_estimate': float(kappa_estimate),
        
        # Price info
        'current_price': float(prices[-1]),
        'price_min': float(np.min(prices)),
        'price_max': float(np.max(prices)),
        
        # Data info
        'n_observations': int(len(prices)),
    }
    
    logger.info(f"  Current volatility (v0 proxy): {current_vol:.2%}")
    logger.info(f"  Long-term volatility (θ proxy): {mean_vol:.2%}")
    logger.info(f"  Vol of vol (σ proxy): {vol_of_vol:.4f}")
    logger.info(f"  Leverage correlation (ρ proxy): {leverage_corr:.3f}")
    logger.info(f"  Mean reversion speed (κ proxy): {kappa_estimate:.2f}")
    
    return stats


def derive_heston_parameter_ranges(stats: dict) -> dict:
    """
    Convert underlying statistics to Heston parameter ranges for synthetic data generation.
    
    Heston parameters:
    - v0: initial variance ~ current_vol^2
    - θ (theta): long-term variance ~ mean_vol^2
    - κ (kappa): mean reversion speed
    - σ (sigma): volatility of variance ~ vol_of_vol
    - ρ (rho): correlation ~ leverage_corr
    
    Args:
        stats: Dictionary from compute_underlying_statistics()
        
    Returns:
        Dictionary with parameter ranges and center values
    """
    logger.info("Deriving Heston parameter ranges from underlying statistics...")
    
    # v0: current variance (variance = vol^2)
    v0_center = stats['current_vol'] ** 2
    v0_range = (
        max(0.001, (stats['vol_21d_p5'] ** 2) * 0.8),
        min(1.0, (stats['vol_21d_p95'] ** 2) * 1.2)
    )
    
    # θ: long-term variance
    theta_center = stats['mean_vol'] ** 2
    theta_range = (
        max(0.001, theta_center * 0.5),
        min(1.0, theta_center * 2.0)
    )
    
    # κ: mean reversion speed (IMPORTANT: keep realistic, typically 0.5-5.0)
    kappa_center = np.clip(stats['kappa_estimate'], 0.5, 4.0)  # Clip to realistic range
    kappa_range = (
        max(0.3, kappa_center * 0.5),
        min(5.0, kappa_center * 2.0)  # REDUCED from 15.0 to 5.0
    )
    
    # σ: vol of vol (scale appropriately, typically 0.2-1.0)
    sigma_center = np.clip(stats['vol_of_vol'] * 2, 0.2, 0.8)  # Clip to realistic range
    sigma_range = (
        max(0.1, sigma_center * 0.5),
        min(1.2, sigma_center * 2.0)  # REDUCED from 3.0 to 1.2
    )
    
    # ρ: leverage effect (usually negative for equities)
    rho_center = np.clip(stats['leverage_corr'], -0.95, 0.0)
    rho_range = (
        max(-0.99, rho_center - 0.3),
        min(0.0, rho_center + 0.3)
    )
    
    ranges = {
        'v0': v0_range,
        'kappa': kappa_range,
        'theta': theta_range,
        'sigma': sigma_range,
        'rho': rho_range,
        # Store centers for reference (useful as starting points)
        'v0_center': v0_center,
        'theta_center': theta_center,
        'kappa_center': kappa_center,
        'sigma_center': sigma_center,
        'rho_center': rho_center,
    }
    
    logger.info(f"  v0 (variance): [{ranges['v0'][0]:.4f}, {ranges['v0'][1]:.4f}] (center: {v0_center:.4f})")
    logger.info(f"  θ (long-term): [{ranges['theta'][0]:.4f}, {ranges['theta'][1]:.4f}] (center: {theta_center:.4f})")
    logger.info(f"  κ (reversion): [{ranges['kappa'][0]:.2f}, {ranges['kappa'][1]:.2f}] (center: {kappa_center:.2f})")
    logger.info(f"  σ (vol of vol): [{ranges['sigma'][0]:.2f}, {ranges['sigma'][1]:.2f}] (center: {sigma_center:.2f})")
    logger.info(f"  ρ (correlation): [{ranges['rho'][0]:.2f}, {ranges['rho'][1]:.2f}] (center: {rho_center:.2f})")
    
    return ranges


def generate_security_specific_data(
    param_ranges: dict,
    n_samples: int = 100000,
    r_range: tuple = (0.01, 0.06),
    moneyness_range: tuple = (0.85, 1.15),
    tau_range: tuple = (0.02, 1.5),
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic Heston training data using security-specific parameter ranges.
    
    Uses importance sampling:
    - More samples near ATM (moneyness = 1.0)
    - More samples near parameter centers
    
    Args:
        param_ranges: Dictionary from derive_heston_parameter_ranges()
        n_samples: Number of samples to generate
        r_range: Risk-free rate range
        moneyness_range: Strike/Spot range (e.g., 0.85-1.15 for near ATM)
        tau_range: Time to maturity range in years
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: log_moneyness, tau, v0, kappa, theta, sigma, rho, r, normalized_price
    """
    logger.info(f"Generating {n_samples:,} security-specific synthetic samples...")
    np.random.seed(seed)
    
    S0 = 100.0  # Normalized spot
    samples = []
    n_valid = 0
    n_attempts = 0
    max_attempts = n_samples * 5
    
    # Pre-compute parameter sampling strategy
    # Mix of uniform (exploration) and centered (exploitation)
    def sample_param_mixed(range_tuple, center, center_weight=0.6):
        """Sample with bias towards center."""
        if np.random.random() < center_weight:
            # Sample from truncated normal near center
            std = (range_tuple[1] - range_tuple[0]) / 4
            val = np.random.normal(center, std)
            return np.clip(val, range_tuple[0], range_tuple[1])
        else:
            # Uniform sampling
            return np.random.uniform(*range_tuple)
    
    while n_valid < n_samples and n_attempts < max_attempts:
        n_attempts += 1
        
        # Sample parameters - mix of uniform and centered sampling
        v0 = sample_param_mixed(param_ranges['v0'], param_ranges['v0_center'])
        kappa = sample_param_mixed(param_ranges['kappa'], param_ranges['kappa_center'])
        theta = sample_param_mixed(param_ranges['theta'], param_ranges['theta_center'])
        sigma = sample_param_mixed(param_ranges['sigma'], param_ranges['sigma_center'])
        rho = sample_param_mixed(param_ranges['rho'], param_ranges['rho_center'])
        r = np.random.uniform(*r_range)
        
        # Feller condition check (ensures variance stays positive)
        if 2 * kappa * theta <= sigma ** 2:
            continue
        
        # IMPROVED: Sample moneyness with MORE weight near ATM
        # 70% near ATM (0.95-1.05), 30% in wings
        if np.random.random() < 0.7:
            # Near ATM: truncated normal centered at 1.0
            moneyness = np.random.normal(1.0, 0.03)
            moneyness = np.clip(moneyness, 0.92, 1.08)
        else:
            # Wings: uniform across full range
            moneyness = np.random.uniform(*moneyness_range)
        
        K = S0 * moneyness
        tau = np.random.uniform(*tau_range)
        
        # Price the option using Heston model
        try:
            model = HestonModel(
                kappa=kappa, theta=theta, sigma=sigma,
                rho=rho, v0=v0, r=r, lambd=0.0
            )
            price = model.call_price_fft(S0, K, tau)
            
            # Skip invalid prices
            if price <= 0 or np.isnan(price) or np.isinf(price) or price < 0.01:
                continue
            
            samples.append({
                'log_moneyness': np.log(moneyness),
                'tau': tau,
                'v0': v0,
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'rho': rho,
                'r': r,
                'normalized_price': price / S0,
            })
            
            n_valid += 1
            
            if n_valid % 10000 == 0:
                logger.info(f"  Generated {n_valid:,}/{n_samples:,} samples...")
            
        except Exception:
            continue
    
    logger.info(f"  Generated {len(samples):,} valid samples ({n_attempts:,} attempts)")
    logger.info(f"  Success rate: {len(samples)/n_attempts*100:.1f}%")
    
    return pd.DataFrame(samples)


# =============================================================================
# YIELD CURVE UTILITIES
# =============================================================================

def load_yield_curve(date=None):
    """
    Load and fit yield curve from Treasury data.
    
    Args:
        date: Date for yield curve (default: today)
        
    Returns:
        YieldCurve object fitted to Treasury data
    """
    # Get Treasury yields for the date
    treasury_data = get_treasury_yields(date=date)
    
    # Create and fit yield curve
    yc = YieldCurve(method='nss')  # Nelson-Siegel-Svensson
    yc.fit_from_treasury_data(
        maturities=treasury_data['maturities'],
        yields=treasury_data['yields']
    )
    
    logger.info(f"Loaded yield curve with {len(treasury_data['maturities'])} points")
    logger.info(f"  Short rate (3m): {yc.get_rate(0.25):.4f}")
    logger.info(f"  Medium rate (1y): {yc.get_rate(1.0):.4f}")
    logger.info(f"  Long rate (10y): {yc.get_rate(10.0):.4f}")
    
    return yc


# =============================================================================
# REAL MARKET DATA FETCHING
# =============================================================================

def fetch_market_options(ticker='QQQ', min_volume=10, min_oi=50, use_yield_curve=True):
    """
    Fetch current market option data for a given ticker.
    
    Args:
        ticker: Stock ticker symbol (default: QQQ)
        min_volume: Minimum daily volume filter
        min_oi: Minimum open interest filter
        use_yield_curve: If True, compute risk-free rate from yield curve for each maturity
        
    Returns:
        DataFrame with option data including:
        - expiry, tau, strike, type, bid, ask, mid_price
        - volume, openInterest, impliedVolatility
        - spot, moneyness, r (risk-free rate from yield curve)
    """
    logger.info(f"Fetching option data for {ticker}...")
    
    # Load yield curve if requested
    yc = None
    if use_yield_curve:
        try:
            yc = load_yield_curve()
            logger.info("Using yield curve for risk-free rates")
        except Exception as e:
            logger.warning(f"Could not load yield curve: {e}. Using constant r=0.03")
            yc = None
    
    try:
        # Get ticker object
        stock = yf.Ticker(ticker)
        
        # Get current stock price
        hist = stock.history(period='1d')
        if hist.empty:
            raise ValueError(f"Could not fetch stock price for {ticker}")
        current_price = hist['Close'].iloc[-1]
        
        # Get available expiration dates
        expirations = stock.options
        logger.info(f"Found {len(expirations)} expiration dates")
        
        option_data = []
        
        for expiry in expirations:
            try:
                # Get option chain
                opt_chain = stock.option_chain(expiry)
                
                # Calculate time to maturity
                expiry_date = pd.to_datetime(expiry)
                today = pd.Timestamp.now().normalize()
                tau = (expiry_date - today).days / 365.25
                
                if tau <= 0.02:  # Skip options expiring in < 1 week
                    continue
                
                # Get risk-free rate for this maturity
                if yc is not None:
                    r = yc.get_rate(tau)
                else:
                    r = 0.03  # Default constant rate
                
                # Process calls
                for _, row in opt_chain.calls.iterrows():
                    if row['volume'] >= min_volume and row['openInterest'] >= min_oi:
                        option_data.append({
                            'expiry': expiry,
                            'tau': tau,
                            'strike': row['strike'],
                            'type': 'call',
                            'bid': row['bid'],
                            'ask': row['ask'],
                            'last': row['lastPrice'],
                            'volume': row['volume'],
                            'openInterest': row['openInterest'],
                            'impliedVolatility': row['impliedVolatility'],
                            'spot': current_price,
                            'r': r  # Risk-free rate from yield curve
                        })
                
                # Process puts
                for _, row in opt_chain.puts.iterrows():
                    if row['volume'] >= min_volume and row['openInterest'] >= min_oi:
                        option_data.append({
                            'expiry': expiry,
                            'tau': tau,
                            'strike': row['strike'],
                            'type': 'put',
                            'bid': row['bid'],
                            'ask': row['ask'],
                            'last': row['lastPrice'],
                            'volume': row['volume'],
                            'openInterest': row['openInterest'],
                            'impliedVolatility': row['impliedVolatility'],
                            'spot': current_price,
                            'r': r  # Risk-free rate from yield curve
                        })
                        
            except Exception as e:
                logger.warning(f"Error processing expiry {expiry}: {e}")
                continue
        
        if not option_data:
            raise ValueError("No option data found after filtering")
        
        # Create DataFrame
        df = pd.DataFrame(option_data)
        
        # Calculate derived fields
        df['mid_price'] = (df['bid'] + df['ask']) / 2
        df['moneyness'] = df['strike'] / df['spot']
        df['log_moneyness'] = np.log(df['moneyness'])
        
        # Filter out bad data
        df = df[
            (df['bid'] > 0) & 
            (df['ask'] > 0) &
            (df['ask'] > df['bid']) &
            (df['impliedVolatility'] > 0) &
            (df['impliedVolatility'] < 2.0)  # Remove outliers
        ]
        
        logger.info(f"Successfully fetched {len(df)} option quotes")
        logger.info(f"  Calls: {len(df[df['type']=='call'])}")
        logger.info(f"  Puts: {len(df[df['type']=='put'])}")
        logger.info(f"  Maturities: {df['tau'].min():.2f} to {df['tau'].max():.2f} years")
        logger.info(f"  Strikes: {df['strike'].min():.2f} to {df['strike'].max():.2f}")
        logger.info(f"  Current spot: ${current_price:.2f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        raise


def save_market_snapshot(df, ticker='QQQ', output_dir='data/raw'):
    """Save market data snapshot with timestamp."""
    output_path = PROJECT_ROOT / output_dir / ticker.lower()
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = output_path / f'options_snapshot_{timestamp}.csv'
    
    df.to_csv(filename, index=False)
    logger.info(f"Saved market snapshot to {filename}")
    
    return filename


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_data(n_samples=100000, regime='normal', seed=None):
    """
    Generate synthetic option data for NN training/testing.
    
    Args:
        n_samples: Number of synthetic options to generate
        regime: Market regime ('normal', 'low_vol', 'high_vol', 'crisis', 'recovery')
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic option data including:
        - Market observables: S0, K, tau, r, option_price
        - True parameters: kappa, theta, sigma, rho, v0
        - Additional features: moneyness, log_moneyness, type
    """
    if seed is not None:
        np.random.seed(seed)
    
    if regime not in MARKET_REGIMES:
        raise ValueError(f"Unknown regime '{regime}'. Choose from: {list(MARKET_REGIMES.keys())}")
    
    logger.info(f"Generating {n_samples} synthetic options for regime: {MARKET_REGIMES[regime]['name']}")
    
    regime_params = MARKET_REGIMES[regime]
    data = []
    
    for i in range(n_samples):
        # Sample Heston parameters from regime ranges
        kappa = np.random.uniform(*regime_params['kappa'])
        theta = np.random.uniform(*regime_params['theta'])
        sigma = np.random.uniform(*regime_params['sigma'])
        rho = np.random.uniform(*regime_params['rho'])
        v0 = np.random.uniform(*regime_params['v0'])
        r = np.random.uniform(*regime_params['r'])
        
        # Ensure Feller condition: 2*kappa*theta > sigma^2
        if 2 * kappa * theta <= sigma**2:
            # Adjust kappa to satisfy Feller
            kappa = (sigma**2 / (2 * theta)) * 1.1
        
        # Sample market conditions
        S0 = 100.0  # Normalize to 100
        
        # Sample strike with variety of moneyness levels
        moneyness = np.random.choice([
            np.random.uniform(0.8, 0.95),   # OTM
            np.random.uniform(0.95, 1.05),  # ATM
            np.random.uniform(1.05, 1.2)    # ITM
        ], p=[0.4, 0.4, 0.2])  # More ATM/OTM for training
        
        K = S0 * moneyness
        
        # Sample maturity (more short-term options)
        tau = np.random.choice([
            np.random.uniform(0.08, 0.25),  # 1-3 months
            np.random.uniform(0.25, 0.75),  # 3-9 months
            np.random.uniform(0.75, 2.0)    # 9-24 months
        ], p=[0.5, 0.35, 0.15])
        
        # Randomly choose call or put
        option_type = np.random.choice(['call', 'put'])
        
        try:
            # Create Heston model and compute price
            model = HestonModel(kappa, theta, sigma, rho, v0, r, q=0.0)
            
            # Use FFT for speed
            call_price = model.call_price_fft(S0, K, tau, N=4096, eta=0.25, alpha=1.5)
            
            # Convert to put if needed using put-call parity
            if option_type == 'put':
                option_price = call_price - S0 + K * np.exp(-r * tau)
            else:
                option_price = call_price
            
            # Only keep valid prices
            if option_price > 0 and not np.isnan(option_price) and not np.isinf(option_price):
                data.append({
                    # Market observables (inputs to NN)
                    'S0': S0,
                    'K': K,
                    'tau': tau,
                    'r': r,
                    'option_price': option_price,
                    'option_type': option_type,
                    'moneyness': moneyness,
                    'log_moneyness': np.log(moneyness),
                    
                    # True parameters (targets for NN)
                    'kappa': kappa,
                    'theta': theta,
                    'sigma': sigma,
                    'rho': rho,
                    'v0': v0,
                    
                    # Metadata
                    'regime': regime
                })
                
        except Exception as e:
            # Skip problematic parameter combinations
            continue
        
        # Progress logging
        if (i + 1) % 10000 == 0:
            logger.info(f"  Generated {i + 1}/{n_samples} samples ({len(data)} valid)")
    
    df = pd.DataFrame(data)
    logger.info(f"Successfully generated {len(df)} valid synthetic options")
    
    return df


def generate_multi_regime_dataset(samples_per_regime=20000, seed=None):
    """
    Generate synthetic dataset spanning multiple market regimes.
    
    Args:
        samples_per_regime: Number of samples per regime
        seed: Random seed
        
    Returns:
        DataFrame with mixed regime data
    """
    logger.info("Generating multi-regime dataset...")
    
    all_data = []
    
    for regime in MARKET_REGIMES.keys():
        regime_data = generate_synthetic_data(
            n_samples=samples_per_regime,
            regime=regime,
            seed=seed
        )
        all_data.append(regime_data)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Shuffle
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    logger.info(f"Total dataset size: {len(combined)} options")
    logger.info(f"Regime distribution:\n{combined['regime'].value_counts()}")
    
    return combined


def save_synthetic_data(df, name='synthetic_train', output_dir='data/processed'):
    """Save synthetic data to CSV."""
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    filename = output_path / f'{name}.csv'
    df.to_csv(filename, index=False)
    
    logger.info(f"Saved synthetic data to {filename}")
    return filename


# =============================================================================
# DATA PREPARATION FOR NN CALIBRATION
# =============================================================================

def prepare_calibration_set(df, target_spot=None):
    """
    Prepare option data for calibration (real or synthetic).
    
    Groups options by common spot/date for simultaneous calibration.
    
    Args:
        df: DataFrame with option data
        target_spot: Filter for specific spot price (for synthetic data)
        
    Returns:
        List of calibration sets, each containing:
        - prices: array of option prices
        - strikes: array of strikes
        - taus: array of maturities
        - types: array of 'call'/'put'
        - spot: spot price
        - r: risk-free rate (single value or array matching taus if from yield curve)
    """
    if target_spot is not None:
        df = df[np.abs(df['S0'] - target_spot) < 0.01]
    
    # Group by spot price and date (for real data) or just spot (for synthetic)
    if 'expiry' in df.columns:
        # Real market data - group by trading day
        groups = df.groupby('spot')
    else:
        # Synthetic data - group by spot
        groups = df.groupby('S0')
    
    calibration_sets = []
    
    for spot_price, group in groups:
        # Determine if we have term structure of rates or constant rate
        if 'r' in group.columns:
            # Check if rates vary by maturity (yield curve)
            unique_rates = group['r'].nunique()
            if unique_rates > 1:
                # Term structure: different rate for each option
                r_values = group['r'].values
            else:
                # Constant rate
                r_values = group['r'].values[0]
        else:
            # Synthetic data without r column
            r_values = 0.03
        
        calib_set = {
            'prices': group['mid_price'].values if 'mid_price' in group.columns else group['option_price'].values,
            'strikes': group['K'].values if 'K' in group.columns else group['strike'].values,
            'taus': group['tau'].values,
            'types': group['option_type'].values if 'option_type' in group.columns else group['type'].values,
            'spot': spot_price,
            'r': r_values,  # Can be scalar or array
            'n_options': len(group)
        }
        
        # Include true parameters if available (synthetic data)
        if 'kappa' in group.columns:
            calib_set['true_params'] = {
                'kappa': group['kappa'].values[0],
                'theta': group['theta'].values[0],
                'sigma': group['sigma'].values[0],
                'rho': group['rho'].values[0],
                'v0': group['v0'].values[0]
            }
        
        calibration_sets.append(calib_set)
    
    logger.info(f"Prepared {len(calibration_sets)} calibration sets")
    
    return calibration_sets


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_regime_info():
    """Return information about available market regimes."""
    return pd.DataFrame([
        {
            'regime': regime,
            'name': params['name'],
            'v0_range': f"{params['v0'][0]:.2f}-{params['v0'][1]:.2f}",
            'rho_range': f"{params['rho'][0]:.2f}-{params['rho'][1]:.2f}"
        }
        for regime, params in MARKET_REGIMES.items()
    ])