"""
Yield Curve Construction using Nelson-Siegel-Svensson Model.

Constructs risk-free rate term structure from Treasury data.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_nss_ols

logger = logging.getLogger(__name__)


class YieldCurve:
    """
    Yield curve construction and interpolation.
    """
    
    def __init__(self, method: str = 'nss'):
        """
        Initialize yield curve.
        
        Args:
            method: 'nss' (Nelson-Siegel-Svensson) or 'linear'
        """
        self.method = method
        self.curve = None
        self.maturities = None
        self.yields = None
    
    def fit_from_treasury_data(self, maturities: np.ndarray, yields: np.ndarray):
        """
        Fit yield curve from Treasury data using standard OLS.
        
        Args:
            maturities: Array of maturities in years (e.g., [1/12, 2/12, 3/12, 1, 2, 3, 5, 10, 30])
            yields: Array of yields (as decimals, e.g., 0.02 for 2%)
        """
        self.maturities = maturities
        self.yields = yields
        
        if self.method == 'nss':
            # Calibrate NSS model using standard OLS
            self.curve, status = calibrate_nss_ols(maturities, yields)
            logger.info(f"NSS curve calibrated: {self.curve}")
        else:
            # Fallback to linear interpolation
            logger.warning("NSS not available, using linear interpolation")
            self.method = 'linear'
    
    def get_rate(self, maturity: float) -> float:
        """
        Get risk-free rate for given maturity.
        
        Args:
            maturity: Time to maturity in years
            
        Returns:
            Risk-free rate (as decimal)
        """
        if self.curve is None:
            raise ValueError("Yield curve not fitted. Call fit_from_treasury_data() first.")
        
        if self.method == 'nss':
            return float(self.curve(maturity))
        else:
            # Linear interpolation
            return np.interp(maturity, self.maturities, self.yields)
    
    def get_rates(self, maturities: np.ndarray) -> np.ndarray:
        """
        Get risk-free rates for multiple maturities.
        
        Args:
            maturities: Array of maturities in years
            
        Returns:
            Array of risk-free rates
        """
        if self.curve is None:
            raise ValueError("Yield curve not fitted. Call fit_from_treasury_data() first.")
        
        if self.method == 'nss':
            return np.array([self.curve(t) for t in maturities])
        else:
            # Linear interpolation
            return np.interp(maturities, self.maturities, self.yields)


DEFAULT_MATURITIES = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 1, 2, 3, 5, 7, 10, 20, 30])
DEFAULT_YIELDS = np.array([0.15, 0.27, 0.50, 0.93, 1.52, 2.13, 2.32, 2.34, 2.37, 2.32, 2.65, 2.52]) / 100

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TREASURY_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "rates" / "2025-daily-treasury-rates.csv"

CSV_COLUMN_TO_MATURITY = {
    "1 Mo": 1 / 12,
    "1.5 Month": 1.5 / 12,
    "2 Mo": 2 / 12,
    "3 Mo": 3 / 12,
    "4 Mo": 4 / 12,
    "6 Mo": 6 / 12,
    "1 Yr": 1.0,
    "2 Yr": 2.0,
    "3 Yr": 3.0,
    "5 Yr": 5.0,
    "7 Yr": 7.0,
    "10 Yr": 10.0,
    "20 Yr": 20.0,
    "30 Yr": 30.0,
}

_TREASURY_CSV_CACHE = None


def get_treasury_yields(date: Optional[datetime] = None, use_cache: bool = True) -> Dict[str, np.ndarray]:
    """
    Fetch US Treasury yield data for the requested date from the local PAR-rate CSV.
    https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?type=daily_treasury_yield_curve&field_tdr_date_value=2025
    
    Args:
        date: Date to fetch yields for (default: most recent available)
        use_cache: If True, use cached data if available (default: True)
    
    Returns:
        Dictionary with 'maturities' and 'yields'
        
    Raises:
        ValueError: If CSV data cannot be loaded and defaults are disabled
    """
    if date is None:
        date = datetime.now()

    try:
        maturities, yields = _get_local_treasury_yields(date)
        return {"maturities": maturities, "yields": yields}
    except Exception as exc:
        logger.error(f"Failed to load Treasury yields from CSV: {exc}")
        logger.warning("Falling back to default yields")
        return {
            "maturities": DEFAULT_MATURITIES,
            "yields": DEFAULT_YIELDS,
        }


def _get_local_treasury_yields(date: datetime) -> (np.ndarray, np.ndarray):
    """
    Load Treasury yields from the local CSV file for the closest available date.
    """
    global _TREASURY_CSV_CACHE

    if _TREASURY_CSV_CACHE is None:
        if not TREASURY_CSV_PATH.exists():
            raise FileNotFoundError(f"Treasury CSV not found at {TREASURY_CSV_PATH}")

        logger.info(f"Loading Treasury yields from local CSV: {TREASURY_CSV_PATH}")
        df = pd.read_csv(TREASURY_CSV_PATH, parse_dates=["Date"])
        df = df.sort_values("Date")
        _TREASURY_CSV_CACHE = df
    else:
        df = _TREASURY_CSV_CACHE

    target_date = pd.Timestamp(date.date())
    available = df[df["Date"] <= target_date]
    if available.empty:
        available = df[df["Date"] == df["Date"].min()]
        if available.empty:
            raise ValueError("Local Treasury CSV contains no valid data rows")

    row = available.iloc[-1]

    maturities = []
    yields = []
    for column, maturity in CSV_COLUMN_TO_MATURITY.items():
        if column not in row or pd.isna(row[column]):
            continue
        rate = row[column]
        if rate > 1.0:
            rate = rate / 100.0
        maturities.append(maturity)
        yields.append(rate)

    if not maturities:
        raise ValueError(f"No usable yields found in CSV for date {row['Date'].date()}")

    maturities = np.array(maturities)
    yields = np.array(yields)
    sort_idx = np.argsort(maturities)
    return maturities[sort_idx], yields[sort_idx]

