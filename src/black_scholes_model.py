"""
Black-Scholes analytical model for quick benchmarking against Heston.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.stats import norm


def _d1(S: float, K: float, tau: float, r: float, q: float, sigma: float) -> float:
    fwd = S * math.exp((r - q) * tau)
    denom = sigma * math.sqrt(tau)
    return (math.log(fwd / K) + 0.5 * sigma**2 * tau) / denom


def _d2(d1_val: float, sigma: float, tau: float) -> float:
    return d1_val - sigma * math.sqrt(tau)


@dataclass
class BlackScholesModel:
    """
    Classic Black-Scholes model with constant volatility.
    """

    sigma: float
    r: float = 0.02
    q: float = 0.0

    def call_price(self, S: float, K: float, tau: float) -> float:
        if tau <= 0:
            return max(S - K, 0.0)
        d1_val = _d1(S, K, tau, self.r, self.q, self.sigma)
        d2_val = _d2(d1_val, self.sigma, tau)
        discount = math.exp(-self.r * tau)
        fwd_disc = math.exp(-self.q * tau)
        return fwd_disc * S * norm.cdf(d1_val) - discount * K * norm.cdf(d2_val)

    def put_price(self, S: float, K: float, tau: float) -> float:
        if tau <= 0:
            return max(K - S, 0.0)
        call = self.call_price(S, K, tau)
        parity = K * math.exp(-self.r * tau) - S * math.exp(-self.q * tau)
        return call - parity

    def option_price(self, S: float, K: float, tau: float, option_type: str) -> float:
        if option_type.lower() == "call":
            return self.call_price(S, K, tau)
        if option_type.lower() == "put":
            return self.put_price(S, K, tau)
        raise ValueError(f"Unknown option type: {option_type}")

    def greeks(self, S: float, K: float, tau: float, option_type: str = "call") -> Dict[str, float]:
        if tau <= 0:
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        d1_val = _d1(S, K, tau, self.r, self.q, self.sigma)
        d2_val = _d2(d1_val, self.sigma, tau)
        pdf = norm.pdf(d1_val)
        sign = 1 if option_type.lower() == "call" else -1
        delta = sign * math.exp(-self.q * tau) * norm.cdf(sign * d1_val)
        gamma = math.exp(-self.q * tau) * pdf / (S * self.sigma * math.sqrt(tau))
        vega = S * math.exp(-self.q * tau) * pdf * math.sqrt(tau) / 100.0
        theta = (
            -math.exp(-self.q * tau) * S * pdf * self.sigma / (2 * math.sqrt(tau))
            - sign * (self.r - self.q) * S * math.exp(-self.q * tau) * norm.cdf(sign * d1_val)
            + sign * self.r * K * math.exp(-self.r * tau) * norm.cdf(sign * d2_val)
        ) / 365.0
        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    @staticmethod
    def implied_vol(price: float, S: float, K: float, tau: float, r: float, q: float) -> float:
        if tau <= 0 or price <= 0:
            return math.nan

        def objective(vol: float) -> float:
            model = BlackScholesModel(vol, r, q)
            return model.call_price(S, K, tau) - price

        try:
            return brentq(objective, 1e-6, 5.0, xtol=1e-8)
        except ValueError:
            return math.nan

    # ---------------------------------------------------------------------
    # Calibration helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def calibrate_constant_sigma(
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        targets: np.ndarray,
        r: float,
        q: float,
        loss: str = "price",
        weights: np.ndarray | None = None,
        sigma_bounds: tuple[float, float] = (1e-6, 5.0),
    ) -> float:
        """
        Calibrate a single Black–Scholes volatility to a surface.
        targets:
          - if loss=='price': option prices (calls) with same shape as strikes/maturities
          - if loss=='vol':   market implied vols for the same grid
        """
        strikes = np.asarray(strikes).ravel()
        maturities = np.asarray(maturities).ravel()
        targets = np.asarray(targets).ravel()
        if weights is None:
            weights = np.ones_like(targets, dtype=float)
        else:
            weights = np.asarray(weights).ravel()

        def residuals(x: np.ndarray) -> np.ndarray:
            sigma = float(x[0])
            model = BlackScholesModel(sigma, r, q)
            if loss == "vol":
                model_prices = np.array([model.call_price(S, K, t) for K, t in zip(strikes, maturities)])
                model_vols = np.array(
                    [BlackScholesModel.implied_vol(p, S, K, t, r, q) for p, K, t in zip(model_prices, strikes, maturities)]
                )
                return weights * (model_vols - targets)
            # price loss
            model_prices = np.array([model.call_price(S, K, t) for K, t in zip(strikes, maturities)])
            return weights * (model_prices - targets)

        x0 = np.array([0.2])
        result = least_squares(
            residuals,
            x0,
            bounds=sigma_bounds,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=10000,
        )
        return float(np.clip(result.x[0], sigma_bounds[0], sigma_bounds[1]))

    @staticmethod
    def calibrate_term_structure_by_maturity(
        S: float,
        strikes: np.ndarray,
        maturities: np.ndarray,
        targets: np.ndarray,
        r: float,
        q: float,
        loss: str = "price",
        weights: np.ndarray | None = None,
        sigma_bounds: tuple[float, float] = (1e-6, 5.0),
    ) -> Dict[float, float]:
        """
        Calibrate a flat Black–Scholes volatility per maturity (piecewise term structure).
        Returns dict {maturity: sigma}.
        """
        strikes = np.asarray(strikes).ravel()
        maturities = np.asarray(maturities).ravel()
        targets = np.asarray(targets).ravel()
        if weights is None:
            weights = np.ones_like(targets, dtype=float)
        else:
            weights = np.asarray(weights).ravel()

        unique_mats = np.unique(maturities)
        term_structure: Dict[float, float] = {}
        for t in unique_mats:
            mask = maturities == t
            sigma_t = BlackScholesModel.calibrate_constant_sigma(
                S=S,
                strikes=strikes[mask],
                maturities=maturities[mask],
                targets=targets[mask],
                r=r,
                q=q,
                loss=loss,
                weights=weights[mask],
                sigma_bounds=sigma_bounds,
            )
            term_structure[float(t)] = sigma_t
        return term_structure

