"""
Heston Stochastic Volatility Model: Pricing, Simulation, and Greeks Computation.

Features:
- Option pricing (FFT, quad integration, rectangular)
- Monte Carlo path simulation (Euler scheme)
- Greeks computation (delta, gamma, vega, theta)
- Variance risk premium (lambda) for P→Q measure transformation
- Batch pricing capabilities
- Market option price calibration support
"""

import numpy as np
from scipy import integrate, optimize
from scipy.stats import norm
from typing import Dict, Tuple, Optional, List
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class HestonModel:
    """
    Heston stochastic volatility model for option pricing.
    
    Model (under risk-neutral measure Q):
    dS_t = r*S_t*dt + sqrt(V_t)*S_t*dW1_t^Q
    dV_t = kappa^Q*(theta^Q - V_t)*dt + sigma*sqrt(V_t)*dW2_t^Q
    dW1_t^Q * dW2_t^Q = rho * dt
    
    Where Q-measure parameters are:
    kappa^Q = kappa + lambda
    theta^Q = kappa*theta/(kappa+lambda)
    rho^Q = rho
    """
    
    def __init__(self, kappa: float, theta: float, sigma: float, rho: float, 
                 v0: float, r: float = 0.05, q: float = 0.0, lambd: float = 0.0):
        """
        Initialize Heston model.
        
        Args:
            kappa: Mean reversion speed (P-measure)
            theta: Long-term variance (P-measure)
            sigma: Volatility of volatility
            rho: Correlation between spot and variance
            v0: Initial variance
            r: Risk-free rate
            q: Dividend yield
            lambd: Variance risk premium (default: 0, assumes already in Q-measure)
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma  # Using 'sigma' instead of 'xi' to match notebook
        self.rho = rho
        self.v0 = v0
        self.r = r
        self.q = q
        self.lambd = lambd
        
        # Transform to Q-measure
        self.kappa_q = kappa + lambd
        self.theta_q = (kappa * theta) / (kappa + lambd) if (kappa + lambd) > 0 else theta
        
        # Validate parameters
        assert kappa > 0, "kappa must be positive"
        assert theta > 0, "theta must be positive"
        assert sigma > 0, "sigma must be positive"
        assert -1 <= rho <= 1, "rho must be in [-1, 1]"
        assert v0 > 0, "v0 must be positive"
    
    def characteristic_function(self, phi: np.ndarray, tau: float, 
                               S0: float, v0: Optional[float] = None) -> np.ndarray:
        """
        Standard Heston characteristic function (Q-measure).
        
        This is the main characteristic function using Q-measure parameters.
        It represents E[exp(i*phi*S_T)] under the risk-neutral measure.
        
        Args:
            phi: Fourier variable (can be array)
            tau: Time to maturity
            S0: Initial stock price
            v0: Initial variance (defaults to self.v0)
            
        Returns:
            Characteristic function values
            
        Note:
            Uses pre-computed Q-measure parameters (kappa_q, theta_q) from initialization.
            This avoids recomputing the P→Q measure transformation on every call.
        """
        if v0 is None:
            v0 = self.v0
        
        # Q-measure parameters (already transformed)
        kappa_q = self.kappa_q
        theta_q = self.theta_q
        sigma = self.sigma
        rho = self.rho
        r = self.r
        q = self.q
        
        # Constants for characteristic function
        a = kappa_q * theta_q
        b = kappa_q  # Already in Q-measure
        
        # Common terms
        rspi = rho * sigma * phi * 1j
        
        # d parameter (discriminant-like term)
        d = np.sqrt((rho * sigma * phi * 1j - b)**2 + (phi * 1j + phi**2) * sigma**2)
        
        # g parameter (ratio of exponential terms)
        g = (b - rspi + d) / (b - rspi - d)
        
        # Calculate characteristic function: φ(phi) = E[exp(i*phi*S_T)]
        # Three components: drift, diffusion scaling, and variance path contribution
        exp1 = np.exp(r * phi * 1j * tau)  # Risk-free drift
        term2 = S0**(phi * 1j) * ((1 - g * np.exp(d * tau)) / (1 - g))**(-2 * a / sigma**2)
        exp2 = np.exp(
            a * tau * (b - rspi + d) / sigma**2 + 
            v0 * (b - rspi + d) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau))) / sigma**2
        )
        
        return exp1 * term2 * exp2
    
    def integrand(self, phi: float, S0: float, K: float, tau: float) -> complex:
        """
        Integrand for option pricing formula (quad integration method).
        
        Uses the Carr-Madan approach with characteristic function.
        
        Args:
            phi: Fourier variable
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            
        Returns:
            Complex integrand value
        """
        # Use the main characteristic function (already in Q-measure)
        numerator = (np.exp(self.r * tau) * self.characteristic_function(phi - 1j, tau, S0, self.v0) 
                     - K * self.characteristic_function(phi, tau, S0, self.v0))
        denominator = 1j * phi * K**(1j * phi)
        return numerator / denominator
    
    def call_price(self, S0: float, K: float, tau: float, method: str = 'fft') -> float:
        """
        Price European call option using numerical integration or FFT.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            method: 'fft' (Carr-Madan FFT, default), 'quad' (scipy), or 'rectangular' (batch)
            
        Returns:
            Call option price
        """
        if method == 'fft':
            return self.call_price_fft(S0, K, tau)
        elif method == 'rectangular':
            return self.call_price_rectangular(S0, K, tau)
        else:
            return self.call_price_quad(S0, K, tau)
    
    def call_price_quad(self, S0: float, K: float, tau: float) -> float:
        """
        Price European call option using scipy quad integration.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            
        Returns:
            Call option price
        """
        def integrand(phi):
            if phi == 0:
                return 0
            return np.real(self.integrand(phi, S0, K, tau))
        
        # Integrate
        real_integral, _ = integrate.quad(integrand, 0, 100, limit=1000)
        
        price = (S0 - K * np.exp(-self.r * tau)) / 2 + real_integral / np.pi
        return max(price, 0)
    
    def _heston_cf_fft(self, u: np.ndarray, tau: float, S0: float) -> np.ndarray:
        """
        Specialized characteristic function for FFT pricing (Little Heston Trap).
        
        This version uses the "Little Heston Trap" parametrization which is more
        numerically stable for FFT-based option pricing. It differs from the standard
        characteristic_function() in that it:
        1. Works with log-prices (x = ln S) instead of prices
        2. Uses different g-parameter formulation: g = (b-d)/(b+d)
        3. Includes numerical guards against division by zero
        4. Outputs in exponential form: exp(C + D*v0 + i*u*x0)
        
        Reference: Albrecher et al. (2007) - "The Little Heston Trap"
        
        Args:
            u: Fourier variable (can be array)
            tau: Time to maturity
            S0: Current stock price
            
        Returns:
            Characteristic function values: φ(u) = E[exp(i*u*ln(S_T))]
            
        Note:
            This should only be used internally by call_price_fft(). For general
            characteristic function evaluation, use characteristic_function() instead.
        """
        i = 1j
        x0 = np.log(S0)
        
        # Q-measure parameters
        kappa_q = self.kappa_q
        theta_q = self.theta_q
        sigma = self.sigma
        rho = self.rho
        r = self.r
        q = self.q
        
        # Little Heston Trap parametrization (matches notebook)
        a = kappa_q * theta_q
        b = kappa_q - rho * sigma * i * u
        d = np.sqrt(b * b + (sigma**2) * (i * u + u * u))
        g = (b - d) / (b + d)
        
        # Guard against division by zero
        eDT = np.exp(-d * tau)
        one_minus_g_eDT = 1 - g * eDT
        one_minus_g = 1 - g
        
        # Small guards to prevent division by zero
        one_minus_g_eDT = np.where(np.abs(one_minus_g_eDT) < 1e-15, 1e-15, one_minus_g_eDT)
        one_minus_g = np.where(np.abs(one_minus_g) < 1e-15, 1e-15, one_minus_g)
        
        # Characteristic function components
        C = i * u * (r - q) * tau + (a / (sigma**2)) * ((b - d) * tau - 2.0 * np.log(one_minus_g_eDT / one_minus_g))
        D = ((b - d) / (sigma**2)) * ((1 - eDT) / one_minus_g_eDT)
        
        return np.exp(C + D * self.v0 + i * u * x0)
    
    def _simpson_weights(self, N: int) -> np.ndarray:
        """Simpson weights on an N-point uniform grid (N must be even)."""
        if N % 2 != 0:
            N += 1  # Make even
        w = np.ones(N)
        w[1:N-1:2] = 4.0
        w[2:N-2:2] = 2.0
        return w / 3.0
    
    def call_price_fft(self, S0: float, K: float, tau: float, 
                      N: int = 4096, alpha: float = 1.5, eta: float = 0.25) -> float:
        """
        Price European call option using Carr-Madan FFT method.
        Matches notebook implementation exactly.
        
        More robust and efficient than direct integration, especially for batch pricing.
        Uses FFT to compute prices for a grid of strikes, then interpolates.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            N: Number of FFT points (should be power of 2, default 4096, must be even)
            alpha: Damping parameter for Carr-Madan (default 1.5)
            eta: Frequency step Δv (default 0.25)
            
        Returns:
            Call option price
        """
        # Ensure N is even and power of 2
        N = int(2 ** np.ceil(np.log2(N)))
        if N % 2 != 0:
            N += 1
        
        # Frequency grid: v_j = j * eta
        n = np.arange(N)
        v = eta * n
        
        i = 1j
        # Carr-Madan: ψ(v) = e^{-rT} φ(v - i(α+1)) / [(α + iv)(α + iv + 1)]
        phi_shift = self._heston_cf_fft(v - (alpha + 1) * i, tau, S0)
        denom = (alpha**2 + alpha - v**2 + i * (2 * alpha + 1) * v)  # (α+iv)(α+iv+1)
        
        # Handle division by zero
        denom = np.where(np.abs(denom) < 1e-15, 1e-15, denom)
        psi = np.exp(-self.r * tau) * phi_shift / denom
        
        # Simpson weights for the v-integral
        w = self._simpson_weights(N) * eta
        
        # FFT coupling: Δk = 2π / (N * eta)
        lam = 2.0 * np.pi / (N * eta)  # Δk (log-strike step)
        b = 0.5 * N * lam  # half-width in k
        
        # Prepare FFT input: multiply by weights and phase factor
        x = psi * np.exp(1j * b * v) * w
        
        # Perform FFT
        F = np.fft.fft(x)
        F = np.real(F)
        
        # Log-strike grid: k = -b + j * lam
        j = np.arange(N)
        k = -b + j * lam  # k = ln K
        K_grid = np.exp(k)
        
        # Undamp: C(k) = exp(-alpha * k) / π * F
        calls = np.exp(-alpha * k) / np.pi * F
        calls = np.maximum(calls, 0.0)
        
        # Sort by strike for interpolation
        order = np.argsort(K_grid)
        K_grid = K_grid[order]
        calls = calls[order]
        
        # Interpolate to get price for desired strike
        if K <= K_grid[0]:
            return calls[0]
        if K >= K_grid[-1]:
            return calls[-1]
        
        idx = np.searchsorted(K_grid, K)
        x0, x1 = K_grid[idx-1], K_grid[idx]
        y0, y1 = calls[idx-1], calls[idx]
        
        return y0 + (y1 - y0) * (K - x0) / (x1 - x0)
    
    def call_price_rectangular(self, S0: float, K: float, tau: float,
                               umax: float = 100, N: int = 10000) -> float:
        """
        Price European call option using rectangular integration.
        
        Simple rectangular rule for numerical integration of the characteristic function.
        Less accurate than FFT but useful for educational purposes.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            umax: Upper integration limit (default: 100)
            N: Number of integration points (default: 10000)
            
        Returns:
            Call option price
        """
        P = 0
        dphi = umax / N  # Width of rectangle
        
        for i in range(1, N):
            # Midpoint for rectangular integration
            phi = dphi * (2 * i + 1) / 2
            
            # Use main characteristic function
            numerator = (np.exp(self.r * tau) * self.characteristic_function(phi - 1j, tau, S0, self.v0) 
                        - K * self.characteristic_function(phi, tau, S0, self.v0))
            denominator = 1j * phi * K**(1j * phi)
            P += dphi * numerator / denominator
        
        return np.real((S0 - K * np.exp(-self.r * tau)) / 2 + P / np.pi)
    
    def call_price_batch(self, S0: float, K: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """
        Price multiple call options efficiently using rectangular integration.
        
        Args:
            S0: Current stock price
            K: Array of strike prices
            tau: Array of times to maturity
            
        Returns:
            Array of call option prices
        """
        prices = []
        for k, t in zip(K, tau):
            prices.append(self.call_price_rectangular(S0, k, t))
        return np.array(prices)

    # ---------------------------------------------------------------------
    # Variance swap term-structure (expected annualized variance)
    # Reference:
    #   (1/T) E[V_T] = θ^Q + ((1 - exp(-κ^Q T)) / (κ^Q T)) · (v0 - θ^Q)
    # This section provides helpers and a least-squares calibration routine.
    # ---------------------------------------------------------------------
    def expected_annualized_variance(self, T: float) -> float:
        """
        Expected annualized variance over horizon T under Q:
            (1/T) E[V_T] = theta_Q + ((1 - exp(-kappa_Q T)) / (kappa_Q T)) * (v0 - theta_Q)

        Where V_T is cumulative variance integral_0^T v_s ds.
        """
        if T <= 0:
            return float(self.v0)
        kq = float(self.kappa_q)
        thq = float(self.theta_q)
        term = (1.0 - np.exp(-kq * T)) / (kq * T)
        return float(thq + term * (self.v0 - thq))

    def expected_annualized_variance_curve(self, maturities: np.ndarray) -> np.ndarray:
        """
        Vectorised version of expected_annualized_variance for an array of maturities.
        """
        maturities = np.asarray(maturities, dtype=float)
        if maturities.ndim == 0:
            return np.array([self.expected_annualized_variance(float(maturities))])
        return np.array([self.expected_annualized_variance(float(T)) for T in maturities])
    
    def put_price(self, S0: float, K: float, tau: float) -> float:
        """
        Price European put option using put-call parity.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            
        Returns:
            Put option price
        """
        call_price = self.call_price(S0, K, tau)
        put_price = call_price - S0 * np.exp(-self.q * tau) + K * np.exp(-self.r * tau)
        return max(put_price, 0)
    
    def option_price(self, S0: float, K: float, tau: float, option_type: str) -> float:
        """
        Price option (call or put).
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        if option_type.lower() == 'call':
            return self.call_price(S0, K, tau)
        elif option_type.lower() == 'put':
            return self.put_price(S0, K, tau)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
    
    def delta(self, S0: float, K: float, tau: float, option_type: str, 
              dS: float = 0.01) -> float:
        """
        Compute delta using finite differences.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            option_type: 'call' or 'put'
            dS: Price bump for finite differences
            
        Returns:
            Delta
        """
        price_up = self.option_price(S0 + dS, K, tau, option_type)
        price_down = self.option_price(S0 - dS, K, tau, option_type)
        return (price_up - price_down) / (2 * dS)
    
    def gamma(self, S0: float, K: float, tau: float, option_type: str,
              dS: float = 0.01) -> float:
        """
        Compute gamma using finite differences.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            option_type: 'call' or 'put'
            dS: Price bump for finite differences
            
        Returns:
            Gamma
        """
        price = self.option_price(S0, K, tau, option_type)
        price_up = self.option_price(S0 + dS, K, tau, option_type)
        price_down = self.option_price(S0 - dS, K, tau, option_type)
        return (price_up - 2 * price + price_down) / (dS**2)
    
    def vega(self, S0: float, K: float, tau: float, option_type: str,
             dv: float = 0.01) -> float:
        """
        Compute vega (sensitivity to volatility) using finite differences.
        
        Note: In Heston model, vega is sensitivity to initial variance v0.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            option_type: 'call' or 'put'
            dv: Variance bump for finite differences
            
        Returns:
            Vega (per unit variance change)
        """
        # Create temporary model with bumped variance
        model_up = HestonModel(self.kappa, self.theta, self.sigma, self.rho,
                              self.v0 + dv, self.r, self.q, self.lambd)
        model_down = HestonModel(self.kappa, self.theta, self.sigma, self.rho,
                                max(1e-6, self.v0 - dv), self.r, self.q, self.lambd)
        
        price_up = model_up.option_price(S0, K, tau, option_type)
        price_down = model_down.option_price(S0, K, tau, option_type)
        
        return (price_up - price_down) / (2 * dv)
    
    def theta_greek(self, S0: float, K: float, tau: float, option_type: str,
                    dT: float = 1/365) -> float:
        """
        Compute theta (time decay) using finite differences.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            option_type: 'call' or 'put'
            dT: Time bump (in years)
            
        Returns:
            Theta (per day, typically negative)
        """
        if tau <= dT:
            return 0.0
        
        price = self.option_price(S0, K, tau, option_type)
        price_decayed = self.option_price(S0, K, tau - dT, option_type)
        return (price_decayed - price) / dT
    
    def compute_greeks(self, S0: float, K: float, tau: float, option_type: str) -> Dict[str, float]:
        """
        Compute all Greeks for an option.
        
        Args:
            S0: Current stock price
            K: Strike price
            tau: Time to maturity
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with delta, gamma, vega, theta
        """
        return {
            'delta': self.delta(S0, K, tau, option_type),
            'gamma': self.gamma(S0, K, tau, option_type),
            'vega': self.vega(S0, K, tau, option_type),
            'theta': self.theta_greek(S0, K, tau, option_type)
        }
    
    # =========================================================================
    # MONTE CARLO SIMULATION
    # =========================================================================
    
    def simulate_paths(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None,
        full_truncation: bool = True,
        scheme: str = 'euler'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate stock price and variance paths using Monte Carlo.
        
        Uses Euler discretization with full truncation (variance stays non-negative).
        The correlated Brownian motions are generated using Cholesky decomposition.
        
        Args:
            S0: Initial stock price
            T: Time horizon (in years)
            n_steps: Number of time steps
            n_paths: Number of simulation paths
            seed: Random seed for reproducibility (optional)
            full_truncation: If True, enforce variance >= 0 at each step
            scheme: Discretization scheme ('euler' is default)
            
        Returns:
            Tuple of (times, spot_paths, var_paths)
            - times: Array of shape (n_steps+1,) with time points
            - spot_paths: Array of shape (n_paths, n_steps+1) with stock prices
            - var_paths: Array of shape (n_paths, n_steps+1) with variances
            
        Example:
            >>> model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3, 
            ...                     rho=-0.7, v0=0.04, r=0.03)
            >>> times, S, V = model.simulate_paths(S0=100, T=1.0, 
            ...                                     n_steps=252, n_paths=1000)
            >>> # Plot average path
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(times, S.mean(axis=0))
            >>> plt.xlabel('Time (years)')
            >>> plt.ylabel('Stock Price')
            >>> plt.show()
        """
        if scheme != 'euler':
            raise ValueError(f"Only 'euler' scheme is currently supported, got '{scheme}'")
        
        # Time grid
        dt = T / n_steps
        times = np.linspace(0, T, n_steps + 1)
        
        # Initialize random number generator
        rng = np.random.default_rng(seed)
        
        # Initialize paths
        spot_paths = np.zeros((n_paths, n_steps + 1))
        var_paths = np.zeros((n_paths, n_steps + 1))
        spot_paths[:, 0] = S0
        var_paths[:, 0] = self.v0
        
        # Correlation matrix and Cholesky decomposition
        # Ensure rho is within valid range for positive definite matrix
        # Clamp to (-1 + eps, 1 - eps) for numerical stability
        rho_safe = np.clip(self.rho, -0.9999, 0.9999)
        if abs(self.rho - rho_safe) > 1e-6:
            import warnings
            warnings.warn(
                f"Correlation rho={self.rho:.6f} clamped to {rho_safe:.6f} "
                f"for numerical stability in Cholesky decomposition",
                RuntimeWarning
            )
        
        corr_matrix = np.array([
            [1.0, rho_safe],
            [rho_safe, 1.0]
        ])
        chol = np.linalg.cholesky(corr_matrix)
        
        # Simulate paths
        for t in range(n_steps):
            # Generate correlated random increments
            z = rng.standard_normal((n_paths, 2))
            dW = z @ chol.T * np.sqrt(dt)  # Correlated Brownian increments
            
            # Current variance (with truncation if enabled)
            v_t = np.maximum(var_paths[:, t], 0.0) if full_truncation else var_paths[:, t]
            
            # Variance process: dV = kappa*(theta - V)*dt + sigma*sqrt(V)*dW2
            var_drift = self.kappa * (self.theta - v_t) * dt
            var_diffusion = self.sigma * np.sqrt(np.maximum(v_t, 0.0)) * dW[:, 1]
            var_paths[:, t + 1] = var_paths[:, t] + var_drift + var_diffusion
            
            # Apply full truncation to variance
            if full_truncation:
                var_paths[:, t + 1] = np.maximum(var_paths[:, t + 1], 0.0)
            
            # Stock process: dS/S = (r - q)*dt + sqrt(V)*dW1
            # Using log-normal formulation: log(S_{t+1}) = log(S_t) + drift + diffusion
            vol = np.sqrt(np.maximum(v_t, 0.0))
            log_drift = (self.r - self.q - 0.5 * v_t) * dt
            log_diffusion = vol * dW[:, 0]
            spot_paths[:, t + 1] = spot_paths[:, t] * np.exp(log_drift + log_diffusion)
        
        return times, spot_paths, var_paths
    
    def simulate_terminal_prices(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Simulate only terminal stock prices (more memory efficient).
        
        Useful for Monte Carlo option pricing where only final values matter.
        
        Args:
            S0: Initial stock price
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            seed: Random seed
            
        Returns:
            Array of terminal stock prices, shape (n_paths,)
            
        Example:
            >>> model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3,
            ...                     rho=-0.7, v0=0.04, r=0.03)
            >>> S_T = model.simulate_terminal_prices(S0=100, T=1.0,
            ...                                      n_steps=252, n_paths=10000)
            >>> # Price European call using MC
            >>> K = 100
            >>> payoffs = np.maximum(S_T - K, 0)
            >>> call_price = np.exp(-model.r * T) * payoffs.mean()
        """
        _, spot_paths, _ = self.simulate_paths(S0, T, n_steps, n_paths, seed)
        return spot_paths[:, -1]
    
    def price_option_monte_carlo(
        self,
        S0: float,
        K: float,
        T: float,
        option_type: str,
        n_steps: int = 252,
        n_paths: int = 10000,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Price European option using Monte Carlo simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            option_type: 'call' or 'put'
            n_steps: Number of time steps (default: 252 for daily)
            n_paths: Number of Monte Carlo paths
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with 'price', 'std_error', 'confidence_interval'
            
        Example:
            >>> model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3,
            ...                     rho=-0.7, v0=0.04, r=0.03)
            >>> result = model.price_option_monte_carlo(S0=100, K=100, T=1.0,
            ...                                          option_type='call',
            ...                                          n_paths=100000)
            >>> print(f"Call price: {result['price']:.4f} ± {result['std_error']:.4f}")
        """
        # Simulate terminal prices
        S_T = self.simulate_terminal_prices(S0, T, n_steps, n_paths, seed)
        
        # Compute payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(S_T - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - S_T, 0)
        else:
            raise ValueError(f"Unknown option type: {option_type}")
        
        # Discount and compute statistics
        discount = np.exp(-self.r * T)
        discounted_payoffs = discount * payoffs
        
        price = discounted_payoffs.mean()
        std_error = discounted_payoffs.std() / np.sqrt(n_paths)
        
        # 95% confidence interval
        confidence_interval = (
            price - 1.96 * std_error,
            price + 1.96 * std_error
        )
        
        return {
            'price': float(price),
            'std_error': float(std_error),
            'confidence_interval': confidence_interval
        }
    
    def get_simulation_statistics(
        self,
        S0: float,
        T: float,
        n_steps: int,
        n_paths: int,
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute statistics from simulated paths.
        
        Returns various statistics useful for model validation and analysis.
        
        Args:
            S0: Initial stock price
            T: Time horizon
            n_steps: Number of time steps
            n_paths: Number of paths
            seed: Random seed
            
        Returns:
            Dictionary with statistics:
            - mean_terminal_price: E[S_T]
            - std_terminal_price: Std[S_T]
            - mean_terminal_variance: E[V_T]
            - mean_realized_vol: Average realized volatility
            - prob_positive_return: P(S_T > S0)
            
        Example:
            >>> model = HestonModel(kappa=2.0, theta=0.04, sigma=0.3,
            ...                     rho=-0.7, v0=0.04, r=0.03)
            >>> stats = model.get_simulation_statistics(S0=100, T=1.0,
            ...                                          n_steps=252, n_paths=10000)
            >>> print(f"Expected terminal price: {stats['mean_terminal_price']:.2f}")
            >>> print(f"Mean realized vol: {stats['mean_realized_vol']:.2%}")
        """
        times, spot_paths, var_paths = self.simulate_paths(S0, T, n_steps, n_paths, seed)
        
        # Terminal values
        S_T = spot_paths[:, -1]
        V_T = var_paths[:, -1]
        
        # Log returns
        log_returns = np.log(spot_paths[:, 1:] / spot_paths[:, :-1])
        
        # Realized volatility (annualized)
        realized_vol = np.std(log_returns, axis=1) * np.sqrt(252)
        
        return {
            'mean_terminal_price': float(S_T.mean()),
            'std_terminal_price': float(S_T.std()),
            'median_terminal_price': float(np.median(S_T)),
            'mean_terminal_variance': float(V_T.mean()),
            'std_terminal_variance': float(V_T.std()),
            'mean_realized_vol': float(realized_vol.mean()),
            'std_realized_vol': float(realized_vol.std()),
            'prob_positive_return': float((S_T > S0).mean()),
            'mean_log_return': float(np.log(S_T / S0).mean()),
            'std_log_return': float(np.log(S_T / S0).std())
        }