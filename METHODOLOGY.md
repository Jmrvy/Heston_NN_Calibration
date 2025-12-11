# Methodology: Deep Learning Framework for Heston Model Calibration

## Table of Contents

1. [Introduction](#introduction)
2. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
3. [Heston Stochastic Volatility Model](#heston-stochastic-volatility-model)
4. [Classical Calibration Framework](#classical-calibration-framework)
5. [Neural Network Surrogate Development](#neural-network-surrogate-development)
6. [Neural Network-Based Calibration](#neural-network-based-calibration)
7. [Hybrid Calibration Approach](#hybrid-calibration-approach)
8. [Evaluation and Validation](#evaluation-and-validation)

---

## 1. Introduction

This document presents the complete methodology for calibrating the Heston stochastic volatility model using deep learning techniques. The framework combines traditional numerical methods with neural network surrogates to achieve both accuracy and computational efficiency. The methodology spans from market data collection to final parameter estimation, providing a comprehensive pipeline for option pricing model calibration.

The framework is designed to:
- Calibrate Heston parameters to real market option prices
- Leverage neural networks for fast surrogate pricing
- Maintain accuracy comparable to traditional FFT-based methods
- Support multiple securities with varying volatility characteristics
- Provide a hybrid approach combining speed and precision

---

## 2. Data Collection and Preprocessing

### 2.1 Market Data Sources

The framework requires three types of market data:

1. **Risk-Free Rate Data**: Treasury yield curve for discounting
2. **Historical Spot Prices**: For computing underlying statistics
3. **Current Option Chains**: For calibration targets

All data is fetched using `yfinance`, providing free access to real-time and historical market data.

### 2.2 Yield Curve Construction

**Objective**: Obtain risk-free rates for all maturities required for option pricing.

**Method**: Nelson-Siegel-Svensson (NSS) model for yield curve interpolation.

The NSS model fits the yield curve using the functional form:

$y(\tau) = \beta_0 + \beta_1 \left( \frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1} \right) + \beta_2 \left( \frac{1 - e^{-\tau/\tau_1}}{\tau/\tau_1} - e^{-\tau/\tau_1} \right) + \beta_3 \left( \frac{1 - e^{-\tau/\tau_2}}{\tau/\tau_2} - e^{-\tau/\tau_2} \right)$

where:
- $\beta_0, \beta_1, \beta_2, \beta_3$: Level, slope, and curvature parameters
- $\tau_1, \tau_2$: Decay parameters controlling the shape
- $\tau$: Time to maturity

**Implementation Steps**:

1. Fetch Treasury yield data for standard maturities (1, 3, 6 months; 1, 2, 3, 5, 7, 10, 20, 30 years)
2. Filter and clean the data (remove missing values, outliers)
3. Fit NSS parameters using non-linear least squares optimization
4. Interpolate/extrapolate rates for arbitrary maturities using the fitted curve

**Output**: A `YieldCurve` object that provides `get_rate(tau)` for any maturity $\tau$.

**Rationale**: The NSS model provides smooth, economically reasonable interpolations across the entire term structure, ensuring consistent risk-free rates for options with varying maturities.

### 2.3 Historical Spot Price Collection

**Objective**: Collect historical price data to estimate underlying security statistics and inform Heston parameter ranges.

**Method**: Daily close prices using `yfinance`.

**Parameters**:
- **Period**: Configurable (default: 2 years or 3 months)
- **Frequency**: Daily observations
- **Data**: Open, High, Low, Close, Volume

**Processing Steps**:

1. Fetch historical data for the specified period
2. Compute daily returns: $r_t = \ln(S_t / S_{t-1})$
3. Calculate rolling volatility (21-day window)
4. Compute realized statistics:
   - Mean and standard deviation of volatility
   - Volatility of volatility $(\sigma_{\text{vol}})$
   - Skewness and kurtosis of returns
   - Leverage correlation (correlation between returns and volatility changes)

**Output Statistics**:

- `current_vol`: Most recent 21-day realized volatility
- `mean_vol`: Average realized volatility over the period
- `vol_of_vol`: Standard deviation of rolling volatility (informs $\sigma$ parameter)
- `leverage_corr`: Correlation between returns and volatility (informs $\rho$ parameter)
- `skewness`, `kurtosis`: Higher moments for parameter range estimation

These statistics directly inform the parameter ranges used for synthetic data generation and calibration bounds.

### 2.4 Option Chain Collection and Filtering

**Objective**: Obtain a high-quality set of market option prices for calibration.

**Data Source**: `yfinance` option chain API

**Filtering Criteria**:

1. **Liquidity Filters**:
   - Minimum trading volume: 50 contracts
   - Minimum open interest: 100 contracts

2. **Moneyness Range**: 
   - Relative moneyness $K/S \in [0.90, 1.10]$ (near ATM focus)
   - Rationale: Near-ATM options are most liquid and informative for calibration

3. **Maturity Range**:
   - Time to maturity: $\tau \in [0.05, 1.5]$ years (3 weeks to 1.5 years)
   - Rationale: Balance between liquidity (shorter) and model fit quality (longer)

4. **Option Type**:
   - Calls only (standard practice for calibration)

**Processing Steps**:

1. Fetch current option chain for the underlying
2. Extract bid, ask, mid prices: $P_{\text{mid}} = (P_{\text{bid}} + P_{\text{ask}})/2$
3. Compute implied volatility (using Black-Scholes as reference)
4. Filter options based on criteria above
5. Match each option with its corresponding risk-free rate from the yield curve
6. Compute moneyness: $m = K/S$
7. Store: strike ($K$), maturity ($\tau$), mid price ($P_{\text{market}}$), implied volatility ($\sigma_{\text{IV}}$), moneyness ($m$)

**Output Format**: CSV file with columns:
- `strike`, `maturity`, `mid_price`, `bid`, `ask`, `volume`, `openInterest`
- `impliedVolatility`, `moneyness`, `spot`, `r` (risk-free rate)

**Rationale**: Filtering ensures we calibrate to liquid, high-quality option prices that accurately reflect market expectations, reducing noise from illiquid or stale prices.

---

## 3. Heston Stochastic Volatility Model

### 3.1 Model Derivation

The Heston model extends the Black-Scholes framework by allowing stochastic volatility. Under the risk-neutral measure $\mathbb{Q}$, the model is defined by the system of stochastic differential equations:

$\begin{align}
dS_t &= r S_t dt + \sqrt{V_t} S_t dW_1^Q(t) \\
dV_t &= \kappa^Q (\theta^Q - V_t) dt + \sigma \sqrt{V_t} dW_2^Q(t) \\
dW_1^Q(t) \cdot dW_2^Q(t) &= \rho dt
\end{align}$

where:
- $S_t$: Asset price at time $t$
- $V_t$: Variance at time $t$ (instantaneous volatility is $\sqrt{V_t}$)
- $r$: Risk-free rate
- $\kappa^Q$: Mean reversion speed (risk-neutral)
- $\theta^Q$: Long-term variance (risk-neutral)
- $\sigma$: Volatility of volatility (vol-of-vol)
- $\rho$: Correlation between price and variance innovations
- $W_1^Q, W_2^Q$: Correlated Brownian motions

### 3.2 Physical vs. Risk-Neutral Measure

The model parameters can be specified in either the physical measure ($\mathbb{P}$) or risk-neutral measure ($\mathbb{Q}$). The relationship is:

$\begin{align}
\kappa^Q &= \kappa + \lambda \\
\theta^Q &= \frac{\kappa \theta}{\kappa + \lambda}
\end{align}$

where $\lambda$ is the variance risk premium. For calibration to option prices (which are risk-neutral), we work directly in the $\mathbb{Q}$ measure, setting $\lambda = 0$ and using $\kappa^Q = \kappa$ and $\theta^Q = \theta$.

### 3.3 Parameter Constraints

The model requires several constraints for mathematical and economic validity:

1. **Positivity Constraints**:
   - $v_0 > 0$: Initial variance must be positive
   - $\theta > 0$: Long-term variance must be positive
   - $\kappa > 0$: Mean reversion speed must be positive
   - $\sigma > 0$: Vol-of-vol must be positive

2. **Feller Condition**:
   $2\kappa\theta \geq \sigma^2$
   This ensures that the variance process $V_t$ remains strictly positive with probability 1. Violation leads to the variance reaching zero, which is economically unrealistic.

3. **Correlation Constraint**:
   $-1 \leq \rho \leq 1$
   Typically, $\rho < 0$ (negative correlation) captures the leverage effect observed in equity markets.

### 3.4 Characteristic Function and Option Pricing

European call option prices are computed using the characteristic function approach. The Heston characteristic function has an analytical form:

$\phi(u, t, S_0, v_0) = E^Q[e^{iu \ln S_t} | S_0, v_0] = e^{C(u,t) + D(u,t)v_0 + iu \ln S_0}$

where $C(u,t)$ and $D(u,t)$ are complex-valued functions of the model parameters (see Heston, 1993, for explicit formulas).

**Option Pricing Methods Implemented**:

1. **FFT Method (Carr-Madan)**:
   - Uses Fast Fourier Transform to compute prices for a grid of strikes simultaneously
   - Most efficient for batch pricing
   - Default method in our framework
   - Complexity: $O(N \log N)$ for $N$ strikes

2. **Quadrature Integration**:
   - Direct numerical integration of the characteristic function
   - More accurate but slower
   - Used for validation

3. **Rectangular Method**:
   - Simplified integration scheme
   - Fastest but less accurate
   - Used for quick approximations

### 3.5 FFT Implementation Details

The Carr-Madan FFT method:

1. **Damped Option Price Transform**:
   $\psi(v) = \frac{e^{-rT} \phi(v - i(\alpha+1))}{(\alpha + iv)(\alpha + iv + 1)}$
   where $\alpha$ is a damping parameter (typically 1.5).

2. **FFT Computation**:
   - Discretize the frequency domain: $v_j = j \cdot \Delta v$
   - Compute $\psi(v_j)$ for $j = 0, 1, ..., N-1$
   - Apply Simpson's rule weights
   - Perform FFT to obtain prices on a log-strike grid

3. **Interpolation**:
   - Extract price for desired strike via linear interpolation

**Advantages**:
- Computes prices for many strikes simultaneously
- Highly optimized (vectorized operations)
- Numerically stable

**Parameters**:
- $N = 4096$: FFT grid size (power of 2 for efficiency)
- $\alpha = 1.5$: Damping parameter
- $\Delta v = 0.25$: Frequency step size

---

## 4. Classical Calibration Framework

### 4.1 Calibration Objective

Calibrate Heston parameters $\eta = \{v_0, \kappa, \theta, \sigma, \rho\}$ to market option prices by minimizing:

$J(\eta) = \sum_{i=1}^{N} w_i \left( C^{\text{Heston}}(\eta, K_i, T_i) - C_i^{\text{Market}} \right)^2$

where:
- $C^{\text{Heston}}(\eta, K_i, T_i)$: Model price for strike $K_i$ and maturity $T_i$
- $C_i^{\text{Market}}$: Market mid price
- $w_i$: Optional weights (typically uniform, $w_i = 1$)

In practice, we minimize the Mean Squared Error (MSE):

$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left( C^{\text{Heston}}(\eta, K_i, T_i) - C_i^{\text{Market}} \right)^2$

### 4.2 Parameter Initialization

Good initial parameter guesses are critical for convergence. Our framework uses historical statistics to inform initial values:

**From Historical Data**:
- $v_0 \leftarrow$ current realized volatility\(^2\)
- $\theta \leftarrow$ mean realized volatility\(^2\)
- $\sigma \leftarrow$ volatility of volatility (scaled appropriately)
- $\rho \leftarrow$ leverage correlation
- $\kappa \leftarrow$ estimate from volatility mean reversion (default: 2.0)

**Parameter Ranges** (used as optimization bounds):

Derived from historical statistics with security-specific ranges:

$\begin{align}
v_0 &\in [(0.5 \times \hat{\sigma}_{\text{current}})^2, (2 \times \hat{\sigma}_{\text{current}})^2] \\
\theta &\in [(0.5 \times \bar{\sigma})^2, (2 \times \bar{\sigma})^2] \\
\kappa &\in [0.5, 10.0] \quad \text{(security-specific, typically narrower based on historical data)} \\
\sigma &\in [0.1, 1.5] \quad \text{(security-specific, informed by volatility of volatility)} \\
\rho &\in [\hat{\rho} - 0.3, \min(\hat{\rho} + 0.3, 0)]
\end{align}$

Note: These ranges are security-specific and derived from each asset's historical volatility statistics, ensuring relevance to the particular security being calibrated.

### 4.3 Optimization Algorithm

**Method**: Sequential Least Squares Programming (SLSQP)

**Rationale**:
- Handles bounds and constraints efficiently
- Good convergence for smooth objectives
- Widely used in finance applications

**Constraints Enforced**:
1. Parameter bounds (from historical analysis)
2. Feller condition: $2\kappa\theta \geq \sigma^2$

**Implementation**:

```python
def objective(params):
    v0, kappa, theta, sigma, rho = params
    
    # Check Feller condition
    if 2 * kappa * theta < sigma**2:
        return 1e10  # Penalty for invalid parameters
    
    # Create Heston model
    model = HestonModel(kappa=kappa, theta=theta, sigma=sigma,
                       rho=rho, v0=v0, r=r)
    
    # Compute model prices using FFT
    model_prices = [model.call_price_fft(S0, K, tau) 
                   for K, tau in zip(strikes, maturities)]
    
    # Compute MSE
    mse = np.mean((model_prices - market_prices)**2)
    return mse

# Optimize
result = optimize.minimize(
    objective, 
    x0=initial_params,
    method='SLSQP',
    bounds=parameter_bounds,
    options={'maxiter': 500, 'ftol': 1e-6}
)
```

**Optimization Settings**:
- Maximum iterations: 500
- Tolerance: $10^{-6}$
- Multiple random restarts (optional) to avoid local minima

### 4.4 Challenges and Limitations

**Computational Cost**:
- Each objective evaluation requires pricing $N$ options using FFT
- Typical calibration: 50-200 iterations → 2,500-10,000 FFT computations
- Time per calibration: 5-20 seconds (depending on $N$ and hardware)

**Local Minima**:
- The objective function is non-convex
- Different initializations may yield different parameters
- Multiple runs recommended to ensure global optimum

**Parameter Identifiability**:
- Some parameter combinations produce similar pricing fits
- Particularly: $\sigma$ (vol-of-vol) can be difficult to identify precisely
- Trade-offs between parameters (e.g., $\kappa$ vs. $\theta$)

**Sensitivity**:
- Solution sensitive to data quality and filtering
- Outliers can significantly affect calibrated parameters
- Robust filtering (liquidity, moneyness) essential

---

## 5. Neural Network Surrogate Development

### 5.1 Motivation

The classical calibration is computationally expensive because each optimization step requires numerous FFT-based option pricings. A neural network surrogate can:

1. **Accelerate Calibration**: Replace expensive FFT calls with fast forward passes
2. **Enable Gradient-Based Optimization**: Leverage automatic differentiation
3. **Scale to Large Portfolios**: Process many securities rapidly

### 5.2 Network Architecture

**Heston Surrogate Network**: A feedforward neural network that approximates the Heston pricing function:

$\hat{C}(K, \tau, v_0, \kappa, \theta, \sigma, \rho, r, S_0) \approx C^{\text{Heston}}(K, \tau, v_0, \kappa, \theta, \sigma, \rho, r, S_0)$

**Architecture**:

- **Input Layer**: 8 features
  - Log moneyness: $\ln(K/S_0)$
  - Time to maturity: $\tau$
  - Heston parameters: $v_0, \kappa, \theta, \sigma, \rho$
  - Risk-free rate: $r$

- **Hidden Layers**: Multi-layer perceptron (MLP)
  - Layer 1: 128 neurons, LeakyReLU activation ($\alpha=0.1$)
  - Layer 2: 128 neurons, LeakyReLU activation ($\alpha=0.1$)
  - Layer 3: 64 neurons, LeakyReLU activation ($\alpha=0.1$)
  - Layer 4: 32 neurons, LeakyReLU activation ($\alpha=0.1$)
  - Batch normalization after each hidden layer
  - Dropout: 10% for regularization

- **Output Layer**: 1 neuron (normalized option price)
  - Output: Price normalized by spot, i.e., $\hat{C}/S_0$
  - Softplus activation to ensure positive outputs

**Scale Invariance**: Normalizing by \(S_0\) makes the network scale-invariant, allowing training on diverse spot prices and generalization to different market conditions.

### 5.3 Synthetic Data Generation

**Objective**: Generate diverse, high-quality training data covering the relevant parameter space.

**Strategy**: Security-specific parameter ranges based on historical statistics.

**Parameter Sampling**:

1. **Within Historical Ranges**:
   - Sample from uniform or truncated normal distributions within bounds derived from historical data
   - Ensures relevance to the specific security

2. **Coverage** (security-specific ranges derived from historical statistics):
   - $v_0$: $[(0.5 \times \hat{\sigma}_{\text{current}})^2, (2 \times \hat{\sigma}_{\text{current}})^2]$
   - $\theta$: $[(0.5 \times \bar{\sigma})^2, (2 \times \bar{\sigma})^2]$
   - $\kappa$: $[0.5, 10.0]$ (security-specific, typically narrower based on historical mean reversion)
   - $\sigma$: $[0.1, 1.5]$ (security-specific, informed by volatility of volatility)
   - $\rho$: $[\hat{\rho} - 0.3, \min(\hat{\rho} + 0.3, 0)]$ (centered around historical leverage correlation)

**Option Contract Generation**:

For each parameter set $\eta$, generate a diverse set of option contracts:

- **Strikes**: Log-uniform distribution in moneyness range $[0.80, 1.20]$
- **Maturities**: Uniform or log-uniform in $[0.05, 1.5]$ years
- **Risk-free rate**: From yield curve (maturity-matched)

**Training Data Size**:
- **Default**: 100,000 synthetic option-price pairs per security
- **Rationale**: Balance between training time and coverage

**Validation Split**: 80% training, 20% validation

### 5.4 Training Procedure

**Loss Function**: Mean Squared Error (MSE) on normalized prices

$\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \left( \frac{\hat{C}_i}{S_{0,i}} - \frac{C_i^{\text{Heston}}}{S_{0,i}} \right)^2$

**Optimizer**: Adam with learning rate scheduling

- Initial learning rate: $10^{-3}$
- Learning rate decay: Reduce on plateau (factor 0.5, patience 10 epochs)
- Weight decay: $10^{-5}$ (L2 regularization)

**Training Process**:

1. **Data Normalization**:
   - Standardize inputs: $x_{\text{norm}} = (x - \mu_x) / \sigma_x$
   - Compute statistics from training set only

2. **Batch Training**:
   - Batch size: 256
   - Shuffle training data each epoch

3. **Early Stopping**:
   - Monitor validation loss
   - Stop if no improvement for 25 epochs
   - Save best model (lowest validation loss)

4. **Training Duration**:
   - Typical: 50-200 epochs
   - Early stopping prevents overfitting

**Training Time**: ~2-5 minutes per security (on MPS/CUDA)

### 5.5 Training Quality Metrics

**Validation Loss**:
- Target: MSE < $10^{-6}$ on normalized prices
- Typical: $10^{-6}$ to $10^{-4}$

**Accuracy Assessment**:
- Mean Absolute Percentage Error (MAPE): < 1% on validation set
- Maximum relative error: < 5% for 95% of samples

**Challenges**:
- **Edge Cases**: Extreme moneyness/maturity combinations may have higher errors
- **Parameter Boundaries**: Near-boundary parameter values require careful sampling
- **Generalization**: Must generalize to market data not seen during training

---

## 6. Neural Network-Based Calibration

### 6.1 Surrogate Calibration Objective

Replace the expensive Heston pricing function with the trained neural network:

$J^{\text{NN}}(\eta) = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{C}^{\text{NN}}(\eta, K_i, \tau_i) - C_i^{\text{Market}} \right)^2$

where $\hat{C}^{\text{NN}}$ is the neural network prediction (denormalized: $\hat{C}^{\text{NN}} = \hat{C}_{\text{norm}} \cdot S_0$).

### 6.2 Optimization Methods

**Option 1: Differential Evolution (Global Optimization)**

- **Rationale**: Avoids local minima, no gradients required
- **Parameters**:
  - Population size: 15
  - Maximum iterations: 200
  - Mutation factor: 0.5
  - Crossover probability: 0.7

- **Advantages**:
  - Robust to initialization
  - Global search capability
  - Works well with surrogate models

- **Disadvantages**:
  - More iterations needed than gradient-based methods
  - Cannot leverage network differentiability

**Option 2: Gradient-Based Optimization (PyTorch Autograd)**

- **Rationale**: Leverage automatic differentiation for efficient optimization
- **Method**: Adam or L-BFGS optimizer

- **Implementation**:
  ```python
  # Parameters as PyTorch tensors with requires_grad=True
  params = torch.tensor([v0, kappa, theta, sigma, rho], 
                       requires_grad=True)
  
  # Forward pass through network
  pred_prices = surrogate_network(features, params)
  
  # Compute loss
  loss = torch.mean((pred_prices - market_prices)**2)
  
  # Backward pass
  loss.backward()
  
  # Update parameters
  optimizer.step()
  ```

- **Advantages**:
  - Faster convergence (fewer iterations)
  - Leverages network structure
  - Can use sophisticated optimizers (L-BFGS, Adam)

- **Challenges**:
  - Requires constraint handling (Feller condition, bounds)
  - Gradient clipping may be needed for stability

**Default Choice**: Differential Evolution for robustness and simplicity.

### 6.3 Constraint Enforcement

**Parameter Bounds**: Enforced by optimizer bounds

**Feller Condition**: Two approaches:

1. **Penalty Method**: Add large penalty if violated
   ```python
   if 2 * kappa * theta < sigma**2:
       return 1e10  # Large penalty
   ```

2. **Projection Method**: Project parameters to valid region after each step

**Implementation**: Penalty method is simpler and works well with global optimizers.

### 6.4 Calibration Workflow

**Complete Pipeline**:

1. **Load Data**:
   - Market option prices
   - Trained surrogate model
   - Normalization parameters

2. **Initialize Parameters**:
   - Use historical statistics
   - Or use traditional calibration result as warm start

3. **Optimize**:
   - Run differential evolution (or gradient descent)
   - Evaluate objective using surrogate network
   - Enforce constraints

4. **Validate**:
   - Compare surrogate prices to FFT prices at calibrated parameters
   - Check Feller condition
   - Assess fit quality (RMSE, MAPE)

5. **Output**:
   - Calibrated parameters
   - Pricing errors
   - Calibration time

**Typical Performance**:
- Calibration time: 1-3 seconds (vs. 5-20 seconds for traditional)
- Speedup: Average 2.27× (range: 0.81× to 3.59×, depending on security)
- Accuracy: RMSE increases of 6.9-160% compared to traditional method, with substantial heterogeneity across securities (QQQ: 6.9% increase with 3.59× speedup; SPY: 68.0% increase with 2.40× speedup; TSLA: 160% increase with 0.81× slowdown)

---

## 7. Hybrid Calibration Approach

### 7.1 Two-Stage Strategy

Combine the speed of neural networks with the accuracy of traditional methods:

**Stage 1: Fast Neural Network Initialization**
- Run NN-based calibration (differential evolution, reduced iterations)
- Obtain approximate solution: $\eta^{\text{NN}}$

**Stage 2: Traditional Refinement**
- Use $\eta^{\text{NN}}$ as initial guess for traditional calibration
- Run SLSQP optimization with FFT pricing
- Fewer iterations needed (good starting point)
- Final solution: $\eta^{\text{Final}}$

### 7.2 Advantages

1. **Speed**: Stage 1 provides good initialization quickly
2. **Accuracy**: Stage 2 ensures final solution matches traditional calibration accuracy
3. **Reliability**: Guaranteed convergence to optimal solution

### 7.3 Implementation

```python
def hybrid_calibration(...):
    # Stage 1: Fast NN calibration (reduced iterations)
    nn_result = differential_evolution(
        nn_objective, bounds, maxiter=50, popsize=10
    )
    nn_params = nn_result.x
    nn_time = elapsed_time()
    
    # Stage 2: Traditional refinement
    fft_result = minimize(
        fft_objective, 
        x0=nn_params,  # Warm start
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 100}  # Fewer iterations needed
    )
    
    total_time = elapsed_time()
    return fft_result.x, total_time, nn_time
```

**Performance**:
- Total time: Similar to traditional (sometimes slightly faster due to better initialization)
- Accuracy: Identical to traditional calibration
- Use case: Production systems requiring guaranteed accuracy

---

## 8. Evaluation and Validation

### 8.1 Performance Metrics

**Pricing Accuracy**:
- **RMSE**: Root Mean Squared Error in dollars
  $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (C_i^{\text{Model}} - C_i^{\text{Market}})^2}$
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
  $\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{C_i^{\text{Model}} - C_i^{\text{Market}}}{C_i^{\text{Market}}} \right| \times 100\%$

**Computational Efficiency**:
- **Calibration Time**: Wall-clock time for complete calibration
- **Speedup**: Ratio of traditional time to NN time

**Parameter Recovery**:
- Compare calibrated parameters to traditional baseline
- Assess stability across multiple runs

### 8.2 Statistical Validation

**Hypothesis Testing**:
- Paired t-tests: Compare pricing errors between methods
- Wilcoxon signed-rank: Non-parametric alternative

**Effect Sizes**:
- Cohen's d: Standardized difference in means
- Practical significance: Not just statistical significance

**Robustness Checks**:
- Cross-validation across different securities
- Sensitivity to data filtering
- Stability across different market regimes

### 8.3 Model Validation Workflow

1. **In-Sample Fit**: Evaluate on calibration data
2. **Out-of-Sample Testing**: Evaluate on held-out option data
3. **Parameter Stability**: Run multiple calibrations, assess variance
4. **Greeks Accuracy**: Compare model Greeks to market-implied Greeks (if available)

### 8.4 Reporting Results

**Summary Tables**:
- Performance comparison (RMSE, MAE, time)
- Parameter comparison (calibrated values)
- Statistical test results

**Visualizations**:
- Market vs. model prices (scatter plots)
- Error distributions
- Parameter comparison (bar charts)
- Volatility surface fits

**Export Formats**:
- CSV tables for analysis
- LaTeX tables for dissertation
- JSON results for programmatic access

---

## 9. Implementation Details

### 9.1 Software Stack

- **Python 3.9+**: Core language
- **PyTorch**: Neural network framework
- **NumPy, SciPy**: Numerical computations
- **pandas**: Data manipulation
- **yfinance**: Market data
- **scikit-learn**: Utilities (normalization, etc.)

### 9.2 Hardware Considerations

**GPU Acceleration**:
- **MPS** (Metal Performance Shaders): Apple Silicon GPU acceleration
- **CUDA**: NVIDIA GPU support
- **CPU Fallback**: Automatic if GPU unavailable

**Performance**:
- Training: 5-10× faster on GPU
- Inference: 2-5× faster on GPU
- Calibration: Benefits from GPU acceleration of batch pricing

### 9.3 Code Organization

```
project/
├── src/
│   ├── heston_model.py              # Heston model implementation
│   ├── neural_networks.py           # Network architectures
│   ├── data_generation.py           # Data fetching and generation
│   ├── training_pipeline.py         # Training procedures
│   ├── nn_calibration.py            # NN-based calibration
│   └── heston_model_classic_calibration.py  # Traditional calibration
├── notebooks/
│   ├── data_collection.ipynb        # Market data fetching
│   ├── assessment.ipynb             # Results evaluation
│   └── statistical_assessment.ipynb # Statistical analysis
├── main.py                          # Main calibration workflow
└── data/
    ├── raw/                         # Raw market data
    ├── processed/                   # Processed calibration data
    └── models/                      # Trained models and results
```

### 9.4 Reproducibility

**Random Seeds**:
- Set seeds for NumPy, PyTorch, random module
- Ensures reproducible training and optimization

**Version Control**:
- Track code versions
- Document dependencies (requirements.txt)

**Data Versioning**:
- Timestamp all data fetches
- Save raw and processed data

---

## 10. Summary

This methodology provides a comprehensive framework for calibrating the Heston stochastic volatility model using deep learning techniques. The key innovations are:

1. **Data-Driven Parameter Initialization**: Historical statistics inform calibration bounds and starting points

2. **Security-Specific Training**: Neural networks trained on security-specific parameter ranges for better generalization

3. **Hybrid Calibration**: Combines neural network speed with traditional accuracy

4. **Robust Validation**: Comprehensive statistical assessment and robustness checks

The framework achieves average computational speedup of 2.27× (with substantial heterogeneity: range from 0.81× slowdown to 3.59× speedup depending on security characteristics) while maintaining acceptable accuracy for moderate-volatility securities, making it suitable for selective deployment in production trading systems and large-scale portfolio calibration.

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Author**: Joris Marvezy

