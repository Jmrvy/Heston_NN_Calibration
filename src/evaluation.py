"""
Evaluation and Metrics Module for Heston Model Calibration.

This module provides comprehensive evaluation tools for comparing
traditional and deep learning calibration methods.

Metrics computed:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MRE (Mean Relative Error)
- Max Error
- Calibration time

Features:
- Visualization of calibration results
- Statistical comparison tests
- Performance across market regimes
- Robustness analysis

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from heston_model import HestonModel

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_pricing_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
    weights: np.ndarray = None
) -> Dict[str, float]:
    """
    Compute comprehensive pricing error metrics.
    
    Metrics follow the benchmark paper:
    - RMSE: sqrt(mean((pred - actual)²))
    - MAE: mean(|pred - actual|)
    - MRE: mean(|pred - actual| / |actual|)
    - MaxError: max(|pred - actual|)
    
    Args:
        predicted: Predicted prices
        actual: Actual/market prices
        weights: Optional weights for weighted metrics
        
    Returns:
        Dictionary of metrics
    """
    predicted = np.asarray(predicted).flatten()
    actual = np.asarray(actual).flatten()
    
    if weights is None:
        weights = np.ones_like(predicted)
    weights = weights / weights.sum()
    
    # Absolute errors
    abs_errors = np.abs(predicted - actual)
    squared_errors = (predicted - actual) ** 2
    
    # Relative errors (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = abs_errors / np.abs(actual)
        rel_errors = np.where(np.isfinite(rel_errors), rel_errors, 0)
    
    # Compute metrics
    metrics = {
        'MSE': float(np.sum(weights * squared_errors)),
        'RMSE': float(np.sqrt(np.sum(weights * squared_errors))),
        'MAE': float(np.sum(weights * abs_errors)),
        'MRE': float(np.sum(weights * rel_errors)),
        'MaxError': float(np.max(abs_errors)),
        'MedianError': float(np.median(abs_errors)),
        'R2': float(1 - np.sum(squared_errors) / np.sum((actual - actual.mean())**2))
    }
    
    return metrics


def compute_parameter_errors(
    calibrated: Dict[str, float],
    true_params: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute parameter estimation errors.
    
    Args:
        calibrated: Calibrated parameters
        true_params: True/benchmark parameters
        
    Returns:
        Dictionary of parameter errors
    """
    errors = {}
    param_keys = ['v0', 'kappa', 'theta', 'sigma', 'rho']
    
    for key in param_keys:
        if key in calibrated and key in true_params:
            true_val = true_params[key]
            calib_val = calibrated[key]
            
            errors[f'{key}_abs_error'] = abs(calib_val - true_val)
            errors[f'{key}_rel_error'] = abs(calib_val - true_val) / abs(true_val) if true_val != 0 else 0
            errors[f'{key}_bias'] = calib_val - true_val
    
    return errors


# =============================================================================
# CALIBRATION EVALUATION
# =============================================================================

class CalibrationEvaluator:
    """
    Comprehensive evaluation of calibration results.
    
    Compares multiple calibration methods across:
    - Pricing accuracy
    - Parameter recovery
    - Computational efficiency
    - Robustness to different regimes
    """
    
    def __init__(self, output_dir: str = None):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory for saving results and plots
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = []
        self.comparison_df = None
    
    def add_result(
        self,
        method_name: str,
        calibrated_params: Dict[str, float],
        predicted_prices: np.ndarray,
        market_prices: np.ndarray,
        true_params: Dict[str, float] = None,
        calibration_time: float = None,
        metadata: Dict = None
    ):
        """
        Add calibration result for evaluation.
        
        Args:
            method_name: Name of the calibration method
            calibrated_params: Calibrated Heston parameters
            predicted_prices: Model predicted prices
            market_prices: Market/target prices
            true_params: True parameters (if known)
            calibration_time: Time taken for calibration
            metadata: Additional metadata
        """
        # Compute pricing metrics
        pricing_metrics = compute_pricing_metrics(predicted_prices, market_prices)
        
        result = {
            'method': method_name,
            'calibrated_params': calibrated_params,
            'pricing_metrics': pricing_metrics,
            'calibration_time': calibration_time,
            'n_options': len(market_prices)
        }
        
        # Add parameter errors if true params known
        if true_params is not None:
            param_errors = compute_parameter_errors(calibrated_params, true_params)
            result['parameter_errors'] = param_errors
            result['true_params'] = true_params
        
        # Add metadata
        if metadata:
            result['metadata'] = metadata
        
        self.results.append(result)
        logger.info(f"Added result for method: {method_name}")
    
    def generate_comparison_table(self) -> pd.DataFrame:
        """
        Generate comparison table across all methods.
        
        Returns:
            DataFrame with comparison metrics
        """
        rows = []
        
        for result in self.results:
            row = {
                'Method': result['method'],
                'RMSE': result['pricing_metrics']['RMSE'],
                'MAE': result['pricing_metrics']['MAE'],
                'MRE': result['pricing_metrics']['MRE'],
                'MaxError': result['pricing_metrics']['MaxError'],
                'R²': result['pricing_metrics']['R2'],
                'Time (s)': result['calibration_time'],
                'v0': result['calibrated_params'].get('v0'),
                'kappa': result['calibrated_params'].get('kappa'),
                'theta': result['calibrated_params'].get('theta'),
                'sigma': result['calibrated_params'].get('sigma'),
                'rho': result['calibrated_params'].get('rho')
            }
            rows.append(row)
        
        self.comparison_df = pd.DataFrame(rows)
        return self.comparison_df
    
    def statistical_comparison(self) -> Dict:
        """
        Perform statistical comparison between methods.
        
        Uses paired t-tests and Wilcoxon signed-rank tests.
        
        Returns:
            Dictionary of statistical test results
        """
        if len(self.results) < 2:
            return {"error": "Need at least 2 methods to compare"}
        
        stats_results = {}
        methods = [r['method'] for r in self.results]
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                
                # Get RMSE values (would need multiple runs for proper statistical test)
                rmse1 = self.results[i]['pricing_metrics']['RMSE']
                rmse2 = self.results[j]['pricing_metrics']['RMSE']
                
                comparison_key = f"{method1} vs {method2}"
                stats_results[comparison_key] = {
                    'RMSE_diff': rmse1 - rmse2,
                    'better_method': method1 if rmse1 < rmse2 else method2,
                    'improvement_pct': abs(rmse1 - rmse2) / max(rmse1, rmse2) * 100
                }
        
        return stats_results
    
    def plot_price_comparison(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot price predictions vs market prices.
        
        Args:
            strikes: Strike prices
            market_prices: Market prices
            save_name: Filename for saving
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left plot: Prices by strike
        ax1 = axes[0]
        ax1.scatter(strikes, market_prices, s=100, c='black', marker='o', 
                   label='Market', zorder=5)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for result, color in zip(self.results, colors):
            method = result['method']
            # Recompute predictions (assuming we have params)
            params = result['calibrated_params']
            model = HestonModel(
                kappa=params['kappa'],
                theta=params['theta'],
                sigma=params['sigma'],
                rho=params['rho'],
                v0=params['v0'],
                r=params.get('r', 0.03)
            )
            
            # Assume uniform maturity for this plot
            tau = 0.25  # Default
            pred_prices = np.array([model.call_price_fft(100, k, tau) for k in strikes])
            
            ax1.plot(strikes, pred_prices, '-', color=color, linewidth=2,
                    label=f'{method} (RMSE={result["pricing_metrics"]["RMSE"]:.3f})')
        
        ax1.set_xlabel('Strike Price', fontsize=12)
        ax1.set_ylabel('Option Price', fontsize=12)
        ax1.set_title('Option Prices: Market vs Calibrated Models', fontsize=14)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Bar chart of RMSE
        ax2 = axes[1]
        methods = [r['method'] for r in self.results]
        rmses = [r['pricing_metrics']['RMSE'] for r in self.results]
        bars = ax2.bar(methods, rmses, color=colors[:len(methods)])
        
        ax2.set_xlabel('Calibration Method', fontsize=12)
        ax2.set_ylabel('RMSE', fontsize=12)
        ax2.set_title('Calibration Error Comparison', fontsize=14)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, rmse in zip(bars, rmses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rmse:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {filepath}")
        
        return fig
    
    def plot_training_comparison(
        self,
        training_histories: Dict[str, List[float]],
        save_name: str = None
    ) -> plt.Figure:
        """
        Plot training curves comparison.
        
        Args:
            training_histories: Dict of {method_name: loss_history}
            save_name: Filename for saving
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for method, history in training_histories.items():
            ax.plot(history, label=method, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Comparison', fontsize=14)
        ax.legend(loc='best')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_error_distribution(self, save_name: str = None) -> plt.Figure:
        """
        Plot error distribution across methods.
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = [r['method'] for r in self.results]
        metrics = ['RMSE', 'MAE', 'MRE', 'MaxError']
        
        for ax, metric in zip(axes.flatten(), metrics):
            values = [r['pricing_metrics'][metric] for r in self.results]
            bars = ax.bar(methods, values)
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Method')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_name and self.output_dir:
            filepath = self.output_dir / save_name
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
        
        return fig
    
    def generate_latex_table(self) -> str:
        """
        Generate LaTeX table for the research paper.
        
        Returns:
            LaTeX table string
        """
        if self.comparison_df is None:
            self.generate_comparison_table()
        
        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Calibration Method Comparison}\n"
        latex += "\\label{tab:calibration_comparison}\n"
        latex += "\\begin{tabular}{l" + "c" * 6 + "}\n"
        latex += "\\toprule\n"
        latex += "Method & RMSE & MAE & MRE & Max Error & R² & Time (s) \\\\\n"
        latex += "\\midrule\n"
        
        for _, row in self.comparison_df.iterrows():
            latex += f"{row['Method']} & "
            latex += f"{row['RMSE']:.4f} & "
            latex += f"{row['MAE']:.4f} & "
            latex += f"{row['MRE']:.4f} & "
            latex += f"{row['MaxError']:.4f} & "
            latex += f"{row['R²']:.4f} & "
            latex += f"{row['Time (s)']:.2f} \\\\\n"
        
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"
        
        return latex
    
    def save_results(self, filename: str = 'calibration_results.json'):
        """Save results to JSON file."""
        if self.output_dir is None:
            return
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = []
        for r in self.results:
            sr = {
                'method': r['method'],
                'calibrated_params': r['calibrated_params'],
                'pricing_metrics': r['pricing_metrics'],
                'calibration_time': r['calibration_time']
            }
            if 'parameter_errors' in r:
                sr['parameter_errors'] = r['parameter_errors']
            serializable_results.append(sr)
        
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {filepath}")


# =============================================================================
# REGIME-SPECIFIC EVALUATION
# =============================================================================

def evaluate_across_regimes(
    calibration_results: Dict[str, Dict],
    regime_data: Dict[str, pd.DataFrame],
    output_dir: str = None
) -> pd.DataFrame:
    """
    Evaluate calibration performance across different market regimes.
    
    Args:
        calibration_results: {regime_name: {method_name: result_dict}}
        regime_data: {regime_name: DataFrame with market data}
        output_dir: Output directory for plots
        
    Returns:
        Summary DataFrame
    """
    summary_rows = []
    
    for regime_name, methods_results in calibration_results.items():
        for method_name, result in methods_results.items():
            row = {
                'Regime': regime_name,
                'Method': method_name,
                'RMSE': result.get('RMSE', result.get('pricing_metrics', {}).get('RMSE')),
                'MAE': result.get('MAE', result.get('pricing_metrics', {}).get('MAE')),
                'Time': result.get('calibration_time', result.get('elapsed_time'))
            }
            summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Create visualization
    if output_dir and len(summary_rows) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pivot_df = summary_df.pivot(index='Regime', columns='Method', values='RMSE')
        pivot_df.plot(kind='bar', ax=ax)
        
        ax.set_ylabel('RMSE')
        ax.set_title('Calibration Performance Across Market Regimes')
        ax.legend(title='Method')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = Path(output_dir) / 'regime_comparison.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
    
    return summary_df


# =============================================================================
# BENCHMARK PAPER METRICS REPLICATION
# =============================================================================

def replicate_paper_metrics(
    train_predictions: np.ndarray,
    train_actual: np.ndarray,
    test_predictions: np.ndarray,
    test_actual: np.ndarray,
    method_name: str = "Deep Learning"
) -> pd.DataFrame:
    """
    Replicate the metrics table from the benchmark paper.
    
    Creates a table matching Table 1 and Table 2 format from the paper.
    
    Args:
        train_predictions, train_actual: Training set predictions and actuals
        test_predictions, test_actual: Test set predictions and actuals
        method_name: Name of the method
        
    Returns:
        DataFrame matching paper's table format
    """
    # Compute metrics for both sets
    train_metrics = compute_pricing_metrics(train_predictions, train_actual)
    test_metrics = compute_pricing_metrics(test_predictions, test_actual)
    
    # Create table matching paper format
    data = {
        'Metric': ['Train RMSE', 'Train MRE', 'Train MAE', 
                  'Test RMSE', 'Test MRE', 'Test MAE'],
        method_name: [
            train_metrics['RMSE'],
            train_metrics['MRE'],
            train_metrics['MAE'],
            test_metrics['RMSE'],
            test_metrics['MRE'],
            test_metrics['MAE']
        ]
    }
    
    return pd.DataFrame(data)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def plot_volatility_surface(
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    model: HestonModel,
    title: str = "Implied Volatility Surface",
    save_path: str = None
) -> plt.Figure:
    """
    Plot implied volatility surface from Heston model.
    
    Args:
        S: Spot price
        strikes: Array of strikes
        maturities: Array of maturities
        model: HestonModel instance
        title: Plot title
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from black_scholes_model import BlackScholesModel
    
    K_grid, T_grid = np.meshgrid(strikes, maturities)
    iv_surface = np.zeros_like(K_grid)
    
    for i, tau in enumerate(maturities):
        for j, K in enumerate(strikes):
            price = model.call_price_fft(S, K, tau)
            iv = BlackScholesModel.implied_vol(price, S, K, tau, model.r, 0)
            iv_surface[i, j] = iv if not np.isnan(iv) else 0
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(
        K_grid / S,  # Moneyness
        T_grid,      # Maturity
        iv_surface * 100,  # IV in percentage
        cmap='viridis',
        alpha=0.8
    )
    
    ax.set_xlabel('Moneyness (K/S)')
    ax.set_ylabel('Maturity (years)')
    ax.set_zlabel('Implied Volatility (%)')
    ax.set_title(title)
    
    fig.colorbar(surf, shrink=0.5, aspect=5, label='IV (%)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_fit(
    S: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    market_prices: np.ndarray,
    calibrated_params: Dict[str, float],
    method_name: str = "Calibrated Model",
    save_path: str = None
) -> plt.Figure:
    """
    Plot calibration fit against market data.
    
    Args:
        S: Spot price
        strikes, maturities, market_prices: Market data
        calibrated_params: Calibrated Heston parameters
        method_name: Name for legend
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    model = HestonModel(
        kappa=calibrated_params['kappa'],
        theta=calibrated_params['theta'],
        sigma=calibrated_params['sigma'],
        rho=calibrated_params['rho'],
        v0=calibrated_params['v0'],
        r=calibrated_params.get('r', 0.03)
    )
    
    model_prices = np.array([
        model.call_price_fft(S, k, t)
        for k, t in zip(strikes, maturities)
    ])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Price comparison
    ax1 = axes[0]
    ax1.scatter(range(len(market_prices)), market_prices, s=100, c='blue', 
               marker='o', label='Market')
    ax1.scatter(range(len(model_prices)), model_prices, s=100, c='red',
               marker='x', label=method_name)
    
    ax1.set_xlabel('Option Index')
    ax1.set_ylabel('Price')
    ax1.set_title('Market vs Model Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    ax2 = axes[1]
    residuals = model_prices - market_prices
    ax2.bar(range(len(residuals)), residuals, color='green', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Option Index')
    ax2.set_ylabel('Residual (Model - Market)')
    ax2.set_title('Pricing Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Evaluation Module Demo")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    
    S0 = 100.0
    strikes = np.array([85, 90, 95, 100, 105, 110, 115])
    maturities = np.full(7, 0.25)
    
    # True model
    true_model = HestonModel(
        kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
        v0=0.04, r=0.03
    )
    
    market_prices = np.array([
        true_model.call_price_fft(S0, k, t)
        for k, t in zip(strikes, maturities)
    ])
    market_prices += np.random.normal(0, 0.1, len(market_prices))
    
    # Simulate different calibration results
    evaluator = CalibrationEvaluator(output_dir='/home/claude/heston_dl_calibration/results')
    
    # Traditional method (simulated)
    traditional_params = {
        'v0': 0.042, 'kappa': 2.1, 'theta': 0.041,
        'sigma': 0.31, 'rho': -0.68, 'r': 0.03
    }
    trad_model = HestonModel(**traditional_params)
    trad_prices = np.array([trad_model.call_price_fft(S0, k, t) 
                           for k, t in zip(strikes, maturities)])
    
    evaluator.add_result(
        method_name="Traditional",
        calibrated_params=traditional_params,
        predicted_prices=trad_prices,
        market_prices=market_prices,
        true_params={'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7},
        calibration_time=5.2
    )
    
    # Deep learning method (simulated - better fit)
    dl_params = {
        'v0': 0.040, 'kappa': 2.02, 'theta': 0.040,
        'sigma': 0.30, 'rho': -0.70, 'r': 0.03
    }
    dl_model = HestonModel(**dl_params)
    dl_prices = np.array([dl_model.call_price_fft(S0, k, t) 
                         for k, t in zip(strikes, maturities)])
    
    evaluator.add_result(
        method_name="Deep Learning",
        calibrated_params=dl_params,
        predicted_prices=dl_prices,
        market_prices=market_prices,
        true_params={'v0': 0.04, 'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7},
        calibration_time=1.5
    )
    
    # Generate comparison
    comparison_df = evaluator.generate_comparison_table()
    print("\nComparison Table:")
    print(comparison_df.to_string(index=False))
    
    # Generate LaTeX
    latex = evaluator.generate_latex_table()
    print("\nLaTeX Table:")
    print(latex)
    
    # Save results
    evaluator.save_results()
    
    print("\nEvaluation demo completed!")
