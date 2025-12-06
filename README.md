# Deep Learning Framework for Heston Model Calibration

A comprehensive framework for calibrating the Heston stochastic volatility model using neural network surrogates. This research project demonstrates how deep learning can accelerate option pricing model calibration while maintaining acceptable accuracy.

## Overview

This framework combines traditional numerical methods (FFT-based Heston pricing) with neural network surrogate models to achieve significant computational speedup in model calibration. The methodology includes:

- **Data Collection**: Automated market data fetching (options, underlying prices, yield curves)
- **Classical Calibration**: Traditional FFT-based Heston model calibration
- **Neural Network Training**: Security-specific surrogate model training
- **Neural Network Calibration**: Fast calibration using trained surrogates
- **Hybrid Approach**: Two-stage calibration combining neural network speed with traditional accuracy

## Key Results

- **Average Speedup**: 2.27× faster than traditional calibration
- **Accuracy Trade-offs**: 7-160% RMSE increase depending on security characteristics
- **Robustness**: Validated across diverse securities (QQQ, SPY, TSLA) and volatility regimes
- **Production Ready**: Hybrid approach provides guaranteed accuracy with faster convergence

## Project Structure

```
.
├── src/                        # Source code
│   ├── heston_model.py         # Heston model implementation
│   ├── neural_networks.py      # Neural network architectures
│   ├── data_generation.py      # Market data fetching and synthetic generation
│   ├── training_pipeline.py    # Neural network training
│   ├── nn_calibration.py       # Neural network-based calibration
│   └── ...
├── notebooks/                  # Jupyter notebooks
│   ├── data_collection.ipynb   # Market data collection
│   ├── assessment.ipynb        # Results evaluation
│   └── statistical_assessment.ipynb  # Statistical analysis
├── data/                       # Data directory
│   ├── raw/                    # Raw market data
│   ├── processed/              # Processed calibration data
│   └── models/                 # Trained models and results
├── results/                    # Results and outputs
│   ├── figures/                # Plots and visualizations
│   ├── tables/                 # Statistical tables
│   └── statistics/             # Statistical summaries
├── main.py                     # Main calibration workflow
└── requirements.txt            # Python dependencies
```

## Installation

### Prerequisites

- Python 3.9+
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/heston-neural-calibration.git
cd heston-neural-calibration
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Market Data

First, fetch market data using the data collection notebook:

```bash
jupyter notebook notebooks/data_collection.ipynb
```

This will:
- Fetch Treasury yield curve data
- Download historical spot prices
- Collect current option chains
- Save data to `data/` directory

### 2. Run Calibration

Run the complete calibration workflow:

```bash
# Calibrate for one security
python main.py --tickers QQQ --history-months 3

# Calibrate for multiple securities
python main.py --tickers QQQ SPY TSLA

# Custom training parameters
python main.py --tickers QQQ --epochs 300 --samples 200000

# Skip training (use existing model)
python main.py --tickers QQQ --skip-training
```

### 3. Evaluate Results

Analyze results using the assessment notebooks:

```bash
jupyter notebook notebooks/assessment.ipynb
jupyter notebook notebooks/statistical_assessment.ipynb
```

## Usage Examples

### Traditional Calibration

```python
from src.heston_model_classic_calibration import calibrate_heston_to_market
import pandas as pd

# Load market data
market_data = pd.read_csv('data/processed/calibration/QQQ_calibration.csv')

# Calibrate
params = calibrate_heston_to_market(
    market_data,
    S0=market_data['spot'].iloc[0],
    pricing_method='fft'
)

print(f"Calibrated parameters: {params}")
```

### Neural Network Calibration

```python
from src.nn_calibration import NeuralNetworkCalibrator
import torch

# Load trained model
calibrator = NeuralNetworkCalibrator(
    model_path='data/models/QQQ_surrogate.pth',
    device='mps'  # or 'cuda', 'cpu'
)

# Calibrate
params = calibrator.calibrate(
    S0=619.25,
    strikes=market_data['strike'].values,
    maturities=market_data['tau'].values,
    market_prices=market_data['mid_price'].values,
    r=0.04
)
```

## Configuration

The framework uses security-specific parameter ranges derived from historical data. Key configuration options:

- **Training epochs**: Number of training epochs (default: 200)
- **Synthetic samples**: Number of synthetic training samples (default: 100,000)
- **Hidden layers**: Neural network architecture (default: [128, 128, 64, 32])
- **Historical period**: Months of historical data for parameter estimation (default: 3)

## Results

### Performance Summary

| Security | Traditional RMSE | NN RMSE | Speedup |
|----------|-----------------|---------|---------|
| QQQ      | $0.492          | $0.526  | 3.59×   |
| SPY      | $0.552          | $0.928  | 2.40×   |
| TSLA     | $0.738          | $1.919  | 0.81×   |

### Key Findings

- Neural network calibration achieves substantial speedup for most securities
- Accuracy trade-offs vary significantly by security characteristics
- High-volatility securities with moderate option counts show best performance
- Hybrid approach combines speed and accuracy effectively

See `results/` directory for detailed statistical analysis and visualizations.

## Methodology

The framework implements:

1. **Data Collection**: Automated fetching of market data (yfinance API)
2. **Parameter Initialization**: Data-driven parameter bounds from historical statistics
3. **Synthetic Data Generation**: Security-specific training data generation
4. **Neural Network Training**: Surrogate model training with early stopping
5. **Calibration**: Optimization-based parameter estimation
6. **Evaluation**: Comprehensive statistical analysis and robustness assessment

For detailed methodology, see the dissertation documentation.

## Requirements

Key dependencies:

- `torch` - Neural network framework
- `numpy`, `scipy` - Numerical computations
- `pandas` - Data manipulation
- `yfinance` - Market data
- `matplotlib`, `seaborn` - Visualization
- `statsmodels` - Statistical analysis

See `requirements.txt` for complete list.

## Hardware Requirements

- **Minimum**: CPU-only (slower training and inference)
- **Recommended**: GPU support (CUDA or MPS for Mac)
- **Training Time**: ~2-5 minutes per security on GPU
- **Calibration Time**: 1-6 seconds per security

## Citation

If you use this framework in your research, please cite:

```bibtex
@masterthesis{marvezy2025heston,
  author = {Marvezy, Joris},
  title = {Deep Learning Framework for Heston Model Calibration},
  school = {TBS},
  year = {2025}
}
```

## License

This project is provided for academic and research purposes. Please cite appropriately if used in research.

## Contributing

This is a research project, but contributions and suggestions are welcome:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

## Acknowledgments

- Built on PyTorch and open-source quantitative finance libraries
- Market data provided by Yahoo Finance via yfinance and the U.S. DEPARTMENT OF THE TREASURY for Daily Treasury Par Yield Curve Rates
- Inspired by research in machine learning for quantitative finance

## Contact

For questions or issues, please open an issue on GitHub.

## References

### Stochastic Volatility Models

- **Heston, S. L.** (1993). "A closed-form solution for options with stochastic volatility with applications to bond and currency options." *Review of Financial Studies*, 6(2), 327-343. [[DOI]](https://doi.org/10.1093/rfs/6.2.327)
- **Hull, J., & White, A.** (1987). "The pricing of options on assets with stochastic volatilities." *Journal of Finance*, 42(2), 281-300. [[DOI]](https://doi.org/10.1111/j.1540-6261.1987.tb02568.x)
- **Stein, E. M., & Stein, J. C.** (1991). "Stock price distributions with stochastic volatility: an analytic approach." *Review of Financial Studies*, 4(4), 727-752. [[DOI]](https://doi.org/10.1093/rfs/4.4.727)
- **Bates, D. S.** (1996). "Jumps and stochastic volatility: exchange rate processes implicit in deutsche mark options." *Review of Financial Studies*, 9(1), 69-107. [[DOI]](https://doi.org/10.1093/rfs/9.1.69)

### Option Pricing and Numerical Methods

- **Black, F., & Scholes, M.** (1973). "The pricing of options and corporate liabilities." *Journal of Political Economy*, 81(3), 637-654. [[DOI]](https://doi.org/10.1086/260062)
- **Carr, P., & Madan, D.** (1999). "Option valuation using the fast Fourier transform." *Journal of Computational Finance*, 2(4), 61-73. [[DOI]](https://doi.org/10.21314/JCF.1999.043)
- **Gatheral, J.** (2006). *The Volatility Surface: A Practitioner's Guide*. John Wiley & Sons.
- **Rouah, F. D.** (2013). *The Heston Model and Its Extensions in Matlab and C#*. John Wiley & Sons.

### Yield Curve Models

- **Nelson, C. R., & Siegel, A. F.** (1987). "Parsimonious modeling of yield curves." *Journal of Business*, 60(4), 473-489. [[DOI]](https://doi.org/10.1086/296409)
- **Svensson, L. E. O.** (1994). "Estimating and interpreting forward interest rates: Sweden 1992-1994." *NBER Working Paper*, No. 4871. [[DOI]](https://doi.org/10.3386/w4871)

### Machine Learning in Quantitative Finance

- **Hutchinson, J. M., Lo, A. W., & Poggio, T.** (1994). "A nonparametric approach to pricing and hedging derivative securities via learning networks." *Journal of Finance*, 49(3), 851-889. [[DOI]](https://doi.org/10.1111/j.1540-6261.1994.tb00081.x)
- **Ruf, J., & Wang, W.** (2020). "Neural networks for option pricing and hedging: a literature review." *Journal of Computational Finance*, 24(1), 1-29. [[DOI]](https://doi.org/10.21314/JCF.2020.376)
- **Liu, S., Oosterlee, C. W., & Bohte, S. M.** (2019). "Pricing options and computing implied volatilities using neural networks." *Risks*, 7(1), 16. [[DOI]](https://doi.org/10.3390/risks7010016)
- **Lapeyre, B., & Lelong, J.** (2021). "Neural network regression for Bermudan option pricing." *Monte Carlo Methods and Applications*, 27(3), 227-247. [[DOI]](https://doi.org/10.1515/mcma-2021-2098)

### Model Calibration

- **Cont, R., & Tankov, P.** (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.
- **Rebonato, R.** (2004). *Volatility and Correlation: The Perfect Hedger and the Fox* (2nd ed.). John Wiley & Sons.
- **Schoutens, W.** (2003). *Levy Processes in Finance: Pricing Financial Derivatives*. John Wiley & Sons.

### Neural Networks and Deep Learning

- **Goodfellow, I., Bengio, Y., & Courville, A.** (2016). *Deep Learning*. MIT Press. [[Online]](https://www.deeplearningbook.org/)
- **LeCun, Y., Bengio, Y., & Hinton, G.** (2015). "Deep learning." *Nature*, 521(7553), 436-444. [[DOI]](https://doi.org/10.1038/nature14539)

### Computational Finance

- **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- **Higham, D. J.** (2004). "An introduction to financial option valuation: mathematics, stochastics and computation." *Cambridge University Press*.
- **Wilmott, P.** (2007). *Paul Wilmott Introduces Quantitative Finance* (2nd ed.). John Wiley & Sons.

### Statistical Methods

- **Cohen, J.** (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Routledge.
- **Benjamini, Y., & Hochberg, Y.** (1995). "Controlling the false discovery rate: A practical and powerful approach to multiple testing." *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300. [[DOI]](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

### Software and Libraries

- **Paszke, A., et al.** (2019). "PyTorch: An imperative style, high-performance deep learning library." *Advances in Neural Information Processing Systems*, 32, 8024-8035. [[Paper]](https://papers.neurips.cc/paper/9015-pytorch-an-imperative-style-high-performance-deep-learning-library.pdf)
- **yfinance**: Yahoo Finance market data downloader. Available at: https://github.com/ranaroussi/yfinance

---

**Author**: Joris Marvezy  
**Institution**: TBS   
**Year**: 2025

