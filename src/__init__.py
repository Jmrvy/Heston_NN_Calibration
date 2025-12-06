"""
Heston Deep Learning Calibration Package.

Modules:
- heston_model: Heston stochastic volatility model
- black_scholes_model: Black-Scholes for implied vol
- neural_networks: PAN, CCN, Surrogate network architectures
- training_pipeline: Data generation and training
- nn_calibration: Neural network-based calibration
- evaluation: Metrics and visualization
"""

from .heston_model import HestonModel
from .neural_networks import (
    HestonSurrogateNetwork,
    PriceApproximatorNetwork,
    CalibrationCorrectionNetwork
)
from .nn_calibration import NeuralNetworkCalibrator, TwoPhaseCalibrator

__version__ = "1.0.0"
__author__ = "Joris Marvezy (2025)"
