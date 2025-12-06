"""
Neural Network Architectures for Heston Model Calibration.

This module implements the deep learning framework

Components:
1. Price Approximator Network (PAN): Approximates the Heston pricing function
2. Calibration Correction Network (CCN): Refines calibration outputs
3. Surrogate Pricing Network: Full pricing surrogate for fast calibration

Author: Joris Marvezy (2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List, Union
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# PRICE APPROXIMATOR NETWORK (PAN)
# =============================================================================

class PriceApproximatorNetwork(nn.Module):
    """
    Price Approximator Network (PAN).
    
    Approximates the relationship between strike price and option price.
    Uses a simple architecture with tanh and ReLU activations.
    
    Architecture:
        Input Layer: 1 neuron (strike K)
        Hidden Layer 1: 8 neurons, tanh activation
        Hidden Layer 2: 8 neurons, ReLU activation
        Output Layer: 1 neuron (predicted price)
    """
    
    def __init__(self, hidden_size: int = 8):
        """
        Initialize PAN architecture.
        
        Args:
            hidden_size: Number of neurons in hidden layers (default: 8)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Layer definitions
        self.fc1 = nn.Linear(1, hidden_size)  # Input: strike price
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output: option price
        
        # Initialize weights using Kaiming initialization for ReLU layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Apply Kaiming initialization for better training stability."""
        # First layer uses tanh, so Xavier init is appropriate
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        # Second layer uses ReLU, so Kaiming init
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        
        # Output layer
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PAN.
        
        Args:
            x: Input tensor of shape (batch_size, 1) containing strike prices
            
        Returns:
            Predicted option prices of shape (batch_size, 1)
        """
        # First hidden layer with tanh
        h1 = torch.tanh(self.fc1(x))
        
        # Second hidden layer with ReLU
        h2 = F.relu(self.fc2(h1))
        
        # Output layer (no activation for regression)
        out = self.fc3(h2)
        
        return out


# =============================================================================
# CALIBRATION CORRECTION NETWORK (CCN)
# =============================================================================

class CalibrationCorrectionNetwork(nn.Module):
    """
    Calibration Correction Network (CCN).
    
    Refines the Heston model's pricing output by learning systematic
    corrections to reduce pricing errors.
    
    Architecture:
        Input Layer: 1 neuron (Heston model price)
        Hidden Layer 1: 7 neurons, sigmoid activation
        Hidden Layer 2: 7 neurons, tanh activation
        Output Layer: 1 neuron (corrected price)
    """
    
    def __init__(self, hidden_size: int = 7):
        """
        Initialize CCN architecture.
        
        Args:
            hidden_size: Number of neurons in hidden layers (default: 7)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        
        # Layer definitions
        self.fc1 = nn.Linear(1, hidden_size)  # Input: Heston price
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Output: corrected price
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Apply appropriate initialization for sigmoid/tanh activations."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CCN.
        
        Args:
            x: Input tensor of shape (batch_size, 1) containing Heston prices
            
        Returns:
            Corrected prices of shape (batch_size, 1)
        """
        # First hidden layer with sigmoid
        h1 = torch.sigmoid(self.fc1(x))
        
        # Second hidden layer with tanh
        h2 = torch.tanh(self.fc2(h1))
        
        # Output layer
        out = self.fc3(h2)
        
        return out


# =============================================================================
# EXTENDED PRICE APPROXIMATOR NETWORK
# =============================================================================

class ExtendedPAN(nn.Module):
    """
    Extended Price Approximator Network.
    
    A more comprehensive network that takes all pricing inputs:
    - Strike price (K) or moneyness (K/S)
    - Time to maturity (tau)
    - Heston parameters: v0, kappa, theta, sigma, rho
    - Risk-free rate (r)
    
    This network serves as a surrogate for the Heston pricing function
    for fast calibration.
    """
    
    def __init__(
        self,
        input_dim: int = 8,  # K/S, tau, v0, kappa, theta, sigma, rho, r
        hidden_dims: List[int] = [64, 64, 32],
        dropout: float = 0.1
    ):
        """
        Initialize Extended PAN.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            
            # Use different activations for different layers
            if i == 0:
                layers.append(nn.Tanh())
            else:
                layers.append(nn.LeakyReLU(0.1))
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Apply Kaiming initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Features: [moneyness, tau, v0, kappa, theta, sigma, rho, r]
               
        Returns:
            Predicted option prices of shape (batch_size, 1)
        """
        return self.network(x)


# =============================================================================
# HESTON SURROGATE NETWORK (Full Pricing Surrogate)
# =============================================================================

class HestonSurrogateNetwork(nn.Module):
    """
    Comprehensive surrogate network for Heston option pricing.
    
    This network learns to approximate the entire Heston pricing function,
    taking all model parameters and option characteristics as input.
    Designed for fast calibration by replacing expensive numerical integration.
    
    Input features:
        - log_moneyness: ln(K/S)
        - tau: time to maturity
        - v0: initial variance
        - kappa: mean reversion speed
        - theta: long-term variance
        - sigma: volatility of volatility
        - rho: correlation
        - r: risk-free rate
        
    Output:
        - Normalized option price (price / S)
    """
    
    def __init__(
        self,
        hidden_layers: List[int] = [128, 128, 64, 32],
        activation: str = 'leaky_relu',
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        """
        Initialize Heston Surrogate Network.
        
        Args:
            hidden_layers: Sizes of hidden layers
            activation: Activation function ('relu', 'leaky_relu', 'tanh', 'elu')
            batch_norm: Whether to use batch normalization
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = 8  # log_moneyness, tau, v0, kappa, theta, sigma, rho, r
        self.hidden_layers = hidden_layers
        
        # Activation function
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'selu': nn.SELU()
        }
        self.activation_fn = activations.get(activation, nn.LeakyReLU(0.1))
        
        # Build the network
        layers = []
        prev_dim = self.input_dim
        
        for dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(self.activation_fn.__class__() if isinstance(self.activation_fn, nn.LeakyReLU) 
                         else type(self.activation_fn)())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Ensures positive output
        
        self.network = nn.Sequential(*layers)
        
        # Initialize
        self._initialize_weights()
        
        # Feature scaling parameters (to be set during training)
        self.register_buffer('input_mean', torch.zeros(self.input_dim))
        self.register_buffer('input_std', torch.ones(self.input_dim))
        self.register_buffer('output_mean', torch.zeros(1))
        self.register_buffer('output_std', torch.ones(1))
        
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def set_normalization(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor = None,
        output_std: torch.Tensor = None
    ):
        """
        Set normalization parameters for input/output scaling.
        
        Args:
            input_mean: Mean of input features
            input_std: Standard deviation of input features
            output_mean: Mean of output (optional)
            output_std: Standard deviation of output (optional)
        """
        self.input_mean = input_mean
        self.input_std = input_std
        if output_mean is not None:
            self.output_mean = output_mean
        if output_std is not None:
            self.output_std = output_std
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input features."""
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output."""
        return y * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass through the surrogate network.
        
        Args:
            x: Input tensor of shape (batch_size, 8)
               Features: [log_moneyness, tau, v0, kappa, theta, sigma, rho, r]
            normalize: Whether to apply input normalization
               
        Returns:
            Predicted normalized prices of shape (batch_size, 1)
        """
        if normalize:
            x = self.normalize_input(x)
        return self.network(x)
    
    def predict_price(
        self,
        S: float,
        K: float,
        tau: float,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        r: float,
        return_tensor: bool = False
    ) -> Union[float, torch.Tensor]:
        """
        Predict option price given market and model parameters.
        
        Args:
            S: Spot price
            K: Strike price
            tau: Time to maturity
            v0: Initial variance
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Vol of vol
            rho: Correlation
            r: Risk-free rate
            return_tensor: If True, return tensor instead of float
            
        Returns:
            Predicted option price
        """
        # Prepare input
        log_moneyness = np.log(K / S)
        x = torch.tensor([[log_moneyness, tau, v0, kappa, theta, sigma, rho, r]], 
                        dtype=torch.float32)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            normalized_price = self.forward(x)
            price = normalized_price * S  # Denormalize by spot
        
        if return_tensor:
            return price
        return float(price.item())
    
    def predict_prices_batch(
        self,
        S: float,
        K: np.ndarray,
        tau: np.ndarray,
        v0: float,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        r: Union[float, np.ndarray]
    ) -> np.ndarray:
        """
        Batch prediction for multiple strikes/maturities.
        
        Args:
            S: Spot price
            K: Array of strike prices
            tau: Array of maturities
            v0, kappa, theta, sigma, rho: Heston parameters
            r: Risk-free rate(s)
            
        Returns:
            Array of predicted option prices
        """
        K = np.asarray(K).flatten()
        tau = np.asarray(tau).flatten()
        n = len(K)
        
        if isinstance(r, (int, float)):
            r = np.full(n, r)
        
        # Prepare batch input
        log_moneyness = np.log(K / S)
        x = np.column_stack([
            log_moneyness,
            tau,
            np.full(n, v0),
            np.full(n, kappa),
            np.full(n, theta),
            np.full(n, sigma),
            np.full(n, rho),
            r
        ])
        x = torch.tensor(x, dtype=torch.float32)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            normalized_prices = self.forward(x)
            prices = normalized_prices.numpy().flatten() * S
        
        return prices


# =============================================================================
# COMBINED CALIBRATION NETWORK
# =============================================================================

class HestonCalibrationNetwork(nn.Module):
    """
    Combined network that performs both pricing approximation and correction.
    
    This is the full two-phase architecture:
    1. Phase 1 (PAN): Approximate pricing surface from market data
    2. Phase 2 (CCN): Correct systematic errors in calibration
    
    The network can be trained end-to-end or in separate phases.
    """
    
    def __init__(
        self,
        pan_hidden_size: int = 8,
        ccn_hidden_size: int = 7,
        use_extended_pan: bool = True,
        pan_input_dim: int = 8
    ):
        """
        Initialize combined calibration network.
        
        Args:
            pan_hidden_size: Hidden size for PAN
            ccn_hidden_size: Hidden size for CCN
            use_extended_pan: Whether to use extended PAN architecture
            pan_input_dim: Input dimension for extended PAN
        """
        super().__init__()
        
        # Price Approximator Network
        if use_extended_pan:
            self.pan = ExtendedPAN(input_dim=pan_input_dim)
        else:
            self.pan = PriceApproximatorNetwork(hidden_size=pan_hidden_size)
        
        # Calibration Correction Network
        self.ccn = CalibrationCorrectionNetwork(hidden_size=ccn_hidden_size)
        
        self.use_extended_pan = use_extended_pan
    
    def forward_pan(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through PAN only."""
        return self.pan(x)
    
    def forward_ccn(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CCN only."""
        return self.ccn(x)
    
    def forward(
        self,
        x: torch.Tensor,
        heston_price: torch.Tensor = None,
        apply_correction: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: Input features for PAN
            heston_price: Pre-computed Heston prices for CCN (optional)
            apply_correction: Whether to apply CCN correction
            
        Returns:
            Dictionary with:
            - 'pan_output': PAN predicted prices
            - 'ccn_output': CCN corrected prices (if apply_correction)
            - 'final_output': Final predicted prices
        """
        # Phase 1: PAN approximation
        pan_output = self.pan(x)
        
        result = {'pan_output': pan_output}
        
        # Phase 2: CCN correction
        if apply_correction:
            if heston_price is not None:
                # Use provided Heston prices for correction
                ccn_input = heston_price
            else:
                # Use PAN output for correction
                ccn_input = pan_output
            
            ccn_output = self.ccn(ccn_input)
            result['ccn_output'] = ccn_output
            result['final_output'] = ccn_output
        else:
            result['final_output'] = pan_output
        
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_surrogate_network(
    architecture: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create surrogate networks.
    
    Args:
        architecture: 'standard', 'pan', 'ccn', 'extended', 'combined'
        **kwargs: Additional arguments for network initialization
        
    Returns:
        Neural network module
    """
    architectures = {
        'pan': PriceApproximatorNetwork,
        'ccn': CalibrationCorrectionNetwork,
        'extended': ExtendedPAN,
        'surrogate': HestonSurrogateNetwork,
        'combined': HestonCalibrationNetwork,
        'standard': HestonSurrogateNetwork
    }
    
    if architecture not in architectures:
        raise ValueError(f"Unknown architecture: {architecture}. "
                        f"Available: {list(architectures.keys())}")
    
    return architectures[architecture](**kwargs)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module, input_size: tuple = None):
    """Print model summary."""
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {count_parameters(model):,}")
    print("\nLayers:")
    for name, module in model.named_modules():
        if name:
            print(f"  {name}: {module.__class__.__name__}")
    
    if input_size:
        print(f"\nInput size: {input_size}")


if __name__ == "__main__":
    # Test network architectures
    print("Testing Neural Network Architectures")
    print("=" * 50)
    
    # Test PAN
    pan = PriceApproximatorNetwork()
    x_pan = torch.randn(32, 1)  # Batch of strike prices
    y_pan = pan(x_pan)
    print(f"PAN - Input: {x_pan.shape}, Output: {y_pan.shape}")
    print(f"PAN parameters: {count_parameters(pan)}")
    
    # Test CCN
    ccn = CalibrationCorrectionNetwork()
    x_ccn = torch.randn(32, 1)  # Batch of Heston prices
    y_ccn = ccn(x_ccn)
    print(f"CCN - Input: {x_ccn.shape}, Output: {y_ccn.shape}")
    print(f"CCN parameters: {count_parameters(ccn)}")
    
    # Test Extended PAN
    epan = ExtendedPAN()
    x_epan = torch.randn(32, 8)  # Full feature set
    y_epan = epan(x_epan)
    print(f"Extended PAN - Input: {x_epan.shape}, Output: {y_epan.shape}")
    print(f"Extended PAN parameters: {count_parameters(epan)}")
    
    # Test Surrogate Network
    surrogate = HestonSurrogateNetwork()
    x_surr = torch.randn(32, 8)
    y_surr = surrogate(x_surr)
    print(f"Surrogate - Input: {x_surr.shape}, Output: {y_surr.shape}")
    print(f"Surrogate parameters: {count_parameters(surrogate)}")
    
    # Test Combined Network
    combined = HestonCalibrationNetwork()
    result = combined(x_epan)
    print(f"Combined - PAN output: {result['pan_output'].shape}")
    print(f"Combined parameters: {count_parameters(combined)}")
    
    print("\nAll tests passed!")
