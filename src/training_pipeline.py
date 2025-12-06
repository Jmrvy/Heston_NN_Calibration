"""
Training Pipeline for Heston Neural Network Calibration.

This module provides complete training functionality for:
1. Surrogate Pricing Network (Phase 1)
2. Calibration Correction Network (Phase 2)
3. Combined end-to-end training

Features:
- Synthetic data generation for training
- Multi-regime training support
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training visualization
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Tuple, Optional, List, Union, Callable
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))

from neural_networks import (
    HestonSurrogateNetwork, 
    PriceApproximatorNetwork,
    CalibrationCorrectionNetwork,
    HestonCalibrationNetwork,
    count_parameters
)
from heston_model import HestonModel

logger = logging.getLogger(__name__)


# =============================================================================
# DATASET CLASSES
# =============================================================================

class HestonPricingDataset(Dataset):
    """
    PyTorch Dataset for Heston option pricing data.
    
    Features (input):
        - log_moneyness: ln(K/S)
        - tau: time to maturity
        - v0: initial variance
        - kappa: mean reversion speed
        - theta: long-term variance
        - sigma: volatility of volatility
        - rho: correlation
        - r: risk-free rate
    
    Target:
        - normalized_price: option_price / S
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str] = None,
        target_col: str = 'option_price',
        spot_col: str = 'S0',
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data: DataFrame with option data
            feature_cols: List of feature column names
            target_col: Target column name
            spot_col: Spot price column name
            normalize: Whether to normalize features
        """
        self.normalize = normalize
        
        # Default feature columns
        if feature_cols is None:
            feature_cols = ['log_moneyness', 'tau', 'v0', 'kappa', 'theta', 'sigma', 'rho', 'r']
        
        # Ensure log_moneyness exists
        if 'log_moneyness' not in data.columns:
            if 'moneyness' in data.columns:
                data = data.copy()
                data['log_moneyness'] = np.log(data['moneyness'])
            elif 'K' in data.columns and spot_col in data.columns:
                data = data.copy()
                data['log_moneyness'] = np.log(data['K'] / data[spot_col])
        
        # Extract features and targets
        self.features = data[feature_cols].values.astype(np.float32)
        
        # Normalize price by spot
        if spot_col in data.columns:
            self.targets = (data[target_col] / data[spot_col]).values.astype(np.float32)
        else:
            self.targets = data[target_col].values.astype(np.float32)
        
        self.targets = self.targets.reshape(-1, 1)
        
        # Compute normalization statistics
        if normalize:
            self.feature_mean = self.features.mean(axis=0)
            self.feature_std = self.features.std(axis=0) + 1e-8
            self.target_mean = self.targets.mean()
            self.target_std = self.targets.std() + 1e-8
        else:
            self.feature_mean = np.zeros(len(feature_cols))
            self.feature_std = np.ones(len(feature_cols))
            self.target_mean = 0.0
            self.target_std = 1.0
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get normalized sample."""
        x = self.features[idx]
        y = self.targets[idx]
        
        if self.normalize:
            x = (x - self.feature_mean) / self.feature_std
        
        return torch.tensor(x), torch.tensor(y)
    
    def get_normalization_params(self) -> Dict[str, np.ndarray]:
        """Return normalization parameters."""
        return {
            'feature_mean': self.feature_mean,
            'feature_std': self.feature_std,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }


class PANDataset(Dataset):
    """
    Dataset for Price Approximator Network.
    Simple mapping from strike to price for a single calibration set.
    """
    
    def __init__(
        self,
        strikes: np.ndarray,
        prices: np.ndarray,
        normalize: bool = True
    ):
        """
        Initialize PAN dataset.
        
        Args:
            strikes: Array of strike prices
            prices: Array of option prices
            normalize: Whether to normalize data
        """
        self.strikes = strikes.astype(np.float32).reshape(-1, 1)
        self.prices = prices.astype(np.float32).reshape(-1, 1)
        self.normalize = normalize
        
        if normalize:
            self.strike_mean = self.strikes.mean()
            self.strike_std = self.strikes.std() + 1e-8
            self.price_mean = self.prices.mean()
            self.price_std = self.prices.std() + 1e-8
        else:
            self.strike_mean = 0.0
            self.strike_std = 1.0
            self.price_mean = 0.0
            self.price_std = 1.0
    
    def __len__(self) -> int:
        return len(self.strikes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.strikes[idx]
        y = self.prices[idx]
        
        if self.normalize:
            x = (x - self.strike_mean) / self.strike_std
        
        return torch.tensor(x), torch.tensor(y)


# =============================================================================
# DATA GENERATION FOR TRAINING
# =============================================================================

def generate_training_data(
    n_samples: int = 100000,
    param_ranges: Dict = None,
    seed: int = None,
    pricing_method: str = 'fft',
    progress_bar: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic training data for surrogate network.
    
    This function creates a comprehensive dataset spanning realistic
    parameter ranges for training the Heston surrogate network.
    
    Args:
        n_samples: Number of samples to generate
        param_ranges: Dictionary of parameter ranges
        seed: Random seed
        pricing_method: 'fft', 'quad', or 'rectangular'
        progress_bar: Show progress bar
        
    Returns:
        DataFrame with training data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Default parameter ranges (covering typical market conditions)
    if param_ranges is None:
        param_ranges = {
            'v0': (0.01, 0.4),        # Initial variance
            'kappa': (0.1, 5.0),       # Mean reversion speed
            'theta': (0.01, 0.3),      # Long-term variance
            'sigma': (0.1, 1.5),       # Vol of vol
            'rho': (-0.95, -0.1),      # Correlation (typically negative)
            'r': (0.0, 0.08),          # Risk-free rate
            'tau': (0.05, 2.0),        # Time to maturity
            'moneyness': (0.7, 1.3)    # K/S ratio
        }
    
    data = []
    S0 = 100.0  # Normalized spot price
    
    iterator = tqdm(range(n_samples), desc="Generating data") if progress_bar else range(n_samples)
    
    for _ in iterator:
        # Sample Heston parameters
        v0 = np.random.uniform(*param_ranges['v0'])
        kappa = np.random.uniform(*param_ranges['kappa'])
        theta = np.random.uniform(*param_ranges['theta'])
        sigma = np.random.uniform(*param_ranges['sigma'])
        rho = np.random.uniform(*param_ranges['rho'])
        r = np.random.uniform(*param_ranges['r'])
        
        # Ensure Feller condition: 2*kappa*theta > sigma^2
        feller_ratio = (2 * kappa * theta) / (sigma**2)
        if feller_ratio <= 1.0:
            # Adjust kappa to satisfy Feller
            kappa = max(kappa, 1.1 * sigma**2 / (2 * theta))
        
        # Sample option parameters
        tau = np.random.uniform(*param_ranges['tau'])
        moneyness = np.random.uniform(*param_ranges['moneyness'])
        K = S0 * moneyness
        
        try:
            # Create model and price option
            model = HestonModel(
                kappa=kappa, theta=theta, sigma=sigma, rho=rho,
                v0=v0, r=r, q=0.0, lambd=0.0
            )
            
            # Compute call price
            if pricing_method == 'fft':
                price = model.call_price_fft(S0, K, tau)
            elif pricing_method == 'quad':
                price = model.call_price_quad(S0, K, tau)
            else:
                price = model.call_price_rectangular(S0, K, tau)
            
            # Validate price
            if price > 0 and not np.isnan(price) and not np.isinf(price):
                data.append({
                    'S0': S0,
                    'K': K,
                    'tau': tau,
                    'r': r,
                    'moneyness': moneyness,
                    'log_moneyness': np.log(moneyness),
                    'v0': v0,
                    'kappa': kappa,
                    'theta': theta,
                    'sigma': sigma,
                    'rho': rho,
                    'option_price': price,
                    'normalized_price': price / S0
                })
                
        except Exception as e:
            continue
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} valid training samples")
    
    return df


# =============================================================================
# TRAINER CLASS
# =============================================================================

class HestonNetworkTrainer:
    """
    Training manager for Heston neural networks.
    
    Supports:
    - Surrogate network training
    - PAN training
    - CCN training
    - Combined training
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        scheduler_type: str = 'reduce_on_plateau',
        checkpoint_dir: str = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Neural network model
            device: 'cuda', 'cpu', or None (auto-detect)
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            scheduler_type: 'reduce_on_plateau', 'cosine', 'step', or None
            checkpoint_dir: Directory to save checkpoints
        """
        # Device setup
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Model
        self.model = model.to(self.device)
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)  # Default Adam betas
        )
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler(scheduler_type)
        
        # Loss function (MSE)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def _create_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler."""
        if scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=50, T_mult=2
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        return None
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        logger.info(f"Model parameters: {count_parameters(self.model):,}")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Early stopping check
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    if self.checkpoint_dir:
                        self.save_checkpoint('best_model.pth')
                else:
                    self.epochs_without_improvement += 1
                    
                    if self.epochs_without_improvement >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                val_loss = None
                if self.scheduler is not None and not isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()
            
            # Record learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rate'].append(current_lr)
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                msg += f" - LR: {current_lr:.2e}"
                print(msg)
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
        
        filepath = self.checkpoint_dir / filename
        
        # Include model architecture info if available
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save architecture info for HestonSurrogateNetwork
        if hasattr(self.model, 'hidden_layers'):
            save_dict['hidden_layers'] = self.model.hidden_layers
        
        torch.save(save_dict, filepath)
        logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        if self.checkpoint_dir is None:
            raise ValueError("Checkpoint directory not set")
        
        filepath = self.checkpoint_dir / filename
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from {filepath}")
    
    def plot_training_history(self, save_path: str = None):
        """Plot training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1 = axes[0]
        ax1.plot(self.history['train_loss'], label='Train Loss', alpha=0.8)
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='Val Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2 = axes[1]
        ax2.plot(self.history['learning_rate'])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training plot to {save_path}")
        
        return fig


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_surrogate_network(
    train_data: pd.DataFrame,
    val_split: float = 0.1,
    batch_size: int = 256,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    hidden_layers: List[int] = [128, 128, 64, 32],
    checkpoint_dir: str = None,
    device: str = None,
    seed: int = 42
) -> Tuple[HestonSurrogateNetwork, Dict]:
    """
    Train the Heston surrogate pricing network.
    
    Args:
        train_data: Training DataFrame
        val_split: Validation split ratio
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        hidden_layers: Network architecture
        checkpoint_dir: Directory for checkpoints
        device: Computing device
        seed: Random seed
        
    Returns:
        Trained model and training history
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dataset
    dataset = HestonPricingDataset(train_data, normalize=True)
    
    # Split into train/val
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Create model
    model = HestonSurrogateNetwork(
        hidden_layers=hidden_layers,
        activation='leaky_relu',
        batch_norm=True,
        dropout=0.1
    )
    
    # Set normalization parameters
    norm_params = dataset.get_normalization_params()
    model.set_normalization(
        input_mean=torch.tensor(norm_params['feature_mean']),
        input_std=torch.tensor(norm_params['feature_std'])
    )
    
    # Create trainer
    trainer = HestonNetworkTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=1e-5,
        scheduler_type='reduce_on_plateau',
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=30,
        verbose=True
    )
    
    # Load best model if checkpointing was used
    if checkpoint_dir:
        try:
            trainer.load_checkpoint('best_model.pth')
        except:
            pass
    
    return trainer.model, history, norm_params


def train_pan_for_calibration_set(
    strikes: np.ndarray,
    prices: np.ndarray,
    hidden_size: int = 8,
    epochs: int = 500,
    learning_rate: float = 0.01,
    device: str = None
) -> Tuple[PriceApproximatorNetwork, Dict]:
    """
    Train PAN for a specific calibration set (single maturity).
    
    Training PAN to approximate the price-strike relationship for a specific set of options.
    
    Args:
        strikes: Array of strike prices
        prices: Array of option prices
        hidden_size: PAN hidden layer size
        epochs: Training epochs
        learning_rate: Learning rate
        device: Computing device
        
    Returns:
        Trained PAN model and normalization parameters
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Create dataset
    dataset = PANDataset(strikes, prices, normalize=True)
    loader = DataLoader(dataset, batch_size=min(32, len(strikes)), shuffle=True)
    
    # Create model
    model = PriceApproximatorNetwork(hidden_size=hidden_size).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50
    )
    
    # Training
    criterion = nn.MSELoss()
    history = {'loss': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        history['loss'].append(avg_loss)
        scheduler.step(avg_loss)
    
    norm_params = {
        'strike_mean': dataset.strike_mean,
        'strike_std': dataset.strike_std,
        'price_mean': dataset.price_mean,
        'price_std': dataset.price_std
    }
    
    return model, norm_params, history


def train_ccn(
    heston_prices: np.ndarray,
    market_prices: np.ndarray,
    hidden_size: int = 7,
    epochs: int = 500,
    learning_rate: float = 0.01,
    device: str = None
) -> Tuple[CalibrationCorrectionNetwork, Dict]:
    """
    Train CCN to correct Heston model pricing errors.
    
    Args:
        heston_prices: Heston model predicted prices
        market_prices: Actual market prices
        hidden_size: CCN hidden layer size
        epochs: Training epochs
        learning_rate: Learning rate
        device: Computing device
        
    Returns:
        Trained CCN model and normalization parameters
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Prepare data
    x = torch.tensor(heston_prices.reshape(-1, 1), dtype=torch.float32)
    y = torch.tensor(market_prices.reshape(-1, 1), dtype=torch.float32)
    
    # Normalize
    x_mean, x_std = x.mean(), x.std() + 1e-8
    x_norm = (x - x_mean) / x_std
    
    # Create model
    model = CalibrationCorrectionNetwork(hidden_size=hidden_size).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training
    criterion = nn.MSELoss()
    history = {'loss': []}
    
    x_norm = x_norm.to(device)
    y = y.to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_norm)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        history['loss'].append(loss.item())
    
    norm_params = {
        'input_mean': float(x_mean),
        'input_std': float(x_std)
    }
    
    return model, norm_params, history


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example training workflow
    print("=" * 60)
    print("Heston Neural Network Training Pipeline")
    print("=" * 60)
    
    # Generate small dataset for testing
    print("\n1. Generating training data...")
    train_data = generate_training_data(
        n_samples=10000,
        seed=42,
        progress_bar=True
    )
    print(f"   Generated {len(train_data)} samples")
    
    # Train surrogate network
    print("\n2. Training surrogate network...")
    model, history, norm_params = train_surrogate_network(
        train_data=train_data,
        val_split=0.1,
        batch_size=128,
        epochs=50,  # Reduced for demo
        learning_rate=1e-3,
        hidden_layers=[64, 64, 32],
        checkpoint_dir='data/models/checkpoints',
        seed=42
    )
    
    # Final validation loss
    final_train_loss = history['train_loss'][-1]
    final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
    
    print(f"\n   Final Training Loss: {final_train_loss:.6f}")
    if final_val_loss:
        print(f"   Final Validation Loss: {final_val_loss:.6f}")
    
    # Test prediction
    print("\n3. Testing predictions...")
    model.eval()
    with torch.no_grad():
        # Sample test case
        test_input = torch.tensor([[
            0.0,    # log_moneyness (ATM)
            0.25,   # tau (3 months)
            0.04,   # v0
            2.0,    # kappa
            0.04,   # theta
            0.3,    # sigma
            -0.7,   # rho
            0.03    # r
        ]], dtype=torch.float32)
        
        pred = model(test_input, normalize=True)
        print(f"   ATM 3-month option (normalized price): {pred.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Training pipeline test completed!")
    print("=" * 60)
