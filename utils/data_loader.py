"""Data loading and preprocessing utilities."""

import h5py
import numpy as np
from typing import Dict, Tuple
from utils.constants import *
from config.settings import *

class PlasmaDataLoader:
    def __init__(self, data_path: str):
        """Initialize data loader.
        
        Args:
            data_path: Path to HDF5 data file
        """
        self.data_path = data_path
        
    def load_data(self) -> Dict[str, np.ndarray]:
        """Load and preprocess plasma data."""
        with h5py.File(self.data_path, 'r') as h5f:
            x_x = h5f['x_x'][:]
            x_y = h5f['x_y'][:]
            x_z = h5f['x_z'][:]
            x_t = h5f['x_t'][:]
            y_den = h5f['y_den'][:]
            y_Te = h5f['y_Te'][:]
            
        # Calculate weights
        init_weight_den = 1.0 / np.median(np.abs(y_den))
        init_weight_Te = 1.0 / np.median(np.abs(y_Te))
        
        # Sample training data
        N_train = int(TRAIN_FRACTION * len(y_den))
        idx = np.random.choice(len(y_den), N_train, replace=False)
        
        return {
            'x_train': x_x[idx],
            'y_train': x_y[idx],
            'z_train': x_z[idx],
            't_train': x_t[idx],
            'v1_train': y_den[idx],
            'v5_train': y_Te[idx],
            'weights': {
                'den': init_weight_den,
                'Te': init_weight_Te
            }
        }
        
    def preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess loaded data."""
        # Compute normalization factors
        init_weight_den = 1.0 / np.median(np.abs(data['y_den']))
        init_weight_Te = 1.0 / np.median(np.abs(data['y_Te']))
        
        # Create training indices
        N_train = int(TRAIN_FRACTION * len(data['y_den']))
        idx = np.random.choice(len(data['y_den']), N_train, replace=False)
        
        # Select training data
        processed_data = {
            'x_train': data['x_x'][idx],
            'y_train': data['x_y'][idx],
            'z_train': data['x_z'][idx],
            't_train': data['x_t'][idx],
            'v1_train': data['y_den'][idx],
            'v5_train': data['y_Te'][idx],
            'weights': {
                'den': init_weight_den,
                'Te': init_weight_Te
            }
        }
        
        return processed_data
        
    def add_noise(self, data: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to data."""
        noise = np.random.normal(NOISE_MEAN, NOISE_STD, data.shape)
        return data * noise 