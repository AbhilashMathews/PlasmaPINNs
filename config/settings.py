"""Configuration settings for the Plasma-PINN model."""

import os
import numpy as np
from pathlib import Path
from typing import Dict

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data/plasma_data.h5"
RESULTS_PATH = PROJECT_ROOT / "results"

# Model architecture
LAYERS = [3, 50, 50, 50, 50, 50, 1]
N_OUTPUTS = 1

# Training hyperparameters
TRAINING_HOURS = 20.0
STEPS_PER_HOUR = 1000
TOTAL_EPOCHS = int(TRAINING_HOURS * STEPS_PER_HOUR)
SAMPLE_BATCH_SIZE = 500
TRAIN_FRACTION = 1.0

# Model weights
INIT_WEIGHT_DEN = 1.0  # Initial weight for density terms
INIT_WEIGHT_TE = 1.0   # Initial weight for temperature terms

# Plotting settings
SCATTER_POINT_SIZE = 2.5  # Size of scatter plot points
PLOT_FREQUENCY = 100    # Number of iterations between plots
SAVE_FREQUENCY = 1000   # Number of iterations between saves

# Data settings
NOISE_MEAN = 1.0
NOISE_STD = 0.25

# Grid settings
NX = 256
NY = 128
NZ = 32

# Time settings
DT = 0.000005  # Time step (normalized)
N_TIMESTEPS = 16000
INITIAL_FRAME = 0
FINAL_FRAME = 398

# Domain settings
LX = 0.35  # Radial size (normalized by minor radius)
LY = 0.25  # Vertical size (normalized by minor radius)
LZ = 20.0  # Connection length (normalized by major radius)

# Grid spacing
DX = LX / NX  # Radial grid spacing
DY = LY / NY  # Vertical grid spacing 
DZ = LZ / NZ  # Toroidal grid spacing
GRID_SPACING = np.array([DX, DY, DZ, DT])

# Model settings
USE_PDE = True  # Whether to use PDE constraints in training

def compute_diffusion_norms(dx: np.ndarray) -> Dict[str, float]:
    """Compute normalized diffusion coefficients."""
    DiffX = 2.0 * np.pi / (dx[0] * 3.0)
    DiffY = 2.0 * np.pi / (dx[1] * 3.0)
    DiffZ = 2.0 * np.pi / (dx[2] * 3.0)
    
    return {
        'DiffX_norm': DiffX**2,
        'DiffY_norm': DiffY**2,
        'DiffZ_norm': DiffZ**2
    }

# Compute normalized diffusion coefficients
DIFF_NORMS = compute_diffusion_norms(GRID_SPACING)

# Export constants for use in other modules
DiffX_norm = DIFF_NORMS['DiffX_norm']
DiffY_norm = DIFF_NORMS['DiffY_norm']
DiffZ_norm = DIFF_NORMS['DiffZ_norm']