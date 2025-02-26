import os
import numpy as np
from pathlib import Path
from utils.plotting import PlasmaPlotter
from utils.data_loader import PlasmaDataLoader
from models.pinn import PINN
from config.settings import *
from utils.constants import *

def load_metadata(metadata_path: str) -> dict:
    """Load and process metadata from NPZ file."""
    metadata = dict(np.load(metadata_path, allow_pickle=True))
    
    # Convert arrays to Python types where needed
    if isinstance(metadata['layers'], np.ndarray):
        # Handle the case where layers is a numpy array of arrays
        if metadata['layers'].dtype == np.dtype('O'):
            metadata['layers'] = metadata['layers'].item()
        # Handle the case where layers is a simple numpy array
        else:
            metadata['layers'] = metadata['layers'].tolist()
            
    # Ensure layers is a list
    if not isinstance(metadata['layers'], list):
        metadata['layers'] = LAYERS  # Use default from settings if conversion fails
        
    metadata['use_pde'] = bool(metadata['use_pde'].item() if isinstance(metadata['use_pde'], np.ndarray) else metadata['use_pde'])
    metadata['diff_norms'] = metadata['diff_norms'].item() if isinstance(metadata['diff_norms'], np.ndarray) else metadata['diff_norms']
    metadata['loss_history'] = metadata['loss_history'].item() if isinstance(metadata['loss_history'], np.ndarray) else metadata['loss_history']
    
    return metadata

def plot_results(run_timestamp: str):
    """Plot results from a trained model.
    
    Args:
        run_timestamp: Timestamp of the training run (format: YYYYMMDD_HHMMSS)
    """
    # Setup paths
    model_dir = RESULTS_PATH / f'run_{run_timestamp}'
    
    # Load data
    data_loader = PlasmaDataLoader(DATA_PATH)
    data = data_loader.load_data()
    
    # Initialize plotter
    plotter = PlasmaPlotter(model_dir)
    
    # Load metadata first
    metadata = load_metadata(os.path.join(model_dir, 'model', 'metadata.npz'))
    
    # Initialize model with loaded metadata
    model = PINN(
        x=data['x_train'], 
        y=data['y_train'],
        t=data['t_train'],
        v1=data['v1_train'],
        v5=data['v5_train'],
        layers=metadata['layers'],
        use_pde=metadata['use_pde']
    )
    
    # Load model weights
    model.load(f'{model_dir}/model')
    
    # Generate predictions
    preds = model.predict(
        x_star=data['x_train'],
        y_star=data['y_train'],
        t_star=data['t_train']
    )
    
    # Get actual data size and grid dimensions
    data_size = len(data['x_train'])
    len_loop_x = NX  # Use actual grid dimensions from settings
    len_loop_y = NY  # Use actual grid dimensions from settings

    # Create all plots
    # 1. Plasma state plots
    plotter.plot_plasma_state(
        X0=data['x_train'],
        X1=data['y_train'],
        y_den=data['v1_train'],
        y_Te=data['v5_train'],
        N_time=0,
        len_skip=1,
        len_2d=data_size
    )
    
    # 2. Loss history plot
    plotter.plot_loss_history(model.loss_history)
    
    if metadata['use_pde']:
        # 3. Electric potential plot
        plotter.plot_electric_potential(
            X0=data['x_train'],
            X1=data['y_train'],
            y_phi=data['v1_train'],
            output_model=[preds['v1'], preds['v2'], preds['v3'], preds['v4'], preds['v5']],
            N_time=0,
            len_skip=1,
            len_2d=data_size
        )
        

if __name__ == "__main__":
    # Replace with your actual run timestamp
    run_timestamp = "20250226_012959"  # Example timestamp
    plot_results(run_timestamp)
