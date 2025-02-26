"""Main training script for Plasma-PINN."""

import os
import logging
import tensorflow as tf
import numpy as np
from datetime import datetime
from models.pinn import PINN
from utils.data_loader import PlasmaDataLoader
from utils.plotting import PlasmaPlotter
from utils.constants import *
from config.settings import *
import time

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def format_losses(losses: Dict[str, float]) -> str:
    """Format loss values for printing."""
    base_losses = f"loss_v1 = {losses['v1']:.3e}, loss_v5 = {losses['v5']:.3e}"
    if USE_PDE:
        pde_losses = f", loss_f1 = {losses['f1']:.3e}, loss_f5 = {losses['f5']:.3e}"
        return base_losses + pde_losses
    return base_losses

def main():
    """Main training function."""
    setup_logging()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = RESULTS_PATH / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data loader
    data_loader = PlasmaDataLoader(DATA_PATH)
    data = data_loader.load_data()
    
    # Set random seeds
    np.random.seed(1234)
    tf.random.set_seed(1234)
    
    # Initialize model
    model = PINN(
        x=data['x_train'], 
        y=data['y_train'],
        t=data['t_train'],
        v1=data['v1_train'],
        v5=data['v5_train'],
        layers=LAYERS,
        use_pde=USE_PDE
    )
    
    # Train model
    for epoch in range(TOTAL_EPOCHS):
        losses = model.train_step()
        if epoch % PLOT_FREQUENCY == 0:
            print(f"Epoch {epoch}: {format_losses(losses)}")
    
    # Save results and plot
    plotter = PlasmaPlotter(output_dir)
    plotter.plot_loss_history(model.loss_history)

    # Save model
    model.save(f'{output_dir}/model')
    logging.info(f"Training completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
    