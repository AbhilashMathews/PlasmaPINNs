"""Plotting utilities for plasma simulation results."""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from utils.constants import *
from config.settings import *

class PlasmaPlotter:
    def __init__(self, save_path: str):
        """Initialize plotter."""
        self.save_path = save_path
        self.space_factor = 100.0 * MINOR_RADIUS  # cm conversion
        
    def plot_plasma_state(self, X0: np.ndarray, X1: np.ndarray, y_den: np.ndarray, 
                         y_Te: np.ndarray, N_time: int, len_skip: int, len_2d: int) -> None:
        """Plot plasma density and temperature."""
        x_min = self.space_factor * np.min(X0)
        x_max = self.space_factor * np.max(X0)
        y_min = self.space_factor * np.min(X1)
        y_max = self.space_factor * np.max(X1)
        
        y_plot = []
        y_plot.append(y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)])
        y_plot.append(y_Te[int(N_time*len_skip):int(N_time*len_skip + len_2d)])
        
        refValmult = [PLASMA_DENSITY, ELECTRON_TEMP]
        
        fig, axes = plt.subplots(nrows=2, ncols=1)
        for i, ax in enumerate(axes.flat):
            im = ax.scatter(self.space_factor*X0-x_min, self.space_factor*X1,
                          c=refValmult[i]*y_plot[i], cmap='YlOrRd_r', s=SCATTER_POINT_SIZE)
            fig.colorbar(im, ax=axes[i])
            ax.set_xlim(x_min-x_min, x_max-x_min)
            ax.set_ylim(y_min, y_max)
            if i % 2 == 0:
                ax.set_ylabel('y (cm)')
                ax.set_title('Observed electron density: $n_e$ (m$^{-3}$)')
            else:
                ax.set_xlabel('x (cm)')
                ax.set_ylabel('y (cm)')
                ax.set_title(r'Observed electron temperature: $T_e$ (eV)')
        
        fig.subplots_adjust(right=0.8)
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'{self.save_path}/redorng_both_n_e_T_e.png')
        plt.savefig(f'{self.save_path}/redorng_both_n_e_T_e.eps')
        plt.close()

    def plot_electric_potential(self, X0: np.ndarray, X1: np.ndarray, y_phi: np.ndarray,
                              output_model: List[np.ndarray], N_time: int, len_skip: int,
                              len_2d: int, colormap: str = 'inferno') -> None:
        """Plot electric potential comparison."""
        factor_space = 100.0 * MINOR_RADIUS
        phi_norm = compute_magnetic_field(MAJOR_RADIUS + MINOR_RADIUS) * \
                  (MINOR_RADIUS**2) / REFERENCE_TIME
        
        xlim_min = factor_space * np.min(X0)
        xlim_max = factor_space * np.max(X0)
        ylim_min = factor_space * np.min(X1)
        ylim_max = factor_space * np.max(X1)
        
        inds = np.where((factor_space*X0[:,0] > xlim_min) & 
                       (factor_space*X0[:,0] < xlim_max))[0]
        
        y_plot = []
        y_plot.append(y_phi[int(N_time*len_skip):int(N_time*len_skip + len_2d)])
        y_plot.append(output_model[2])
        
        fig, axes = plt.subplots(nrows=2, ncols=1)
        for i, ax in enumerate(axes.flat):
            im = ax.scatter(factor_space*X0[inds]-xlim_min, factor_space*X1[inds],
                          c=phi_norm*y_plot[i][inds], cmap=colormap, s=SCATTER_POINT_SIZE)
            ax.set_xlim(xlim_min-xlim_min)
            ax.set_ylim(ylim_min, ylim_max)
            fig.colorbar(im, ax=axes[i])
            if i % 2 == 0:
                ax.set_ylabel('y (cm)')
                ax.set_title(r'Target electric potential: $\phi$ (V)')
            else:
                ax.set_xlabel('x (cm)')
                ax.set_ylabel('y (cm)')
                ax.set_title(r'Predicted electric potential: $\phi$ (V)')
        
        fig.subplots_adjust(right=0.8)
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'{self.save_path}/phi.png')
        plt.savefig(f'{self.save_path}/phi.eps')
        plt.close()

    def plot_loss_history(self, history: Dict[str, List[float]]) -> None:
        """Plot training loss history."""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(history['f5'], label='$\mathcal{L(f)}_{T_e}$')
        ax.plot(history['f1'], label='$\mathcal{L(f)}_{n_e}$')
        ax.plot(history['v5'], label='$\mathcal{L}_{T_e}$')
        ax.plot(history['v1'], label='$\mathcal{L}_{n_e}$')
        ax.set_yscale('log')
        ax.set_xlabel('iterations')
        ax.set_ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/loss_history.png')
        plt.close()

    def plot_electric_field(self, X0: np.ndarray, X1: np.ndarray, y_plot: List[np.ndarray],
                           len_loop_x: int, len_loop_y: int, N_time: int, len_skip: int,
                           len_2d: int, colormap: str = 'inferno') -> None:
        """Plot electric field comparison."""
        factor_space = 100.0 * MINOR_RADIUS
        phi_norm = compute_magnetic_field(MAJOR_RADIUS + MINOR_RADIUS) * (MINOR_RADIUS**2) / self.get_reference_time()
        
        # Setup plot limits
        xlim_min = factor_space * np.min(X0)
        xlim_max = factor_space * np.max(X0)
        ylim_min = factor_space * np.min(X1)
        ylim_max = factor_space * np.max(X1)
        
        # Calculate E-field row by row
        tot_e_field_true = []
        tot_e_field_pred = []
        
        for i in range(len_loop_y):
            ind_start = i * len_loop_x
            ind_end = ind_start + len_loop_x
            
            # Extract x coordinates and potentials for this row
            x_row = X0[ind_start:ind_end,0]
            v_true = y_plot[0][ind_start:ind_end]
            v_pred = y_plot[1][ind_start:ind_end]
            
            # Sort to ensure monotonic x coordinates
            sort_idx = np.argsort(x_row)
            x_row = x_row[sort_idx]
            v_true = v_true[sort_idx]
            v_pred = v_pred[sort_idx]
            
            # Calculate E-field as negative gradient of potential
            e_field_true = -np.gradient(v_true.flatten(), x_row)
            e_field_pred = -np.gradient(v_pred.flatten(), x_row)
            
            # Store results
            tot_e_field_true.extend(e_field_true)
            tot_e_field_pred.extend(e_field_pred)
        
        # Convert to arrays
        tot_e_field_true = np.array(tot_e_field_true)
        tot_e_field_pred = np.array(tot_e_field_pred)
        
        # Get plotting indices
        inds = np.where((factor_space*X0[:,0] > xlim_min) & 
                        (factor_space*X0[:,0] < xlim_max))[0]
        
        # Create plot
        fig, axes = plt.subplots(nrows=2, ncols=1)
        for i, (Er, title) in enumerate([(tot_e_field_true, 'Target'), (tot_e_field_pred, 'Predicted')]):
            im = axes[i].scatter(
                factor_space*X0[inds]-xlim_min,
                factor_space*X1[inds],
                c=-phi_norm*Er[inds]/MINOR_RADIUS,
                cmap=colormap,
                s=SCATTER_POINT_SIZE
            )
            axes[i].set_xlim(0, xlim_max-xlim_min)
            axes[i].set_ylim(ylim_min, ylim_max)
            fig.colorbar(im, ax=axes[i])
            axes[i].set_ylabel('y (cm)')
            axes[i].set_title(f'{title} electric field: $E_r$ (V/m)')
            if i == 1:
                axes[i].set_xlabel('x (cm)')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/E_r_lim.png')
        plt.savefig(f'{self.save_path}/E_r_lim.eps')
        plt.close()
        
        # Calculate error metrics
        E1 = -phi_norm*tot_e_field_true[inds]/MINOR_RADIUS
        E2 = -phi_norm*tot_e_field_pred[inds]/MINOR_RADIUS
        error = np.abs(E1 - E2)
        print('Average electric field absolute error is: ')
        print(f"{np.mean(error):.2f} V/m")
        print(f"Max error: {np.max(error):.2f} V/m")
        print(f"Min error: {np.min(error):.2f} V/m")

    def plot_1d_potential(self, X0: np.ndarray, X1: np.ndarray, y_plot: List[np.ndarray],
                         len_loop_y: int, inds: np.ndarray) -> None:
        """Plot 1D potential profile."""
        factor_space = 100.0 * MINOR_RADIUS
        phi_norm = compute_magnetic_field(MAJOR_RADIUS + MINOR_RADIUS) * (MINOR_RADIUS**2) / self.get_reference_time()
        
        # Fix scalar indexing
        xlim_min = factor_space * np.min(X0)
        
        x_line = factor_space*X0[inds]-xlim_min
        len_x_line = int(len(x_line)/len_loop_y)
        y_line_full = factor_space*X1[inds]
        y_line = int(len_loop_y/2.)  # selecting point approximately halfway
        
        x_plot_1d = x_line[y_line*len_x_line:y_line*len_x_line+len_x_line]
        phi_plot_1d_actual = phi_norm*y_plot[0][inds][y_line*len_x_line:y_line*len_x_line+len_x_line]
        phi_plot_1d_pred = phi_norm*y_plot[1][inds][y_line*len_x_line:y_line*len_x_line+len_x_line]
        
        fig, ax1 = plt.subplots(figsize=(12., 6.25))
        ax1.plot(x_plot_1d, phi_plot_1d_actual, color='r', label='Target')
        ax1.set_ylabel(r'Target $\phi$ (V)', color='r', fontsize=20, labelpad=10)
        ax1.tick_params(axis='y', labelcolor='r', labelsize=20)
        ax1.set_xlabel('x (cm)', fontsize=20, labelpad=5)
        ax1.tick_params(axis='x', labelcolor='k', labelsize=20)
        
        ax2 = ax1.twinx()
        ax2.set_ylabel(r'Predicted $\phi$ (V)', color='k', fontsize=20, labelpad=-5)
        ax2.tick_params(axis='y', labelcolor='k', labelsize=20)
        ax2.plot(x_plot_1d, phi_plot_1d_pred, color='k', label='Prediction')
        
        ax1.set_ylim(80., 490.)
        ax2.set_ylim(-125., 285.)
        plt.subplots_adjust(top=1.0)
        plt.savefig(f'{self.save_path}/1d_phi_char.png')
        plt.savefig(f'{self.save_path}/1d_phi_char.eps')
        fig.tight_layout()
        plt.close()

    def plot_boltzmann_neoclassical(self, X0: np.ndarray, X1: np.ndarray, 
                                  y_den: np.ndarray, y_Te: np.ndarray, y_Ti: np.ndarray,
                                  N_time: int, len_skip: int, len_2d: int,
                                  len_loop_x: int, len_loop_y: int,
                                  colormap: str = 'inferno') -> None:
        """Plot Boltzmann potential and neoclassical electric field."""
        factor_space = 100.0 * MINOR_RADIUS
        phi_norm = compute_magnetic_field(MAJOR_RADIUS + MINOR_RADIUS) * (MINOR_RADIUS**2) / self.get_reference_time()
        
        # Fix scalar indexing
        xlim_min = factor_space * np.min(X0)
        xlim_max = factor_space * np.max(X0)
        ylim_min = factor_space * np.min(X1)
        ylim_max = factor_space * np.max(X1)
        
        # Calculate arrays in physical units
        den_array = PLASMA_DENSITY * y_den[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
        n0_ref = np.max(den_array)
        Te_array = ELECTRON_TEMP * y_Te[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
        Ti_array = ION_TEMP * y_Ti[int(N_time*len_skip):int(N_time*len_skip + len_2d)]
        
        # Calculate Boltzmann potential
        phi_array = np.log(den_array/n0_ref) * Te_array * ELECTRON_CHARGE / ELECTRON_CHARGE
        phi_plot = phi_array/phi_norm
        
        # Calculate neoclassical field with safety checks
        grad_n_field_true = []
        grad_Ti_field_true = []
        for i in range(len_loop_y):
            ind_start = i * len_loop_x
            ind_end = min(ind_start + len_loop_x, X0.shape[0])  # Prevent overflow
            if ind_end <= ind_start:
                continue
                
            # Get coordinate slice and ensure proper shape
            x_slice = MINOR_RADIUS * X0[ind_start:ind_end, 0].flatten()
            den_slice = den_array[ind_start:ind_end].flatten()
            Ti_slice = Ti_array[ind_start:ind_end].flatten()
            
            if len(x_slice) < 2:  # Need at least 2 points for gradient
                continue
                
            try:
                # Sort by x coordinate to ensure proper gradient calculation
                sort_idx = np.argsort(x_slice)
                x_slice = x_slice[sort_idx]
                den_slice = den_slice[sort_idx]
                Ti_slice = Ti_slice[sort_idx]
                
                # Remove any duplicate x values that could cause division by zero
                unique_idx = np.concatenate(([True], np.diff(x_slice) > 1e-10))
                x_slice = x_slice[unique_idx]
                den_slice = den_slice[unique_idx]
                Ti_slice = Ti_slice[unique_idx]
                
                if len(x_slice) < 2:  # Check again after removing duplicates
                    continue
                    
                # Calculate gradients
                grad_n = np.gradient(den_slice, x_slice, edge_order=1)
                grad_Ti = np.gradient(Ti_slice, x_slice, edge_order=1)
                
                # Append to lists
                grad_n_field_true.append(grad_n)
                grad_Ti_field_true.append(grad_Ti)
                
            except (ValueError, IndexError) as e:
                print(f"Warning: Gradient calculation failed for slice {i}: {e}")
                continue
        
        if not grad_n_field_true or not grad_Ti_field_true:
            print("Warning: No valid gradient data calculated")
            return
            
        # Convert to arrays ensuring same length
        grad_n_field_true = np.concatenate(grad_n_field_true)
        grad_Ti_field_true = np.concatenate(grad_Ti_field_true)
        
        # Ensure all arrays have same dimensions for force balance calculation
        valid_len = min(len(grad_n_field_true), len(grad_Ti_field_true), len(den_array))
        grad_n_field_true = grad_n_field_true[:valid_len]
        grad_Ti_field_true = grad_Ti_field_true[:valid_len]
        den_array_true = np.array(den_array)[:valid_len]
        Ti_array_true = np.array(Ti_array)[:valid_len]
        
        # Calculate force balance
        Er_ion_force_balance = (Ti_array_true*grad_n_field_true + 
                              den_array_true*grad_Ti_field_true)/(den_array_true)
        
        xlim_min = factor_space * np.min(X0)[0]
        xlim_max = factor_space * np.max(X0)[0]
        ylim_min = factor_space * np.min(X1)[0]
        ylim_max = factor_space * np.max(X1)[0]
        
        inds = np.where((factor_space*X0[:,0] > xlim_min) & 
                       (factor_space*X0[:,0] < xlim_max))[0]
        
        y_plot = [phi_plot, Er_ion_force_balance/phi_norm]
        refs = [phi_norm, phi_norm/MINOR_RADIUS]
        
        fig, axes = plt.subplots(nrows=2, ncols=1)
        for i, ax in enumerate(axes.flat):
            im = ax.scatter(factor_space*X0[inds]-xlim_min, factor_space*X1[inds],
                          c=refs[i]*y_plot[i][inds], cmap=colormap, s=SCATTER_POINT_SIZE)
            ax.set_xlim(xlim_min-xlim_min)
            ax.set_ylim(ylim_min, ylim_max)
            fig.colorbar(im, ax=axes[i])
            if i == 0:
                ax.set_ylabel('y (cm)')
                ax.set_title(r'Boltzmann potential: $\phi$ (V)')
            else:
                ax.set_xlabel('x (cm)')
                ax.set_ylabel('y (cm)')
                ax.set_title('Neoclassical electric field: $E_r$ (V/m)')
        
        fig.subplots_adjust(right=0.8)
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f'{self.save_path}/Boltz_neo.png')
        plt.savefig(f'{self.save_path}/Boltz_neo.eps')
        plt.close()

    def get_reference_time(self) -> float:
        """Calculate reference time."""
        cse = np.sqrt(ELECTRON_CHARGE * ELECTRON_TEMP / (MASS_RATIO * ELECTRON_MASS))
        return np.sqrt((MAJOR_RADIUS * MINOR_RADIUS) / 2.0) / cse